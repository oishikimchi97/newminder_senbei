import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False, activation=True,
                 batch_norm=True):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else None
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        y = self.conv2d(x)
        y = self.batch_norm(y) if self.batch_norm else y
        y = self.activation(y) if self.activation else y
        return y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, activation=True, batch_norm=True)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, activation=False, batch_norm=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = torch.add(x, y)
        return y


class DownConv(nn.Module):
    def __init__(self, input_channel, input_size, output_size):
        super().__init__()

        self.conv1 = ConvBlock(input_channel, 16, 7, 1, activation=True, batch_norm=False)
        self.conv2 = ConvBlock(16, 16, 7, 1, activation=True, batch_norm=True)
        self.resblock1 = ResBlock(16, 16)
        self.conv3 = ConvBlock(16, 16, 3, 1, activation=False, batch_norm=True)
        self.conv4 = ConvBlock(16, 3, 7, 1, activation=False, batch_norm=False)

        self.scale_factor = tuple(output_size[i] / input_size[i] for i in range(2))
        self.binear_resizer = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.conv2(y1)
        y1 = self.binear_resizer(y1)

        res = self.resblock1(y1)
        res = self.conv3(res)

        y1 = torch.add(res, y1)
        y1 = self.conv4(y1)

        y2 = self.binear_resizer(x)

        output = torch.add(y1, y2)

        return output


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class DownconvUnet(nn.Module):
    def __init__(self, in_channel=3, seg_classes=1, cls_classes=2, mode=0):
        super().__init__()

        self.downconv = DownConv(3, input_size=(2448, 2048), output_size=(256, 256))
        self.unet = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights="noisy-student",
            in_channels=in_channel,
            classes=seg_classes
        )
        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        encoder_channel = 352

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(encoder_channel, int(encoder_channel / 2)),
            nn.Dropout(p=0.2),
            nn.Linear(int(encoder_channel / 2), cls_classes)
        )

        self._mode = mode  # 0: seg, cls mode, 2: seg mode, 3: cls mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        mode2name = {
            0: "ALL",
            1: "SEGMENTATION",
            2: "CLASSIFICATION",
        }
        if mode in [0, 1, 2]:
            self._mode = mode
            print(f"{mode2name[mode]} mode is activated")
        else:
            raise ValueError("mode must be the one of 0 (ALL) , 1 (SEGMENTATION) , 2 (CLASSIFICATION)")

    def forward(self, x):
        """

        :rtype: return predict value
        mode 0:
        return pred of segmentation and classification
        mode 1:
        return pred of segmentation
        mode 2:
        return pred of classification
        """
        x = self.downconv(x)
        x = self.encoder(x)

        if self.mode in [0, 1]:
            seg_y = self.decoder(x)
        if self.mode in [0, 2]:
            cls_y = self.classifier(x)

        if self.mode == 0:
            return seg_y, cls_y
        elif self.mode == 1:
            return seg_y
        elif self.mode == 2:
            return cls_y

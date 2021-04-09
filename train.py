import torch
from torch.utils.data import DataLoader
from data import SegmentationDataset
from model import DownconvUnet
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow

SMOOTH = 1e-6


class Params:
    def __init__(self, batch_size=8, epochs=100, lr=0.01, swa_lr=0.005, seed=2021, weight=[1., 1.]):
        self.swa_lr = swa_lr
        self.weight = weight
        self.lr = lr
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size


class Metrics:  # for logging by each step
    def __init__(self):
        self.loss = 0.
        self.iou = 0.
        self.acc = 0.

    @staticmethod
    def _iou(outputs: torch.Tensor, labels: torch.Tensor):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1)

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded.mean()  # Orf thresholded.mean() if you are interested in average across the batch

    @staticmethod
    def _accuracy(outputs: torch.Tensor, labels: torch.Tensor):
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        positive = outputs > 0.5
        positive_num = torch.sum(positive)

        positive_true = (positive | labels)
        positive_true_num = torch.sum(positive_true)

        acc = positive_num / (positive_true + SMOOTH)

        return acc

    def update(self, outputs, labels, loss):
        self.loss = loss
        self.iou = self._iou(outputs, labels)
        self.acc = self._accuracy(outputs, labels)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_train = True
    classification_train = True

    train_dir = ''
    val_dir = ''

    train_dataset = SegmentationDataset(img_dir=train_dir, mask_dir=train_dir,
                                        n_channels=3, classes=1, train=True)
    val_dataset = SegmentationDataset(img_dir=val_dir, mask_dir=val_dir,
                                      n_channels=3, classes=1, train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    my_model = DownconvUnet(in_channel=3, seg_classes=1, cls_classes=2)
    avg_model = AveragedModel(my_model)

    my_model.to(device)
    avg_model.to(device)

    with mlflow.start_run() as run:
        seg_args = Params(8, 100, 0.01, 0.005, 2021, [0.1, 1.])
        cls_args = Params(8, 20, 0.01, 0.005, 2021, [1, 1.])
        mode_list = ["seg", "cls"]
        for mode in mode_list:
            for key, value in vars(seg_args).items():
                mlflow.log_param(f"{mode}_{key}", value)

        # Segmentation train

        if segmentation_train:
            print("-" * 5 + "Segmentation training start" + "-" * 5)

            my_model.mode = 1
            train_metrics = Metrics()
            train_loss = 0.
            train_iou = 0.
            train_acc = 0.

            val_metrics = Metrics()
            val_loss = 0.
            val_iou = 0.
            val_acc = 0.

            my_model.train()

            optimizer = torch.optim.Adam(my_model.parameters(), lr=seg_args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=100)
            bce = torch.nn.BCELoss(weight=seg_args.weight)
            swa_start = int(seg_args.epochs * 0.75)
            swa_scheduler = SWALR(optimizer, swa_lr=seg_args.swa_lr)

            for epoch in range(seg_args.epochs):
                for batch_idx, (seg_x, seg_y) in enumerate(train_loader):
                    pred_y = my_model(seg_x)

                    loss = bce(pred_y, seg_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_metrics.update(pred_y, seg_y, loss.item())
                    train_iou += train_metrics.iou
                    train_acc += train_metrics.acc

                    step = epoch * len(train_loader) + batch_idx
                    for metric, value in vars(train_metrics).items():
                        mlflow.log_metric(f"seg_train_{metric}", value, step=step)

                train_loss /= len(train_loader)
                train_iou /= len(train_loader)
                train_acc /= len(train_loader)

                my_model.eval()

                for batch_idx, (seg_x, seg_y) in enumerate(val_loader):
                    pred_y = my_model(seg_x)

                    loss = bce(pred_y, seg_y)

                    val_loss += loss.item()
                    val_metrics.update(pred_y, seg_y, loss.item())
                    val_iou += val_metrics.iou
                    val_acc += val_metrics.acc

                    step = epoch * len(val_loader) + batch_idx
                    for metric, value in vars(val_metrics).items():
                        mlflow.log_metric(f"seg_val_{metric}", value, step=step)

                val_loss /= len(val_loader)
                val_iou /= len(val_loader)
                val_acc /= len(val_loader)

                print(f"Epoch {epoch + 1}:")
                print("-" * 10)
                print(f"train_loss {train_loss.data.item():.3f}, train_iou: {train_iou.data.item():.3f}, "
                      f"train_accuracy: {train_acc.data.item():.3f}")
                print(f"val_loss {val_loss.data.item():.3f}, val_iou: {val_iou.data.item():.3f}, "
                      f"val_accuracy: {val_acc.data.item():.3f}")

                if epoch > swa_start:
                    print("Stochastic average start")
                    avg_model.update_parameters(my_model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

            print("Segmentation train completed")

            # Classification train

            if classification_train:
                print("-" * 5 + "Classification training start" + "-" * 5)

                my_model.mode = 2

                train_metrics = Metrics()
                train_loss = 0.
                train_iou = 0.
                train_acc = 0.

                val_metrics = Metrics()
                val_loss = 0.
                val_iou = 0.
                val_acc = 0.

                my_model.train()

                optimizer = torch.optim.Adam(my_model.parameters(), lr=cls_args.lr)
                scheduler = CosineAnnealingLR(optimizer, T_max=100)
                bce = torch.nn.BCELoss(weight=cls_args.weight)
                swa_start = int(cls_args.epochs * 0.75)
                swa_scheduler = SWALR(optimizer, swa_lr=cls_args.swa_lr)

                for epoch in range(cls_args.num_epchos):
                    for batch_idx, (cls_x, cls_y) in enumerate(train_loader):
                        pred_y = my_model(cls_x)

                        loss = bce(pred_y, cls_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_metrics.update(pred_y, cls_y, loss.item())
                        train_acc += train_metrics.acc

                        step = epoch * len(train_loader) + batch_idx
                        for metric, value in vars(train_metrics).items():
                            mlflow.log_metric(f"cls_train_{metric}", value, step=step)

                    train_loss /= len(train_loader)
                    train_acc /= len(train_loader)


                    my_model.eval()

                    for batch_idx, (cls_x, cls_y) in enumerate(val_loader):
                        pred_y = my_model(cls_x)

                        loss = bce(pred_y, cls_y)

                        val_loss += loss.item()
                        val_metrics.update(pred_y, cls_y, loss.item())
                        val_acc += val_metrics.acc

                        step = epoch * len(train_loader) + batch_idx
                        for metric, value in vars(val_metrics).items():
                            mlflow.log_metric(f"cls_train_{metric}", value, step=step)

                    val_loss /= len(val_loader)
                    val_acc /= len(val_loader)

                    print(f"Epoch {epoch + 1}:")
                    print("-" * 10)
                    print(f"train_loss {train_loss:.3f}, train_iou: {train_iou:.3f}, train_accuracy: {train_acc:.3f}")
                    print(f"val_loss {val_loss:.3f}, val_iou: {val_iou:.3f}, val_accuracy: {val_acc:.3f}")

                print("Classification train completed")

                if epoch > swa_start:
                    print("Stochastic average start")
                    avg_model.update_parameters(my_model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
    weight_path = "weights/donwconv_swa_weights.pth"
    torch.save(my_model.state_dict(), weight_path)
    print(f"model weight saved to {weight_path}")

import torch
from torch.utils.data import DataLoader
from data import SegmentationDataset
from model import DownconvUnet

SMOOTH = 1e-6


def iou(outputs: torch.Tensor, labels: torch.Tensor):
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


def accuracy(outputs, labels):
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)

    positive = outputs > 0.5
    positive_num = torch.sum(positive)

    positive_true = (positive | labels)
    positive_true_num = torch.sum(positive_true)

    acc = positive_num / (positive_true + SMOOTH)

    return acc


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

    my_model.to(device)

    # Segmentation train

    if segmentation_train:
        print("-" * 5 + "Segmentation training start" + "-" * 5)

        my_model.mode = 1
        seg_num_epochs = 20

        train_loss = 0.
        train_iou = 0.
        train_acc = 0.

        val_loss = 0.
        val_iou = 0.
        val_acc = 0.

        my_model.train()

        params = [p for p in my_model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(params, lr=0.005)
        bce = torch.nn.BCELoss(weight=[0.1, 1])

        for epoch in range(seg_num_epochs):
            for seg_x, seg_y in train_loader:
                pred_y = my_model(seg_x)

                loss = bce(pred_y, seg_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_iou += iou(pred_y, seg_y)
                train_acc += accuracy(pred_y, seg_y)

            train_loss /= len(train_loader)
            train_iou /= len(train_loader)
            train_acc /= len(train_loader)

            my_model.eval()

            for seg_x, seg_y in val_loader:
                pred_y = my_model(seg_x)

                loss = bce(pred_y, seg_y)

                val_loss += loss.item()
                val_iou += iou(pred_y, seg_y)
                val_acc += accuracy(pred_y, seg_y)

            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            val_acc /= len(val_loader)

            print(f"Epoch {epoch}:")
            print("-" * 10)
            print(f"train_loss {train_loss}, train_iou: {train_iou}, train_accuracy: {train_acc}")
            print(f"val_loss {val_loss}, val_iou: {val_iou}, val_accuracy: {val_acc}")

        print("Segmentation train completed")

        # Classification train #TODO: detached encoder model and train separately

        if classification_train:
            print("-" * 5 + "Segmentation training start" + "-" * 5)

            cls_num_epochs = 20
            my_model.mode = 2

            train_loss = 0.
            train_acc = 0.

            val_loss = 0.
            val_acc = 0.

            my_model.train()

            params = [p for p in my_model.parameters() if p.requires_grad]

            optimizer = torch.optim.Adam(params, lr=0.005)
            bce = torch.nn.BCELoss(weight=[1, 1])

            for epoch in range(cls_num_epochs):
                for cls_x, cls_y in train_loader:
                    pred_y = my_model(cls_x)

                    loss = bce(pred_y, cls_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_acc += accuracy(pred_y, cls_y)

                train_loss /= len(train_loader)
                train_acc /= len(train_loader)

                my_model.eval()

                for cls_x, cls_y in val_loader:
                    pred_y = my_model(cls_x)

                    loss = bce(pred_y, cls_y)

                    val_loss += loss.item()
                    val_acc += accuracy(pred_y, cls_y)

                val_loss /= len(val_loader)
                val_iou /= len(val_loader)
                val_acc /= len(val_loader)

                print(f"Epoch {epoch}:")
                print("-" * 10)
                print(f"train_loss {train_loss}, train_accuracy: {train_acc}")
                print(f"val_loss {val_loss}, val_accuracy: {val_acc}")

            print("Classification train completed")

        #TODO: save weights method
        #TODO: save results by writing output.txt
        #TODO: make logger and maintain with logger class



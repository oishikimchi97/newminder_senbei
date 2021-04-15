import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from data import SegmentationDataset, ClassificationDataset
from model import DownconvUnet
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow

SMOOTH = 1e-6


class WeightedBCELoss(nn.Module):
    def __init__(self, weight: List):
        super().__init__()
        self.weight = weight

    def _weighted_binary_cross_entropy(self, input: torch.tensor, target: torch.tensor):
        return self.weight[0] * target * torch.log(input) + \
               self.weight[1] * (1 - target) * torch.log(1 - input)

    def forward(self, input: torch.tensor, target: torch.tensor):
        loss = self._weighted_binary_cross_entropy(input, target)
        return loss


class Params:
    def __init__(self, batch_size=8, num_epoch=100, lr=0.01, swa_lr=0.005, seed=2021, weight=[1., 1.]):
        self.swa_lr = swa_lr
        self.weight = weight
        self.lr = lr
        self.seed = seed
        self.num_epoch = num_epoch
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="batch size of both segmentation and classification training"
    )
    parser.add_argument(
        "--seg_epoch",
        default=100,
        type=int,
        help="the number of epoch in the segmentation training"
    )
    parser.add_argument(
        "--cls_epoch",
        default=20,
        type=int,
        help="the number of epoch in the classification training"
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="the learning rate of training"
    )
    parser.add_argument(
        "--swa_lr",
        default=0.005,
        type=float,
        help="the stochastic learning rate of training"
    )
    parser.add_argument(
        "--seg_weight",
        default=[0.1, 1],
        type=list,
        nargs='+',
        help="the weight of Binary Cross Entropy in the segmentation learning"
    )
    parser.add_argument(
        "--cls_weight",
        default=[1, 1],
        type=list,
        nargs='+',
        help="the weight of Binary Cross Entropy in the classification learning"
    )
    parser.add_argument(
        "--seed",
        default=2021,
        type=int,
        help="the random seed"
    )
    parser.add_argument(
        "--train_dir",
        default="/train_dir",
        type=str,
        help="the train data directory. it consists of the both ng and ok directorys, and they have img and mask folders."
    )
    parser.add_argument(
        "--val_dir",
        default="/val_dir",
        type=str,
        help="the validation data directory. it consists of the both ng and ok directorys, and they have img and mask folders."
    )

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_train = True
    classification_train = True

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)

    train_ok_dir = str(train_dir / "ok")
    train_mask_dir = str(train_dir / "mask")
    train_ng_dir = str(train_dir / "ng")

    val_ok_dir = str(val_dir / "ok")
    val_mask_dir = str(val_dir / "mask")
    val_ng_dir = str(val_dir / "ng")

    seg_train_dataset = SegmentationDataset(img_dir=train_ng_dir, mask_dir=train_mask_dir,
                                            n_channels=3, classes=1, train=True)
    seg_val_dataset = SegmentationDataset(img_dir=val_ng_dir, mask_dir=val_mask_dir,
                                          n_channels=3, classes=1, train=False)

    cls_train_dataset = ClassificationDataset(ok_dir=train_ok_dir, ng_dir=train_ng_dir,
                                              n_channels=3, classes=1, train=True)
    cls_val_dataset = ClassificationDataset(ok_dir=val_ok_dir, ng_dir=val_ng_dir,
                                            n_channels=3, classes=1, train=False)

    seg_train_loader = DataLoader(seg_train_dataset, batch_size=8, shuffle=True)
    seg_val_loader = DataLoader(seg_val_dataset, batch_size=8, shuffle=True)
    cls_train_loader = DataLoader(cls_train_dataset, batch_size=8, shuffle=True)
    cls_val_loader = DataLoader(cls_val_dataset, batch_size=8, shuffle=True)

    my_model = DownconvUnet(in_channel=3, seg_classes=1, cls_classes=2)
    avg_model = AveragedModel(my_model)

    my_model.to(device)
    avg_model.to(device)

    with mlflow.start_run() as run:
        seg_args = Params(args.batch_size, args.seg_epoch, args.lr, args.seed, args.seg_weight)
        cls_args = Params(args.batch_size, args.cls_epoch, args.lr, args.seed, args.cls_weight)
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
            bce = WeightedBCELoss(weight=seg_args.weight)
            swa_start = int(seg_args.num_epoch * 0.75)
            swa_scheduler = SWALR(optimizer, anneal_strategy='linear', anneal_epochs=swa_start,
                                  swa_lr=seg_args.swa_lr)

            for epoch in range(seg_args.num_epoch):
                for batch_idx, batch in enumerate(seg_train_loader):
                    batch = tuple(t.to(device) for t in batch)
                    seg_x, seg_y = batch

                    optimizer.zero_grad()

                    pred_y = my_model(seg_x)
                    loss = bce(pred_y, seg_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_metrics.update(pred_y, seg_y, loss.item())
                    train_iou += train_metrics.iou
                    train_acc += train_metrics.acc

                    step = epoch * len(seg_train_loader) + batch_idx
                    for metric, value in vars(train_metrics).items():
                        mlflow.log_metric(f"seg_train_{metric}", value, step=step)

                train_loss /= len(seg_train_loader)
                train_iou /= len(seg_train_loader)
                train_acc /= len(seg_train_loader)

                my_model.eval()

                for batch_idx, batch in enumerate(seg_val_loader):
                    batch = tuple(t.to(device) for t in batch)
                    seg_x, seg_y = batch
                    pred_y = my_model(seg_x)

                    loss = bce(pred_y, seg_y)

                    val_loss += loss.item()
                    val_metrics.update(pred_y, seg_y, val_loss)
                    val_iou += val_metrics.iou
                    val_acc += val_metrics.acc

                    step = epoch * len(seg_val_loader) + batch_idx
                    for metric, value in vars(val_metrics).items():
                        mlflow.log_metric(f"seg_val_{metric}", value, step=step)

                val_loss /= len(seg_val_loader)
                val_iou /= len(seg_val_loader)
                val_acc /= len(seg_val_loader)

                print(f"Epoch {epoch + 1}:")
                print("-" * 10)
                print(f"train_loss {train_loss:.3f}, train_iou: {train_iou:.3f}, "
                      f"train_accuracy: {train_acc:.3f}")
                print(f"val_loss {val_loss:.3f}, val_iou: {val_iou:.3f}, "
                      f"val_accuracy: {val_acc:.3f}")

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
                bce = WeightedBCELoss(weight=cls_args.weight)
                swa_start = int(cls_args.num_epoch * 0.75)
                swa_scheduler = SWALR(optimizer, anneal_strategy='linear', anneal_epochs=swa_start,
                                      swa_lr=cls_args.swa_lr)

                for epoch in range(cls_args.num_epoch):
                    for batch_idx, batch in enumerate(cls_train_loader):
                        batch = tuple(t.to(device) for t in batch)
                        cls_x, cls_y = batch
                        pred_y = my_model(cls_x)

                        loss = bce(pred_y, cls_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        train_metrics.update(pred_y, cls_y, train_loss)
                        train_acc += train_metrics.acc

                        step = epoch * len(seg_train_loader) + batch_idx
                        for metric, value in vars(train_metrics).items():
                            mlflow.log_metric(f"cls_train_{metric}", value, step=step)

                    train_loss /= len(seg_train_loader)
                    train_acc /= len(seg_train_loader)

                    my_model.eval()

                    for batch_idx, batch in enumerate(cls_val_loader):
                        batch = tuple(t.to(device) for t in batch)
                        cls_x, cls_y = batch
                        pred_y = my_model(cls_x)

                        loss = bce(pred_y, cls_y)

                        val_loss += loss.item()
                        val_metrics.update(pred_y, cls_y, loss.item())
                        val_acc += val_metrics.acc

                        step = epoch * len(seg_train_loader) + batch_idx
                        for metric, value in vars(val_metrics).items():
                            mlflow.log_metric(f"cls_train_{metric}", value, step=step)

                    val_loss /= len(seg_val_loader)
                    val_acc /= len(seg_val_loader)

                    print(f"Epoch {epoch + 1}:")
                    print("-" * 10)
                    print(f"train_loss {train_loss:.3f}, train_iou: {train_iou:.3f}, "
                          f"train_accuracy: {train_acc:.3f}")
                    print(f"val_loss {val_loss:.3f}, val_iou: {val_iou:.3f}, "
                          f"val_accuracy: {val_acc:.3f}")

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


if __name__ == "__main__":
    main()

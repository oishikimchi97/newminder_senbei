import torch
import numpy as np
import albumentations as albm
import glob
from PIL import Image


# TODO: Add ClassificiationDataset Class

class SegmentationDataset(torch.utils.data.Dataset):
    'Generates data for Keras'

    def __init__(self, img_dir, mask_dir, resize=(256, 192), n_channels=3, classes=1, train=False):
        'Initialization'
        self.img_paths = glob.glob(img_dir + '/*')

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.resize = resize

        self.n_channels = n_channels
        self.classes = classes
        self.train = train
        self.entries = None
        self.augment = albm.Compose([
            albm.GaussNoise(p=0.3),
            albm.RandomBrightness(limit=0.1, p=0.3),
            albm.ToFloat(max_value=1)
        ], p=1)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.img_paths)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        idx_img_path = self.img_paths[index]

        # Generate data
        X, y = self.data_generation(idx_img_path)

        if self.train is False:
            return X, np.array(y) / 255
        else:
            augmented = self.augment(image=X, mask=y)
            X = augmented['image']
            y = augmented['mask']

            return np.array(X), np.array(y) / 255

    def data_generation(self, img_path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        mask_path = img_path.replace(self.img_dir, self.mask_dir)
        # change file extension jpg to png
        mask_path = mask_path.replace(mask_path.split('.')[-1], 'png')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.resize:
            img = img.resize(self.resize)
            mask = mask.resize(self.resize)

        img = np.array(img)
        mask = np.array(mask)

        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, 2)
        mask = mask[..., np.newaxis]

        X = img
        y = mask
        y[y > 0] = 255

        return np.uint8(X), np.uint8(y)


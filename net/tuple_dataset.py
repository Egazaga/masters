import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage


class TupleDataset(Dataset):
    def __init__(self, path="data/triples/", train=True):
        paths = sorted(glob(path + "pic/*.png"))
        self.paths = paths[:int(len(paths) * 0.8)] if train else paths[int(len(paths) * 0.8):]
        labels = np.load(path + "/data.npy")
        self.labels = labels[:int(len(paths) * 0.8)] if train else labels[int(len(paths) * 0.8):]

        self.transforms = [
            CropZeros(),
            ToPILImage(),
            Resize((512, 512)),
            ToTensor()
        ]

    def __len__(self):
        return len(self.paths)

    def transform(self, img, idx):
        for t in self.transforms:
            if isinstance(t, CropZeros):
                img = t(img, idx)
            else:
                img = t(img)
        return img

    def __getitem__(self, idx):
        img1 = cv2.imread(self.paths[idx])[:, :, 0]
        img1 = self.transform(img1, idx)
        label1 = self.labels[idx][0][:4]

        img2 = cv2.imread(self.paths[idx])[:, :, 1]
        img2 = self.transform(img2, idx + self.__len__())
        label2 = self.labels[idx][1][:4]

        return img1, img2, label1, label2

    def visualize(self, idx):
        img1, img2, label1, label2 = self.__getitem__(idx)
        print(label1, label2)
        cv2.imshow("img1", img1.numpy().squeeze())
        cv2.imshow("img2", img2.numpy().squeeze())
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class CropZeros(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, img, idx=None):
        if idx is None or idx not in self.cache:
            # argwhere will give you the coordinates of every non-zero point
            true_points = np.argwhere(img)
            # take the smallest points and use them as the top left of your crop
            top_left = true_points.min(axis=0)
            # take the largest points and use them as the bottom right of your crop
            bottom_right = true_points.max(axis=0)
            self.cache[idx] = (top_left, bottom_right)
        else:
            top_left, bottom_right = self.cache[idx]
        out = img[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
              top_left[1]:bottom_right[1] + 1]  # inclusive
        return out

import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class TupleDataset(Dataset):
    def __init__(self, path="data/dataset/canny", train=True):
        paths = sorted(glob(path + "/*.png"))
        self.paths = paths[:int(len(paths) * 0.8)] if train else paths[int(len(paths) * 0.8):]
        labels = np.load(path + "/../canny.npy")
        self.labels = labels[:int(len(paths) * 0.8)] if train else labels[int(len(paths) * 0.8):]
        self.transforms = self.get_transforms()

    def __len__(self):
        return len(self.paths)

    def get_transforms(self):
        transform = [
            ToTensor()
        ]

        transform = Compose(transform)
        return transform

    def __getitem__(self, idx):
        img1 = cv2.imread(self.paths[idx])[:, :, 0]
        img1 = self.transforms(img1)
        label1 = self.labels[idx][0][:4]

        img2 = cv2.imread(self.paths[idx])[:, :, 1]
        img2 = self.transforms(img2)
        label2 = self.labels[idx][1][:4]

        return img1, img2, label1, label2

    def visualize(self, idx):
        img1, img2, label1, label2 = self.__getitem__(idx)
        print(label1, label2)
        cv2.imshow("img1", img1.numpy().squeeze())
        cv2.imshow("img2", img2.numpy().squeeze())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

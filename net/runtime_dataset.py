import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from interactive_canny import get_imgs
from utils.model_subset import get_models_subset
from utils.render import get_renderer, json_to_verts_faces, construct_mesh


class RuntimeDataset(Dataset):
    def __init__(self, train=True):
        self.subset = get_models_subset(20)
        self.meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if
                       i in self.subset]
        self.renderer = get_renderer(imsize=512)
        self.dist_range, self.elev_range, self.azim_range, self.class_id_range = (4.5, 12), (3, 45), (0, 360), \
            (0, len(self.meshes) - 1)
        self.transforms = self.get_transforms()
        self.len = 800 if train else 200

    def __len__(self):
        return self.len

    def get_transforms(self):
        transform = [
            ToTensor()
        ]

        transform = Compose(transform)
        return transform

    def __getitem__(self, idx):
        dist, elev, azim, class_id = np.random.uniform(*self.dist_range), np.random.uniform(*self.elev_range), \
            np.random.uniform(*self.azim_range), np.random.randint(*self.class_id_range)
        fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
        img1 = get_imgs(self.meshes[class_id], self.renderer, dist, elev, azim, fov)[1]
        label1 = torch.FloatTensor([dist, elev, azim, class_id])

        diff_scale = 0.1
        new_vals = []
        for orig, val_range in zip((dist, elev, azim), (self.dist_range, self.elev_range, self.azim_range)):
            deviation = (val_range[1] - val_range[0]) * diff_scale * random.random() / 2
            if random.random() > 0.5:
                deviation *= -1
            new_vals.append(orig + deviation)
        class_id2 = random.randint(*self.class_id_range)
        dist2, elev2, azim2 = new_vals
        fov2 = 2 * np.arctan(0.5 * 6 / dist2) * 180 / np.pi
        img2 = get_imgs(self.meshes[class_id2], self.renderer, dist2, elev2, azim2, fov2)[1]
        label2 = torch.FloatTensor([dist2, elev2, azim2, class_id2])

        return self.transforms(img1), self.transforms(img2), label1, label2

    def visualize(self, idx):
        img1, img2, label1, label2 = self.__getitem__(idx)
        print(label1, label2)
        cv2.imshow("img1", img1.numpy().squeeze())
        cv2.imshow("img2", img2.numpy().squeeze())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from net.model import MyModel
from net.tuple_dataset import TupleDataset
from train import labels_to_value, PF

# load MyModel
model = MyModel(out_channels=3).cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()

path = "data/triples5k/"
paths = sorted(glob(path + "pic/*.png"))
labels = np.load(path + "/data.npy")

res = []
for img_idx in range(-1, -100, -1):
    path = paths[img_idx]
    img1, img2 = cv2.imread(path)[:, :, 0], cv2.imread(path)[:, :, 1]
    img1, img2 = ToTensor()(img1).unsqueeze(0).cuda(), ToTensor()(img2).unsqueeze(0).cuda()
    labels1 = labels[img_idx][0][:4]
    labels2 = labels[img_idx][1][:4]
    pred = model(img1, img2)
    gt = labels_to_value(torch.tensor(labels1[None]).cuda(), torch.tensor(labels2[None]).cuda())
    pr, gt = pred.cpu().detach().numpy(), gt.cpu().detach().numpy()
    print(pr, gt)
    res.append((pr.mean(), gt.mean()))
# correlation
print(np.corrcoef(np.array(res).T))

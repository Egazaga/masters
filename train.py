# Load the MNIST dataset
from time import sleep

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from net.dataset import MyDataset
from net.model import MyModel


def loss_function(pred, labels1, labels2):
    are_equal_pred = pred
    dist1, elev1, azim1, class_id1, fov1 = torch.hsplit(labels1, 5)
    dist2, elev2, azim2, class_id2, fov2 = torch.hsplit(labels2, 5)
    gt = torch.eq(class_id1, class_id2).float()
    return torch.mean(torch.abs(are_equal_pred - gt))


train_dataset = MyDataset(train=True)
test_dataset = MyDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Initialize the VAE model and the optimizer
model = MyModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the VAE model
model.train()
for epoch in range(100):
    # train
    train_loss = 0
    for batch_idx, (imgs1, imgs2, labels1, labels2) in enumerate(train_loader):
        imgs1, imgs2, labels1, labels2 = imgs1.cuda(), imgs2.cuda(), labels1.cuda(), labels2.cuda()
        optimizer.zero_grad()
        pred = model(imgs1, imgs2)
        loss = loss_function(pred, labels1, labels2)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # test
    test_loss = 0
    for batch_idx, (imgs1, imgs2, labels1, labels2) in enumerate(test_loader):
        imgs1, imgs2, labels1, labels2 = imgs1.cuda(), imgs2.cuda(), labels1.cuda(), labels2.cuda()
        pred = model(imgs1, imgs2)
        loss = loss_function(pred, labels1, labels2)
        test_loss += loss.item()

    print('Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader), test_loss / len(test_loader)))

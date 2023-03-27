from time import sleep

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from net.disk_dataset import DiskDataset
from net.model import MyModel
from net.runtime_dataset import RuntimeDataset


def labels_to_value(labels1, labels2):
    err_ranges = torch.FloatTensor(((4.5, 12), (3, 45), (0, 360)))
    errs = torch.abs(labels1 - labels2)[..., :3]
    normalized_errs = errs / (err_ranges[:, 1] - err_ranges[:, 0]).cuda()

    # add column for class equality
    c1, c2 = labels1[..., 3].short(), labels2[..., 3].short()
    class_equal_err = ~torch.eq(c1, c2).unsqueeze(1)
    normalized_errs = torch.cat((normalized_errs, class_equal_err), dim=1)
    gt = torch.mean(normalized_errs, dim=1)
    return gt


def loss_function(pred, labels1, labels2):
    gt = labels_to_value(labels1, labels2)
    pred = pred.squeeze()
    loss = torch.mean(torch.abs(pred - gt))
    return loss


batch_size = 16
# train_dataset = DiskDataset(train=True)
# test_dataset = DiskDataset(train=False)
train_dataset = RuntimeDataset()
test_dataset = RuntimeDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# for i in range(100):
#     train_dataset.visualize(i)

model = MyModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        # print(pred[0], labels_to_value(labels1, labels2)[0])

        test_loss += loss.item()

    print('Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader), test_loss / len(test_loader)))

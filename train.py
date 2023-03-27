import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.svs_model import HQEncoder
from net.tuple_dataset import TupleDataset
from net.model import MyModel


class PF(float):
    def __repr__(self):
        return "%0.5f" % self


def labels_to_value(labels1, labels2):
    err_ranges = torch.FloatTensor(((4.5, 12), (3, 45), (0, 360)))
    errs = torch.abs(labels1 - labels2)[..., :3]
    deviation_percent = 0.1
    normalized_errs = errs / (err_ranges[:, 1] - err_ranges[:, 0]).cuda() / deviation_percent

    # add column for class equality
    # c1, c2 = labels1[..., 3].short(), labels2[..., 3].short()
    # class_equal_err = ~torch.eq(c1, c2).unsqueeze(1)
    # normalized_errs = torch.cat((normalized_errs, class_equal_err), dim=1)

    # gt = torch.mean(normalized_errs, dim=1)
    gt = normalized_errs[:, 2]
    return gt


# gts = []


def loss_function(pred, labels1, labels2):
    gt = labels_to_value(labels1, labels2)
    # gts.extend(gt.cpu().detach().numpy())
    pred = pred.squeeze()
    loss = torch.mean(torch.abs(pred - gt))
    return loss


batch_size = 4
train_dataset, test_dataset = TupleDataset(train=True), TupleDataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = HQEncoder().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for num_epoch, epoch in enumerate(range(100)):
    # train
    train_loss = 0
    for batch_idx, (imgs1, imgs2, labels1, labels2) in enumerate(tqdm(train_loader)):
        imgs1, imgs2, labels1, labels2 = imgs1.cuda(), imgs2.cuda(), labels1.cuda(), labels2.cuda()
        optimizer.zero_grad()
        pred = model(imgs1, imgs2)
        loss = loss_function(pred, labels1, labels2)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # plot gts distribution
    # plt.hist(gts, bins=10)
    # plt.show()

    # test
    test_loss = 0
    for batch_idx, (imgs1, imgs2, labels1, labels2) in enumerate(tqdm(test_loader)):
        imgs1, imgs2, labels1, labels2 = imgs1.cuda(), imgs2.cuda(), labels1.cuda(), labels2.cuda()
        pred = model(imgs1, imgs2)
        loss = loss_function(pred, labels1, labels2)
        if num_epoch > 5:
            print(PF(pred[0][0].item()), PF(labels_to_value(labels1, labels2)[0].item()))

        test_loss += loss.item()

    print('Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader), test_loss / len(test_loader)))

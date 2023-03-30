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
    normalized_errs = errs / (err_ranges[:, 1] - err_ranges[:, 0]).cuda()

    # add column for class equality
    # c1, c2 = labels1[..., 3].short(), labels2[..., 3].short()
    # class_equal_err = ~torch.eq(c1, c2).unsqueeze(1)
    # normalized_errs = torch.cat((normalized_errs, class_equal_err), dim=1)

    # return normalized_errs[:, [1, 2]]
    return normalized_errs[:, 2].unsqueeze(1)
    # return normalized_errs


# gts = []


def loss_function(pred, labels1, labels2):
    gt = labels_to_value(labels1, labels2)
    # gts.extend(gt.cpu().detach().numpy().flatten())
    assert pred.shape == gt.shape
    loss = torch.mean(torch.abs(pred - gt))
    return loss


if __name__ == '__main__':
    # dataset_path = "data/tuples5k/"
    # dataset_path = "data/triples5k/"
    dataset_path = "data/azim5k/"
    out_channels = 1
    multihead = True
    batch_size = 64

    train_dataset = TupleDataset(path=dataset_path, train=True)
    test_dataset = TupleDataset(path=dataset_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # for i in range(10):
    #     test_dataset.visualize(i)

    model = MyModel(out_channels=out_channels, multihead=multihead).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_test_loss = 1000

    for num_epoch in range(100):
        # train
        model.train()
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
        # plt.hist(gts, bins=50)
        # plt.show()

        # test
        torch.cuda.empty_cache()
        model.eval()
        test_loss = 0
        for batch_idx, (imgs1, imgs2, labels1, labels2) in enumerate(tqdm(test_loader)):
            imgs1, imgs2, labels1, labels2 = imgs1.cuda(), imgs2.cuda(), labels1.cuda(), labels2.cuda()
            with torch.no_grad():
                pred = model(imgs1, imgs2)
                loss = loss_function(pred, labels1, labels2)
            test_loss += loss.item()

            # if num_epoch > 5:
            #     print(PF(pred[0][0].item()), PF(labels_to_value(labels1, labels2)[0].item()))

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "model.pth")

        torch.cuda.empty_cache()

        print('Epoch: {} Train loss: {:.4f} Test loss: {:.4f}'.format(
            num_epoch, train_loss / len(train_loader), test_loss / len(test_loader)))

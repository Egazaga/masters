import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=(32, 64, 128, 256), imsize=512, out_channels=3, multihead=False):
        super(MyModel, self).__init__()

        self.multihead = multihead

        # encoder
        self.encoder = []
        for h_dim in hidden_dims:
            self.encoder.extend([
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = h_dim

        self.encoder = nn.Sequential(*self.encoder, nn.Flatten())

        # fc
        n_features = (imsize ** 2 // 4 ** len(hidden_dims)) * hidden_dims[-1] * 2
        head_sizes = [n_features, 128, 64, 32]
        if multihead:
            self.heads = nn.ModuleList()
            for i in range(out_channels):
                head = []
                for hs1, hs2 in zip(head_sizes, head_sizes[1:]):
                    head.extend([nn.Linear(hs1, hs2), nn.BatchNorm1d(hs2), nn.ReLU()])
                head.extend([nn.Linear(head_sizes[-1], 1), nn.Sigmoid()])
                self.heads.append(nn.Sequential(*head))
        else:
            head = []
            for hs1, hs2 in zip(head_sizes, head_sizes[1:]):
                head.extend([nn.Linear(hs1, hs2), nn.BatchNorm1d(hs2), nn.ReLU()])
            head.extend([nn.Linear(head_sizes[-1], out_channels), nn.Sigmoid()])
            self.head = nn.Sequential(*head)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.cat((x1, x2), dim=1)
        if self.multihead:
            x = torch.concatenate([head(x) for head in self.heads], dim=1)
        else:
            x = self.head(x)
        return x


if __name__ == "__main__":
    model = MyModel(multihead=True, out_channels=1)
    img = torch.randn(2, 1, 512, 512)

    print(model(img, img).shape)

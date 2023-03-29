import torch
from torch import nn

from net.csp import ConvolutionBlock, CspBlock


class HQEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        sizes = [16, 32, 64, 128, 256]
        self.features_x2 = nn.Sequential(
            ConvolutionBlock(1, sizes[0], kernel=3),
            ConvolutionBlock(sizes[0], sizes[1], kernel=3, stride=2),  # x2
            CspBlock(sizes[1], sizes[1], 2),
        )
        self.features_x4 = nn.Sequential(
            ConvolutionBlock(sizes[1], sizes[2], kernel=3, stride=2),  # x4
            CspBlock(sizes[2], sizes[2], 2),
        )
        self.features_x8 = nn.Sequential(
            ConvolutionBlock(sizes[2], sizes[3], kernel=3, stride=2),  # x8
            CspBlock(sizes[3], sizes[3], 2),
        )
        self.features_x16 = nn.Sequential(
            ConvolutionBlock(sizes[3], sizes[4], kernel=3, stride=2),  # x16
            CspBlock(sizes[4], sizes[4], 2),
        )
        # self.features_x32 = nn.Sequential(
        #     ConvolutionBlock(sizes[4], sizes[5], kernel=3, stride=2),  # x32
        #     CspBlock(sizes[5], sizes[5], 2),
        # )

        # fc
        n_features = sizes[-1] * 2 * (512 // 32) * (512 // 32)
        self.head = [nn.AvgPool2d(512 // 16, 512 // 16),
                     nn.Flatten(),
                     nn.Linear(sizes[-1] * 2, 256),
                     nn.BatchNorm1d(256),
                     nn.ReLU(),
                     nn.Linear(256, 3),
                     nn.Sigmoid()]
        self.head = nn.Sequential(*self.head)

    def encode(self, img):
        f_x2 = self.features_x2(img)
        f_x4 = self.features_x4(f_x2)
        f_x8 = self.features_x8(f_x4)
        f_x16 = self.features_x16(f_x8)
        # f_x32 = self.features_x32(f_x16)

        return {'x2': f_x2, 'x4': f_x4, 'x8': f_x8, 'x16': f_x16
            # , 'x32': f_x32
                }

    def forward(self, img1, img2):
        x1 = self.encode(img1)
        x2 = self.encode(img2)
        x = torch.cat((x1['x16'], x2['x16']), dim=1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    print(HQEncoder())

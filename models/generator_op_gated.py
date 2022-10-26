import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    # (convolution => [BN] => ReLU) * 2

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    # Downscaling with maxpooling then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    # Upscaling then double conv

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class optical_encoder(nn.Module):
    def __init__(self, in_channels = 2):
        super(optical_encoder, self).__init__()
        self.in_channels = in_channels
        feature_num = 64
        feature_num1 = 128
        feature_num2 = 256
        feature_num3 = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, feature_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_num, feature_num1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_num1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_num1, feature_num2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_num2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_num2, feature_num3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_num3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Bottleneck(nn.Module):  # 瓶颈层
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # x = torch.cat([torch.mul(x1, x2), x1], dim=1)
        # x = torch.cat([x1, x2], dim=1)
        x = torch.add(torch.mul(torch.sigmoid(x1), x2), x1)
        # return self.inconv(x)
        return x


class Net(nn.Module):
    def __init__(self, out_channel = 3):
        super(Net, self).__init__()
        self.rgb = 3
        self.op = 2
        self.length = 4
        self.inc = DoubleConv(self.rgb*self.length, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.op_encoder = optical_encoder(self.op*self.length)
        self.bottleneck = Bottleneck(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, out_channel)

    def forward(self, frame, optical):
        # two-streams
        frame_in = frame[:, :-self.rgb, :, :]
        frame_target = frame[:, -self.rgb:, :, :]
        x1 = self.inc(frame_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y1 = self.op_encoder(optical)
        x4_new = self.bottleneck(x4, y1)
        x = self.up1(x4_new, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        out = dict(frame_pred=torch.sigmoid(x), frame_target=frame_target,
                   op_embedding=y1, RGB_embedding=x4)
        return out


if __name__ == "__main__":
    op = torch.randn((1, 2, 256, 256))
    rgb_embedding = torch.randn((1, 512, 32, 32))
    op_encoder = optical_encoder()
    op_embedding = op_encoder(op)
    print(op_embedding.size())

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2. BatchNorm keeps activations from
    shrinking to near-zero as they pass through the five pooling stages,
    which otherwise starves the encoder's deeper layers of gradient."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        # Encoder
        self.e1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e5 = DoubleConv(512, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d1 = DoubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe1 = self.e1(x)
        xp1 = self.pool1(xe1)

        xe2 = self.e2(xp1)
        xp2 = self.pool2(xe2)

        xe3 = self.e3(xp2)
        xp3 = self.pool3(xe3)

        xe4 = self.e4(xp3)
        xp4 = self.pool4(xe4)

        xe5 = self.e5(xp4)

        # Decoder
        xu1 = self.upconv1(xe5)
        xd1 = self.d1(torch.cat([xu1, xe4], dim=1))   # skip connection from encoder

        xu2 = self.upconv2(xd1)
        xd2 = self.d2(torch.cat([xu2, xe3], dim=1))

        xu3 = self.upconv3(xd2)
        xd3 = self.d3(torch.cat([xu3, xe2], dim=1))

        xu4 = self.upconv4(xd3)
        xd4 = self.d4(torch.cat([xu4, xe1], dim=1))

        return self.outconv(xd4)


if __name__ == "__main__":
    from torchinfo import summary

    model = UNet(1, 1)
    summary(model, input_size=(1, 1, 128, 128))

import torch
import torch.nn as nn
from torch.nn.functional import relu


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe12 = relu(self.e12(relu(self.e11(x))))
        xp1 = self.pool1(xe12)

        xe22 = relu(self.e22(relu(self.e21(xp1))))
        xp2 = self.pool2(xe22)

        xe32 = relu(self.e32(relu(self.e31(xp2))))
        xp3 = self.pool3(xe32)

        xe42 = relu(self.e42(relu(self.e41(xp3))))
        xp4 = self.pool4(xe42)

        xe52 = relu(self.e52(relu(self.e51(xp4))))

        # Decoder
        xu1 = self.upconv1(xe52)
        xcat1 = torch.cat([xu1, xe42], dim=1)   # skip connection from encoder
        xd12 = relu(self.d12(relu(self.d11(xcat1))))

        xu2 = self.upconv2(xd12)
        xcat2 = torch.cat([xu2, xe32], dim=1)
        xd22 = relu(self.d22(relu(self.d21(xcat2))))

        xu3 = self.upconv3(xd22)
        xcat3 = torch.cat([xu3, xe22], dim=1)
        xd32 = relu(self.d32(relu(self.d31(xcat3))))

        xu4 = self.upconv4(xd32)
        xcat4 = torch.cat([xu4, xe12], dim=1)
        xd42 = relu(self.d42(relu(self.d41(xcat4))))

        return self.outconv(xd42)


if __name__ == "__main__":
    model = UNet(n_class=1)
    dummy = torch.randn(1, 3, 128, 128)
    print("Output shape:", model(dummy).shape)  # expected: [1, 1, 128, 128]
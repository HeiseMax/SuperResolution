import torch.nn as nn
import torch


# Low Resolution Encoder Module
class AELREncoder(nn.Module):
    def __init__(self, in_channels, channels, latent_dims, base_width):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # base_width x base_width
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # base_width/2 x base_width/2
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)  # base_width/4 x base_width/4
        
        width = base_width // 4
        self.fc = nn.Linear(channels[2] * width * width, latent_dims[0])

    def forward(self, x):
        x0 = self.in_conv(x)  # base_width x base_width
        x1 = self.down1(x0)  # base_width/2 x base_width/2
        x2 = self.down2(x1)  # base_width/4 x base_width/4

        z = self.fc(x2.view(x2.size(0), -1))
        return z


# Decoder Module
class AEDecoder(nn.Module):
    def __init__(self, out_channels, channels, latent_dims):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dims[0], channels[3] * 4 * 4 * 64)
        
        self.deconv1 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)  # 16x16 -> 32x32

        self.out_conv = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, z):
        x1 = self.fc(z).view(-1, self.channels[3], 4*8, 4*8)

        x2 = self.deconv1(x1)  # Expected: (batch, 64, 16, 16)
        x2 = nn.ReLU()(x2)
        x3 = self.deconv2(x2)  # Expected: (batch, 32, 32, 32)
        x3 = nn.ReLU()(x3)
        x4 = self.deconv3(x3)
        output = self.out_conv(x4)
        return torch.sigmoid(output)


# VAE Module
class AE(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], latent_dims=[64, 128, 256], base_width=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels = channels
        self.loss =[]
        self.encoder = AELREncoder(in_channels, channels, latent_dims, base_width)
        self.decoder = AEDecoder(in_channels, channels, latent_dims)

    def forward(self, x_lr):
        z = self.encoder(x_lr)
        recon_x = self.decoder(z)
        return recon_x
    
    def sample(self, x_lr):
        return self.forward(x_lr)

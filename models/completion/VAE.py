import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import Resize
from utils.utils import sample_latent


# Low Resolution Encoder Module
class VAELREncoder(nn.Module):
    def __init__(self, in_channels, channels, latent_dims, base_width):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # base_width x base_width
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # base_width/2 x base_width/2
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)  # base_width/4 x base_width/4
        
        width = base_width // 4
        self.fc_mu = nn.Linear(channels[2] * width * width, latent_dims[0])
        self.fc_logvar = nn.Linear(channels[2] * width * width, latent_dims[0])

    def forward(self, x):
        x0 = self.in_conv(x)  # base_width x base_width
        x1 = self.down1(x0)  # base_width/2 x base_width/2
        x2 = self.down2(x1)  # base_width/4 x base_width/4

        mu, logvar = self.fc_mu(x2.view(x2.size(0), -1)), self.fc_logvar(x2.view(x2.size(0), -1))
        return [(mu, logvar)]


# Decoder Module
class VAEDecoder(nn.Module):
    def __init__(self, out_channels, channels, latent_dims):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dims[0], channels[3] * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)  # 16x16 -> 32x32

        self.out_conv = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, z):
        z, = z
        x1 = self.fc(z).view(-1, self.channels[3], 4, 4)

        x2 = self.deconv1(x1)  # Expected: (batch, 64, 16, 16)
        x2 = nn.ReLU()(x2)
        x3 = self.deconv2(x2)  # Expected: (batch, 32, 32, 32)
        x3 = nn.ReLU()(x3)
        x4 = self.deconv3(x3)
        output = self.out_conv(x4)
        return torch.tanh(output)


# VAE Module
class VAE(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], latent_dims=[64, 128, 256], base_width=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels = channels

        self.encoder = VAELREncoder(in_channels, channels, latent_dims, base_width)
        self.decoder = VAEDecoder(in_channels, channels, latent_dims)
        self.combiner = nn.Conv2d(in_channels*2, in_channels, 1, 1, 0)

    def forward(self, x_lr, x_lr_up):
        latents = self.encoder(x_lr)

        # latent sampling with projection
        z = sample_latent(latents[0][0], latents[0][1])

        sampled_latents = [z]
        output = self.decoder(sampled_latents)
        # recon_x = F.sigmoid(self.combiner(torch.cat([output, x_lr_up], dim=1)))
        recon_x = output + x_lr_up
        recon_x = torch.clamp(recon_x, 0, 1)
        return recon_x, latents
    
    def sample(self, x_lr):
        x_lr_up = Resize((32, 32))(x_lr)
        return self.forward(x_lr, x_lr_up)[0]



# Conditional Encoder Module
class ConditionalVAEEncoder(nn.Module):
    def __init__(self, in_channels, channels, latent_dims):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # 32x32 -> 32x32
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)  # 16x16 -> 8x8
        self.down3 = nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=1)  # 8x8 -> 4x4

        self.fc_mu = nn.Linear(channels[3] * 4 * 4, latent_dims[0])
        self.fc_logvar = nn.Linear(channels[3] * 4 * 4, latent_dims[0])

    def forward(self, x):
        x0 = self.in_conv(x)  # 32x32 -> 32x32
        x1 = self.down1(x0)  # 32x32 -> 16x16
        x2 = self.down2(x1)  # 16x16 -> 8x8
        x3 = self.down3(x2)  # 8x8 -> 4x4

        mu, logvar = self.fc_mu(x3.view(x3.size(0), -1)), self.fc_logvar(x3.view(x3.size(0), -1))        
        return [(mu, logvar)]


# Conditional Decoder Module
class ConditionalVAEDecoder(nn.Module):
    def __init__(self, out_channels, channels, latent_dims, condition_dims):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dims[0], channels[3] * 4 * 4)

        self.fc_cond1 = nn.Linear(condition_dims[0], channels[3] * 4 * 4)
        self.fc_cond2 = nn.Linear(condition_dims[1], channels[2] * 8 * 8)
        
        self.deconv1 = nn.ConvTranspose2d(channels[3] * 2, channels[2], 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)  # 16x16 -> 32x32

        self.out_conv = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, z, condition):
        z, = z
        c1, c2 = condition

        z = self.fc(z).view(-1, self.channels[3], 4, 4)

        c1 = self.fc_cond1(c1).view(-1, self.channels[3], 4, 4)
        c2 = self.fc_cond2(c2).view(-1, self.channels[2], 8, 8)

        x1 = torch.cat([z, c1], dim=1)
        x2 = self.deconv1(x1)  # Expected: (batch, 64, 16, 16)
        x2 = nn.ReLU()(x2)
        x2 = torch.cat([x2, c2], dim=1)
        x3 = self.deconv2(x2)  # Expected: (batch, 32, 32, 32)
        x3 = nn.ReLU()(x3)
        x4 = self.deconv3(x3)
        output = self.out_conv(x4)
        output = nn.Tanh()(output)
        return output


# Conditional Low Resolution Encoder Module
class ConditionalVAELREncoder(nn.Module):
    def __init__(self, in_channels, channels, condition_dims, base_width):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # base_width x base_width
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # base_width/2 x base_width/2
        
        width = base_width // 2
        self.fc1 = nn.Linear(channels[1] * width * width, condition_dims[0])
        self.fc2 = nn.Linear(channels[0] * base_width * base_width, condition_dims[1])

    def forward(self, x):
        x0 = self.in_conv(x)  # base_width x base_width
        x1 = self.down1(x0)  # base_width/2 x base_width/2

        c1 = self.fc1(x1.view(x1.size(0), -1))
        c2 = self.fc2(x0.view(x0.size(0), -1))

        return [c1, c2]
    
class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], cond_channels=[32, 64], latent_dims=[64, 128, 256], condition_dims=[64, 128], cond_base_width=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.condition_dims = condition_dims
        self.channels = channels

        self.encoder = ConditionalVAEEncoder(in_channels, channels, latent_dims)
        self.lr_encoder = ConditionalVAELREncoder(in_channels,cond_channels, condition_dims, cond_base_width)
        self.decoder = ConditionalVAEDecoder(in_channels, channels, latent_dims, condition_dims)
        self.combiner = nn.Conv2d(in_channels*2, in_channels, 1, 1, 0)

    def forward(self, x, x_lr, x_lr_up):
        latents = self.encoder(x)
        condition = self.lr_encoder(x_lr)

        # latent sampling with projection
        z = sample_latent(latents[0][0], latents[0][1])

        sampled_latents = [z]
        output = self.decoder(sampled_latents, condition)
        # recon_x = F.sigmoid(self.combiner(torch.cat([output, x_lr_up], dim=1)))
        recon_x = output + x_lr_up
        recon_x = torch.clamp(recon_x, 0, 1)
        return recon_x, latents
    
    def sample(self, x_lr):
        num_samples = x_lr.size(0)
        device = next(self.parameters()).device
        x_lr_up = Resize((32, 32))(x_lr)

        z = torch.randn(num_samples, self.latent_dims[0]).to(device)
        sampled_latents = [z]
        
        condition = self.lr_encoder(x_lr)
        output = self.decoder(sampled_latents, condition)
        # recon_x = F.sigmoid(self.combiner(torch.cat([output, x_lr_up], dim=1)))
        recon_x = output + x_lr_up
        recon_x = torch.clamp(recon_x, 0, 1)
        return recon_x

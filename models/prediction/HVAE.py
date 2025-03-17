import torch.nn as nn
import torch
from utils.utils import sample_latent

import torch.nn as nn
import torch
from utils.utils import sample_latent


# Low Resolution Encoder Module
class HVAELREncoder(nn.Module):
    def __init__(self, in_channels, channels, latent_dims, base_width):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # base_width x base_width
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # base_width/2 x base_width/2
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)  # base_width/4 x base_width/4
        
        width = base_width // 4
        self.fc_mu1 = nn.Linear(channels[2] * width * width, latent_dims[0])
        self.fc_logvar1 = nn.Linear(channels[2] * width * width, latent_dims[0])

        width = width * 2
        self.fc_mu2 = nn.Linear(channels[1] * 8 * 8, latent_dims[1])
        self.fc_logvar2 = nn.Linear(channels[1] * 8 * 8, latent_dims[1])

        width = width * 2
        self.fc_mu3 = nn.Linear(channels[0] * 16 * 16, latent_dims[2])
        self.fc_logvar3 = nn.Linear(channels[0] * 16 * 16, latent_dims[2])

    def forward(self, x):
        x0 = self.in_conv(x)  # base_width x base_width
        x1 = self.down1(x0)  # base_width/2 x base_width/2
        x2 = self.down2(x1)  # base_width/4 x base_width/4

        mu1, logvar1 = self.fc_mu1(x2.view(x2.size(0), -1)), self.fc_logvar1(x2.view(x2.size(0), -1))
        mu2, logvar2 = self.fc_mu2(x1.view(x1.size(0), -1)), self.fc_logvar2(x1.view(x1.size(0), -1))
        mu3, logvar3 = self.fc_mu3(x0.view(x0.size(0), -1)), self.fc_logvar3(x0.view(x0.size(0), -1))
        return [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]


# Decoder Module
class HVAEDecoder(nn.Module):
    def __init__(self, out_channels, channels, latent_dims):
        super().__init__()
        self.channels = channels
        self.fc1 = nn.Linear(latent_dims[0], channels[3] * 4 * 4)
        self.fc2 = nn.Linear(latent_dims[1], channels[2] * 8 * 8)
        self.fc3 = nn.Linear(latent_dims[2], channels[1] * 16 * 16)
        
        self.deconv1 = nn.ConvTranspose2d(channels[3], channels[2], 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(channels[2], channels[1], 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)  # 16x16 -> 32x32

        self.out_conv = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, z):
        z1, z2, z3 = z
        z1 = self.fc1(z1).view(-1, self.channels[3], 4, 4)
        z2 = self.fc2(z2).view(-1, self.channels[2], 8, 8)
        z3 = self.fc3(z3).view(-1, self.channels[1], 16, 16)

        x0 = z1
        x1 = self.deconv1(x0)
        x1 = nn.ReLU()(x1)
        x1 = x1 + z2
        x2 = self.deconv2(x1)
        x2 = nn.ReLU()(x2)
        x2 = x2 + z3
        x3 = self.deconv3(x2)
        output = self.out_conv(x3)
        return torch.sigmoid(output)


# VAE Module
class HVAE(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], latent_dims=[64, 128, 256], base_width=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.channels = channels

        self.encoder = HVAELREncoder(in_channels, channels, latent_dims, base_width)
        self.decoder = HVAEDecoder(in_channels, channels, latent_dims)

    def forward(self, x_lr):
        latents = self.encoder(x_lr)

        # latent sampling with projection
        z1 = sample_latent(latents[0][0], latents[0][1])
        z2 = sample_latent(latents[1][0], latents[1][1])
        z3 = sample_latent(latents[2][0], latents[2][1])

        sampled_latents = [z1, z2, z3]
        recon_x = self.decoder(sampled_latents)
        return recon_x, latents
    
    def sample(self, x_lr):
        return self.forward(x_lr)[0]



# Conditional Encoder Module
class ConditionalHierarchicalEncoder(nn.Module):
    def __init__(self, in_channels, channels, latent_dims):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 1, 1, 0)  # 32x32 -> 32x32
        self.down1 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.down2 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=1)  # 16x16 -> 8x8
        self.down3 = nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=1)  # 8x8 -> 4x4

        self.fc_mu1 = nn.Linear(channels[3] * 4 * 4, latent_dims[0])
        self.fc_logvar1 = nn.Linear(channels[3] * 4 * 4, latent_dims[0])

        self.fc_mu2 = nn.Linear(channels[2] * 8 * 8, latent_dims[1])
        self.fc_logvar2 = nn.Linear(channels[2] * 8 * 8, latent_dims[1])

        self.fc_mu3 = nn.Linear(channels[1] * 16 * 16, latent_dims[2])
        self.fc_logvar3 = nn.Linear(channels[1] * 16 * 16, latent_dims[2])

    def forward(self, x):
        x0 = self.in_conv(x)  # 32x32 -> 32x32
        x1 = self.down1(x0)  # 32x32 -> 16x16
        x2 = self.down2(x1)  # 16x16 -> 8x8
        x3 = self.down3(x2)  # 8x8 -> 4x4

        mu1, logvar1 = self.fc_mu1(x3.view(x3.size(0), -1)), self.fc_logvar1(x3.view(x3.size(0), -1))
        mu2, logvar2 = self.fc_mu2(x2.view(x2.size(0), -1)), self.fc_logvar2(x2.view(x2.size(0), -1))
        mu3, logvar3 = self.fc_mu3(x1.view(x1.size(0), -1)), self.fc_logvar3(x1.view(x1.size(0), -1))
        
        return [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]
    

# Conditional Decoder Module
class ConditionalHierarchicalDecoder(nn.Module):
    def __init__(self, out_channels, channels, latent_dims, condition_dims):
        super().__init__()
        self.channels = channels
        self.fc1 = nn.Linear(latent_dims[0], channels[3] * 4 * 4)
        self.fc2 = nn.Linear(latent_dims[1], channels[2] * 8 * 8)
        self.fc3 = nn.Linear(latent_dims[2], channels[1] * 16 * 16)

        self.fc_cond1 = nn.Linear(condition_dims[0], channels[3] * 4 * 4)
        self.fc_cond2 = nn.Linear(condition_dims[1], channels[2] * 8 * 8)
        
        self.deconv1 = nn.ConvTranspose2d(channels[3] * 2, channels[2], 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(channels[1], channels[0], 4, stride=2, padding=1)  # 16x16 -> 32x32

        self.out_conv = nn.Conv2d(channels[0], out_channels, 1, 1, 0)

    def forward(self, z_hierarchical, condition):
        z1, z2, z3 = z_hierarchical
        c1, c2 = condition

        z1 = self.fc1(z1).view(-1, self.channels[3], 4, 4)
        z2 = self.fc2(z2).view(-1, self.channels[2], 8, 8)
        z3 = self.fc3(z3).view(-1, self.channels[1], 16, 16)

        c1 = self.fc_cond1(c1).view(-1, self.channels[3], 4, 4)
        c2 = self.fc_cond2(c2).view(-1, self.channels[2], 8, 8)

        x1 = torch.cat([z1, c1], dim=1)
        x2 = self.deconv1(x1)  # Expected: (batch, 64, 16, 16)
        x2 = nn.ReLU()(x2)
        x2 = x2 + z2
        x2 = torch.cat([x2, c2], dim=1)
        x3 = self.deconv2(x2)  # Expected: (batch, 32, 32, 32)
        x3 = nn.ReLU()(x3)
        x3 = x3 + z3
        x4 = self.deconv3(x3)
        output = self.out_conv(x4)
        return torch.sigmoid(output)
    

# Conditional Low Resolution Encoder Module
class ConditionalHierarchicalLREncoder(nn.Module):
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
    
# Conditional HVAE
class ConditionalHierarchicalVAE(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256], cond_channels=[32, 64], latent_dims=[64, 128, 256], condition_dims=[64, 128], cond_base_width=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.condition_dims = condition_dims
        self.channels = channels

        self.encoder = ConditionalHierarchicalEncoder(in_channels, channels, latent_dims)
        self.lr_encoder = ConditionalHierarchicalLREncoder(in_channels,cond_channels, condition_dims, cond_base_width)
        self.decoder = ConditionalHierarchicalDecoder(in_channels, channels, latent_dims, condition_dims)

    def forward(self, x, x_lr):
        latents = self.encoder(x)
        condition = self.lr_encoder(x_lr)

        # Hierarchical latent sampling with projection
        z1 = sample_latent(latents[0][0], latents[0][1])
        z2 = sample_latent(latents[1][0], latents[1][1])
        z3 = sample_latent(latents[2][0], latents[2][1])

        sampled_latents = [z1, z2, z3]
        recon_x = self.decoder(sampled_latents, condition)
        return recon_x, latents
    
    def sample(self, x_lr):
        num_samples = x_lr.size(0)
        device = next(self.parameters()).device

        z1 = torch.randn(num_samples, self.latent_dims[0]).to(device)
        z2 = torch.randn(num_samples, self.latent_dims[1]).to(device)
        z3 = torch.randn(num_samples, self.latent_dims[2]).to(device)
        sampled_latents = [z1, z2, z3]
        
        condition = self.lr_encoder(x_lr)
        recon_x = self.decoder(sampled_latents, condition)
        return recon_x

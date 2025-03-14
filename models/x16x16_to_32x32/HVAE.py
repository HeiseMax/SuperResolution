import torch.nn as nn
import torch
from utils.utils import sample_latent

# Encoder Module
class HierarchicalLREncoder(nn.Module):
    def __init__(self, in_channels=3, base_width=32, latent_dims=None):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 32, 1, 1, 0)  # 16x16
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 8x8
        
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(32 * 16 * 16, 128)

    def forward(self, x):
        x0 = self.in_conv(x)  # Expected: (batch, 32, 16, 16)
        x1 = self.down1(x0)  # Expected: (batch, 64, 8, 8)

        c1 = self.fc1(x1.view(x1.size(0), -1))
        c2 = self.fc2(x0.view(x0.size(0), -1))

        return [c1, c2]
    
class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3, base_width=32, latent_dims=None):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 32, 1, 1, 0)  # 32x32
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 16x16
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 8x8
        
        self.fc_mu1 = nn.Linear(128 * 8 * 8, 64)
        self.fc_logvar1 = nn.Linear(128 * 8 * 8, 64)

        self.fc_mu2 = nn.Linear(64 * 16 * 16, 128)
        self.fc_logvar2 = nn.Linear(64 * 16 * 16, 128)

        self.fc_mu3 = nn.Linear(32 * 32 * 32, 256)
        self.fc_logvar3 = nn.Linear(32 * 32 * 32, 256)

    def forward(self, x):
        x0 = self.in_conv(x)  # Expected: (batch, 32, 32, 32)
        x1 = self.down1(x0)  # Expected: (batch, 32, 16, 16)
        x2 = self.down2(x1)  # Expected: (batch, 64, 8, 8)

        mu1, logvar1 = self.fc_mu1(x2.view(x2.size(0), -1)), self.fc_logvar1(x2.view(x2.size(0), -1))
        mu2, logvar2 = self.fc_mu2(x1.view(x1.size(0), -1)), self.fc_logvar2(x1.view(x1.size(0), -1))
        mu3, logvar3 = self.fc_mu3(x0.view(x0.size(0), -1)), self.fc_logvar3(x0.view(x0.size(0), -1))

        return [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]


# Decoder Module
class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dims=[32, 64, 128, 256], base_width=32, out_channels=3):
        super().__init__()
        self.fc1 = nn.Linear(64, 128 * 8 * 8)
        self.fc2 = nn.Linear(128, 64 * 16 * 16)
        self.fc3 = nn.Linear(256, 32 * 32 * 32)

        self.fc_cond1 = nn.Linear(64, 128 * 8 * 8)
        self.fc_cond2 = nn.Linear(128, 64 * 16 * 16)
        
        self.deconv1 = nn.ConvTranspose2d(128*2, 64, 4, stride=2, padding=1)  # 16x16
        self.deconv2 = nn.ConvTranspose2d(64*2, 32, 4, stride=2, padding=1)  # 32x32

        self.out_conv = nn.Conv2d(32, 3, 1, 1, 0)

    def forward(self, z_hierarchical, condition):
        z1, z2, z3 = z_hierarchical
        c1, c2 = condition

        z1 = self.fc1(z1).view(-1, 128, 8, 8)
        z2 = self.fc2(z2).view(-1, 64, 16, 16)
        z3 = self.fc3(z3).view(-1, 32, 32, 32)

        c1 = self.fc_cond1(c1).view(-1, 128, 8, 8)
        c2 = self.fc_cond2(c2).view(-1, 64, 16, 16)
        
        x1 = torch.cat([z1, c1], dim=1)
        x2 = self.deconv1(x1)  # Expected: (batch, 64, 16, 16)
        x2 = nn.ReLU()(x2)
        x2 = x2 + z2
        x2 = torch.cat([x2, c2], dim=1)
        x3 = self.deconv2(x2)  # Expected: (batch, 32, 32, 32)
        x3 = nn.ReLU()(x3)
        x3 = x3 + z3
        x3 = self.out_conv(x3)
        return torch.sigmoid(x3)


# Hierarchical VDVAE Model
class HierarchicalVDVAE(nn.Module):
    def __init__(self, in_channels=3, base_width=32, latent_dims = [64, 128, 256]):
        super().__init__()
        self.encoder = HierarchicalEncoder(in_channels, base_width, latent_dims)
        self.lr_encoder = HierarchicalLREncoder(in_channels, base_width, latent_dims)
        self.decoder = HierarchicalDecoder(latent_dims, base_width, in_channels)

        # Add Linear layers to project latent dimensions before conditioning
        # self.project_z1 = nn.Linear(latent_dims[0], latent_dims[1])  # Projects 32 → 64
        # self.project_z2 = nn.Linear(latent_dims[1], latent_dims[2])  # Projects 64 → 128

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
        z1 = torch.randn(num_samples, 64).to(next(self.parameters()).device)
        z2 = torch.randn(num_samples, 128).to(next(self.parameters()).device)
        z3 = torch.randn(num_samples, 256).to(next(self.parameters()).device)
        sampled_latents = [z1, z2, z3]
        condition = self.lr_encoder(x_lr)
        recon_x = self.decoder(sampled_latents, condition)
        return recon_x

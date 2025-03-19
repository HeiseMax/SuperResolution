import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Utility Functions
def gaussian_kl(mu1, logsigma1, mu2, logsigma2):
    """Computes the KL divergence between two diagonal Gaussians."""
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def sample_latent(mu, logvar, prev_z=None):
    """Samples a latent variable conditioned on the previous latent."""
    eps = torch.empty_like(mu).normal_(0., 1.)
    z = torch.exp(logvar) * eps + mu
    if prev_z is not None:
        z = z + prev_z
    return z


def get_conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
    """Creates a 2D convolutional layer."""
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
    nn.init.kaiming_normal_(c.weight)
    return c


def get_3x3(in_dim, out_dim):
    """Creates a 3x3 convolutional layer."""
    return get_conv(in_dim, out_dim, 3, 1, 1)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]  # Use first layers
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.vgg = vgg

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.mse_loss(pred_features, target_features)  # Feature-level loss


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

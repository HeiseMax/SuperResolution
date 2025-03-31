import torch.nn as nn
from torchvision.transforms import Resize


# Returns the interpolated image
class InterpolationModel(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.size = size

    def forward(self, LR):
        size = self.size
        SR = Resize((size, size))(LR)
        return SR
    
    def sample(self, LR):
        return self(LR)

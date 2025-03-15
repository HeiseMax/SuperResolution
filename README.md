# Super-Resolution Imaging with Autoencoders
by Max Heise, Kevin Heibel, Marcel Kessler

## 🔧 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/HeiseMax/SuperResolution.git
cd SuperResolution
pip install -r requirements.txt
```

## 🤖 Models
All available models are stored in ```/models```
  - Autoencoder
  - Variational Autoencoder
  - Hierarchical Variational Autoencoder

Example usage of a model can be found in ```model_template.ipynb```

```python
from models.x16x16_to_32x32.HVAE import HierarchicalVDVAE

vdvae = HierarchicalVDVAE()
print(vdvae)
```

All trained models can be found on Heibox:
[Models](https://heibox.uni-heidelberg.de/d/234abe7858064a32bf2c/)

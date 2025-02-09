<div align="center">
    <img src="assets/banner.png" width="50%" alt="Overcomplete logo" align="center" />
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.8, 3.9, 3.10-efefef">
    </a>
    <a href="https://github.com/fel-thomas/Overcomplete/actions/workflows/lint.yml/badge.svg">
        <img alt="PyLint" src="https://github.com/fel-thomas/Overcomplete/actions/workflows/lint.yml/badge.svg">
    </a>
    <a href="https://github.com/fel-thomas/Overcomplete/actions/workflows/tox.yml/badge.svg">
        <img alt="Tox" src="https://github.com/fel-thomas/Overcomplete/actions/workflows/tox.yml/badge.svg">
    </a>
    <a href="https://github.com/fel-thomas/Overcomplete/actions/workflows/publish.yml/badge.svg">
        <img alt="Pypi" src="https://github.com/fel-thomas/Overcomplete/actions/workflows/publish.yml/badge.svg">
    </a>
    <a href="https://kempnerinstitute.github.io/Overcomplete/">
        <img src="https://img.shields.io/badge/Documentation-Online-00BCD4">
    </a>
    <a href="https://pepy.tech/project/overcomplete">
        <img alt="Pepy" src="https://static.pepy.tech/badge/overcomplete">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>

**Overcomplete** is a compact research library in Pytorch designed to study (Overcomplete)-Dictionary learning methods to extract concepts from large **Vision models**. In addition, this repository also introduces various visualization methods, attribution and metrics. However, Overcomplete emphasizes **experimentation**.

# 🚀 Getting Started with Overcomplete

Overcomplete requires Python 3.8 or newer and several dependencies, including Numpy. It supports both only Torch. Installation is straightforward with Pypi:

```bash
pip install overcomplete
```

With Overcomplete installed, you can dive into an optimisation based dictionary learning method to extract visual features or use the latest SAEs variant. The API is designed to be intuitive, requiring only a few hyperparameters to get started.

Example usage:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from overcomplete.sae import TopKSAE, train_sae

Activations = torch.randn(N, d)
sae = TopKSAE(d, nb_concepts=16_000, top_k=10, device='cuda')

dataloader = DataLoader(TensorDataset(Activations), batch_size=1024)
optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

def criterion(x, x_hat, pre_codes, codes, dictionary):
  mse = (x - x_hat).square().mean()
  return mse

logs = train_sae(sae, dataloader, criterion, optimizer,
                 nb_epochs=20, device='cuda')
```

# 🧪 Notebooks

- Getting started: [![Open](https://img.shields.io/badge/Starter-Notebook-00BCD4?style=flat&logo=jupyter)](https://colab.research.google.com/drive/1rB71_RdmCzr50I1Ebwfq49cEqoxt1G3X?usp=drive_link)
- TopK vs JumpReLU vs Vanilla SAE: _Coming soon_
- Stable Dictionary with Archetypal-SAE: _Coming soon_
- Advanced metrics to study the solution of SAE: _Coming soon_
- The visualization module: _Coming soon_
- NMF, ConvexNMF and Semi-NMF: _Coming soon_
- Modern Feature visualization to visualize concepts: _Coming soon_


# 👏 Credits
<div align="right">
  <picture>
    <source srcset="https://kempnerinstitute.harvard.edu/app/uploads/2024/08/Kempner-logo_Full-Color-Kempner-and-Harvard-Logo-Lockup-2048x552.png"  width="60%" align="right">
    <img  style="background-color: rgba(255, 255, 255, 0.8)" alt="Kempner Logo" src="https://kempnerinstitute.harvard.edu/app/uploads/2024/08/Kempner-logo_Full-Color-Kempner-and-Harvard-Logo-Lockup-2048x552.png" width="60%" align="right">
  </picture>
</div>

This work has been made possible in part by the generous support provided by the Kempner Institute at Harvard University. The institute, established through a gift from the Chan Zuckerberg Initiative Foundation, is dedicated to advancing research in natural and artificial intelligence. The resources and commitment of the Kempner Institute have been instrumental in the development and completion of this project.

# Additional Resources

For a complete LLM implementation of the SAE, we strongly recommend exploring the following resources. The Sparsify library by EleutherAI (https://github.com/EleutherAI/sparsify) provides a comprehensive toolset for implementing the SAE. The original TopK implementation is available through OpenAI's Sparse Autoencoder (https://github.com/openai/sparse_autoencoder). Additionally, SAE Lens by jbloomAus (https://github.com/jbloomAus/SAELens) is an excellent resource, especially if you are interested in using the SAE-vis associated tools found at (https://github.com/callummcdougall/sae_vis).

# Citation

```
@article{coming soon}
```


# Authors

- [Thomas Fel](https://thomasfel.me) - tfel@g.harvard.edu, Kempner Research Fellow, Harvard University.
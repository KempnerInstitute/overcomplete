# JumpReLU SAE

[JumpReLU SAE](https://arxiv.org/html/2407.14435) is a variation of the standard Sparse Autoencoder (SAE) that incorporates the JumpReLU activation function to have an adaptive sparsity without shriking.
This involve a learnable thresholding mechanism on each concept. As all SAEs, it include an encoder, a decoder, and a forward method.

- `encode` returns the pre-codes (`z_pre`, before ReLU) and codes (`z`) given an input (`x`).
- `decode` returns a reconstructed input (`x_hat`) based on an input (`x`).
- `forward` returns the pre-codes, codes, and reconstructed input.

kernel='silverman', bandwith=1e-3,

The specificity of this architecture is that i contains 2 hyperparamter, a bandwith and a kernel.
We strongly encourage you to check the original paper [^1] to know more about JumpReLU.

###  Basic Usage
```python
from overcomplete import JumpSAE

# define a JumpReLU SAE with input dimension 768 and 10k concepts
sae = JumpSAE(768, 10_000)

# adjust kernel and bandwith
sae = JumpSAE(768, 10_000, bandwith = 1e-2,
              kernel='silverman')
```

{{overcomplete.sae.jump_sae.JumpSAE | num_parents=1, skip_methods=fit}}

[^1]:[Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders (2024)](https://arxiv.org/html/2407.14435) by Rajamanoharan et al. (2024).

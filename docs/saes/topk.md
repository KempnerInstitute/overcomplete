# TopK SAE

[TopK SAE](https://arxiv.org/abs/2406.04093) is a variation of the standard Sparse Autoencoder (SAE) that enforces structured sparsity using a **Top-K selection mechanism**. This method ensures that only the **K most significant activations** are retained in the encoded representation, promoting interpretability and feature selection.

The architecture follows the standard SAE framework, consisting of an encoder, a decoder, and a forward method:

- `encode` returns the pre-codes (`z_pre`, before activation) and codes (`z`) given an input (`x`).
- `decode` returns a reconstructed input (`x_hat`) based on an input (`x`).
- `forward` returns the pre-codes, codes, and reconstructed input.

We strongly encourage you to check the original paper [^1] to learn more about TopK SAE.

### Basic Usage
```python
from overcomplete import TopKSAE

# define a TopK SAE with input dimension 768, 10k concepts
sae = TopKSAE(768, 10_000, top_k=5)

# adjust the encoder module (you can also)
# directly pass your own encoder module
sae = TopKSAE(768, 10_000, top_k=10,
              encoder_module='mlp_bn_1')
```

{{overcomplete.sae.topk_sae.TopKSAE | num_parents=1, skip_methods=fit}}

[^1]: [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093v1) by Gao et al. (2024).
# Batch TopK SAE

[Batch TopK SAE](https://arxiv.org/abs/2412.06410) is a variation of the standard Sparse Autoencoder (SAE) that enforces structured sparsity at the batch level using a **global Top-K selection mechanism**. Instead of selecting the K largest activations per sample, this method selects the **top-K activations across the entire batch**, ensuring a controlled level of sparsity.

The architecture follows the standard SAE framework, consisting of an encoder, a decoder, and a forward method:

- `encode` returns the pre-codes (`z_pre`, before thresholding) and codes (`z`) given an input (`x`).
- `decode` returns a reconstructed input (`x_hat`) based on an input (`x`).
- `forward` returns the pre-codes, codes, and reconstructed input.

We strongly encourage you to check the original paper [^1] to learn more about Batch TopK SAE.

## Basic Usage
```python
from overcomplete import BatchTopKSAE

# define a Batch TopK SAE with input dimension 768, 10k concepts
# and top_k = 50 (for the entire batch!)
sae = BatchTopKSAE(768, 10_000, top_k=50)

# the threshold momentum is used to estimate
# the final threshold (when in eval)
sae = BatchTopKSAE(768, 10_000, top_k=10, threshold_momentum=0.95)
# ... training sae
sae = sae.eval()
# now top_k is no longer use and instead an
# internal threshold is used
print(sae.running_threshold)
```

{{overcomplete.sae.batchtopk_sae.BatchTopKSAE | num_parents=1, skip_methods=fit}}

[^1]: [Batch Top-k Sparse Autoencoders](https://arxiv.org/pdf/2412.06410) by Bussmann et al. (2024).


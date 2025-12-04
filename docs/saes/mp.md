# Matching Pursuit SAE (MpSAE)

MpSAE replaces thresholding with a **greedy matching pursuit** loop: at each step it picks the atom most correlated with the residual, updates the codes, and subtracts the atom’s contribution, yielding sparse codes that track reconstruction progress. We encourage reading [^1] for the full method.

## Basic Usage
```python
from overcomplete import MpSAE

# define a Matching Pursuit SAE with input dimension 512, 4k concepts
sae = MpSAE(512, 4_096, k=4, dropout=0.1)

# k = number of pursuit steps, dropout optionally masks atoms each step
residual, codes = sae.encode(x)
```

## Advanced: auxiliary loss to revive dead codes

To ensure high dictionary utilization in MP-SAE, we strongly recommend implementing an auxiliary loss term.
Here is an example of such loss:

```python
def criterion(x, x_hat, residual, z, d):
    recon_loss = ((x - x_hat) ** 2).mean()

    revive_mask = (z.amax(dim=0) < 1e-2).detach()  # shape: [c]

    if revive_mask.sum() > 10:
        projected = residual @ d.T  # shape: [n, c]
        revive_term = projected[:, revive_mask].mean()
        recon_loss -= revive_term * 1e-2

    return recon_loss
```

{{overcomplete.sae.mp_sae.MpSAE | num_parents=1, skip_methods=fit}}

[^1]: [From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit](https://arxiv.org/abs/2506.03093).

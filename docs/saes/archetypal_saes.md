# Archetypal Sparse Autoencoders (TopK & Jump)

Archetypal SAEs combine the archetypal dictionary constraint with familiar sparse encoders:
- **RATopKSAE**: TopK selection with archetypal atoms.
- **RAJumpSAE**: JumpReLU selection with archetypal atoms.

Dictionary atoms stay close to convex combinations of provided data points (controlled by `delta` and an optional multiplier), stabilizing training and improving interpretability. This is the SAE-form of the [Archetypal SAE](https://arxiv.org/abs/2502.12892) idea.

## Basic Usage
```python
import torch
from overcomplete.sae import RATopKSAE, RAJumpSAE

points = torch.randn(2_000, 768)  # e.g. k-means centroids or sampled activations

ra_topk = RATopKSAE(
    input_shape=768,
    nb_concepts=10_000,
    points=points,
    top_k=20,
    delta=1.0,          # relaxation radius
    use_multiplier=True # learnable scaling of the archetypal hull
)

ra_jump = RAJumpSAE(
    input_shape=768,
    nb_concepts=10_000,
    points=points,
    bandwidth=1e-3,
    delta=1.5
)
```

Tips:
- Provide reasonably diverse `points` (e.g., k-means cluster centers) for stable archetypes.
- `use_multiplier` allows atoms to scale beyond the convex hull; set False to stay tighter.
- All standard training utilities (`train_sae`, custom losses) work unchanged.

{{overcomplete.sae.rasae.RATopKSAE | num_parents=1, skip_methods=fit}}
{{overcomplete.sae.rasae.RAJumpSAE | num_parents=1, skip_methods=fit}}

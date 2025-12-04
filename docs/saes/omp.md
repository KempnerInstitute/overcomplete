# Orthogonal Matching Pursuit SAE (OMPSAE)

OMPSAE uses **orthogonal matching pursuit** for sparse coding. Each iteration picks the atom most correlated with the current residual, then resolves NNLS on all selected atoms to refine the codes. This tighter refit often improves reconstruction over plain matching pursuit. For background, see [Sparse Autoencoders via Matching Pursuit](https://arxiv.org/pdf/2506.03093).

## Basic Usage
```python
import torch
from overcomplete.sae import OMPSAE

x = torch.randn(64, 512)
sae = OMPSAE(
    input_shape=512,
    nb_concepts=4_096,
    k=4,            # pursuit steps
    max_iter=15,    # NNLS iterations
    dropout=0.1,    # optional atom dropout
    encoder_module="identity",
    device="cuda"
)

residual, codes = sae.encode(x)
```

Notes:
- `encode` returns `(residual, codes)`; residual is the reconstruction error after pursuit steps.
- Set `dropout` to randomly mask atoms each iteration.
- Inputs must be 1D features (no 3D/4D tensors); `k` and `max_iter` must be positive.

{{overcomplete.sae.omp_sae.OMPSAE | num_parents=1, skip_methods=fit}}

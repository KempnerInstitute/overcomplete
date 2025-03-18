# Relaxed Archetypal Dictionary

[Archetypal SAE](https://arxiv.org/abs/2502.12892) introduces a constraint on the dictionary where each atom is formed as a convex combination of data points with an additional relaxation term. This method enhances stability and interpretability in dictionary learning, making it a robust drop-in replacement for the dictionary layer in any Sparse Autoencoder. Simply initialize the archetypal dictionary and assign it to your SAE (e.g., `sae.dictionary = archetypal_dict`).

## Basic Usage

```python
import torch
from overcomplete.sae.batchtopk_sae import BatchTopKSAE
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary

# initialize any sae
sae = BatchTopKSAE(768, 10_000, top_k=50)

# assume 'points' is a tensor of candidate data points (e.g., sampled from your dataset)
# the original paper recommend k-means
points = torch.randn(1000, 768)

# create our ra-sae
archetypal_dict = RelaxedArchetypalDictionary(
    in_dimensions=768,
    nb_concepts=10_000,
    points=points,
    delta=1.0,
)

# set the SAE's dictionary with the archetypal dictionary
sae.dictionary = archetypal_dict
# you can now train normally your sae
```
{{overcomplete.sae.archetypal_dictionary.RelaxedArchetypalDictionary | num_parents=0, skip_methods=fit}}

[^1]: [Archetypal-SAE](https://arxiv.org/abs/2502.12892) by Fel et al. (2025).

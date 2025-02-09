# Semi-Nonnegative Matrix Factorization (Semi-NMF)

Semi-Nonnegative Matrix Factorization (Semi-NMF) is a variant of NMF that relaxes the constraint on the code matrix (`Z`), allowing negative values while keeping the dictionary (`D`) nonnegative. The model consists of:

- **Input Matrix (`A`)**: The pattern of activations in a neural network, shaped as `(batch_size, n_features)`.
- **Codes (`Z`)**: The representation of data in terms of discovered concepts, shaped as `(batch_size, nb_concepts)`, and can take negative values.
- **Dictionary (`D`)**: A learned basis of concepts, shaped as `(nb_concepts, n_features)`, constrained to be nonnegative.

## Basic Usage
```python
from overcomplete import SemiNMF

# define a Semi-NMF model with 10k concepts using
# the multiplicative update solver
semi_nmf = SemiNMF(nb_concepts=10_000, solver='mu')

# fit the model to input activations A
Z, D = semi_nmf.fit(A)

# encode new data
Z = semi_nmf.encode(A)
# decode (reconstruct) data from codes
A_hat = semi_nmf.decode(Z)
```

## Solvers
The Semi-NMF module supports different optimization strategies:
- **MU** (Multiplicative Updates) - Standard Semi-NMF update rule.
- **PGD** (Projected Gradient Descent) - Allows for sparsity control via L1 penalty on `Z`.

For further details, we encourage you to check the original references [^1].

{{overcomplete.optimization.semi_nmf.SemiNMF | num_parents=1}}


[^1]: [Convex and Semi-Nonnegative Matrix Factorizations](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf) by Ding, Li, and Jordan (2008).



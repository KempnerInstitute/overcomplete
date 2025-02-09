# Convex Non-Negative Matrix Factorization (Convex NMF)

[Convex Non-Negative Matrix Factorization (Convex NMF)](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf) is a variation of NMF that constrains the dictionary (`D`) to be a convex combination of input data (`A`). This ensures that learned basis components are directly interpretable in terms of the original data and the dictionary elements to be in the conical hull of the data.

where:

- **`A` (Data Matrix):** Input data, shape `(n_samples, n_features)`.
- **`Z` (Codes Matrix):** Latent representation, shape `(n_samples, nb_concepts)`, constrained to be non-negative.
- **`W` (Coefficient Matrix):** Convex coefficients, shape `(nb_concepts, n_samples)`, constrained to be non-negative. The dictionary is computed as `D = W A`.

## Basic Usage
```python
from overcomplete import ConvexNMF

# define a Convex NMF model with 10k concepts using
# the multiplicative update solver
convex_nmf = ConvexNMF(nb_concepts=10_000, solver='mu')

# fit the model to input data A
Z, D = convex_nmf.fit(A)

# encode new data
Z = convex_nmf.encode(A)
# decode (reconstruct) data from codes
A_hat = convex_nmf.decode(Z)
```

## Solvers
The Convex NMF module supports different optimization strategies:
- **MU** (Multiplicative Updates) - Standard Convex NMF update rule.
- **PGD** (Projected Gradient Descent) - Allows for sparsity control via L1 penalty on `Z`.

For further details, we encourage you to check the original references [^1].

{{overcomplete.optimization.convex_nmf.ConvexNMF | num_parents=1}}

[^1]: [Convex and Semi-Nonnegative Matrix Factorizations](https://people.eecs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf) by Ding, Li, and Jordan (2008).




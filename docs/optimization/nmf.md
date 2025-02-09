# Non-Negative Matrix Factorization (NMF)

Non-Negative Matrix Factorization (NMF) is a factorization method that decomposes a non-negative matrix into two lower-rank non-negative matrices. This technique is widely used in feature learning, dictionary learning, and dimensionality reduction, particularly in neural network activations.

The model consists of:

- **Input Matrix (`A`)**: The pattern of activations in a neural network, shaped as `(batch_size, n_features)`.
- **Codes (`Z`)**: The representation of data in terms of discovered concepts, shaped as `(batch_size, nb_concepts)`.
- **Dictionary (`D`)**: A learned basis of concepts, shaped as `(nb_concepts, n_features)`.

## Basic Usage
```python
from overcomplete import NMF

# define an NMF model with 10k concepts using the HALS solver
nmf = NMF(nb_concepts=10_000, solver='hals')

# fit the model to input activations A
Z, D = nmf.fit(A)

# encode new data
Z = nmf.encode(A)
# decode (reconstruct) data from codes
A_hat = nmf.decode(Z)
```

## Solvers
The NMF module supports different optimization strategies:
- **HALS** (Hierarchical Alternating Least Squares) - Efficient for large-scale problems.
- **MU** (Multiplicative Updates) - Standard NMF update rule.
- **ANLS** (Alternating Non-Negative Least Squares) - Robust least squares optimization.
- **PGD** (Projected Gradient Descent) - Suitable for constrained optimization.

With HALS usually providing best reconstruction.

{{overcomplete.optimization.nmf.NMF | num_parents=1}}
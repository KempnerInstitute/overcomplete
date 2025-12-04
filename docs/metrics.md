# Metrics for Dictionary Learning

The `overcomplete.metrics` module provides a collection of evaluation metrics designed for **dictionary learning algorithms**. These metrics help assess sparsity, reconstruction accuracy, and dictionary quality.

## Overview
This module includes metrics for:

- **Norm-based evaluations**: L0, L1, L2, and Lp norms.
- **Reconstruction losses**: Absolute and relative errors.
- **Sparsity metrics**: Hoyer score, L1/L2 ratio, and Kappa-4.
- **Dictionary similarity**: Hungarian loss, cosine Hungarian loss, and collinearity.
- **Distribution-based metrics**: Wasserstein-1D and Fréchet distance.
- **Code analysis**: Detecting dead codes and assessing sparsity.

## Example Usage (Key Metrics)
```python
from overcomplete.metrics import (dead_codes, r2_score, hungarian_loss,
                                 cosine_hungarian_loss, wasserstein_1d)
# inputs
x = torch.randn(100, 10)
x_hat = torch.randn(100, 10)
# dictionaries
dict1 = torch.randn(512, 256)
dict2 = torch.randn(512, 256)
# concept values (codes)
codes = torch.randn(100, 512)
codes_2 = torch.randn(100, 512)

# check for inactive dictionary elements
dead_code_ratio = dead_codes(codes).mean()

# compare dictionary structures
hungarian_dist = hungarian_loss(dict1, dict2)
cosine_hungarian_dist = cosine_hungarian_loss(dict1, dict2)

# compute reconstruction quality
r2 = r2_score(x, x_hat)
# distrib. reconstruction quality
wasserstein_dist = wasserstein_1d(x, x_hat)
```

## Available Metrics
### **Norm-Based Metrics**
- `l0(x)`, `l1(x)`, `l2(x)`, `lp(x, p)`
- `l1_l2_ratio(x)`: Ratio of L1 to L2 norm.
- `hoyer(x)`: Normalized sparsity measure.

### **Reconstruction Losses**
- `avg_l2_loss(x, x_hat)`, `avg_l1_loss(x, x_hat)`
- `relative_avg_l2_loss(x, x_hat)`, `relative_avg_l1_loss(x, x_hat)`
- `r2_score(x, x_hat)`: Measures reconstruction accuracy.

### **Sparsity Metrics**
- `l0(x)`: Cardinality of the support of `x`.
- `sparsity_eps(x, threshold)`: L0 with an epsilon threshold.
- `kappa_4(x)`: Kurtosis-based sparsity measure.
- `dead_codes(x)`: Identifies unused codes in a dictionary.

### **Dictionary Evaluation**
- `hungarian_loss(dict1, dict2)`: Finds best-matching dictionary elements.
- `cosine_hungarian_loss(dict1, dict2)`: Cosine distance-based Hungarian loss.
- `dictionary_collinearity(dict)`: Measures collinearity in dictionary elements.

### **Distribution-Based Metrics**
- `wasserstein_1d(x1, x2)`: 1D Wasserstein-1 distance.
- `frechet_distance(x1, x2)`: Fréchet distance for distributions.

For further details, refer to the module documentation.

{{overcomplete.metrics.l2}}
{{overcomplete.metrics.l1}}
{{overcomplete.metrics.lp}}
{{overcomplete.metrics.avg_l2_loss}}
{{overcomplete.metrics.avg_l1_loss}}
{{overcomplete.metrics.relative_avg_l2_loss}}
{{overcomplete.metrics.relative_avg_l1_loss}}
{{overcomplete.metrics.l0}}
{{overcomplete.metrics.l1_l2_ratio}}
{{overcomplete.metrics.hoyer}}
{{overcomplete.metrics.kappa_4}}
{{overcomplete.metrics.r2_score}}
{{overcomplete.metrics.dead_codes}}
{{overcomplete.metrics.hungarian_loss}}
{{overcomplete.metrics.cosine_hungarian_loss}}
{{overcomplete.metrics.dictionary_collinearity}}
{{overcomplete.metrics.wasserstein_1d}}
{{overcomplete.metrics.frechet_distance}}
{{overcomplete.metrics.codes_correlation_matrix}}
{{overcomplete.metrics.energy_of_codes}}


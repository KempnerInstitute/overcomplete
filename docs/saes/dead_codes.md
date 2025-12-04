# Handling Dead Codes in SAEs

One of the most persistent challenges in training Sparse Autoencoders (SAEs) is the issue of **dead dictionary elements** (often called "dead codes" or feature collapse). These are latent features that, early in training, fall into a regime where they never activate (i.e., they have zero magnitude), preventing them from receiving gradients and learning useful representations.

Below is a simple auxiliary loss to gently nudge inactive atoms back into use while keeping the main reconstruction objective intact.

### How it works
1.  **Identify Dead Codes:** It calculates a boolean mask for features that have not fired a single time across the current batch.
2.  **Boost Pre-activations:** It isolates the "pre-codes" (the values *before* the activation function like ReLU or TopK is applied) for these dead atoms.
3.  **Revive:** It subtracts these pre-activation values from the loss. Since the optimizer minimizes loss, this effectively **pushes the pre-activations toward the positive direction**, making them more likely to cross the activation threshold in future steps.

## Recommended Auxiliary Loss

```python
def criterion(x, x_hat, pre_codes, codes):
    # 1. Standard reconstruction loss (MSE)
    loss = (x - x_hat).square().mean()

    # 2. Identify dead codes
    # is_dead has shape [dict_size] and is 1.0 when a code never fires in the batch
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()

    # 3. Calculate re-animation term
    # We want to maximize the pre_codes of dead atoms to push them > 0
    # Therefore, we subtract their mean value from the loss.
    reanim_loss = (pre_codes * is_dead[None, :]).mean()

    # 4. Combine
    loss -= reanim_loss * 1e-3  # Keep this factor small

    return loss
```

### Guidance for Implementation

* **Coefficient Sensitivity:** Use a **small coefficient** (e.g., `1e-4,1e-3`) so the reconstruction error remains the dominant term. If the coefficient is too high, the model may hallucinate features just to satisfy the auxiliary loss.
* **Monitoring:** Monitor the `dead_codes` metric (e.g., via `overcomplete.metrics.dead_codes`) to confirm the auxiliary term is reducing the dead count without simply creating "dense" noise.
* **Scheduling:** This is primarily useful during the early to mid-stages of training. You can anneal the coefficient to `0` once the dictionary utilization stabilizes.
* **Compatibility:** This auxiliary pairs well with any SAE variant (TopK, JumpReLU, Standard ReLU) provided you have access to the `pre_codes`.
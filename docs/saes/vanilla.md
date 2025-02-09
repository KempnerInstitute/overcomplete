# Vanilla SAE

The most basic SAE. It consists of an encoder and a decoder. The decoder is simply a dictionary, while the encoder can be configured. By default, it is a linear module with bias and ReLU activation. All SAEs include an encoder, a decoder, and a forward method.

- `encode` returns the pre-codes (`z_pre`, before ReLU) and codes (`z`) given an input (`x`).
- `decode` returns a reconstructed input (`x_hat`) based on an input (`x`).
- `forward` returns the pre-codes, codes, and reconstructed input.

###  Basic Usage
```python
from overcomplete import SAE

# Define a basic SAE where input dimension is 768, with 10k concepts
# Using a simple linear encoding
sae = SAE(768, 10_000)

# Define a more complex SAE with batch normalization in the encoder
# The dictionary is normalized on the L1 ball instead of L2
sae = SAE(768, 10_000, encoder_module='mlp_bn_1',
          dictionary_params={'normalization': 'l1'})
```




{{overcomplete.sae.base.SAE | num_parents=0, skip_methods=fit}}

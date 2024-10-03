"""
Module for Quantized Sparse SAE (Q-SAE).
"""

import torch
from torch import nn

from overcomplete.sae.base import SAE, SAEOutput


class QSAE(SAE):
    """
    Quantized SAE.

    @tfel: The Quantized Sparse Autoencoder is a sparse autoencoder that will learn
    level of codes for each dimension, and project the pre_code to the closest
    quantization point in a set of 2q points. The quantization points are
    symmetric around 0 (Q = [-q, -q+1, ..., q-1, q]).

    @tfel: this is an unpublished work, please cite the Overcomplete library
    if you use it.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    n_components : int
        Number of components in the dictionary.
    q : int, optional
        Number of top quantized steps to keep in the latent representation.
    hard : bool, optional
        Whether to use hard quantization (True) or soft quantization (False),
        by default False. Hard quantization is slower and more memory intensive.
    encoder_module : nn.Module or string, optional
        Custom encoder module, by default None.
        If None, a simple Linear + BatchNorm default encoder is used.
        If string, the name of the registered encoder module.
    dictionary_initializer : str, optional
        Method for initializing the dictionary, e.g 'svd', 'kmeans', 'ica',
        see dictionary module to see all the possible initialization.
    data_initializer : torch.Tensor, optional
        Data used to fit a first approximation and initialize the dictionary, by default None.
    device : str, optional
        Device to run the model on, by default 'cpu'.


    Methods
    -------
    get_dictionary():
        Return the learned dictionary.
    forward(x):
        Perform a forward pass through the autoencoder.
    encode(x):
        Encode input data to latent representation.
    decode(z):
        Decode latent representation to reconstruct input data.
    """

    def __init__(self, input_shape, n_components, q=2, hard=False,
                 encoder_module=None, dictionary_initializer=None, data_initializer=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))

        super().__init__(input_shape, n_components, encoder_module,
                         dictionary_initializer, data_initializer, device)

        # from 1..q on each concepts. we will symmetrize it later
        # todo check arange ou faire juste un linspace 0, 1?
        self._Q = torch.arange(q, device=device).float() + 1.0
        self._Q = self._Q.unsqueeze(0).repeat(n_components, 1)

        self.hard = hard

    def encode(self, x):
        """
        Encode input data to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor or tuple
            Latent representation tensor (z) of shape (batch_size, nb_components).
            If the encoder returns the pre-codes, it returns a tuple (pre_z, z).
        """
        pre_codes, codes = self.encoder(x)

        # symmetrize Q: concatenate -Q and Q along the last dimension
        Q = torch.cat([-self._Q, self._Q], dim=-1)

        # compute the distance from pre_codes to each quantization state
        dist = (pre_codes.unsqueeze(-1) - Q).square()

        # take the closest id
        if self.hard:
            closest_idx = dist.argmin(dim=-1)

            # could be more efficient, i simply convert to one-hot and
            # do a matrix multiplication to get the closest Q value
            one_hot = torch.nn.functional.one_hot(closest_idx, num_classes=Q.size(-1)).float()
            quantized_codes = torch.sum(one_hot * Q[None, :, :], -1)
        else:
            # soft quantization
            # compute the softmax of the negative distance
            closest_idx = torch.nn.functional.softmax(-dist, dim=-1)
            quantized_codes = torch.sum(closest_idx * Q[None, :, :], -1)

        # straight-through estimator
        quantized_codes = codes + quantized_codes - codes.detach()

        quantized_codes = torch.relu(quantized_codes)

        return pre_codes, quantized_codes

    def decode(self, z):
        """
        Decode latent representation to reconstruct input data.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, input_size).
        """
        return self.dictionary(z)

    def fit(self, x):
        """
        Method not implemented for SAE. See train_sae function for training the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.
        """
        raise NotImplementedError('SAE does not support fit method. You have to train the model \
                                  using a custom training loop.')

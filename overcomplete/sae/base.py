"""
Base module for Sparse Autoencoder (SAE) model for dictionary learning.
"""

import torch
from torch import nn
from dataclasses import dataclass

from ..base import BaseDictionaryLearning
from .dictionary import DictionaryLayer
from .factory import EncoderFactory


@dataclass
class SAEOutput:
    """
    SAEOuput dataclass to store the output of the SAE model.

    Pre-codes may not be available if the encoder does not return them. In this case,
    the pre-codes will usually be the same as the codes.

    pre_codes: torch.Tensor
        Pre-activation outputs.
    codes: torch.Tensor
        Post-activation (latent codes) (Z).
    reconstruction: torch.Tensor
        Reconstructed input.
    """
    pre_codes: torch.Tensor
    codes: torch.Tensor
    reconstruction: torch.Tensor

    def __iter__(self):
        """
        Allow to simply unpack the dataclass.

            data = SAEOutput(a, b, c)
            a, b, c = sae_output

        """
        return iter((self.pre_codes, self.codes, self.reconstruction))


class SAE(BaseDictionaryLearning):
    """
    Sparse Autoencoder (SAE) model for dictionary learning.

    The SAE for Overcomplete models follows a common structure, consisting of:
    - (i) An encoder that returns code embeddings (concepts) for each token. This
          embedding is always 2D (N, n_components). The encoder can handle 2D
          (seq_len, dim) or 3D (dim, height, width) input data. At the end of
          encoding, a rearrangement/flattening is applied so that each token has a
          unique description.
    - (ii) A dictionary layer, which is a matrix multiplication between the code
           values produced by the encoder and a dictionary. To reconstruct the
           activation, the dimension 'dim' will be chosen. In the 1D case, it is
           the unique dimension; in the 2D case (seq_len, dim), it is the second
           dimension; and in the 3D case (dim, height, width), it is the first
           dimension.
    After the dictionary layer, we end up with a vector of dimension 2 (N, dim),
    where dim is the 'channel' dimension.

    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    n_components : int
        Number of components in the dictionary.
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

    def __init__(self, input_shape, n_components, encoder_module=None, dictionary_initializer=None,
                 data_initializer=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))

        super().__init__(n_components=n_components, device=device)

        # initialize the encoder
        if isinstance(encoder_module, str):
            assert encoder_module in EncoderFactory.list_modules(), f"Encoder '{encoder_module}' not found in registry."
            self.encoder = EncoderFactory.create_module(encoder_module, input_shape, n_components, device=device)
        elif encoder_module is not None:
            self.encoder = encoder_module
        else:
            # default encoder
            assert isinstance(input_shape, int), "Default encoder assumes input_shape is an int."
            self.encoder = EncoderFactory.create_module("linear", input_shape, n_components, device=device)

        # initialize the dictionary, but first find the channel dimension
        # tfel: do we really need this parameter if an encoder module is passed?
        if isinstance(input_shape, int):
            dim = input_shape
        elif len(input_shape) == 2:
            # attention model case with shape (T C)
            dim = input_shape[1]
        elif len(input_shape) == 3:
            # convolutional model case with shape (C H W)
            dim = input_shape[0]
        else:
            raise ValueError("Input shape must be 1D, 2D or 3D.")

        self.dictionary = DictionaryLayer(n_components, dim).to(device)
        # if needed, initialize the dictionary layer (e.g with SVD)
        if dictionary_initializer is not None:
            if data_initializer is None:
                raise ValueError("You must provide data_initializer if you want to initialize\
                                 the dictionary.")
            self.dictionary.initialize_dictionary(data_initializer, dictionary_initializer)

    def get_dictionary(self):
        """
        Return the learned dictionary.

        Returns
        -------
        torch.Tensor
            Learned dictionary tensor of shape (nb_components, input_size).
        """
        return self.dictionary.get_dictionary()

    def forward(self, x):
        """
        Perform a forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        SAEOuput
            Return the pre_codes (z_pre), codes (z) and reconstructed input tensor (x_hat).
        """
        pre_codes, codes = self.encode(x)

        x_reconstructed = self.decode(codes)

        return SAEOutput(pre_codes, codes, x_reconstructed)

    def encode(self, x):
        """
        Encode input data to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        tuple (pre_codes, codes)
            Latent representation tensor before (z_pre) and after (z) activation function.
            The shape are the same (batch_size, nb_components).
        """
        return self.encoder(x)

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

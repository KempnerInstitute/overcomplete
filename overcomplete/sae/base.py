from torch import nn

from ..base import BaseDictionaryLearning
from .dictionary import DictionaryLayer
from .factory import SAEFactory


class SAE(BaseDictionaryLearning):
    """
    Sparse Autoencoder (SAE) model for dictionary learning.

    Parameters
    ----------
    input_size : int or tuple
        Dimensionality of the input data.
    n_components : int
        Number of components in the dictionary.
    encoder_module : nn.Module or string, optional
        Custom encoder module, by default None.
        If None, a simple Linear + BatchNorm default encoder is used.
        If string, the name of the registered encoder module.
    dictionary_initializer : str, optional
        Method for initializing the dictionary, by default 'svd'.
    data_initializer : torch.Tensor, optional
        Data used to fit a first approximation and initialize the dictionary, by default None.

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

    def __init__(self, input_size, n_components, encoder_module=None, dictionary_initializer='svd',
                 data_initializer=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))

        super().__init__(n_components=n_components, device=device)

        if isinstance(encoder_module, str):
            self.encoder = SAEFactory.create_module(encoder_module, input_size, n_components)
        elif encoder_module is not None:
            self.encoder = encoder_module
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, n_components),
                nn.BatchNorm1d(n_components),
                nn.ReLU(),
            )
        self.dictionary = DictionaryLayer(n_components, input_size)

        if data_initializer is not None:
            self.dictionary.initialize_dictionary(data_initializer, dictionary_initializer)

    def get_dictionary(self):
        """
        Return the learned dictionary.

        Returns
        -------
        torch.Tensor
            Learned dictionary tensor of shape (nb_components, input_size).
        """
        return self.dictionary.dictionary

    def forward(self, x):
        """
        Perform a forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        tuple
            Latent representation tensor and reconstructed input tensor.
        """
        z = self.encode(x)
        return z, self.decode(z)

    def encode(self, x):
        """
        Encode input data to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).
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
        raise NotImplementedError('SAE does not support fit method. You have to train the model \
                                  using a custom training loop.')

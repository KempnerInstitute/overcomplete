import torch
from torch import nn
from ..base import BaseDictionaryLearning
from ..optimization import (OptimKMeans, OptimPCA, OptimICA, OptimNMF,
                            OptimSparsePCA, OptimSVD, OptimDictionaryLearning)


class DictionaryLayer(nn.Module):
    """
    A neural network layer representing a dictionary for reconstructing input data.

    Parameters
    ----------
    nb_components : int
        Number of components in the dictionary.
    dimensions : int
        Dimensionality of the input data.

    Methods
    -------
    forward(z):
        Perform a forward pass to reconstruct input data from latent representation.
    initialize_dictionary(x, method='svd'):
        Initialize the dictionary using a specified method.
    """

    def __init__(self, nb_components, dimensions):
        super(DictionaryLayer, self).__init__()
        self.nb_components = nb_components
        self.dimensions = dimensions
        self.dictionary = nn.Parameter(torch.randn(nb_components, dimensions))

    def forward(self, z):
        """
        Reconstruct input data from latent representation.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape (batch_size, nb_components).

        Returns
        -------
        torch.Tensor
            Reconstructed input tensor of shape (batch_size, dimensions).
        """
        x_hat = torch.matmul(z, self.dictionary)
        return x_hat

    def initialize_dictionary(self, x, method='svd'):
        """
        Initialize the dictionary using a specified method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, dimensions).
        method : str or BaseDictionaryLearning, optional
            Method for initializing the dictionary, by default 'svd'.
        """
        print('Initializing dictionary with', method, '...')
        if method == 'kmeans':
            init = OptimKMeans(self.nb_components)
        elif method == 'pca':
            init = OptimPCA(self.nb_components)
        elif method == 'ica':
            init = OptimICA(self.nb_components)
        elif method == 'nmf':
            init = OptimNMF(self.nb_components)
        elif method == 'sparse_pca':
            init = OptimSparsePCA(self.nb_components)
        elif method == 'svd':
            init = OptimSVD(self.nb_components)
        elif method == 'dictionary_learning':
            init = OptimDictionaryLearning(self.nb_components)
        elif isinstance(method, BaseDictionaryLearning):
            init = method
        else:
            raise ValueError("Invalid method")

        init.fit(x)
        self.dictionary.data = init.get_dictionary()

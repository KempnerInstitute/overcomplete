"""
Module dedicated to everything around the Dictionary Layer of SAE.
"""

import torch
from torch import nn
from ..base import BaseDictionaryLearning
from ..optimization import (SkPCA, SkICA, SkNMF, SkKMeans,
                            SkDictionaryLearning, SkSparsePCA, SkSVD)


class DictionaryLayer(nn.Module):
    """
    A neural network layer representing a dictionary for reconstructing input data.

    Parameters
    ----------
    nb_components : int
        Number of components in the dictionary.
    dimensions : int
        Dimensionality of the input data.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward(z):
        Perform a forward pass to reconstruct input data from latent representation.
    initialize_dictionary(x, method='svd'):
        Initialize the dictionary using a specified method.
    """

    def __init__(self, nb_components, dimensions, device='cpu'):
        super().__init__()
        self.nb_components = nb_components
        self.dimensions = dimensions
        self.dictionary = nn.Parameter(torch.randn(nb_components, dimensions)).to(device)

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
        if method == 'kmeans':
            init = SkKMeans(self.nb_components)
        elif method == 'pca':
            init = SkPCA(self.nb_components)
        elif method == 'ica':
            init = SkICA(self.nb_components)
        elif method == 'nmf':
            init = SkNMF(self.nb_components)
        elif method == 'sparse_pca':
            init = SkSparsePCA(self.nb_components)
        elif method == 'svd':
            init = SkSVD(self.nb_components)
        elif method == 'dictionary_learning':
            init = SkDictionaryLearning(self.nb_components)
        elif isinstance(method, BaseDictionaryLearning):
            init = method
        else:
            raise ValueError("Invalid method")

        init.fit(x)
        self.dictionary.data = init.get_dictionary().to(self.dictionary.device)

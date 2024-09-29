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
    normalize : str or callable, optional
        Whether to normalize the dictionary, by default 'l2' normalization is applied.
        Current options are 'l2', 'max_l2', 'l1', 'max_l1', 'identity'.
        If a custom normalization is needed, a callable can be passed.

    Methods
    -------
    forward(z):
        Perform a forward pass to reconstruct input data from latent representation.
    initialize_dictionary(x, method='svd'):
        Initialize the dictionary using a specified method.
    """
    _NORMALIZATIONS = {
        # project each concept in the dictionary "on" the l2 ball
        'l2': lambda x: x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8),
        # re-project (if necessary) each concept of the dictionary "inside" the l2 ball
        'max_l2': lambda x: x / torch.amax(torch.norm(x, p=2, dim=1, keepdim=True), 1, keepdim=True),
        # project each concept in the dictionary "on" the l1 ball
        'l1': lambda x: x / (torch.norm(x, p=1, dim=1, keepdim=True) + 1e-8),
        # re-project (if necessary) each concept in the dictionary "inside" the l1 ball
        'max_l1': lambda x: x / torch.amax(torch.norm(x, p=1, dim=1, keepdim=True), 1, keepdim=True),
        # no projection
        'identity': lambda x: x
    }

    def __init__(self, nb_components, dimensions, normalize='l2', device='cpu'):
        super().__init__()
        self.nb_components = nb_components
        self.dimensions = dimensions
        self.device = device

        # weights should not be accessed directly because of possible normalization/projections
        self._weights = nn.Parameter(torch.randn(nb_components, dimensions)).to(device)

        if isinstance(normalize, str):
            self.normalize = self._NORMALIZATIONS[normalize]
        elif callable(normalize):
            self.normalize = normalize
        else:
            raise ValueError("Invalid normalization function")

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
        dictionary = self.get_dictionary()
        x_hat = torch.matmul(z, dictionary)
        return x_hat

    def get_dictionary(self):
        """
        Get the dictionary.

        Returns
        -------
        torch.Tensor
            The dictionary tensor of shape (nb_components, dimensions).
        """
        return self.normalize(self._weights)

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
            # if we can call .fit(x) on the method, we consider it valid
            assert hasattr(method, 'fit'), "Invalid initialization object, must have a .fit() method"
            init = method

        init.fit(x)
        self._weights.data = init.get_dictionary().to(self.device)

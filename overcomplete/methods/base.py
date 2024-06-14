from abc import ABC, abstractmethod


class BaseSparseCoding(ABC):
    """
    Abstract base class for Sparse Coding models.

    Parameters
    ----------
    n_components : int
        Number of components to learn.

    Methods
    -------
    encode(x):
        Encode the input tensor.
    decode(x):
        Decode the input tensor.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    @abstractmethod
    def encode(self, x):
        """
        Encode the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Encoded features.
        """
        pass

    @abstractmethod
    def decode(self, x):
        """
        Decode the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Encoded tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Decoded output.
        """
        pass

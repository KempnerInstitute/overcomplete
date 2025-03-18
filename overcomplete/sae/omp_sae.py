"""
Orthogonal Matching Pursuit Sparse Autoencoder (OMP-SAE) implementation.
"""

import torch
from torch import nn
from overcomplete.sae import SAE


class OMPSAE(SAE):
    """
    Orthogonal Matching Pursuit Sparse Autoencoder (OMP-SAE).

    This autoencoder applies Orthogonal Matching Pursuit (OMP) to obtain sparse
    codes. Unlike standard Matching Pursuit (MP), OMP ensures that previously
    selected dictionary elements remain orthogonal by solving a least squares
    problem at each iteration.
    The encoding process is non-differentiable, the gradient comes only from the
    reconstruction part after the codes are obtained.
    Warning: for this SAE, the encoding is returning (1) the residual and (2) the
    codes -- as the pre_codes are just the input.

    Parameters
    ----------
    input_shape : int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of latent dimensions (components) of the autoencoder.
    k : int, optional
        The number of matching pursuit iterations to perform (must be > 0).
    dropout : float, optional
        Probability of dropping a dictionary element at each iteration
        (range: 0.0 - 1.0). If None, no dropout is applied.
    encoder_module : nn.Module or str, optional
        Custom encoder module (or its registered name). If None, a default encoder is used.
    dictionary_params : dict, optional
        Parameters that will be passed to the dictionary layer.
    device : str, optional
        Device on which to run the model (default is 'cpu').
    """

    def __init__(self, input_shape, nb_concepts, k=1, dropout=None,
                 encoder_module="identity", dictionary_params=None, device='cpu'):
        assert isinstance(input_shape, int) or len(input_shape) == 1, \
            "OMPSAE doesn't support 3D or 4D data format."
        assert k > 0, "k must be a positive integer."
        if dropout is not None:
            assert 0.0 <= dropout <= 1.0, "Dropout must be in range [0,1]."

        super().__init__(input_shape, nb_concepts, encoder_module, dictionary_params, device)
        self.k = k
        self.dropout = dropout

    def encode(self, x):
        """
        Encode input data using Orthogonal Matching Pursuit (OMP).

        Unlike standard Matching Pursuit, OMP ensures that selected dictionary elements
        remain orthogonal, improving representation sparsity and stability.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        residual : torch.Tensor
            The residual after k iterations of OMP.
        codes : torch.Tensor
            The final sparse codes obtained after k iterations of OMP.
        """
        W = self.get_dictionary()  # dictionary / encoder are tied

        if self.dropout is not None:
            drop_mask = torch.bernoulli((1.0 - self.dropout) * torch.ones(W.shape[0], device=self.device))
            # dropout is removing the possibility of using some atoms/concepts
            W = W * drop_mask.unsqueeze(1)

        batch_size = x.shape[0]
        codes = torch.zeros(batch_size, self.nb_concepts, device=self.device)
        residual = x.clone()

        with torch.no_grad():
            selected_atoms = torch.zeros((batch_size, self.k), dtype=torch.long, device=self.device)

            for i in range(self.k):
                z = residual @ W.T
                val, idx = torch.max(z, dim=1)

                selected_atoms[:, i] = idx

                # omp part: solve least squares for the selected atoms
                W_selected = W[idx]
                codes_selected = torch.linalg.lstsq(W_selected, x).solution

                codes.scatter_add_(1, idx.unsqueeze(1), codes_selected)

                residual = x - codes @ W

        return residual, codes

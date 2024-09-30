"""
Module for JumpReLU Sparse Autoencoder (JumpSAE).
"""

import torch
from torch import nn

from .base import SAE, SAEOutput
from .dictionary import DictionaryLayer
from .factory import EncoderFactory
from .kernels import (rectangle_kernel, gaussian_kernel, triangular_kernel, cosine_kernel,
                      epanechnikov_kernel, quartic_kernel, silverman_kernel, cauchy_kernel)


class JumpReLU(torch.autograd.Function):
    """
    JumpReLU activation function with pseudo-gradient for threshold.
    """
    @staticmethod
    def forward(ctx, x, threshold, kernel_fn, bandwith):
        """
        Forward pass of the JumpReLU activation function.
        Save the necessary variables for the backward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        threshold : torch.Tensor
            Threshold tensor, learnable parameter.
        kernel_fn : callable
            Kernel function.
        bandwith : float
            Bandwith of the kernel.
        """
        ctx.save_for_backward(x, threshold)
        ctx.bandwith = bandwith
        ctx.kernel_fn = kernel_fn

        output = x.clone()
        output[x < threshold] = 0

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Pseudo-gradient for the JumpReLU activation function.
        Use the kernel function to compute the pseudo-gradient w.r.t. the threshold.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss w.r.t. the output.
        """
        x, threshold = ctx.saved_tensors
        bandwith = ctx.bandwith
        kernel_fn = ctx.kernel_fn

        # gradient w.r.t. input (normal gradient)
        grad_input = grad_output.clone()
        grad_input[x < threshold] = 0

        # pseudo-gradient w.r.t. threshold parameters
        delta = x - threshold
        kernel_values = kernel_fn(delta, bandwith)

        # @tfel: we have a singularity at threshold=0, thus the
        # re-parametrization trick in JumpSAE class
        grad_threshold = - (threshold / bandwith) * kernel_values * grad_output
        grad_threshold = grad_threshold.sum(0)

        return grad_input, grad_threshold, None, None


class HeavisidePseudoGradient(torch.autograd.Function):
    """
    Heaviside step function with pseudo-gradient for threshold.

    The Heaviside step function is defined as:
        H(x) = 1 if x > 0, 0 otherwise.

    The pseudo-gradient is used to approximate the gradient at the threshold.
    """
    @staticmethod
    def forward(ctx, x, threshold, kernel_fn, bandwith):
        """
        Forward pass of the Heaviside step function.
        Save the necessary variables for the backward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        threshold : torch.Tensor
            Threshold tensor, learnable parameter.
        kernel_fn : callable
            Kernel function.
        bandwith : float
            Bandwith of the kernel.
        """
        ctx.save_for_backward(x, threshold)
        ctx.bandwith = bandwith
        ctx.kernel_fn = kernel_fn

        output = (x > threshold).float()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Pseudo-gradient for the Heaviside step function.
        Use the kernel function to compute the pseudo-gradient w.r.t. the threshold.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss w.r.t. the output.
        """
        x, threshold = ctx.saved_tensors
        bandwith = ctx.bandwith
        kernel_fn = ctx.kernel_fn

        delta = x - threshold
        kernel_values = kernel_fn(delta, bandwith)

        # see the paper for the formula
        grad_threshold = - (1 / bandwith) * kernel_values * grad_output
        grad_threshold = grad_threshold.sum(0)

        grad_input = torch.zeros_like(x)

        return grad_input, grad_threshold, None, None


def jump_relu(x, threshold, kernel_fn, bandwith):
    """
    Apply the JumpReLU activation function to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    threshold : torch.Tensor
        Threshold tensor, learnable parameter.
    kernel_fn : callable
        Kernel function.
    bandwith : float
        Bandwith of the kernel.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    return JumpReLU.apply(x, threshold, kernel_fn, bandwith)


def heaviside(x, threshold, kernel_fn, bandwith):
    """
    Apply the Heaviside step function to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    threshold : torch.Tensor
        Threshold tensor, learnable parameter.
    kernel_fn : callable
        Kernel function.
    bandwith : float
        Bandwith of the kernel.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    return HeavisidePseudoGradient.apply(x, threshold, kernel_fn, bandwith)


class JumpSAE(SAE):
    """
    JumpReLU Sparse Autoencoder (SAE).

    The JumpReLU activation function is a sparsity-inducing activation function that
    approximates the L0 norm. It is defined as:

            JumpReLU(x) = x if x > threshold, 0 otherwise.

    The JumpReLU function is not differentiable at the threshold, the idea
    is to use a pseudo-gradient to approximate the gradient at the threshold.

    For more information, see:
        - "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
            by Rajamanoharan et al. (2024), https://arxiv.org/abs/2407.14435
        - "Scaling and evaluating sparse autoencoders"
            by Gao et al. (2024), https://arxiv.org/abs/2406.04093v1
        - "JumpReLU: A Retrofit Defense Strategy for Adversarial Attacks"
            by Erichson et al. (2019), https://arxiv.org/abs/1904.03750


    Parameters
    ----------
    input_shape : int or tuple of int
        Dimensionality of the input data, do not include batch dimensions.
        It is usually 1d (dim), 2d (seq length, dim) or 3d (dim, height, width).
    n_components : int
        Number of components in the dictionary.
    kernel : str, optional
        Kernel function to use in the JumpReLU activation, by default 'silverman'.
        Current options are :
            - 'rectangle'
            - 'gaussian'
            - 'triangular'
            - 'cosine'
            - 'epanechnikov'
            - 'quartic'
            - 'silverman'
            - 'cauchy'.
    bandwith : float, optional
        Bandwith of the kernel, by default 1e-3.
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

    _KERNELS = {
        'rectangle': rectangle_kernel,
        'gaussian': gaussian_kernel,
        'triangular': triangular_kernel,
        'cosine': cosine_kernel,
        'epanechnikov': epanechnikov_kernel,
        'quartic': quartic_kernel,
        'silverman': silverman_kernel,
        'cauchy': cauchy_kernel
    }

    def __init__(self, input_shape, n_components, kernel='silverman', bandwith=1e-3,
                 encoder_module=None, dictionary_initializer=None, data_initializer=None, device='cpu'):
        assert isinstance(encoder_module, (str, nn.Module, type(None)))
        assert isinstance(input_shape, (int, tuple, list))
        assert kernel in self._KERNELS, f"Kernel '{kernel}' not found in the registry."

        super().__init__(input_shape, n_components, encoder_module,
                         dictionary_initializer, data_initializer, device)

        self.kernel = self._KERNELS[kernel]
        self.bandwith = torch.tensor(bandwith, device=device)
        self.threshold = nn.Parameter(torch.zeros(n_components), requires_grad=True).to(device)

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
        tuple (z, x_hat)
            The codes (z) and and reconstructed input tensor (x_hat).
        """
        z = self.encode(x)

        if isinstance(z, tuple):
            # if z is a tuple, it means that the encoder returns the pre-codes and the codes
            # (before the activation fn)
            pre_codes, codes = z
        else:
            pre_codes = z
            codes = z

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
        torch.Tensor or tuple
            Latent representation tensor (z) of shape (batch_size, nb_components).
            If the encoder returns the pre-codes, it returns a tuple (pre_z, z).
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

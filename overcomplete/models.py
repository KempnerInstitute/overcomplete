from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision import transforms
import timm


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for models.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Return the features for the input tensor.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__()
        self.use_half = use_half
        self.device = device

    @abstractmethod
    def forward_features(self, x):
        """
        Return the features for the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Output features.
        """
        pass


class DinoV2Model(BaseModel):
    """
    Concrete class for DiNoV2 model.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Perform a forward pass on the input tensor.

    preprocess(img):
        Preprocess input images for the DiNoV2 model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        if self.use_half:
            self.model = self.model.half()

        self.preprocess = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Assume input is in the same device as the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output features.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)['x_norm_patchtokens']


class SigLIPModel(BaseModel):
    """
    Concrete class for SigLIP model.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Perform a forward pass on the input tensor.

    preprocess(img):
        Preprocess input images for the SigLIP model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        self.model = timm.create_model('vit_base_patch16_siglip_224', pretrained=True, num_classes=0).eval().to(
            self.device)
        if self.use_half:
            self.model = self.model.half()

        self.preprocess = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Assume the input (x) is on the same device as the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output features.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)


class ViTModel(BaseModel):
    """
    Concrete class for ViT model.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.

    Methods
    -------
    forward_features(x):
        Perform a forward pass on the input tensor.

    preprocess(img):
        Preprocess input images for the ViT model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(
            self.device)
        if self.use_half:
            self.model = self.model.half()

        self.preprocess = transforms.Compose(
            [transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])])

    def forward_features(self, x):
        """
        Perform a forward pass on the input tensor.
        Assume the input (x) is on the same device as the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output features.
        """
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            return self.model.forward_features(x)

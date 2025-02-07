"""
Module dedicated for models loading.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision import transforms


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


class DinoV2(BaseModel):
    """
    Concrete class for DiNoV2 model.

    Specifically, we use the DiNoV2 model from Oquab, Darcet & al (2021),
    see https://github.com/facebookresearch/dinov2

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


class SigLIP(BaseModel):
    """
    Concrete class for SigLIP model.

    Specifically, we use the SigLIP model (https://arxiv.org/abs/2303.15343) from timm.

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
        import timm  # lazy import to avoid loading timm if not needed
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


class ViT(BaseModel):
    """
    Concrete class for ViT model.

    Specifically, we use a base ViT model from timm.

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
        import timm  # lazy import to avoid loading timm if not needed
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
            activations = self.model.forward_features(x)
            activations = activations[:, 1:]  # remove the class token
            return activations


class ResNet(BaseModel):
    """
    Concrete class for ResNet50 model.

    Specifically, we use a ResNet 50 BiT, see https://arxiv.org/abs/2106.05237.

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
        Preprocess input images for the ResNet model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        import timm  # lazy import to avoid loading timm if not needed
        self.model = timm.create_model('resnetv2_50x1_bit.goog_distilled_in1k', pretrained=True).eval().to(self.device)
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


class ConvNeXt(BaseModel):
    """
    Concrete class for ConvNeXt model.

    Specifically, we use a small ConvNeXt from timm, see https://arxiv.org/abs/2201.03545.

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
        Preprocess input images for the ConvNeXt model.
    """

    def __init__(self, use_half=False, device='cpu'):
        super().__init__(use_half, device)
        import timm  # lazy import to avoid loading timm if not needed
        self.model = timm.create_model('convnext_small.fb_in1k', pretrained=True).eval().to(self.device)
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

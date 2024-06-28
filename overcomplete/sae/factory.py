"""
Collection of module creation functions for SAE encoder
and a factory class to create modules using string identifier.
"""

from torch import nn

from .modules import MLPEncoder, AttentionEncoder, ResNetEncoder


_module_registry = {}


def register_module(name):
    """
    Decorator to register a module creation function.

    Parameters
    ----------
    name : str
        The name to register the module creation function under.
    """
    def decorator(func):
        _module_registry[name] = func
        return func
    return decorator


class ModuleFactory:
    """
    Factory class to create modules using registered module creation functions.
    """
    @staticmethod
    def create_module(name, *args, **kwargs):
        """
        Creates a module based on the registered name.

        Parameters
        ----------
        name : str
            The name of the registered module creation function.
        *args : tuple
            Positional arguments to pass to the module creation function.
        **kwargs : dict
            Keyword arguments to pass to the module creation function.

        Returns
        -------
        nn.Module
            The initialized module.
        """
        if name not in _module_registry:
            raise ValueError(f"Module '{name}' not found in registry.")
        return _module_registry[name](*args, **kwargs)


@register_module("mlp_bn_1")
def bn_1(input_size, n_components):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        nb_blocks=1,
        norm_layer=nn.BatchNorm1d
    )


@register_module("mlp_ln_1")
def ln_1(input_size, n_components):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        nb_blocks=1,
        norm_layer=nn.LayerNorm
    )


@register_module("mlp_bn_3")
def bn_3(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        norm_layer=nn.BatchNorm1d
    )


@register_module("mlp_ln_3")
def ln_3(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        norm_layer=nn.LayerNorm
    )


@register_module("mlp_bn_3_no_res")
def bn_3_no_res(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        norm_layer=nn.BatchNorm1d,
        residual=False
    )


@register_module("mlp_ln_3_no_res")
def ln_3_no_res(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        norm_layer=nn.LayerNorm,
        residual=False
    )


@register_module("mlp_gelu_bn_3")
def gelu_bn_3(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        hidden_activation=nn.GELU,
        norm_layer=nn.BatchNorm1d
    )


@register_module("mlp_gelu_ln_3")
def gelu_ln_3(input_size, n_components, hidden_dim=None):
    return MLPEncoder(
        input_size=input_size,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=3,
        hidden_activation=nn.GELU,
        norm_layer=nn.LayerNorm
    )


@register_module("resnet_big")
def resnet_bn(input_channels, n_components, hidden_dim=None, nb_blocks=1):
    return ResNetEncoder(
        input_channels=input_channels,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )


@register_module("resnet_big_3")
def resnet_bn(input_channels, n_components, hidden_dim=None, nb_blocks=3):
    return ResNetEncoder(
        input_channels=input_channels,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )


@register_module("resnet_small")
def resnet_bn(input_channels, n_components, hidden_dim=128, nb_blocks=1):
    return ResNetEncoder(
        input_channels=input_channels,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )


@register_module("resnet_small_3")
def resnet_bn(input_channels, n_components, hidden_dim=128, nb_blocks=3):
    return ResNetEncoder(
        input_channels=input_channels,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )


@register_module("attention")
def attention_encoder(input_shape, n_components, hidden_dim=None, nb_blocks=1):
    return AttentionEncoder(
        input_shape=input_shape,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )


@register_module("attention_3")
def attention_encoder(input_shape, n_components, hidden_dim=None, nb_blocks=3):
    return AttentionEncoder(
        input_shape=input_shape,
        n_components=n_components,
        hidden_dim=hidden_dim,
        nb_blocks=nb_blocks,
    )

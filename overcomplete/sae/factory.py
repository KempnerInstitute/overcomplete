"""
Collection of module creation functions for SAE encoder
and a factory class to create modules using string identifier.

example usage:

model = SAEFactory.create_module("mlp_ln_3")
model = SAEFactory.create_module("mlp_bn_1_gelu_no_res")

model = SAEFactory.create_module("resnet_3b")
model = SAEFactory.create_module("attention_1b")

you can also pass additional arguments to the module creation function:

model = SAEFactory.create_module("mlp_ln_3", hidden_dim=128)
model = SAEFactory.create_module("resnet_3b", hidden_dim=128)
model = SAEFactory.create_module("attention_1b", attention_heads=2)
"""

from torch import nn

from .modules import MLPEncoder, AttentionEncoder, ResNetEncoder


class SAEFactory:
    """
    Factory class to create modules using registered module creation functions.
    """
    _module_registry = {}

    @staticmethod
    def register_module(name):
        """
        Decorator to register a module creation function.

        Parameters
        ----------
        name : str
            The name to register the module creation function under.
        """
        def decorator(func):
            SAEFactory._module_registry[name] = func
            return func
        return decorator

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
        if name not in SAEFactory._module_registry:
            raise ValueError(f"Module '{name}' not found in registry.")
        return SAEFactory._module_registry[name](*args, **kwargs)

    @staticmethod
    def list_modules():
        """
        Lists all registered modules.

        Returns
        -------
        list
            A list of names of all registered modules.
        """
        return list(SAEFactory._module_registry.keys())


def register_basic_templates():
    # pylint: disable=W0640
    # register some template mlp models for the factory
    for norm in ['ln', 'bn']:
        for nb_blocks in [1, 3]:
            for act in [None, 'gelu']:
                for res in [None, 'no_res']:

                    name = f"mlp_{norm}_{nb_blocks}"
                    if act is not None:
                        name = f"{name}_{act}"
                    if res is not None:
                        name = f"{name}_{res}"

                    @SAEFactory.register_module(name)
                    def create_mlp(input_size, n_components, **kwargs):
                        return MLPEncoder(
                            input_size=input_size,
                            n_components=n_components,
                            nb_blocks=nb_blocks,
                            norm_layer=nn.LayerNorm if norm == 'ln' else nn.BatchNorm1d,
                            hidden_activation=nn.GELU if act == 'gelu' else nn.ReLU,
                            residual=res != 'no_res',
                            **kwargs
                        )

    # register basics resnet and attention template
    for nb_blocks in [1, 3]:

        name_resnet = f"resnet_{nb_blocks}b"

        @SAEFactory.register_module(name_resnet)
        def create_resnet(input_channels, n_components, **kwargs):
            return ResNetEncoder(
                input_channels=input_channels,
                n_components=n_components,
                nb_blocks=nb_blocks,
                **kwargs
            )

        name_attention = f"attention_{nb_blocks}b"

        @SAEFactory.register_module(name_attention)
        def create_attention(input_shape, n_components, hidden_dim=None, **kwargs):
            return AttentionEncoder(
                input_shape=input_shape,
                n_components=n_components,
                hidden_dim=hidden_dim,
                nb_blocks=nb_blocks,
                **kwargs
            )


register_basic_templates()

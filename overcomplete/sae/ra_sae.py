import torch
from torch import nn
from typing import Optional, Any

from .topk_sae import TopKSAE
from .jump_sae import (
    JumpSAE,
) 
from .archetypal_dictionary import RelaxedArchetypalDictionary


class RATopKSAE(TopKSAE):
    """
    Relaxed Archetypal TopK SAE.

    This class implements a TopK SAE that utilizes a Relaxed Archetypal Dictionary.
    The dictionary atoms are initialized/constrained to be convex combinations of
    data points.
    """

    def __init__(
        self,
        input_shape: int,
        nb_concepts: int,
        points: torch.Tensor,
        top_k: int,
        **kwargs: Any
    ):
        """
        Args:
            input_shape (int): Dimension of the input data.
            nb_concepts (int): Number of dictionary atoms (concepts).
            points (torch.Tensor): The data points used to initialize/define the archetypes.
                                   Shape should be (num_points, input_shape).
            top_k (int): The k in TopK.
            **kwargs: Additional arguments passed to the parent TopKSAE.
        """
        # Initialize the parent TopKSAE
        super().__init__(
            input_shape=input_shape, nb_concepts=nb_concepts, top_k=top_k, **kwargs
        )

        # Overwrite the standard dictionary with the Relaxed Archetypal Dictionary
        # We assume points are on the correct device or will be moved via .to(device) later
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape, nb_concepts=nb_concepts, points=points
        )


# TODO: chck if it's "bandwith" or "bandwidth" in the original paper
class RAJumpSAE(JumpSAE):
    """
    Relaxed Archetypal Jump SAE.

    This class implements a Jump SAE that utilizes a Relaxed Archetypal Dictionary.
    """

    def __init__(
        self,
        input_shape: int,
        nb_concepts: int,
        points: torch.Tensor,
        bandwith: float = 0.001,
        **kwargs: Any
    ):
        """
        Args:
            input_shape (int): Dimension of the input data.
            nb_concepts (int): Number of dictionary atoms.
            points (torch.Tensor): The data points used for the dictionary.
            bandwidth (float): Bandwidth parameter for Jump SAE.
            **kwargs: Additional arguments passed to the parent JumpSAE.
        """
        # Initialize the parent JumpSAE
        super().__init__(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            bandwith=bandwith,
            **kwargs
        )

        # Overwrite the dictionary
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape, nb_concepts=nb_concepts, points=points
        )

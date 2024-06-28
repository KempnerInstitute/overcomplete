"""
Overcomplete: Personal toolbox for experimenting with Dictionary Learning
"""

__version__ = '0.0.1'


from .optimization import (OptimDictionaryLearning, OptimSVD, OptimKMeans,
                           OptimICA, OptimNMF, OptimPCA, OptimSparsePCA)
from .models import (DinoV2, SigLIP, ViT, ResNet, ConvNeXt)
from .sae import (MLPEncoder, ResNetEncoder, AttentionEncoder, ModuleFactory)

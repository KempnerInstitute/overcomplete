"""
Overcomplete: Personal toolbox for experimenting with Dictionary Learning.
"""

__version__ = '0.1.0'


from .optimization import (SkPCA, SkICA, SkNMF, SkKMeans,
                           SkDictionaryLearning, SkSparsePCA, SkSVD)
from .models import (DinoV2, SigLIP, ViT, ResNet, ConvNeXt)
from .sae import (MLPEncoder, ResNetEncoder, AttentionEncoder, EncoderFactory)

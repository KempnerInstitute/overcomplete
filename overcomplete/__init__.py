"""
Overcomplete: Personal toolbox for experimenting with Dictionary Learning.
"""

__version__ = '0.2.8'


from .optimization import (SkPCA, SkICA, SkNMF, SkKMeans,
                           SkDictionaryLearning, SkSparsePCA, SkSVD,
                           NMF, ConvexNMF, SemiNMF)
from .models import (DinoV2, SigLIP, ViT, ResNet, ConvNeXt)
from .sae import (SAE, TopKSAE, BatchTopKSAE, JumpSAE,
                  DictionaryLayer, RelaxedArchetypalDictionary,
                  MLPEncoder, ResNetEncoder, AttentionEncoder, EncoderFactory)
from .visualization import (overlay_top_heatmaps, evidence_top_images,
                            zoom_top_images, contour_top_image)
from .metrics import (l0, l1, l2, lp, avg_l1_loss, avg_l2_loss,
                      relative_avg_l1_loss, relative_avg_l2_loss,
                      hoyer, kappa_4, r2_score, dead_codes, hungarian_loss,
                      cosine_hungarian_loss, dictionary_collinearity,
                      wasserstein_1d, frechet_distance)

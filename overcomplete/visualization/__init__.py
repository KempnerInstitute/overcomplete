"""
Visualization module of Overcomplete.
"""

from .plot_utils import show, interpolate_torch, interpolate_cv2, get_image_dimensions
from .top_concepts import overlay_top_heatmaps, zoom_top_images, contour_top_image, evidence_top_images
from .cmaps import VIRIDIS_ALPHA, JET_ALPHA, TAB10_ALPHA

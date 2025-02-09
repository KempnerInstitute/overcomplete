# Visualization Modules

The `overcomplete.visualization` module provides a set of tools for analyzing and visualizing **top-concept activations** in batches of images. These tools help for understanding which part of an images contribute for some concept.

## Overview
The visualization module includes functions for:
- Overlaying **heatmaps** onto images to highlight top-concept activations.
- Displaying the **most representative** images for a given concept.
- Applying **contour visualizations** to emphasize highly activating regions.
- Zooming into the **hottest points** of a heatmap.
- Highlighting **evidence areas** using percentile-based heatmap thresholding.

## Example Usage
```python
from overcomplete.visualization import (overlay_top_heatmaps,
evidence_top_images, zoom_top_images, contour_top_image)
# lets imagine we have 100 images and 10k concepts maps
# and we want to visualize concept 3
images = torch.randn(100, 3, 256, 256)
heatmaps = torch.randn(100, 14, 14, 10_000)

# heatmap + transparency (recommended)
overlay_top_heatmaps(images, heatmaps, concept_id=3,
                     cmap='jet', alpha=0.35)
# transparency based
evidence_top_images(images, heatmaps, concept_id=3)
# zoom into max activating crops
zoom_top_images(images, heatmaps, concept_id=3, zoom_size=100)
# contour of most important part (boundary)
contour_top_image(images, heatmaps, concept_id=3)
```

For more details, see the module in `Overcomplete.visualization`.


{{overcomplete.visualization.top_concepts.overlay_top_heatmaps}}

{{overcomplete.visualization.top_concepts.evidence_top_images}}

{{overcomplete.visualization.top_concepts.zoom_top_images}}

{{overcomplete.visualization.top_concepts.contour_top_image}}


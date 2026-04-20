"""
xAILab Bamberg
University of Bamberg

@description:
Package initialization for the metrics module.
"""

from .color_metrics import (
    compute_wasserstein_distance,
    compute_histogram_distance,
    compute_color_moment_distance
)

from .content_metrics import (
    compute_luminance_ssim,
    compute_lpips_distance,
    compute_edge_similarity,
    compute_depth_consistency
)
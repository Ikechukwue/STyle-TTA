"""
Geometric TTA — 16-view standard geometric augmentation baseline.

Reference: Section 1.2, 6.2 of the RetriStyle-TTA report.

Generates 16 augmented views of a test image using combinations of
horizontal flips, vertical flips, 90°-rotations, and five-crops.
Predictions are averaged (soft voting) across all views.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


class GeometricTTA:
    """Standard 16-view geometric Test-Time Augmentation.

    Generates transformed copies via crops, flips, and rotations, runs
    each through the frozen classifier, and aggregates via soft voting.

    Args:
        model: Frozen classifier ``(B, C, H, W) -> (B, num_classes)``.
        input_size: Spatial resolution the classifier expects.
        crop_fraction: Fraction of the image kept for corner / centre crops.
        device: Target torch device.
    """

    FLIP_MODES: List[str] = ["none", "hflip", "vflip", "hflip+vflip"]
    ROTATION_ANGLES: List[int] = [0, 90, 180, 270]

    def __init__(
        self,
        model: nn.Module,
        input_size: int = 224,
        crop_fraction: float = 0.875,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model.eval()
        self.input_size = input_size
        self.crop_fraction = crop_fraction
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # View generation
    # ------------------------------------------------------------------
    def _generate_views(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Return up to 16 augmented views of *image* ``(1, 3, H, W)``."""
        views: List[torch.Tensor] = []
        img = image.squeeze(0)  # (3, H, W)

        for angle in self.ROTATION_ANGLES:
            rotated = torch.rot90(img, k=angle // 90, dims=[1, 2])
            for flip in self.FLIP_MODES:
                view = rotated.clone()
                if "hflip" in flip:
                    view = view.flip(dims=[2])
                if "vflip" in flip:
                    view = view.flip(dims=[1])
                # Resize to model input size
                view = F.interpolate(
                    view.unsqueeze(0),
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                views.append(view)
        return views  # length 16

    # ------------------------------------------------------------------
    # Inverse transforms for optional pixel-level tasks
    # ------------------------------------------------------------------
    @staticmethod
    def _invert_view(
        view: torch.Tensor, angle: int, flip: str
    ) -> torch.Tensor:
        """Reverse flip then rotation (inverse of generation order)."""
        if "vflip" in flip:
            view = view.flip(dims=[-2])
        if "hflip" in flip:
            view = view.flip(dims=[-1])
        view = torch.rot90(view, k=-(angle // 90), dims=[-2, -1])
        return view

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run TTA on a single image and return the soft-voted logits.

        Args:
            image: ``(1, 3, H, W)`` tensor in [0, 1].

        Returns:
            Averaged logits ``(1, num_classes)``.
        """
        image = image.to(self.device)
        views = self._generate_views(image)
        batch = torch.stack(views, dim=0).to(self.device)  # (16, 3, H, W)
        logits = self.model(batch)  # (16, num_classes)
        return logits.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Run TTA on a batch of images.

        Args:
            images: ``(B, 3, H, W)`` tensor.

        Returns:
            Averaged logits ``(B, num_classes)``.
        """
        results = []
        for i in range(images.size(0)):
            results.append(self.predict(images[i : i + 1]))
        return torch.cat(results, dim=0)

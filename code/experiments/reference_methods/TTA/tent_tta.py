"""
TENT — Test-time Entropy Minimization.

Reference: Section 1.2, 2.3, 6.2 of the RetriStyle-TTA report.
Paper: "Tent: Fully Test-Time Adaptation by Entropy Minimization"
       Wang et al., ICLR 2021

Updates *only* the affine parameters (gamma, beta) of BatchNorm layers
on-the-fly by minimising the Shannon entropy of the model's predictions.
All other parameters stay frozen.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TENT:
    """Test-time Entropy Minimization (TENT) baseline.

    Args:
        model: Pre-trained classifier.
        lr: Learning rate for the BN affine parameter update.
        steps: Number of gradient steps per test sample / batch.
        episodic: If ``True``, reset BN parameters after every sample
            (single-sample adaptation).  If ``False``, adapt continuously.
        device: Target torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        steps: int = 1,
        episodic: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.lr = lr
        self.steps = steps
        self.episodic = episodic
        self.device = device or next(model.parameters()).device

        # Prepare model: freeze everything, then un-freeze BN affine params
        self.model.eval()
        self.model.requires_grad_(False)
        self._bn_params: List[nn.Parameter] = []
        self._bn_state_backup = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Keep running stats but enable training of affine params
                module.requires_grad_(True)
                module.track_running_stats = False  # use batch stats
                if module.weight is not None:
                    self._bn_params.append(module.weight)
                if module.bias is not None:
                    self._bn_params.append(module.bias)
                # Back up original state for episodic resets
                self._bn_state_backup[name] = {
                    k: v.clone() for k, v in module.state_dict().items()
                }

        self.optimizer = torch.optim.Adam(self._bn_params, lr=self.lr)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of softmax probabilities (mean over batch)."""
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        return -(probs * log_probs).sum(dim=1).mean()

    def _reset_bn(self) -> None:
        """Restore BN parameters to their original values."""
        for name, module in self.model.named_modules():
            if name in self._bn_state_backup:
                module.load_state_dict(self._bn_state_backup[name])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Adapt BN parameters on *images* and return the final logits.

        Args:
            images: ``(B, 3, H, W)`` tensor in [0, 1] (pre-normalised).

        Returns:
            Logits ``(B, num_classes)``.
        """
        if self.episodic:
            self._reset_bn()

        images = images.to(self.device)

        for _ in range(self.steps):
            logits = self.model(images)
            loss = self._entropy(logits)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Final forward pass (eval mode — clean logits)
        with torch.no_grad():
            logits = self.model(images)
        return logits

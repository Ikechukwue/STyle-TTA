"""
CLIP and DINOv2 Classifier Wrappers for Thesis
===============================================

Provides unified classifier interfaces for VLMs and foundation models
that can be loaded via the existing ``load_classifier`` pathway.

Supported models:
    - **CLIP ViT-B/16** — zero-shot classification via text prompts
    - **DINOv2 ViT-B/14** — linear probe on frozen features

Usage::

    # Zero-shot CLIP (no training needed)
    model = load_clip_classifier(
        model_name="ViT-B-16", num_classes=200,
        class_names=[...], device="cuda"
    )
    logits = model(images)  # (B, num_classes)

    # DINOv2 linear probe (needs training or loading)
    model = load_dinov2_classifier(
        num_classes=200, device="cuda",
        weights_path="path/to/linear_head.pth"  # optional
    )
    logits = model(images)  # (B, num_classes)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable
from experiments.train import prepare_dataloaders
from experiments.utils.reproducibility import random_seed
# ====================================================================
# CLIP Zero-Shot Classifier
# ====================================================================
class CLIPZeroShotClassifier(nn.Module):
    """Wraps an open_clip model for zero-shot classification.

    The text encoder produces a fixed text embedding matrix from class
    names at init time.  At inference, the visual encoder embeds images
    and classification is done via cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        class_names: Optional[List[str]] = None,
        num_classes: int = 1000,
        prompt_template: str = "a photo of a {}.",
        device: str = "cuda",
    ):
        super().__init__()
        import open_clip

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        # If class names not given, use ImageNet-1k names
        if class_names is None:
            class_names = self._imagenet_class_names(num_classes)

        # Pre-compute text embeddings for all classes
        prompts = [prompt_template.format(name) for name in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        self.register_buffer("text_features", text_features)  # (C, D)

        # CLIP logit scale
        self.logit_scale = self.model.logit_scale
        print("Zero Shot Model loaded")
    @staticmethod
    def _imagenet_class_names(num_classes: int) -> List[str]:
        """Return ImageNet class names (falls back to generic labels)."""
        try:
            # Try to use timm's ImageNet class map
            import json
            data_dir = 'data/imagenet/imagenet1k'
            idx_to_label = data_dir / "imagenet_class_index.json"
            if idx_to_label.exists():
                with open(idx_to_label) as f:
                    mapping = json.load(f)
                names = [mapping[str(i)][1].replace("_", " ") for i in range(num_classes)]
                return names
        except Exception:
            pass
        return [f"class {i}" for i in range(num_classes)]

    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of images.

        Args:
            images: (B, 3, H, W) in [0, 1], already resized to model input size.
                    The images should be normalised with CLIP's own preprocessing
                    (mean/std), which this module applies internally.

        Returns:
            (B, C) logits (cosine similarity × temperature).
        """
        # Apply CLIP's normalization (images come in [0,1])
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.T  # (B, C)
        return logits

    def eval(self):
        self.model.eval()
        return super().eval()


class CLIPLinearProbeClassifier(nn.Module):
    """CLIP visual encoder + trainable linear head."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        num_classes: int = 1000,
        device: str = "cuda",
    ):
        super().__init__()
        import open_clip

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            feat = self.backbone.encode_image(dummy)
            feat_dim = feat.shape[-1]

        self.head = nn.Linear(feat_dim, num_classes).to(self.device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_image(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # If input is 4D (B, C, H, W), it's an image: run through backbone
        x = images
        if x.ndim == 4:
            with torch.no_grad():
                features = self.backbone.encode_image(images)
        # If input is 2D (B, D), it's already features: skip backbone
        else:
            features = x
                
        return self.head(features)


# ====================================================================
# DINOv2 Linear Probe Classifier
# ====================================================================
class DINOv2Classifier(nn.Module):
    """DINOv2 visual encoder + linear classification head.

    The backbone is frozen; only the linear head is trainable.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        num_classes: int = 1000,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load DINOv2 from Facebook's hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name,
        ).to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[-1]

        self.head = nn.Linear(feat_dim, num_classes).to(self.device)

        # Normalization stats (ImageNet)
        #self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        #self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Normalize
        #x = (images - self.mean) / self.std
        # If input is 4D (B, C, H, W), it's an image: run through backbone
        x = images
        if x.ndim == 4:
            with torch.no_grad():
                features = self.backbone(x)
        # If input is 2D (B, D), it's already features: skip backbone
        else:
            features = x
        
        return self.head(features)


# ====================================================================
# Factory functions
# ====================================================================
def load_clip_classifier(
    model_name: str = "ViT-B-16",
    num_classes: int = 1000,
    class_names: Optional[List[str]] = None,
    device: str = "cuda",
    weights_path: Optional[str] = None,
    mode: str = "linear_probe",
) -> nn.Module:
    """Load a CLIP-based classifier.

    Args:
        mode: "zero_shot" or "linear_probe"
        weights_path: Path to saved linear head weights (for linear_probe mode)
    """
    if mode == "zero_shot":
        model = CLIPZeroShotClassifier(
            model_name=model_name,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
    else:
        model = CLIPLinearProbeClassifier(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
        )
        if weights_path is not None and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=device, weights_only=True)
            if "head" in state:
                model.head.load_state_dict(state["head"])
            else:
                model.head.load_state_dict(state, strict=False)
        
    model.eval()
    return model


def load_dinov2_classifier(
    model_name: str = "dinov2_vitb14",
    num_classes: int = 1000,
    device: str = "cuda",
    weights_path: Optional[str] = None,
) -> nn.Module:
    """Load a DINOv2-based classifier with linear head."""
    model = DINOv2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
    )
    if weights_path is not None and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        if "head" in state:
            model.head.load_state_dict(state["head"])
        else:
            model.head.load_state_dict(state, strict=False)

    model.eval()
    return model

#======================================================================================
#   Pre-extraction of the relevant Datasets
#=======================================================================================

def extract_and_cache_features(
    backbone,
    dataloader: DataLoader,
    cache_path: str,
    device: torch.device,
) -> TensorDataset:
    """Run the frozen backbone once and cache features to disk."""
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached features from {cache}")
        data = torch.load(cache, weights_only=True)
        return TensorDataset(data["features"], data["labels"])


    print("Extracting features (one-time cost)...")
    all_features, all_labels = [], []
    backbone.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting..."):
            images = images.to(device)
            if hasattr(backbone, 'encode_image'):
                features = backbone.encode_image(images)
            else:
                features = backbone(images)          # (B, D)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)
    print(f"Creating  Cache at {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features_tensor, "labels": labels_tensor}, cache)
    print(f"Saved features to {cache}  shape={features_tensor.shape}")
    return TensorDataset(features_tensor, labels_tensor)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    g = random_seed(seed_value=42, use_cuda='store_true')
    
    print("Start")
    
    for model_name in ["dinov2_vitb14"]:
        if model_name == "ViT-B-16":
            model = load_clip_classifier(model_name=model_name, num_classes=200, device="cuda")
        else:
            model = load_dinov2_classifier(num_classes=200, device="cuda")
        print("Loaded Model")
        train_loader = prepare_dataloaders(dataset="imagenet", 
                                        data_path="./data",
                                        input_size=224,
                                        batch_size=256,
                                        num_workers=4,
                                        color_transfer_params=None,
                                        augmentations=[],
                                        g=g,
                                        classifier=model_name
                                        )
        print("Prepared Dataloader successfully")
        # Train Features
        _ = extract_and_cache_features(
            backbone=model.backbone,
            dataloader=train_loader,
            cache_path=f"./data/feature_cache/{model_name}/test_r.pt",
            device=device
        )

        """    
        # Val Features
        _ = extract_and_cache_features(
            backbone=model.model,
            dataloader=val_loader,
            cache_path="./data/feature_cache/clip_val.pt",
            device=device
        ) 
        """

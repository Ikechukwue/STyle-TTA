"""
Network architectures for DomainBed algorithms.

This module provides utilities to decompose timm models into featurizer (backbone)
and classifier (head) components, as required by DomainBed algorithms.
"""

import timm
import torch
import torch.nn as nn
from typing import Tuple


class TimmFeaturizer(nn.Module):
    """
    Wrapper around timm models to extract features (backbone without classification head).
    
    Args:
        classifier_name: Name of the timm model (e.g., 'resnet50', 'efficientnet_b0')
        pretrained: Whether to load pretrained weights
    """
    
    def __init__(self, classifier_name: str, pretrained: bool = False):
        super().__init__()
        
        # Create model without classification head (num_classes=0)
        self.model = timm.create_model(
            classifier_name,
            pretrained=pretrained,
            num_classes=0  # This removes the classification head
        )
        
        # Get the number of output features
        # timm models with num_classes=0 have a num_features attribute
        self.n_outputs = self.model.num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.model(x)


class Classifier(nn.Module):
    """
    Simple linear classifier head.
    
    Args:
        in_features: Number of input features (from featurizer)
        num_classes: Number of output classes
        use_bias: Whether to use bias in the linear layer
    """
    
    def __init__(self, in_features: int, num_classes: int, use_bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=use_bias)
        self.n_inputs = in_features
        self.n_outputs = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from features."""
        return self.fc(x)


class FeaturizerClassifierNetwork(nn.Module):
    """
    Combined network wrapping featurizer and classifier for easy forward pass.
    
    This is useful for saving/loading models in a format compatible with
    standard timm models.
    """
    
    def __init__(self, featurizer: TimmFeaturizer, classifier: Classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through featurizer and classifier."""
        features = self.featurizer(x)
        return self.classifier(features)


def create_featurizer_classifier(
    classifier_name: str,
    num_classes: int,
    pretrained: bool = False
) -> Tuple[TimmFeaturizer, Classifier, FeaturizerClassifierNetwork]:
    """
    Create a featurizer-classifier pair from a timm model name.
    
    Args:
        classifier_name: Name of the timm model
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights for the featurizer
        
    Returns:
        Tuple of (featurizer, classifier, network) where network is the combined model
    """
    featurizer = TimmFeaturizer(classifier_name, pretrained=pretrained)
    classifier = Classifier(featurizer.n_outputs, num_classes)
    network = FeaturizerClassifierNetwork(featurizer, classifier)
    
    return featurizer, classifier, network


def get_network_state_dict_for_timm(network, classifier_name: str, num_classes: int) -> dict:
    """
    Convert a DomainBed network state dict to a format compatible with
    standard timm models for inference.

    Handles both nn.Sequential (keys: 0.model.*, 1.fc.*) and
    FeaturizerClassifierNetwork (keys: featurizer.model.*, classifier.fc.*).

    This allows saving models in a format that can be loaded directly with
    timm.create_model() without the DomainBed wrapper.

    Args:
        network: The trained network (nn.Sequential or FeaturizerClassifierNetwork)
        classifier_name: Name of the timm model
        num_classes: Number of output classes

    Returns:
        State dict compatible with timm.create_model()
    """
    # Create a reference timm model to get the expected state dict structure
    reference_model = timm.create_model(classifier_name, pretrained=False, num_classes=num_classes)

    # Get state dicts
    network_state = network.state_dict()
    reference_state = reference_model.state_dict()

    # Detect key format: nn.Sequential uses '0.' / '1.' prefixes,
    # FeaturizerClassifierNetwork uses 'featurizer.' / 'classifier.' prefixes.
    sample_key = next(iter(network_state), '')
    uses_sequential = sample_key.startswith('0.') or sample_key.startswith('1.')

    if uses_sequential:
        featurizer_prefix = '0.model.'
        classifier_weight_key = '1.fc.weight'
        classifier_bias_key = '1.fc.bias'
    else:
        featurizer_prefix = 'featurizer.model.'
        classifier_weight_key = 'classifier.fc.weight'
        classifier_bias_key = 'classifier.fc.bias'

    # Map our state dict to timm format
    new_state_dict = {}

    # Map featurizer weights (remove featurizer prefix)
    for key, value in network_state.items():
        if key.startswith(featurizer_prefix):
            new_key = key[len(featurizer_prefix):]
            new_state_dict[new_key] = value

    # Map classifier weights to the head
    # Find the classifier key names in the reference model
    classifier_keys = [
        k for k in reference_state.keys()
        if 'fc' in k or 'head' in k or 'classifier' in k
    ]

    # Map classifier weight
    if classifier_weight_key in network_state:
        mapped = False
        for ref_key in classifier_keys:
            if 'weight' in ref_key:
                new_state_dict[ref_key] = network_state[classifier_weight_key]
                mapped = True
                break
        if not mapped:
            # Fallback: use common naming conventions
            for fallback in ['fc.weight', 'head.fc.weight', 'head.weight', 'classifier.weight']:
                if fallback in reference_state:
                    new_state_dict[fallback] = network_state[classifier_weight_key]
                    break

    # Map classifier bias
    if classifier_bias_key in network_state:
        mapped = False
        for ref_key in classifier_keys:
            if 'bias' in ref_key:
                new_state_dict[ref_key] = network_state[classifier_bias_key]
                mapped = True
                break
        if not mapped:
            for fallback in ['fc.bias', 'head.fc.bias', 'head.bias', 'classifier.bias']:
                if fallback in reference_state:
                    new_state_dict[fallback] = network_state[classifier_bias_key]
                    break

    # Validate: ensure we mapped a reasonable number of keys
    if len(new_state_dict) == 0:
        raise RuntimeError(
            f"Failed to convert network state dict to timm format. "
            f"Network has {len(network_state)} keys, sample: "
            f"{list(network_state.keys())[:5]}"
        )

    n_ref = len(reference_state)
    n_mapped = len(new_state_dict)
    if n_mapped < n_ref * 0.5:
        print(
            f"Warning: Only mapped {n_mapped}/{n_ref} keys. "
            f"Sample loaded: {list(new_state_dict.keys())[:3]}, "
            f"Sample expected: {list(reference_state.keys())[:3]}"
        )
    
    return new_state_dict

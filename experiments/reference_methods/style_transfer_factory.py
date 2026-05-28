"""Factory utilities for loading implemented style transfer methods.

This module provides a stable API used across training, inference, and
evaluation code:

- ``create_color_transfer_method`` (preferred)
- ``create_style_transfer_method`` (backward-compatible alias)
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types
from typing import Optional

from torch import nn


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# NOTE:
# - Keys are public method names accepted by CLI/code.
# - Values are import paths to modules exposing a `Method` class.
_ART = "experiments.reference_methods.style_transfer.artistic"
_DIFF = "experiments.reference_methods.style_transfer.artistic_diffusion"
_FM = "experiments.reference_methods.style_transfer.flow_matching"
_PHOTO = "experiments.reference_methods.style_transfer.photorealistic"

METHOD_MODULES = {
    # Artistic (training-required)
    "adain": f"{_ART}.adain.method",
    "adaattn": f"{_ART}.adaattn.method",
    "adaconv": f"{_ART}.adaconv.method",
    "aesfa": f"{_ART}.aesfa.method",
    "aespanet": f"{_ART}.aespanet.method",
    "artflow": f"{_ART}.artflow.method",
    "efdm": f"{_ART}.efdm.method",
    "iecontrast": f"{_ART}.iecontrast.method",
    "lst": f"{_ART}.lst.method",
    "mast": f"{_ART}.mast.method",
    "rast": f"{_ART}.rast.method",
    "sanet": f"{_ART}.sanet.method",
    "styleformer": f"{_ART}.styleformer.method",
    "stytr2": f"{_ART}.stytr2.method",

    # Artistic diffusion (training-free)
    "diffstyle": f"{_DIFF}.diffstyle.method",
    "diffuseit": f"{_DIFF}.diffuseit.method",
    "inst": f"{_DIFF}.inst.method",
    "instantstyle": f"{_DIFF}.instantstyle.method",
    "instantstyle_plus": f"{_DIFF}.instantstyle_plus.method",
    "lsast": f"{_DIFF}.lsast.method",
    "stylealign": f"{_DIFF}.stylealign.method",
    "styleid": f"{_DIFF}.styleid.method",
    "stylessp": f"{_DIFF}.stylessp.method",

    # Flow matching (FLUX family + VeloEdit)
    "flux_dev": f"{_FM}.flux.flux_dev.method",
    "flux_kontext": f"{_FM}.flux.flux_kontext.method",
    "flux_krea": f"{_FM}.flux.flux_krea.method",
    "flux_schnell": f"{_FM}.flux.flux_schnell.method",
    "flux2_dev": f"{_FM}.flux.flux2_dev.method",
    "flux2_klein_4b": f"{_FM}.flux.flux2_klein_4b.method",
    "flux2_klein_9b": f"{_FM}.flux.flux2_klein_9b.method",
    "veloedit": f"{_FM}.veloedit.method",

    # Photorealistic
    "modflows": f"{_PHOTO}.modflows.method",
    "wct2": f"{_PHOTO}.wct2.method",
    "photowct": f"{_PHOTO}.photowct.method",
    "deeppreset": f"{_PHOTO}.deeppreset.method",
    "photonas": f"{_PHOTO}.photonas.method",

    # Own contributions (outside reference_methods/)
    "lcs_veloedit": "experiments.lcs_veloedit.method",

    # PhysFlowMatch — proposed method variants (live in physflowmatch/ package)
    "physflowmatch": "physflowmatch.method",
    "physflowmatch_gradient": "physflowmatch.method_gradient",
    "physflowmatch_viscous": "physflowmatch.method_viscous",
    "physflowmatch_schrodinger": "physflowmatch.method_schrodinger",
    # PhysFlowMatch backbone variants
    "physflowmatch_flux1_dev": "physflowmatch.method_flux1_dev",
    "physflowmatch_flux1_schnell": "physflowmatch.method_flux1_schnell",
    "physflowmatch_flux2_dev": "physflowmatch.method_flux2_dev",
    "physflowmatch_flux2_klein_4b": "physflowmatch.method_flux2_klein_4b",
    "physflowmatch_flux2_klein_9b": "physflowmatch.method_flux2_klein_9b",
}

_THIS_DIR = Path(__file__).resolve().parent
_STYLE_TRANSFER_DIR = _THIS_DIR / "style_transfer"
_EXPERIMENTS_DIR = _THIS_DIR.parent  # experiments/

METHOD_FILES = {}
for name, module in METHOD_MODULES.items():
    if "style_transfer." not in module:
        # Handled via override below (e.g. lcs_veloedit lives outside style_transfer/)
        continue
    suffix_parts = module.split("style_transfer.", 1)[1].split(".")
    METHOD_FILES[name] = _STYLE_TRANSFER_DIR / Path(*suffix_parts[:-1]) / (
        f"{suffix_parts[-1]}.py"
    )

# Own contributions that live outside the style_transfer/ subtree
METHOD_FILES["lcs_veloedit"] = _EXPERIMENTS_DIR / "lcs_veloedit" / "method.py"

# PhysFlowMatch package (top-level physflowmatch/)
_PHYSFLOWMATCH_DIR = _EXPERIMENTS_DIR.parent / "physflowmatch"
METHOD_FILES["physflowmatch"]              = _PHYSFLOWMATCH_DIR / "method.py"
METHOD_FILES["physflowmatch_gradient"]    = _PHYSFLOWMATCH_DIR / "method_gradient.py"
METHOD_FILES["physflowmatch_viscous"]     = _PHYSFLOWMATCH_DIR / "method_viscous.py"
METHOD_FILES["physflowmatch_schrodinger"] = _PHYSFLOWMATCH_DIR / "method_schrodinger.py"
METHOD_FILES["physflowmatch_flux1_dev"]      = _PHYSFLOWMATCH_DIR / "method_flux1_dev.py"
METHOD_FILES["physflowmatch_flux1_schnell"]  = _PHYSFLOWMATCH_DIR / "method_flux1_schnell.py"
METHOD_FILES["physflowmatch_flux2_dev"]      = _PHYSFLOWMATCH_DIR / "method_flux2_dev.py"
METHOD_FILES["physflowmatch_flux2_klein_4b"] = _PHYSFLOWMATCH_DIR / "method_flux2_klein_4b.py"
METHOD_FILES["physflowmatch_flux2_klein_9b"] = _PHYSFLOWMATCH_DIR / "method_flux2_klein_9b.py"


def _ensure_style_transfer_namespace_package() -> None:
    """Ensure imports under `experiments.reference_methods.style_transfer.*` work.

    The repository contains both:
    - `experiments/reference_methods/style_transfer.py` (module)
    - `experiments/reference_methods/style_transfer/` (method implementations)

    Many method files import submodules under
    `experiments.reference_methods.style_transfer.<...>`, which requires
    `style_transfer` to be a package. This shim installs a namespace package
    entry in `sys.modules` for compatibility.
    """
    pkg_name = "experiments.reference_methods.style_transfer"
    existing = sys.modules.get(pkg_name)

    # If already a package-like module, keep it.
    if existing is not None and hasattr(existing, "__path__"):
        return

    ns_pkg = types.ModuleType(pkg_name)
    ns_pkg.__path__ = [str(_STYLE_TRANSFER_DIR)]
    ns_pkg.__file__ = str(_STYLE_TRANSFER_DIR)
    sys.modules[pkg_name] = ns_pkg

    # Also register flow_matching as a sub-package so that imports like
    # `from experiments.reference_methods.style_transfer.flow_matching.flux.base …`
    # work when those files are loaded via importlib.util.spec_from_file_location.
    _fm_dir = _STYLE_TRANSFER_DIR / "flow_matching"
    _fm_pkg_name = f"{pkg_name}.flow_matching"
    if _fm_pkg_name not in sys.modules:
        fm_pkg = types.ModuleType(_fm_pkg_name)
        fm_pkg.__path__ = [str(_fm_dir)]
        fm_pkg.__file__ = str(_fm_dir)
        sys.modules[_fm_pkg_name] = fm_pkg

    _flux_dir = _fm_dir / "flux"
    _flux_pkg_name = f"{_fm_pkg_name}.flux"
    if _flux_pkg_name not in sys.modules:
        flux_pkg = types.ModuleType(_flux_pkg_name)
        flux_pkg.__path__ = [str(_flux_dir)]
        flux_pkg.__file__ = str(_flux_dir)
        sys.modules[_flux_pkg_name] = flux_pkg

    # Register top-level physflowmatch package so that its sub-imports work
    # when method files are loaded via importlib.util.spec_from_file_location.
    _pfm_dir = _PHYSFLOWMATCH_DIR
    _pfm_pkg_name = "physflowmatch"
    if _pfm_pkg_name not in sys.modules and _pfm_dir.is_dir():
        pfm_pkg = types.ModuleType(_pfm_pkg_name)
        pfm_pkg.__path__ = [str(_pfm_dir)]
        pfm_pkg.__file__ = str(_pfm_dir / "__init__.py")
        sys.modules[_pfm_pkg_name] = pfm_pkg


# Backward-compatible aliases
METHOD_ALIASES = {
    "photowct2": "photowct",
}


# Methods that require local checkpoints/weights.
WEIGHTS_REQUIRED_METHODS = {
    # artistic
    "adain", "adaattn", "adaconv", "aesfa", "aespanet", "artflow",
    "avatarnet", "cast", "efdm", "iecontrast", "lst", "mast", "rast",
    "sanet", "styleformer", "stytr2", "ucast",
    # diffusion / others requiring explicit model path
    "diffstyle", "diffuseit",
    # photorealistic
    "modflows", "wct2", "photowct", "deeppreset", "photonas",
}


def _resolve_method_name(method_name: str) -> str:
    name = method_name.lower().strip()
    return METHOD_ALIASES.get(name, name)


def _build_method_instance(method_name: str, pretrained_weights: Optional[str]):
    _ensure_style_transfer_namespace_package()

    module_file = METHOD_FILES[method_name]
    if not module_file.exists():
        raise ValueError(
            f"Method file not found for '{method_name}': {module_file}"
        )

    module_name = f"_retristyle_style_method_{method_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"Could not create import spec for '{method_name}' from {module_file}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Method"):
        raise ValueError(f"Module file '{module_file}' has no 'Method' class.")

    MethodCls = module.Method
    sig = inspect.signature(MethodCls.__init__)

    kwargs = {}
    if "pretrained_weights" in sig.parameters:
        kwargs["pretrained_weights"] = pretrained_weights
    if "weights" in sig.parameters:
        kwargs["weights"] = pretrained_weights
    if "model_path" in sig.parameters:
        kwargs["model_path"] = pretrained_weights
    if "checkpoint_path" in sig.parameters:
        kwargs["checkpoint_path"] = pretrained_weights

    return MethodCls(**kwargs)


def _extract_network_container(method) -> nn.Module:
    """Build a module container for accelerator.prepare()."""

    # Common single-module attributes.
    for attr in ("network", "model", "encoder", "unet", "pipe", "wct2"):
        obj = getattr(method, attr, None)
        if isinstance(obj, nn.Module):
            return obj

    # Common paired modules.
    encoder = getattr(method, "encoder", None)
    decoder = getattr(method, "decoder", None)
    if isinstance(encoder, nn.Module) and isinstance(decoder, nn.Module):
        return nn.ModuleDict({"encoder": encoder, "decoder": decoder})

    # Wrapper objects exposing encoder/decoder modules (e.g. WCT2 wrapper).
    wct2_obj = getattr(method, "wct2", None)
    if wct2_obj is not None:
        wct2_encoder = getattr(wct2_obj, "encoder", None)
        wct2_decoder = getattr(wct2_obj, "decoder", None)
        if isinstance(wct2_encoder, nn.Module) and isinstance(wct2_decoder, nn.Module):
            return nn.ModuleDict({"encoder": wct2_encoder, "decoder": wct2_decoder})

    # DeepPreset exposes a wrapper model with generator `G`.
    model_obj = getattr(method, "model", None)
    if model_obj is not None:
        model_g = getattr(model_obj, "G", None)
        if isinstance(model_g, nn.Module):
            return model_g

    # Diffusion-style multi-component containers.
    components = {}
    for name in (
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "controlnet",
        "blip2_model",
        "vit_loss_model",
    ):
        obj = getattr(method, name, None)
        if isinstance(obj, nn.Module):
            components[name] = obj

    # Nested pipelines used by several diffusion methods.
    for parent_name in ("pipe", "pipe_inference", "pipe_inversion"):
        parent = getattr(method, parent_name, None)
        if parent is None:
            continue
        for child_name in (
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
        ):
            obj = getattr(parent, child_name, None)
            if isinstance(obj, nn.Module):
                components[f"{parent_name}_{child_name}"] = obj

    # DiffuseIT-style clip model dict.
    clip_models = getattr(method, "clip_models_dict", None)
    if isinstance(clip_models, dict):
        for clip_name, clip_model in clip_models.items():
            if isinstance(clip_model, nn.Module):
                components[f"clip_{clip_name}"] = clip_model

    if components:
        return nn.ModuleDict(components)

    # Fallback: method is a lazy-loading pipeline (e.g. diffusion models that
    # load their weights on first forward pass).  Return an empty placeholder
    # so accelerator.prepare() has something to work with.
    return nn.ModuleDict()


def create_color_transfer_method(
    method_name: str,
    pretrained_weights: Optional[str] = None,
):
    """Create and return a style/color transfer method and network container.

    Args:
        method_name: Public method name, e.g. ``adain``, ``styleid``, ``wct2``.
        pretrained_weights: Optional checkpoint path for methods requiring weights.

    Returns:
        (method_instance, network_module)
    """
    resolved = _resolve_method_name(method_name)

    if resolved not in METHOD_MODULES:
        available = ", ".join(sorted(METHOD_MODULES.keys()))
        raise ValueError(
            f"Unknown style transfer method: '{method_name}'. "
            f"Resolved as '{resolved}'. Available methods: {available}."
        )

    if resolved in WEIGHTS_REQUIRED_METHODS and pretrained_weights is None:
        raise ValueError(
            f"pretrained_weights must be provided for method '{resolved}'"
        )

    method = _build_method_instance(resolved, pretrained_weights)
    network = _extract_network_container(method)
    network.eval()
    network.requires_grad_(False)
    return method, network


def create_style_transfer_method(
    method_name: str,
    pretrained_weights: Optional[str] = None,
):
    """Backward-compatible alias for legacy call sites."""
    return create_color_transfer_method(
        method_name=method_name,
        pretrained_weights=pretrained_weights,
    )

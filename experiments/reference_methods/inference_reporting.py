"""
xAILab Bamberg
University of Bamberg

@description:
Inference speed and compute requirements evaluation script for reference color/style transfer methods.

Follows best practices for accurate GPU inference measurement:
- GPU warm-up phase (to prevent initialization latency artifacts)
- CUDA Events for precise GPU timing (avoids CPU-GPU async issues)
- Proper synchronization before/after inference
- Statistical reporting: mean, std, median, p95, p99 latencies
- Throughput measurement (samples/second)
- FLOPs/MACs computation for theoretical complexity analysis

Reference: "The Correct Way to Measure Inference Time of Deep Neural Networks"
https://medium.com/data-science/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
"""

import argparse
import json
import torch
import numpy as np
import traceback
import sys
from datetime import datetime
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# For FLOPs/MACs computation
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("Warning: ptflops not available. FLOPs/MACs will not be computed.")

from experiments.data import create_dataset
from experiments.utils import (
    ResizeWhileRetainAspectRatio, random_seed, worker_seed
)
from experiments.reference_methods.style_transfer_factory import (
    create_color_transfer_method
)


# =============================================================================
# Configuration Constants
# =============================================================================
DEFAULT_NUM_WARMUP = 50  # Warm-up iterations before timing
DEFAULT_NUM_ITERATIONS = 1000  # Number of timed iterations
DEFAULT_SEED = 265017005


def load_or_create_metrics_json(metrics_file: Path) -> dict:
    """Load existing metrics JSON or create new structure."""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {"metrics": {}}


def update_metrics(metrics_data: dict, method_name: str, metrics: dict) -> dict:
    """
    Update metrics data with comprehensive timing and compute statistics.
    
    Args:
        metrics_data: Existing metrics dictionary
        method_name: Name of the method being evaluated
        metrics: Dictionary containing all computed metrics
        
    Returns:
        Updated metrics dictionary
    """
    if "metrics" not in metrics_data:
        metrics_data["metrics"] = {}
    
    metrics_data["metrics"][method_name] = metrics
    return metrics_data


def load_or_create_error_log(error_log_file: Path) -> dict:
    """Load existing error log or create new structure."""
    if error_log_file.exists():
        with open(error_log_file, 'r') as f:
            return json.load(f)
    return {"errors": [], "last_updated": None}


def save_error_report(
    error_log_file: Path,
    method_name: str,
    device: str,
    error_type: str,
    error_message: str,
    error_traceback: str,
    stage: str = "unknown"
) -> None:
    """
    Save error information to a JSON report file for later analysis.
    
    Args:
        error_log_file: Path to the error log JSON file
        method_name: Name of the method that failed
        device: Device type ('cuda' or 'cpu')
        error_type: Type/class of the exception
        error_message: The error message
        error_traceback: Full traceback string
        stage: Stage where error occurred (e.g., 'model_loading', 'warmup', 'inference')
    """
    error_data = load_or_create_error_log(error_log_file)
    
    error_entry = {
        "method": method_name,
        "device": device,
        "method_key": f"{method_name}_{device}",
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": error_traceback,
    }
    
    # Remove any previous errors for this method+device combination
    error_data["errors"] = [
        e for e in error_data["errors"] 
        if e.get("method_key") != error_entry["method_key"]
    ]
    
    error_data["errors"].append(error_entry)
    error_data["last_updated"] = datetime.now().isoformat()
    
    with open(error_log_file, 'w') as f:
        json.dump(error_data, f, indent=4)


def enforce_device_placement(
    method_name: str,
    color_transfer_transform,
    color_transfer_model,
    device: torch.device,
    accelerator
) -> None:
    """
    Enforce device placement for all model components.
    
    Many reference methods have internal components that don't properly move
    when calling .to(device) on the main model. This function handles
    method-specific device placement to ensure all components are on the
    correct device.
    
    Args:
        method_name: Name of the color transfer method
        color_transfer_transform: The callable transform/method
        color_transfer_model: The underlying model
        device: Target device (cuda or cpu)
        accelerator: Accelerator instance for logging
    """
    accelerator.print(f"Enforcing device placement for {method_name} -> {device}")
    
    def move_all_to_device(module, target_device):
        """Recursively move all parameters and buffers to device."""
        if module is None:
            return
        if hasattr(module, 'to'):
            module.to(target_device)
        # Also handle nested modules that might be stored as attributes
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.nn.Module):
                    attr.to(target_device)
            except (AttributeError, RuntimeError):
                pass
    
    # =========================================================================
    # Training-Required Style Transfer Methods
    # =========================================================================
    if method_name == "stytr2":
        # StyTr2 has encoder, decoder, and transformer components
        method = color_transfer_transform
        if hasattr(method, 'network'):
            method.network.to(device)
        if hasattr(method, 'patch_embed'):
            method.patch_embed.to(device)
        if hasattr(method, 'encoder'):
            method.encoder.to(device)
        if hasattr(method, 'decoder'):
            method.decoder.to(device)
        if hasattr(method, 'transformer'):
            method.transformer.to(device)
        # Move the entire method's network
        for attr_name in ['network', 'model', 'net']:
            if hasattr(method, attr_name):
                net = getattr(method, attr_name)
                if net is not None:
                    move_all_to_device(net, device)
    
    # =========================================================================
    # Training-Free Diffusion Methods
    # =========================================================================
    elif method_name == "styleid":
        # StyleID uses VAE, UNet, and text encoder from diffusers
        method = color_transfer_transform
        if hasattr(method, 'vae'):
            method.vae.to(device)
        if hasattr(method, 'unet'):
            method.unet.to(device)
        if hasattr(method, 'text_encoder'):
            method.text_encoder.to(device)
        if hasattr(method, 'pipe'):
            if hasattr(method.pipe, 'to'):
                method.pipe.to(device)
        # Update internal device reference if exists
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "instantstyle":
        # InstantStyle uses SDXL pipeline with IP-Adapter
        method = color_transfer_transform
        if hasattr(method, 'pipe'):
            method.pipe.to(device)
        if hasattr(method, 'image_encoder'):
            method.image_encoder.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "diffstyle":
        # DiffStyle uses improved DDPM UNet
        method = color_transfer_transform
        if hasattr(method, 'model'):
            method.model.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "diffuseit":
        # DiffuseIT uses CLIP and diffusion model
        method = color_transfer_transform
        if hasattr(method, 'model'):
            method.model.to(device)
        if hasattr(method, 'clip_model'):
            method.clip_model.to(device)
        if hasattr(method, 'clip_models'):
            for clip_model in method.clip_models:
                if clip_model is not None:
                    clip_model.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "stylessp":
        # StyleSSP uses BLIP-2 and SDXL with IP-Adapter
        method = color_transfer_transform
        if hasattr(method, 'blip_model'):
            method.blip_model.to(device)
        if hasattr(method, 'blip_processor'):
            pass  # Processor doesn't need device
        if hasattr(method, 'pipe'):
            method.pipe.to(device)
        if hasattr(method, 'device'):
            method.device = device
    
    # =========================================================================
    # Photorealistic Color Transfer Methods
    # =========================================================================
    elif method_name == "photonas":
        # PhotoNAS has encoder and decoder
        method = color_transfer_transform
        if hasattr(method, 'encoder'):
            method.encoder.to(device)
        if hasattr(method, 'decoder'):
            method.decoder.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "deeppreset":
        # DeepPreset has a generator network
        method = color_transfer_transform
        if hasattr(method, 'model'):
            method.model.to(device)
            if hasattr(method.model, 'G'):
                method.model.G.to(device)
        if hasattr(method, 'G'):
            method.G.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "modflows":
        # ModFlows has encoder and flow model
        method = color_transfer_transform
        if hasattr(method, 'encoder'):
            method.encoder.to(device)
        if hasattr(method, 'flow'):
            method.flow.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "wct2":
        # WCT2 has encoder, decoder, and whitening/coloring transforms
        method = color_transfer_transform
        if hasattr(method, 'wct2'):
            wct2 = method.wct2
            if hasattr(wct2, 'encoder'):
                wct2.encoder.to(device)
            if hasattr(wct2, 'decoder'):
                wct2.decoder.to(device)
            if hasattr(wct2, 'device'):
                wct2.device = device
        if hasattr(method, 'device'):
            method.device = device
    
    # =========================================================================
    # Medical Color Transfer Methods
    # =========================================================================
    elif method_name == "contrimix":
        method = color_transfer_transform
        if hasattr(method, 'network'):
            method.network.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "sgvits":
        method = color_transfer_transform
        if hasattr(method, 'network'):
            method.network.to(device)
        if hasattr(method, 'device'):
            method.device = device
            
    elif method_name == "stylizing_vit":
        method = color_transfer_transform
        if hasattr(method, 'network'):
            method.network.to(device)
        if hasattr(method, 'device'):
            method.device = device
    
    # =========================================================================
    # Generic fallback for other methods
    # =========================================================================
    else:
        # Try generic approach for unknown methods
        method = color_transfer_transform
        for attr_name in ['network', 'model', 'net', 'encoder', 'decoder', 
                          'vae', 'unet', 'pipe', 'G', 'D']:
            if hasattr(method, attr_name):
                attr = getattr(method, attr_name)
                if attr is not None and hasattr(attr, 'to'):
                    try:
                        attr.to(device)
                    except Exception:
                        pass
        if hasattr(method, 'device'):
            method.device = device
    
    accelerator.print(f"Device enforcement complete for {method_name}")


def compute_percentile(times: np.ndarray, percentile: float) -> float:
    """Compute percentile of timing measurements."""
    return float(np.percentile(times, percentile))


def compute_model_complexity(model, input_size: tuple, device: torch.device, accelerator) -> dict:
    """
    Compute FLOPs/MACs for a model.
    
    Note: For style transfer models that take two inputs (content, style),
    we approximate by measuring single-input complexity and noting this limitation.
    
    Args:
        model: PyTorch model
        input_size: Tuple of (C, H, W) for input
        device: Device to run computation on
        accelerator: Accelerator instance for logging
        
    Returns:
        Dictionary with complexity metrics
    """
    complexity_metrics = {
        "flops": None,
        "macs": None,
        "parameters": None,
        "note": "Complexity measured for single forward pass; style transfer uses content+style inputs"
    }
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        complexity_metrics["parameters"] = {
            "total": total_params,
            "trainable": trainable_params
        }
        accelerator.print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
        
    except Exception as e:
        accelerator.print(f"  Could not count parameters: {e}")
    
    # Try ptflops first (more robust for complex models)
    if PTFLOPS_AVAILABLE:
        try:
            with torch.no_grad():
                macs, params = get_model_complexity_info(
                    model,
                    input_size,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False
                )
                complexity_metrics["macs"] = macs
                complexity_metrics["flops"] = macs * 2  # 1 MAC ≈ 2 FLOPs
                accelerator.print(f"  MACs: {macs:,} ({macs/1e9:.2f} GMACs)")
                accelerator.print(f"  FLOPs: {macs*2:,} ({macs*2/1e9:.2f} GFLOPs)")
        except Exception as e:
            accelerator.print(f"  ptflops failed: {e}")

    return complexity_metrics


def measure_inference_speed(
    method_name: str,
    model_weights: str,
    data_path: str,
    dataset_name: str,
    output_path: str,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    num_warmup: int = DEFAULT_NUM_WARMUP,
    seed: int = DEFAULT_SEED,
    num_workers: int = 4,
    compute_flops: bool = True,
    device_type: str = 'cuda',
):
    """
    Measure inference speed and compute requirements of a color/style transfer method.
    
    Follows best practices from:
    "The Correct Way to Measure Inference Time of Deep Neural Networks"
    
    Key implementation details:
    1. GPU Warm-up: Run NUM_WARMUP_ITERATIONS dummy inferences before timing
    2. CUDA Events: Use torch.cuda.Event for accurate GPU timing
    3. Synchronization: Proper cuda.synchronize() before recording
    4. No data transfer in timing: Data already on GPU before timing starts
    5. Statistical reporting: Mean, std, median, p95, p99 latencies
    6. Throughput: Samples processed per second
    
    Args:
        method_name: Name of the color transfer method
        model_weights: Path to pretrained weights (if required)
        data_path: Path to dataset
        dataset_name: Name of the dataset
        output_path: Directory for saving metrics
        num_iterations: Number of timed iterations (default: 1000)
        num_warmup: Number of warmup iterations (default: 50)
        seed: Random seed for reproducibility
        num_workers: Number of data loader workers
        compute_flops: Whether to compute FLOPs/MACs (default: True)
        device_type: Device to run inference on ('cuda' or 'cpu')
    """
    accelerator = Accelerator()
    
    # Determine device based on user preference and availability
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if device_type == 'cuda':
            accelerator.print("Warning: CUDA requested but not available. Falling back to CPU.")
    
    # ==========================================================================
    # Setup error logging
    # ==========================================================================
    base_output_dir = Path(output_path) / "inference_speed" / dataset_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    error_log_file = base_output_dir / "error_log.json"
    device_suffix = 'gpu' if device_type == 'cuda' else 'cpu'
    
    # ==========================================================================
    # Load Model
    # ==========================================================================
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Inference Speed Evaluation: {method_name.upper()}")
    accelerator.print(f"Device: {device} | Iterations: {num_iterations} | Warmup: {num_warmup}")
    accelerator.print(f"{'='*70}")
    
    accelerator.print(f"\nLoading {method_name}...")
    try:
        color_transfer_transform, color_transfer_model = create_color_transfer_method(
            method_name=method_name,
            pretrained_weights=model_weights
        )
        # Move model to specified device (not using accelerator.prepare to control device)
        color_transfer_model = color_transfer_model.to(device)
        color_transfer_model.eval()
        
        # Enforce device placement for all internal components
        # Many methods have internal modules that don't move with .to(device)
        enforce_device_placement(
            method_name=method_name,
            color_transfer_transform=color_transfer_transform,
            color_transfer_model=color_transfer_model,
            device=device,
            accelerator=accelerator
        )
        
    except Exception as e:
        error_tb = traceback.format_exc()
        accelerator.print(f"Error loading model: {e}")
        accelerator.print(error_tb)
        
        # Save error to report
        if accelerator.is_main_process:
            save_error_report(
                error_log_file=error_log_file,
                method_name=method_name,
                device=device_suffix,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=error_tb,
                stage="model_loading"
            )
            accelerator.print(f"Error logged to: {error_log_file}")
        
        sys.exit(1)

    # Get native size for the method
    native_size = 256  # Default
    if hasattr(color_transfer_transform, 'get_native_image_size'):
        native_size = color_transfer_transform.get_native_image_size()
    elif hasattr(color_transfer_transform, 'native_size'):
        native_size = color_transfer_transform.native_size
        
    accelerator.print(f"Native image size: {native_size}x{native_size}")

    # Metrics file is in the same directory as error log
    metrics_file = base_output_dir / "metrics.json"
    
    metrics_data = load_or_create_metrics_json(metrics_file)

    # ==========================================================================
    # Compute Model Complexity (FLOPs/MACs/Parameters)
    # ==========================================================================
    complexity_metrics = {}
    if compute_flops:
        accelerator.print(f"\nComputing model complexity...")
        complexity_metrics = compute_model_complexity(
            model=color_transfer_model,
            input_size=(3, native_size, native_size),
            device=device,
            accelerator=accelerator
        )

    # ==========================================================================
    # Prepare Data
    # ==========================================================================
    accelerator.print(f"\nPreparing datasets (seed={seed})...")
    g = random_seed(seed)
    
    transform = v2.Compose([
        ResizeWhileRetainAspectRatio(size=native_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    # Content dataset
    content_dataset = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split='train',
        transform=transform,
        use_subset=True,
        subset_size=num_iterations + num_warmup,  # Extra for warmup
        subset_seed=seed
    )
    
    # Style dataset (different seed for different random selection)
    style_dataset = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split='train',
        transform=transform,
        use_subset=True,
        subset_size=num_iterations + num_warmup,
        subset_seed=seed + 1 
    )
    
    content_loader = DataLoader(
        content_dataset, 
        batch_size=1, 
        num_workers=num_workers, 
        shuffle=False,
        worker_init_fn=worker_seed,
        generator=g,
        pin_memory=True  # Faster host-to-device transfer
    )
    
    style_loader = DataLoader(
        style_dataset, 
        batch_size=1, 
        num_workers=num_workers, 
        shuffle=False,
        worker_init_fn=worker_seed,
        generator=g,
        pin_memory=True
    )
    
    content_loader, style_loader = accelerator.prepare(content_loader, style_loader)

    # ==========================================================================
    # Warm-up Phase
    # ==========================================================================
    device_name = 'GPU' if device.type == 'cuda' else 'CPU'
    accelerator.print(f"\n{device_name} Warm-up ({num_warmup} iterations)...")
    
    warmup_iter = iter(zip(content_loader, style_loader))
    try:
        for warmup_i in tqdm(range(num_warmup), 
                      desc="Warm-up", 
                      disable=not accelerator.is_local_main_process):
            try:
                (content_img, _), (style_img, _) = next(warmup_iter)
            except StopIteration:
                break
                
            # Ensure correct shape [1, C, H, W]
            if content_img.dim() == 3:
                content_img = content_img.unsqueeze(0)
            if style_img.dim() == 3:
                style_img = style_img.unsqueeze(0)
                
            content_img = content_img.to(device)
            style_img = style_img.to(device)
            
            with torch.no_grad():
                _ = color_transfer_transform(content_img, style_img)
    except Exception as e:
        error_tb = traceback.format_exc()
        accelerator.print(f"Error during warmup (iteration {warmup_i}): {e}")
        accelerator.print(error_tb)
        
        if accelerator.is_main_process:
            save_error_report(
                error_log_file=error_log_file,
                method_name=method_name,
                device=device_suffix,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=error_tb,
                stage="warmup"
            )
            accelerator.print(f"Error logged to: {error_log_file}")
        
        sys.exit(1)
    
    # Synchronize after warmup
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ==========================================================================
    # Timed Inference Measurement
    # ==========================================================================
    accelerator.print(f"\nMeasuring inference time ({num_iterations} iterations)...")
    
    # Use CUDA Events for accurate GPU timing (only if on GPU)
    use_cuda_events = device.type == 'cuda'
    
    if use_cuda_events:
        # Pre-allocate CUDA events for all iterations
        starters = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        enders = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    
    timings = np.zeros(num_iterations)
    
    # Create fresh iterator for timed runs
    timed_iter = iter(zip(content_loader, style_loader))
    
    # Skip warmup samples (they've already been consumed from the iterator)
    # Actually, iterators are independent, so we need to skip in the main loop
    
    try:
        with torch.no_grad():
            for i in tqdm(range(num_iterations), 
                          desc="Measuring", 
                          disable=not accelerator.is_local_main_process):
                try:
                    (content_img, _), (style_img, _) = next(timed_iter)
                except StopIteration:
                    accelerator.print(f"Warning: Dataset exhausted at iteration {i}")
                    timings = timings[:i]
                    break
                    
                # Ensure correct shape [1, C, H, W]
                if content_img.dim() == 3:
                    content_img = content_img.unsqueeze(0)
                if style_img.dim() == 3:
                    style_img = style_img.unsqueeze(0)
                
                # Move to device BEFORE timing (don't measure data transfer)
                content_img = content_img.to(device)
                style_img = style_img.to(device)
                
                if use_cuda_events:
                    # CUDA Events timing (most accurate for GPU)
                    torch.cuda.synchronize()  # Wait for data transfer to complete
                    starters[i].record()
                    
                    _ = color_transfer_transform(content_img, style_img)
                    
                    enders[i].record()
                    torch.cuda.synchronize()  # Wait for inference to complete
                    
                    # Get elapsed time in milliseconds, convert to seconds
                    timings[i] = starters[i].elapsed_time(enders[i]) / 1000.0
                else:
                    # CPU timing
                    import time
                    start = time.perf_counter()
                    
                    _ = color_transfer_transform(content_img, style_img)
                    
                    end = time.perf_counter()
                    timings[i] = end - start
    except Exception as e:
        error_tb = traceback.format_exc()
        accelerator.print(f"Error during inference measurement (iteration {i}): {e}")
        accelerator.print(error_tb)
        
        if accelerator.is_main_process:
            save_error_report(
                error_log_file=error_log_file,
                method_name=method_name,
                device=device_suffix,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=error_tb,
                stage="inference"
            )
            accelerator.print(f"Error logged to: {error_log_file}")
        
        sys.exit(1)

    # ==========================================================================
    # Compute Statistics
    # ==========================================================================
    accelerator.print(f"\nComputing statistics...")
    
    # Convert to milliseconds for reporting
    timings_ms = timings * 1000
    
    # Core statistics
    mean_time = float(np.mean(timings))
    std_time = float(np.std(timings))
    median_time = float(np.median(timings))
    min_time = float(np.min(timings))
    max_time = float(np.max(timings))
    
    # Percentile latencies (important for tail latency analysis)
    p50 = compute_percentile(timings, 50)
    p90 = compute_percentile(timings, 90)
    p95 = compute_percentile(timings, 95)
    p99 = compute_percentile(timings, 99)
    
    # Throughput (samples per second)
    throughput = 1.0 / mean_time if mean_time > 0 else 0.0
    
    # Total time for all iterations
    total_time = float(np.sum(timings))

    # ==========================================================================
    # Report Results
    # ==========================================================================
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Results: {method_name.upper()}")
    accelerator.print(f"{'='*70}")
    accelerator.print(f"\nLatency Statistics (seconds):")
    accelerator.print(f"  Mean:   {mean_time:.6f} ± {std_time:.6f} s ({mean_time*1000:.3f} ms)")
    accelerator.print(f"  Median: {median_time:.6f} s ({median_time*1000:.3f} ms)")
    accelerator.print(f"  Min:    {min_time:.6f} s ({min_time*1000:.3f} ms)")
    accelerator.print(f"  Max:    {max_time:.6f} s ({max_time*1000:.3f} ms)")
    accelerator.print(f"\nPercentile Latencies:")
    accelerator.print(f"  p50: {p50*1000:.3f} ms")
    accelerator.print(f"  p90: {p90*1000:.3f} ms")
    accelerator.print(f"  p95: {p95*1000:.3f} ms")
    accelerator.print(f"  p99: {p99*1000:.3f} ms")
    accelerator.print(f"\nThroughput: {throughput:.2f} samples/second")
    accelerator.print(f"Total time for {len(timings)} iterations: {total_time:.2f} s")

    # ==========================================================================
    # Save Metrics
    # ==========================================================================
    if accelerator.is_main_process:
        # Create method key with device suffix for separate GPU/CPU metrics
        device_suffix = 'gpu' if device.type == 'cuda' else 'cpu'
        method_key = f"{method_name}_{device_suffix}"
        
        method_metrics = {
            # Timing metrics
            "latency": {
                "mean_s": mean_time,
                "std_s": std_time,
                "median_s": median_time,
                "min_s": min_time,
                "max_s": max_time,
                "mean_ms": mean_time * 1000,
                "std_ms": std_time * 1000,
            },
            "percentiles": {
                "p50_s": p50,
                "p90_s": p90,
                "p95_s": p95,
                "p99_s": p99,
                "p50_ms": p50 * 1000,
                "p90_ms": p90 * 1000,
                "p95_ms": p95 * 1000,
                "p99_ms": p99 * 1000,
            },
            "throughput": {
                "samples_per_second": throughput,
            },
            "experiment_config": {
                "num_iterations": len(timings),
                "num_warmup_iterations": num_warmup,
                "native_size": native_size,
                "seed": seed,
                "dataset": dataset_name,
                "device": device.type,
                "timing_method": "cuda_events" if use_cuda_events else "cpu_time",
            },
            # Complexity metrics (if computed)
            "complexity": complexity_metrics if complexity_metrics else None,
        }
        
        metrics_data = update_metrics(metrics_data, method_key, method_metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        accelerator.print(f"\nMetrics saved to: {metrics_file}")
        accelerator.print(f"Method key: {method_key}")
            
    accelerator.print(f"\n{'='*70}")
    accelerator.print("Done.")
    accelerator.print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Measure inference speed and compute requirements of color transfer methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure inference speed for AdaIN on GPU (default)
  python experiments/reference_methods/inference_reporting.py \\
      --method adain \\
      --model_weights /path/to/adain.pth \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results
      
  # Measure for training-free method on GPU
  python experiments/reference_methods/inference_reporting.py \\
      --method styleid \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results \\
      --device cuda
      
  # Measure on CPU with fewer iterations
  python experiments/reference_methods/inference_reporting.py \\
      --method styleid \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results \\
      --device cpu \\
      --num_iterations 50 \\
      --num_warmup 1
        """
    )
    parser.add_argument("--method", type=str, required=True, 
                        help="Method name (e.g., adain, adaattn, styleid, etc.)")
    parser.add_argument("--model_weights", type=str, default=None, 
                        help="Path to model weights (required for training-based methods)")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data directory")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name (e.g., pathmnist)")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Output path for saving metrics")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on (default: cuda)")
    parser.add_argument("--num_iterations", type=int, default=DEFAULT_NUM_ITERATIONS, 
                        help=f"Number of timed iterations (default: {DEFAULT_NUM_ITERATIONS})")
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_NUM_WARMUP, 
                        help=f"Number of warmup iterations (default: {DEFAULT_NUM_WARMUP})")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, 
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--no-flops", action="store_true",
                        help="Skip FLOPs/MACs computation")
    
    args = parser.parse_args()
    
    measure_inference_speed(
        method_name=args.method,
        model_weights=args.model_weights,
        data_path=args.data_path,
        dataset_name=args.dataset,
        output_path=args.output_path,
        num_iterations=args.num_iterations,
        num_warmup=args.num_warmup,
        num_workers=args.num_workers,
        seed=args.seed,
        compute_flops=not args.no_flops,
        device_type=args.device,
    )


if __name__ == "__main__":
    main()

"""
xAILab Bamberg
University of Bamberg

@description:
Inference speed and compute requirements evaluation script for colorist's
color transfer methods.

Follows best practices for accurate inference measurement:
- Warm-up phase (to stabilize timing)
- High-precision timing with proper synchronization
- Statistical reporting: mean, std, median, p95, p99 latencies
- Throughput measurement (samples/second)

Reference: "The Correct Way to Measure Inference Time of Deep Neural Networks"
https://medium.com/data-science/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
"""

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import List, Optional, Dict, Any

from code.experiments.data import create_dataset
from code.experiments.utils import ResizeWhileRetainAspectRatio, random_seed, worker_seed
from code.experiments.color_transfer_factory import create_color_transfer_method


# =============================================================================
# Configuration Constants
# =============================================================================
NUM_WARMUP_ITERATIONS = 50  # Warm-up iterations before timing
DEFAULT_NUM_ITERATIONS = 1000  # Number of timed iterations
DEFAULT_SEED = 265017005

# Available options for colorist methods
COLOR_SPACES = [
    'rgb', 'lab', 'hsv', 'hsi', 'hsd', 'hed',
    'lch', 'luv', 'yuv', 'ycbcr', 'yiq', 'ypbpr', 'ydbdr'
]

COLOR_TRANSFER_METHODS = ['mean_std', 'histogram', 'ehm']

STRATEGIES = ['direct', 'wavelet', 'regional', 'wavelet_regional']

WAVELET_TYPES = ['dwt', 'swt']

WAVELETS = ['haar', 'db2', 'db4', 'sym2', 'sym4', 'coif1', 'bior1.3']

WAVELET_LEVELS = [1, 2, 3, 4]

K_METHODS = [
    'silhouette', 'elbow', 'variance', 'dbscan', 'gmm_bic', 'entropy_gap'
]

MATCHING_METHODS = [
    'hungarian', 'greedy', 'jonker_volgenant',
    'min_cost_max_flow', 'gale_shapley', 'sinkhorn'
]


def load_or_create_metrics_json(metrics_file: Path) -> dict:
    """Load existing metrics JSON or create new structure."""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {"metrics": {}}


def compute_percentile(times: np.ndarray, percentile: float) -> float:
    """Compute percentile of timing measurements."""
    return float(np.percentile(times, percentile))


def create_method_key(config: Dict[str, Any]) -> str:
    """
    Create a unique key for a method configuration.
    
    Args:
        config: Dictionary with method configuration
        
    Returns:
        String key for the configuration
    """
    color_space = config.get('color_space', 'lab')
    method = config.get('color_transfer_method', 'mean_std')
    strategy = config.get('strategy', 'direct')
    
    if strategy == 'direct':
        return f"{color_space}_{method}_{strategy}"
    
    elif strategy == 'wavelet':
        wavelet_type = config.get('wavelet_type', 'swt')
        wavelet = config.get('wavelet', 'haar')
        wavelet_level = config.get('wavelet_level', 2)
        return f"{color_space}_{method}_{strategy}_{wavelet_type}_{wavelet}_L{wavelet_level}"
    
    elif strategy == 'regional':
        k_method = config.get('k_method', 'dbscan')
        matching = config.get('matching_method', 'hungarian')
        return f"{color_space}_{method}_{strategy}_{k_method}_{matching}"
    
    elif strategy == 'wavelet_regional':
        wavelet_type = config.get('wavelet_type', 'swt')
        wavelet = config.get('wavelet', 'haar')
        wavelet_level = config.get('wavelet_level', 2)
        k_method = config.get('k_method', 'dbscan')
        matching = config.get('matching_method', 'hungarian')
        return (f"{color_space}_{method}_{strategy}_"
                f"{wavelet_type}_{wavelet}_L{wavelet_level}_{k_method}_{matching}")
    
    return f"{color_space}_{method}_{strategy}_unknown"


def measure_inference_speed(
    # Color space and method
    color_space: str,
    color_transfer_method: str,
    channels: List[int],
    # Strategy parameters
    strategy: str,
    wavelet_type: str,
    wavelet: str,
    wavelet_level: int,
    wavelet_coeff_level: int,
    k_method: str,
    matching_method: str,
    min_k: int,
    max_k: int,
    spatial_weight: float,
    regional_channels: Optional[List[int]],
    wavelet_channels: Optional[List[int]],
    # Data parameters
    data_path: str,
    dataset_name: str,
    output_path: str,
    input_size: int = 256,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    seed: int = DEFAULT_SEED,
    num_workers: int = 4,
):
    """
    Measure inference speed of a colorist color transfer configuration.
    
    Follows best practices for accurate timing:
    1. Warm-up: Run NUM_WARMUP_ITERATIONS before timing
    2. High-precision timing: Use time.perf_counter()
    3. Statistical reporting: Mean, std, median, p95, p99 latencies
    4. Throughput: Samples processed per second
    
    Args:
        color_space: Color space name (e.g., 'lab', 'rgb', 'hsv')
        color_transfer_method: Transfer method ('mean_std', 'histogram', 'ehm')
        channels: Channels to apply transfer on
        strategy: Transfer strategy ('direct', 'wavelet', 'regional', etc.)
        wavelet_type: 'dwt' or 'swt'
        wavelet: Wavelet family ('haar', 'db2', etc.)
        wavelet_level: Decomposition level
        wavelet_coeff_level: SWT coefficient level
        k_method: Clustering method
        matching_method: Optimal transport method
        min_k: Minimum clusters
        max_k: Maximum clusters
        spatial_weight: Spatial weight for clustering
        regional_channels: Channels for regional transfer
        wavelet_channels: Channels for wavelet transfer
        data_path: Path to dataset
        dataset_name: Name of the dataset
        output_path: Directory for saving metrics
        input_size: Image size for processing
        num_iterations: Number of timed iterations
        seed: Random seed
        num_workers: Data loader workers
    """
    # Build configuration dict for method key
    config = {
        'color_space': color_space,
        'color_transfer_method': color_transfer_method,
        'strategy': strategy,
        'wavelet_type': wavelet_type,
        'wavelet': wavelet,
        'wavelet_level': wavelet_level,
        'k_method': k_method,
        'matching_method': matching_method,
    }
    method_key = create_method_key(config)
    
    # ==========================================================================
    # Setup
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"Inference Speed Evaluation: {method_key}")
    print(f"{'='*70}")
    
    print(f"\nConfiguration:")
    print(f"  Color Space: {color_space}")
    print(f"  Transfer Method: {color_transfer_method}")
    print(f"  Strategy: {strategy}")
    if strategy in ['wavelet', 'wavelet_regional']:
        print(f"  Wavelet Type: {wavelet_type}")
        print(f"  Wavelet: {wavelet}")
        print(f"  Wavelet Level: {wavelet_level}")
    if strategy in ['regional', 'wavelet_regional']:
        print(f"  K-Method: {k_method}")
        print(f"  Matching Method: {matching_method}")
    
    # Load the transfer function
    print(f"\nLoading {color_space} transfer function...")
    try:
        transfer_func = create_color_transfer_method(color_space=color_space)
    except Exception as e:
        print(f"Error loading transfer function: {e}")
        return

    # Setup output directory
    base_output_dir = Path(output_path) / "inference_speed_colorist" / dataset_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = base_output_dir / "metrics.json"
    
    metrics_data = load_or_create_metrics_json(metrics_file)

    # ==========================================================================
    # Prepare Data
    # ==========================================================================
    print(f"\nPreparing datasets (seed={seed})...")
    g = random_seed(seed)
    
    transform = v2.Compose([
        ResizeWhileRetainAspectRatio(size=input_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    # Source dataset
    source_dataset = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split='train',
        transform=transform,
        use_subset=True,
        subset_size=num_iterations + NUM_WARMUP_ITERATIONS,
        subset_seed=seed
    )
    
    # Reference dataset (different seed)
    reference_dataset = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split='train',
        transform=transform,
        use_subset=True,
        subset_size=num_iterations + NUM_WARMUP_ITERATIONS,
        subset_seed=seed + 1
    )
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=worker_seed,
        generator=g,
        pin_memory=True
    )
    
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=worker_seed,
        generator=g,
        pin_memory=True
    )

    # Build transfer kwargs
    transfer_kwargs = {
        'method': color_transfer_method,
        'channels': channels,
        'strategy': strategy,
        'wavelet_type': wavelet_type,
        'wavelet': wavelet,
        'wavelet_level': wavelet_level,
        'wavelet_coeff_level': wavelet_coeff_level,
        'k_method': k_method,
        'matching_method': matching_method,
        'min_k': min_k,
        'max_k': max_k,
        'spatial_weight': spatial_weight,
        'regional_channels': regional_channels,
        'wavelet_channels': wavelet_channels,
    }

    # ==========================================================================
    # Warm-up Phase
    # ==========================================================================
    print(f"\nWarm-up ({NUM_WARMUP_ITERATIONS} iterations)...")
    
    warmup_iter = iter(zip(source_loader, reference_loader))
    for _ in tqdm(range(NUM_WARMUP_ITERATIONS), desc="Warm-up"):
        try:
            (source_img, _), (reference_img, _) = next(warmup_iter)
        except StopIteration:
            break
        
        # Convert to numpy for colorist functions
        # colorist expects [H, W, C] numpy arrays or torch tensors
        source_np = source_img.squeeze(0).permute(1, 2, 0).numpy()
        reference_np = reference_img.squeeze(0).permute(1, 2, 0).numpy()
        
        _ = transfer_func(source_np, reference_np, **transfer_kwargs)

    # ==========================================================================
    # Timed Inference Measurement
    # ==========================================================================
    print(f"\nMeasuring inference time ({num_iterations} iterations)...")
    
    timings = np.zeros(num_iterations)
    timed_iter = iter(zip(source_loader, reference_loader))
    
    for i in tqdm(range(num_iterations), desc="Measuring"):
        try:
            (source_img, _), (reference_img, _) = next(timed_iter)
        except StopIteration:
            print(f"Warning: Dataset exhausted at iteration {i}")
            timings = timings[:i]
            break
        
        # Convert to numpy
        source_np = source_img.squeeze(0).permute(1, 2, 0).numpy()
        reference_np = reference_img.squeeze(0).permute(1, 2, 0).numpy()
        
        # Time the transfer
        start = time.perf_counter()
        _ = transfer_func(source_np, reference_np, **transfer_kwargs)
        end = time.perf_counter()
        
        timings[i] = end - start

    # ==========================================================================
    # Compute Statistics
    # ==========================================================================
    print(f"\nComputing statistics...")
    
    # Core statistics
    mean_time = float(np.mean(timings))
    std_time = float(np.std(timings))
    median_time = float(np.median(timings))
    min_time = float(np.min(timings))
    max_time = float(np.max(timings))
    
    # Percentile latencies
    p50 = compute_percentile(timings, 50)
    p90 = compute_percentile(timings, 90)
    p95 = compute_percentile(timings, 95)
    p99 = compute_percentile(timings, 99)
    
    # Throughput
    throughput = 1.0 / mean_time if mean_time > 0 else 0.0
    
    # Total time
    total_time = float(np.sum(timings))

    # ==========================================================================
    # Report Results
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"Results: {method_key}")
    print(f"{'='*70}")
    print(f"\nLatency Statistics (seconds):")
    print(f"  Mean:   {mean_time:.6f} ± {std_time:.6f} s ({mean_time*1000:.3f} ms)")
    print(f"  Median: {median_time:.6f} s ({median_time*1000:.3f} ms)")
    print(f"  Min:    {min_time:.6f} s ({min_time*1000:.3f} ms)")
    print(f"  Max:    {max_time:.6f} s ({max_time*1000:.3f} ms)")
    print(f"\nPercentile Latencies:")
    print(f"  p50: {p50*1000:.3f} ms")
    print(f"  p90: {p90*1000:.3f} ms")
    print(f"  p95: {p95*1000:.3f} ms")
    print(f"  p99: {p99*1000:.3f} ms")
    print(f"\nThroughput: {throughput:.2f} samples/second")
    print(f"Total time for {len(timings)} iterations: {total_time:.2f} s")

    # ==========================================================================
    # Save Metrics
    # ==========================================================================
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
            "num_warmup_iterations": NUM_WARMUP_ITERATIONS,
            "input_size": input_size,
            "seed": seed,
            "dataset": dataset_name,
            "color_space": color_space,
            "color_transfer_method": color_transfer_method,
            "strategy": strategy,
            "channels": channels,
        },
    }
    
    # Add strategy-specific config
    if strategy in ['wavelet', 'wavelet_regional']:
        method_metrics["experiment_config"]["wavelet_config"] = {
            "wavelet_type": wavelet_type,
            "wavelet": wavelet,
            "wavelet_level": wavelet_level,
            "wavelet_coeff_level": wavelet_coeff_level,
        }
    
    if strategy in ['regional', 'wavelet_regional']:
        method_metrics["experiment_config"]["regional_config"] = {
            "k_method": k_method,
            "matching_method": matching_method,
            "min_k": min_k,
            "max_k": max_k,
            "spatial_weight": spatial_weight,
        }
    
    # Update metrics file
    if "metrics" not in metrics_data:
        metrics_data["metrics"] = {}
    metrics_data["metrics"][method_key] = method_metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference speed of colorist color transfer methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure direct transfer in LAB color space
  python experiments/inference_reporting.py \\
      --color_space lab \\
      --color_transfer_method mean_std \\
      --strategy direct \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results

  # Measure wavelet transfer
  python experiments/inference_reporting.py \\
      --color_space lab \\
      --color_transfer_method mean_std \\
      --strategy wavelet \\
      --wavelet_type swt \\
      --wavelet haar \\
      --wavelet_level 2 \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results

  # Measure regional transfer
  python experiments/inference_reporting.py \\
      --color_space lab \\
      --color_transfer_method mean_std \\
      --strategy regional \\
      --k_method dbscan \\
      --matching_method hungarian \\
      --dataset pathmnist \\
      --data_path /data \\
      --output_path /results
        """
    )
    
    # Color space and method
    parser.add_argument("--color_space", type=str, required=True,
                        choices=COLOR_SPACES,
                        help="Color space for transfer")
    parser.add_argument("--color_transfer_method", type=str, default="mean_std",
                        choices=COLOR_TRANSFER_METHODS,
                        help="Color transfer method")
    parser.add_argument("--channels", type=int, nargs='+', default=[0, 1, 2],
                        help="Channels to apply transfer on")
    
    # Strategy parameters
    parser.add_argument("--strategy", type=str, default="direct",
                        choices=STRATEGIES,
                        help="Transfer strategy")
    parser.add_argument("--wavelet_type", type=str, default="swt",
                        choices=WAVELET_TYPES,
                        help="Wavelet type (dwt or swt)")
    parser.add_argument("--wavelet", type=str, default="haar",
                        help="Wavelet family")
    parser.add_argument("--wavelet_level", type=int, default=2,
                        help="Wavelet decomposition level")
    parser.add_argument("--wavelet_coeff_level", type=int, default=0,
                        help="SWT coefficient level")
    
    # Regional parameters
    parser.add_argument("--k_method", type=str, default="dbscan",
                        choices=K_METHODS,
                        help="Clustering method for regional transfer")
    parser.add_argument("--matching_method", type=str, default="hungarian",
                        choices=MATCHING_METHODS,
                        help="Optimal transport matching method")
    parser.add_argument("--min_k", type=int, default=2,
                        help="Minimum number of clusters")
    parser.add_argument("--max_k", type=int, default=6,
                        help="Maximum number of clusters")
    parser.add_argument("--spatial_weight", type=float, default=0.3,
                        help="Spatial weight for clustering")
    parser.add_argument("--regional_channels", type=int, nargs='+', default=None,
                        help="Channels for regional transfer")
    parser.add_argument("--wavelet_channels", type=int, nargs='+', default=None,
                        help="Channels for wavelet transfer")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for saving metrics")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Image size for processing")
    parser.add_argument("--num_iterations", type=int, default=DEFAULT_NUM_ITERATIONS,
                        help=f"Number of timed iterations (default: {DEFAULT_NUM_ITERATIONS})")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    
    args = parser.parse_args()
    
    measure_inference_speed(
        color_space=args.color_space,
        color_transfer_method=args.color_transfer_method,
        channels=args.channels,
        strategy=args.strategy,
        wavelet_type=args.wavelet_type,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_coeff_level=args.wavelet_coeff_level,
        k_method=args.k_method,
        matching_method=args.matching_method,
        min_k=args.min_k,
        max_k=args.max_k,
        spatial_weight=args.spatial_weight,
        regional_channels=args.regional_channels,
        wavelet_channels=args.wavelet_channels,
        data_path=args.data_path,
        dataset_name=args.dataset,
        output_path=args.output_path,
        input_size=args.input_size,
        num_iterations=args.num_iterations,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

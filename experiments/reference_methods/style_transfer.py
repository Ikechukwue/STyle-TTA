"""
xAILab Bamberg
University of Bamberg

@description:
Reference methods evaluation script to evaluate reference color/style transfer methods
using the same protocol as color_space_evaluation.py

Now supports multi-GPU evaluation using HuggingFace Accelerate library.
"""
import time
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Generator
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from accelerate import Accelerator

# Import data utilities
from experiments.data import create_dataset, ImageFolder

# Import evaluation metrics helper functions
from experiments.metrics.edge_metrics import set_edge_model_dir, get_edges
from experiments.metrics.depth_metrics import get_depth_map

# Import evaluation metrics
from experiments.metrics.color_metrics import (
    compute_wasserstein,
    compute_histogram_distance,
    compute_color_moment,
    compute_gatys_style_loss,
)
from experiments.metrics.distribution_metrics import compute_fid_tensor
from experiments.metrics.combined_metrics import compute_artfid_tensor
from experiments.metrics.feature_metrics import (
    compute_luminance_ssim,
    compute_ssim,
    compute_lpips,
)
from experiments.metrics.edge_metrics import (
    compute_edge_ssim,
    compute_edge_dists,
    compute_edge_adists,
)
from experiments.metrics.depth_metrics import (
    compute_depth_spearman,
    compute_depth_dists,
    compute_depth_adists,
)

from experiments.utils import ResizeWhileRetainAspectRatio, random_seed, worker_seed
from experiments.data import ImageFolder
from experiments.reference_methods.style_transfer_factory import create_style_transfer_method


def worker_seed(worker_id):
    """Set worker seed for reproducibility."""
    np.random.seed(torch.initial_seed() % 2**32)


def resize_image_tensor(img: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Resize an image tensor to target size while maintaining aspect ratio.
    
    Args:
        img: Image tensor [C, H, W] in range [0, 1]
        target_size: Target size for the shorter side
        
    Returns:
        Resized image tensor [C, H', W'] in range [0, 1]
    """
    if img.dim() == 3:
        # Single image [C, H, W]
        img = img.unsqueeze(0)  # Add batch dim [1, C, H, W]
        remove_batch = True
    else:
        remove_batch = False
    
    # Use ResizeWhileRetainAspectRatio
    resize_transform = ResizeWhileRetainAspectRatio(size=target_size)
    resized = resize_transform(img)
    
    if remove_batch:
        resized = resized.squeeze(0)  # Remove batch dim
    
    return resized


def save_edge_maps(img: torch.Tensor, file_path_prefix: str, methods: List[str] = ['ldc']) -> None:
    """
    Compute and save edge maps for an image using all specified methods.
    
    Args:
        img: Image as torch tensor in range [0,1], shape [C, H, W]
        file_path_prefix: Path prefix to save the edge maps (method name will be appended)
        methods: List of edge detection methods
    """    
    for method in methods:
        try:
            # Create file path with method name
            file_path = f"{file_path_prefix}_{method}.png"

            edges = get_edges(img, method=method)  # Returns torch.Tensor [1, H, W]
            
            # Convert grayscale to RGB for saving
            edges = edges.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
            
            save_image(edges, file_path)
        except Exception as e:
            print(f"Failed to save edge map for {method}: {e}")


def save_depth_maps(img: torch.Tensor, file_path_prefix: str, methods: List[str] = ['depthanything_v2_large']) -> None:
    """
    Compute and save depth maps for an image using all specified methods.
    
    Args:
        img: Image as torch tensor in range [0,1], shape [C, H, W]
        file_path_prefix: Path prefix to save the depth maps (method name will be appended)
        methods: List of depth estimation methods (depthpro, depthanything_v2_large, dpt_large)
    """
    
    for method in methods:
        try:
            # Create file path with method name
            file_path = f"{file_path_prefix}_{method}.png"
                        
            depth = get_depth_map(img, method=method)  # Returns torch.Tensor [1, H, W]
            
            # Normalize depth to [0, 1] range
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # Convert grayscale to RGB for saving
            depth_norm = depth_norm.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
            
            save_image(depth_norm, file_path)
        except Exception as e:
            print(f"Failed to save depth map for {method}: {e}")


def load_or_create_metrics_json(metrics_file: Path, experiment_config: Dict) -> Dict:
    """
    Load existing metrics JSON or create a new one.
    
    Args:
        metrics_file: Path to the metrics JSON file
        experiment_config: Configuration dictionary for the experiment
        
    Returns:
        Dictionary with metrics structure
    """
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        # Verify base configuration matches (excluding split/method specific fields)
        stored_config = data.get("experiment_config", {})
        base_config_keys = ["mode", "dataset_source", "dataset_reference", "num_images_source", 
                           "num_images_reference", "edge_methods", "depth_methods", "use_subset", "input_size"]
        for key in base_config_keys:
            if key in experiment_config and stored_config.get(key) != experiment_config.get(key):
                print(f"Warning: Config mismatch for {key}: stored={stored_config.get(key)}, current={experiment_config.get(key)}")
        return data
    else:
        return {
            "experiment_config": experiment_config,
            "run_seeds": [],
            "metrics": {}
        }


def update_metrics_with_seed_run(metrics_data: Dict, seed: int, run_metrics: List[Dict],
                                split_key: str, method_name: str) -> Dict:
    """
    Update the metrics data with results from a new seed run.
    
    Args:
        metrics_data: Existing metrics data structure
        seed: Seed value for this run
        run_metrics: List of per-image metrics for this run
        split_key: Key for split combination (e.g., "train-val")
        method_name: Method used (e.g., "adain", "adaattn")
        
    Returns:
        Updated metrics data
    """
    # Check if this seed already exists
    if "run_seeds" not in metrics_data:
        metrics_data["run_seeds"] = []
    
    if seed in metrics_data["run_seeds"]:
        print(f"Seed {seed} already exists. Replacing metrics for this seed.")
        seed_index = metrics_data["run_seeds"].index(seed)
    else:
        print(f"Adding new seed {seed} to metrics.")
        metrics_data["run_seeds"].append(seed)
        seed_index = len(metrics_data["run_seeds"]) - 1
    
    # Initialize metrics dict if needed
    if "metrics" not in metrics_data:
        metrics_data["metrics"] = {}
    
    # Create nested structure: split -> method
    if split_key not in metrics_data["metrics"]:
        metrics_data["metrics"][split_key] = {}
    
    if method_name not in metrics_data["metrics"][split_key]:
        metrics_data["metrics"][split_key][method_name] = {}
        
    # Compute mean values for this run
    if len(run_metrics) > 0:
        # Get all metric keys (exclude IDs)
        metric_keys = [k for k in run_metrics[0].keys() 
                      if isinstance(run_metrics[0][k], (int, float)) 
                      and k not in ["source_id", "reference_id", "output_id"]]
        
        for key in metric_keys:
            # Compute mean over all images in this run
            values = [m[key] for m in run_metrics if m[key] is not None]
            if len(values) > 0:
                mean_value = float(np.mean(values))
                
                # Initialize list if this is the first time we see this metric
                if key not in metrics_data["metrics"][split_key][method_name]:
                    metrics_data["metrics"][split_key][method_name][key] = []
                
                # Ensure the list is long enough
                while len(metrics_data["metrics"][split_key][method_name][key]) <= seed_index:
                    metrics_data["metrics"][split_key][method_name][key].append(None)
                
                # Update or add the value at the seed index
                metrics_data["metrics"][split_key][method_name][key][seed_index] = mean_value
    
    return metrics_data


def prepare_dataloaders(
    data_path: str,
    dataset_name_source: str,
    dataset_name_reference: str,
    dataset_split_source: str,
    dataset_split_reference: str,
    input_size: int = 256,
    batch_size: int = 1,  # Process one at a time
    use_subset: bool = True,
    subset_size_source: int = 20,
    subset_size_reference: int = 20,
    g: Optional[Generator] = None,
    num_workers: int = 4,
    seed: int = 265017005,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the dataloaders for source and reference images.
    """
    # Create simple transform (resize to image size and convert to tensor, keep in [0,1] range)
    transform = v2.Compose([
        ResizeWhileRetainAspectRatio(size=input_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Converts to [0,1]
    ])

    # If source and reference use the same dataset and split, use different seeds
    same_dataset_and_split = (
        dataset_name_source == dataset_name_reference and 
        dataset_split_source == dataset_split_reference
    )
    
    if same_dataset_and_split:
        source_subset_seed = seed
        reference_subset_seed = seed + 999983  # Large prime offset
        g_source = torch.Generator().manual_seed(seed)
        g_reference = torch.Generator().manual_seed(seed + 999983)
        print(f"Same dataset and split detected. Using different seeds:")
        print(f"  Source subset seed: {source_subset_seed}, dataloader seed: {seed}")
        print(f"  Reference subset seed: {reference_subset_seed}, dataloader seed: {seed + 999983}")
    else:
        source_subset_seed = seed
        reference_subset_seed = seed
        g_source = torch.Generator().manual_seed(seed)
        g_reference = torch.Generator().manual_seed(seed)
    
    # Create source dataset
    source_dataset = create_dataset(
        dataset_name=dataset_name_source,
        data_path=data_path,
        split=dataset_split_source,
        transform=transform,
        use_subset=use_subset,
        subset_size=subset_size_source,
        subset_seed=source_subset_seed
    )

    # Create reference dataset
    reference_dataset = create_dataset(
        dataset_name=dataset_name_reference,
        data_path=data_path,
        split=dataset_split_reference,
        transform=transform,
        use_subset=use_subset,
        subset_size=subset_size_reference,
        subset_seed=reference_subset_seed
    )

    # Create dataloaders
    source_loader = DataLoader(
        dataset=source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=g_source
    )

    reference_loader = DataLoader(
        dataset=reference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=g_reference
    )

    return source_loader, reference_loader


def evaluate_color_transfer(
    method_name: str,
    model_weights: Optional[str],
    data_path: str,
    dataset_source: str,
    dataset_reference: str,
    dataset_split_source: str = 'train',
    dataset_split_reference: str = 'val',
    output_path: str = './experiments/results/reference_methods',
    input_size: int = 256,
    seed: int = 265017005,
    num_images_source: int = 20,
    num_images_reference: int = 20,
    num_workers: int = 4,
    edge_methods: List[str] = ['ldc'],
    depth_methods: List[str] = ['depthanything_v2_large'],
    edge_models_dir: Optional[str] = None,
    all_combinations: bool = False,
):
    """
    Evaluate a reference method using the same protocol as color_space_evaluation.py with multi-GPU support
    
    Args:
        method_name: Name of the method to evaluate
        model_weights: Path to pretrained checkpoint
        data_path: Path to datasets
        dataset_source: Source dataset name
        dataset_reference: Reference dataset name
        dataset_split_source: Source dataset split
        dataset_split_reference: Reference dataset split
        output_path: Path to save results
        input_size: Image size for processing
        seed: Random seed
        num_images_source: Number of source images to process
        num_images_reference: Number of reference images to process
        num_workers: Number of dataloader workers
        edge_methods: Edge detection methods for evaluation
        depth_methods: Depth estimation methods for evaluation
        edge_models_dir: Directory containing edge detection model weights
        all_combinations: If True, evaluate all N×M combinations instead of pairwise
    """
    # Initialize Accelerator for multi-GPU evaluation
    accelerator = Accelerator()
    
    # Only print on main process to avoid duplicate output
    if accelerator.is_main_process:
        print("="*80)
        print(f"Evaluating Reference Method: {method_name.upper()}")
        print("="*80)
        print(f"Dataset Source: {dataset_source} ({dataset_split_source})")
        print(f"Dataset Reference: {dataset_reference} ({dataset_split_reference})")
        print(f"Number of images: {num_images_source} x {num_images_reference}")
        print(f"Mode: {'All Combinations (N×M)' if all_combinations else 'Pairwise (1-to-1)'}")
        if model_weights:
            print(f"Checkpoint: {model_weights}")
        else:
            print(f"Checkpoint: None (training-free method)")
        print(f"Output: {output_path}")
        print(f"Device: {accelerator.device}")
        print(f"Num Processes: {accelerator.num_processes}")
        print(f"Distributed Type: {accelerator.distributed_type}")
        print("="*80)

    # Set random seed
    g = random_seed(seed_value=seed)

    # Load the reference method
    accelerator.print(f"\nLoading {method_name} model...")
    color_transfer_transform, color_transfer_model = create_style_transfer_method(
        method_name=method_name,
        pretrained_weights=model_weights
    )

    color_transfer_model = accelerator.prepare(color_transfer_model)

    # Get native image size from the color transfer method
    if hasattr(color_transfer_transform, 'get_native_image_size'):
        native_size = color_transfer_transform.get_native_image_size()
        accelerator.print(f"Using native image size for color transfer: {native_size}")
    else:
        accelerator.print(f"Warning: Color transfer method does not have get_native_image_size(), using input_size: {input_size}")
        native_size = input_size

    # Fall back to requested input_size if the method returns None
    if native_size is None:
        accelerator.print(f"  → native_size is None, using requested input_size: {input_size}")
        native_size = input_size

    accelerator.print(f"✓ Model loaded successfully")
    accelerator.print(f"✓ Method native image size: {native_size}x{native_size}")
    
    # Check if we need to resize outputs
    needs_resize = (input_size != native_size)
    if needs_resize:
        accelerator.print(f"⚠ Image size mismatch: requested {input_size}x{input_size}, native {native_size}x{native_size}")
        accelerator.print(f"  → Will process at {native_size}x{native_size} and resize outputs to {input_size}x{input_size}")
    
    # Set model paths for edge/depth if provided
    if edge_models_dir is not None:
        set_edge_model_dir(path=edge_models_dir)
        accelerator.print(f"Model paths configured:")
        if edge_models_dir:
            accelerator.print(f"  Edge models: {edge_models_dir}")

    # Prepare dataloaders at NATIVE size (not requested size)
    # This ensures the model sees images at the resolution it was trained on
    accelerator.print(f"\nPreparing dataloaders at native resolution ({native_size}x{native_size})...")
    source_loader, reference_loader = prepare_dataloaders(
        data_path=data_path,
        dataset_name_source=dataset_source,
        dataset_name_reference=dataset_reference,
        dataset_split_source=dataset_split_source,
        dataset_split_reference=dataset_split_reference,
        input_size=native_size,  # Use native size for loading
        batch_size=1,
        use_subset=True,
        subset_size_source=num_images_source,
        subset_size_reference=num_images_reference,
        g=g,
        num_workers=num_workers,
        seed=seed
    )

    # Create output directories
    mode_suffix = "all_combinations" if all_combinations else "pairwise"
    base_output_dir = Path(output_path) / f"{dataset_source}-{dataset_reference}_{mode_suffix}"
    split_dir = base_output_dir / f"{dataset_split_source}-{dataset_split_reference}"
    seed_dir = split_dir / f"seed_{seed}"
    
    sources_dir = seed_dir / "sources"
    references_dir = seed_dir / "references"
    results_dir = seed_dir / "outputs" / method_name
    
    for d in [sources_dir, references_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"✓ Output directories created")
    accelerator.print(f"  Sources: {sources_dir}")
    accelerator.print(f"  References: {references_dir}")
    accelerator.print(f"  Results: {results_dir}")

    # Create maps directories
    maps_base_dir = seed_dir / "maps"
    maps_sources_dir = maps_base_dir / "sources"
    maps_references_dir = maps_base_dir / "references"
    maps_outputs_dir = maps_base_dir / "outputs" / method_name
    
    for d in [maps_base_dir, maps_sources_dir, maps_references_dir, maps_outputs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Process images
    accelerator.print(f"\nProcessing images...")
    start_time = time.time()

    source_images_list = []
    reference_images_list = []
    # inference_times = []  # Track inference time for each pair
    # mean_inference_time = None  # Initialize to None

    if all_combinations:
        # All combinations mode: Each source with each reference (N×M)
        accelerator.print("Processing in all-combinations mode (N×M)...")
        
        # First, collect all source and reference images
        accelerator.print("Loading source images...")
        for i, (source_img, _) in enumerate(tqdm(source_loader)):
            source_img = source_img.squeeze(0)  # Remove batch dimension [C, H, W]
            source_id = f"{i:04d}"
            
            # Resize to requested size if needed (for saving only)
            source_img_to_save = resize_image_tensor(source_img, input_size) if needs_resize else source_img
            
            # Save source image at requested resolution
            save_image(source_img_to_save, sources_dir / f"source_{source_id}.png")
            
            # Save edge and depth maps for source at requested resolution
            save_edge_maps(source_img_to_save, str(maps_sources_dir / f"source_{source_id}_edges"), edge_methods)
            save_depth_maps(source_img_to_save, str(maps_sources_dir / f"source_{source_id}_depth"), depth_methods)
            
            # Keep native resolution for model inference
            source_images_list.append((source_img, source_id))
        
        accelerator.print("Loading reference images...")
        for i, (ref_img, _) in enumerate(tqdm(reference_loader)):
            ref_img = ref_img.squeeze(0)  # Remove batch dimension [C, H, W]
            reference_id = f"{i:04d}"
            
            # Resize to requested size if needed (for saving only)
            ref_img_to_save = resize_image_tensor(ref_img, input_size) if needs_resize else ref_img
            
            # Save reference image at requested resolution
            save_image(ref_img_to_save, references_dir / f"reference_{reference_id}.png")
            
            # Save edge and depth maps for reference at requested resolution
            save_edge_maps(ref_img_to_save, str(maps_references_dir / f"reference_{reference_id}_edges"), edge_methods)
            save_depth_maps(ref_img_to_save, str(maps_references_dir / f"reference_{reference_id}_depth"), depth_methods)
            
            # Keep native resolution for model inference
            reference_images_list.append((ref_img, reference_id))
        
        accelerator.print(f"Loaded {len(source_images_list)} sources and {len(reference_images_list)} references.")
        accelerator.print(f"Will generate {len(source_images_list) * len(reference_images_list)} outputs...")
        
        # Apply transfer to all combinations
        accelerator.print(f"\nApplying {method_name} transfer...")
        total_combinations = len(source_images_list) * len(reference_images_list)
        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for source_img, source_id in source_images_list:
                for ref_img, reference_id in reference_images_list:
                    # Apply transfer at native resolution (add batch dimension)
                    source_batch = source_img.unsqueeze(0).to(accelerator.device)
                    ref_batch = ref_img.unsqueeze(0).to(accelerator.device)
                    
                    # Measure inference time
                    # inference_start = time.time()
                    with torch.no_grad():
                        output_img = color_transfer_transform(source_batch, ref_batch)
                    
                    # # Synchronize GPU if using CUDA
                    # if accelerator.device.type == 'cuda':
                    #     torch.cuda.synchronize()
                    # inference_end = time.time()
                    # inference_times.append(inference_end - inference_start)
                    
                    # Remove batch dimension and move to CPU
                    output_img = output_img.squeeze(0).cpu()
                    
                    # Resize output to requested size if needed
                    output_img_to_save = resize_image_tensor(output_img, input_size) if needs_resize else output_img
                    
                    # Save result at requested resolution
                    output_path = results_dir / f"output_src{source_id}_ref{reference_id}.png"
                    save_image(output_img_to_save, output_path)
                    
                    # Save edge and depth maps for output at requested resolution
                    save_edge_maps(output_img_to_save, str(maps_outputs_dir / f"output_src{source_id}_ref{reference_id}_edges"), edge_methods)
                    save_depth_maps(output_img_to_save, str(maps_outputs_dir / f"output_src{source_id}_ref{reference_id}_depth"), depth_methods)
                    
                    pbar.update(1)
        
        print(f"Saved {total_combinations} output images.")
    
    else:
        # Pairwise mode: Each source with corresponding reference (1-to-1)
        print("Processing in pairwise mode (1-to-1)...")
        
        num_pairs = min(len(source_loader.dataset), len(reference_loader.dataset))
        print(f"Will generate {num_pairs} outputs...")
        
        print("Loading and processing images...")
        for i, ((source_img, _), (ref_img, _)) in enumerate(tqdm(zip(source_loader, reference_loader), total=num_pairs)):
            source_img = source_img.squeeze(0)  # Remove batch dimension [C, H, W]
            ref_img = ref_img.squeeze(0)  # Remove batch dimension [C, H, W]
            
            source_id = f"{i:04d}"
            reference_id = f"{i:04d}"
            
            # Resize to requested size if needed (for saving only)
            source_img_to_save = resize_image_tensor(source_img, input_size) if needs_resize else source_img
            ref_img_to_save = resize_image_tensor(ref_img, input_size) if needs_resize else ref_img
            
            # Save source and reference images at requested resolution
            save_image(source_img_to_save, sources_dir / f"source_{source_id}.png")
            save_image(ref_img_to_save, references_dir / f"reference_{reference_id}.png")
            
            # Save edge and depth maps for source and reference at requested resolution
            save_edge_maps(source_img_to_save, str(maps_sources_dir / f"source_{source_id}_edges"), edge_methods)
            save_edge_maps(ref_img_to_save, str(maps_references_dir / f"reference_{reference_id}_edges"), edge_methods)
            save_depth_maps(source_img_to_save, str(maps_sources_dir / f"source_{source_id}_depth"), depth_methods)
            save_depth_maps(ref_img_to_save, str(maps_references_dir / f"reference_{reference_id}_depth"), depth_methods)
            
            # Apply transfer at native resolution
            source_batch = source_img.unsqueeze(0).to(accelerator.device)
            ref_batch = ref_img.unsqueeze(0).to(accelerator.device)
            
            # Measure inference time
            # inference_start = time.time()
            with torch.no_grad():
                output_img = color_transfer_transform(source_batch, ref_batch)
            
            # # Synchronize GPU if using CUDA
            # if accelerator.device.type == 'cuda':
            #     torch.cuda.synchronize()
            # inference_end = time.time()
            # inference_times.append(inference_end - inference_start)
            
            # Remove batch dimension and move to CPU
            output_img = output_img.squeeze(0).cpu()
            
            # Resize output to requested size if needed
            output_img_to_save = resize_image_tensor(output_img, input_size) if needs_resize else output_img
            
            # Save result at requested resolution
            output_path = results_dir / f"output_src{source_id}_ref{reference_id}.png"
            save_image(output_img_to_save, output_path)
            
            # Save edge and depth maps for output at requested resolution
            save_edge_maps(output_img_to_save, str(maps_outputs_dir / f"output_src{source_id}_ref{reference_id}_edges"), edge_methods)
            save_depth_maps(output_img_to_save, str(maps_outputs_dir / f"output_src{source_id}_ref{reference_id}_depth"), depth_methods)
        
        accelerator.print(f"Saved {num_pairs} output images.")

    end_time = time.time()
    elapsed = end_time - start_time
    
    # # Calculate mean inference speed
    # if len(inference_times) > 0:
    #     mean_inference_time = float(np.mean(inference_times))
    #     std_inference_time = float(np.std(inference_times))
    #     accelerator.print(f"\n✓ Transfer complete! Elapsed time: {elapsed:.2f} seconds")
    #     accelerator.print(f"✓ Mean inference time per image pair: {mean_inference_time:.4f} ± {std_inference_time:.4f} seconds")
    #     accelerator.print(f"✓ Throughput: {1.0/mean_inference_time:.2f} image pairs/second")
    # else:
    #     mean_inference_time = None
    #     std_inference_time = None
    #     accelerator.print(f"\n✓ Transfer complete! Elapsed time: {elapsed:.2f} seconds")
    #     accelerator.print(f"⚠ Warning: No inference times recorded")

    # Compute metrics
    accelerator.print("\nComputing metrics...")
    all_metrics = []
    
    # Create transform for loading images
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Converts to [0,1]
    ])
    
    source_dataset = ImageFolder(root=str(sources_dir), transform=transform)
    reference_dataset = ImageFolder(root=str(references_dir), transform=transform)
    output_dataset = ImageFolder(root=str(results_dir), transform=transform)
    
    num_sources = len(source_dataset)
    num_references = len(reference_dataset)
    num_outputs = len(output_dataset)
    
    accelerator.print(f"Dataset sizes:")
    accelerator.print(f"  Sources: {num_sources}")
    accelerator.print(f"  References: {num_references}")
    accelerator.print(f"  Outputs: {num_outputs}")
    
    # Step 1: Compute per-image metrics (batch_size=1)
    accelerator.print("\nStep 1/3: Computing per-image metrics (SSIM, LPIPS, edge, depth, etc.)...")
    
    output_loader_single = DataLoader(output_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for output_img, output_meta in tqdm(output_loader_single, total=num_outputs, desc="Per-image metrics"):
        # Keep as tensor [C, H, W]
        output_tensor = output_img[0]
        output_id = output_meta['image_id'][0]
        
        # Parse output filename to get source and reference IDs
        # Format: output_src{source_id}_ref{reference_id}
        parts = output_id.split('_')
        source_id = parts[1].replace('src', '')  # Remove 'src' prefix
        reference_id = parts[2].replace('ref', '')  # Remove 'ref' prefix
        
        # Load corresponding source and reference images
        source_path = sources_dir / f"source_{source_id}.png"
        reference_path = references_dir / f"reference_{reference_id}.png"
        
        # Load source image
        source_pil = Image.open(source_path)
        source_tensor = transform(source_pil)  # Keep as tensor [C, H, W]
        
        # Load reference image
        reference_pil = Image.open(reference_path)
        reference_tensor = transform(reference_pil)  # Keep as tensor [C, H, W]
        
        # Compute per-image metrics
        metrics = {}
        
        # Color metrics (per-image, compare output vs reference)
        metrics["wasserstein_distance"] = compute_wasserstein(output_tensor, reference_tensor, "rgb").value
        hist_group = compute_histogram_distance(output_tensor, reference_tensor)
        metrics["kl_divergence"] = hist_group.metrics["histogram_kl"].value
        metrics["js_divergence"] = hist_group.metrics["histogram_js"].value
        metrics["chi_square"] = hist_group.metrics["histogram_chi2"].value
        metrics["histogram_intersection"] = hist_group.metrics["histogram_intersection"].value
        metrics["color_moment_distance"] = compute_color_moment(output_tensor, reference_tensor).value
        
        # Content metrics (per-image, compare output vs source)
        metrics["luminance_ssim"] = compute_luminance_ssim(source_tensor, output_tensor).value
        metrics["ssim"] = compute_ssim(source_tensor, output_tensor).value
        metrics["lpips"] = compute_lpips(source_tensor, output_tensor).value
        
        # Edge similarity metrics (output vs source)
        for edge_method in edge_methods:
            try:
                metrics[f"edge_similarity_{edge_method}"] = compute_edge_ssim(source_tensor, output_tensor, method=edge_method).value
                metrics[f"edge_dists_{edge_method}"] = compute_edge_dists(source_tensor, output_tensor, method=edge_method).value
                metrics[f"edge_adists_{edge_method}"] = compute_edge_adists(source_tensor, output_tensor, method=edge_method).value
            except Exception as e:
                accelerator.print(f"Edge metrics ({edge_method}) failed: {e}")
                metrics[f"edge_similarity_{edge_method}"] = None
                metrics[f"edge_dists_{edge_method}"] = None
                metrics[f"edge_adists_{edge_method}"] = None
        
        # Depth consistency metrics (output vs source)
        for depth_method in depth_methods:
            try:
                metrics[f"depth_consistency_{depth_method}"] = compute_depth_spearman(source_tensor, output_tensor, method=depth_method).value
                metrics[f"depth_dists_{depth_method}"] = compute_depth_dists(source_tensor, output_tensor, method=depth_method).value
                metrics[f"depth_adists_{depth_method}"] = compute_depth_adists(source_tensor, output_tensor, method=depth_method).value
            except Exception as e:
                accelerator.print(f"Depth metrics ({depth_method}) failed: {e}")
                metrics[f"depth_consistency_{depth_method}"] = None
                metrics[f"depth_dists_{depth_method}"] = None
                metrics[f"depth_adists_{depth_method}"] = None
        
        # Add image IDs
        metrics["source_id"] = source_id
        metrics["reference_id"] = reference_id
        metrics["output_id"] = output_id
        
        all_metrics.append(metrics)
    
    # Step 2: Compute distribution-based metrics (FID, Gatys) with larger batch size
    accelerator.print("\nStep 2/3: Computing distribution-based metrics (FID, Gatys style loss)...")
    
    batch_size_distribution = min(32, max(num_outputs, num_references))
    
    output_loader_batch = DataLoader(output_dataset, batch_size=batch_size_distribution, shuffle=False, num_workers=0)
    reference_loader_batch = DataLoader(reference_dataset, batch_size=batch_size_distribution, shuffle=False, num_workers=0)
    
    # Collect all output and reference images for FID
    accelerator.print(f"Loading all outputs ({num_outputs} images)...")
    all_output_tensors = []
    for output_img, _ in output_loader_batch:
        all_output_tensors.append(output_img)
    all_outputs = torch.cat(all_output_tensors, dim=0)
    
    accelerator.print(f"Loading all references ({num_references} images)...")
    all_reference_tensors = []
    for reference_img, _ in reference_loader_batch:
        all_reference_tensors.append(reference_img)
    all_references = torch.cat(all_reference_tensors, dim=0)
    
    accelerator.print(f"Computing FID between {all_outputs.shape[0]} outputs and {all_references.shape[0]} references...")
    try:
        fid_score = compute_fid_tensor(all_outputs, all_references).value
        accelerator.print(f"FID: {fid_score:.4f}")
    except Exception as e:
        accelerator.print(f"FID computation failed: {e}")
        fid_score = None
    
    # Compute Gatys style loss on batches and average
    accelerator.print(f"Computing Gatys style loss on batches...")
    gatys_scores = []
    
    # Re-create batch loaders for synchronized iteration
    output_loader_gatys = DataLoader(output_dataset, batch_size=min(8, num_outputs), shuffle=False, num_workers=0)
    reference_loader_gatys = DataLoader(reference_dataset, batch_size=min(8, num_references), shuffle=False, num_workers=0)
    
    # If different sizes, we'll sample references cyclically for outputs
    reference_iter = iter(reference_loader_gatys)
    for output_batch, _ in output_loader_gatys:
        try:
            # Get next reference batch, cycle if needed
            try:
                reference_batch, _ = next(reference_iter)
            except StopIteration:
                reference_iter = iter(reference_loader_gatys)
                reference_batch, _ = next(reference_iter)
            
            # Match batch sizes (take minimum)
            min_batch = min(output_batch.shape[0], reference_batch.shape[0])
            output_batch = output_batch[:min_batch]
            reference_batch = reference_batch[:min_batch]
            
            gatys_loss = compute_gatys_style_loss(output_batch, reference_batch).value
            gatys_scores.append(gatys_loss)
        except Exception as e:
            accelerator.print(f"Gatys style loss failed on batch: {e}")
    
    if gatys_scores:
        gatys_mean = float(np.mean(gatys_scores))
        accelerator.print(f"Gatys style loss (mean): {gatys_mean:.4f}")
    else:
        gatys_mean = None
    
    # Step 3: Add distribution-based metrics to all_metrics
    accelerator.print("\nStep 3/3: Adding distribution-based metrics to results...")
    for metrics in all_metrics:
        metrics["fid"] = fid_score
        metrics["gatys_style_loss"] = gatys_mean
        # metrics["mean_inference_time"] = mean_inference_time
        
        # Compute ArtFID if both FID and LPIPS are available
        if fid_score is not None and metrics.get("lpips") is not None:
            try:
                metrics["artfid"] = compute_artfid_tensor(fid_score, metrics["lpips"]).value
            except Exception as e:
                accelerator.print(f"ArtFID computation failed: {e}")
                metrics["artfid"] = None
        else:
            metrics["artfid"] = None
    
    # Prepare experiment configuration
    experiment_config = {
        "mode": "all_combinations" if all_combinations else "pairwise",
        "dataset_source": dataset_source,
        "dataset_reference": dataset_reference,
        "num_images_source": num_images_source,
        "num_images_reference": num_images_reference,
        "edge_methods": edge_methods,
        "depth_methods": depth_methods,
        "use_subset": True,
        "input_size": input_size,
    }
    
    # Single metrics file for the entire dataset comparison
    metrics_file = base_output_dir / "metrics.json"
    metrics_data = load_or_create_metrics_json(metrics_file, experiment_config)
    
    # Create keys for nested structure
    split_key = f"{dataset_split_source}-{dataset_split_reference}"
    
    # Update metrics with this seed run
    metrics_data = update_metrics_with_seed_run(
        metrics_data, seed, all_metrics,
        split_key, method_name
    )
    
    # Save updated metrics to JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    accelerator.print(f"\nExperiment complete!")
    accelerator.print(f"  Seed-specific results saved to: {seed_dir}")
    accelerator.print(f"  Sources: {sources_dir}")
    accelerator.print(f"  References: {references_dir}")
    accelerator.print(f"  Outputs: {results_dir}")
    accelerator.print(f"  Maps: {maps_base_dir}")
    accelerator.print(f"  Metrics saved to: {metrics_file}")
    accelerator.print(f"  Configuration: {split_key} / {method_name}")
    accelerator.print(f"  Seed: {seed}")
    accelerator.print(f"  Total seeds in metrics file: {len(metrics_data['run_seeds'])}")
    accelerator.print(f"  Seeds: {metrics_data['run_seeds']}")
    accelerator.print(f"  Total images processed in this run: {len(all_metrics)}")
    # if mean_inference_time is not None:
    #     accelerator.print(f"  Mean inference time: {mean_inference_time:.4f} seconds/pair")
    #     accelerator.print(f"  Throughput: {1.0/mean_inference_time:.2f} pairs/second")
    
    # Print summary statistics across all runs for this specific configuration
    if len(metrics_data["run_seeds"]) > 1:
        try:
            config_metrics = metrics_data["metrics"][split_key][method_name]
            accelerator.print(f"\n=== Summary across {len(metrics_data['run_seeds'])} seed runs ===")
            accelerator.print(f"Configuration: {split_key} / {method_name}")
            for metric_name, values in config_metrics.items():
                valid_values = [v for v in values if v is not None]
                if len(valid_values) > 0:
                    mean_across_runs = np.mean(valid_values)
                    std_across_runs = np.std(valid_values)
                    accelerator.print(f"  {metric_name}: {mean_across_runs:.4f} ± {std_across_runs:.4f}")
        except KeyError:
            accelerator.print(f"\n=== Configuration {split_key}/{method_name} not yet complete ===")
    
    accelerator.print("\n" + "="*80)
    accelerator.print(f"✓ Evaluation complete for {method_name}!")
    accelerator.print(f"Results saved to: {seed_dir}")
    accelerator.print("="*80)


def main(args):
    """Main evaluation function."""
    evaluate_color_transfer(
        method_name=args.method,
        model_weights=args.model_weights,
        data_path=args.data_path,
        dataset_source=args.dataset_source,
        dataset_reference=args.dataset_reference,
        dataset_split_source=args.dataset_split_source,
        dataset_split_reference=args.dataset_split_reference,
        output_path=args.output_path,
        input_size=args.input_size,
        seed=args.seed,
        num_images_source=args.num_images_source,
        num_images_reference=args.num_images_reference,
        num_workers=args.num_workers,
        edge_methods=args.edge_methods,
        depth_methods=args.depth_methods,
        edge_models_dir=args.edge_models_dir,
        all_combinations=args.all_combinations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate reference methods for color/style transfer'
    )
    
    # Method arguments
    parser.add_argument('--method', type=str, required=True, help='Reference method to evaluate')
    parser.add_argument('--model_weights', type=str, default=None, help='Path to the color transfer model weights (if applicable)')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='./data', help='Path to datasets')
    parser.add_argument('--dataset_source', type=str, required=True, help='Source dataset name')
    parser.add_argument('--dataset_reference', type=str, required=True, help='Reference dataset name')
    parser.add_argument('--dataset_split_source', type=str, default='train', help='Source dataset split')
    parser.add_argument('--dataset_split_reference', type=str, default='val', help='Reference dataset split')
    
    # Image arguments
    parser.add_argument('--num_images_source', type=int, default=20, help='Number of source images')
    parser.add_argument('--num_images_reference', type=int, default=20, help='Number of reference images')
    parser.add_argument('--input_size', type=int, default=256, help='Image size for processing')
    parser.add_argument('--all_combinations', action='store_true', help='Evaluate all N×M combinations instead of pairwise')
    
    # Evaluation arguments
    parser.add_argument('--edge_methods', nargs='+', default=['ldc'], help='Edge detection methods for evaluation')
    parser.add_argument('--depth_methods', nargs='+', default=['depthanything_v2_large'], help='Depth estimation methods for evaluation')
    parser.add_argument('--edge_models_dir', type=str, default=None, help='Directory containing edge detection model weights')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, default='./results/reference_methods', help='Path to save results')
    parser.add_argument('--seed', type=int, default=265017005, help='Random seed for reproducibility')
    
    # Device arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    main(args)

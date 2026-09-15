import time
import argparse
import json
import statistics
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from accelerate import Accelerator

from experiments.data import create_dataset
from experiments.classifier_evaluation import load_classifier
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.reference_methods.style_transfer.artistic.adain.method import Method as AdaINMethod
from retristyle.infer_style_base import StyleIDMethod
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.tta.augmentation import augment_views
from experiments.tta.evaluation import eval_vanilla


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------

def synchronize(device):
    """
    Synchronize CUDA before taking a wall-clock measurement.

    This is necessary because CUDA operations are asynchronous.
    """
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def time_gpu_operation(fn, device):
    """
    Measure a GPU operation using CUDA events.

    Falls back to perf_counter on CPU.
    Returns elapsed time in seconds.
    """
    if torch.cuda.is_available() and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        synchronize(device)

        start_event.record()
        result = fn()
        end_event.record()

        synchronize(device)

        elapsed_ms = start_event.elapsed_time(end_event)
        return result, elapsed_ms / 1000.0

    else:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        return result, elapsed


def summarize(times):
    """
    Return mean, std, median and min/max in milliseconds.
    """
    times_ms = [t * 1000.0 for t in times]

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def benchmark_method(
    method_name: str,
    args,
    model: nn.Module,
    test_loader,
    device,
    accelerator,
):
    """
    Benchmark view generation and classifier evaluation separately.

    Timing is performed per sample. Per-view timing is derived from the
    number of generated views.

    Important:
        - augmentation timing includes augment_views() and transfer of
          the generated views to the target device;
        - classifier timing measures eval_zero() only;
        - reference database construction and retriever construction are
          deliberately excluded from per-sample latency.
    """

    torch.cuda.empty_cache()

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------------------
    # Determine number of views
    # -------------------------------------------------------------

    if method_name == "geometric":
        effective_n_views = args.n_views
        effective_n_refs = 0
    else:
        effective_n_views = args.n_refs + 1
        effective_n_refs = args.n_refs

    # -------------------------------------------------------------
    # Build style infrastructure OUTSIDE timed loop
    # -------------------------------------------------------------

    retristyle_infer = None
    retriever = None

    if method_name in ("retristyle", "adain_tta"):

        ref_db = build_reference_db(
            args.dataset,
            args.data_path,
            args.input_size,
            seed=args.seed,
            split=args.split,
        )

        retriever = build_retriever(
            strategy=args.retrieval_strategy,
            db=ref_db,
            seed=args.seed,
            metric_type="ssim",
            embedding_dir="./data/embeddings",
            dataset=args.dataset,
            device=str(device),
        )

        if method_name == "retristyle":
            retristyle_infer = StyleIDMethod()

        elif method_name == "adain_tta":
            adain_path = Path(args.method_weights) / "adain.pth"

            retristyle_infer = AdaINMethod(
                pretrained_weights=adain_path
            )

            if hasattr(retristyle_infer, "to"):
                retristyle_infer = retristyle_infer.to(device)

            elif hasattr(retristyle_infer, "model"):
                retristyle_infer.model = retristyle_infer.model.to(device)

    # -------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------

    augmentation_times = []
    evaluation_times = []
    total_sample_times = []

    # -------------------------------------------------------------
    # Benchmark samples
    # -------------------------------------------------------------

    for sample_idx, (x, _) in enumerate(test_loader):

        x = x.to(device, non_blocking=True)

        # ---------------------------------------------------------
        # Warm-up
        # ---------------------------------------------------------

        if sample_idx < args.warmup_samples:

            with torch.inference_mode():

                _ = augment_views(
                    x,
                    method_name,
                    effective_n_views,
                    start_idx=0,
                    end_idx=effective_n_refs,
                    retriever=retriever,
                    n_refs=effective_n_refs,
                    retristyle_infer=retristyle_infer,
                    native_size=args.input_size,
                    classifier_size=args.input_size,
                    input_size=args.input_size,
                    dataset=args.dataset,
                    style_batch_size=args.style_batch_size,
                )

            # Warm up classifier separately
            with torch.inference_mode():

                dummy_views = augment_views(
                    x,
                    method_name,
                    effective_n_views,
                    start_idx=0,
                    end_idx=effective_n_refs,
                    retriever=retriever,
                    n_refs=effective_n_refs,
                    retristyle_infer=retristyle_infer,
                    native_size=args.input_size,
                    classifier_size=args.input_size,
                    input_size=args.input_size,
                    dataset=args.dataset,
                    style_batch_size=args.style_batch_size,
                )

                dummy_views = dummy_views.to(device)

                _ = eval_vanilla(
                    dummy_views,
                    model,
                    normalize_fn=None,
                )

            synchronize(device)

            continue

        # ---------------------------------------------------------
        # Measure multiple repetitions
        # ---------------------------------------------------------

        sample_aug_times = []
        sample_eval_times = []
        sample_total_times = []

        for repeat_idx in range(args.repeats):

            # -----------------------------------------------------
            # Augmentation / view generation
            # -----------------------------------------------------

            synchronize(device)

            total_start = time.perf_counter()

            views, aug_time = time_gpu_operation(
                lambda: augment_views(
                    x,
                    method_name,
                    effective_n_views,
                    start_idx=0,
                    end_idx=effective_n_refs,
                    retriever=retriever,
                    n_refs=effective_n_refs,
                    retristyle_infer=retristyle_infer,
                    native_size=args.input_size,
                    classifier_size=args.input_size,
                    input_size=args.input_size,
                    dataset=args.dataset,
                    style_batch_size=args.style_batch_size,
                ),
                device,
            )

            # Make sure the generated views are physically on the
            # target device before evaluating them.
            views = views.to(
                device,
                non_blocking=True,
            )

            synchronize(device)

            # Include .to(device) in augmentation/view-generation cost.
            aug_time = time.perf_counter() - total_start

            # -----------------------------------------------------
            # Classifier evaluation
            # -----------------------------------------------------

            _, eval_time = time_gpu_operation(
                lambda: eval_vanilla(
                    views,
                    model,
                    normalize_fn=None,
                ),
                device,
            )

            # -----------------------------------------------------
            # Total
            # -----------------------------------------------------

            total_time = aug_time + eval_time

            sample_aug_times.append(aug_time)
            sample_eval_times.append(eval_time)
            sample_total_times.append(total_time)

        # ---------------------------------------------------------
        # Average repetitions for this sample
        # ---------------------------------------------------------

        augmentation_times.append(
            statistics.mean(sample_aug_times)
        )

        evaluation_times.append(
            statistics.mean(sample_eval_times)
        )

        total_sample_times.append(
            statistics.mean(sample_total_times)
        )

    # -----------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------

    aug_summary = summarize(augmentation_times)
    eval_summary = summarize(evaluation_times)
    total_summary = summarize(total_sample_times)

    # Per-view values
    aug_per_view = {
        key: value / effective_n_views
        for key, value in aug_summary.items()
    }

    eval_per_view = {
        key: value / effective_n_views
        for key, value in eval_summary.items()
    }

    total_per_view = {
        key: value / effective_n_views
        for key, value in total_summary.items()
    }

    # -----------------------------------------------------------------
    # VRAM
    # -----------------------------------------------------------------

    if torch.cuda.is_available() and device.type == "cuda":
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
    else:
        peak_vram_mb = 0.0

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    metrics = {
        "method": method_name,

        "processed_samples": len(augmentation_times),

        "views_per_sample": effective_n_views,

        "references_per_sample": effective_n_refs,

        "augmentation_latency_per_sample_ms": {
            k: round(v, 4)
            for k, v in aug_summary.items()
        },

        "augmentation_latency_per_view_ms": {
            k: round(v, 4)
            for k, v in aug_per_view.items()
        },

        "evaluation_latency_per_sample_ms": {
            k: round(v, 4)
            for k, v in eval_summary.items()
        },

        "evaluation_latency_per_view_ms": {
            k: round(v, 4)
            for k, v in eval_per_view.items()
        },

        "total_latency_per_sample_ms": {
            k: round(v, 4)
            for k, v in total_summary.items()
        },

        "total_latency_per_view_ms": {
            k: round(v, 4)
            for k, v in total_per_view.items()
        },

        "peak_vram_mb": round(peak_vram_mb, 2),
    }

    if accelerator.is_main_process:

        accelerator.print(
            f"\n--- Completed: {method_name.upper()} ---"
        )

        accelerator.print(
            json.dumps(metrics, indent=2)
        )

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Accurate TTA efficiency profiler"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test_r",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        default="resnet18",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        default="pretrained",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--n_views",
        type=int,
        default=4,
        help="Number of geometric views.",
    )

    parser.add_argument(
        "--n_refs",
        type=int,
        default=4,
        help="Number of style references.",
    )

    parser.add_argument(
        "--style_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="random",
    )

    parser.add_argument(
        "--method_weights",
        type=str,
        default="./data/models/style_transfer",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="tta_benchmark_results.json",
    )

    parser.add_argument(
        "--warmup_samples",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of samples used for warm-up.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of timing repetitions per sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    set_seed(args.seed)

    # -------------------------------------------------------------
    # Accelerator
    # -------------------------------------------------------------

    accelerator = Accelerator()

    device = accelerator.device

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(
            size=args.input_size
        ),
    ])

    full_dataset = create_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=args.split,
        transform=transform,
    )

    subset = Subset(
        full_dataset,
        range(
            min(
                args.num_samples,
                len(full_dataset)
            )
        ),
    )

    test_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------
    # Classifier
    # -------------------------------------------------------------

    model = load_classifier(
        weights_path=args.weights_path,
        classifier=args.classifier,
        num_classes=200,
        device=device,
    )

    model = accelerator.prepare(model)

    model.eval()

    # -------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------

    results = {

        "metadata": {

            "dataset": args.dataset,

            "split": args.split,

            "classifier": args.classifier,

            "num_samples": len(test_loader),

            "input_size": args.input_size,

            "retrieval_strategy": args.retrieval_strategy,

            "seed": args.seed,

            "warmup_samples": args.warmup_samples,

            "repeats_per_sample": args.repeats,

            "batch_size": 1,

            "timing": {
                "augmentation": (
                    "view generation including transfer "
                    "of generated views to device"
                ),
                "evaluation": (
                    "eval_zero over all generated views"
                ),
                "per_view": (
                    "per-sample latency divided by "
                    "number of generated views"
                ),
            },
        },

        "methods": {},
    }

    target_methods = [
        "geometric",
        "retristyle",
        "adain_tta",
    ]

    for method in target_methods:

        method_metrics = benchmark_method(
            method,
            args,
            model,
            test_loader,
            device,
            accelerator,
        )

        results["methods"][method] = method_metrics

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    if accelerator.is_main_process:

        out_path = Path(args.output_json)

        out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(out_path, "w") as f:

            json.dump(
                results,
                f,
                indent=4,
            )

        accelerator.print(
            f"\nBenchmark results saved to:"
            f" {out_path.resolve()}"
        )


if __name__ == "__main__":
    main()

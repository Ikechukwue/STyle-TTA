import time
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from accelerate import Accelerator

from experiments.data import create_dataset, TASK_TYPE, DATASET_SPLITS
from experiments.classifier_evaluation import load_classifier
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.reference_methods.style_transfer.artistic.adain.method import Method as AdaINMethod
from retristyle.infer_style_base import StyleIDMethod
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.tta.augmentation import augment_views
from experiments.tta.evaluation import eval_zero


def parse_args():
    parser = argparse.ArgumentParser(description="TTA Efficiency Profiler")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_r")
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default="pretrained")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--n_views", type=int, default=16, help="Views for geometric")
    parser.add_argument("--n_refs", type=int, default=4, help="Refs for style methods")
    parser.add_argument("--style_batch_size", type=int, default=4, help="Batch size for style transfer")
    parser.add_argument("--retrieval_strategy", type=str, default="random")
    parser.add_argument("--method_weights", type=str, default="./data/models/style_transfer")
    parser.add_argument("--output_json", type=str, default="tta_benchmark_results.json")
    return parser.parse_args()


def benchmark_method(method_name: str, args, model, test_loader, device, accelerator) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    retristyle_infer = None
    retriever = None
    effective_n_views = args.n_views if method_name == "geometric" else args.n_refs + 1
    effective_n_refs = 0 if method_name == "geometric" else args.n_refs

    if method_name in ("retristyle", "adain_tta"):
        ref_db = build_reference_db(args.dataset, args.data_path, args.input_size, seed=42, split=args.split)
        retriever = build_retriever(
            strategy=args.retrieval_strategy, db=ref_db, seed=42,
            metric_type="ssim", embedding_dir="./data/embeddings", dataset=args.dataset, device=str(device)
        )
        if method_name == "retristyle":
            retristyle_infer = StyleIDMethod()
        elif method_name == "adain_tta":
            adain_path = Path(args.method_weights) / "adain.pth"
            retristyle_infer = AdaINMethod(pretrained_weights=adain_path)
            if hasattr(retristyle_infer, "to"):
                retristyle_infer = retristyle_infer.to(device)
            elif hasattr(retristyle_infer, "model"):
                retristyle_infer.model = retristyle_infer.model.to(device)

    aug_times = []
    eval_times = []
    
    total_start = time.perf_counter()

    for x, _ in test_loader:
        x = x.to(device)

        t0 = time.perf_counter()
        views = augment_views(
            x, method_name, effective_n_views,
            start_idx=0,
            end_idx=effective_n_refs,
            retriever=retriever,
            n_refs=effective_n_refs,
            retristyle_infer=retristyle_infer,
            native_size=args.input_size,
            classifier_size=args.input_size,
            input_size=args.input_size,
            dataset=args.dataset,
            style_batch_size=args.style_batch_size
        )
        views = views.to(device)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        _ = eval_zero(views, model, normalize_fn=None)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        aug_times.append(t1 - t0)
        eval_times.append(t2 - t1)

    total_time = time.perf_counter() - total_start
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    avg_aug_per_sample = sum(aug_times) / len(aug_times)
    avg_aug_per_view = avg_aug_per_sample / effective_n_views
    avg_eval_per_sample = sum(eval_times) / len(eval_times)

    metrics = {
        "processed_samples": len(test_loader),
        "views_per_sample": effective_n_views,
        "total_runtime_s": round(total_time, 4),
        "aug_latency_per_sample_ms": round(avg_aug_per_sample * 1000, 4),
        "aug_latency_per_view_ms": round(avg_aug_per_view * 1000, 4),
        "eval_latency_per_sample_ms": round(avg_eval_per_sample * 1000, 4),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }

    if accelerator.is_main_process:
        accelerator.print(f"\n--- Completed: {method_name.upper()} ---")
        accelerator.print(json.dumps(metrics, indent=2))

    return metrics


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=args.input_size),
    ])
    full_dataset = create_dataset(
        dataset_name=args.dataset, data_path=args.data_path,
        split=args.split, transform=transform
    )
    subset = Subset(full_dataset, range(min(args.num_samples, len(full_dataset))))
    test_loader = DataLoader(subset, batch_size=1, shuffle=False)

    model = load_classifier(
        weights_path=args.weights_path, classifier=args.classifier,
        num_classes=200, device=device
    )
    model = accelerator.prepare(model)
    model.eval()

    results = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "classifier": args.classifier,
            "num_samples": len(test_loader),
            "input_size": args.input_size,
            "retrieval_strategy": args.retrieval_strategy
        },
        "methods": {}
    }

    target_methods = ["geometric", "retristyle", "adain_tta"]
    for method in target_methods:
        method_metrics = benchmark_method(method, args, model, test_loader, device, accelerator)
        results["methods"][method] = method_metrics

    if accelerator.is_main_process:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        accelerator.print(f"\nBenchmark results saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()

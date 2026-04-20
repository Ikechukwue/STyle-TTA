"""
build_retrieval_db.py — Offline Retrieval Database Generator
============================================================

Pre-computes and saves structural embeddings / metrics for the training
set to disk so that online retrieval is fast.

Reference: Section 4.1.1 of the RetriStyle-TTA report.

Usage
-----
::

    python -m retristyle.retrieval.build_retrieval_db \\
        --data_path /data \\
        --dataset pathmnist \\
        --output_path ./retrieval_db \\
        --embedding_type dino \\
        --device cuda

The script produces a ``.pt`` file containing:

- ``embeddings``: ``(N, D)`` tensor of structural descriptors.
- ``labels``: ``(N,)`` tensor of ground-truth labels.
- ``edges``: ``(N, 1, H, W)`` Sobel edge maps (for SSIM metric retriever).
- ``metadata``: dict with dataset name, embedding_type, image size, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from tqdm import tqdm


def _to_gray(images: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device, dtype=images.dtype)
    return (images * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx**2 + gy**2).sqrt()


@torch.no_grad()
def build_database(
    data_path: str,
    dataset: str,
    output_path: str,
    embedding_type: str = "dino",
    image_size: int = 224,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    """Build and save the retrieval database."""
    from experiments.data import create_dataset

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((image_size, image_size)),
    ])
    ds = create_dataset(dataset, data_path, split="train", transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_images, all_labels = [], []
    for imgs, lbls in tqdm(loader, desc="Loading training set"):
        all_images.append(imgs)
        all_labels.append(lbls)
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Training set: {images.shape[0]} images, {labels.unique().numel()} classes")

    # ------------------------------------------------------------------
    # Compute Sobel edges
    # ------------------------------------------------------------------
    print("Computing Sobel edge maps …")
    edges = _sobel_edges(_to_gray(images))

    # ------------------------------------------------------------------
    # Compute embeddings
    # ------------------------------------------------------------------
    embeddings = None
    if embedding_type == "dino":
        print("Computing DINOv2 embeddings …")
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()

        all_emb = []
        for start in tqdm(range(0, images.shape[0], batch_size), desc="DINOv2"):
            batch = images[start : start + batch_size].to(device)
            if batch.shape[-1] != 224 or batch.shape[-2] != 224:
                batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            emb = dino(batch)
            all_emb.append(emb.cpu())
        embeddings = torch.cat(all_emb, dim=0)
        del dino
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    db = {
        "embeddings": embeddings,
        "labels": labels,
        "edges": edges,
        "metadata": {
            "dataset": dataset,
            "embedding_type": embedding_type,
            "image_size": image_size,
            "n_samples": images.shape[0],
            "n_classes": labels.unique().numel(),
        },
    }
    save_path = output_dir / f"{dataset}_{embedding_type}_retrieval_db.pt"
    torch.save(db, save_path)
    print(f"Saved retrieval database → {save_path}")


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RetriStyle retrieval database")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./retrieval_db")
    parser.add_argument("--embedding_type", type=str, default="dino", choices=["dino", "edges_only"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    build_database(
        data_path=args.data_path,
        dataset=args.dataset,
        output_path=args.output_path,
        embedding_type=args.embedding_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )

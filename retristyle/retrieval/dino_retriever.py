"""
DinoRetriever — timm-based embedding retrieval with FAISS.

Uses any ``timm`` vision encoder (default: DINOv3 ViT-B/16) to compute
image embeddings, then indexes them with FAISS for sub-linear nearest-
neighbour search.

The ``timm`` integration allows trivial model swapping via the
``model_name`` parameter — every timm-compatible backbone works out of
the box because the model's own data config (resize, normalisation) is
applied automatically.

Default model
~~~~~~~~~~~~~
``vit_base_patch16_dinov3.lvd1689m`` — DINOv3 ViT-B/16 distilled on
LVD-1689M (85.6 M params, 256×256 native resolution).

Reference: Siméoni et al., *DINOv3*, arXiv 2508.10104, 2025.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseRetriever
from .reference_db import ReferenceDatabase

# Default backbone — DINOv3 base distilled on LVD-1689M
DEFAULT_EMBEDDING_MODEL = "vit_base_patch16_dinov3.lvd1689m"


class DinoRetriever(BaseRetriever):
    """Retrieve references via a timm vision-encoder + FAISS.

    Parameters
    ----------
    db : ReferenceDatabase | None
        Lazy reference-image database.  If the database already has
        ``dino_embeddings`` attached, those are reused.
    images : Tensor | None
        **(deprecated)** In-memory images ``(N, 3, H, W)`` in [0, 1].
    labels : Tensor | None
        Training-set labels ``(N,)``.
    model_name : str
        Any timm model name.  Defaults to DINOv3 ViT-B/16.
    device : str | torch.device
        Device for the embedding model.
    embeddings : Tensor | None
        If provided, skip re-computation.  Shape ``(N, D)``.
    use_faiss : bool
        Use FAISS index for fast search.
    """

    def __init__(
        self,
        db: ReferenceDatabase | None = None,
        *,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda",
        embeddings: Optional[torch.Tensor] = None,
        use_faiss: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_faiss = use_faiss
        self._model = None
        self._transform = None
        self._model_name = model_name

        if db is not None:
            self.db: Optional[ReferenceDatabase] = db
            self.images = None
            self.labels = db.labels
            self.n = len(db)
            if embeddings is not None:
                self._embeddings = embeddings
            elif db.dino_embeddings is not None:
                self._embeddings = db.dino_embeddings
            else:
                self._load_model()
                self._embeddings = self._compute_embeddings_lazy(db)
                db.dino_embeddings = self._embeddings
        elif images is not None:
            self.db = None
            self.images = images
            self.labels = labels
            self.n = images.shape[0]
            if embeddings is not None:
                self._embeddings = embeddings
            else:
                self._load_model()
                self._embeddings = self._compute_embeddings(images)
        else:
            raise ValueError("Either db or images must be provided")

        # Normalise for cosine similarity
        self._embeddings = F.normalize(self._embeddings.float(), dim=1)

        # Build FAISS index
        self._index = None
        if self.use_faiss:
            self._build_faiss_index()

    # ------------------------------------------------------------------
    # Model loading (timm)
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load a timm-compatible model and its data transforms."""
        import timm
        from timm.data import resolve_model_data_config, create_transform

        self._model = timm.create_model(
            self._model_name,
            pretrained=True,
            num_classes=0,  # remove classifier head → feature extractor
        ).to(self.device).eval()

        data_cfg = resolve_model_data_config(self._model)
        self._transform = create_transform(**data_cfg, is_training=False)

    def _preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply the timm transform to a ``(B, 3, H, W)`` [0, 1] tensor.

        The model-specific transform handles resize + normalisation.
        """
        if self._transform is None:
            self._load_model()
        # timm transforms expect (C,H,W)-tensor; apply per image
        out = torch.stack([self._transform(img) for img in batch])
        return out.to(self.device)

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_embeddings(
        self, images: torch.Tensor, batch_size: int = 64
    ) -> torch.Tensor:
        """Compute embeddings for an in-memory ``(N, 3, H, W)`` tensor."""
        if self._model is None:
            self._load_model()

        all_emb: List[torch.Tensor] = []
        for start in range(0, images.shape[0], batch_size):
            batch = self._preprocess_batch(images[start : start + batch_size])
            emb = self._model(batch)  # (B, D)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, dim=0)

    @torch.no_grad()
    def _compute_embeddings_lazy(
        self, db: ReferenceDatabase, batch_size: int = 64
    ) -> torch.Tensor:
        """Compute embeddings by loading images lazily from *db*."""
        if self._model is None:
            self._load_model()

        all_emb: List[torch.Tensor] = []
        n = len(db)
        buf: List[torch.Tensor] = []
        for i in range(n):
            buf.append(db.get_image(i))
            if len(buf) == batch_size or i == n - 1:
                batch = self._preprocess_batch(torch.stack(buf))
                emb = self._model(batch)  # (B, D)
                all_emb.append(emb.cpu())
                buf = []
        return torch.cat(all_emb, dim=0)

    # ------------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------------
    def _build_faiss_index(self) -> None:
        try:
            import faiss

            d = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(d)  # cosine on normalised vecs
            self._index.add(self._embeddings.numpy())
        except ImportError:
            self._index = None
            self.use_faiss = False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[List[int], Optional[List[float]]]:
        q_emb = self._compute_embeddings(query)  # (1, D)
        q_emb = F.normalize(q_emb.float(), dim=1)

        if self._index is not None:
            scores, indices = self._index.search(q_emb.numpy(), k)
            return indices[0].tolist(), scores[0].tolist()

        # Fallback: brute-force cosine
        sims = (q_emb @ self._embeddings.T).squeeze(0)  # (N,)
        topk = torch.topk(sims, k=min(k, sims.shape[0]))
        return topk.indices.tolist(), topk.values.tolist()

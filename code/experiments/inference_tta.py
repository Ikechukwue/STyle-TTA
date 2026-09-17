"""
Unified TTA Inference — backward-compatible thin wrapper.
==========================================================

This module delegates to :mod:`experiments.tta.run_inference` which
contains the full modular implementation.

The CLI is identical so existing bash scripts (including HPC SLURM
scripts) keep working without any changes::

    python -m experiments.inference_tta \\
        --dataset pathmnist --data_path /data \\
        --classifier densenet121 --weights_path ./checkpoints/model.pth \\
        --tta_method retristyle --eval_strategy zero \\
        --retrieval_strategy dino --n_refs 64

For direct usage of the modular package, see :mod:`experiments.tta`.
"""

from code.experiments.tta.run_inference import main  # noqa: F401


if __name__ == "__main__":
    main()

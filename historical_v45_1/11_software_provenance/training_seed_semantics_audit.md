
# Training-seed semantics audit

**Status: PASS with terminology corrections applied.**

## Source-level finding

- Both `tools/run_seed_expansion_v46.py` and `tools/run_v48_scope_confirmation.py` load the same fixed initial YOLO checkpoint before every new condition/seed run. The checkpoint hash is `7aab2bd4aebb6181df0350c80883e0c1b2615f363481649d579a23b0e4a7140f`.
- Neither runner reinitializes the model or detection head. Interrupted jobs resume their own `last.pt`, preserving model and optimizer state.
- The paired value passed to Ultralytics is the training `seed`; `deterministic=True` is also passed. Therefore, the paired factor is training stochasticity, not model-initialization stochasticity.
- The exact training ledger records Python 3.12.10, Ultralytics 8.4.50, PyTorch 2.11.0+cu128, and NVIDIA GeForce RTX 5080.

## Exact Ultralytics 8.4.50 semantics

The exact PyPI wheel was source-inspected. Trainer initialization calls `init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)`. `init_seeds` sets Python `random`, NumPy, PyTorch CPU, CUDA, and CUDA-all-device seeds. Deterministic mode requests deterministic PyTorch algorithms with `warn_only=True`, sets deterministic cuDNN, `CUBLAS_WORKSPACE_CONFIG`, and `PYTHONHASHSEED`. Data-loader worker seeding derives NumPy and Python seeds from `torch.initial_seed`; the reported YOLO robustness and scope runs used `workers=0`.

## What the seed plausibly controls

- Training-data order and sampler state.
- Stochastic augmentation choices, including mosaic scheduling/selection.
- PyTorch/CUDA random operations addressed by the library initialization.
- Any Python/NumPy randomness used after trainer seed initialization.

It does **not** denote condition-specific weight initialization because weights were loaded from the same checkpoint. Deterministic mode improves repeatability but does not guarantee bitwise identity for every hardware/kernel path because the library uses warning-only deterministic enforcement.

## Required manuscript terminology

- Use **paired training seeds**, **condition-paired training stochasticity**, or **paired differences within training seed**.
- State **identical fixed starting weights**.
- Do not use **paired initialization**, **initialization seed**, or **initialization noise** for the YOLO contrasts.
- “COCO initialization” remains valid for the distinct Faster R-CNN replication because it describes the pretrained starting model, not the pairing variable.

## Applied corrections

The manuscript abstract, contribution statement, control rationale, head-only methods, statistical methods, figure/table labels, discussion, and supplement were corrected accordingly. Numerical results and frozen decisions were unchanged.

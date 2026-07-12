import os

import torch


class Config:
    # ---- dataset paths ----
    KAGGLE_INPUT = "/kaggle/input"
    LOCAL_INPUT = "./input"
    DATASET_ROOT = KAGGLE_INPUT if os.path.exists(KAGGLE_INPUT) else LOCAL_INPUT

    # Name of the dataset layout used by LOLDataset (see dataset.py LAYOUTS dict).
    # Common values: "lol_v1", "lolv2_real", "lolv2_syn".
    DATASET_LAYOUT = os.environ.get("DATASET_LAYOUT", "lol_v1")

    # ---- output directories ----
    SAVE_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    TEST_SAMPLES_DIR = "./test_samples"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(TEST_SAMPLES_DIR, exist_ok=True)

    # ---- image / training resolution ----
    # IMG_SIZE was used in the old loader to resize the WHOLE image. Kept for backward
    # compat only. The new loader uses CROP_SIZE for random crops during training.
    IMG_SIZE = 128
    CROP_SIZE = int(os.environ.get("CROP_SIZE", 256))

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
    EPOCHS = int(os.environ.get("EPOCHS", 300))
    VAL_EVERY = int(os.environ.get("VAL_EVERY", 5))  # validate every N epochs

    # ---- optimizer ----
    LR_START = 1e-4
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    BETAS = (0.9, 0.999)
    GRAD_CLIP = 1.0

    # ---- diffusion ----
    TIMESTEPS = 100
    INFERENCE_STEPS = 20

    # ---- network ----
    CHANNELS = 32
    CHANNEL_MULT = [1, 2, 4, 8]
    RES_BLOCKS = 1

    # ---- LuminaDiff-R switches ----
    # Enable Retinex-style illumination prior as extra conditioning (change B in the plan).
    # This changes the first conv's input shape, so a prior-disabled checkpoint CANNOT load
    # a prior-enabled state dict. Set via env var USE_ILLUM_PRIOR=1 to retrain with it.
    USE_ILLUM_PRIOR = bool(int(os.environ.get("USE_ILLUM_PRIOR", "0")))

    # ---- reproducibility ----
    SEED = int(os.environ.get("SEED", 42))

    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    )

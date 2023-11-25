from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataset import CustomDataset
from src.utils import train_model

# PATHS
PROJECT_NAME = "cp_hackton"

DATASET_PATHS = "/home/jovyan/storage/nikonov/hackaton/splitted_optical_dataset"
LOAD_CHECKPOINT_FROM = None
WORK_DIR = "work_dirs"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
UNFREEZE_AT_EPOCH = 5
NUM_WORKERS = 4
NUM_EPOCHS = 40
NUM_CLASSES = 4
MAX_SIZE = 256
GPU_ID = 0
SEED = 13
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pl.seed_everything(SEED)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# CREATE DATASETS, DATALOADERS
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(always_apply=False, limit=(-45, 45)),
        A.ColorJitter(
            always_apply=False,
            p=1.0,
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.2, 0.2),
        ),
        A.AdvancedBlur(
            always_apply=False,
            p=1.0,
            blur_limit=(3, 7),
            sigmaX_limit=(0.2, 1.0),
            sigmaY_limit=(0.2, 1.0),
            rotate_limit=(-90, 90),
            beta_limit=(0.5, 8.0),
            noise_limit=(0.9, 1.1),
        ),
        A.Resize(MAX_SIZE, MAX_SIZE),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(MAX_SIZE, MAX_SIZE),
        ToTensorV2(),
    ]
)

DATASET_PATHS = Path(DATASET_PATHS)
train_dataset = CustomDataset(DATASET_PATHS / "train", train_transform)
valid_dataset = CustomDataset(DATASET_PATHS / "val", test_transform)
test_dataset = CustomDataset(DATASET_PATHS / "test", test_transform)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)

# TRAINING MODEL
facenet_model, facenet_results = train_model(
    save_name="Baseline",
    model_hparams={
        "num_classes": NUM_CLASSES,
        "max_size": MAX_SIZE,
        "unfreeze_at_epoch": UNFREEZE_AT_EPOCH,
    },
    scheduler_hparams={
        "t_max": NUM_EPOCHS,
    },
    optimizer_hparams={
        "optimizer_name": "Adam",
        "lr": LEARNING_RATE,
    },
    load_checkpoint_from=LOAD_CHECKPOINT_FROM,
    project_name=PROJECT_NAME,
    work_dir=WORK_DIR,
    device=DEVICE,
    gpu_id=[GPU_ID],
    max_epochs=NUM_EPOCHS,
    seed=SEED,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
)

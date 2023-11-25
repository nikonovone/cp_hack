import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.model import CustomModel, FeatureExtractorFreezeUnfreeze


def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        save_name (optional) - If specified, this name will be used for
        creating the checkpoint and logging directory.
    """
    
    wandb_logger = WandbLogger(
        project=kwargs["project_name"],
        save_dir=kwargs["work_dir"],
        log_model="all",
    )

    defaulr_root_dir = os.path.join(kwargs["work_dir"], save_name)
    accelerator = "gpu" if str(kwargs["device"]) == "cuda" else "cpu"

    trainer = pl.Trainer(
        default_root_dir=defaulr_root_dir,
        accelerator=accelerator,
        devices=kwargs["gpu_id"],
        # How many epochs to train for if no patience is set
        max_epochs=kwargs["max_epochs"],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="val_loss",
                save_top_k=5,
                filename="{epoch}-{val_loss:.5f}",
                save_last=True,
            ),
            LearningRateMonitor("epoch"),
            RichProgressBar(),
            FeatureExtractorFreezeUnfreeze(
                kwargs["model_hparams"]["unfreeze_at_epoch"]
            ),
        ],
        logger=wandb_logger,
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    model = CustomModel(
        model_name,
        model_hparams=kwargs["model_hparams"],
        scheduler_hparams=kwargs["scheduler_hparams"],
        optimizer_hparams=kwargs["optimizer_hparams"],
    )

    if kwargs["load_checkpoint_from"]:
        pretrained_filename = kwargs["load_checkpoint_from"]
        if os.path.isfile(pretrained_filename):
            trainer.fit(
                model,
                kwargs["train_loader"],
                kwargs["valid_loader"],
                ckpt_path=pretrained_filename,
            )
    else:
        pl.seed_everything(kwargs["seed"])  # To be reproducable
        trainer.fit(
            model,
            kwargs["train_loader"],
            kwargs["valid_loader"],
        )
        model = CustomModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on validation and test set
    valid_result = trainer.test(
        model, dataloaders=kwargs["valid_loader"], verbose=False
    )
    test_result = trainer.test(model, dataloaders=kwargs["test_loader"], verbose=False)

    result = {
        "test_acc": test_result[0]["test_acc"],
        "val_acc": valid_result[0]["val_acc"],
    }

    wandb_logger.finalize("success")

    return model, result

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import BaseFinetuning
from torch import optim
from torchmetrics.classification import Precision


class CustomModel(pl.LightningModule):
    def __init__(
        self,
        model_hparams,
        scheduler_hparams,
        optimizer_hparams,
    ):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary.
            This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        # Create model
        self.model = timm.create_model(
            "hrnet_w32.ms_in1k",
            # "efficientvit_b3.r256_in1k",
            pretrained=True,
            num_classes=model_hparams["num_classes"],
        )

        # dropout_rate = 0.5
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(
        #         2048, model_hparams["num_classes"]
        #     ),  # Замените num_classes на число классов вашей задачи
        # )
        # Create loss module
        self.max_size = model_hparams["max_size"]
        self.loss_module = nn.CrossEntropyLoss()
        self.valid_precision = Precision(
            task="multiclass", num_classes=model_hparams["num_classes"]
        )
        self.test_precision = Precision(
            task="multiclass", num_classes=model_hparams["num_classes"]
        )
        self.example_input_array = torch.zeros(
            (1, 3, self.max_size, self.max_size), dtype=torch.float32
        )

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer_name = self.hparams.optimizer_hparams["optimizer_name"]
        t_max = self.hparams.scheduler_hparams["t_max"]

        if optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.optimizer_hparams["lr"],
            )
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.optimizer_hparams["lr"],
            )
        else:
            assert False, f'Unknown optimizer: "{optimizer_name}"'

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = self.valid_precision(preds.argmax(dim=-1), labels)
        metrics = {"val_acc": acc, "val_loss": loss}

        self.valid_precision.update(preds.argmax(dim=-1), labels)
        self.log_dict(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {"test_acc": metrics["val_acc"], "test_loss": metrics["val_loss"]}
        self.log_dict(metrics)

        # imgs, labels = batch
        # preds = self.model(imgs)
        # self.test_precision.update(preds.argmax(dim=-1), labels)

        # return self.test_precision(preds.argmax(dim=-1), labels)


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        for module in list(pl_module.model.children())[:-1]:
            self.freeze(module)

        model_parameters = filter(lambda p: p.requires_grad, pl_module.parameters())
        sum([np.prod(p.size()) for p in model_parameters])

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch // 2:
            self.unfreeze_and_add_param_group(
                modules=list(pl_module.model.children())[-3:],
                optimizer=optimizer,
                train_bn=True,
            )
            model_parameters = filter(lambda p: p.requires_grad, pl_module.parameters())
            sum([np.prod(p.size()) for p in model_parameters])

        if current_epoch == self._unfreeze_at_epoch:
            for module in list(pl_module.model.children())[:-1]:
                self.unfreeze_and_add_param_group(
                    modules=module,
                    optimizer=optimizer,
                    train_bn=True,
                )

            model_parameters = filter(lambda p: p.requires_grad, pl_module.parameters())
            sum([np.prod(p.size()) for p in model_parameters])

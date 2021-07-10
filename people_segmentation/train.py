import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk
from pytorch_lightning.loggers import WandbLogger
from pytorch_toolbelt.losses import JaccardLoss, FocalLoss
#from pytorch_toolbelt.utils.catalyst.metrics import multiclass_dice_iou_score
from torch.utils.data import DataLoader

from people_segmentation.dataloaders import SegmentationDataset
from people_segmentation.utils import get_samples
from people_segmentation.metrics import multiclass_dice_iou_score

train_path = Path(os.environ["TRAIN_PATH"])
val_path = Path(os.environ["VAL_PATH"])


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class SegmentPeople(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.model = object_from_dict(self.hparams["model"])
        if "resume_from_checkpoint" in self.hparams:
            corrections: Dict[str, str] = {"model.": ""}

            state_dict = state_dict_from_disk(
                file_path=self.hparams["resume_from_checkpoint"],
                rename_in_layers=corrections,
            )
            self.model.load_state_dict(state_dict)

        self.losses = [
            ("jaccard", 0.1, JaccardLoss(mode="multiclass", from_logits=True)),
            ("focal", 0.9, FocalLoss()),
        ]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        self.train_samples = []
        for dataset_name in self.hparams["train_datasets"]:
            self.train_samples += get_samples(
                train_path / dataset_name / "images", train_path / dataset_name / "labels"
            )

        self.val_samples = []

        self.val_dataset_names = {}

        for dataset_id, dataset_name in enumerate(self.hparams["val_datasets"]):
            self.val_dataset_names[dataset_id] = dataset_name

            self.val_samples += [get_samples(val_path / dataset_name / "images", val_path / dataset_name / "labels")]

            print(f"Len val samples {dataset_name} = ", len(self.val_samples[dataset_id]))

        print("Len train samples = ", len(self.train_samples))

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])
        degrad_aug = from_dict(self.hparams["degradation_aug"])

        if "epoch_length" not in self.hparams["train_parameters"]:
            epoch_length = None
        else:
            epoch_length = self.hparams["train_parameters"]["epoch_length"]

        result = DataLoader(
            SegmentationDataset(self.train_samples, train_aug, degrad_aug, epoch_length),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        result = []

        for val_samples in self.val_samples:
            result += [
                DataLoader(
                    SegmentationDataset(val_samples, val_aug, length=None),
                    batch_size=self.hparams["val_parameters"]["batch_size"],
                    num_workers=self.hparams["num_workers"],
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                )
            ]
        return result

    def configure_optimizers(self):
        print("OPTIM:", self.hparams["optimizer"])
        optimizer = object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        print(self.hparams["scheduler"])
        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        total_loss = 0
        logs = {}
        for loss_name, weight, loss in self.losses:
            if loss_name == 'focal':
                ls_mask = loss(logits, masks.long().squeeze(dim=1))
            else:
                ls_mask = loss(logits, masks.long())
            total_loss += weight * ls_mask
            logs[f"train_mask_{loss_name}"] = ls_mask

        logs["train_loss"] = total_loss

        logs["lr"] = self._get_current_lr()

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_id, dataloader_idx=0):
        features = batch["features"]
        masks = batch["masks"]

        logits = self.forward(features)

        result = {}
        for loss_name, _, loss in self.losses:
            if loss_name == 'focal':
                result[f"val_mask_{loss_name}"] = loss(logits, masks.long().squeeze(dim=1))
            else:
                result[f"val_mask_{loss_name}"] = loss(logits, masks.long())

        dataset_type = self.val_dataset_names[dataloader_idx]

        result[f"{dataset_type}_val_iou"] = multiclass_dice_iou_score(logits, masks)

        return result

    def validation_epoch_end(self, outputs):
        logs = {"epoch": self.trainer.current_epoch}

        for output_id, output in enumerate(outputs):
           # dataset_type = self.val_dataset_names[output_id]
           # avg_val_iou = find_average(output, f"{dataset_type}_val_iou")
            avg_val_iou = output['celeba_val_iou']
            logs[f"{output_id}_val_iou"] = avg_val_iou

        return {"val_iou": avg_val_iou, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    print(hparams)
    pipeline = SegmentPeople(hparams)

    Path(hparams["checkpoint_callback"]["dirpath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=WandbLogger(hparams["experiment_name"]),
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()

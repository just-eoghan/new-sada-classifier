import os
from typing import Optional

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.datasets.frcnn_dataset import FrcnnDataset

class Collater:
    # https://shoarora.github.io/2020/02/01/collate_fn.html
    def __call__(self, batch):
        return tuple(zip(*batch))

class FrcnnDatamodule(LightningDataModule):
    """
    FrcnnDatamodule for faster rcnn object detection.

    """

    def __init__(
        self,
        train_data_dir: str = "data/",
        val_data_dir: str = "data/",
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 2,
        image_width: int = 160,
        image_height: int = 120
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = A.Compose(
            [
                A.Resize(self.hparams.image_height, self.hparams.image_width),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.notransforms = A.Compose(
            [
                A.Resize(self.hparams.image_height, self.hparams.image_width),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.hparams.image_height,  self.hparams.image_width)
        # Flir dataset (labelled) 10228
        self.data_train: Optional[Dataset] = None
        # Griffith dataset (labelled) 1062
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val:
            self.data_train = FrcnnDataset(
                self.hparams.train_data_dir + "train",
                self.hparams.train_data_dir + "train_annotations.json",
                transform=self.transforms
            )

            self.data_val = FrcnnDataset(
                self.hparams.val_data_dir + "val",
                self.hparams.val_data_dir + "val_annotations.json",
                transform=self.notransforms
            )

            return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # https://github.com/pytorch/vision/issues/2624#issuecomment-681811444
            collate_fn=self.collater,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,            
        )
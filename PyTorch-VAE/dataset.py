import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import webdataset as wds
import math
import random


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

#     def __init__(
#         self,
#         data_path: str,
#         train_batch_size: int = 8,
#         val_batch_size: int = 8,
#         patch_size: Union[int, Sequence[int]] = (256, 256),
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         **kwargs,
#     ):
#         super().__init__()

#         self.data_dir = data_path
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         self.patch_size = patch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#     def setup(self, stage: Optional[str] = None) -> None:
# #       =========================  OxfordPets Dataset  =========================
            
# #         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
# #                                               transforms.CenterCrop(self.patch_size),
# # #                                               transforms.Resize(self.patch_size),
# #                                               transforms.ToTensor(),
# #                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
# #         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
# #                                             transforms.CenterCrop(self.patch_size),
# # #                                             transforms.Resize(self.patch_size),
# #                                             transforms.ToTensor(),
# #                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# #         self.train_dataset = OxfordPets(
# #             self.data_dir,
# #             split='train',
# #             transform=train_transforms,
# #         )
        
# #         self.val_dataset = OxfordPets(
# #             self.data_dir,
# #             split='val',
# #             transform=val_transforms,
# #         )
        
# #       =========================  CelebA Dataset  =========================
    
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(148),
#                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(148),
#                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),])
        
#         self.train_dataset = MyCelebA(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#             download=False,
#         )
        
#         # Replace CelebA with your dataset
#         self.val_dataset = MyCelebA(
#             self.data_dir,
#             split='test',
#             transform=val_transforms,
#             download=False,
#         )
# #       ===============================================================

    def __init__(self, *args, **kwargs):
        super().__init__()

        
    def train_dataloader(self) -> DataLoader:
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.train_batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=True,
        #     pin_memory=self.pin_memory,
        # )
        train_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_train.tar', resampled=False, nodesplitter=wds.split_by_node)\
            .shuffle(500, initial=500, rng=random.Random(0))\
            .decode("torch")\
            .rename(images="jpg;png")\
            .to_tuple("images")\
            .batched(1024, partial=False)\
            .with_epoch(38)
        train_dl = torch.utils.data.DataLoader(train_data, 
                                num_workers=4,
                                batch_size=None, shuffle=False, persistent_workers=True)
        return train_dl

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # return DataLoader(
        #     self.val_dataset,
        #     batch_size=self.val_batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        #     pin_memory=self.pin_memory,
        # )
        val_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_val.tar', resampled=False, nodesplitter=wds.split_by_node)\
            .decode("torch")\
            .rename(images="jpg;png")\
            .to_tuple("images")\
            .batched(4096, partial=False)
        val_dl = torch.utils.data.DataLoader(val_data, num_workers=1,
                        batch_size=None, shuffle=False, persistent_workers=True)
        return val_dl

    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # return DataLoader(
        #     self.val_dataset,
        #     batch_size=144,
        #     num_workers=self.num_workers,
        #     shuffle=True,
        #     pin_memory=self.pin_memory,
        # )
        return self.val_dataloader()
     

def get_dataloaders_eeg(
    batch_size,
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    num_train=None,
    cache_dir="/tmp/wds-cache",
    seed=0,
    val_batch_size=None,
    local_rank=0,
):
    if local_rank==0: print("Getting dataloaders...")
    
    def my_split_by_node(urls):
        return urls

    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)

    train_data = wds.WebDataset(train_url, resampled=False, cache_dir=cache_dir, nodesplitter=wds.split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(seed))\
        .decode("torch")\
        .rename(images="jpg;png", eeg="eeg.npy")\
        .to_tuple("eeg", "images")\
        .batched(batch_size, partial=False, collation_fn=eeg_collation_fn)\
        .with_epoch(num_worker_batches)
    train_dl = torch.utils.data.DataLoader(train_data, 
                            num_workers=num_workers,
                            batch_size=None, shuffle=False, persistent_workers=True)

    val_data = wds.WebDataset(val_url, resampled=False, nodesplitter=wds.split_by_node)\
        .decode("torch")\
        .rename(images="jpg;png", eeg="eeg.npy")\
        .to_tuple("eeg", "images")\
        .batched(val_batch_size, partial=False, collation_fn=eeg_collation_fn)
    val_dl = torch.utils.data.DataLoader(val_data, num_workers=1,
                    batch_size=None, shuffle=False, persistent_workers=True)

    return train_dl, val_dl, None, None
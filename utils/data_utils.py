import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image
import numpy as np

from pathlib import Path

from anomalib import TaskType
from anomalib.data import MVTec

# NOTE: Provide the path to the dataset root directory.
#   If the datasets is not downloaded, it will be downloaded
#   to this directory.
dataset_root = Path.cwd().parent / "datasets" / "mvtec_anomaly_detection"




def patch_imgs(imgs, patch_size):
    """
    Split images into patches.
    
    Args:
        imgs: Tensor, shape [N, C, H, W], where N is the batch size, C is the number of channels
        patch_size: int, size of one side of the square patch
    
    Returns:
        Tensor, shape [N, L, patch_size**2 * C], where L is the number of patches per image
    """
    N, C, H, W = imgs.size()
    assert H == W and H % patch_size == 0, "Image dimensions must be square and divisible by the patch size."

    h = w = H // patch_size
    # Reshape to [N, C, h, patch_size, w, patch_size]
    x = imgs.reshape(N, C, h, patch_size, w, patch_size)
    # Rearrange the patches to bring patches to the front and flatten
    x = torch.einsum('nchpwq->nhwpqc', x)
    # Flatten the patches
    x = x.reshape(N, h * w, patch_size**2 * C)
    return x


# def image_to_patches(images, patch_size):
#     """
#     이미지를 패치로 분할하는 함수

#     Parameters:
#     images (Tensor): 입력 이미지 텐서, 크기는 (batch_size, 3, height, width)
#     patch_size (int): 한 패치의 크기 (가정: 정사각형 패치)

#     Returns:
#     Tensor: 패치 텐서, 크기는 (batch_size, num_patches, channels * patch_size * patch_size)
#     """
#     batch_size, channels, height, width = images.shape
#     patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#     patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
#     patches = patches.permute(0, 2, 1, 3, 4).contiguous()
#     patches = patches.view(batch_size, -1, channels * patch_size * patch_size)  # (batch_size, num_patches, patch_vector)
#     return patches

def mask_images(images, mask_rate):
    """
    이미지 패치를 주어진 비율로 마스킹합니다.
    
    Args:
    patches (Tensor): 입력 패치 텐서, 차원은 (batch_size, num_patches, patch_vector)
    mask_rate (float): 마스킹할 패치의 비율, 0에서 1 사이의 값

    Returns:
    Tensor: 마스킹된 패치 텐서
    """
    batch_size, num_patches, _ = images.size()
    num_masked = int(mask_rate * num_patches)

    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    
    for i in range(batch_size):
        mask_indices = torch.randperm(num_patches)[:num_masked]
        mask[i, mask_indices] = True

    images[mask] = 0
    return images


# mae encoder -> decoder의 출력값을 여기에 넣어서 다시 이미지로 만들 때 사용
def patches_to_image(patches, image_size, patch_size, channels):
    """
    재조합 함수: 패치들을 원본 이미지로 재조합
    """
    batch_size, num_patches, _ = patches.shape
    # 패치를 원본 이미지 차원으로 reshape
    patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
    # (batch_size, num_patches, channels, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    # (batch_size, channels, num_patches, patch_size, patch_size)
    grid_size = int(image_size / patch_size)
    images = patches.view(batch_size, channels, grid_size, grid_size, patch_size, patch_size)
    images = images.permute(0, 1, 2, 4, 3, 5).contiguous()
    images = images.view(batch_size, channels, image_size, image_size)
    return images


# def load_data(category, image_size, train_batch_size, num_workers, eval_batch_size=None, mode='train'):
#     mvtec_datamodule = MVTec(
#         root='datasets/MVTec',
#         category=category,
#         image_size=image_size,
#         train_batch_size=train_batch_size,
#         eval_batch_size=eval_batch_size if eval_batch_size is not None else train_batch_size,
#         num_workers=num_workers,
#         task=TaskType.CLASSIFICATION,
#     )

#     mvtec_datamodule.prepare_data()
#     mvtec_datamodule.setup()

#     # 데이터 로더 선택
#     dataloader = None
#     if mode == 'train':
#         dataloader = mvtec_datamodule.train_dataloader()
#     elif mode == 'test':
#         dataloader = mvtec_datamodule.test_dataloader()
#     return dataloader

def load_data(categories, image_size, train_batch_size, num_workers, ratio, eval_batch_size=None, mode='train'):
    all_datasets = []
    for category in categories:
        mvtec_datamodule = MVTec(
            root='datasets/MVTec',
            category=category,
            image_size=image_size,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size if eval_batch_size is not None else train_batch_size,
            num_workers=num_workers,
            task=TaskType.CLASSIFICATION,
        )
        mvtec_datamodule.prepare_data()
        mvtec_datamodule.setup()
        
        if mode == 'train':
            full_dataset = mvtec_datamodule.train_dataloader().dataset
        elif mode == 'test':
            full_dataset = mvtec_datamodule.test_dataloader().dataset
        
        subset_size = int(len(full_dataset) * ratio)
        indices = np.random.choice(len(full_dataset), subset_size, replace=False)
        subset_dataset = Subset(full_dataset, indices)
        
        all_datasets.append(subset_dataset)
    
    combined_dataset = ConcatDataset(all_datasets)
    combined_dataloader = DataLoader(combined_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    return combined_dataloader

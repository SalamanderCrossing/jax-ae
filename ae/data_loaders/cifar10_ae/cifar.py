import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import numpy as np
import jax

# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    if img.max() > 1:
        img = img / 255.0 * 2.0 - 1.0
    return img


# For visualization, we might want to map JAX or numpy tensors back to PyTorch
def jax_to_torch(imgs):
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_loaders(train_batch_size: int, test_batch_size: int, dataset_path: str):
    dataset_path = dataset_path
    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(
        root=dataset_path, train=True, transform=image_to_numpy, download=True
    )
    train_set, val_set = random_split(
        train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )

    # Loading the test set
    test_set = CIFAR10(
        root=dataset_path, train=False, transform=image_to_numpy, download=True
    )

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=numpy_collate,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate,
    )
    return train_loader, val_loader, test_loader

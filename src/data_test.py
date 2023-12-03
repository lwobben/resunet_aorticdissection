"""
Provides test data (for use in `test.py`).
Here we move through the data in a sequential manner (instead of by taking random slabs).
"""

import torch
import numpy as np
import random
import Constants


def load_dataset(path):
    """
    input - path = path to load img and mask slab from
    return - imgs[None,:,...] = image volume slab
    return - masks[None,:,...] = mask volume slab
    """
    seedvalue = random.randint(1, 100000)
    mask = load_volume(path + "mask", seedvalue)
    img = load_volume(path + "img", seedvalue)
    return img, mask


def load_volume(path, seed):
    """
    input - v_info = header info
    input - z = depth to start reading from
    return - vol.astype(dtype=float) = volume with Constants.NB_SLICES slices starting from depth z
    """
    vol = np.load(path + ".npy")  # , mmap_mode='r')
    z_size = np.shape(vol)[2]
    random.seed(seed)
    if Constants.DATASET == "AX":
        z = random.randint(
            int(z_size * 0.15), int(z_size * 0.776) - (Constants.NB_SLICES + 1)
        )  # returns random integer between two arguments
        slab = vol[:, :, z : z + Constants.NB_SLICES]
    else:
        z = random.randint(
            0, z_size - (Constants.NB_SLICES + 1)
        )  # returns random integer between two arguments
        if Constants.MPRCROP[0]:
            slab = vol[:, :, z : z + Constants.NB_SLICES]
        else:
            hw = Constants.MPRCROP[1]
            slab = vol[
                ((128 - hw) / 2) : ((128 - hw) / 2) + hw,
                ((128 - hw) / 2) : ((128 - hw) / 2) + hw,
                z : z + Constants.NB_SLICES,
            ]
    return slab


class ImageFolder:
    def __init__(self, root_path):
        """
        input - root_path = path to folder that contains a folder for each scan with the image and mask in there
        """
        self.root = root_path
        self.img = np.load(root_path + "img.npy")
        self.mask = np.load(root_path + "mask.npy")
        self.slices = np.shape(self.img)[2]  # or 2? #Total nb of slices in scan
        self.slab = Constants.NB_SLICES  # slices per slabl

    def GetItem(self, index):
        start = index * self.slab
        if start + self.slab > self.slices:
            img = self.img[..., self.slices - self.slab : self.slices]
            mask = self.img[..., self.slices - self.slab : self.slices]
        else:
            img = self.img[..., start : start + self.slab]
            mask = self.mask[..., start : start + self.slab]

        img = img[None, None, ...]
        mask = mask[None, None, ...]
        return torch.Tensor(img.copy()), torch.Tensor(mask.copy())

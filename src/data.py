# Imports the preprocessed data during training and validation.

import torch
import torch.utils.data as data
import numpy as np
import os
import random
import Constants
from augmentation import augment
import random


def load_dataset(path, slab):
    """
    input - path = path to load img and mask slab from
    return - imgs[None,:,...] = image volume slab
    return - masks[None,:,...] = mask volume slab
    """
    seedvalue = random.randint(1, 100000)
    mask = load_volume(path + "mask", seedvalue, slab)
    img = load_volume(path + "img", seedvalue, slab, isimg=True)

    return img, mask


def load_volume(path, seed, slab, isimg=False):
    """
    input - v_info = header info
    input - z = depth to start reading from
    return - vol.astype(dtype=float) = volume with Constants.NB_SLICES slices starting from depth z
    """

    if Constants.PHASE == 2:
        if isimg:
            vol = np.load(Constants.DIR_IMGS + slab + "-newimg.npy")
        else:
            vol = np.load(path + ".npy")  # , mmap_mode='r')
    else:
        vol = np.load(path + ".npy")  # , mmap_mode='r')
    z_size = np.shape(vol)[2]
    random.seed(seed)
    if Constants.DATASET.startswith("AX"):
        z = random.randint(
            int(z_size * 0.036), int(z_size * 0.776) - (Constants.NB_SLICES + 1)
        )  # returns random integer between two arguments
        slab = vol[:, :, z : z + Constants.NB_SLICES]
    else:
        z = random.randint(
            0, z_size - (Constants.NB_SLICES + 1)
        )  # returns random integer between two arguments
        slab = vol[:, :, z : z + Constants.NB_SLICES]
    return slab


class ImageFolder(data.Dataset):
    # Class derived from base class 'Dataset' from module 'data'

    def __init__(self, root_path):
        # input - root_path = path to folder that contains a folder for each scan with the image and mask in there
        self.root = root_path
        self.scanlist = os.listdir(self.root)
        self.scanlist.sort()
        if self.root.endswith("test/"):
            self.slablist = [
                i for i in self.scanlist for j in range(Constants.NB_SLABS)
            ]
        else:
            self.slablist = self.get_slablist(self.scanlist)

    def get_slablist(self, scanlist):
        """
        the number of slabs per scan differs based on the number of present scans of that patient.
        if a patient has many scans less slabs per scan will be taken, so that the network doesn't develop an excessive focus on this patient
        the list that is returned by this function contains the names of the scans to take slabs from
        """
        Dict = {}
        nb_scans = 0
        patient_id = scanlist[0][0:4]

        for scan in scanlist:
            current_patient_id = scan[0:4]
            if current_patient_id == patient_id:
                nb_scans += 1
            else:
                Dict[patient_id] = nb_scans
                patient_id = current_patient_id
                nb_scans = 1
        Dict[patient_id] = nb_scans

        complete_list = []
        for patient_id, nb_scans in Dict.items():
            nb_slabs = round((22.5 + 2.5 * nb_scans) * (Constants.NB_SLABS / 10))
            patient_scanlist = []
            for scan in scanlist:
                if scan.startswith(patient_id):
                    patient_scanlist.append(scan)
            slablist = []
            for slab in range(nb_slabs):
                slablist.append(random.choice(patient_scanlist))
            complete_list.extend(slablist)

        random.shuffle(complete_list)
        return complete_list

    def __getitem__(self, index):
        """
        input - index = index of image and mask to get (inside Torch - between 0 and __len___())
        return = image as tensor
        return = mask as tensor
        """
        slab = self.slablist[index]  # scanid of scan to take slab from
        img, mask = load_dataset(self.root + slab + "/", slab)

        # Data augmentation:
        if self.root == Constants.ROOT_TRAIN:
            if Constants.AUGM[0]:  # True if augmentation should be performed
                augm_frac = Constants.AUGM[1]
                augm = np.random.choice([True, False], p=[augm_frac, 1 - augm_frac])
                if augm:
                    [img, mask], nbnot = augment(img, mask)
                    # Uncomment code below for second augmentation:
                    # augm_frac2 = Constants.AUGM[2]
                    # augm2 = np.random.choice([True, False], p=[augm_frac2, 1-augm_frac2])
                    # if augm2: #second augmentations
                    #     [img, mask], nb = augment(img, mask, nb_not=nbnot)

        # Returning tensors:
        img = img[None, ...]
        mask = mask[None, ...]
        mask[mask > Constants.MAX_LABEL] = Constants.MAX_LABEL
        return torch.Tensor(img.copy()), torch.Tensor(mask.copy())

    def __len__(self):
        return len(self.slablist)

"""
Script for testing (providing average dice scores and standard deviations) and saving the predictions.
Here we move through the data in a sequential manner (instead of by taking random slabs).
"""

import os
from time import time
import time as pytime
import torch
import numpy as np
from framework import MyFrame
from visualizer import Visualizer
from data_test import ImageFolder
import Constants
from dice_calculation import Dices
from loss import Generalized_SoftDiceloss
from scipy.ndimage.measurements import label
import models

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = Constants.GPU  # ID of graphics card you wish to use


def check_connect(maskslice):
    labeled, num_comp = label(maskslice, np.ones((3, 3), dtype=np.int))
    if num_comp > 1:
        xysize = np.shape(labeled)[0]
        midvalue = labeled[int(xysize / 2), int(xysize / 2)]
        maskslice[labeled != midvalue] = 0
    return maskslice


def test_save():
    tic = time()  # record time
    NAME = Constants.TESTNAME  # should be name of the saved model you wish to use
    model = models.ResUNet.cuda()
    labels = Constants.MAX_LABEL
    batchsize = (
        torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    )  # number of available GPUs * 2 (2 comes from Constants.py)
    solver = MyFrame(model, Generalized_SoftDiceloss)
    show = Visualizer()
    Constants.log_vars()
    show.empty_visdom()
    show.Log(
        "******************************\n Saving & testing process started at "
        + pytime.strftime("%d/%m/%Y %H:%M:%S")
        + ".\nName = "
        + NAME
        + ".\n******************************",
        showvisdom=True,
    )

    if os.path.isfile("../weights/" + NAME + ".th"):
        solver.load_existing("../weights/" + NAME + ".th")
    else:
        raise ValueError("no model present with this name")

    for DataSet in ["train/", "test/", "val/"]:
        show.Log("Start with dataset: " + DataSet, showvisdom=True)
        PATH = Constants.DIR_DATA + Constants.DATASET + "/" + DataSet
        scanlist = os.listdir(PATH)
        scanlist.sort()

        # Initializing lists for dices:
        dices_tl = []
        dices_fl = []
        if labels == 3:
            dices_flt = []
        else:
            dices_flt = [None]

        for scan in scanlist:
            show.Log("Start with scan: " + scan)
            dataset = ImageFolder(PATH + scan + "/")

            fullmask = np.load(PATH + scan + "/" + "mask.npy")
            fullmask[fullmask > labels] = labels
            fullimg = np.load(PATH + scan + "/" + "img.npy")
            slices = np.shape(fullmask)[2]  # nb of slices in total scan
            slab = Constants.NB_SLICES  # nb of slices per slab
            indices = int(np.ceil(slices / slab))

            prediction = np.zeros_like(fullmask)

            for i in range(indices):
                img, mask = dataset.GetItem(i)
                solver.set_input(img, mask)
                pred = solver.test()
                pred = pred.cpu().detach().numpy()
                pred = np.argmax(pred, axis=1) / labels
                pred = np.expand_dims(pred, axis=1)
                if (
                    i * slab + slab > slices
                ):  # this is the case for the last slab, you cannot just take the last slices here. the z size should always be a multiple of 16.
                    slices_left = slices - i * slab
                    realpred = pred[..., slab - slices_left : slab]
                    prediction[..., i * slab : i * slab + slices_left] = realpred
                else:
                    prediction[..., i * slab : i * slab + slab] = pred

            fullmask = fullmask / labels

            newprediction = np.zeros_like(prediction)
            slices = np.shape(prediction)[2]
            for i in range(slices):
                predslice = prediction[..., i]
                predslice = check_connect(predslice)
                newprediction[..., i] = predslice

            if Constants.SAVE_PREDS:
                np.save(Constants.ROOT_PRED1 + scan, newprediction)
            show.show_vis(
                fullimg[None, None, ...],
                fullmask[None, None, ...],
                newprediction[None, None, ...],
                int(slices / 4),
                int(scan[0:2]),
                "saving",
                close=True,
                argmax=False,
            )  # show org images, fullmasks & preds on visdom
            show.show_vis(
                fullimg[None, None, ...],
                fullmask[None, None, ...],
                newprediction[None, None, ...],
                int(slices / 2),
                int(scan[0:2]),
                "saving",
                close=False,
                argmax=False,
            )  # show org images, fullmasks & preds on visdom
            show.show_vis(
                fullimg[None, None, ...],
                fullmask[None, None, ...],
                newprediction[None, None, ...],
                int(slices * (3 / 4)),
                int(scan[0:2]),
                "saving",
                close=False,
                argmax=False,
            )  # show org images, fullmasks & preds on visdom

            dicevalues = Dices(newprediction, fullmask, argmax=False)
            dices_tl.append(dicevalues[0])
            dices_fl.append(dicevalues[1])
            if labels == 3:
                dices_flt.append(dicevalues[2])

            show.Log(
                "Dices: TL = "
                + str(dices_tl[-1])
                + "; PFL = "
                + str(dices_fl[-1])
                + "; FLT = "
                + str(dices_flt[-1])
            )

        avg_dices_tl = np.nanmean(dices_tl)
        std_dices_tl = np.nanstd(dices_tl)
        show.Log("Average TL dices - " + DataSet + ": " + str(avg_dices_tl))
        show.Log("Std TL dices - " + DataSet + ": " + str(std_dices_tl))
        np.save(
            Constants.MAIN_DIR + "dices_totalscan/dices_tl_" + DataSet.replace("/", ""),
            dices_tl,
        )  # specify output!!!
        avg_dices_fl = np.nanmean(dices_fl)
        std_dices_fl = np.nanstd(dices_fl)
        show.Log("Average PFL dices - " + DataSet + ": " + str(avg_dices_fl))
        show.Log("Std PFL dices - " + DataSet + ": " + str(std_dices_fl))
        np.save(
            Constants.MAIN_DIR + "dices_totalscan/dices_fl_" + DataSet.replace("/", ""),
            dices_fl,
        )
        if labels == 3:
            avg_dices_flt = np.nanmean(dices_flt)
            std_dices_flt = np.nanstd(dices_flt)
            show.Log("Average FLT dices - " + DataSet + ": " + str(avg_dices_flt))
            show.Log("Std FLT dices - " + DataSet + ": " + str(std_dices_flt))
            np.save(
                Constants.MAIN_DIR
                + "dices_totalscan/dices_flt_"
                + DataSet.replace("/", ""),
                dices_flt,
            )

    total_time = int(time() - tic)
    show.Log(
        "Saving & testing finished after "
        + str(total_time)
        + " seconds! (= "
        + str(round(total_time / 3600, 1))
        + " hours)",
        showvisdom=True,
    )


if __name__ == "__main__":
    phase = Constants.PHASE
    assert phase == 2
    test_save()

"""
Main training script
Want to image on Visdom? Run 'python3 -m visdom.server' before running this script.
"""

import os
from time import time
import time as pytime
import torch
import numpy as np
from framework import MyFrame
from visualizer import Visualizer
from data import ImageFolder
import Constants
from validation import Validate
from dice_calculation import Dices
from loss import Generalized_SoftDiceloss
import models

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = Constants.GPU  # Contains ID of graphics card you wish to use


def Train():
    tic = time()
    NAME = Constants.NAME
    model = models.ResUNet.cuda()
    labels = Constants.MAX_LABEL
    batchsize = (
        torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    )  ##number of available GPUs * 2 (2 comes from Constants.py)
    solver = MyFrame(model, Generalized_SoftDiceloss)
    show = Visualizer()
    Constants.log_vars()
    show.empty_visdom()
    show.Log(
        "******************************\nmain.py executed at "
        + pytime.strftime("%d/%m/%Y %H:%M:%S")
        + "with loss function "
        + Constants.LOSS
        + ".\nName = "
        + NAME
        + ". GPU id(s): "
        + Constants.GPU
        + ".\n******************************",
        showvisdom=True,
    )

    # If you want to train further on weights from earlier training, keep the name the same!:
    if os.path.isfile("../weights/" + NAME + ".th"):
        if Constants.PREV_WEIGHTS:
            solver.load_existing("../weights/" + NAME + ".th")
            show.Log("Started with existing weights from previous execution.")

    no_optim = 0  # updated later when loss function is higher than best
    best_loss = (
        5000  # start with extremely high best loss so first epoch has better loss
    )
    last_epoch = Constants.TOTAL_EPOCH

    # Initializing lists:
    trainlosses = []
    vallosses = []
    traindices_tl = []
    valdices_tl = []
    traindices_fl = []
    valdices_fl = []
    if labels == 3:
        valdices_flt = []
        traindices_flt = []
    else:
        valdices_flt = [None]
        traindices_flt = [None]
    epochs = []

    for epoch in range(1, last_epoch + 1):
        dataset = ImageFolder(Constants.ROOT_TRAIN)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=Constants.NUM_WORKERS,
        )
        epochs.append(epoch)
        data_loader_iter = iter(data_loader)  # iterate through dataset
        sum_loss = 0
        traindices_tl_epoch = []
        traindices_fl_epoch = []
        traindices_flt_epoch = []

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            loss, pred = solver.optimize(epoch)
            sum_loss += loss
            dicevalues = Dices(pred, mask)
            traindices_tl_epoch.append(dicevalues[0])
            traindices_fl_epoch.append(dicevalues[1])
            if labels == 3:
                traindices_flt_epoch.append(dicevalues[2])
            if Constants.CODE_TESTING:
                show.Log(
                    "Broke epoch after one image. Reason: quick code testing; train_loss not correct!"
                )
                break

        traindices_tl.append(np.nanmean(traindices_tl_epoch))
        traindices_fl.append(np.nanmean(traindices_fl_epoch))
        if labels == 3:
            traindices_flt.append(np.nanmean(traindices_flt_epoch))
        train_loss = (sum_loss / len(data_loader_iter)).item()
        trainlosses.append(train_loss)

        val_loss, vdices = Validate(Constants.ROOT_VAL, solver, batchsize, epoch)

        vallosses.append(val_loss)
        valdices_tl.append(vdices[0])
        valdices_fl.append(vdices[1])
        if labels == 3:
            valdices_flt.append(vdices[2])

        if Constants.CODE_TESTING:
            save = False
        else:
            save = True
        show.plotdice(
            epochs, traindices_tl, traindices_fl, traindices_flt, "training", save=save
        )
        show.plotdice(
            epochs, valdices_tl, valdices_fl, valdices_flt, "validation", save=save
        )
        show.plotloss(epochs, trainlosses, vallosses, save=save)

        show.show_vis(
            img, mask, pred, 8, epoch, "training"
        )  # show org images, masks & preds on visdom
        show.Log(
            "**********\nEpoch "
            + str(epoch)
            + "! train_loss = "
            + "{:.3f}".format(train_loss)
            + "; val_loss = "
            + "{:.3f}".format(val_loss)
            + "; val_dice_tl = "
            + "{:.3f}".format(valdices_tl[-1])
            + "; val_dice_fl = "
            + "{:.3f}".format(valdices_fl[-1])
            + "; val_dice_flt = "
            + str(valdices_flt[-1])
        )

        if val_loss >= best_loss:
            no_optim += 1
        else:
            no_optim = 0
            best_loss = val_loss
            best_epoch = epoch
            if not Constants.CODE_TESTING:
                solver.save("../weights/" + NAME + ".th")
        show.Log(
            "Time = "
            + pytime.strftime("%H:%M:%S")
            + " (avg epoch time = "
            + "{:.2f}".format((int(time() - tic)) / epoch)
            + " sec.). Best val_loss so far = "
            + "{:.3f}".format(best_loss)
            + " (epoch "
            + str(best_epoch)
            + ")."
        )
        if (
            no_optim > Constants.NUM_EARLY_STOP
        ):  # stop after ... epochs without improvement
            show.Log(
                "Stop after epoch "
                + str(epoch)
                + ", reason: "
                + str(Constants.NUM_EARLY_STOP)
                + " epochs without improvement (early stop)",
                showvisdom=True,
            )
            break

    if epoch == last_epoch:
        show.Log(
            "Stopped after epoch "
            + str(epoch)
            + ", reason: max nb. of epochs reached.",
            showvisdom=True,
        )
    if not Constants.CODE_TESTING:
        show.save_diceandloss_info(
            trainlosses,
            vallosses,
            traindices_tl,
            traindices_fl,
            traindices_flt,
            valdices_tl,
            valdices_fl,
            valdices_flt,
        )
    total_time = int(time() - tic)
    show.Log(
        "Training finished after "
        + str(total_time)
        + " seconds! (= "
        + str(round(total_time / 3600, 1))
        + " hours)",
        showvisdom=True,
    )


if __name__ == "__main__":
    phase = Constants.PHASE
    assert phase == 1
    Train()

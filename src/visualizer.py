# Live imaging and plotting on Visdom (an online visualization tool) during the trainings

import visdom
import logging
import Constants
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np


class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom()
        self.font = {
            "family": "georgia",
            "color": "black",
            "weight": "normal",
            "size": 12,
        }
        self.font_title = {
            "family": "georgia",
            "color": "black",
            "weight": "normal",
            "size": 16,
        }
        self.labels = Constants.MAX_LABEL

    def empty_visdom(self):
        self.vis.close(None)

    def Log(self, string, printing=True, showvisdom=False):
        if printing:
            print(string)
        logging.basicConfig(
            filename=Constants.MAIN_DIR + "logs/" + Constants.NAME + ".log",
            level=logging.INFO,
            format="%(message)s",
        )
        logging.info(string)
        if showvisdom:
            self.vis.text(string, opts=dict(height=150))

    def show_vis(
        self,
        img,
        mask,
        pred,
        slice_hor,
        epoch,
        stage,
        close=True,
        argmax=True,
        pred_prev=False,
    ):
        # Prepare for showing:
        maxlabel = Constants.MAX_LABEL
        if type(pred) != np.ndarray:
            pred = pred.cpu().detach().numpy()  # convert to numpy array
        if argmax:
            pred = np.argmax(pred, axis=1) / maxlabel
            pred = np.expand_dims(pred, axis=1)
            mask = mask / maxlabel

        if close:
            self.vis.close("source " + stage)
            self.vis.close("mask " + stage)
            self.vis.close("pred " + stage)
            if pred_prev:
                self.vis.close("predprev " + stage)

        self.vis.image(
            img[0, 0, :, :, slice_hor],
            opts=dict(caption="source image; epoch " + str(epoch) + " - " + stage),
            win="source " + stage,
            env="exampleimgs_" + stage,
        )
        self.vis.image(
            mask[0, 0, :, :, slice_hor],
            opts=dict(caption="mask epoch; " + str(epoch) + " - " + stage),
            win="mask " + stage,
            env="exampleimgs_" + stage,
        )
        self.vis.image(
            pred[0, 0, :, :, slice_hor],
            opts=dict(caption="prediction; epoch " + str(epoch) + " - " + stage),
            win="pred " + stage,
            env="exampleimgs_" + stage,
        )
        if pred_prev:
            self.vis.image(
                img[0, 1, :, :, slice_hor],
                opts=dict(
                    caption="prediction previous phase; epoch "
                    + str(epoch)
                    + " - "
                    + stage
                ),
                win="predprev " + stage,
                env="exampleimgs_" + stage,
            )

    def plotloss(self, epochs, trainloss, valloss, save=True):
        self.vis.close("loss")
        plt.figure(4)
        plt.plot(epochs, trainloss, color="coral", label="Training loss")
        plt.plot(epochs, valloss, color="indigo", label="Validation loss")
        plt.title(
            "Loss during training and validation", fontdict=self.font_title, pad=15
        )
        plt.ylabel("Loss", fontdict=self.font)
        plt.xlabel("Epoch", fontdict=self.font)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True)

        self.vis.matplot(plt, win="loss")
        if save:
            plt.savefig(
                Constants.MAIN_DIR + "loss_and_dice/" + Constants.NAME + "_Losses_graph"
            )

    def plotdice(self, epochs, dices_tl, dices_fl, dices_flt, stage, save=True):
        self.vis.close("dice" + stage)
        if stage == "training":
            plt.figure(1)
        elif stage == "validation":
            plt.figure(2)
        else:
            plt.figure(3)
        plt.plot(epochs, dices_tl, color="m", label="Dice TL")
        if self.labels > 1:
            plt.plot(epochs, dices_fl, color="c", label="Dice FL")
        if self.labels == 3:
            plt.plot(epochs, dices_flt, color="gold", label="Dice FLT")
        plt.title("Dice coefficients during " + stage, fontdict=self.font_title, pad=15)
        plt.ylabel("Dice coefficient", fontdict=self.font)
        plt.xlabel("Epoch", fontdict=self.font)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fancybox=True, shadow=True)

        self.vis.matplot(plt, win="dice" + stage)
        if save:
            plt.savefig(
                Constants.MAIN_DIR
                + "loss_and_dice/"
                + Constants.NAME
                + "_"
                + stage
                + "_Dices_graph"
            )

    def save_diceandloss_info(
        self,
        trainlosses,
        vallosses,
        traindices_tl,
        traindices_fl,
        traindices_flt,
        valdices_tl,
        valdices_fl,
        valdices_flt,
    ):
        np.save(
            Constants.MAIN_DIR + "loss_and_dice/" + Constants.NAME + "_Train_losses",
            np.asarray(trainlosses),
        )
        np.save(
            Constants.MAIN_DIR
            + "loss_and_dice/"
            + Constants.NAME
            + "_Validation_losses",
            np.asarray(vallosses),
        )
        np.save(
            Constants.MAIN_DIR + "loss_and_dice/" + Constants.NAME + "_Train_Dices_TL",
            np.asarray(traindices_tl),
        )
        np.save(
            Constants.MAIN_DIR + "loss_and_dice/" + Constants.NAME + "_Train_Dices_FL",
            np.asarray(traindices_fl),
        )
        np.save(
            Constants.MAIN_DIR
            + "loss_and_dice/"
            + Constants.NAME
            + "_Validation_Dices_TL",
            np.asarray(valdices_tl),
        )
        np.save(
            Constants.MAIN_DIR
            + "loss_and_dice/"
            + Constants.NAME
            + "_Validation_Dices_FL",
            np.asarray(valdices_fl),
        )
        if self.labels == 3:
            np.save(
                Constants.MAIN_DIR
                + "loss_and_dice/"
                + Constants.NAME
                + "_Train_Dices_FLT",
                np.asarray(traindices_flt),
            )
            np.save(
                Constants.MAIN_DIR
                + "loss_and_dice/"
                + Constants.NAME
                + "_Validation_Dices_FLT",
                np.asarray(valdices_flt),
            )

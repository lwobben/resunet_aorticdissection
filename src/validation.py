import torch
import numpy as np
import data
import Constants
from visualizer import Visualizer
from dice_calculation import Dices


def Validate(path, frame, batchsize, epoch):
    show = Visualizer()
    dataset = data.ImageFolder(path)
    labels = Constants.MAX_LABEL
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, shuffle=True, num_workers=Constants.NUM_WORKERS
    )
    data_loader_iter = iter(data_loader)
    sum_loss = 0
    dices_tl = []
    dices_fl = []
    dices_flt = []

    for img, mask in data_loader_iter:
        frame.set_input(img, mask)
        loss, pred = frame.valid()
        sum_loss += loss

        dicevalues = Dices(pred, mask)
        dices_tl.append(dicevalues[0])
        dices_fl.append(dicevalues[1])
        if labels == 3:
            dices_flt.append(dicevalues[2])

        if Constants.CODE_TESTING:
            show.Log(
                "Only validated on one image. Reason: quick code testing; val_loss not correct!"
            )
            break
    avg_loss = sum_loss / len(data_loader_iter)

    show.show_vis(
        img, mask, pred, 2, epoch, "validation"
    )  # show org images, masks & preds on visdom

    avg_dices = []
    avg_dices.append(np.nanmean(dices_tl))
    avg_dices.append(np.nanmean(dices_fl))
    if labels == 3:
        avg_dices.append(np.nanmean(dices_flt))

    return avg_loss.item(), avg_dices

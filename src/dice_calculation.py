# Calculation of the Dice similarity coefficients between masks and predictions

import Constants
import numpy as np


def Dices(pred, mask, argmax=True):
    maxlabel = Constants.MAX_LABEL

    if type(pred) != np.ndarray:
        pred = pred.cpu().detach().numpy()  # convert to numpy array ([batch,3,x,y,z])
    if type(mask) != np.ndarray:
        mask = (
            mask.cpu().detach().numpy().squeeze()
        )  # convert to numpy array, remove axis 1 (which has 1 dimension) --> [batch,x,y,z]
    if argmax:
        pred = np.argmax(
            pred, axis=1
        )  # now at axis 1 the argmax will taken --> [batch,x,y,z]
    labels = list(
        (range(1, maxlabel + 1))
    )  # is for example [1, 2] when 2 (FL) is max label

    dicevalues = []
    for label in labels:
        prednew = pred * 0
        masknew = mask * 0
        prednew[pred == label] = 1
        masknew[mask == label] = 1
        sum_pred = np.sum(prednew)
        sum_mask = np.sum(masknew)
        intersect = np.sum(prednew * masknew)
        totalsum = sum_pred + sum_mask
        if totalsum != 0:
            dicevalues.append((2 * intersect) / totalsum)
        else:
            dicevalues.append(np.nan)
    return dicevalues

import random
import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import elasticdeform


def augment(img, mask, predprev=None, nb_not=None):
    """Main augment function in which type of augmentation is randomly decided and applied."""
    options = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]  # define which augmentation options to choose from (see `augdict` below)
    if nb_not:
        options.remove(nb_not)
    nb = np.random.choice(options)
    augdict = augdict = {
        0: Rotate,
        1: Flip,
        2: Translate,
        3: Deform,
        4: Gamma,
        5: Gaussian,
        6: Zoom,
        7: Stretch,
    }
    if predprev == None:
        return augdict[nb](img, mask), nb
    else:
        return augdict[nb](img, mask, predprev), nb


# The 8 augmentation functions, which are randomly called by `augment`:


def Rotate(img, mask, predprev=None):
    theta = random.randint(-15, 15)
    img = rotate(img, float(theta), reshape=False, order=0, mode="nearest")
    mask = rotate(mask, float(theta), reshape=False, order=0, mode="nearest")
    if predprev == None:
        return img, mask
    else:
        predprev = rotate(mask, float(theta), reshape=False, order=0, mode="nearest")
        # reshape = False: so that input shape == output shape
        # order = 0: nearest neighbour interpolation
        # mode = nearest: takes closest pixel-value at edge and replicates it
        return img, mask, predprev


def Flip(img, mask, predprev=None):
    img = np.fliplr(img)
    mask = np.fliplr(mask)
    if predprev == None:
        return img, mask
    else:
        predprev = np.fliplr(predprev)
        return img, mask, predprev


def Translate(img, mask, predprev=None):
    shape_xy = np.asarray(np.shape(img))[0:2]
    shiftmaxes = np.round(0.15 * shape_xy)
    shiftx = random.randint(-shiftmaxes[0], shiftmaxes[0])
    shifty = random.randint(-shiftmaxes[1], shiftmaxes[1])
    img = interpolation.shift(img, [shiftx, shifty, 0], order=0, mode="nearest")
    mask = interpolation.shift(mask, [shiftx, shifty, 0], order=0, mode="nearest")
    # order = 0: nearest neighbour interpolation
    # mode = nearest: takes closest pixel-value at edge and replicates it
    if predprev == None:
        return img, mask
    else:
        predprev = interpolation.shift(
            predprev, [shiftx, shifty, 0], order=0, mode="nearest"
        )
        return img, mask, predprev


def Deform(img, mask, predprev=None):
    Sigma = random.randint(4, 6)
    Points = random.randint(2, 4)
    [Img, Mask] = elasticdeform.deform_random_grid(
        [img, mask], sigma=Sigma, points=Points, order=[3, 0], axis=(0, 1)
    )
    # normalize:
    Max, Min = np.max(Img), np.min(Img)
    scale = 1 / (Max - Min)
    Img = scale * (Img - Min)
    return Img, Mask


def Gamma(img, mask, predprev=None):
    gamma = random.randint(5, 15) / 10
    # print(np.unique(img))
    Img = img**gamma
    return Img, mask, predprev


def Gaussian(img, mask, predprev=None):
    Sigma = random.randint(3, 7) / 10
    Img = gaussian_filter(img, sigma=Sigma)
    Mask = gaussian_filter(mask.astype(dtype="uint8"), sigma=Sigma)
    if predprev == None:
        return Img, Mask.astype(dtype=float)
    else:
        Pred = gaussian_filter(predprev.astype(dtype="uint8"), sigma=Sigma)
        return Img, Mask.astype(dtype=float), Pred.astype(dtype=float)


def Zoom(img, mask, predprev=None):
    factor = random.randint(85, 115) / 100
    if predprev == None:
        Img, Mask = Stretching(img, mask, factor)
        return Img, Mask
    else:
        Img, Mask, Pred = Stretching(img, mask, factor, predprev)
        return Img, Mask, Pred


def Stretch(img, mask, predprev=None):
    rand = np.random.choice([0, 1])
    if predprev == None:
        if rand == 0:
            return Stretch_in(img, mask)
        else:
            return Stretch_out(img, mask)
    else:
        if rand == 0:
            return Stretch_in(img, mask, predprev)
        else:
            return Stretch_out(img, mask, predprev)


# Subfunctions called by the Zoom and Stretch augmentation functions:


def Stretch_in(img, mask, predprev=None):
    factorx = random.randint(100, 115) / 100
    factory = random.randint(100, 115) / 100
    factorz = random.randint(100, 115) / 100
    if predprev == None:
        Img, Mask = Stretching(img, mask, [factorx, factory, factorz])
        return Img, Mask
    else:
        Img, Mask, Pred = Stretching(img, mask, [factorx, factory, factorz], predprev)
        return Img, Mask, Pred


def Stretch_out(img, mask, predprev=None):
    factorx = random.randint(85, 100) / 100
    factory = random.randint(85, 100) / 100
    factorz = random.randint(85, 100) / 100
    if predprev == None:
        Img, Mask = Stretching(img, mask, [factorx, factory, factorz])
        return Img, Mask
    else:
        Img, Mask, Pred = Stretching(img, mask, [factorx, factory, factorz], predprev)
        return Img, Mask, Pred


def Stretching(img, mask, factor, predprev=None):
    # factor = random.randint(85,115)/100
    newimg = zoom(img, factor, order=0, mode="nearest")
    newmask = zoom(mask, factor, order=0, mode="nearest")
    if predprev != None:
        newpred = zoom(predprev, factor, order=0, mode="nearest")
    newshape = np.shape(newimg)  # x,y
    goodshape = np.shape(img)  # cropx/cropy

    if (
        np.mean(factor) > 1
    ):  # in this case: input shape != output shape --> so below we crop:
        startx = newshape[0] // 2 - (goodshape[0] // 2)
        starty = newshape[1] // 2 - (goodshape[1] // 2)
        startz = newshape[2] // 2 - (goodshape[2] // 2)
        Img = newimg[
            startx : startx + goodshape[0],
            starty : starty + goodshape[1],
            startz : startz + goodshape[2],
        ]
        Mask = newmask[
            startx : startx + goodshape[0],
            starty : starty + goodshape[1],
            startz : startz + goodshape[2],
        ]
        if predprev != None:
            Pred = newpred[
                startx : startx + goodshape[0],
                starty : starty + goodshape[1],
                startz : startz + goodshape[2],
            ]

    elif np.mean(factor) < 1:
        Img = extend_bound(img, newimg, goodshape, newshape)
        Mask = extend_bound(mask, newmask, goodshape, newshape)
        if predprev != None:
            Pred = extend_bound(predprev, newpred, goodshape, newshape)

    else:  # factor == 1
        Img = newimg
        Mask = newmask
        if predprev != None:
            Pred = newpred

    if predprev != None:
        return Img, Mask, Pred
    else:
        return Img, Mask


def extend_bound(ar_org, ar_new, goodshape, newshape):
    xedge = (goodshape[0] - newshape[0]) // 2
    yedge = (goodshape[1] - newshape[1]) // 2
    zedge = (goodshape[2] - newshape[2]) // 2
    ar = np.zeros_like(ar_org)

    ar[
        xedge : xedge + newshape[0],
        yedge : yedge + newshape[1],
        zedge : zedge + newshape[2],
    ] = ar_new

    ar[:xedge, yedge : yedge + newshape[1], zedge : zedge + newshape[2]] = ar_new[
        0, :, :
    ][None, :, :]
    ar[
        xedge + newshape[0] :, yedge : yedge + newshape[1], zedge : zedge + newshape[2]
    ] = ar_new[-1, :, :][None, :, :]
    ar[xedge : xedge + newshape[0], :yedge, zedge : zedge + newshape[2]] = ar_new[
        :, 0, :
    ][:, None, :]
    ar[
        xedge : xedge + newshape[0], yedge + newshape[1] :, zedge : zedge + newshape[2]
    ] = ar_new[:, -1, :][:, None, :]

    ar[:xedge, :yedge, zedge : zedge + newshape[2]] = ar_new[0, 0, :][None, None, :]
    ar[xedge + newshape[0] :, :yedge, zedge : zedge + newshape[2]] = ar_new[-1, 0, :][
        None, None, :
    ]
    ar[:xedge, yedge + newshape[1] :, zedge : zedge + newshape[2]] = ar_new[0, -1, :][
        None, None, :
    ]
    ar[
        xedge + newshape[0] :, yedge + newshape[1] :, zedge : zedge + newshape[2]
    ] = ar_new[-1, -1, :][None, None, :]

    ar[:, :, :zedge] = ar[:, :, zedge][:, :, None]
    ar[:, :, zedge + newshape[2] :] = ar[:, :, zedge + newshape[2] - 1][:, :, None]
    return ar

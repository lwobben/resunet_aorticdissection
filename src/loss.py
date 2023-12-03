# Loss functions

import torch
import torch.nn as nn
import Constants
import torch.nn.functional as F


class Generalized_SoftDiceloss(nn.Module):
    """
    This class calculates the generalized soft diceloss, a weighted version of the dice loss,
    where weights are 1/vol**2 (vol being the sum of voxels with a specific label in the mask)
    Pred input: number of channels = number of classes
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        num_classes = Constants.MAX_LABEL + 1
        mask = F.one_hot(mask.long(), num_classes).squeeze().float()
        weighted_intersect = 0
        weighted_totalsum = 0

        for label in range(num_classes):
            mask_label = mask[..., label]
            pred_label = pred[:, label, ...]
            sum_mask = torch.sum(mask_label)
            sum_pred = torch.sum(pred_label)
            totalsum = sum_mask + sum_pred

            if totalsum != 0:
                intersect = torch.sum(pred_label * mask_label)
                weight = 1 / (1 + sum_mask**2)
                weighted_intersect += weight * intersect
                weighted_totalsum += weight * totalsum

        loss = 1 - 2 * (weighted_intersect / weighted_totalsum)
        return loss


class CE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = torch.tensor([1, 95.4, 69.7]).cuda()
        self.ce_loss = nn.CrossEntropyLoss()  # weight=self.weights

    def forward(self, pred, mask):
        mask = mask.squeeze().long()
        if torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD == 1:
            mask = mask[None, ...]
        return self.ce_loss(pred, mask)

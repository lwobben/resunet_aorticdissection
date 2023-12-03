# Framework that defines our main operations, called in main and validation

import torch
import torch.nn as nn
import numpy as np
import Constants


class MyFrame:
    def __init__(self, net, loss):
        torch.manual_seed(1)
        self.net = net
        if len(Constants.GPU) > 1:
            self.net = nn.DataParallel(
                self.net, device_ids=range(torch.cuda.device_count())
            )  # makes it possible to use more than 1 GPU
        self.params = self.net.parameters()
        self.init_lr = Constants.LR_RESUNET
        self.optimizer = torch.optim.Adam(
            params=self.params, lr=self.init_lr, weight_decay=Constants.L2
        )
        self.loss = loss()
        self.total_ep = Constants.TOTAL_EPOCH

    def set_input(self, img_batch, mask_batch):
        self.img = img_batch
        self.mask = mask_batch

    def forward(self, volatile=False):
        self.img = self.img.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()

    def optimize(self, epoch):
        self.net.train()
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net(self.img)
        loss = self.loss(pred, self.mask)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), pred

    def valid(self):
        self.net.eval()
        self.forward()
        pred = self.net(self.img)
        loss = self.loss(pred, self.mask)
        return loss.detach(), pred

    def test(self):
        self.net.eval()
        self.forward()
        pred = self.net(self.img)
        return pred

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load_existing(self, path):
        self.net.load_state_dict(torch.load(path))

    def var_lr(self, epoch):
        lr = self.init_lr * (np.cos(np.pi * (epoch - 1) / self.total_ep) + 1) / 2
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

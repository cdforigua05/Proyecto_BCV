#! /usr/bin/env python

# import libraries
import os
import time
import math
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
from Metricas.py import F_score_PR
import torch.nn.functional as F
from torch.autograd import Variable

# utility class for training VGG_16 model
class Trainer(object):
    # init function for class
    def __init__(self, model, optimizer, trainDataloader, valDataloader, loss_function,\
                 nBatch=10, maxEpochs=15, cuda=True, lrDecayEpochs={} ,gamma=2):
        self.cuda = cuda
        # define an optimizer
        self.optim = optimizer
        # set the network
        self.model = model
        self.loss_func = loss_function
        if self.cuda:
            self.loss_func = self.loss_func.cuda()
        # set the data loaders
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader

        # set training parameters
        self.epoch = 0
        self.nBatch = nBatch
        self.maxepochs = maxEpochs

        self.lrDecayEpocs = lrDecayEpochs
        self.gamma = gamma
        self.valInterval = 500
        self.dispInterval = 100
        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        # function to train network
        for epoch in range(self.epoch, self.maxepochs):
            # set function to training mode
            self.model.train()

            # initialize gradients
            self.optim.zero_grad()

            if epoch in self.lrDecayEpochs:
                self.adjustLR()

            # train the network
            losses = []
            lossAcc = 0.0
            for i, sample in enumerate(self.trainDataloader, 0):
                # get the training batch
                data, target = sample
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # model forward
                output = self.model(data)

                # compute loss for batch
                loss = self.loss_func(output,target)

                if np.isnan(float(loss.data.item())):
                    raise ValueError('loss is nan while training')
                    break

                losses.append(loss)
                lossAcc += loss.item()

                # perform backpropogation and update network
                if i % self.nBatch == 0:
                    bLoss = sum(losses)
                    bLoss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    losses = []

                # visualize the loss
                if (i + 1) % self.dispInterval == 0:
                    timestr = time.strftime(self.timeformat, time.localtime())
                    print("%s epoch: %d iter:%d loss:%.6f"%(
                        timestr, epoch+1, i+1, lossAcc/self.dispInterval))
                    lossAcc = 0.0

                # perform validation every 500 iters
                if (i+1) % self.valInterval == 0:
                    self.val(epoch + 1)

    def val(self, epoch):
        # eval model on validation set
        print('Evaluation:')
        # convert to test mode
        self.model.eval()
        # perform test inference
        for i, sample in enumerate(self.valDataloader, 0):
            # get the test sample
            data, target = sample
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # perform forward computation
            preds = self.model.(data)
            F_score_PR(target, preds)

        print('evaluation done')
        self.model.train()

    def adjustLR(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] *= self.gamma
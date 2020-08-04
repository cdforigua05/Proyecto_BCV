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
from Metricas import F_score_PR, F_Medida
import torch.nn.functional as F
from torch.autograd import Variable

# utility class for training VGG_16 model
class Trainer(object):
    # init function for class
    def __init__(self, model, optimizer, trainDataloader, valDataloader, loss_function, maxEpochs=15, gamma=2, cuda=True):
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
        self.maxepochs = maxEpochs
        self.gamma = gamma
        self.dispInterval = 10
        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self,epoch):
        # function to train network
        # set function to training mode
        self.model.train()

        for index, sample in enumerate(self.trainDataloader):
            # get the training batch
            data, target = sample
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            # model forward
            output = self.model(data)
            # compute loss for batch
            loss = self.loss_func(output,target.long())
            loss.backward()
            self.optim.step()

            if np.isnan(float(loss.data.item())):
                raise ValueError('loss is nan while training')

            # visualize the loss
            if index % self.dispInterval == 0:
                timestr = time.strftime(self.timeformat, time.localtime())
                print("{} epoch: {} per:{}/{} ({:.0f}%) loss:{:.6f}".format(timestr,epoch,(index+1)*len(data),\
                    len(self.trainDataloader.dataset),100.*(index+1)/len(self.trainDataloader),loss.data.item()))

    def val(self, epoch):
        # eval model on validation set
        print('Evaluation:')
        # convert to test mode
        self.model.eval()
        test_loss = 0
        target_total = torch.tensor([],dtype=torch.int32).cuda()
        pred_total = torch.tensor([],dtype=torch.float32).cuda()
        # perform test inference
        for index, sample in enumerate(self.valDataloader):
            data, target = sample
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # perform forward computation
            preds = self.model(data)
            test_loss += self.loss_func(preds,target.long()).data
            pred_total = torch.cat((pred_total,preds.data),0)
            target_total = torch.cat((target_total,target.data))

        x = F_score_PR(target_total, pred_total)

        print('evaluation done')
        return test_loss, x

    def adjustLR(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] *= self.gamma
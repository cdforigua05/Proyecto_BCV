#! /usr/bin/env python

import time
import argparse
import numpy as np
import os.path as osp
import scipy.io as scio

import torch
import torch.nn as nn
# import Dataset.py import MelanomaDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='VGG training file')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 2)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model_vgg16.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = transforms.Compose([transforms.ToPILImage(),\
    transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize(mean, std)])

train_loader = torch.utils.data.DataLoader(MelanomaDataset('./data', distribution=0,\
    #transform=transforms),batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(MelanomaDataset('./data', distribution=1,\
    #transform=transforms),batch_size=args.batch_size, shuffle=True, **kwargs)

vgg_16 = models.vgg16()
vgg_16.classifier[6] = nn.Linear(in_features=4096, out_features=9,bias=True)

if args.cuda:
    vgg_16.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        vgg_16.load_state_dict(state)
        load_model = True

optimizer = optim.SGD(vgg_16.parameters(), lr=args.lr, momentum=args.momentum)
loss_function = torch.nn.CrossEntropyLoss(weight=None)



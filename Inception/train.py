#! /usr/bin/env python
from Inceptionv4 import inceptionv4
import time
import argparse
import numpy as np
import os.path as osp
import scipy.io as scio
import pretrainedmodels
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from trainer import Trainer
from torch.autograd import Variable
from Dataset import MelanomaDataset
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

num_classes = 8
input_size = pretrainedmodels.pretrained_settings['inceptionv4']['imagenet']['input_size']
mean = pretrainedmodels.pretrained_settings['inceptionv4']['imagenet']['mean']
std = pretrainedmodels.pretrained_settings['inceptionv4']['imagenet']['std']

transforms = transforms.Compose([transforms.ToPILImage(),\
    transforms.Resize((input_size[1],input_size[1])),transforms.ToTensor(), transforms.Normalize(mean, std)])


train_loader = torch.utils.data.DataLoader(MelanomaDataset('../data', distribution=0,\
    transform=transforms),batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(MelanomaDataset('../data', distribution=1,\
    transform=transforms),batch_size=args.batch_size, shuffle=True, **kwargs)

model = inceptionv4(pretrained='imagenet')
model.last_linear = nn.Linear(1536, num_classes)

if args.cuda:
    model.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        vgg_16.load_state_dict(state)
        load_model = True

for param in model.parameters(): 
    param.requires_grad

optimizer = optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

weights = [17606/5086,17606/17606,17606/3477,17606/1077,17606/3338,17606/313,17606/350,17606/564]
class_weights = torch.FloatTensor(weights).cuda()
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

lrDecayEpoch = {3, 5, 8, 10, 12}

trainer = Trainer(model, optimizer, train_loader, val_loader,loss_function, nBatch = args.batch_size,\
    maxEpochs=args.epochs, cuda = args.cuda, lrDecayEpochs = lrDecayEpoch, gamma = args.gamma)

trainer.train()
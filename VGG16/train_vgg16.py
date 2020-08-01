#! /usr/bin/env python

import time
import argparse
import numpy as np
import os.path as osp
import scipy.io as scio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from trainer import Trainer
from torch.autograd import Variable
from Dataset import MelanomaDataset
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='VGG training file')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
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
parser.add_argument('--save', type=str, default='model_vgg16_pretrained_dataaug.pt',
                    help='file on which to save model weights')
parser.add_argument('--input_size',type=int, default=224, metavar='N',
                    help='Size needed for VGG')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

data_path = '../melanoma_dataset'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transf = transforms.Compose([transforms.ToPILImage(),transforms.Resize((args.input_size,args.input_size)),\
transforms.ToTensor(), transforms.Normalize(mean, std)])

transf_data_aug = transforms.Compose([transforms.ToPILImage(),transforms.RandomResizedCrop((args.input_size,args.input_size)),\
transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean, std)])
#breakpoint()

train_loader = torch.utils.data.DataLoader(MelanomaDataset(data_path, distribution=0,transform=transf_data_aug),\
    batch_size=args.batch_size, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(MelanomaDataset(data_path, distribution=1,transform=transf_data_aug),\
    batch_size=args.batch_size, shuffle=True, **kwargs)

vgg_16 = models.vgg16(pretrained=True)
vgg_16.classifier[6] = nn.Linear(in_features=4096, out_features=8,bias=True)

if args.cuda:
    vgg_16.cuda()

load_model = False
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        vgg_16.load_state_dict(state)
        load_model = True

optimizer = optim.SGD(vgg_16.parameters(), lr=args.lr, momentum=args.momentum)

weights = [17606/5086,17606/17606,17606/3477,17606/1077,17606/3338,17606/313,1760/6,17606/564]
class_weights = torch.FloatTensor(weights).cuda()
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

trainer = Trainer(vgg_16, optimizer, train_loader, val_loader,loss_function,maxEpochs=args.epochs,gamma = args.gamma,cuda = args.cuda)

if __name__ == '__main__':
    if torch.cuda.current_device() == 0:
        print('Currently running on GPU 1')
    elif torch.cuda.current_device() == 1:
        print('Currently running on GPU 2')
    else:
        print('Currently running on GPU 0')
    best_loss = None
    mean_fscore=0
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            trainer.train(epoch)
            validation = trainer.val(epoch)
            test_loss = validation[0]
            mean_fscore += validation[1]

            print('-' * 90)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, time.time() - epoch_start_time))
            print('-' * 90)

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                with open(args.save, 'wb') as fp:
                    state = vgg_16.state_dict()
                    torch.save(state, fp)
            else:
                trainer.adjustLR()

            if epoch == 15:
                print('-' * 90)
                print('| end of epoch {:3d} | time: {:5.2f}s '.format(epoch, time.time() - epoch_start_time))
                print('| Total mean F_score {:4d}'.format(mean_fscore))
                print('-' * 90)


    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')
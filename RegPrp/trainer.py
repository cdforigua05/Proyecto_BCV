from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import Dataset_prop as dset
from MLP import CrisAyoNet
from torch.utils.data import DataLoader

#Check if cuda is avaiable
cuda = torch.cuda.is_available()

# Establish manual seed
seed = 1
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Establish cuda kwargs
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}


def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name):
    """
    @param model: Model architecture
    @param dataloaders: Dictionary containing the training and validation datasets
    @param criterion: Loss function
    @param optimizer: Optimizer function
    @param num_epochs: Number of epochs used for training
    """
    # We start the timer
    since = time.time()
    # We define a list to store the validation accuracy history
    val_acc_history = []
    # We get the weights from the model
    best_model_wts = copy.deepcopy(model.state_dict())
    # We define a best accuracy for validation
    best_acc = 0.0
    # We create a progress bar
    with tqdm(total=num_epochs) as pbar:
        # We start the epoch iteration
        for epoch in range(num_epochs):
            pbar.set_description('Epoch {}/{}. Best acc: {}'.format(epoch, num_epochs - 1, best_acc))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Define a loss and correct variable
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                with tqdm(total=len(dataloaders[phase])) as pbar2:
                    for inputs, labels in dataloaders[phase]:
                        labels = labels.long()
                        if cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # Optimization and back propagation
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        pbar2.update(1)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Append the validation accuracy to the history
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
            pbar.update(1)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open(model_name, 'wb') as fp:
        print('Best model saved!')
        state = model.state_dict()
        torch.save(state, fp)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

num_classes = 8
bs = 16
num_epochs = 15
model = CrisAyoNet()
dataloader_train = DataLoader(dset.PropsDataset())
dataloader_val = DataLoader(dset.PropsDataset())
dataloaders = {'train':dataloader_train, 'val':dataloader_val}


# Send the model to Cuda
if cuda:
    model.cuda()


params_to_update = model.parameters()
print("Params to learn:")
for name,param in model.named_parameters():
    if param.requires_grad == True:
        print("\t",name)

# Define the SGD Optimizer
learning_rate = 0.001
momentum = 0.9
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

# Setup the loss function
weights = [17606/5086,17606/17606,17606/3477,17606/1077,17606/3338,17606/313,17606/350,17606/564]
class_weights = torch.FloatTensor(weights).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

model_ft, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, model_name='TestRegPrp.pt')
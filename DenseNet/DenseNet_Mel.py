"""
DenseNet Dataset adaptation for melanoma classification
This code is based on PyTorch's guide to fine tune torchvision models
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
# First we import the libraries
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
import Dataset

#Check if cuda is avaiable
cuda = torch.cuda.is_available()

# Establish manual seed
seed = 1
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# Establish cuda kwargs
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# Now we define the working parameters
num_classes = 8
bs = 16
num_epochs = 15

# We define a train method to do the fine-tuning

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

# We create a function to initialize the model
def initialize_model(num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Densenet
    """
    model_ft = models.densenet121(pretrained=use_pretrained)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224


    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(num_classes, use_pretrained=False)

# Data Transformations

# Normalization according to DenseNet. Retrieved from: https://github.com/andreasveit/densenet-pytorch/blob/master/train.py
augment = True
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
if augment:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
            ])
transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
            ])

# Create the dictionary with the transformations


print("Initializing Datasets and Dataloaders...")

# Import the train dataset
train_dataset = Dataset.MelanomaDataset('data',0,transform=transform_train)
# Import the val dataset
val_dataset = Dataset.MelanomaDataset('data',1,transform=transform_val)
# Create the train data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, **kwargs)
# Create the test data loader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, **kwargs)

dataloaders_dict = {'train':train_loader, 'val':val_loader}

# Send the model to Cuda
if cuda:
    model_ft.cuda()


params_to_update = model_ft.parameters()
print("Params to learn:")
for name,param in model_ft.named_parameters():
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

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, model_name='ScratchAug01.pt')


#Pretrained + Finetuned no data aug: Training complete in 176m 7s Best val ACA: 0.621958
#Pretrained + Finetuned data aug: Training complete in 187m F_score: 0.3971391960357016
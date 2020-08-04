# Code for mAP testing
from Metricas import F_score_PR as F
from torchvision import datasets, models, transforms
import Dataset
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
#Check if cuda is avaiable
cuda = torch.cuda.is_available()
# Define the working parameters
bs = 100
input_size = 244
num_classes = 8
model_name = 'ScratchAug01.pt'
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

# Define the test transformation function
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
            ])
# Import the val dataset
test_dataset = Dataset.MelanomaDataset('data',1,transform=transform_test)
# Create the train data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, **kwargs)

# Define the model initialization function
def initialize_model(num_classes, use_pretrained=False):
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
# Define the testing function
def test(model_name, dataset):
    # Initialize the model for this run
    model, _ = initialize_model(num_classes, use_pretrained=False)
    # Load the model given the name 
    with open(model_name, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        print('{} Succesfully loaded!'.format(model_name))
    # Define a variable to calculate the average f_score and average mAP
    F_Score = []
    mAP = []
    # Start the testing
    with tqdm(total=len(dataset)) as pbar:
        for inputs, labels in dataset:
            outputs = model(inputs)
            f_scores, mAPs = F(labels, outputs)
            F_Score.append(f_scores)
            mAP.append(mAPs)
            pbar.update(1)
    F_score = np.mean(F_Score)
    mAP = np.mean(mAP)
    print('Total average F_score: {} | Total average mAP: {}'.format(F_score, mAP))
    return (F_score, mAP)

# Run the file
if __name__ == '__main__':
    test(model_name, test_loader)

import torch.nn as nn
import torch
import torch.optim as optim
import argparse
import resnet as RN
import trainer as tr
import Dataset as data
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=60, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

ID = 0
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)
model = model.cuda()

trans = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(),])

dta_train = data.MelanomaDataset('Train_Val_Test',0,trans)
dta_val = data.MelanomaDataset('Train_Val_Test',1,trans)
dta_test = data.MelanomaDataset('Train_Val_Test',2,trans)
dtload_train = DataLoader(dta_train)
dtload_val = DataLoader(dta_val)
dtload_test = DataLoader(dta_test)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
weights = [17606/5086,17606/17606,17606/3477,17606/1077,17606/3338,17606/313,17606/350,17606/564]
class_weights = torch.FloatTensor(weights).cuda()
loss_fn = nn.CrossEntropyLoss(weight=class_weights).cuda()
#loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

trainer = tr.Trainer(model,optimizer,dtload_train,dtload_val,dtload_test,loss_fn,ID,cuda=args.cuda)

trainer.train()
print('FINISH TRAIN ---------')
#trainer.val()
#print('FINISH VAL --------')
#trainer.test

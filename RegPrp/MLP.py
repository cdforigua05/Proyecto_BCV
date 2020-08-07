import torch.nn as nn
import torch.nn.functional as F

class CrisAyoNet(nn.Module):
    def __init__(self):
        super(CrisAyoNet, self).__init__()
        self.fc1 = nn.Linear(7, 512, bias = True)
        self.fc2 = nn.Linear(512,1024, bias = True)
        self.classifier = nn.Linear(1024, 8, bias = True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x
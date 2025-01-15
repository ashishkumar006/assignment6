import torch
import torch.nn as nn
from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def check_batchnorm(model):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            return True
    return False

def check_dropout(model):
    for layer in model.modules():
        if isinstance(layer, nn.Dropout) or isinstance(layer, nn.Dropout2d):
            return True
    return False

def check_fully_connected_or_gap(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.AdaptiveAvgPool2d):
            return True
    return False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example model (replace this with loading your model)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(20, 10, 3, padding=1)
        self.conv4 = nn.Conv2d(10, 10, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(10, 10, 3)
        self.conv6 = nn.Conv2d(10, 10, 3)
        self.conv7 = nn.Conv2d(10, 120, 3)
        self.fc = nn.Linear(120, 10)

        #Dropout layers
        self.dropout= nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = x.view(x.size(0), -1) #flatten tensor
        x = self.fc(x)
        x = self.dropout(x)

        return F.log_softmax(x,dim=1)

# Load your model here (ExampleModel is just for illustration)
model = ExampleModel()

batchnorm_used = check_batchnorm(model)
dropout_used = check_dropout(model)
fully_connected_or_gap_used = check_fully_connected_or_gap(model)
param_count = count_parameters(model)

# Check conditions
if batchnorm_used:
    print("BatchNormalization is used.")
else:
    print("BatchNormalization is NOT used.")

if dropout_used:
    print("Dropout is used.")
else:
    print("Dropout is NOT used.")

if fully_connected_or_gap_used:
    print("Fully connected layer or GAP is used.")
else:
    print("Fully connected layer or GAP is NOT used.")

if param_count < 20000:
    print(f"Total parameters are {param_count}, which is less than 20k.")
else:
    print(f"Total parameters are {param_count}, which exceeds 20k.")

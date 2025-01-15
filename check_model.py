import torch
import torch.nn as nn

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
class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc1 = nn.Linear(32*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 32*28*28)
        x = self.fc1(x)
        return x

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

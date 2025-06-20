import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import random


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=ToTensor())

print(training_data)
print(test_data)
num_train = len(training_data)
indices = list(range(num_train))
random.shuffle(indices)
print(num_train)
#784 dimensions (1 dimension par pixel)

print(torch.cuda.is_available())

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 2),
    nn.Sigmoid()
)

x = torch.tensor([1.0,2.0,3.0])
y = model(x)
print(y)
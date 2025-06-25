import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from AI import SimpleNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Utilisation du :", device)

writer = SummaryWriter()


# 1. Transformation : convertit l’image en tenseur, puis la met à plat (784 valeurs)
transform = transforms.Compose([
    #transforms.RandomAffine(
    #    degrees=15,            # Rotation entre -15 et +15 degrés
    #    translate=(0.1, 0.1),  # Translation jusqu'à 10% dans chaque direction
    #    scale=(0.9, 1.1),      # Zoom entre 90% et 110%
    #    shear=10               # Shear jusqu'à ±10 degrés
    #),
    transforms.RandomRotation(10),
    transforms.ToTensor(),            # Convertit en tenseur [0, 1]
    #transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))  # Aplatit l’image en vecteur de 784
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# 2. Télécharger les données d'entraînement
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=12800, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
manual_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)

model = SimpleNet()
model = SimpleNet().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


list = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

list_number = [0 for i in range(10)]
for image,labels in train_loader:
    for label in labels:
        
        list_number[label.item()] += 1

print(list_number)



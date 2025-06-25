from torch import nn as nn
import torch.nn.functional as F
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 2500)   # couche cachée : 784 entrées → 128 neurones
        self.hidden = nn.Linear(2500,2000)
        self.hidden2 = nn.Linear(2000,1500)
        self.hidden3 = nn.Linear(1500,1000)
        self.hidden4 = nn.Linear(1000,500)
        self.output = nn.Linear(500, 10)    # sortie : 128 → 10 (pour les chiffres 0 à 9)

    def forward(self, x): #2500, 2000, 1500, 1000, 500, 10
        x = self.hidden1(x)          # passe par la couche cachée
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)          # passe par la couche cachée
        x = F.relu(x)               # activation ReLU
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        x = self.output(x)          # passe par la couche de sortie
        return x
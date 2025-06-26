import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train_model import SimpleNet, evaluate


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = SimpleNet()
    model = model.to(device)
    model.load_state_dict(torch.load("models/Best_model.pth", map_location=device))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    test_dataset = datasets.MNIST("./data",train=False,download=True,transform=transform_test)
    test_loader = DataLoader(test_dataset,6400,shuffle=True)


    avg_loss, accuracy, precision, recall, f1 = evaluate(model, test_loader, criterion, device)

    print(f"loss : {avg_loss}, accuracy: {accuracy * 100}%")


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AI import SimpleNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNet()
model = model.to(device)
model.load_state_dict(torch.load("Best_model.pth"))
transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
test_dataset = datasets.MNIST("./data",train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,6400,shuffle=True)

def evaluate(model, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total = 0

    model.eval()  # mode évaluation (désactive dropout, etc.)

    error_list = [0 for i in range(10)]


    with torch.no_grad():  # pas de calcul de gradients pendant le test
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()

            for i in range(len(predictions)):
                if predictions[i] != labels[i]:
                    error_list[labels[i]] += 1

            total += labels.size(0)

    accuracy = total_correct / total
    avg_loss = total_loss / len(test_loader)
    print(error_list)
    return avg_loss, accuracy

avg_loss,acc = evaluate(model,test_loader)

print(f"loss : {avg_loss}, accuracy: {acc * 100}%")
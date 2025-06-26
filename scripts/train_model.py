import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt


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


def evaluate(model, eval_loader, criterion, device):
    total_loss = 0
    total_correct = 0
    total = 0

    model.eval()  # mode évaluation (désactive dropout, etc.)

    all_labels = []
    all_predictions = []

    with torch.no_grad():  # pas de calcul de gradients pendant le test
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()

            total += labels.size(0)
            labels = labels.to("cpu")
            outputs = outputs.to("cpu")

            all_labels += labels.cpu().numpy().tolist()
            all_predictions += predictions.cpu().numpy().tolist()

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")

    avg_loss = total_loss / len(eval_loader)

    return avg_loss, accuracy, precision, recall, f1


def view_prediction(manual_loader, model):
    for image,label in manual_loader:
        output = model(image)
        prediction = torch.argmax(output,dim=1)

        print(output)
        print(prediction)

        img = image.view(28, 28)  # remettre en 2D pour l’afficher

        plt.imshow(img, cmap="gray")
        plt.title(f"Chiffre : {label.item()} ; Chiffre prédit ; {prediction.item()}")
        plt.show()


def create_datasets():
    # 1. Transformation : convertit l’image en tenseur, puis la met à plat (784 valeurs)
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),            # Convertit en tenseur [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Aplatit l’image en vecteur de 784
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # 2. Télécharger les données d'entraînement
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=12800, shuffle=True)
    eval_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=True)
    manual_loader = DataLoader(eval_dataset,batch_size=1,shuffle=True)

    return train_loader, eval_loader, manual_loader


def main():
    num_epoch = 30

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    writer = SummaryWriter()

    train_loader, eval_loader, manual_loader = create_datasets()

    for epoch in range(num_epoch):
        start_time = time.time()
        total_loss = 0

        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)  # image : [1, 784]

            prediction = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)

            Loss = criterion(output, label)
            total_loss += Loss.item()

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time

        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = evaluate(model, eval_loader, criterion, device)

        # Create scalar for Tensorboard
        writer.add_scalar("epoch/time", epoch_time, epoch)
        writer.add_scalar("train/loss", total_loss, epoch)
        writer.add_scalar("eval/loss", eval_loss, epoch)
        writer.add_scalar("eval/accuracy", eval_acc, epoch)
        writer.add_scalar("eval/Precision", eval_precision, epoch)
        writer.add_scalar("eval/Recall", eval_recall, epoch)
        writer.add_scalar("eval/F1", eval_f1, epoch)

        print(f'Epoch: {epoch}/{num_epoch}')
        print(f"Perte sur eval: {eval_loss:.4f}, Accuracy : {eval_acc*100:.2f}%")
        print(f"loss moyenne: {total_loss/num_epoch}")

    if input("Sauvegarder le modèle ?  y/n ") == "y":
        torch.save(model.state_dict(), "models/mon_model.pth")

    eval_loss, eval_acc, eval_precision, eval_recall, eval_f1  = evaluate(model, eval_loader, criterion, device)
    print(f"Perte sur test : {eval_loss:.4f}, Accuracy : {eval_acc*100:.2f}%")

    writer.flush()


if __name__ == '__main__':
    main()

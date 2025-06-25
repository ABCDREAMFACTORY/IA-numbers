import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import time

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

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
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
manual_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)

model = SimpleNet()
model = SimpleNet().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


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
    return avg_loss, accuracy,error_list

def evaluate2(model, test_loader):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total = 0

    model.eval()  # mode évaluation (désactive dropout, etc.)

    error_list = [0 for i in range(10)]

    all_labels = []
    all_predictions = []

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
            labels = labels.to("cpu")
            outputs = outputs.to("cpu")
            all_labels += labels.cpu().numpy().tolist() 
            all_predictions += predictions.cpu().numpy().tolist()
    #print(f"labels : {all_labels}\n predictions : {all_predictions}")
    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels,all_predictions,average="macro")
    recall = recall_score(all_labels,all_predictions,average="macro")
    f1 = f1_score(all_labels,all_predictions,average="macro")

    avg_loss = total_loss / len(test_loader)
    print(error_list)  
    return avg_loss, accuracy,precision,recall,f1,error_list


def train(model,train_loader,num_epoch):
    # On prend une image au hasard
    loss_evolution = []
    precision_evolution = []
    for epoch in range(num_epoch):
        start_time = time.time()
        total_loss = 0
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)  # image : [1, 784]
            prediction = torch.argmax(output, dim=1)
            probs = F.softmax(output, dim=1)

            Loss = loss(output,label)
            total_loss += Loss.item()


            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
        epoch_time = time.time()-start_time
        writer.add_scalar("epoch/time",epoch_time,epoch)
        writer.add_scalar("train/loss", total_loss, epoch)
        train_loader = DataLoader(train_dataset,batch_size=12800, shuffle=True)

        loss_evolution.append(total_loss/len(train_loader))
        print(epoch)
        test_loss, test_acc,test_precision,test_recall,test_f1,error_list = evaluate2(model, test_loader)

        #num_number = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
        #for i,number in enumerate(error_list):
        #    number_accuracy = 100 - (number/num_number[i]*100)
        #    writer.add_scalar(f"accuracy/{i}",number_accuracy, epoch)

        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        writer.add_scalar("test/Precision", test_precision, epoch)
        writer.add_scalar("test/Recall", test_recall, epoch)
        writer.add_scalar("test/F1", test_f1, epoch)

        print(f"Perte sur test : {test_loss:.4f}, Précision : {test_acc*100:.2f}%")
        precision_evolution.append(test_loss)

    print(f"loss moyenne: {total_loss/num_epoch}")
    view_loss_evolution(num_epoch,loss_evolution,precision_evolution)

    if input("Sauvegarder le modèle ?  y/n ") == "y":
        torch.save(model.state_dict(), "mon_modele.pth")

def view_prediction(manual_loader):
    for image,label in manual_loader:
        output = model(image)
        print(output)
        prediction = torch.argmax(output,dim=1)
        print(prediction)
        img = image.view(28, 28)  # remettre en 2D pour l’afficher
        plt.imshow(img, cmap="gray")
        plt.title(f"Chiffre : {label.item()} ; Chiffre prédit ; {prediction.item()}")
        plt.show()
def view_loss_evolution(epoch,loss_epoch,precision_evol):
    fig, ax1 = plt.subplots()
    print(loss_epoch)
    ax1.plot(range(epoch),loss_epoch, "blue")
    plt.plot(range(epoch),precision_evol,"red")
    fig.set_size_inches(5,3)
    fig.set_dpi(100)
    plt.title(f"Minimum: {min(precision_evol)} a l'epoch : {precision_evol.index(min(precision_evol))}")


train(model,train_loader,30)


test_loss, test_acc,test_precision,test_recall,test_f1,error_list = evaluate2(model, test_loader)
print(f"Perte sur test : {test_loss:.4f}, Accuracy : {test_acc*100:.2f}%")

writer.flush()
plt.show()

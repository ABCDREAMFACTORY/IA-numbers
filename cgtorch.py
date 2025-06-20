import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Utilisation du :", device)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)   # couche cachée : 784 entrées → 128 neurones
        #self.hidden2 = nn.Linear(256,128)
        self.hidden3 = nn.Linear(128,64)
        self.output = nn.Linear(64, 10)    # sortie : 128 → 10 (pour les chiffres 0 à 9)

    def forward(self, x):
        x = self.hidden1(x)          # passe par la couche cachée
        x = F.relu(x)
        #x = self.hidden2(x)          # passe par la couche cachée
        #x = F.relu(x)               # activation ReLU
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.output(x)          # passe par la couche de sortie
        return x                    # pas besoin de softmax ici, on le fera plus tard
    

# 1. Transformation : convertit l’image en tenseur, puis la met à plat (784 valeurs)
transform = transforms.Compose([
    transforms.ToTensor(),            # Convertit en tenseur [0, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Aplatit l’image en vecteur de 784
])

# 2. Télécharger les données d'entraînement
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=12800, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
manual_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)

model = SimpleNet()
model = SimpleNet().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)


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




def train(model,train_loader,num_epoch):
    # On prend une image au hasard
    loss_evolution = []
    precision_evolution = []
    for epoch in range(num_epoch):
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
        
        loss_evolution.append(total_loss/len(train_loader))
        print(epoch)
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Perte sur test : {test_loss:.4f}, Précision : {test_acc*100:.2f}%")
        precision_evolution.append(test_loss)


    print(f"loss moyenne: {total_loss/num_epoch}")
    view_loss_evolution(num_epoch,loss_evolution,precision_evolution)

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
    fig.set_size_inches(epoch,3)
    fig.set_dpi(100)
    plt.title(f"Minimum: {min(precision_evol)} a l'epoch : {precision_evol.index(min(precision_evol))}")


torch.manual_seed(575)
train(model,train_loader,30)


test_loss, test_acc = evaluate(model, test_loader)
print(f"Perte sur test : {test_loss:.4f}, Accuracy : {test_acc*100:.2f}%")

plt.show()

#Tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_model import SimpleNet



def view_prediction(manual_loader, model):
    with torch.no_grad():
        for image,label in manual_loader:
            output = model(image)
            print(output)
            prediction = torch.argmax(output,dim=1)
            print(prediction)
            img = image.view(28, 28)  # remettre en 2D pour l’afficher
            plt.imshow(img, cmap="gray")
            plt.title(f"Chiffre : {label.item()} ; Chiffre prédit ; {prediction.item()}")
            plt.show()


def main():
    model = SimpleNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load("models/Best_model.pth", map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),            # Convertit en tenseur [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Aplatit l’image en vecteur de 784
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    manual_loader = DataLoader(test_dataset,batch_size=1,shuffle=True)

    view_prediction(manual_loader, model)


if __name__ == '__main__':
    main()

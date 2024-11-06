import time
import cv2

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    best_train_loss = 1e9

    train_losses = []
    test_losses = []

    for epoch in range(1, 100):

        t = time.time_ns()

        model.train()
        train_loss = 0

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss = train_loss + loss.item()

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):

            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss = test_loss + loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        train_losses.append(train_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-11/best_model.pth')

        torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-11/current_model.pth')

        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-11/task2_loss_plot.png')
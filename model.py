import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchsummary import summary
import torchvision.models as models




import torch
from torchvision import models
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model = model.to(device)
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_acc.append(correct_train / total_train)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_acc.append(correct_val / total_val)

        print(f'Epoch {epoch+1}/{epochs}, '
            f'Train Loss: {train_losses[-1]:.4f}, '
            f'Val Loss: {val_losses[-1]:.4f}, '
            f'Train Acc: {train_acc[-1]:.4f}, '
            f'Val Acc: {val_acc[-1]:.4f}')

    #output:Plotting the losses
    plt.figure(figsize=(10, 6))
    min_length = min(len(train_losses), len(val_losses))
    plt.plot(range(1, min_length + 1), train_losses[:min_length], label="Training Loss", marker="o")
    plt.plot(range(1, min_length + 1), val_losses[:min_length], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_name+"Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    save_path = f"{model_name}_loss.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    #output:Plotting the losses
    plt.figure(figsize=(10, 6))
    min_length = min(len(train_acc), len(val_acc))
    plt.plot(range(1, min_length + 1), train_losses[:min_length], label="Training Loss", marker="o")
    plt.plot(range(1, min_length + 1), val_losses[:min_length], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_name+"Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    save_path = f"{model_name}_acc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def test_model(model_name,model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # 獲取預測結果
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

#ResNet_34
# 初始化 VGG19_BN 模型，并设置输出类别数为 10
model = models.vgg19_bn(num_classes=10)
# 查看模型结构
summary(model, input_size=(3, 32, 32))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_model("vgg19_bn",model, trainloader, valloader, criterion, optimizer, epochs=40)
test_model("vgg19_bn",model, testloader)
FILE = 'vgg19_bn.pth'
torch.save(model, FILE)
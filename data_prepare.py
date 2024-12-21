import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=0.5),#隨機水平翻轉圖像，概率為0.5。
    transforms.RandomVerticalFlip(p=0.5),#隨機垂直翻轉圖像，概率為0.5。
    transforms.RandomRotation(degrees=30),#隨機旋轉 -30° 到 30° 之間
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,shuffle=False, num_workers=2)

test_indices = list(range(10))
testset = Subset(valset, test_indices)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
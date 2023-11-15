import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet34, resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        resnet = resnet50(pretrained=False)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        resnet = resnet34(pretrained=False)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        resnet = resnet18(pretrained=False)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


# 모델, 손실 함수, 최적화기 정의
device = 'mps'
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 및 정확도 기록
num_epochs = 1
train_accuracy_list = []

# 에포크를 1회만 수행
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    batch_count = 0  # 배치 수를 세기 위한 변수 추가

    # 각 배치마다 학습 및 정확도 계산
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # 각 배치에서 정확도를 계산하고 리스트에 추가
        batch_count += 1
        batch_accuracy = correct_train / total_train
        train_accuracy_list.append(batch_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy:.4f}")

# 테스트
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 정확도 그래프
plt.plot(train_accuracy_list, label='Train Accuracy (Per Batch)')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy per Batch')
plt.legend()
plt.show()


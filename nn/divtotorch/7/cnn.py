import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 1
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 2
class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1_0 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_0(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5) #88 = 24 +24 + 24 + 16 经过卷积层维数都是 88
        self.incept1 = Inception(in_channels=10)
        self.incept2 = Inception(in_channels=20)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(1408, 10) #1408 = 88 * 4 * 4 是卷积的结果

    def forward(self, x):
        in_size = x.size(0)
        # print('in size ' + str(in_size))
        x = self.pooling(torch.nn.functional.relu(self.conv1(x))) # (28 - 4) / 2 = 12
        x = self.incept1(x)
        x = self.pooling(torch.nn.functional.relu(self.conv2(x))) # (12 - 4) / 2 = 4
        x = self.incept2(x)
        # inception 不改变w,h pooling w,h / 2 , conv 如果 kernel == 5 则 - 4
        x = x.view(in_size, -1)
        x = self.fc1(x)
        return x
    
model = Net()
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# 3
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        opt.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

# print(torch.__version__)
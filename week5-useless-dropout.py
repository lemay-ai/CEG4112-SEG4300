# A pytorch dnn with dropout on MNIST dataset compared to a dnn without dropout
# This is intentionally underpowered. Add layers and grow the layer width to get better results. Dropout did nothing for me here. LOL.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 10

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network architecture with dropout
class NetWithDropout(nn.Module):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.dropout2 = nn.Dropout(0.2) # Dropout layer
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x) # Apply dropout
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Define the neural network architecture without dropout
class NetWithoutDropout(nn.Module):
    def __init__(self):
        super(NetWithoutDropout, self).__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Train and test the model without dropout
model_no_dropout = NetWithoutDropout().to(device)
optimizer_no_dropout = optim.SGD(model_no_dropout.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train(model_no_dropout, optimizer_no_dropout, epoch)
    test(model_no_dropout)

# Train and test the model with dropout
model_dropout = NetWithDropout().to(device)
optimizer_dropout = optim.SGD(model_dropout.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train(model_dropout, optimizer_dropout, epoch)
    test(model_dropout)
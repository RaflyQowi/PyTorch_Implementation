# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# Hyperparameter
load_model = True
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 4
model_name = "my_checkpoint.pth.tar"
DIR_NAME = "models"

if DIR_NAME not in os.listdir('.'):
    os.mkdir(DIR_NAME)

model_path = os.path.join(DIR_NAME, model_name)

# Create Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channel =1, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels= 8, kernel_size= (3,3), stride= (1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels= 16, kernel_size= (3,3), stride= (1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

def save_checkpoint(state, filename = model_path):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model_path):
    print("=> Loading checkpoint")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Set Devide
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
train_dataset = datasets.MNIST(root = 'datasets/', train = True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_dataset = datasets.MNIST(root = 'datasets/', train = False, transform= transforms.ToTensor(), download= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint(model_path)

# Train Network
for epoch in range(1, num_epochs+1):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device= device)
        targets = targets.to(device= device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent or adam step
        optimizer.step()
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
    if epoch % 2 == 0 and epoch != 0:
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

# Check accuracy on training and test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_sample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f'Got {num_correct} / {num_sample} with accuracy {float(num_correct) / float(num_sample) * 100:.2f}')

        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
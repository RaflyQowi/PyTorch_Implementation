# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Coneected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Set Devide
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root = 'datasets/', train = True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_dataset = datasets.MNIST(root = 'datasets/', train = False, transform= transforms.ToTensor(), download= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= True)

# Initialize network
model = NN(input_size= input_size,
           num_classes= num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device= device)
        targets = targets.to(device= device)

        # Get into correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent or adam step
        optimizer.step()


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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f'Got {num_correct} / {num_sample} with accuracy {float(num_correct) / float(num_sample) * 100:.2f}')

        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Hyperparameter
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
learning_rate = 0.001
batch_size = 64
num_epochs = 2
num_classes = 10

# Create Biderectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0,c0))
        out = self.linear(out[:,-1,:])
        return out

# Set Devide
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
train_dataset = datasets.MNIST(root = 'datasets/', train = True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_dataset = datasets.MNIST(root = 'datasets/', train = False, transform= transforms.ToTensor(), download= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= True)

# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device= device).squeeze(1)
        targets = targets.to(device= device)

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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_sample += predictions.size(0)

        print(f'Got {num_correct} / {num_sample} with accuracy {float(num_correct) / float(num_sample) * 100:.2f}')

        model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
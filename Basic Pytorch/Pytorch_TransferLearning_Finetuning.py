# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

# Set Devide
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# Load pretrained model and modify it
model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Linear(576, 10)
model.to(device)

# print(model)
# exit()

# load data
train_dataset = datasets.CIFAR10(root = 'datasets/', train = True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
# Batch size will now be 1, try changing the batch_size parameter above and see what happens
img, label = next(iter(train_loader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device= device)
        targets = targets.to(device= device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent or adam step
        optimizer.step()
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")


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
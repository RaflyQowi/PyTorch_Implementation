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
import os

# Set Devide
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 16
num_epochs = 10
DIR_PATH = os.path.join("datasets", "Cat_dogs")
TRAIN_DIR = os.path.join(DIR_PATH, "train")
TEST_DIR = os.path.join(DIR_PATH, "test")
load_model = True

model_name = "cat_VS_dogs.pth.tar"
DIR_MODEL = "models"
if DIR_MODEL not in os.listdir('.'):
    os.mkdir(DIR_MODEL)
model_path = os.path.join(DIR_MODEL, model_name)

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
model.classifier = nn.Linear(576, 2)
model.to(device)

# print(model)
# exit()

def save_checkpoint(state, filename = model_path):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(model_path):
    print("=> Loading checkpoint")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# load data
## Saw how much image stores
def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir(TRAIN_DIR)
# walk_through_dir(TEST_DIR)

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size = (32,32)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root = TRAIN_DIR,
                                  transform= data_transform)
test_data = datasets.ImageFolder(root = TEST_DIR, 
                                 transform= data_transform)
print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# Turn train and test Datasets into DataLoaders
train_loader = DataLoader(dataset= train_data, 
                          batch_size= batch_size,
                          shuffle= True)
test_loader = DataLoader(dataset= test_data,
                         batch_size= batch_size,
                         shuffle= True)


# Batch size will now be 1, try changing the batch_size parameter above and see what happens
img, label = next(iter(train_loader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

if load_model:
    load_checkpoint(model_path)
# Train Network
for epoch in range(1, num_epochs+1):
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
    if epoch % 2 == 0 and epoch != 0:
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)


# Check accuracy on training and test to see how good our model
def check_accuracy(loader, model, training = True):
    if training:
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
check_accuracy(test_loader, model, training= False)
import torch
import torch.nn as nn

# LeNet architecture
# 1x32x32 Input -> (5x5), s=1, p=0 -> avg pool s=2, p=0 -> (5x5), s=1, p=0 -> avg pool s=2, p=0
# -> (5x5) to 120 channels -> Linear 84 -> Linear 10

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv1 = nn.Conv2d(1, 6, kernel_size = (5,5), stride = (1,1), padding = (0,0))
        self.conv2 = nn.Conv2d(6, 16, kernel_size = (5,5), stride = (1,1), padding = (0,0))
        self.conv3 = nn.Conv2d(16, 120, kernel_size = (5,5), stride = (1,1), padding = (0,0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x)) # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

x = torch.randn(64, 1, 32, 32)
model = LeNet5()
print(model(x).shape)

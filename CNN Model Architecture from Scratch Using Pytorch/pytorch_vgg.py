import torch
import torch.nn as nn

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# Then Flatten and 4096x4096x1000 Linear Layers


class VGG(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10000, types = 'VGG16'):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[types])
        self.fcs = nn.Sequential(
            nn.Linear(in_features= 512*7*7, out_features= 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:

            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                    in_channels= in_channels,
                    out_channels= out_channels,
                    kernel_size=(3,3),
                    stride = (1,1),
                    padding = (1,1)
                    ), 
                    nn.BatchNorm2d(x), 
                    nn.ReLU()
                ]
                in_channels = x

            elif x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=(2,2),
                                 stride= (2,2))
                ]
        return nn.Sequential(*layers)
    
def test():
    model = VGG(types='VGG11')
    x = torch.randn(1, 3, 244, 244)
    print(model(x).shape)

if __name__ == '__main__':
    test()
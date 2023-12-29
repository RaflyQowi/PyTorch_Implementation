import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Method for dealing with imbalance datasets
# 1. Oversampling
# 2. Class Weighting

def get_loader(root_dir, batch_size):
    my_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root= root_dir, transform= my_transform)
    class_weight = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weight.append(1/len(files))

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        sample_weights[idx] = class_weight[label]

    sampler = WeightedRandomSampler(sample_weights, 
                                    num_samples= len(sample_weights),
                                    replacement= True)
    
    loader = DataLoader(dataset, batch_size= batch_size, sampler= sampler)
    return loader

def main():
    loader = get_loader(root_dir='datasets\imbalance_dog_data', batch_size= 8)
    num_retrivers = 0
    num_elkhounds = 0
    for epoch in range(10):
        for data, labels in loader:
            num_retrivers += torch.sum(labels == 1)
            num_elkhounds += torch.sum(labels == 0)
    print(num_retrivers)
    print(num_elkhounds)

if __name__ == "__main__":
    main()
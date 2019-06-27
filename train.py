import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models

import json

from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse

ap = argparse.ArgumentParser()

ap.add_argument('--arch', default = 'vgg' )
ap.add_argument('--dir', default = 'flowers' )
ap.add_argument('--learning_rate',type = float, default = 0.001)
ap.add_argument('--epochs',type = int, default = 20)
ap.add_argument('--hidden_layer1', type= int, default = 512)
ap.add_argument('--power', default ='gpu')
ap.add_argument('--save_dir', default="./checkpoint.pth")
ap.add_argument('--dropout',type = float, default = 0.5)

pa = ap.parse_args()
diri = pa.dir
arch = pa.arch
lr = pa.learning_rate
e = pa.epochs
h1 = pa.hidden_layer1
power = pa.power
save_dir = pa.save_dir
dropout = pa.dropout

data_dir = diri
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=102, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=102, shuffle=True)
testloaders = torch.utils.data.DataLoader(test_datasets,batch_size=102, shuffle=True)


if power == gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    inputs = 25088
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    inputs = 1024
elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
    inputs = 9216
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('dropout',nn.Dropout(dropout)),
                          ('fc1', nn.Linear(inputs, h1)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(h1, 256)),
                          ('relu2',nn.ReLU()),
                          ('fc3', nn.Linear(256, 128)),
                          ('relu3', nn.ReLU()),
                          ('fc4',nn.Linear(128,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
model.to(device)
epochs = e
steps = 0
running_loss = 0
print_every = 5
if pa.gpu == gpu:
    model.to('cuda')
for epoch in range(epochs):
    for inputs, labels in trainloaders:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validation loss: {test_loss/len(validloaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloaders):.3f}")
            running_loss = 0
            model.train()
model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'input_size': inputs,
              'output_size': 102,
              'hidden_layers': h1,
              'state_dict': model.state_dict(),
               'model.class_to_idx': model.class_to_idx,
               'arch': models.arch}


torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

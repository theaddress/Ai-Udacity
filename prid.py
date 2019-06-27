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
ap.add_argument('--top_k',type = int, default = 5)
ap.add_argument('--img_path')
ap.add_argument('--dir', default = 'flowers' )
ap.add_argument('--checkpoint_pth', default="./checkpoint.pth")
ap.add_argument('--gpu', default ='gpu')
ap.add_argument('--cat_file', default="cat_to_name.json")
pa = ap.parse_args()
topk = pa.top_k
diri = pa.dir
ck = pa.checkpoint_pth
power = pa.gpu
img_path = pa.img_path
cat_file = pa.cat_file

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

def load_checkpoint(ck):
    checkpoint = torch.load(ck)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             checkpoint['model.class_to_idx'],
                             checkpoint['arch'])
    model.load_state_dict(checkpoint['state_dict'])

    return model




import json

with open(cat_file, 'r') as json_file:
    cat_to_name = json.load(json_file)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    adj = transforms.Compose([transforms.Resize(256),
                              transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    im = Image.open(image_path)
    im = adj(im)
    return im

def predict(image_path, model, topk, power):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.class_to_idx = train_datasets.class_to_idx
    if power == gpu:
        model.to('cuda:0')
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    with torch.no_grad():
        logps = model.forward(img.cuda())
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class

probabilities = predict(image_path, model)
image = process_image(image_path)
probabilities = probabilities

ps = np.array(probabilities[0][0])
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]

i = 0

while i < topk:
    print('{} is most likely {}'.format(labels[i], ps[i]))

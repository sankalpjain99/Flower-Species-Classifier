#Import Required Libraries
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import UtilFuncs

#Creating Arguments for CLI
parser = argparse.ArgumentParser(description='Training File')
parser.add_argument('data_dir', nargs=1, action="store", default=["./flowers"])
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", type=float, action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--arch', dest="arch", action="store"   , default="vgg16", type=str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)

pa = parser.parse_args()

if pa.arch=="help":
    print("List of availabke networks : ")
    print("1. vgg16 (default) ")
    print("2. densenet121")
    quit()
    
if(pa.epochs<=0):
    print("Epochs should be greater than 0")

where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structre = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

train_data,trainloader,validloader,testloader = UtilFuncs.load_data(where)
model,criterion,optimizer = UtilFuncs.buildNN(structre,lr,dropout,hidden_layer1,power)
UtilFuncs.trainNN(model, criterion, optimizer,trainloader, validloader, epochs, 25, power)
UtilFuncs.save_checkpoint(model,train_data,path,structre,hidden_layer1,dropout,lr,epochs)

print("Model trained and saved successfully!!")

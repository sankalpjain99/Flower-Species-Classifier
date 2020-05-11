import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import nn,optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
import UtilFuncs

parser = argparse.ArgumentParser(description='Prediction File')
parser.add_argument('input_img', default='./flowers/test/1/image_06752.jpg', nargs=1, action="store", type = str)
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs=1, action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

argParser = parser.parse_args()
path_img = argParser.input_img
no_of_outputs = argParser.top_k
mode = argParser.gpu
path = argParser.checkpoint
json_file = argParser.category_names

# print(path_img)
# print(path)

traindata,trainloader,validloader,testloader = UtilFuncs.load_data(["./flowers"])
model = UtilFuncs.load_checkpoint(path)
with open(json_file, 'r') as json_file:
    cat_to_name = json.load(json_file)
    
probs = UtilFuncs.predict(path_img,model,no_of_outputs,mode)
labels = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
final_probs = np.array(probs[0][0])

for i in range(no_of_outputs):
    print(" {}. {} : {}".format(i+1,labels[i],final_probs[i]))
    
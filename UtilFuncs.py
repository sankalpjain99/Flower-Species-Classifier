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

#Returns DataLoaders for Train,Validation,Test Data
def load_data(where):
    #Load Data Folders
    for i in where:
        where = str(i)
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Perform Transforms on Image
    train_t = transforms.Compose([transforms.RandomRotation(30),
                                  transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    valid_t = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])
    test_t = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform = train_t)
    valid_data = datasets.ImageFolder(valid_dir,transform = valid_t)
    test_data = datasets.ImageFolder(test_dir,transform = test_t)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = 50,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size = 50)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = 50)
    
    return train_data,trainloader,validloader,testloader

#Return Model,Criterion,Optimizer
def buildNN(arch,lr=0.001,dropout=0.5,hidden_units=4096,mode='gpu'):
    
    if(arch=='vgg16'):
        model = models.vgg16(pretrained=True)
    elif(arch=='densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        print("Pls choose vgg16 or densenet121, Other networks are not available!")
    
    for param in model.parameters():
        param.requires_grad = False
        if(arch=="vgg16"):
            model.classifier = nn.Sequential(OrderedDict([
                                     ('fc1',nn.Linear(25088,hidden_units,bias=True)),
                                     ('relu1',nn.ReLU()),
                                     ('drop1',nn.Dropout(p=0.5)),
                                     ('fc2',nn.Linear(hidden_units,102,bias=True)),
                                     ('softmax1',nn.LogSoftmax(dim=1))]))
        elif(arch=="densenet121"):
            model.classifier = nn.Sequential(OrderedDict([
                                     ('fc1',nn.Linear(1024,hidden_units,bias=True)),
                                     ('relu1',nn.ReLU()),
                                     ('drop1',nn.Dropout(p=0.5)),
                                     ('fc2',nn.Linear(hidden_units,102,bias=True)),
                                     ('softmax1',nn.LogSoftmax(dim=1))]))
        else:
            print("Pls try to use vgg16 or densenet121")
            
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        if torch.cuda.is_available() and mode == 'gpu':
            model.cuda()
            
        return model, criterion, optimizer

#Validate Model
def validation(model, testloader, criterion,mode='gpu'):
    val_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        if torch.cuda.is_available() and mode=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy
    
#Train our Neural Network
def trainNN(model,criterion,optimizer,loader1, loader2, epochs=5,print_every=25,mode='gpu'):
    steps=0
    running_loss=0
    
    print("----------------Training Started------------------\n")

    for e in range(epochs):
        for inputs,labels in loader1:
            steps+=1
            if torch.cuda.is_available() and mode=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            #Forwara and Backward Propogation
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                #VALIDATION
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, loader2, criterion)

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(loader2):.3f}.. "
                      f"Validation accuracy: {accuracy/len(loader2):.3f}")

                running_loss = 0
                model.train()

    print("\n---------------Training Completed---------------")

#Save Checkpoint
def save_checkpoint(model,train_data ,path='./checkpoint.pth',structure ='vgg16', hidden_layer1=4096,dropout=0.5,lr=0.001,epochs=5):
    
    model.cpu
    model.class_to_idx = train_data.class_to_idx
    chpt = {'structure' :structure,
            'hidden_layer1':hidden_layer1,
            'dropout':dropout,
            'lr':lr,
            'no_of_epochs':epochs,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx}
    
    if structure=="resnet101":
        chpt['fc'] = model.fc
    else:
        chpt['classifier'] = model.classifier
    if(path!="./checkpoint.pth"):
        path = path + "/checkpoint.pth"
    torch.save(chpt,path)
    print("Model Saved")

#Load Checkpoint
def load_checkpoint(path='checkpoint.pth'):
    for i in path:
        path=str(i)
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    model,_,_ = buildNN(structure,lr,dropout,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model
   
#Process PIL Image
def process_image(img_path):
    for i in img_path:
        path = str(i)
    img = Image.open(path)
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
    final_img = process(img)
    return final_img

#Make Prediction
def predict(image_path, model, topk=5,power='gpu'):
    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img)
    probability = torch.exp(output)
    return probability.topk(topk)
    

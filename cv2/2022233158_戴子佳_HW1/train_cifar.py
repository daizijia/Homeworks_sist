import torch
from torchvision import datasets, transforms,models
from torch.utils.data import dataloader
from resnet18 import ResNet18
import torch.nn as nn

import copy
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt

batch_sz = 128
learn_rate = 0.001
epochs = 200
modes = ['vanilla', 'imgnet1k']
mode = 'imgnet1k'
model_weight_path = "/home/daizj/Homeworks/cv/hw1/resnet50_8xb32_in1k_20210831-ea4938fc.pth"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_data = dataloader.DataLoader(
    datasets.CIFAR10(root='datasets/', train=True, transform=transforms.Compose([

        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(p=0.5),     
        transforms.ToTensor(),      
        normalize        
    ]), download=True), shuffle=True, batch_size=batch_sz)

train_test = dataloader.DataLoader(
    datasets.CIFAR10(root='datasets/', train=False, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize 
    ]), download=True), shuffle=True, batch_size=batch_sz)


device = torch.device('cuda')          
model = ResNet18()

losslist = []
acclist = []


if mode == modes[0]:
    model.to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    
    for epoch in tqdm.tqdm(range(epochs)):
    
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.94)
        model.train()
        for batch_idx, (x, label) in enumerate(train_data):
            x = x.to(device)
            label = label.to(device)
            logits = model(x)       
            loss = criteon(logits, label)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx == len(train_data) - 1:
                print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in train_test:
                x = x.to(device)
                label = label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)
        lr_scheduler.step() ###
        
        losslist.append(loss.item())
        acclist.append(acc)
        if epoch%50 == 0:
            name = './models/{}_{}_{}.pth'.format("resnet18_cifar10",epoch, acc)
            torch.save(model, name)  
            
else:
    pass


        
print(losslist)
print(acclist)

plt.plot(losslist, label='loss')
plt.plot(acclist, label='acc')
plt.title('model loss')
plt.ylabel('loss and acc')
plt.xlabel('epoch')
plt.legend(['loss', 'acc'], loc='upper left')
plt.savefig('./loss2.png')


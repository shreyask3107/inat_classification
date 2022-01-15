from dataloader import inaturalist
from model import Classifier
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import Adam
import numpy as np

torch.manual_seed(42)

############################################# DEFINE HYPERPARAMS #####################################################
batch_size = 16
epochs = 500
learning_rate = 0.0045 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir='Data/inaturalist_12K', mode='train')
valset = inaturalist(root_dir='Data/inaturalist_12K', mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)

################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
criterion = nn.CrossEntropyLoss()  # Cross entropy loss
model = Classifier(n_classes = 10, layers = [3, 4, 6, 3])
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9898)

# model.load_state_dict(torch.load('best_128pixels_resnet50_lr45.ckpt'))
# epoch_end = 100

################################### CREATE CHECKPOINT DIRECTORY ####################################################
checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#################################### HELPER FUNCTIONS ##############################################################

def get_model_summary(model, input_tensor_shape):
    summary(model, input_tensor_shape)

def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct/total

def train(model, dataset, optimizer, criterion, device, epoch_num):

    model.train()
    model.to(device)
    train_loss = []
    train_acc = []
    
    for i, (data, target) in enumerate(dataset):
        data, target = data.to(device), target.to(device)
        output = model(data)
                
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred_cls = output.max(1)[1]
        correct = pred_cls.eq(target.long().data).cpu().sum()
        
        train_acc.append(correct.item()/data.shape[0])
        train_loss.append(loss.item())
        
        if i % 100 == 0:
            print('\rTrain Epoch: {} [({:.0f}%)]\tLoss: {:.3f}\tAcc: {:.2f}% '.format(
            epoch_num+1, 100. * i / len(dataset), np.mean(train_loss), 100*np.mean(train_acc)), end="")
            
    return 
        
    
    

def eval(model, dataset, device):

    val_acc = []
    model.eval()
    
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred_cls = output.max(1)[1]
            correct = pred_cls.eq(target.long().data).cpu().sum()
            val_acc.append(correct.item()/data.shape[0])
            
    print("Val Acc: {:.2f}%".format(100*np.mean(val_acc)))
    
    return np.mean(val_acc)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################################################### TRAINING #######################################################
#Training and Validation
best_valid_loss = float('inf')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best = 0

for epoch in range(epochs):
    
    start_time = time.monotonic()
    train(model, trainloader, optimizer, criterion, device, epoch)
    val_acc = eval(model, valloader, device )
    my_lr_scheduler.step()
    if val_acc > best:
        best = val_acc
        torch.save(model.state_dict(), "checkpoints/best_128pixels_centrecropped_resnet50_lr45.ckpt")

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
    print("\n\n\n TIME TAKEN FOR THE EPOCH: {} mins and {} seconds".format(epoch_mins, epoch_secs))


print("OVERALL TRAINING COMPLETE")

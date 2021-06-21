#Author: Ian Edwards
#Date: 18/06/2021
#Description: CNN Shape classifier to train on and classify squares, circles, triangles, stars and pentagons.

#Necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from PIL import Image
from sklearn.metrics import confusion_matrix
import os
import ast
import sys
import argparse
from shapeDataset import shapeDataset
import time

text = "Shape classification Neural Network (CNN)."
parser = argparse.ArgumentParser(description=text) #setup argument parser.
parser.add_argument('-batchsize', '--batchsize', type = int, required=False)
parser.add_argument('-epochs', '--epochs', type = int, required=False)
parser.add_argument('-lr', '--lr', type = float, required=False)
args = parser.parse_args()

#Use GPU instead of CPU - makes the training a lot faster.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#Hyperparameters
if args.batchsize == None:
    BATCH_SIZE = 64
else:
    BATCH_SIZE = args.batchsize
if args.epochs == None:
    MAX_EPOCHS = 25
else:
    MAX_EPOCHS = args.epochs
if args.lr == None:
    learning_rate = 0.001
else:
    learning_rate = args.lr

modelSaveName = './shapeClassifierModel.pt'

#Record data
history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
}

#Shape classifications
shapeClass = {
    0 : "Circle",
    1 : "Triangle",
    2 : "Square",
    3 : "Pentagon",
    4 : "Star"
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # create 6, 5x5 kernels
        # Pytorch does valid padding by default. 
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size = 5)

        # 2x2 max-pooling 
        self.pool = nn.MaxPool2d(2, 2)        
        # 6, 25x25 feature maps going out of the pooling stage 
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 6, kernel_size = 5)
        # output 16, 20x20 feature maps
        self.conv3 = nn.Conv2d(in_channels=6, out_channels = 16, kernel_size = 5)
        # there will be another pooling stage in the forward pass before fc1
        # output 16, 10x10 feature maps 
        
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, input):
        output = self.pool(F.relu(self.conv1(input)))
        output = self.pool(F.relu(self.conv2(output)))
        output = self.pool(F.relu(self.conv3(output)))
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
    # Return mean loss, accuracy
    return running_loss / len(loader), correct / total

def train():
    start_time = time.time()
    running_loss = 0.0
    total = 0.0

    for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
        print("Starting Epoch: {}".format(epoch+1))    
        for i, data in enumerate(train_loader, 0):
            model.train()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # keep record of loss across mini-batches until logged
            running_loss += loss.item()
            
            # log results
            if i % 50 == 9:    # log every 100 mini-batches
                mean_loss = running_loss / 50 
                
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                train_acc = correct / labels.size(0)
                
                history['train_loss'].append(mean_loss)
                history['train_acc'].append(train_acc)
                
                print('# mini-batch {}\ntrain loss: {} train accuracy: {}'.format(
                    i + 1, mean_loss, train_acc))
                running_loss = 0.0
                
                # evaluate on validation dataset
                mean_loss, val_acc = evaluate(val_loader)
                history['val_loss'].append(mean_loss)
                history['val_acc'].append(val_acc)
                print("validation loss: {} validation accuracy: {}\n".format(mean_loss, val_acc))

    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))
    saveHistory()
    torch.save(model, modelSaveName) 

def plotData():
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Training and validation loss and accuracy curves")
    ax1.plot(history['train_loss'], label='train_loss')
    ax1.plot(history['val_loss'], label='val_loss')
    ax1.set(xlabel = "Logging iterations", ylabel = "Cross-entropy Loss")
    ax1.legend()
    # plt.show()

    #fig = plt.figure(figsize=(8,8))
    ax2.plot(history['train_acc'], label='train_acc')
    ax2.plot(history['val_acc'], label='val_acc')
    ax2.set(xlabel = "Logging iterations", ylabel = "Accuracy")
    ax2.legend()
    plt.show()

def plotConfusionMatrix():
    # In this case we know there will only be one batch consisting of the entire test set
    y_pred = []
    y = []
    for x, labels in test_loader:
        x = x.to(device)
        labels = labels.to(device)
        outputs = model(x)
        y_pred.extend(torch.max(outputs, 1)[1].data.cpu().numpy())
        y.extend(labels.data.cpu().numpy())

    cm = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=4)
    #print(cm)

    plt.figure(figsize = (10,10))
    cm = confusion_matrix(y, y_pred, normalize="true")
    plt.matshow(cm, fignum=1)

    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.xlabel("Prediction")
    plt.ylabel("True")

    # We can retrieve the categories used by the LabelEncoder
    classes = list(shapeClass.values())
    plt.gca().set_xticklabels(classes)
    plt.gca().set_yticklabels(classes)

    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    plt.show()

def checkImage(path, model):
    img = Image.open(path)
    preprocess = transforms.Compose([transforms.ToTensor()])
    img = preprocess(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    return shapeClass[predicted.cpu().detach().numpy()[0]]

def saveHistory(path='historyData.txt', data=history):
    with open(path, 'w') as f:
        print(history, file=f)

def loadHistory(path='./historyData.txt', data=history):
    file = open(path, "r")
    contents = file.read()
    history = ast.literal_eval(contents)
    file.close()
    return history


#Load in the data - needed for training, validation or testing.
dataset = shapeDataset(csvfile = 'shape_data.csv', rootdir = './greyscale/',  
    transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [52321, 2000])
train_set, val_set = torch.utils.data.random_split(train_set, [50321, 2000])
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

#Create model
model = Net().to(device)
#Use cross entropy loss function
criterion = nn.CrossEntropyLoss()
#Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("""\
╔═╗┬ ┬┌─┐┌─┐┌─┐  ┌─┐┬  ┌─┐┌─┐┌─┐┬┌─┐┬┌─┐┬─┐
╚═╗├─┤├─┤├─┘├┤   │  │  ├─┤└─┐└─┐│├┤ │├┤ ├┬┘
╚═╝┴ ┴┴ ┴┴  └─┘  └─┘┴─┘┴ ┴└─┘└─┘┴└  ┴└─┘┴└─ 
""")

option = input("Please choose a number from below:\n1 : Train model\n2 : Statistics\n3 : Evaluate\n4 : Exit program\n")
while (option != '4'):
    if option == '1':
        #modelSaveName = input("Please enter a name for the model (for saving)") --> customize saving model name
        train()
    elif (option == '2' or option == '3'):
        if (os.path.isfile(modelSaveName)):
            model = torch.load(modelSaveName)
            print("Model loaded successfully!", model.parameters)
            history = loadHistory(data=history)
        else: #no model exists, train one up.
            train()

        if (option == '2'):
            print("Fetching statistics...\n")
            t_loss, t_acc = evaluate(test_loader)
            print("Testing accuracy:\n", t_acc)
            print("Testing loss:\n", t_loss)
            plotData()
            plotConfusionMatrix()

        if (option == '3'):
            suboption = input("Enter the path to an image (q to exit)\n")
            while (suboption != 'q'):
                print(checkImage(suboption, model))
                suboption = input("Enter the path to an image (q to exit)\n")

    option = input("Please choose a number from below:\n1 : Train model\n2 : Statistics\n3 : Evaluate\n4 : Exit program\n")

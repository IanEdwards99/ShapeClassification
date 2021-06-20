#Author: Ian Edwards
#Date: 18/06/2021
#Description: CNN Shape classifier 
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

from shapeDataset import shapeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU instead of CPU.

#HyperparametersL
BATCH_SIZE = 32
MAX_EPOCHS = 20
learning_rate = 0.001

#Load in the data
dataset = shapeDataset(csvfile = 'shape_data.csv', rootdir = './greyscale/',  
    transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [48999, 1000])
train_set, val_set = torch.utils.data.random_split(train_set, [43999, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # create 6, 5x5 kernels
        # Pytorch does valid padding by default. 
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size = 3)

        
        # 2x2 max-pooling 
        self.pool = nn.MaxPool2d(2, 2)        
        # 6, 25x25 feature maps going out of the pooling stage 
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 6, kernel_size = 3)
        # output 16, 20x20 feature maps
        self.conv3 = nn.Conv2d(in_channels=6, out_channels = 16, kernel_size = 3)
        # there will be another pooling stage in the forward pass before fc1
        # output 16, 10x10 feature maps 
        
        self.fc1 = nn.Linear(16 * 23*23, 120)
        self.fc2 = nn.Linear(84, 32)
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

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, loader):
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

history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
}

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
        if i % 100 == 9:    # log every 10 mini-batches
            mean_loss = running_loss / 10 
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            train_acc = correct / labels.size(0)
            
            history['train_loss'].append(mean_loss)
            history['train_acc'].append(train_acc)
            
            
            print('# mini-batch {}\ntrain loss: {} train accuracy: {}'.format(
                  i + 1, mean_loss, train_acc))
            running_loss = 0.0
            
            # evaluate on validation dataset
            mean_loss, val_acc = evaluate(model, val_loader)
            history['val_loss'].append(mean_loss)
            history['val_acc'].append(val_acc)
                  
            print("validation loss: {} validation accuracy: {}\n".format(mean_loss, val_acc))

print('Finished Training')
torch.save(model, './my_mnist_model.pt') 

# fig = plt.figure(figsize=(8,8))
# plt.plot(history['train_loss'], label='train_loss')
# plt.plot(history['val_loss'], label='val_loss')
# plt.xlabel("Logging iterations")
# plt.ylabel("Cross-entropy Loss")
# plt.legend()
# plt.show()

# fig = plt.figure(figsize=(8,8))
# plt.plot(history['train_acc'], label='train_acc')
# plt.plot(history['val_acc'], label='val_acc')
# plt.xlabel("Logging iterations")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# from sklearn.metrics import confusion_matrix
# # In this case we know there will only be one batch consisting of the entire test set
# it = iter(val_loader)
# x, y = next(it)

# outputs = model(x)
# _, y_pred = torch.max(outputs, 1)

# cm = confusion_matrix(y.numpy(), y_pred.numpy())
# np.set_printoptions(precision=4)
# print(cm)

# plt.figure(figsize = (10,10))
# cm = confusion_matrix(y.numpy(), y_pred.numpy(), normalize="true")
# plt.matshow(cm, fignum=1)

# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    
# plt.xticks(range(8))
# plt.yticks(range(8))
# plt.xlabel("Prediction")
# plt.ylabel("True")

# # We can retrieve the categories used by the LabelEncoder
# classes = val_set.enc.classes_.tolist()
# plt.gca().set_xticklabels(classes)
# plt.gca().set_yticklabels(classes)

# plt.title("Normalized Confusion Matrix")
# plt.colorbar()
# plt.show()
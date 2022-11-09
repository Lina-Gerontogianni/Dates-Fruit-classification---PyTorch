# Importing libraries
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
       
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import train_test_split
import sklearn as sk

import matplotlib.pyplot as plt
import time

# For reproducibilty purposes
sk.utils.check_random_state(42)
from numpy.random import seed
seed(42)
torch.manual_seed(42)

# Loading data
data = pd.read_excel('Date_Fruit_Datasets.xlsx', engine='openpyxl')

# Mapping dates's types to indices
types_mapping = {
    categ: value for value, categ in enumerate(data.Class.unique()) }

y_data = data.Class.map(types_mapping).values

# Creating the feature dataset 
X_data = data.drop('Class', axis=1).values


# Splitting data to training and test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, shuffle= True, test_size=0.2 )

# Scaling the feature dataset
scaler = sk.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting the numpy data to Pytorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Building the ANN 
class ANN(nn.Module):
    
    def __init__(self, input_size, output_size, layers = []):
        super().__init__()
        self.L1 = nn.Linear(input_size, layers[0])
        self.L2 = nn.Linear(layers[0], layers[1])
        self.L3 = nn.Linear(layers[1], output_size)        
        
    def forward(self, X):
        X = F.relu(self.L1(X))
        X = F.relu(self.L2(X))
        X = F.softmax(self.L3(X))
        return X
 
# Instanciating the ANN model
model = ANN(input_size=X_train.shape[1], output_size=7, layers=[600,500])

# Instanciating the loss function and the optimiser
loss_fun = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        
# Lists to save results for visualisations later on
epochs = 190
train_losses = []
train_correct = []
test_losses = []
test_correct = []

# Timing the training process
start_point = time.time()

for i in range(epochs):
    
    train_corr = 0
    test_corr = 0
    i += 1
    # Apply the model
    y_pred = model.forward(X_train)
    loss = loss_fun(y_pred, y_train)
    
    # Tally the number of correct predictions
    predicted = torch.max(y_pred.data, 1)[1]
    train_corr = (predicted == y_train).sum()
    
    if i%10==0:
        print(f'Epoch: {i} ---> Loss: {loss.item():10.8f} & Train Accuracy: {round(100 * train_corr.item()/X_train.shape[0], 2)} %')
        
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    train_losses.append(loss.item())
    train_correct.append(train_corr.item())
    
    # Predict the test data
    with torch.no_grad():
        
        # Apply the model
        y_fitted = model.forward(X_test) 

        # Tally the number of correct predictions
        predicted = torch.max(y_fitted.data, 1)[1] 
        test_corr  = (predicted == y_test).sum()
    
    # Update test loss and accuracy for the epoch
    loss = loss_fun(y_fitted, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)
    
print(f'Training time: {round((time.time() - start_point), 2)} secs')      
        
# Plotting the losses
plt.figure(figsize=(10, 10))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.title('Loss function')
plt.xlabel('Epoch')
plt.ylabel('Loss values')
plt.legend()
plt.show()

# Plotting the accuracies
plt.figure(figsize=(10, 10))
plt.plot([t/X_train.shape[0] for t in train_correct], label='Training accuracy')
plt.plot([t/X_test.shape[0] for t in test_correct], label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Proportions')
plt.title('Accuracy')
plt.legend()
        
# the confusion matrix of classified dates
conf_mat = confusion_matrix(predicted, y_test)

# percentage of correctly classified dates per date type
perc = 100* np.diag(conf_mat) / np.sum(conf_mat, axis=0)
perc = pd.DataFrame(perc.round(2), index=data.Class.unique(), columns = ['Correctly_classified_%'])
print(perc)

# test accuracy
test_acc = 100 * (test_correct[189]/X_test.shape[0]).numpy()
print(f'Test Accuracy: {test_acc.round(1)} %')

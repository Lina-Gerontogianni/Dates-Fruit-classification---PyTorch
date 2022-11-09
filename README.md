# Dates-Fruit-classification---PyTorch
Mutliclass classification via ANN with PyTorch '1.13.0'

A [kaggle](https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets) dataset concerning 34 characteristics of 718 dates fruit of seven different types. 

For the prediction of the dates' type, an Artificial Neural Network (ANN) is employed with two hidden layers. The test accuracy is 92.2% whereas the CPU convergence training time is only a few seconds.

The plots of the training/testing loss and accuracy can be found inside the image folder, while the script is named as *Dates_fruit.py*.

Here are the results of the:
- training process 
- the training time 
- the correctly classified dates per date type (ROTANA seems to be harder to distinguish -85.71 % compared to the rest types, like SAFAVI which is 100% correctly classified)**
- the test accuracy 

```
Epoch: 10 ---> Loss: 1.38627064 & Train Accuracy: 77.86 %
Epoch: 20 ---> Loss: 1.26948118 & Train Accuracy: 89.83 %
Epoch: 30 ---> Loss: 1.23669517 & Train Accuracy: 92.76 %
Epoch: 40 ---> Loss: 1.21450937 & Train Accuracy: 95.26 %
Epoch: 50 ---> Loss: 1.20545685 & Train Accuracy: 96.1 %
Epoch: 60 ---> Loss: 1.20443320 & Train Accuracy: 96.1 %
Epoch: 70 ---> Loss: 1.20242202 & Train Accuracy: 96.38 %
Epoch: 80 ---> Loss: 1.20737994 & Train Accuracy: 95.82 %
Epoch: 90 ---> Loss: 1.20538068 & Train Accuracy: 96.1 %
Epoch: 100 ---> Loss: 1.21722007 & Train Accuracy: 94.71 %
Epoch: 110 ---> Loss: 1.20463216 & Train Accuracy: 96.1 %
Epoch: 120 ---> Loss: 1.20293689 & Train Accuracy: 96.38 %
Epoch: 130 ---> Loss: 1.19872355 & Train Accuracy: 96.66 %
Epoch: 140 ---> Loss: 1.19724596 & Train Accuracy: 96.8 %
Epoch: 150 ---> Loss: 1.19465399 & Train Accuracy: 97.08 %
Epoch: 160 ---> Loss: 1.19456255 & Train Accuracy: 97.08 %
Epoch: 170 ---> Loss: 1.19449413 & Train Accuracy: 97.08 %
Epoch: 180 ---> Loss: 1.19445348 & Train Accuracy: 97.08 %
Epoch: 190 ---> Loss: 1.19440508 & Train Accuracy: 97.08 %

Training time: 4.6 secs

        Correctly_classified_%
BERHI                    91.67
DEGLET                   90.00
DOKOL                    94.00
IRAQI                   100.00
ROTANA                   85.71
SAFAVI                  100.00
SOGAY                    85.00

Test Accuracy: 92.2 %

```
** the distribution of the dates' types in the dataset:

DOKOL:     204, 
SAFAVI:     199, 
ROTANA:    166, 
DEGLET:      98, 
SOGAY:      94, 
IRAQI:      72, 
BERHI:      65

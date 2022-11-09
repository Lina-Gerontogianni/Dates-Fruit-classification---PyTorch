# Dates-Fruit-classification---PyTorch
Mutliclass classification via ANN with PyTorch '1.13.0'

A [kaggle](https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets) dataset concerning 34 characteristics of 718 dates fruit of seven different types. 

For the prediction of the dates' type, an Artificial Neural Network (ANN) is employed with two hidden layers. The test accuracy is 93.9% (higher than the cited paper in [here]) whereas the CPU training time is only a few seconds (< 3 secs in an Apple M1 Pro device).

The plots of the training/testing loss and accuracy can be found inside the image folder, while the script is named as *Dates_fruit.py*.

Here are the results of the predicted classes for the test data: 

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

Training time: 2.28 secs

        Correctly_classified_%
BERHI                    91.67
DEGLET                   85.00
DOKOL                    94.00
IRAQI                   100.00
ROTANA                   97.14
SAFAVI                  100.00
SOGAY                    85.00

Test Accuracy: 93.9 %

```

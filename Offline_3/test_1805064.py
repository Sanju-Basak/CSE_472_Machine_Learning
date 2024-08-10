import pickle
import torchvision.datasets as ds
import torch.nn.functional as F
from torchvision import transforms
from train_1805064 import Dense, ReLU, Softmax
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix




independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())

# Create DataLoader instances for the test set
test_loader = DataLoader(independent_test_dataset, batch_size=len(independent_test_dataset), shuffle=False)

network = [
    Dense(784, 1024),
    ReLU(),
    Dense(1024, 26),
    Softmax()
]

# Load the model
with open('model_1805064.pickle', 'rb') as f:
    params= pickle.load(f)

weights, biases = params

layer_index = 0
# Set the parameters
for i in range(len(network)):
    if isinstance(network[i], Dense):
        network[i].weights= weights[layer_index]
        network[i].bias= biases[layer_index]
        layer_index += 1


for batch in test_loader:
    x_test, y_test = batch
    break

x_test= x_test.view(x_test.shape[0], -1).numpy().T
one_hot_y_test= np.zeros((26, x_test.shape[1]))
for j in range(x_test.shape[1]):
    one_hot_y_test[y_test[j]-1, j]= 1

#Test
output = x_test
for layer in network:
    output = layer.forward(output)
predicted_test = np.argmax(output, axis=0)
true_test = np.argmax(one_hot_y_test, axis=0)
accuracy_test = accuracy_score(true_test, predicted_test)
f1_score_test = f1_score(true_test, predicted_test, average='macro')
confusion_matrix_test = confusion_matrix(true_test, predicted_test)
print("Accuracy on test set: ", accuracy_test)
print("F1 score on test set: ", f1_score_test)

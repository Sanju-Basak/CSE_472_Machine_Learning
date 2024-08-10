import torchvision.datasets as ds
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import pickle

np.random.seed(42)


def cross_entropy(output, target):
    return -np.sum(target * np.log(output + 1e-8)) / output.shape[1]


class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None
        self.t = 0

    def initialize(self, weights_shape, bias_shape):
        self.m_weights = np.zeros(weights_shape)
        self.v_weights = np.zeros(weights_shape)
        self.m_bias = np.zeros(bias_shape)
        self.v_bias = np.zeros(bias_shape)
        self.t = 0

    def update(self, weights, bias, weights_gradient, bias_gradient, learning_rate):
        if self.m_weights is None:
            self.initialize(weights.shape, bias.shape)

        self.t += 1

        # Update weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * weights_gradient
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (weights_gradient ** 2)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        weights_update = learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        weights -= weights_update

        # Update bias
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_gradient
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (bias_gradient ** 2)
        m_hat_bias = self.m_bias / (1 - self.beta1 ** self.t)
        v_hat_bias = self.v_bias / (1 - self.beta2 ** self.t)
        bias_update = learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
        bias -= bias_update

        return weights, bias


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

    def set_parameters(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def get_parameters(self):
        return self.weights, self.bias

class Dense(Layer):
    def __init__(self, input_size, output_size):
        #he initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.optimizer= AdamOptimizer()

    def forward(self, input):
        self.input = input
        forward= np.dot(self.weights, self.input) + self.bias
        return forward
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) / output_gradient.shape[1]
        input_gradient = np.dot(self.weights.T, output_gradient) 
        bias_gradient = np.reshape(np.mean(output_gradient, axis=1), (output_gradient.shape[0], 1))
        # self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * bias_gradient
        self.weights, self.bias= self.optimizer.update(self.weights, self.bias, weights_gradient, bias_gradient, learning_rate)
        return input_gradient
    
class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        forward= np.maximum(0, self.input)
        return forward
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * (self.input > 0)
        return input_gradient
    
def softmax(x):
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0)

class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        forward= softmax(self.input)
        return forward
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient
    
class Dropout(Layer):
    def __init__(self, p):
        self.p = p
        self.input = None
        self.output = None
        self.mask = None

    def forward(self, input):
        self.input = input
        if(self.p == 0):
            self.output = input
            return self.output
        elif (self.p == 1):
            self.output = np.zeros(input.shape)
            return self.output
        self.mask = (np.random.rand(*self.input.shape) > self.p).astype(float)
        self.output = np.multiply(self.input, self.mask) / (1 - self.p)

    def backward(self, output_gradient, learning_rate):
        return (output_gradient * self.mask) / (1-self.p)
    


def create_graph(training_loss_array, validation_loss_array, training_accuracy_array, validation_accuracy_array, f1_score_array):
    #create 3 subplots and set their titles and labels and save the figure and the subplots should be square and normal sized
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Loss and Accuracy')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax3.set_title('F1 Score')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('F1 Score')
    #plot the training loss and validation loss in the first subplot
    ax1.plot(training_loss_array, label='Training Loss')
    ax1.plot(validation_loss_array, label='Validation Loss')
    ax1.legend()
    #plot the training accuracy and validation accuracy in the second subplot
    ax2.plot(training_accuracy_array, label='Training Accuracy')
    ax2.plot(validation_accuracy_array, label='Validation Accuracy')
    ax2.legend()
    #plot the training f1 score and validation f1 score in the third subplot
    ax3.plot(f1_score_array, label='F1 Score')
    ax3.legend()
    #save the figure
    plt.savefig('loss_accuracy.png')
    #show the figure
    plt.show()
    
def create_confusion(confusion_matrix_test):
    sns.set_theme()
    plt.figure(figsize=(20, 20))
    sns.heatmap(confusion_matrix_test, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.show()

def predict(network, x):
    output = x
    for layer in network:
        output = layer.forward(output)
    return output

def predict_val(network, x):
    #ignore dropout
    output = x
    for layer in network:
        if isinstance(layer, Dropout):
            continue
        output = layer.forward(output)


train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())

# Define the split ratio
train_ratio = 0.85
validation_ratio = 0.15

# Calculate the sizes of the training and validation sets
num_samples = len(train_validation_dataset)
train_size = int(train_ratio * num_samples)
validation_size = num_samples - train_size
test_size = len(independent_test_dataset)

# Use random_split to create the training and validation sets
train_set, validation_set = random_split(train_validation_dataset, [train_size, validation_size])




network = [
    Dense(784, 1024),
    ReLU(),
    Dense(1024, 26),
    Softmax()
]

# Hyperparameters
learning_rate = 1e-3
batch_size = 1024
epochs = 20

# Create DataLoader objects for the training and validation sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=validation_size, shuffle=True)

# Create DataLoader object for the test set
test_loader = DataLoader(independent_test_dataset, batch_size=test_size, shuffle=False)


for batch in validation_loader:
    x_val, y_val = batch
    break

x_val= x_val.view(x_val.shape[0], -1).numpy().T
one_hot_y_val= np.zeros((26, x_val.shape[1]))
for j in range(x_val.shape[1]):
    one_hot_y_val[y_val[j]-1, j]= 1

for batch in test_loader:
    x_test, y_test = batch
    break

x_test= x_test.view(x_test.shape[0], -1).numpy().T
one_hot_y_test= np.zeros((26, x_test.shape[1]))
for j in range(x_test.shape[1]):
    one_hot_y_test[y_test[j]-1, j]= 1


training_loss_array = []
validation_loss_array = []
training_accuracy_array = []
validation_accuracy_array = []
f1_score_array = []


for i in range(epochs):
    predictions = []
    true_labels = []
    training_loss = 0
    for batch in tqdm(train_loader):
        # Get batch
        x, y = batch

        x= x.view(x.shape[0], -1).numpy().T
        
        one_hot_y= np.zeros((26, x.shape[1]))
        for j in range(x.shape[1]):
            one_hot_y[y[j]-1, j]= 1
        
        # Forward pass
        output= predict(network, x)
        # output = softmax(output)

        predicted = np.argmax(output, axis=0)
        true = np.argmax(one_hot_y, axis=0)
        predictions.append(predicted)
        true_labels.append(true)

        # Calculate loss
        training_loss = training_loss+ cross_entropy(output, one_hot_y)

        # Backward pass
        output_gradient = output - one_hot_y
        for layer in reversed(network):
            output_gradient = layer.backward(output_gradient, learning_rate)
    training_loss= training_loss/ len(train_loader)
    training_loss_array.append(training_loss)
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    accuracy = accuracy_score(true_labels, predictions)
    training_accuracy_array.append(accuracy)
    print("Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1, training_loss, accuracy))

    # Validation
    validation_loss= 0
    output = predict(network, x_val)
    # output = softmax(output)
    validation_loss = validation_loss+ cross_entropy(output, one_hot_y_val)
    validation_loss_array.append(validation_loss)
    
    predicted_val = np.argmax(output, axis=0)
    true_val = np.argmax(one_hot_y_val, axis=0)
    accuracy_val = accuracy_score(true_val, predicted_val)
    f1_score_val = f1_score(true_val, predicted_val, average='macro')
    validation_accuracy_array.append(accuracy_val)
    f1_score_array.append(f1_score_val)
    print("Validation Loss: {:.3f}, Accuracy: {:.3f}, F1 Score: {:.3f}".format(validation_loss, accuracy_val, f1_score_val))

create_graph(training_loss_array, validation_loss_array, training_accuracy_array, validation_accuracy_array, f1_score_array)

# Test
output = predict(network, x_test)
test_loss = cross_entropy(output, one_hot_y_test)
predicted_test = np.argmax(output, axis=0)
true_test = np.argmax(one_hot_y_test, axis=0)
accuracy_test = accuracy_score(true_test, predicted_test)
f1_score_test = f1_score(true_test, predicted_test, average='macro')
confusion_matrix_test = confusion_matrix(true_test, predicted_test)
print("Test Loss: {:.3f}, Accuracy: {:.3f}, F1 Score: {:.3f}".format(test_loss, accuracy_test, f1_score_test))

# Plot confusion matrix using seaborn
create_confusion(confusion_matrix_test)

# create pickle file and save only the weights and biases of the network if the layer is Dense
# an array of tuples of weights and biases
weights= []
biases= []
for layer in network:
    if isinstance(layer, Dense):
        weights.append(layer.weights)
        biases.append(layer.bias)

with open('model_1805064.pickle', 'wb') as f:
    pickle.dump([weights, biases], f)






        

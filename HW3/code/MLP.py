import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_size=3, output_size=3):
        super(MLP, self).__init__()
        ### YOUR CODE HERE
        #defining the layers
        self.f1 = nn.Linear(input_size, hidden_size) #first hidden layer
        self.f2 = nn.Linear(hidden_size, output_size) # output layer
        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        #forward pass through the network
        x = F.relu(self.f1(x))  #first linear transformation within ReLU activation
        x = self.f2(x)  #second linear transformation
        ### END YOUR CODE
        return x

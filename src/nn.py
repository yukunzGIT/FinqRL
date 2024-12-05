import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format}) # we ensure floating-point numbers are printed with three decimal places.
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def key_value_attention(key, value):
    # a simpler version to compute attention weights from key and value tensors and applies them to the value
    # key = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # value = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
    # output = key_value_attention(key, value)

    # Example Output:
    # tensor([[ 1.286,  2.286],
    #         [ 3.429,  4.429]])

    product = key.matmul(value.transpose(0, 1)) # Computes the dot product between key and the transpose of value. This is the raw similarity score matrix.
    relued_product = torch.relu(product) # Applies the ReLU activation function, ensuring non-negative values.
    attn = relued_product / relued_product.sum(dim=1, keepdim=True) # Normalizes the scores along each row (dim=1) to compute attention weights.
    return attn.matmul(value) # Applies the attention weights to the value tensor, resulting in a weighted output.


class MLP(nn.Module):
    # a simple Multi-Layer Perceptron class
    def __init__(self, in_dim, hid_dim, out_dim, layer_num=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.in_dim = in_dim # Input feature size
        self.hid_dim = hid_dim # Hidden layer size.
        self.out_dim = out_dim # Output feature size
        self.lay_num = layer_num # Number of layers in the network (default is 2)
        self.activation = activation # Activation function (default is ReLU).
        layers = [nn.Linear(in_dim, hid_dim)] # The first layer maps input features to the hidden dimension.
        for i in range(self.lay_num-1): # loop adds layer_num-1 hidden layers, each followed by an activation function.
            layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            layers.append(self.activation())
        layers.append(nn.Linear(hid_dim, out_dim)) # The final layer maps the hidden layer to the output dimension.

        self.layers = nn.Sequential(*layers) # Combines all layers into a Sequential module.

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device) # Moves the model to the appropriate device.

    def forward(self, x, stop_layer=None): # Allows stopping at a specific layer (useful for debugging or feature extraction).
        # Implements the forward pass, where the input tensor x is processed through the layers.
        for l in self.layers[:stop_layer]:
            x = l(x)
        return x

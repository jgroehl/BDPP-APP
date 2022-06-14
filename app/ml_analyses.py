import pickle
import numpy as np
import torch
from torch import nn


def calculate_random_forest_score(test_data, model_save_path):
    with open(model_save_path, 'rb') as save_file:
        rf = pickle.load(save_file)
    return rf.predict(test_data)


def estimate_data_with_descriptor_network(data, network_path):
    input_features = np.shape(data)[1]
    network = FeedForwardNeuralNetwork(input_features)
    network.load_state_dict(torch.load(network_path))
    network.float()
    network.eval()

    data = torch.from_numpy(data).float()
    estimates = network(data)
    return estimates.cpu().detach().numpy()


def estimate_data_with_image_network(data, network_path):
    data = data.reshape((-1, 1, 256, 256)).astype(np.float32)
    network = ConvolutionalNeuralNetwork()
    network.load_state_dict(torch.load(network_path, map_location=torch.device('cpu')))
    network.float()
    network.eval()

    data = torch.from_numpy(data).float()
    estimates = network(data)
    return estimates.cpu().detach().numpy()


class FeedForwardNeuralNetwork(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.hidden_size = 2 * input_size

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LeakyReLU())

        self.hidden_layer_1 = self.hidden_layer()
        self.hidden_layer_2 = self.hidden_layer()
        self.hidden_layer_3 = self.hidden_layer()
        self.hidden_layer_4 = self.hidden_layer()

        self.output_layer = nn.Linear(self.hidden_size, 1)

    def hidden_layer(self):
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.input_layer(out)
        out = self.hidden_layer_1(out)
        out = self.hidden_layer_2(out)
        out = self.hidden_layer_3(out)
        out = self.hidden_layer_4(out)
        out = self.output_layer(out)
        return out


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        num_channels = 8
        #256 x 256
        self.conv_layer_1 = self.conv_layer(1, num_channels)
        #128 x 128
        self.conv_layer_2 = self.conv_layer(num_channels, 2*num_channels)
        #64 x 64
        self.conv_layer_3 = self.conv_layer(2*num_channels, 4*num_channels)
        #32 x 32
        self.conv_layer_4 = self.conv_layer(4*num_channels, 8*num_channels)
        #16 x 16
        self.conv_layer_5 = self.conv_layer(8*num_channels, 16*num_channels)
        #8 x 8

        hidden_layer_size = (8 * 8 * (16*num_channels))
        hidden_size = 200

        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),#padding macht einen 1px rahmen, damit Px Anzahl stimmt
            nn.LeakyReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        out = self.conv_layer_4(out)
        out = self.conv_layer_5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fully_connected(out)
        return out

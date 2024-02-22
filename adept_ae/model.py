import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    




class ConvAutoencoder(nn.Module):
    def __init__(self,  input_dims, latent_dims, channels = 3):
        super(ConvAutoencoder, self).__init__()


        self.conv1 = nn.Conv2d(channels, 16, 3, stride=2, padding=0)
        self.fc1 = nn.Linear(3600,latent_dims)
        self.fc2 = nn.Linear(latent_dims,3600)
        self.conv_t3 = nn.ConvTranspose2d(16, channels, 3, stride=2, padding=0, output_padding=1)

     
        self.kl = torch.zeros(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        temp_shape = x.shape
        x = torch.flatten(x, start_dim=1)
        z = self.fc1(x)
        z = self.fc2(z)
        z = z.reshape(temp_shape) 
        z = F.sigmoid(self.conv_t3(z))
        return z




class encoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super().__init__()
        self.dim = 200
        self.linear1 = nn.Linear(input_dims, self.dim)
        self.linear2 = nn.Linear(self.dim, latent_dims)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        z =  self.linear2(x)
        return z

class decoder(nn.Module):
    def __init__(self, latent_dims, input_dims):
        super().__init__()
        self.dim = 200
        self.linear1dec = nn.Linear(latent_dims, self.dim)
        self.linear2dec = nn.Linear(self.dim, input_dims)

    def forward(self, x):
        z = F.relu(self.linear1dec(x))
        z = self.linear2dec(z)
        return z

class Autoencoder(nn.Module):
    
    def __init__(self,  input_dims, latent_dims):
        super(Autoencoder, self).__init__()
        self.input_dims = input_dims
        self.linear_b = nn.Linear(input_dims, latent_dims)

        self.lineardec_b = nn.Linear(latent_dims, input_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = torch.zeros(1)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear_b(x)
        z = self.lineardec_b(x)
        return z.reshape((-1, 1, int(np.sqrt(self.input_dims)), int(np.sqrt(self.input_dims))))




class Autoencoder_gen(nn.Module):
    def __init__(self,  input_dims, latent_dims):
        super(Autoencoder_gen, self).__init__()
 
        self.lineardec_b = nn.Linear(latent_dims, input_dims)

        self.N = torch.distributions.Normal(0, 1)

        self.kl = torch.zeros(1)
    
    def forward(self, x):

        z = F.sigmoid(self.lineardec_b(x))
        return z.reshape((-1, 1, 64, 64))
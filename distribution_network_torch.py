import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor


epochs = 500
batch_size = 50
learning_rate = 0.0003
display_step = 50




class DistributionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 12)
        self.linear2 = nn.Linear(12, 12)
        self.linear3 = nn.Linear(12, 12)
        self.linear4 = nn.Linear(12, 12)
        self.mu = nn.Linear(12, 1)
        self.sigma = nn.ELU(12, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, action):
        action = action.transpose(1,0)
        x = torch.cat((state.to(torch.float32), action.to(torch.float32)), 1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        mu = self.mu(x)
        sigma = self.sigma(x)+1
        return mu, sigma

    def mdn_cost(self, mu, sigma, y):
        dist = Normal(torch.Tensor(mu).squeeze(), torch.Tensor(sigma).squeeze())
        return -dist.log_prob(y)

    def optimize(self,x):
        input = Tensor(input).to(device)
        self.zero_grad()
        mu_out, sigma_out = self.forward(input)
        loss = self.mdn_cost(mu_out,sigma_out, x)
        loss.backward()

def f(x):
    return x**2-6*x+9


def data_generator(x,sigma_0,samples):
    return np.random.normal(f(x),sigma_0*x,samples)

sigma_0 = 0.1
x_vals = np.arange(1,5.2,0.2)
x_arr = np.array([])
y_arr = np.array([])
samples = 50
for x in x_vals:
    #Take the previous array and stack a (50,) array with values x
    x_arr = np.append(x_arr, np.full(samples,x))
    # Take the previous array and stack a (50,) array with values y_generated
    y_arr = np.append(y_arr, data_generator(x,sigma_0,samples))

x_arr, y_arr = shuffle(x_arr, y_arr)
x_test = np.arange(1.1,5.1,0.2)

batch_num = int(len(x_arr) / batch_size)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")


###### SOURCE DISTRIBUTION NETWORK #######
class DistributionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 12)
        self.linear2 = nn.Linear(12, 12)
        self.linear3 = nn.Linear(12, 12)
        self.linear4 = nn.Linear(12, 12)
        self.mu = nn.Linear(12, 1)
        self.sigma = nn.Linear(12, 1)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), learning_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        mu = self.mu(self.tanh(x))
        sigma = self.sigma(self.elu(x) + 1)
        return mu, sigma

    def mdn_cost(self, mu, sigma, y):
        dist = Normal(mu, sigma)
        return -dist.log_prob(y)

    def optimize(self, x_sample, y_sample):
        self.zero_grad()
        x_sample = x_sample.to(device)
        y_sample = y_sample.to(device)
        mu_out, sigma_out = self.forward(x_sample)
        loss = self.mdn_cost(mu_out, sigma_out, y_sample)
        loss.backward()
        self.optimizer.step()
##########################################

###### HYPERPARAMETERS #######
epochs = 500
batch_size = 50
learning_rate = 0.0003
display_step = 50
sigma_0 = 0.1
x_vals = np.arange(1,5.2,0.2)
x_arr = np.array([])
y_arr = np.array([])
samples = 50

##############################

###### DATA GENERATION #######
def f(x):
    return x**2-6*x+9
def data_generator(x,sigma_0,samples):
    return np.random.normal(f(x),sigma_0*x,samples)

for x in x_vals:
    #Take the previous array and stack a (50,) array with values x
    x_arr = np.append(x_arr, np.full(samples,x))
    # Take the previous array and stack a (50,) array with values y_generated
    y_arr = np.append(y_arr, data_generator(x,sigma_0,samples))
x_arr, y_arr = shuffle(x_arr, y_arr)
x_test = np.arange(1.1,5.1,0.2)
batch_num = int(len(x_arr) / batch_size)
x_batches = np.array_split(x_arr, batch_num)
x_batches = torch.tensor(x_batches)
y_batches = np.array_split(y_arr, batch_num)
y_batches = torch.tensor(y_batches)

##############################

###### TRAINING #######
source_distribution = DistributionNetwork()
source_distribution = source_distribution.to(device)

for epoch in range(epochs):
    avg_cost = 0.0
    #x_batches, y_batches = shuffle(x_batches, y_batches)
    for i in range(batch_num):
        x_batch = x_batches[i]
        y_batch = y_batches[i]
        for x_sample, y_sample in x_batch, y_batch:x_sample, y_sample in zip(range(x), range(y))
            source_distribution.optimize(x_sample.to(torch.float32), y_sample.to(torch.float32))
            avg_cost += batch_num
        if epoch % display_step == 0:
            print('Epoch {0} | cost = {1:.4f}'.format(epoch, avg_cost))
    print('Final cost: {0:.4f}'.format(avg_cost))
#######################

'''
env = gym.make('Pendulum-v0')


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

num_episodes = 10

steps_total = []

for i_episode in range(num_episodes):

    state = env.reset()

    step = 0
        while True:
            step += 1
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)

            if done:
                steps_total.append(step)
                print("Episode finished after %i steps" % step)
                break
env.render()

print("Average reward: %.2f" % (sum(steps_total) / num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:]) / 100))

plt.figure(figsize=(12, 5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()
'''
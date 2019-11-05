import gym
import torch
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

#If gpu is to be used
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor

env = gym.make('CartPole-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

#### PARAMETERS ####
learning_rate = 0.01
num_episodes = 1000
gamma = 0.99
####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs =env.action_space.n

def calculate_epsilon(steps_done):
        epsilon = egreedy_final + (egreedy-egreedy_final)*\math.exp(-1. * steps_done / egreedy_decay)
        return epsilon

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(number_of_inputs , number_of_outputs)

    def forward(self, x):
        output = self.linear1(x)
        return output


class QNet_Agent(object):

    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss()
        #loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=nn.parameters(), lr=learning_rate)

    def select_action(self, state):
        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > egreedy:
            with torch.no_grad()
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
        else:
            action = env.action_space.sample()
        return action

    def optimize(self, state,action,new_state,reward,done):
        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = Tensor([reward]).to(device)
        if done:
            target_value = reward
        else:
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            target_value = reward + gamma * max_new_state_values
        predicted_value = self.nn(state)[action]
        loss = self.loss_func(predicted_value,target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

qnet_agent= QNet_Agent
steps_total = []

for i_episode in range(num_episodes):

    state = env.reset()

    step = 0
    # for step in range(100):
    while True:

        step += 1

        #action = env.action_space.sample()
        action = qnet_agent.select_action(state)
        new_state, reward, done, info = env.step(action)
        qnet_agent.optimize(state,action,new_state,reward,done)
        state = new_state

        # env.render()

        if done:
            steps_total.append(step)
            print("Episode finished after %i steps" % step)
            break


print("Average reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()
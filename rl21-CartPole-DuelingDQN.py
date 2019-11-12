#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Dueling DQN implementation
Tuning - version stable - last 100 episodes - no reward lost

@author: udemy
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import math
import time

import matplotlib.pyplot as plt

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

env = gym.make('CartPole-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.001
learning_rate_fwd = 0.001
learning_rate_source = 0.001
learning_rate_planning = 0.001

num_episodes = 500
gamma = 0.9999

hidden_layer = 64

replay_mem_size = 50000
batch_size = 32

double_dqn = True

update_target_frequency = 1500

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500

report_interval = 10
score_to_solve = 195

clip_error = False

####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = ( self.position + 1 ) % self.capacity
        
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))
        
        
    def __len__(self):
        return len(self.memory)
        

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,hidden_layer)
        
        self.advantage = nn.Linear(hidden_layer,number_of_outputs)
        self.value = nn.Linear(hidden_layer,1)

        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()
        
        
    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        
        output_advantage = self.advantage(output1)
        output_value = self.value(output1)
        
        output_final = output_value + output_advantage - output_advantage.mean()
        
        return output_final

class ForwardDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(5, 64)
        self.linear2 = nn.Linear(64, 4)

        self.criterion = nn.MSELoss().to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate_fwd)


    def forward(self, state, action):
        action = action.transpose(1,0)
        x = torch.cat((state.to(torch.float32), action.to(torch.float32)), 1)
        x = self.linear1(x)
        output = self.linear2(x)
        return output

    def optimize(self):

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        action = LongTensor(action).to(device)

        self.zero_grad()
        output = self.forward(state, action.unsqueeze(0).to(device))
        loss = self.criterion(output, new_state)
        loss.backward()
        self.optimizer.step()


class SourceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 2)
        self.soft_max = nn.Softmax()
        #self.sigmoid = nn.Sigmoid()

        self.criterion = nn.MSELoss().to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate_fwd)

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        return self.soft_max(x)

    def optimize(self):
        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = Tensor(state).to(device)

        self.zero_grad()
        output = self.forward(state)
        loss = self.criterion(output, new_state)
        loss.backward()
        self.optimizer.step()


# class PlanningNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
#
#     def forward(self, state, action):
#
#
#     def optimize(self):
#         if (len(memory) < batch_size):
#             return
#
#         state, action, new_state, reward, done = memory.sample(batch_size)
#
#         state = Tensor(state).to(device)
#         new_state = Tensor(new_state).to(device)
#         reward = Tensor(reward).to(device)
#         action = LongTensor(action).to(device)
#         done = Tensor(done).to(device)

    
class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.update_target_counter = 0
        
    def select_action(self,state,epsilon):
        
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self):
        
        if (len(memory) < batch_size):
            return
        
        state, action, new_state, reward, done = memory.sample(batch_size)
        
        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]  
            
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]
        
        
        target_value = reward + ( 1 - done ) * gamma * max_new_state_values
  
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        loss = self.loss_func(predicted_value, target_value)
    
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)
        
        self.optimizer.step()
        
        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        self.update_target_counter += 1
        
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])

        
        

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()
fwd_model = ForwardDynamics().to(device)

steps_total = []

frames_total = 0 
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    #for step in range(100):
    while True:
        
        step += 1
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)

        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        fwd_model.optimize()

        state = new_state
        
        if done:
            steps_total.append(step)
            
            mean_reward_100 = sum(steps_total[-100:])/100
            
            if (mean_reward_100 > score_to_solve and solved == False):
                print("SOLVED! After %i episodes " % i_episode)
                solved_after = i_episode
                solved = True
            
            if (i_episode % report_interval == 0):
                
                print("\n*** Episode %i *** \
                      \nAv.reward: [last %i]: %.2f, [last 100]: %.2f, [all]: %.2f \
                      \nepsilon: %.2f, frames_total: %i" 
                  % 
                  ( i_episode,
                    report_interval,
                    sum(steps_total[-report_interval:])/report_interval,
                    mean_reward_100,
                    sum(steps_total)/len(steps_total),
                    epsilon,
                    frames_total
                          ) 
                  )
                  
                elapsed_time = time.time() - start_time
                print("Elapsed time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


            break
        if i_episode > 490:
            env.render()
print("\n\n\n\nAverage reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))
if solved:
    print("Solved after %i episodes" % solved_after)
plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

env.close()
env.env.close()

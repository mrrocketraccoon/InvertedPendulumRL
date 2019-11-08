import gym
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

#If gpu is to be used
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor
env = gym.make('CartPole-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

#### PARAMETERS ####
learning_rate = 0.01
num_episodes = 500
gamma = 0.99

hidden_layer = 64

replay_mem_size = 50000
batch_size = 3
egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500

report_interval = 10
score_to_solve = 195
####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs =env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon
class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        #memory storage
        self.memory = []
        #to track entries pushed into memory
        self.position = 0
    #to push all entries, all needed/received information from the environment
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position=(self.position+1)%self.capacity

    #it will sample some random entries from the memory
    def sample(self, batch_size):
        #to separate into state batch, action batch, etc. zip(*)
        return zip(*random.sample(self.memory, batch_size))

    #to verify current memory size
    def __len__(self):
        return len(self.memory)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, number_of_outputs)
        self.activation = nn.Tanh()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)
        return output2


class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.loss_func = nn.MSELoss()
        #loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > epsilon:
            #Better than the detach method. Won't execute back prop, so it won't even start to build
            #the graph for the gradient, a better memory-efficient solution.
            with torch.no_grad():
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = env.action_space.sample()
        return action

    def optimize(self):
        if (len(memory)<batch_size):
            return
        state, action, new_state, reward, done = memory.sample(batch_size)

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        #reward must be in brackets because it's single value
        #otherwise you get a 5-entry tensor.
        ####----Update, since we're using batches now, we don't need the brackets anymore----####
        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        #Get max value of all possible states, add detach() because we want to have the values
        #but we also don't want to calculate the gradient and update it's network parameters here.
        new_state_values = self.nn(new_state).detach()
        #cause we wanna get max value for each row we add 1
        max_new_state_values = torch.max(new_state_values,1)[0]
        #Q-Learning equation
        ####----Update: since we're using tensors now we perform (1-done) to stop computations if done flag is set
        target_value = reward + (1-done)*gamma * max_new_state_values
        #This time we call nn without detach because we want to calculate the gradient and let the agent
        #learn from its predictions.
        predicted_value = self.nn(state)[action]

        #print(self.nn(state))
        #tensor([[0.0212, 0.1090],
        #        [0.0471, -0.0182],
        #        [0.0459, -0.0123]], device='cuda:0', grad_fn= < AddmmBackward >)
        #print(action.unsqueeze(1))
        #tensor([[0],
        #[1],
        #[0]], device='cuda:0')
        #print(self.nn(state).gather(1, action.unsqueeze(1)))
        #tensor([[0.0212],
        #        [-0.0182],
        #        [0.0459]], device='cuda:0', grad_fn= < GatherBackward >)

        loss = self.loss_func(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

memory = ExperienceReplay(replay_mem_size)
qnet_agent= QNet_Agent()
steps_total = []
frames_total = 0
solved_after = 0
solved = False
start_time = time.time()
for i_episode in range(num_episodes):

    state = env.reset()

    step = 0
    # for step in range(100):
    while True:

        step += 1
        frames_total += 1
        epsilon = calculate_epsilon(frames_total)
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)
        #first collect actions into memory and then pull it from there
        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        state = new_state

        # env.render()

        if done:
            steps_total.append(step)
            mean_reward_100 = sum(steps_total[-100:])/100
            if(mean_reward_100>score_to_solve and solved == False):
                print("Solved after %i episodes" % i_episode)
                solved_after = i_episode
                solved = True
            if (i_episode%10 ==0):
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
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            break
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
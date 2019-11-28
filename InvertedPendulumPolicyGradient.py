import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from gym.wrappers import Monitor



def wrap_env(env):
  env = Monitor(env, './video', video_callable=lambda episode_id: episode_id%999==0, force=True)
  return env

env = wrap_env(gym.make('CartPole-v0'))

# Constants
GAMMA = 0.9
###########

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):

        #unsqueeztakes you from this [-0.00417724 -0.40476312  0.06895009  0.69969184] to this
        # [[-0.0042, -0.4048,  0.0690,  0.6997]], torch.from_numpy transforms it into a tensor
        state = torch.from_numpy(state).float().unsqueeze(0)

        #forward(Variable(state)) delivers a 2-d prob. dist. [[0.4980, 0.5020]].
        probs = self.forward(Variable(state))
        #probs.detach().numpy() transforms probs to numpy
        #np.squeeze(probs.detach().numpy()) takes you from [[0.5653084 0.4346916]] to [0.5653084 0.4346916]
        #np.random.choice(actions, p) gives you a non-uniform random sample from the actions array
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        #torch.log(probs.squeeze(0)[highest_prob_action]) takes probs, unsqueezes it, searches for highest
        #probability and computes its logarithm. --> highest_prob_action comes from the gradient.
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_prob


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    #### Compute Gt term and append it to a list####
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
    ####################################

    ###transform to tensor and normalize the rewards tensor
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    policy_network.optimizer.zero_grad() ###clears gradients of optimized tensor
     # takes list of  tensors tensor(0.8492, grad_fn=<MulBackward0>), stacks them on top of each other
    # and retrieves single tensor(0.1396, grad_fn=<SumBackward0>) with sum of list elements.
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

max_episode_num = 5000
max_steps = 10000
numsteps = []
avg_numsteps = []
all_rewards = []

for episode in range(max_episode_num):
    state = env.reset()
    log_probs = []
    rewards = []

    for steps in range(max_steps):
        action, log_prob = policy_net.get_action(state) #retrieve best action and log_prob from policy
        new_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            update_policy(policy_net, rewards, log_probs)
            numsteps.append(steps)
            avg_numsteps.append(np.mean(numsteps[-10:]))
            all_rewards.append(np.sum(rewards))
            sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                                                                                                          np.round(
                                                                                                              np.sum(
                                                                                                                  rewards),
                                                                                                              decimals=3),
                                                                                                          np.round(
                                                                                                              np.mean(
                                                                                                                  all_rewards[
                                                                                                                  -10:]),
                                                                                                              decimals=3),
                                                                                                          steps))
            break
        state = new_state

env.render()

plt.plot(numsteps)
plt.plot(avg_numsteps)
plt.xlabel('Episode')
plt.show()

env.close()
env.env.close()
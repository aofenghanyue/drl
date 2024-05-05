import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# actor network
# 2 Linear layers with ReLU activation ending with a tanh activation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# critic network
# 2 Linear layers with ReLU activation ending with no activation
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# replay buffer
# deque with 15000 max length
# sample, push, and __len__ methods
class ReplayBuffer:
    def __init__(self, max_size=15000):
        self.buffer = deque(maxlen=max_size)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

# DDPG agent
# get_action, update, and __init__ methods
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=0.0001, critic_lr=0.001, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer()

        self.gamma = gamma
        self.tau = tau

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()
        return action
    
    def update(self, BATCH_SIZE=64):
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # update critic
        next_actions = self.actor_target(next_states)
        target_values = rewards + self.gamma * self.critic_target(next_states, next_actions) * (1 - dones)
        values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(values, target_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
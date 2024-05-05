# Description: Main file for running the DDPG algorithm
import torch
import gym
from ddpg import DDPGAgent
import numpy as np

# initialize environment
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

# hyperparameters
MAX_EPISODES = 200
MAX_STEPS = 200
BATCH_SIZE = 64

# training loop
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM) # [-1,1]
        next_state, reward, done, _, _ = env.step(2*action)
        agent.memory.push(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)

        if done:
            break

    print(f'Episode: {episode+1}, Reward: {episode_reward}')

env.close()

# save model
torch.save(agent.actor.state_dict(), 'actor.pth')
torch.save(agent.critic.state_dict(), 'critic.pth')

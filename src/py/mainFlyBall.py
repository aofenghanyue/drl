import torch
import os
os.add_dll_directory(os.environ['minGWPath'])

from build import flyBall as env
from ddpg import DDPGAgent
import numpy as np

DEBUG_MODE = False

# initialize environment
env.reset()
STATE_DIM = 7
ACTION_DIM = 1

# initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM, hidden_dim=STATE_DIM**3)

# create save path
model_save_dir = os.environ['MODEL_PATH']
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# hyperparameters
MAX_EPISODES = 2 if DEBUG_MODE else 2000
MAX_STEPS = 101
BATCH_SIZE = 64
model_save_period = 500

# training loop
for episode in range(MAX_EPISODES):
    env.reset()
    state = env.state()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM) # [-1,1]
        env.step(12*np.clip(action, -1, 1))
        next_state = env.state()
        reward = env.reward()
        done = env.done()
        agent.memory.push(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        if len(agent.memory) > BATCH_SIZE:
            agent.update(BATCH_SIZE)

        if DEBUG_MODE:
            print(f'Step: {step+1}, State: {state[:4]}, Action: {action}, Reward: {reward}, done: {done}')

        if done:
            break

    print(f'Episode: {episode+1}, Reward: {episode_reward}')
    # save model
    if (episode+1) % model_save_period == 0:
        torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, f'{episode+1}actor.pth'))
        torch.save(agent.critic.state_dict(), os.path.join(model_save_dir, f'{episode+1}critic.pth'))






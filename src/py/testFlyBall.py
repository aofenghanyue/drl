import os
os.add_dll_directory(os.environ['minGWPath'])

from build import flyBall as env
from ddpg import Actor
import torch
import matplotlib.pyplot as plt
import numpy as np

# initialize environment
env.reset()

# load actor
model_path = os.environ['MODEL_PATH']
actor = Actor(7, 1, hidden_dim=7**3)
actor.load_state_dict(torch.load(os.path.join(model_path, '2000actor.pth')))

# initialize store lists
states = []
actions = []
rewards = []

# run for 3 episodes
episodes = 1
for episode in range(episodes):
    states = []
    actions = []
    rewards = []
    env.reset()
    while not env.done():
        # state:[t, y, vy, ay, target_t, target_y, target_vy]
        state = env.get_real_state()
        state_norm = env.state()
        action = actor(torch.tensor(state_norm, dtype=torch.float32)).detach().numpy()
        reward = env.reward()

        env.step(12*action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        print(f"Episode: {episode}, State: {state[:4]}, Action: {action}, Reward: {reward}")
    print(f"Episode: {episode}, Total Reward: {sum(rewards)}")

states = np.array(states)
actions = np.array(actions)

# plot curves: y-t, vy-t, action-t
fig, ax = plt.subplots(3, 1)
ax[0].plot(states[:, 0], states[:, 1], label='y-t')
ax[0].set_xlabel('t')
ax[0].set_ylabel('y')
ax[0].legend()

ax[1].plot(states[:, 0], states[:, 2], label='vy-t')
ax[1].set_xlabel('t')
ax[1].set_ylabel('vy')
ax[1].legend()

ax[2].plot(states[:, 0], actions, label='action-t')
ax[2].set_xlabel('t')
ax[2].set_ylabel('action')
ax[2].legend()

plt.show()

    
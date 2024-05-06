import gym
import torch
import numpy as np
from ddpg import Actor
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize environment
env = gym.make('Pendulum-v1', render_mode='rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# load actor model
actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load('actor.pth'))

# initialize pygame
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption('Pendulum-v1')
clock = pygame.time.Clock()

# run environment
EPISODES = 5
MAX_STEPS = 200
for episode in range(EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        
        frame = env.render()
        frame = np.transpose(frame,(1,0,2))
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (500,500))
        screen.blit(frame, (0,0))
        pygame.display.flip()
        
        clock.tick(60)

        state = torch.FloatTensor(state).to(device)
        action = actor(state).detach().cpu().numpy()
        next_state, reward, done, _, _ = env.step(2*action)
        episode_reward += reward
        state = next_state

        if done:
            break

    print(f'Episode: {episode+1}, Reward: {episode_reward}')


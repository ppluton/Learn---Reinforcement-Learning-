import re

import gymnasium as gym
from stable_baselines3 import PPO, A2C

env = gym.make("LunarLander-v3")
env.reset()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10_000)

episodes = 10

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated

env.close()

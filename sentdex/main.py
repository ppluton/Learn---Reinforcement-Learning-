import gymnasium as gym
from stable_baselines3 import PPO

# Entraînement
env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
env.close()

# 3 tests visuels
eval_env = gym.make("LunarLander-v3", render_mode="human")

for ep in range(3):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    resultat = "Atterri !" if terminated and total_reward > 0 else "Crash"
    print(f"Test {ep + 1}/3 — Score : {total_reward:.0f} — {resultat}")

eval_env.close()

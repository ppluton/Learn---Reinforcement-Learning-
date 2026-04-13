# tuto_baseline_rl

Tutoriel stable-baselines3 avec gymnasium et PyTorch.

## Stack

- Python 3.11, uv
- `stable-baselines3==2.8.0` — algorithmes RL (PPO, A2C, SAC…)
- `gymnasium[box2d]==1.2.3` — environnements (LunarLander-v3, etc.)
- `torch==2.11.0`

## Commandes

```bash
uv run python main.py   # entraîner + évaluer
```

## Notes API gymnasium (breaking vs gym)

- `env.reset()` → retourne `(obs, info)` et non juste `obs`
- `env.step()` → retourne `(obs, reward, terminated, truncated, info)` et non `(obs, reward, done, info)`
- `done = terminated or truncated`
- `render_mode` se passe à `gym.make(...)`, pas à `env.render()`
- LunarLander : `v3` dans gymnasium (était `v2` dans gym)

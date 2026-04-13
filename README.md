# Reinforcement Learning — Notes & Exercices

Apprentissage du Reinforcement Learning avec Gymnasium, Stable-Baselines3 et PyTorch.

## Stack

| Outil | Version | Rôle |
|-------|---------|------|
| Python | 3.11 | Runtime |
| [Gymnasium](https://gymnasium.farama.org/) | 1.2.3 | Environnements RL (CartPole, FrozenLake, LunarLander...) |
| [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | 2.8.0 | Algorithmes pré-implémentés (PPO, A2C, DQN, SAC...) |
| [PyTorch](https://pytorch.org/) | 2.11.0 | Réseaux de neurones (DQN from scratch) |
| uv | — | Gestion des dépendances |

## Structure

```
.
├── sentdex/                          # Tuto pythonprogramming.net (sentdex)
│   ├── 00_fiche_algorithmes_rl.ipynb     # Fiche de référence : algorithmes RL
│   ├── 00_fiche_stable_baselines3.ipynb  # Fiche de référence : API SB3
│   ├── 01_introduction_rl.ipynb          # Intro RL, A2C, PPO sur LunarLander
│   ├── 02_save_load_tensorboard.ipynb    # Checkpoints et TensorBoard
│   ├── 03_custom_environment.ipynb       # Créer un environnement custom
│   ├── 04_reward_engineering.ipynb       # Reward shaping
│   ├── rl_intro.ipynb                    # Comparaison PPO / A2C / DQN
│   └── main.py                           # Script d'entraînement autonome
│
└── OC/                               # Exercices OpenClassrooms
    ├── 1. introduction a RL (simple)/
    │   └── 01_intro_rl_cartpole.ipynb    # Espaces obs/action, agent aléatoire
    ├── 2. q-learning/
    │   └── 05_q_learning_frozenlake.ipynb # Q-Learning from scratch, FrozenLake-v1
    └── 3. exo 3/
        └── 03_dqn_cartpole.ipynb         # DQN PyTorch + DQN SB3, CartPole-v1
```

## Notebooks Sentdex

Série de notebooks suivant le tutoriel [pythonprogramming.net](https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/) sur Stable-Baselines3, appliqués à LunarLander-v3.

### `00_fiche_algorithmes_rl.ipynb` — Fiche de référence : algorithmes
Tableau comparatif des algorithmes SB3 (PPO, A2C, DQN, SAC, DDPG, TD3) : compatibilité avec les espaces discrets/continus, caractéristiques principales et cas d'usage recommandés.

### `00_fiche_stable_baselines3.ipynb` — Fiche de référence : API SB3
Mémo de l'API Stable-Baselines3 : instanciation, entraînement, évaluation, sauvegarde/chargement, callbacks et intégration TensorBoard.

### `01_introduction_rl.ipynb` — Introduction au RL
Concepts fondamentaux (agent, environnement, reward), exploration de LunarLander-v3 (espace d'observation 8D, 4 actions), agent aléatoire comme baseline, premier entraînement A2C puis PPO, évaluation avec `evaluate_policy`, visualisation avec `render_mode='human'`.

### `02_save_load_tensorboard.ipynb` — Checkpoints et TensorBoard
Entraînement par tranches avec sauvegarde périodique (`model.save`), reprise d'entraînement (`reset_num_timesteps=False`), chargement de checkpoints (`PPO.load`), comparaison de performances à différents stades, visualisation des courbes dans TensorBoard (`tensorboard --logdir=logs`).

### `03_custom_environment.ipynb` — Environnement custom
Implémentation d'un environnement de navigation sur grille 10×10 héritant de `gym.Env` : définition des `observation_space` / `action_space`, méthodes `reset` et `step`, validation avec `check_env`. Expérimentation sur la conception des observations (coordonnées absolues vs vecteur delta).

### `04_reward_engineering.ipynb` — Reward shaping
Comparaison de 4 stratégies de récompense sur le même environnement de navigation : récompense binaire, pénalité de survie, pénalité de distance naïve (échoue : incite l'agent à terminer rapidement), distance avec offset positif (guidage progressif vers la cible). Principes généraux du reward engineering.

### `rl_intro.ipynb` — Comparaison PPO / A2C / DQN
Entraînement en parallèle des trois algorithmes sur LunarLander-v3 (300 000 steps), évaluation sur 10 épisodes et visualisation comparative sous forme de graphique en barres.

### `main.py` — Script autonome
Script minimal : entraîne PPO sur LunarLander-v3 (100 000 steps) puis lance 3 épisodes de visualisation avec `render_mode='human'`.

```bash
uv run python sentdex/main.py
```

## Exercices OC

### 1. Introduction au RL — CartPole-v1
Exploration des briques fondamentales : espace d'observation (`Box`), espace d'action (`Discrete`), boucle `reset` / `step`, agent aléatoire comme baseline.

### 2. Q-Learning — FrozenLake-v1
Implémentation from scratch d'une Q-table avec la stratégie epsilon-greedy et la mise à jour de Bellman. Évaluation du taux de réussite après entraînement.

### 3. DQN — CartPole-v1
Passage de la Q-table au réseau de neurones. Implémentation PyTorch complète : classe `DQN` (`nn.Module`), `ReplayBuffer` (experience replay), target network, boucle d'entraînement. Puis reproduction en 5 lignes avec SB3.

## Lancer un notebook

```bash
uv run jupyter lab
```

## Références

- [Tuto Sentdex — Stable-Baselines3](https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/)
- [Documentation Gymnasium](https://gymnasium.farama.org/)
- [Documentation Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Tuto DQN — PyTorch officiel](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Article DQN original — DeepMind (2013)](https://arxiv.org/abs/1312.5602)

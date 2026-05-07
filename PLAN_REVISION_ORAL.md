# Plan de révision — Oral RL Eagle-1
**Durée totale : 1h30**

> Objectif : pouvoir expliquer le Reinforcement Learning, la progression entre les 3 exercices OC (CartPole aléatoire → Q-Learning FrozenLake → DQN CartPole), les différences entre les méthodes, et tout le vocabulaire associé.

---

## Découpage temporel

| Phase | Durée | Sujet |
|-------|-------|-------|
| 1 | 15 min | Fondamentaux du RL (vocabulaire de base) |
| 2 | 10 min | Exercice 1 — CartPole + agent aléatoire |
| 3 | 20 min | Exercice 2 — Q-Learning sur FrozenLake |
| 4 | 25 min | Exercice 3 — DQN sur CartPole |
| 5 | 15 min | Comparaisons & lien avec la mission Eagle-1 |
| 6 | 5 min | Questions probables à l'oral + cheat sheet final |

---

# PHASE 1 — Fondamentaux du RL (15 min)

## 1.1 Le pitch en une phrase
Le **Reinforcement Learning (RL)** est un paradigme d'apprentissage où un **agent** apprend à prendre de bonnes décisions par **essai-erreur**, en interagissant avec un **environnement** et en étant guidé uniquement par un **signal de récompense**. Pas de "bonne réponse" fournie à l'avance — l'agent doit la découvrir.

## 1.2 RL vs autres types de Machine Learning

| Paradigme | Données | Exemple |
|---|---|---|
| **Supervisé** | (x, y) — couples entrée/sortie étiquetés | Classifier des emails spam/non-spam |
| **Non-supervisé** | x seuls — pas de label | Clustering de clients |
| **Renforcement** | Pas de label, juste un signal de récompense différé | Apprendre à jouer aux échecs |

**Différence clé** : en RL, la "bonne action" peut n'être révélée que **plusieurs steps plus tard** (récompense différée). On parle de **credit assignment problem** : à quelle action attribuer le mérite (ou la faute) ?

## 1.3 La boucle fondamentale

```
        ┌─────────────────────────────────┐
        ▼                                 │
   ┌─────────┐    action a_t      ┌──────────────┐
   │  AGENT  │ ──────────────────▶│ ENVIRONNEMENT│
   └─────────┘                     └──────────────┘
        ▲                                 │
        │   observation s_{t+1}, reward r_t│
        └─────────────────────────────────┘
```

À chaque **timestep** :
1. L'agent observe l'état `s_t`
2. Il choisit une action `a_t` selon sa **politique** π
3. L'environnement renvoie `s_{t+1}` et `r_t`
4. On boucle jusqu'à la fin de l'**épisode**

## 1.4 Vocabulaire — à connaître par cœur

| Terme | Définition courte |
|---|---|
| **Agent** | Le décideur (le réseau de neurones, la Q-table…) |
| **Environnement** | Le monde simulé qui répond aux actions |
| **État (state) / observation** | Ce que perçoit l'agent à un instant t |
| **Action** | Décision prise par l'agent |
| **Reward** | Signal scalaire de feedback (un nombre) |
| **Épisode** | Une partie complète, du `reset()` à la fin |
| **Timestep** | Un cycle observation→action→récompense |
| **Politique (policy) π** | Fonction état → action (la "stratégie" de l'agent) |
| **Trajectoire / rollout** | Suite (s₀, a₀, r₀, s₁, a₁, r₁, …) d'un épisode |
| **Return G_t** | Somme actualisée des récompenses futures : G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … |
| **Value V(s)** | Espérance de return depuis l'état s |
| **Q-value Q(s,a)** | Espérance de return si on prend a en s, puis qu'on suit π |
| **Discount factor γ** | ∈ [0,1] — pondère les récompenses futures (myopie ↔ vision long terme) |
| **Learning rate α** | Vitesse de mise à jour des estimations |
| **Epsilon ε** | Probabilité de choisir une action aléatoire (exploration) |
| **MDP** | Markov Decision Process — formalisme mathématique du RL |

## 1.5 Le dilemme exploration vs exploitation
- **Explorer** : tester des actions inconnues pour découvrir si elles sont meilleures
- **Exploiter** : utiliser ce qu'on sait déjà pour maximiser les rewards

C'est un compromis permanent. Trop explorer = on ne progresse pas. Trop exploiter = on reste bloqué dans un optimum local. **Solution standard : epsilon-greedy** (avec décroissance d'epsilon dans le temps).

## 1.6 Propriété de Markov
> "Le futur ne dépend que de l'état présent, pas du passé."

Formellement : P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, …) = P(s_{t+1} | s_t, a_t)

Tous les algos qu'on a vus supposent que l'environnement est un **MDP**.

---

# PHASE 2 — Exercice 1 : CartPole + agent aléatoire (10 min)

> **Fichier** : `OC/1. introduction a RL (simple)/01_intro_rl_cartpole.ipynb`
> **But** : comprendre l'API Gymnasium et établir une **baseline** avant de coder un vrai agent.

## 2.1 L'environnement CartPole-v1
Un chariot sur un rail, une perche en équilibre dessus. **Objectif** : empêcher la perche de tomber en poussant le chariot à gauche ou à droite.

- **Reward** : +1 à chaque step où la perche tient
- **Maximum** : 500 steps (timeout de CartPole-v1)
- **Fin** : perche tombée OU chariot sorti du rail OU 500 steps atteints

## 2.2 Les espaces — la distinction la plus importante du brief

| Type | Définition | Exemple CartPole |
|---|---|---|
| `Box(4,)` | Espace **continu** : vecteur de réels | Observation = [x, ẋ, θ, θ̇] (position, vitesse, angle, vitesse angulaire) |
| `Discrete(2)` | Espace **discret** : nb fini de choix | Action ∈ {0=gauche, 1=droite} |

C'est crucial parce que **certains algos ne marchent qu'avec l'un ou l'autre** :
- DQN → actions discrètes uniquement
- SAC, DDPG → actions continues uniquement
- PPO, A2C → les deux ✅

## 2.3 L'API Gymnasium (à connaître par cœur)

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()              # ← démarre un nouvel épisode

while not (terminated or truncated):
    action = env.action_space.sample()           # action aléatoire
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Les 5 valeurs de `step()`
| Valeur | Sens |
|---|---|
| `obs` | nouvel état |
| `reward` | récompense reçue |
| `terminated` | fin "logique" (perche tombée) |
| `truncated` | fin "technique" (timeout 500 steps) |
| `info` | dict de debug |

> **Piège classique de l'oral** : ancienne API `gym` retournait `(obs, reward, done, info)` — 4 valeurs avec `done = terminated or truncated`. La nouvelle API gymnasium les sépare. Idem `reset()` retourne `(obs, info)` et plus juste `obs`.

## 2.4 Ce que fait l'agent aléatoire
Il appelle `env.action_space.sample()` à chaque step. Sur 100 épisodes, il atteint en moyenne **~20-25 steps** avant que la perche tombe. C'est notre **baseline** : tout algo de RL doit faire **bien mieux** (objectif ≥ 450 sur CartPole).

## 2.5 Pourquoi un agent aléatoire avant tout ?
Trois raisons à expliquer à l'oral :
1. **Vérifier que l'environnement marche** (l'API, les rewards, les épisodes)
2. **Avoir une référence chiffrée** pour mesurer le progrès
3. **Comprendre la difficulté** du problème

---

# PHASE 3 — Exercice 2 : Q-Learning sur FrozenLake (20 min)

> **Fichier** : `OC/2. q-learning/05_q_learning_frozenlake.ipynb`
> **But** : implémenter from scratch un vrai algo de RL — le **Q-Learning**.

## 3.1 L'environnement FrozenLake-v1
Une grille 4×4. L'agent part de S, doit atteindre G en évitant les trous H.

```
S F F F
F H F H
F F F H
H F F G
```
- **16 états** (chaque case), **4 actions** (←↓→↑)
- **Reward** : 0 partout, **+1** uniquement si on atteint G
- **Mode `is_slippery=False`** : déterministe (l'agent va où il veut)
- **Mode `is_slippery=True`** : stochastique (l'agent peut glisser dans une autre direction)

## 3.2 Pourquoi FrozenLake et pas CartPole ?

| | CartPole | FrozenLake |
|---|---|---|
| Espace d'états | Continu (∞) | **Discret (16)** |
| Q-table possible ? | ❌ | ✅ |

Le Q-Learning **classique** ne fonctionne qu'avec un **nombre fini d'états**. C'est précisément pour ça que l'exercice 3 (DQN) introduira un réseau de neurones.

## 3.3 La Q-table — concept central
Tableau 2D de dimensions [n_states × n_actions] = [16 × 4] dans ce cas.

`Q(s, a)` = valeur estimée de prendre l'action `a` dans l'état `s`, c'est-à-dire **l'espérance de récompense cumulée future** si on agit optimalement ensuite.

Au départ : Q-table remplie de zéros. Après entraînement : chaque case `(état, action)` contient une valeur qui guide les décisions de l'agent (l'agent prend l'action de plus grande Q-value).

## 3.4 La stratégie epsilon-greedy
À chaque décision :

```python
if random() < epsilon:
    action = env.action_space.sample()      # EXPLORE
else:
    action = np.argmax(q_table[state, :])   # EXPLOITE
```

Avec **decay** (réduction progressive d'epsilon) : on commence à 100% d'exploration, on finit à 1%.

```
Épisode 1   → ε = 1.000  (100% aléatoire)
Épisode 500 → ε ≈ 0.082  (8% exploration)
Épisode 2000→ ε = 0.010  (plancher)
```

## 3.5 L'équation de Bellman — le cœur du Q-Learning

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Décomposition (très important pour l'oral)** :

| Terme | Signification |
|---|---|
| `Q(s, a)` | Valeur actuelle (ce qu'on pensait) |
| `r` | Récompense **réelle** reçue |
| `γ · max Q(s', a')` | Meilleur return possible **depuis le nouvel état** |
| `r + γ · max Q(s', a')` | **Cible** (target) — la "vraie" valeur estimée |
| `[cible − Q(s,a)]` | **Erreur TD (Temporal Difference)** |
| `α · erreur` | Correction partielle (on n'écrase pas tout) |

**Intuition** : à chaque pas, on rapproche notre estimation Q(s,a) de la "vraie" valeur observée, par petits pas (α).

## 3.6 Hyperparamètres et leur effet

| Hyperparamètre | Symbole | Valeur | Effet si on l'augmente |
|---|---|---|---|
| Learning rate | α | 0.8 | Apprend plus vite mais plus instable |
| Discount factor | γ | 0.95 | L'agent valorise plus le long terme |
| Epsilon initial | ε | 1.0 | Plus d'exploration au début |
| Epsilon decay | — | 0.995 | Décroissance plus lente d'ε |

**γ = 0** → agent **myope** (ne pense qu'au reward immédiat)
**γ → 1** → agent **prévoyant** (mais risque de ne pas converger)

## 3.7 La boucle d'entraînement (squelette à pouvoir réciter)

```
Pour chaque épisode :
    state ← env.reset()
    Tant que pas done :
        action ← epsilon-greedy(state)
        new_state, reward, done ← env.step(action)
        # Mise à jour Bellman :
        target = reward + γ · max(q_table[new_state])
        q_table[state, action] += α · (target - q_table[state, action])
        state ← new_state
    epsilon ← max(epsilon_min, epsilon · decay)
```

## 3.8 Évaluation — différence clé avec l'entraînement
Pendant l'éval :
- **Plus d'epsilon** : on prend toujours `argmax(q_table[state])`
- **Plus de mise à jour** : la Q-table est **figée**

C'est la phase où on mesure ce que l'agent a vraiment appris.

## 3.9 Limites du Q-Learning classique (TRÈS important pour la suite)

1. **Espaces d'états discrets et finis uniquement** → impossible avec CartPole (états continus)
2. **Pas de généralisation** : Q(s, a) est appris séparément pour chaque état
3. **Mémoire** : pour un Atari avec ~10²⁰ états, une Q-table est impensable

→ C'est pour ça qu'en exercice 3, on remplace la Q-table par un **réseau de neurones**.

---

# PHASE 4 — Exercice 3 : DQN sur CartPole (25 min)

> **Fichier** : `OC/3. exo 3/03_dqn_cartpole.ipynb`
> **But** : passer de la Q-table au **Deep Q-Network**, d'abord en PyTorch from scratch puis avec SB3.

## 4.1 L'idée fondamentale du DQN
On remplace la Q-table par un **réseau de neurones** qui prend un état en entrée et prédit `Q(s, a)` pour toutes les actions :

```
Q-TABLE (FrozenLake)         DQN (CartPole)
─────────────────            ────────────────────────────────
État entier (0..15)         État vecteur 4D
   ↓                            ↓
Tableau [16×4]              Réseau de neurones
   ↓                         (3 couches, ReLU)
Q-values                       ↓
                            [Q(s, gauche), Q(s, droite)]
```

Article fondateur : **Mnih et al., DeepMind 2013** (le DQN qui a appris Atari à partir des pixels).

## 4.2 Architecture du DQN

```
Entrée (4 obs) → Linear(4→128) → ReLU → Linear(128→128) → ReLU → Linear(128→2 actions)
```

**Pourquoi ReLU ?** Fonction d'activation `f(x) = max(0, x)`. Elle introduit la **non-linéarité** (sans elle, empiler des couches linéaires = une seule couche linéaire). Simple, rapide, évite les gradients qui disparaissent.

**Pourquoi pas d'activation sur la dernière couche ?** Les Q-values peuvent être **négatives**, donc on garde une sortie linéaire.

## 4.3 Les 3 innovations clés du DQN (à connaître impérativement)

### 1. Experience Replay (ReplayBuffer)
**Problème** : si on entraîne le réseau sur la transition courante uniquement, les données sont **fortement corrélées** (s_t, s_{t+1}, s_{t+2} se ressemblent). Le réseau **sur-apprend** sur la séquence récente et oublie ce qu'il avait appris.

**Solution** : on stocke toutes les transitions `(s, a, r, s', done)` dans un buffer (de taille fixe), et à chaque mise à jour on tire un **mini-batch aléatoire**. Le mélange casse la corrélation temporelle → apprentissage stable.

```python
self.buffer = collections.deque(maxlen=10_000)  # mémoire circulaire
```

### 2. Target Network
**Problème** : pour calculer la cible Bellman, on a besoin de `max Q(s', a')`. Si on utilise le **même réseau** pour calculer Q(s,a) ET la cible, **la cible bouge à chaque mise à jour** → instabilité (comme essayer d'attraper sa propre ombre).

**Solution** : maintenir **deux réseaux identiques en architecture** :

| Réseau | Rôle | Mis à jour |
|---|---|---|
| `policy_net` | Choisit les actions, calcule Q(s,a) | À chaque batch (gradient descent) |
| `target_net` | Calcule la cible Q(s', a') | **Périodiquement** (copie de policy_net tous les N épisodes) |

```python
target_net.load_state_dict(policy_net.state_dict())  # copie périodique
```

### 3. Epsilon-greedy (déjà vu en Q-Learning)
Même principe que le Q-Learning classique : tirage aléatoire avec proba ε, sinon exploitation. Décroissance progressive.

## 4.4 La fonction `optimize_model()` — étape par étape

```
1. Tirer un batch du ReplayBuffer
2. Calculer Q(s, a) avec policy_net           ← ce qu'on PRÉDIT
3. Calculer max Q(s', a') avec target_net     ← ce qu'on CIBLE
4. target = r + γ · max Q(s', a')             ← Bellman
5. loss = SmoothL1(Q prédit, target)          ← Huber loss
6. optimizer.zero_grad() + loss.backward() + optimizer.step()
```

### Pourquoi Huber loss (Smooth L1) plutôt que MSE ?
Plus **robuste aux outliers**. Comportement :
- Pour des petites erreurs : se comporte comme MSE (quadratique)
- Pour des grosses erreurs : se comporte comme MAE (linéaire) → ne fait pas exploser le gradient

### Gradient clipping
`torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)`
On limite la norme des gradients pour éviter qu'ils n'explosent et ne déstabilisent l'apprentissage.

## 4.5 La boucle d'entraînement DQN (squelette)

```
Pour chaque épisode :
    state ← env.reset()
    Tant que pas done :
        action ← epsilon-greedy(state) via policy_net
        next_state, reward, done ← env.step(action)
        memory.push(state, action, reward, next_state, done)
        state ← next_state
        optimize_model()                        # apprentissage à chaque step
    epsilon ← decay(epsilon)
    Si épisode % TARGET_UPDATE == 0 :
        target_net.load_state_dict(policy_net.state_dict())
```

## 4.6 Hyperparamètres DQN (typiques)

| Param | Valeur typique | Rôle |
|---|---|---|
| `BATCH_SIZE` | 128 | Taille du mini-batch tiré du buffer |
| `GAMMA` | 0.99 | Discount factor |
| `EPSILON_START` / `END` | 1.0 / 0.05 | Bornes de l'exploration |
| `LR` | 1e-4 | Learning rate (Adam) |
| `MEMORY_SIZE` | 10 000 | Capacité du ReplayBuffer |
| `TARGET_UPDATE` | 10 | Fréquence de copie target_net |

## 4.7 Phases typiques de la courbe d'apprentissage

| Phase | Épisodes | Comportement |
|---|---|---|
| **Exploration** | 0 → ~100 | ε élevé, agent erratique, rewards ~20-50 |
| **Transition** | ~100 → ~300 | ε décroît, le réseau apprend, rewards montent |
| **Exploitation** | ~300 → 500 | ε bas, l'agent converge vers ~400-500 |

## 4.8 La même chose en 5 lignes avec SB3

```python
from stable_baselines3 import DQN
env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25_000)
model.save("dqn_cartpole")
```

| | From scratch | SB3 |
|---|---|---|
| Lignes de code | ~150 | ~5 |
| Bugs potentiels | Nombreux | Quasi-nuls |
| Compréhension | Maximale | Limitée |
| Usage recommandé | Apprendre | Produire |

> **Argument à ressortir** : on fait les deux pour avoir le **meilleur des deux mondes** — comprendre les rouages, puis utiliser des outils éprouvés en pratique.

### Détail SB3 important : timesteps vs épisodes
SB3 raisonne en **timesteps**, pas en épisodes :
- 1 timestep = 1 appel à `env.step()`
- 25 000 timesteps ≈ 125 épisodes (sur CartPole bien entraîné)
- `MlpPolicy` = Multi-Layer Perceptron = bon pour observations vectorielles (vs `CnnPolicy` pour des images Atari)

---

# PHASE 5 — Comparaisons & lien avec la mission Eagle-1 (15 min)

## 5.1 La progression pédagogique des 3 exercices

```
EXO 1 — CartPole aléatoire        EXO 2 — Q-Learning              EXO 3 — DQN
────────────────────────         ──────────────────              ─────────────
Comprendre l'API et les          Premier vrai algo de RL,        Passage au Deep RL :
espaces (Box vs Discrete)        mais limité aux espaces         réseau de neurones,
+ établir une baseline            d'états discrets et finis       experience replay,
                                                                  target network
```

À l'oral, raconte ça comme une **histoire de complexification progressive** :
1. "D'abord j'ai voulu **comprendre la mécanique** de base (l'API, le cycle agent-environnement)."
2. "Ensuite j'ai implémenté un **vrai algo from scratch**, le Q-Learning, pour comprendre ce qu'il y a sous le capot — la Q-table, Bellman, epsilon-greedy."
3. "Enfin j'ai franchi le pas du **Deep RL** avec le DQN, pour gérer les espaces continus comme CartPole — et c'est ce qui prépare la mission Eagle-1 sur LunarLander."

## 5.2 Tableau comparatif des méthodes

| Critère | Random | Q-Learning | DQN | PPO / A2C |
|---|---|---|---|---|
| **Stocke Q via** | — | Q-table | Réseau de neurones | Politique paramétrique π |
| **Espace d'états** | tout | discret fini | continu OK | continu OK |
| **Espace d'actions** | tout | discret | discret | discret + continu |
| **Type** | — | value-based | value-based | policy-based / actor-critic |
| **Sample efficiency** | nulle | basse | moyenne | moyenne |
| **Stabilité** | — | bonne | moyenne | bonne |
| **Use case** | baseline | FrozenLake, Taxi | CartPole, Atari | LunarLander, MuJoCo |

## 5.3 Concepts à savoir distinguer (souvent demandés à l'oral)

### Value-based vs Policy-based vs Actor-Critic

| Famille | Idée | Exemple | Sortie du modèle |
|---|---|---|---|
| **Value-based** | On apprend Q(s,a), on déduit la politique en prenant argmax | Q-Learning, **DQN** | Q-values |
| **Policy-based** | On apprend directement la politique π(a∣s) | REINFORCE | Distribution de probas sur les actions |
| **Actor-Critic** | Hybride : un réseau "acteur" (politique) + un réseau "critique" (value) | A2C, **PPO**, SAC | Politique + valeur |

### On-policy vs Off-policy

| | On-policy | Off-policy |
|---|---|---|
| **Définition** | Apprend depuis les données générées par la politique **actuelle** | Peut apprendre depuis des données générées par **n'importe quelle** politique (ex: ancienne) |
| **Replay buffer** | ❌ (les données vieillissent vite) | ✅ |
| **Exemples** | A2C, PPO | **Q-Learning, DQN**, SAC |

→ **Le DQN est off-policy** précisément parce qu'il a un ReplayBuffer, donc apprend sur des transitions générées il y a longtemps avec une autre politique.

### Model-free vs Model-based
- **Model-free** : pas de modèle de l'environnement, on apprend par interaction directe (tous nos algos !)
- **Model-based** : on apprend aussi un modèle de la dynamique de l'environnement (ex: AlphaZero pour le go)

## 5.4 Pourquoi l'algo recommandé pour Eagle-1 est le DQN (puis PPO en alternative)

Le brief dit : "Vous vous en sortirez avec le DQN, mais testez PPO si vous cherchez une alternative plus optimisée."

**LunarLander-v2/v3** :
- Observation = `Box(8,)` — 8 valeurs continues (position x/y, vitesse, angle…)
- Actions = `Discrete(4)` — rien / propulseur gauche / principal / droit

→ Compatible DQN ✅ (actions discrètes) et compatible PPO ✅
→ **Critère de succès du brief** : reward moyen ≥ 200 sur 100 épisodes consécutifs

## 5.5 Les biais classiques à connaître pour l'oral

| Concept | Explication courte |
|---|---|
| **Overestimation bias du DQN** | Le `max` dans Bellman tend à sur-estimer les Q-values → solution : Double DQN |
| **Catastrophic forgetting** | Le réseau "oublie" les anciennes situations s'il ne les revoit plus → atténué par le replay buffer |
| **Sparse reward** | Reward très rare (FrozenLake : 1 fois en fin d'épisode) → apprentissage difficile |
| **Reward hacking** | L'agent trouve une faille dans la fonction de récompense → cf. notebook 04 reward engineering |

---

# PHASE 6 — Préparation à l'oral (5 min)

## 6.1 Questions probables

### Niveau 1 — Compréhension de base
1. **C'est quoi le RL ?** → Apprentissage par essai-erreur via reward. Pas de label, juste un signal scalaire.
2. **Différence entre observation et action ?** → Observation = ce que l'agent perçoit ; action = ce qu'il décide.
3. **Différence entre `Box` et `Discrete` ?** → Continu (vecteur de réels) vs discret (nb fini de valeurs).
4. **C'est quoi `terminated` vs `truncated` ?** → Fin logique (échec/succès) vs fin technique (timeout).

### Niveau 2 — Q-Learning
5. **Donne l'équation de Bellman.** → `Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') − Q(s,a)]`
6. **C'est quoi epsilon-greedy ?** → Stratégie qui choisit au hasard avec proba ε, sinon argmax.
7. **Pourquoi un epsilon decay ?** → Passer progressivement d'exploration à exploitation.
8. **C'est quoi γ et α ?** → γ = poids des rewards futurs ; α = vitesse d'apprentissage.
9. **Pourquoi le Q-Learning classique ne marche pas sur CartPole ?** → Espace d'états continu (infini), une Q-table ne peut pas le représenter.

### Niveau 3 — DQN
10. **Pourquoi un Replay Buffer ?** → Casser la corrélation temporelle entre transitions consécutives.
11. **Pourquoi un Target Network ?** → Stabiliser la cible Bellman ; sinon la cible bouge en même temps que le réseau (instable).
12. **À quelle fréquence on met à jour le target_net ?** → Périodiquement, ex tous les 10 épisodes (ou tous les 500 steps en SB3).
13. **Pourquoi Huber loss et pas MSE ?** → Plus robuste aux outliers, gradient stable même pour des grosses erreurs.
14. **C'est quoi le forward pass ?** → Le passage en avant : on pousse l'entrée à travers le réseau pour obtenir les Q-values.
15. **Pourquoi on hérite de `nn.Module` ?** → Gestion auto des paramètres, save/load, mode train/eval.

### Niveau 4 — Comparaisons
16. **DQN vs Q-Learning, différence principale ?** → DQN = Q-Learning + réseau de neurones + replay + target net. Permet les espaces continus.
17. **DQN vs PPO ?** → DQN = value-based, off-policy, actions discrètes seulement ; PPO = actor-critic, on-policy, supporte continu et discret.
18. **From scratch vs SB3 ?** → SB3 = ~5 lignes pour la même chose, code testé en prod ; from scratch = comprendre les rouages.
19. **Quel algo recommandé pour LunarLander ?** → DQN (actions discrètes, marche bien) ou PPO (plus optimisé).

### Niveau 5 — Pièges
20. **Si le DQN diverge, qu'est-ce qu'on fait ?** → Vérifier batch_size, target_update_interval, learning_rate (le baisser), gradient clipping, taille du buffer.
21. **C'est quoi l'erreur TD ?** → `cible − estimation actuelle = r + γ·max Q(s',a') − Q(s,a)`. C'est l'écart qu'on cherche à minimiser.

## 6.2 Cheat sheet final — à relire 5 min avant

```
╔══════════════════════════════════════════════════════════════════════════╗
║                  CHEAT SHEET — RL en 1 page                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  RL = agent apprend par essai-erreur via REWARD                          ║
║                                                                          ║
║  BOUCLE :  obs → action → reward + new_obs → ...                         ║
║                                                                          ║
║  GYMNASIUM API :                                                         ║
║    obs, info = env.reset()                                               ║
║    obs, reward, terminated, truncated, info = env.step(action)           ║
║                                                                          ║
║  ESPACES :                                                               ║
║    Box(n,)     = continu (vecteur de réels)                              ║
║    Discrete(n) = discret (n choix)                                       ║
║                                                                          ║
║  EXPLORATION vs EXPLOITATION → epsilon-greedy + decay                    ║
║                                                                          ║
║  ÉQUATION DE BELLMAN :                                                   ║
║    Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') − Q(s,a)]                    ║
║                                                                          ║
║  HYPERPARAMÈTRES :                                                       ║
║    α (alpha)   = learning rate                                           ║
║    γ (gamma)   = discount factor (0 myope → 1 prévoyant)                 ║
║    ε (epsilon) = proba exploration                                       ║
║                                                                          ║
║  Q-LEARNING : Q-table, espaces DISCRETS et FINIS uniquement              ║
║                                                                          ║
║  DQN = Q-Learning + ANN + 3 innovations :                                ║
║    1. Replay Buffer  → casse la corrélation temporelle                   ║
║    2. Target Network → stabilise la cible Bellman                        ║
║    3. ε-greedy       → exploration                                       ║
║                                                                          ║
║  ALGOS / FAMILLES :                                                      ║
║    Value-based  : Q-Learning, DQN          (off-policy)                  ║
║    Actor-Critic : A2C, PPO, SAC            (PPO=on-policy)               ║
║                                                                          ║
║  EAGLE-1 / LunarLander :                                                 ║
║    obs Box(8,), action Discrete(4)                                       ║
║    Critère : reward moyen ≥ 200 / 100 épisodes                           ║
║    Algos : DQN ✅ / PPO ✅                                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 6.3 Petits trucs pour l'oral
- Quand tu ne te souviens plus d'un mot, **reformule avec tes propres mots** : c'est mieux que de bafouiller un terme technique.
- Pour Bellman : **dis-le à voix haute** maintenant, plusieurs fois. C'est l'équation à connaître.
- Si on te demande quelque chose que tu ne sais pas : "Je ne l'ai pas exploré dans les exercices mais d'après ce que j'ai compris, je dirais que…" → mieux que de mentir.
- **Toujours raccrocher au concret** : "dans CartPole c'est… dans FrozenLake c'est…"
- L'examinateur veut voir que tu **comprends ce que tu as fait**, pas que tu récites par cœur.

---

**Bon courage Pierre — tu as largement le niveau si tu retiens ce document. Le plus important : la boucle agent-env, Bellman, et les 3 innovations du DQN.**

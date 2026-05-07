# Soutenance Eagle-1 — Q&A technique

Document de préparation : questions probables du jury + réponses argumentées + ce qu'il faut savoir montrer dans le code.

---

## 1. Concepts fondamentaux du RL

### Q1.1 — C'est quoi le Reinforcement Learning ?
Un paradigme d'apprentissage où un **agent** apprend à agir dans un **environnement** pour maximiser une **récompense cumulative**. Il n'y a pas de dataset étiqueté : l'agent apprend par essais-erreurs.

> Différence vs supervisé : pas de "bonne réponse" donnée, juste un signal (reward) après chaque action.

### Q1.2 — Quels sont les 4 blocs de construction du RL ?
1. **Agent** — celui qui décide (notre PPO)
2. **Environnement** — le monde qui réagit (LunarLander-v3)
3. **Action** — ce que l'agent peut faire (4 actions discrètes)
4. **Reward** — feedback chiffré après chaque action

Boucle : `state → action → reward + new_state → action → ...`

### Q1.3 — Différence entre épisode et step ?
- **Step** : un tick de temps (l'agent observe, agit, reçoit reward)
- **Épisode** : suite de steps jusqu'à la fin (atterrissage ou crash)

LunarLander : ~80–300 steps par épisode.

### Q1.4 — Qu'est-ce que la policy (politique) ?
La fonction `π(action | state)` qui dicte quelle action prendre dans chaque état. Pour PPO/DQN, c'est un réseau de neurones.

### Q1.5 — Qu'est-ce que la value function ?
`V(state)` = somme attendue des rewards futurs si on part de cet état en suivant π.
`Q(state, action)` = pareil, mais on conditionne aussi sur la première action.

PPO entraîne **policy + value**. DQN entraîne **uniquement Q**.

### Q1.6 — Qu'est-ce que gamma (facteur d'actualisation) ?
Poids des rewards futurs : `G = r₀ + γr₁ + γ²r₂ + ...`
- γ = 0 → agent myope (ne voit que le reward immédiat)
- γ = 0.99 → planifie ~100 steps en avance
- γ = 0.999 → planifie ~1000 steps

> **Mon choix** : γ=0.995, on a vu en exp 2 que c'était le sweet spot sur LunarLander.

---

## 2. Q-Learning et Q-table (Exercice 2 — FrozenLake)

### Q2.1 — Comment fonctionne le Q-Learning ?
On stocke une **table** `Q[state][action]`. À chaque step, on met à jour :

```
Q[s][a] ← Q[s][a] + α · (r + γ·max(Q[s']) - Q[s][a])
```

C'est l'**équation de Bellman**. Α = learning rate.

### Q2.2 — Pourquoi Q-table ne marche pas sur LunarLander ?
LunarLander a un espace d'observation **continu** (8 floats : position, vitesse, angle...). Pour une Q-table il faudrait discrétiser → explosion combinatoire.

> Sur FrozenLake (16 états × 4 actions = 64 cases) la Q-table marche. Sur LunarLander, c'est intractable.

### Q2.3 — Epsilon-greedy, c'est quoi ?
Stratégie d'exploration : avec proba ε, on prend une action aléatoire ; sinon, on prend `argmax(Q)`. On décroît ε au fil de l'entraînement (au début on explore, à la fin on exploite).

### Q2.4 — Off-policy vs on-policy ?
- **Off-policy** (Q-Learning, DQN) : peut apprendre depuis n'importe quelle expérience, même passée → replay buffer possible
- **On-policy** (PPO, A2C) : doit utiliser ses dernières expériences sous la policy actuelle → pas de replay buffer

---

## 3. DQN (Exercice 3)

### Q3.1 — Différence entre Q-Learning et DQN ?
DQN = Q-Learning où la Q-table est remplacée par un **réseau de neurones** `Q(s, a; θ)`. On peut donc gérer des espaces continus.

### Q3.2 — Pourquoi DQN a-t-il besoin d'un replay buffer ?
Deux raisons :
1. **Décorréler les samples** : les transitions consécutives dans un épisode sont très corrélées, ça fait diverger le NN
2. **Réutiliser les données** : sample efficiency (chaque transition est utilisée plusieurs fois)

### Q3.3 — C'est quoi le target network ?
Deuxième copie du réseau, mise à jour lentement (soft update ou hard copy). Sert à stabiliser la cible pendant l'apprentissage. Sans, le réseau "court après lui-même".

### Q3.4 — Pourquoi DQN ne marche que sur action **discrète** ?
DQN apprend `Q(s, a)` pour chaque action et fait `argmax_a Q(s,a)`. Avec actions continues, le argmax n'est pas calculable → on utilise SAC, DDPG, TD3 à la place.

---

## 4. PPO

### Q4.1 — Pourquoi PPO et pas DQN sur LunarLander ?
On l'a **testé** (notebook 03). Résultats :
- PPO optimisé : **243.7 ± 51.3**
- DQN optimisé : **115.0 ± 104.4**

PPO est plus stable, converge plus vite, et l'écart-type est 2× plus petit.

### Q4.2 — Comment fonctionne PPO ?
1. Collecte `n_steps` transitions sous la policy actuelle
2. Calcule l'avantage `A(s, a) = Q(s,a) - V(s)` (avec GAE)
3. Met à jour la policy en optimisant un objectif **clippé** :
   `L = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)`
   avec `ratio = π_new(a|s) / π_old(a|s)`

Le clip empêche des updates trop violents → stabilité.

### Q4.3 — C'est quoi l'avantage A(s,a) ?
"Cette action a-t-elle été meilleure que la moyenne dans cet état ?"
- A > 0 → on encourage cette action
- A < 0 → on la décourage

### Q4.4 — Hyperparamètres PPO et leur rôle ?
| Param | Mon choix | Rôle |
|---|---|---|
| `learning_rate` | 1e-3 | Vitesse d'apprentissage du NN |
| `gamma` | 0.995 | Horizon de planification |
| `n_steps` | 4096 | Taille du rollout entre 2 updates |
| `n_epochs` | 10 (défaut) | Passes sur les données collectées |
| `clip_range` | 0.2 (défaut) | Limite des updates de la policy |

### Q4.5 — Pourquoi tuner un seul paramètre à la fois ?
Sinon impossible d'attribuer le gain à un paramètre précis. C'est la méthode scientifique : isoler les variables. (Trade-off : on rate les interactions entre params, mais c'est OK pour un tuning rapide.)

---

## 5. Mon process Eagle-1 (à raconter au jury)

### Étape 1 — Exploration (notebook 01)
- Compris l'environnement : 8 obs continues, 4 actions discrètes
- Reward shaping LunarLander : pénalités carburant, bonus atterrissage, bonus contact pieds
- Baseline aléatoire : **−183** (référence à battre)
- PPO 100k steps brut : **0.3** (mieux que random, mais loin du seuil 200)

### Étape 2 — Optimisation (notebook 02)
3 expériences un-paramètre-à-la-fois sur 300k steps :
- LR : 1e-3 gagne (184.9)
- Gamma : 0.995 gagne (249.8)
- n_steps : 4096 gagne (267.9)
- Run final 500k steps : **242.6 ± 58.2**

### Étape 3 — Comparaison DQN (notebook 03)
3 configs DQN testées, toutes < 0 sur 300k steps. PPO gagne largement.

### Étape 4 — Évaluation finale (notebook 04)
100 épisodes :
- Moyenne **247.0 ± 49.9**
- 91% de réussite (reward > 200)
- Vidéo à 287.7

---

## 6. Architecture du livrable

### Q6.1 — Pourquoi une API ?
Découpler le **modèle** (entraîné une fois) de ses **consommateurs** (GUI, dashboard, applis tierces). Si je change le modèle, les UIs ne bougent pas.

### Q6.2 — Endpoints ?
- `POST /predict` : reçoit une observation, renvoie l'action (usage temps réel par un système externe)
- `POST /play` : joue un épisode complet, renvoie les stats
- `POST /play-video` : joue un épisode, renvoie un mp4
- `GET /model-info` : métadonnées du modèle

### Q6.3 — Pourquoi FastAPI ?
- Validation automatique avec Pydantic
- Doc Swagger gratuite (`/docs`)
- Async-ready
- Standard dans l'écosystème Python moderne

### Q6.4 — Pourquoi Streamlit pour GUI et dashboard ?
- Prototypage ultra rapide (Python pur, pas de HTML/CSS/JS)
- Idéal pour data viz interactive
- Adapté à un livrable de démo, pas une prod publique

### Q6.5 — Différence GUI vs Dashboard ?
| | GUI | Dashboard |
|---|---|---|
| Cible | Visualiser **un** atterrissage | Statistiques sur **N** épisodes |
| Public | Démo qualitative | Analyse quantitative |

---

## 7. Chiffres à retenir par cœur

| Métrique | Valeur |
|---|---|
| Baseline aléatoire | −183 |
| PPO 100k steps brut | 0.3 |
| PPO optimisé final | **247.0 ± 49.9** |
| DQN optimisé | 115.0 ± 104.4 |
| Taux de réussite final | **91%** |
| Hyperparams gagnants | lr=1e-3, γ=0.995, n_steps=4096 |
| Timesteps total entraînement final | 500 000 |

---

## 8. Pièges classiques du jury

### Q8.1 — "Pourquoi ne pas avoir fait du grid search ?"
Coût en temps. Avec 3 params × 3 valeurs = 27 runs × 5 min = 2h+. Le tuning séquentiel a donné un bon résultat en ~10 runs. Si je voulais 280+, je passerais à Optuna/Ray Tune.

### Q8.2 — "Vos résultats sont-ils reproductibles ?"
Pas seedés explicitement → variance run-à-run. Visible exp 1 : `lr=3e-4` a donné −73.6 alors que c'est un défaut connu pour bien marcher. Pour la prod, il faudrait : seed fixe, plusieurs seeds par config, médianes.

### Q8.3 — "Pourquoi 200 comme seuil ?"
C'est le seuil officiel défini par les auteurs de LunarLander (Brockman et al., OpenAI Gym). Reward > 200 = atterrissage propre dans la zone cible avec carburant raisonnable.

### Q8.4 — "Que feriez-vous pour passer de 91% à 99% ?"
- Plus de timesteps (1M+)
- Reward shaping custom
- Tester `ent_coef > 0` (force l'exploration)
- Ensemble de plusieurs PPO entraînés sur seeds différents
- Architecture réseau plus large

### Q8.5 — "Pourquoi gymnasium et pas gym ?"
`gym` est déprécié depuis 2022, repris par la fondation Farama sous `gymnasium`. API quasi-identique mais :
- `reset()` renvoie `(obs, info)` (pas juste obs)
- `step()` renvoie `(obs, reward, terminated, truncated, info)` (séparation explicite fin-d'épisode vs timeout)

### Q8.6 — "C'est quoi `deterministic=True` à l'évaluation ?"
À l'inférence on prend l'action la plus probable (argmax). Pendant l'entraînement on sample dans la distribution (exploration). On évalue toujours en déterministe pour mesurer la vraie performance.

---

## 9. Ouverture (questions de fin)

- **Reward shaping custom** : ajouter une pénalité supplémentaire pour atterrissage hors-zone, ou bonus pour atterrissage symétrique
- **Curriculum learning** : commencer sur une version simplifiée (gravité réduite) puis durcir
- **Multi-agent** : plusieurs landers coopérant pour un essaim
- **Sim-to-real** : appliquer la policy à un vrai drone (gros gap de reality)
- **Algorithmes plus récents** : SAC discret, Rainbow DQN, MuZero

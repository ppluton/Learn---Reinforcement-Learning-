import streamlit as st
import requests
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Eagle-1 — Dashboard", layout="wide")
st.title("Eagle-1 — Tableau de bord de performance")


def play_episodes(n):
    episodes = []
    progress = st.progress(0, text="Lancement des episodes...")
    for i in range(n):
        response = requests.post(f"{API_URL}/play", timeout=30)
        response.raise_for_status()
        episodes.append(response.json())
        progress.progress((i + 1) / n, text=f"Episode {i + 1}/{n}")
    progress.empty()
    return episodes


n_episodes = st.sidebar.slider("Nombre d'episodes", min_value=10, max_value=200, value=50, step=10)

if st.sidebar.button("Lancer l'evaluation", type="primary"):
    episodes = play_episodes(n_episodes)

    rewards = [ep["total_reward"] for ep in episodes]
    successes = [ep["success"] for ep in episodes]
    n_steps_list = [ep["n_steps"] for ep in episodes]

    # Metriques en haut
    st.subheader("Metriques globales")
    st.caption(
        "Vue d'ensemble sur les "
        f"{len(episodes)} episodes joues. **Reward moyen** > 200 = pilote considere comme operationnel "
        "(seuil officiel LunarLander). **Taux de reussite** = % d'atterrissages avec reward > 200."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reward moyen", f"{np.mean(rewards):.1f}")
    col2.metric("Taux de reussite", f"{np.mean(successes) * 100:.0f}%")
    col3.metric("Meilleur score", f"{np.max(rewards):.1f}")
    col4.metric("Pire score", f"{np.min(rewards):.1f}")

    st.divider()

    # Graphiques
    left, right = st.columns(2)

    with left:
        st.subheader("Rewards par episode")
        st.caption(
            "Score brut de chaque episode dans l'ordre chronologique. "
            "Permet de reperer la variabilite : des pics negatifs isoles indiquent des crashes, "
            "une courbe lissee au-dessus de 200 indique un pilote regulier."
        )
        df_rewards = pd.DataFrame({
            "Episode": range(1, len(rewards) + 1),
            "Reward": rewards,
        })
        st.line_chart(df_rewards.set_index("Episode"))

        st.subheader("Distribution des scores")
        st.caption(
            "Histogramme des rewards regroupes par tranches. "
            "Une distribution concentree a droite (vers 250-300) signifie un pilote stable. "
            "Une longue queue a gauche revele les episodes problematiques."
        )
        hist = pd.cut(pd.Series(rewards), bins=15).value_counts().sort_index()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)

    with right:
        st.subheader("Moyenne glissante (fenetre = 10)")
        st.caption(
            "Moyenne sur les 10 derniers episodes. "
            "Lisse le bruit episode-par-episode pour reveler la performance reelle. "
            "Si la courbe reste plate au-dessus de 200, le pilote est fiable."
        )
        window = min(10, len(rewards))
        rolling_mean = pd.Series(rewards).rolling(window).mean()
        st.line_chart(rolling_mean.dropna())

        st.subheader("Statistiques detaillees")
        st.caption(
            "Resume chiffre. **Ecart-type** : plus il est faible, plus le pilote est constant. "
            "**Mediane** vs **moyenne** : un ecart important signale des outliers (crashes rares)."
        )
        st.dataframe(pd.DataFrame({
            "Metrique": ["Moyenne", "Ecart-type", "Mediane", "Min", "Max", "Steps moyen"],
            "Valeur": [
                f"{np.mean(rewards):.1f}",
                f"{np.std(rewards):.1f}",
                f"{np.median(rewards):.1f}",
                f"{np.min(rewards):.1f}",
                f"{np.max(rewards):.1f}",
                f"{np.mean(n_steps_list):.0f}",
            ]
        }), hide_index=True)

    st.divider()

    # Analyse des actions
    st.subheader("Decisions du pilote automatique")
    st.caption(
        "LunarLander a 4 actions possibles : ne rien faire, allumer le moteur gauche, "
        "le moteur principal (bas), ou le moteur droit. Comparer les reussites et les crashes "
        "permet de voir si le pilote sur-utilise ou sous-utilise certaines manoeuvres."
    )

    actions_success = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}
    actions_failure = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}

    for ep in episodes:
        target = actions_success if ep["success"] else actions_failure
        for step in ep["steps"]:
            target[step["action_name"]] += 1

    all_actions = {k: actions_success[k] + actions_failure[k] for k in actions_success}

    col_act1, col_act2 = st.columns(2)

    with col_act1:
        st.write("**Repartition globale des actions**")
        st.bar_chart(all_actions)

    with col_act2:
        st.write("**Actions dans les atterrissages reussis vs crashes**")
        df_actions = pd.DataFrame({
            "Action": list(actions_success.keys()),
            "Reussis": list(actions_success.values()),
            "Crashes": list(actions_failure.values()),
        }).set_index("Action")
        st.bar_chart(df_actions)


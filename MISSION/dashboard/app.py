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
        df_rewards = pd.DataFrame({
            "Episode": range(1, len(rewards) + 1),
            "Reward": rewards,
        })
        st.line_chart(df_rewards.set_index("Episode"))

        st.subheader("Distribution des scores")
        st.bar_chart(pd.cut(pd.Series(rewards), bins=15).value_counts().sort_index())

    with right:
        st.subheader("Moyenne glissante (fenetre = 10)")
        window = min(10, len(rewards))
        rolling_mean = pd.Series(rewards).rolling(window).mean()
        st.line_chart(rolling_mean.dropna())

        st.subheader("Statistiques detaillees")
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

    all_actions = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}
    actions_success = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}
    actions_failure = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}

    for ep in episodes:
        target = actions_success if ep["success"] else actions_failure
        for step in ep["steps"]:
            all_actions[step["action_name"]] += 1
            target[step["action_name"]] += 1

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

    # Sauvegarde des donnees dans session_state
    st.session_state["last_results"] = {
        "rewards": rewards,
        "successes": successes,
        "n_episodes": n_episodes,
    }

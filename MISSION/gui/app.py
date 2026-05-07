import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Eagle-1 — Visualisation", layout="centered")
st.title("Eagle-1 — Visualisation d'atterrissage")
st.caption("Regardez le pilote automatique en action")


def fetch_episode():
    response = requests.post(f"{API_URL}/play", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_video():
    response = requests.post(f"{API_URL}/play-video", timeout=60)
    response.raise_for_status()
    return response.content, {
        "total_reward": float(response.headers.get("X-Total-Reward", 0)),
        "success": response.headers.get("X-Success") == "1",
    }


col_btn1, col_btn2 = st.columns(2)
launch_stats = col_btn1.button("Lancer un atterrissage", type="primary")
launch_video = col_btn2.button("Lancer avec video")

if launch_video:
    with st.spinner("Le module est en approche... (rendu video)"):
        video_bytes, meta = fetch_video()

    if meta["success"]:
        st.success(f"Atterrissage reussi ! Score : {meta['total_reward']:.1f}")
    else:
        st.error(f"Crash... Score : {meta['total_reward']:.1f}")

    st.video(video_bytes)
    st.caption("Replay de l'episode genere par l'API (rendu cote serveur, retourne en mp4).")

if launch_stats:
    with st.spinner("Le module est en approche..."):
        episode = fetch_episode()

    total_reward = episode["total_reward"]
    n_steps = episode["n_steps"]
    success = episode["success"]

    if success:
        st.success(f"Atterrissage reussi ! Score : {total_reward:.1f}")
    else:
        st.error(f"Crash... Score : {total_reward:.1f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Score total", f"{total_reward:.1f}")
    col2.metric("Nombre de steps", n_steps)
    col3.metric("Statut", "Reussi" if success else "Crash")

    st.subheader("Deroulement de l'episode")

    actions_count = {"Ne rien faire": 0, "Moteur gauche": 0, "Moteur principal": 0, "Moteur droit": 0}
    rewards_over_time = []
    cumulative = 0

    for step in episode["steps"]:
        actions_count[step["action_name"]] += 1
        cumulative += step["reward"]
        rewards_over_time.append(cumulative)

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Repartition des actions**")
        st.bar_chart(actions_count)

    with col_b:
        st.write("**Reward cumule au fil du temps**")
        st.line_chart(rewards_over_time)

    with st.expander("Details step par step"):
        for i, step in enumerate(episode["steps"]):
            st.text(f"Step {i:3d} | Action: {step['action_name']:20s} | Reward: {step['reward']:+.2f}")

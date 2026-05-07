import io
import tempfile
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path

app = FastAPI(title="Eagle-1 Autopilot API", version="1.0.0")

MODEL_PATH = Path(__file__).parent.parent / "models" / "ppo_optimized"
model = PPO.load(str(MODEL_PATH))


class State(BaseModel):
    observation: list[float]


class PredictResponse(BaseModel):
    action: int
    action_name: str


class StepRecord(BaseModel):
    observation: list[float]
    action: int
    action_name: str
    reward: float


class PlayResponse(BaseModel):
    steps: list[StepRecord]
    total_reward: float
    n_steps: int
    success: bool


class ModelInfo(BaseModel):
    algorithm: str
    observation_size: int
    n_actions: int


ACTION_NAMES = {0: "Ne rien faire", 1: "Moteur gauche", 2: "Moteur principal", 3: "Moteur droit"}


@app.post("/predict", response_model=PredictResponse)
def predict(state: State):
    obs = np.array(state.observation, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    return PredictResponse(action=action, action_name=ACTION_NAMES[action])


@app.post("/play", response_model=PlayResponse)
def play():
    env = gym.make("LunarLander-v3")
    obs, info = env.reset()
    done = False
    steps = []
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        step_obs = obs.tolist()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        steps.append(StepRecord(
            observation=step_obs,
            action=action,
            action_name=ACTION_NAMES[action],
            reward=float(reward),
        ))

    env.close()

    return PlayResponse(
        steps=steps,
        total_reward=float(total_reward),
        n_steps=len(steps),
        success=total_reward > 200,
    )


@app.post("/play-video")
def play_video():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    frames = []
    total_reward = 0.0

    while not done:
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        done = terminated or truncated

    for _ in range(20):
        frames.append(env.render())
    env.close()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        imageio.mimsave(tmp.name, frames, fps=30, macro_block_size=1)
        tmp.seek(0)
        video_bytes = Path(tmp.name).read_bytes()

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Total-Reward": f"{total_reward:.2f}",
            "X-Success": "1" if total_reward > 200 else "0",
            "X-N-Steps": str(len(frames)),
        },
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        algorithm="PPO",
        observation_size=8,
        n_actions=4,
    )

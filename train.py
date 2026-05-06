"""
Trains a PPO agent on the LidarCarEnv.
Run: python train.py
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env import LidarCarEnv


def make_env():
    """Factory function that creates one environment instance."""
    return LidarCarEnv('map2.png')


# --- Set up the training environment ---
# DummyVecEnv: wraps our env in SB3's vectorized format (required)
# VecMonitor: tracks episode rewards and lengths
# VecFrameStack: stacks the last 4 observations so the policy can sense motion
train_env = DummyVecEnv([make_env])
train_env = VecMonitor(train_env)
train_env = VecFrameStack(train_env, n_stack=4)

# --- Set up an eval environment (separate copy) ---
eval_env = DummyVecEnv([make_env])
eval_env = VecMonitor(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)

# --- Callbacks ---
# Save the best model whenever eval reward improves
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/eval/',
    eval_freq=10_000,        # evaluate every 1k training steps
    n_eval_episodes=5,        # average over 5 episodes
    deterministic=True,
    render=False,
)

# Also save periodic checkpoints in case training crashes
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path='./checkpoints/',
    name_prefix='ppo_lidar_car',
)

# --- Create the PPO model ---
import os

# --- Create or load the PPO model ---
BEST_MODEL_PATH = 'best_model/best_model.zip'

if os.path.exists(BEST_MODEL_PATH):
    print(f"Found existing model at {BEST_MODEL_PATH}")
    print("Loading and continuing training...\n")
    model = PPO.load(
        BEST_MODEL_PATH,
        env=train_env,
        device='cpu',
        tensorboard_log='./logs/',
    )
    print(f"Resumed at step: {model.num_timesteps:,}")
    RESET_TIMESTEPS = False
else:
    print("No existing model found. Starting fresh training.\n")
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log='./logs/',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu',
    )
    RESET_TIMESTEPS = True

print("=" * 60)
print(f"Training device: {model.device}")
print(f"Total training steps: 1,000,000")
print(f"Estimated time: 1-3 hours")
print("=" * 60)

# --- Train ---
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
    reset_num_timesteps=RESET_TIMESTEPS,
)

# --- Save the final model ---
model.save('final_model')
print("\nTraining complete! Saved as final_model.zip")
print("Best model saved in best_model/best_model.zip")
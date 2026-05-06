# rlcar

PPO agent learning to drive a simulated lidar-equipped car around a 2D map.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

What happens:

- Trains for 1,000,000 steps using PPO from Stable-Baselines3 (CPU).
- If `best_model/best_model.zip` already exists, training **resumes** from it. Delete that folder to start fresh.
- Every 10,000 steps the agent is evaluated over 5 episodes; whenever the mean reward improves, the new best policy is written to `best_model/best_model.zip`.
- Periodic checkpoints land in `checkpoints/` (in case training crashes).
- TensorBoard logs go to `logs/`. The final policy is saved as `final_model.zip`.

Estimated wall time: ~1–3 hours.

### Monitor training (optional)

In a second terminal:

```bash
tensorboard --logdir logs
```

Then open the URL it prints (usually http://localhost:6006).

## Watch the current best model drive

Open a **separate** terminal (you can do this while training is still running):

```bash
python watch.py
```

This loads `best_model/best_model.zip` and renders the car driving with its lidar rays in a matplotlib window. Close the window to stop. Re-run the script any time to pick up the latest improved checkpoint.

If you see `No best model yet`, training hasn't completed its first eval — wait until ~10k steps have been logged and try again.

## Files

- `env.py` — the lidar + bicycle-model car environment.
- `train.py` — PPO training loop with eval + checkpoint callbacks.
- `watch.py` — live visualizer for the saved best model.
- `map2.png` — the track the car trains on (white = free, dark = wall).
- `best_model/` — best policy seen during eval (auto-saved).
- `checkpoints/` — periodic snapshots every 10k steps.
- `logs/` — TensorBoard + eval logs.

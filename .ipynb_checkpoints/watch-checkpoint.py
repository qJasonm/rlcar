"""
Watch the current best model drive in real time.
Run this in a SEPARATE Anaconda Prompt while training is happening.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env import LidarCarEnv
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms



def make_env():
    return LidarCarEnv('map2.png')


# Load the latest best model
print("Loading best_model/best_model.zip...")
try:
    model = PPO.load('best_model/best_model')
    print("Loaded successfully")
except FileNotFoundError:
    print("No best model yet. Training needs to run for ~5k steps first.")
    print("Try again in a few minutes.")
    exit()

# Create env (matching training setup)
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

# Get a reference to the underlying env so we can read its state
raw_env = env.envs[0] if hasattr(env, 'envs') else env.unwrapped.envs[0]

# Set up the plot
plt.ion()  # interactive mode = real-time updates
fig, ax = plt.subplots(figsize=(10, 10))

# Show the map
img_data = raw_env.grid * 255  # invert for display
ax.imshow(raw_env.grid, cmap='gray_r',
          extent=[0, raw_env.map_w * raw_env.resolution,
                  raw_env.map_h * raw_env.resolution, 0])

# Plot elements that we'll update each frame
# Car rectangle — width is along travel direction (length), height is the side width
CAR_LENGTH = 0.65   # 65cm long
CAR_WIDTH = 0.3    # 30cm wide
car_rect = Rectangle(
    (-CAR_LENGTH/2, -CAR_WIDTH/2),  # anchor at center
    CAR_LENGTH, CAR_WIDTH,
    color='red', alpha=0.8, zorder=10
)
ax.add_patch(car_rect)
heading_arrow, = ax.plot([], [], 'darkred', linewidth=2, zorder=11)
trail, = ax.plot([], [], 'r-', alpha=0.3, linewidth=1)
lidar_lines = [ax.plot([], [], 'b-', alpha=0.15, linewidth=0.5)[0]
               for _ in range(50)]

ax.set_title('Watching trained policy drive (close window to stop)')
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')

# Run episodes forever
trail_x, trail_y = [], []
obs = env.reset()
episode_count = 1
step_in_episode = 0

print("\nWatching the policy. Close the plot window to stop.")
print("Tip: re-run this script to load the latest improved model.\n")

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Read state from the raw env
    x, y, theta = raw_env.x, raw_env.y, raw_env.theta
    scan = raw_env.last_scan.reshape(50, -1).min(axis=1)
    
    # Update car position
    # Update car rectangle (rotate around its center, then translate to position)
    t = (transforms.Affine2D()
         .rotate(theta)
         .translate(x, y) + ax.transData)
    car_rect.set_transform(t)
    
    # Heading line from front of car
    front_x = x + (CAR_LENGTH/2) * np.cos(theta)
    front_y = y + (CAR_LENGTH/2) * np.sin(theta)
    heading_arrow.set_data(
        [x, front_x + 0.3 * np.cos(theta)],
        [y, front_y + 0.3 * np.sin(theta)]
    )
    
    # Update trail
    trail_x.append(x)
    trail_y.append(y)
    if len(trail_x) > 200:  # keep last 200 points
        trail_x.pop(0)
        trail_y.pop(0)
    trail.set_data(trail_x, trail_y)
    
    # Update lidar rays
    for i, line in enumerate(lidar_lines):
        ray_angle = theta + (i / 50) * 2 * np.pi
        line.set_data(
            [x, x + scan[i] * np.cos(ray_angle)],
            [y, y + scan[i] * np.sin(ray_angle)]
        )
    
    step_in_episode += 1
    ax.set_title(
        f'Episode {episode_count}, step {step_in_episode}, '
        f'speed {raw_env.v:.1f} m/s'
    )
    
    plt.pause(0.05)  # ~20 FPS
    
    if done[0]:
        print(f"Episode {episode_count} ended after {step_in_episode} steps "
              f"({'crashed' if raw_env._check_collision() else 'survived'})")
        episode_count += 1
        step_in_episode = 0
        trail_x.clear()
        trail_y.clear()
        time.sleep(1)  # brief pause between episodes
        obs = env.reset()
    
    # Stop if user closed the window
    if not plt.fignum_exists(fig.number):
        break

print("Stopped watching.")
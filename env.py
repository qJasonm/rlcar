"""
Gymnasium environment for RL car training with virtual LD19 lidar.

Components:
- VirtualLD19: vectorized 2D ray-casting lidar matching LD19 specs
- LidarCarEnv: bicycle-model car with rectangular (oriented) collision check
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image


class VirtualLD19:
    """
    Simulates an LD19 lidar by casting rays into a 2D occupancy grid.
    Vectorized version - all 450 rays computed in parallel using numpy.
    """

    def __init__(self, grid, resolution=0.05):
        # grid: 2D numpy array, 1 = wall, 0 = free space
        # resolution: meters per pixel
        self.grid = grid
        self.resolution = resolution
        self.height, self.width = grid.shape

        # LD19 specs
        self.num_rays = 450
        self.max_range = 12.0
        self.min_range = 0.02
        self.ray_angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        # Pre-compute step distances along each ray (in meters)
        step_size = resolution * 0.5
        self.num_steps = int(self.max_range / step_size)
        self.step_distances = np.arange(1, self.num_steps + 1) * step_size

        # Noise parameters from LD19 datasheet
        self.range_noise_std = 0.010              # 10mm standard deviation
        self.angle_noise_std = np.deg2rad(2.0)    # ±2° angular error

    def scan(self, x, y, theta):
        """Vectorized scan: cast all 450 rays simultaneously."""
        # Add angular noise to each ray
        angles = (theta + self.ray_angles +
                  np.random.normal(0, self.angle_noise_std, self.num_rays))

        # Compute (x, y) at each step distance for every ray
        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]
        ray_xs = x + cos_a * self.step_distances[None, :]
        ray_ys = y + sin_a * self.step_distances[None, :]

        # Convert to pixel indices
        px = (ray_xs / self.resolution).astype(np.int32)
        py = (ray_ys / self.resolution).astype(np.int32)

        # Mark out-of-bounds as walls
        out_of_bounds = (px < 0) | (px >= self.width) | (py < 0) | (py >= self.height)
        px = np.clip(px, 0, self.width - 1)
        py = np.clip(py, 0, self.height - 1)

        # Sample the grid: 1 = wall hit
        hits = self.grid[py, px] | out_of_bounds.astype(np.uint8)

        # Find first hit per ray
        any_hit = hits.any(axis=1)
        first_hit_idx = hits.argmax(axis=1)

        # Distance to first hit (or max_range if no hit)
        ranges = np.where(
            any_hit,
            self.step_distances[first_hit_idx],
            self.max_range
        )

        # Add range noise and clip
        ranges = ranges + np.random.normal(0, self.range_noise_std, self.num_rays)
        ranges = np.clip(ranges, self.min_range, self.max_range)

        return ranges


class LidarCarEnv(gym.Env):
    """
    A 2D car in a 2D map, controlled by [steering, throttle].
    Observation = downsampled lidar scan + velocity + previous action.
    Reward = forward speed - wall proximity penalty - jerky steering penalty - idle penalty.
    Collision check uses an oriented (rotating) rectangular footprint.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, map_path, resolution=0.05):
        super().__init__()

        # --- Load the map ---
        img = np.array(Image.open(map_path).convert('L'))
        self.grid = (img < 240).astype(np.uint8)  # 1 = wall, 0 = free
        self.resolution = resolution
        self.map_h, self.map_w = self.grid.shape

        # --- Create lidar ---
        self.lidar = VirtualLD19(self.grid, resolution)
        self.num_rays_obs = 50  # downsample 450 -> 50 (450/50 = 9 per bin)

        # --- Car physical parameters ---
        self.dt = 0.1                       # 10 Hz, matches LD19 scan rate
        self.max_steer = np.deg2rad(30)     # ±30° steering
        self.max_accel = 2.0                # m/s²
        self.max_speed = 3.0                # m/s
        self.wheelbase = 0.3                # 30cm front-to-rear
        self.car_length = 0.40              # 40cm long
        self.car_width = 0.20               # 20cm wide
        # Diagonal half-length used for spawn-margin check
        self.car_radius = 0.5 * np.hypot(self.car_length, self.car_width)

        # --- Episode parameters ---
        self.max_steps = 1000   # 100 seconds at 10 Hz

        # --- Spaces ---
        # 50 lidar + 1 velocity + 2 prev action = 53
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(53,), dtype=np.float32
        )
        # [steering, throttle], both in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # State (set in reset)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.last_scan = None

    def _find_free_spawn(self):
        """Find a random free pixel with clearance margin around it."""
        margin = int(self.car_radius / self.resolution) + 5
        for _ in range(1000):
            py = np.random.randint(margin, self.map_h - margin)
            px = np.random.randint(margin, self.map_w - margin)
            patch = self.grid[py-margin:py+margin+1, px-margin:px+margin+1]
            if patch.sum() == 0:
                return px * self.resolution, py * self.resolution
        raise RuntimeError("Couldn't find a clear spawn point. Map too tight?")

    def _check_collision(self):
        """
        Rectangular collision check. The rectangle rotates with the car heading.
        Returns True if any wall pixel falls inside the car's footprint.
        """
        hl = self.car_length / 2
        hw = self.car_width / 2

        # Corners in car-local frame
        local_corners = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ])

        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        world_corners = (rot @ local_corners.T).T + np.array([self.x, self.y])

        # Out-of-bounds = collision
        if (world_corners[:, 0].min() < 0 or
            world_corners[:, 0].max() >= self.map_w * self.resolution or
            world_corners[:, 1].min() < 0 or
            world_corners[:, 1].max() >= self.map_h * self.resolution):
            return True

        # Bounding box in pixel coords
        px_min = int(world_corners[:, 0].min() / self.resolution)
        px_max = int(world_corners[:, 0].max() / self.resolution) + 1
        py_min = int(world_corners[:, 1].min() / self.resolution)
        py_max = int(world_corners[:, 1].max() / self.resolution) + 1

        # Walls inside bounding box
        sub_grid = self.grid[py_min:py_max, px_min:px_max]
        if sub_grid.sum() == 0:
            return False

        wall_ys, wall_xs = np.where(sub_grid == 1)
        if len(wall_xs) == 0:
            return False

        # Wall pixel centers in world coords
        wall_world_x = (wall_xs + px_min + 0.5) * self.resolution
        wall_world_y = (wall_ys + py_min + 0.5) * self.resolution

        # Transform walls into car-local frame (inverse rotation)
        dx = wall_world_x - self.x
        dy = wall_world_y - self.y
        local_x =  cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy

        # Inside rectangle if |local_x| < hl AND |local_y| < hw
        inside = (np.abs(local_x) < hl) & (np.abs(local_y) < hw)
        return bool(inside.any())

    def _get_observation(self):
        """Build the observation vector the policy sees."""
        # Downsample 450 rays -> 50 by min-pooling each chunk
        scan = self.last_scan.reshape(self.num_rays_obs, -1).min(axis=1)
        scan_norm = (scan / self.lidar.max_range).astype(np.float32)
        obs = np.concatenate([
            scan_norm,
            [self.v / self.max_speed],
            self.prev_action,
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = self._find_free_spawn()
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.v = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.last_scan = self.lidar.scan(self.x, self.y, self.theta)
        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        steer_cmd = action[0] * self.max_steer
        accel_cmd = action[1] * self.max_accel

        # Bicycle model
        self.v = np.clip(self.v + accel_cmd * self.dt, 0.0, self.max_speed)
        self.theta += (self.v / self.wheelbase) * np.tan(steer_cmd) * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        # Collision and lidar update
        collided = self._check_collision()
        self.last_scan = self.lidar.scan(self.x, self.y, self.theta)

        # Reward
        if collided:
            reward = -10.0
        else:
            reward = 0.0
            # Forward speed bonus (main signal)
            reward += 0.3 * self.v
            # Idle penalty - discourages standing still
            if self.v < 0.3:
                reward -= 0.1
            # Wall proximity penalty
            min_dist = np.min(self.last_scan)
            if min_dist < 0.4:
                reward -= 0.5 * (0.4 - min_dist)
            # Smoothness penalty
            reward -= 0.05 * abs(action[0] - self.prev_action[0])
            # NO alive bonus (would let agent farm reward by standing still)

        self.prev_action = action
        self.steps += 1
        terminated = collided
        truncated = self.steps >= self.max_steps

        info = {
            'x': self.x, 'y': self.y, 'theta': self.theta,
            'v': self.v, 'collided': collided
        }
        return self._get_observation(), reward, terminated, truncated, info
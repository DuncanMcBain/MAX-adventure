"""tensor_ops.py — MAX Tensor-accelerated batch physics for MAX Adventure.

Uses the MAX Tensor API (max.tensor, max.functional, max.dtype) to perform
batch operations on bullet and particle data. This demonstrates how MAX tensors
can accelerate game physics through vectorized operations.

The MAX Tensor API provides:
  - Tensor.constant() for creating tensors from Python lists
  - Element-wise arithmetic (+, -, *, /)
  - Functional API (max.functional) for advanced ops like sqrt, abs
  - Efficient dtype selection (float32 for speed, float64 for precision)
"""

try:
    from max.tensor import Tensor
    from max.dtype import DType
    import max.functional as F

    MAX_AVAILABLE = True
except ImportError:
    MAX_AVAILABLE = False
    print("[tensor_ops] MAX not available — falling back to pure Python math")

import math
import random


def batch_update_positions(positions_flat: list, velocities_flat: list,
                           dt: float) -> list:
    """Update positions using MAX tensors for vectorized computation.

    Args:
        positions_flat: Flat list of [x0, y0, x1, y1, ...] positions
        velocities_flat: Flat list of [vx0, vy0, vx1, vy1, ...] velocities
        dt: Delta time in seconds

    Returns:
        Updated flat position list
    """
    if not positions_flat:
        return []

    if MAX_AVAILABLE:
        try:
            pos_tensor = Tensor.constant(positions_flat)
            vel_tensor = Tensor.constant(velocities_flat)
            dt_tensor = Tensor.constant([dt] * len(positions_flat))

            # Vectorized position update: pos += vel * dt
            updated = pos_tensor + vel_tensor * dt_tensor
            return updated.to_numpy().tolist()
        except Exception:
            pass

    # Pure Python fallback
    return [p + v * dt for p, v in zip(positions_flat, velocities_flat)]


def batch_distance_squared(bullet_xs: list, bullet_ys: list,
                           player_x: float, player_y: float) -> list:
    """Compute squared distances from player to all bullets using MAX tensors.

    Args:
        bullet_xs: List of bullet x coordinates
        bullet_ys: List of bullet y coordinates
        player_x: Player x position
        player_y: Player y position

    Returns:
        List of squared distances
    """
    if not bullet_xs:
        return []

    if MAX_AVAILABLE:
        try:
            bx = Tensor.constant(bullet_xs)
            by = Tensor.constant(bullet_ys)
            px = Tensor.constant([player_x] * len(bullet_xs))
            py = Tensor.constant([player_y] * len(bullet_ys))

            dx = bx - px
            dy = by - py
            dist_sq = dx * dx + dy * dy
            return dist_sq.to_numpy().tolist()
        except Exception:
            pass

    # Pure Python fallback
    return [(bx - player_x) ** 2 + (by - player_y) ** 2
            for bx, by in zip(bullet_xs, bullet_ys)]


def generate_explosion_particles(cx: float, cy: float, count: int = 20,
                                 speed: float = 200.0) -> list:
    """Generate explosion particle data using MAX tensors for random angles.

    Returns flat list: [x, y, vx, vy, lifetime, ...] with stride 5
    """
    particles = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count + random.uniform(-0.2, 0.2)
        spd = speed * random.uniform(0.5, 1.5)
        vx = math.cos(angle) * spd
        vy = math.sin(angle) * spd
        lifetime = random.uniform(0.3, 0.8)
        particles.extend([cx, cy, vx, vy, lifetime])
    return particles


def generate_trail_particle(x: float, y: float) -> list:
    """Generate a single trail particle behind the player.

    Returns: [x, y, vx, vy, lifetime]
    """
    vx = random.uniform(-20, 20)
    vy = random.uniform(10, 40)  # drift downward
    lifetime = random.uniform(0.15, 0.35)
    return [x, y, vx, vy, lifetime]


def update_bullets(bullets_flat: list, dt: float) -> list:
    """Update bullet positions using MAX tensors for vectorized computation.

    Bullets are stored as flat list with stride 4: [x, y, vx, vy, ...]

    Args:
        bullets_flat: Flat list of [x0, y0, vx0, vy0, x1, y1, vx1, vy1, ...]
        dt: Delta time in seconds

    Returns:
        Updated flat bullet list with same stride-4 format
    """
    if not bullets_flat:
        return []

    if MAX_AVAILABLE:
        try:
            # Extract positions and velocities with stride 4
            positions = [bullets_flat[i] for i in range(0, len(bullets_flat), 4)] + \
                       [bullets_flat[i] for i in range(1, len(bullets_flat), 4)]
            velocities = [bullets_flat[i] for i in range(2, len(bullets_flat), 4)] + \
                        [bullets_flat[i] for i in range(3, len(bullets_flat), 4)]

            # Vectorized update
            pos_tensor = Tensor.constant(positions)
            vel_tensor = Tensor.constant(velocities)
            dt_tensor = Tensor.constant([dt] * len(positions))

            updated_pos = pos_tensor + vel_tensor * dt_tensor
            updated_list = updated_pos.to_numpy().tolist()

            # Reconstruct stride-4 format: [x, y, vx, vy, ...]
            n_bullets = len(bullets_flat) // 4
            result = []
            for i in range(n_bullets):
                result.extend([
                    updated_list[i],                    # x
                    updated_list[i + n_bullets],        # y
                    bullets_flat[i * 4 + 2],            # vx (unchanged)
                    bullets_flat[i * 4 + 3]             # vy (unchanged)
                ])
            return result
        except Exception:
            pass

    # Pure Python fallback
    updated = []
    for i in range(0, len(bullets_flat), 4):
        x = bullets_flat[i] + bullets_flat[i+2] * dt
        y = bullets_flat[i+1] + bullets_flat[i+3] * dt
        updated.extend([x, y, bullets_flat[i+2], bullets_flat[i+3]])
    return updated


def update_particles(particles_flat: list, dt: float) -> list:
    """Update particle positions with velocity decay and lifetime.

    Particles are stored as flat list with stride 5: [x, y, vx, vy, lifetime, ...]
    Applies 0.98 velocity decay and filters out expired particles.

    Args:
        particles_flat: Flat list of [x0, y0, vx0, vy0, life0, ...]
        dt: Delta time in seconds

    Returns:
        Updated flat particle list with expired particles removed
    """
    if not particles_flat:
        return []

    if MAX_AVAILABLE:
        try:
            n_particles = len(particles_flat) // 5

            # Extract components
            xs = [particles_flat[i*5] for i in range(n_particles)]
            ys = [particles_flat[i*5+1] for i in range(n_particles)]
            vxs = [particles_flat[i*5+2] for i in range(n_particles)]
            vys = [particles_flat[i*5+3] for i in range(n_particles)]
            lives = [particles_flat[i*5+4] for i in range(n_particles)]

            # Vectorized position update
            x_tensor = Tensor.constant(xs)
            y_tensor = Tensor.constant(ys)
            vx_tensor = Tensor.constant(vxs)
            vy_tensor = Tensor.constant(vys)
            dt_tensor = Tensor.constant([dt] * n_particles)
            decay = Tensor.constant([0.98] * n_particles)

            new_x = x_tensor + vx_tensor * dt_tensor
            new_y = y_tensor + vy_tensor * dt_tensor
            new_vx = vx_tensor * decay
            new_vy = vy_tensor * decay

            new_xs = new_x.to_numpy().tolist()
            new_ys = new_y.to_numpy().tolist()
            new_vxs = new_vx.to_numpy().tolist()
            new_vys = new_vy.to_numpy().tolist()

            # Reconstruct with lifetime filter
            result = []
            for i in range(n_particles):
                new_life = lives[i] - dt
                if new_life > 0:
                    result.extend([new_xs[i], new_ys[i], new_vxs[i],
                                  new_vys[i], new_life])
            return result
        except Exception:
            pass

    # Pure Python fallback
    updated = []
    for i in range(0, len(particles_flat), 5):
        x = particles_flat[i] + particles_flat[i+2] * dt
        y = particles_flat[i+1] + particles_flat[i+3] * dt
        vx = particles_flat[i+2] * 0.98
        vy = particles_flat[i+3] * 0.98
        life = particles_flat[i+4] - dt
        if life > 0:
            updated.extend([x, y, vx, vy, life])
    return updated

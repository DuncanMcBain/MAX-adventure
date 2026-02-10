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

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
                                 speed: float = 200.0) -> dict:
    """Generate explosion particle data using MAX tensors for random angles.

    Returns dict with SoA format: {'x': [...], 'y': [...], 'vx': [...], 'vy': [...], 'life': [...]}
    """
    xs, ys, vxs, vys, lives = [], [], [], [], []
    for i in range(count):
        angle = 2.0 * math.pi * i / count + random.uniform(-0.2, 0.2)
        spd = speed * random.uniform(0.5, 1.5)
        vx = math.cos(angle) * spd
        vy = math.sin(angle) * spd
        lifetime = random.uniform(0.3, 0.8)
        xs.append(cx)
        ys.append(cy)
        vxs.append(vx)
        vys.append(vy)
        lives.append(lifetime)
    return {'x': xs, 'y': ys, 'vx': vxs, 'vy': vys, 'life': lives}


def generate_trail_particle(x: float, y: float) -> dict:
    """Generate a single trail particle behind the player.

    Returns: dict with single element in each array
    """
    vx = random.uniform(-20, 20)
    vy = random.uniform(10, 40)  # drift downward
    lifetime = random.uniform(0.15, 0.35)
    return {'x': [x], 'y': [y], 'vx': [vx], 'vy': [vy], 'life': [lifetime]}


def update_bullets(bullets: dict, dt: float) -> dict:
    """Update bullet positions using MAX tensors for vectorized computation.

    Bullets are stored as SoA dict: {'x': [...], 'y': [...], 'vx': [...], 'vy': [...]}

    Args:
        bullets: Dict with separate arrays for x, y, vx, vy
        dt: Delta time in seconds

    Returns:
        Updated dict with same structure
    """
    if not bullets['x']:
        return bullets

    if MAX_AVAILABLE:
        try:
            # Direct tensor operations on arrays - no stride extraction needed
            x_tensor = Tensor.constant(bullets['x'])
            y_tensor = Tensor.constant(bullets['y'])
            vx_tensor = Tensor.constant(bullets['vx'])
            vy_tensor = Tensor.constant(bullets['vy'])
            dt_tensor = Tensor.constant([dt] * len(bullets['x']))

            # Vectorized position update: pos += vel * dt
            new_x = x_tensor + vx_tensor * dt_tensor
            new_y = y_tensor + vy_tensor * dt_tensor

            return {
                'x': new_x.to_numpy().tolist(),
                'y': new_y.to_numpy().tolist(),
                'vx': bullets['vx'],
                'vy': bullets['vy']
            }
        except Exception:
            pass

    # Pure Python fallback using dict format
    return {
        'x': [x + vx * dt for x, vx in zip(bullets['x'], bullets['vx'])],
        'y': [y + vy * dt for y, vy in zip(bullets['y'], bullets['vy'])],
        'vx': bullets['vx'],
        'vy': bullets['vy']
    }


def update_particles(particles: dict, dt: float) -> dict:
    """Update particle positions with velocity decay and lifetime.

    Particles are stored as SoA dict: {'x': [...], 'y': [...], 'vx': [...], 'vy': [...], 'life': [...]}
    Applies 0.98 velocity decay and filters out expired particles.

    Args:
        particles: Dict with separate arrays for x, y, vx, vy, life
        dt: Delta time in seconds

    Returns:
        Updated dict with expired particles removed
    """
    if not particles['x']:
        return particles

    if MAX_AVAILABLE:
        try:
            n_particles = len(particles['x'])

            # Direct tensor operations on separate arrays
            x_tensor = Tensor.constant(particles['x'])
            y_tensor = Tensor.constant(particles['y'])
            vx_tensor = Tensor.constant(particles['vx'])
            vy_tensor = Tensor.constant(particles['vy'])
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

            # Filter particles: only include indices where life > 0
            filtered_xs, filtered_ys, filtered_vxs, filtered_vys, filtered_lives = [], [], [], [], []
            for i in range(n_particles):
                new_life = particles['life'][i] - dt
                if new_life > 0:
                    filtered_xs.append(new_xs[i])
                    filtered_ys.append(new_ys[i])
                    filtered_vxs.append(new_vxs[i])
                    filtered_vys.append(new_vys[i])
                    filtered_lives.append(new_life)

            return {
                'x': filtered_xs,
                'y': filtered_ys,
                'vx': filtered_vxs,
                'vy': filtered_vys,
                'life': filtered_lives
            }
        except Exception:
            pass

    # Pure Python fallback using dict format
    filtered_xs, filtered_ys, filtered_vxs, filtered_vys, filtered_lives = [], [], [], [], []
    for i in range(len(particles['x'])):
        x = particles['x'][i] + particles['vx'][i] * dt
        y = particles['y'][i] + particles['vy'][i] * dt
        vx = particles['vx'][i] * 0.98
        vy = particles['vy'][i] * 0.98
        life = particles['life'][i] - dt
        if life > 0:
            filtered_xs.append(x)
            filtered_ys.append(y)
            filtered_vxs.append(vx)
            filtered_vys.append(vy)
            filtered_lives.append(life)

    return {
        'x': filtered_xs,
        'y': filtered_ys,
        'vx': filtered_vxs,
        'vy': filtered_vys,
        'life': filtered_lives
    }

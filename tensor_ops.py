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


def update_bullets_tensors(x_t, y_t, vx_t, vy_t, dt: float):
    """Update bullet positions using MAX tensors (no conversion overhead).

    Args:
        x_t, y_t: Position tensors
        vx_t, vy_t: Velocity tensors
        dt: Delta time in seconds

    Returns:
        Tuple of (new_x, new_y, vx, vy) tensors
    """
    if MAX_AVAILABLE:
        try:
            n = len(x_t.to_numpy())
            dt_tensor = Tensor.constant([dt] * n)

            # Vectorized position update: pos += vel * dt
            new_x = x_t + vx_t * dt_tensor
            new_y = y_t + vy_t * dt_tensor

            return new_x, new_y, vx_t, vy_t
        except Exception:
            pass

    # Should not reach here if called correctly
    raise RuntimeError("update_bullets_tensors called without MAX_AVAILABLE")


def update_bullets(bullets: dict, dt: float) -> dict:
    """Update bullet positions (list-based fallback).

    Args:
        bullets: Dict with separate arrays for x, y, vx, vy
        dt: Delta time in seconds

    Returns:
        Updated dict with same structure
    """
    if not bullets['x']:
        return bullets

    # Pure Python fallback using dict format
    return {
        'x': [x + vx * dt for x, vx in zip(bullets['x'], bullets['vx'])],
        'y': [y + vy * dt for y, vy in zip(bullets['y'], bullets['vy'])],
        'vx': bullets['vx'],
        'vy': bullets['vy']
    }


def update_particles_tensors(x_t, y_t, vx_t, vy_t, life_t, dt: float):
    """Update particle positions with velocity decay (no conversion overhead).

    Args:
        x_t, y_t: Position tensors
        vx_t, vy_t: Velocity tensors
        life_t: Lifetime tensor (or list if not tensor)
        dt: Delta time in seconds

    Returns:
        Tuple of (new_x, new_y, new_vx, new_vy, new_life, alive_mask) tensors/arrays
        alive_mask is a boolean numpy array indicating which particles are still alive
    """
    if MAX_AVAILABLE:
        try:
            n_particles = len(x_t.to_numpy())
            dt_tensor = Tensor.constant([dt] * n_particles)
            decay = Tensor.constant([0.98] * n_particles)

            new_x = x_t + vx_t * dt_tensor
            new_y = y_t + vy_t * dt_tensor
            new_vx = vx_t * decay
            new_vy = vy_t * decay

            # Life is kept as list for now since we need to filter
            # (MAX tensors don't have great boolean indexing support yet)
            import numpy as np
            life_np = np.array(life_t) - dt
            alive_mask = life_np > 0

            return new_x, new_y, new_vx, new_vy, life_np.tolist(), alive_mask
        except Exception:
            pass

    raise RuntimeError("update_particles_tensors called without MAX_AVAILABLE")


def update_particles(particles: dict, dt: float) -> dict:
    """Update particle positions with velocity decay (list-based fallback).

    Args:
        particles: Dict with separate arrays for x, y, vx, vy, life
        dt: Delta time in seconds

    Returns:
        Updated dict with expired particles removed
    """
    if not particles['x']:
        return particles

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


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Management Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def dict_to_tensors(data: dict):
    """Convert dict-of-lists to tensors (for bullets/particles).

    Returns tuple of tensors, or None if MAX not available.
    """
    if not MAX_AVAILABLE or not data['x']:
        return None

    try:
        if 'life' in data:
            # Particles
            return (
                Tensor.constant(data['x']),
                Tensor.constant(data['y']),
                Tensor.constant(data['vx']),
                Tensor.constant(data['vy']),
                data['life']  # Keep life as list for now
            )
        else:
            # Bullets
            return (
                Tensor.constant(data['x']),
                Tensor.constant(data['y']),
                Tensor.constant(data['vx']),
                Tensor.constant(data['vy'])
            )
    except Exception:
        return None


def concat_tensors(existing_tensors, new_dict: dict):
    """Concatenate new dict data to existing tensors.

    Args:
        existing_tensors: Tuple of existing tensors (or None)
        new_dict: Dict with new data to append

    Returns:
        New tuple of concatenated tensors
    """
    if not MAX_AVAILABLE or not new_dict['x']:
        return existing_tensors

    try:
        import numpy as np

        if existing_tensors is None or len(existing_tensors[0].to_numpy()) == 0:
            # No existing data, just create new tensors
            return dict_to_tensors(new_dict)

        # Convert to numpy, concatenate, convert back to tensor
        if len(existing_tensors) == 5:
            # Particles (with life)
            x_np = np.concatenate([existing_tensors[0].to_numpy(), new_dict['x']])
            y_np = np.concatenate([existing_tensors[1].to_numpy(), new_dict['y']])
            vx_np = np.concatenate([existing_tensors[2].to_numpy(), new_dict['vx']])
            vy_np = np.concatenate([existing_tensors[3].to_numpy(), new_dict['vy']])
            life_list = existing_tensors[4] + new_dict['life']

            return (
                Tensor.constant(x_np.tolist()),
                Tensor.constant(y_np.tolist()),
                Tensor.constant(vx_np.tolist()),
                Tensor.constant(vy_np.tolist()),
                life_list
            )
        else:
            # Bullets (no life)
            x_np = np.concatenate([existing_tensors[0].to_numpy(), new_dict['x']])
            y_np = np.concatenate([existing_tensors[1].to_numpy(), new_dict['y']])
            vx_np = np.concatenate([existing_tensors[2].to_numpy(), new_dict['vx']])
            vy_np = np.concatenate([existing_tensors[3].to_numpy(), new_dict['vy']])

            return (
                Tensor.constant(x_np.tolist()),
                Tensor.constant(y_np.tolist()),
                Tensor.constant(vx_np.tolist()),
                Tensor.constant(vy_np.tolist())
            )
    except Exception:
        return existing_tensors


def filter_tensors(tensors, mask):
    """Filter tensors using boolean mask.

    Args:
        tensors: Tuple of tensors
        mask: Boolean numpy array

    Returns:
        New tuple of filtered tensors
    """
    if not MAX_AVAILABLE or tensors is None:
        return tensors

    try:
        import numpy as np

        if len(tensors) == 5:
            # Particles (with life)
            x_np = tensors[0].to_numpy()[mask]
            y_np = tensors[1].to_numpy()[mask]
            vx_np = tensors[2].to_numpy()[mask]
            vy_np = tensors[3].to_numpy()[mask]
            life_list = [tensors[4][i] for i in range(len(mask)) if mask[i]]

            if len(x_np) == 0:
                return (
                    Tensor.constant([0.0]),  # Empty placeholder
                    Tensor.constant([0.0]),
                    Tensor.constant([0.0]),
                    Tensor.constant([0.0]),
                    []
                )

            return (
                Tensor.constant(x_np.tolist()),
                Tensor.constant(y_np.tolist()),
                Tensor.constant(vx_np.tolist()),
                Tensor.constant(vy_np.tolist()),
                life_list
            )
        else:
            # Bullets (no life)
            x_np = tensors[0].to_numpy()[mask]
            y_np = tensors[1].to_numpy()[mask]
            vx_np = tensors[2].to_numpy()[mask]
            vy_np = tensors[3].to_numpy()[mask]

            if len(x_np) == 0:
                return (
                    Tensor.constant([0.0]),  # Empty placeholder
                    Tensor.constant([0.0]),
                    Tensor.constant([0.0]),
                    Tensor.constant([0.0])
                )

            return (
                Tensor.constant(x_np.tolist()),
                Tensor.constant(y_np.tolist()),
                Tensor.constant(vx_np.tolist()),
                Tensor.constant(vy_np.tolist())
            )
    except Exception:
        return tensors

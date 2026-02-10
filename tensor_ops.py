"""tensor_ops.py — MAX Graph API-accelerated batch physics for MAX Adventure.

Uses the MAX Graph API to compile optimized compute graphs for batch
particle/bullet physics. Graphs are compiled once at import time and
executed each frame with numpy arrays as I/O.

The MAX Graph API provides:
  - Graph() for defining symbolic computation graphs
  - TensorType with symbolic dimensions for dynamic batch sizes
  - InferenceSession for compiling graphs to optimized machine code
  - Model.execute() for running compiled graphs with numpy array I/O
"""

import numpy as np
import math
import random

from max.graph import Graph, TensorType
from max.dtype import DType
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession


# ═══════════════════════════════════════════════════════════════════════════════
# Compile MAX Graphs (once at module load)
# ═══════════════════════════════════════════════════════════════════════════════

_gpu = Accelerator()


def _build_bullet_update_graph():
    """Graph: new_pos = pos + vel * dt"""

    def forward(x, y, vx, vy, dt):
        return x + vx * dt, y + vy * dt, vx, vy

    return Graph(
        "bullet_update",
        forward=forward,
        input_types=[
            TensorType(DType.float32, ("n",), _gpu),  # x
            TensorType(DType.float32, ("n",), _gpu),  # y
            TensorType(DType.float32, ("n",), _gpu),  # vx
            TensorType(DType.float32, ("n",), _gpu),  # vy
            TensorType(DType.float32, (1,), _gpu),     # dt (broadcasts)
        ],
    )


def _build_particle_update_graph():
    """Graph: pos += vel * dt; vel *= decay; life -= dt"""

    def forward(x, y, vx, vy, life, dt, decay):
        return (
            x + vx * dt,
            y + vy * dt,
            vx * decay,
            vy * decay,
            life - dt,
        )

    return Graph(
        "particle_update",
        forward=forward,
        input_types=[
            TensorType(DType.float32, ("n",), _gpu),  # x
            TensorType(DType.float32, ("n",), _gpu),  # y
            TensorType(DType.float32, ("n",), _gpu),  # vx
            TensorType(DType.float32, ("n",), _gpu),  # vy
            TensorType(DType.float32, ("n",), _gpu),  # life
            TensorType(DType.float32, (1,), _gpu),     # dt (broadcasts)
            TensorType(DType.float32, (1,), _gpu),     # decay (broadcasts)
        ],
    )


_session = InferenceSession(devices=[_gpu])
_bullet_model = _session.load(_build_bullet_update_graph())
_particle_model = _session.load(_build_particle_update_graph())
print(f"MAX Graph models compiled successfully on {_gpu}")


# ═══════════════════════════════════════════════════════════════════════════════
# Physics Update Functions
# ═══════════════════════════════════════════════════════════════════════════════

def update_bullets(x: np.ndarray, y: np.ndarray,
                   vx: np.ndarray, vy: np.ndarray,
                   dt: float) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Update bullet positions using compiled MAX graph.

    Args:
        x, y: Position arrays (float32)
        vx, vy: Velocity arrays (float32)
        dt: Delta time in seconds

    Returns:
        Tuple of (new_x, new_y, vx, vy) numpy arrays
    """
    if len(x) == 0:
        return x, y, vx, vy

    dt_arr = np.array([dt], dtype=np.float32)
    results = _bullet_model.execute(
        Buffer.from_numpy(x).to(_gpu),
        Buffer.from_numpy(y).to(_gpu),
        Buffer.from_numpy(vx).to(_gpu),
        Buffer.from_numpy(vy).to(_gpu),
        Buffer.from_numpy(dt_arr).to(_gpu),
    )
    return (
        results[0].to_numpy(),
        results[1].to_numpy(),
        results[2].to_numpy(),
        results[3].to_numpy(),
    )


def update_particles(x: np.ndarray, y: np.ndarray,
                     vx: np.ndarray, vy: np.ndarray,
                     life: np.ndarray,
                     dt: float) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray,
                                         np.ndarray]:
    """Update particle positions/velocities/life using compiled MAX graph.

    Applies velocity decay (0.98) and removes dead particles.

    Args:
        x, y: Position arrays (float32)
        vx, vy: Velocity arrays (float32)
        life: Lifetime array (float32)
        dt: Delta time in seconds

    Returns:
        Tuple of (new_x, new_y, new_vx, new_vy, new_life) with dead
        particles filtered out.
    """
    if len(x) == 0:
        return x, y, vx, vy, life

    dt_arr = np.array([dt], dtype=np.float32)
    decay_arr = np.array([0.98], dtype=np.float32)
    results = _particle_model.execute(
        Buffer.from_numpy(x).to(_gpu),
        Buffer.from_numpy(y).to(_gpu),
        Buffer.from_numpy(vx).to(_gpu),
        Buffer.from_numpy(vy).to(_gpu),
        Buffer.from_numpy(life).to(_gpu),
        Buffer.from_numpy(dt_arr).to(_gpu),
        Buffer.from_numpy(decay_arr).to(_gpu),
    )

    new_x = results[0].to_numpy()
    new_y = results[1].to_numpy()
    new_vx = results[2].to_numpy()
    new_vy = results[3].to_numpy()
    new_life = results[4].to_numpy()

    # Filter out dead particles (shape-changing op, done outside the graph)
    alive = new_life > 0
    return (
        new_x[alive], new_y[alive],
        new_vx[alive], new_vy[alive],
        new_life[alive],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Particle Generation (pure Python — randomness not suited for graphs)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_explosion_particles(cx: float, cy: float, count: int = 20,
                                 speed: float = 200.0) -> dict:
    """Generate explosion particle data.

    Returns dict with SoA format:
        {'x': [...], 'y': [...], 'vx': [...], 'vy': [...], 'life': [...]}
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

    Returns: dict with single-element lists for each field.
    """
    vx = random.uniform(-20, 20)
    vy = random.uniform(10, 40)  # drift downward
    lifetime = random.uniform(0.15, 0.35)
    return {'x': [x], 'y': [y], 'vx': [vx], 'vy': [vy], 'life': [lifetime]}


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy Array Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def empty_bullets() -> dict:
    """Create empty bullet arrays (SoA format)."""
    return {
        'x': np.array([], dtype=np.float32),
        'y': np.array([], dtype=np.float32),
        'vx': np.array([], dtype=np.float32),
        'vy': np.array([], dtype=np.float32),
    }


def empty_particles() -> dict:
    """Create empty particle arrays (SoA format)."""
    return {
        'x': np.array([], dtype=np.float32),
        'y': np.array([], dtype=np.float32),
        'vx': np.array([], dtype=np.float32),
        'vy': np.array([], dtype=np.float32),
        'life': np.array([], dtype=np.float32),
    }


def append_data(target: dict, source: dict):
    """Append source data (lists or arrays) to target numpy arrays in-place."""
    for key in target:
        if key in source and len(source[key]) > 0:
            target[key] = np.concatenate([
                target[key],
                np.asarray(source[key], dtype=np.float32),
            ])


# ═══════════════════════════════════════════════════════════════════════════════
# Commented-out pure Python versions (previously used as fallbacks)
# ═══════════════════════════════════════════════════════════════════════════════

# def batch_update_positions(positions_flat, velocities_flat, dt):
#     """Pure Python: pos + vel * dt."""
#     if not positions_flat:
#         return []
#     return [p + v * dt for p, v in zip(positions_flat, velocities_flat)]
#
#
# def batch_distance_squared(bullet_xs, bullet_ys, player_x, player_y):
#     """Pure Python: squared distances."""
#     if not bullet_xs:
#         return []
#     return [(bx - player_x) ** 2 + (by - player_y) ** 2
#             for bx, by in zip(bullet_xs, bullet_ys)]
#
#
# def update_bullets_py(bullets, dt):
#     """Pure Python bullet update (list-based)."""
#     if not bullets['x']:
#         return bullets
#     return {
#         'x': [x + vx * dt for x, vx in zip(bullets['x'], bullets['vx'])],
#         'y': [y + vy * dt for y, vy in zip(bullets['y'], bullets['vy'])],
#         'vx': bullets['vx'],
#         'vy': bullets['vy'],
#     }
#
#
# def update_particles_py(particles, dt):
#     """Pure Python particle update with decay and filtering."""
#     if not particles['x']:
#         return particles
#     result = {'x': [], 'y': [], 'vx': [], 'vy': [], 'life': []}
#     for i in range(len(particles['x'])):
#         life = particles['life'][i] - dt
#         if life > 0:
#             result['x'].append(particles['x'][i] + particles['vx'][i] * dt)
#             result['y'].append(particles['y'][i] + particles['vy'][i] * dt)
#             result['vx'].append(particles['vx'][i] * 0.98)
#             result['vy'].append(particles['vy'][i] * 0.98)
#             result['life'].append(life)
#     return result

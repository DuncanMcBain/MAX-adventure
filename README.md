# 🚀 MAX Adventure — A Bullet Hell Game

**Powered by Mojo 🔥 + MAX Tensors + Pygame**

A space-themed bullet hell game where you play as **MAX the Astronaut**, dodging
waves of increasingly chaotic bullet patterns while fighting a boss. The game
demonstrates Mojo↔Python interop by offloading intensive particle physics to a
compiled Mojo module, with MAX Tensor operations for batch vectorized math.

![MAX Adventure](https://img.shields.io/badge/Mojo-🔥_Physics-orange)
![MAX Tensors](https://img.shields.io/badge/MAX-Tensor_Ops-blue)
![Pygame](https://img.shields.io/badge/Pygame-Rendering-green)

---

## 🎮 Gameplay

- **Arrow keys / WASD** — Move MAX around the screen
- **SPACE / Z** — Fire bullets at the boss
- **Left Shift** — Focus mode (slower movement + visible hitbox)
- Dodge bullets, graze them for bonus points, and defeat the boss!

### Bullet Hell Mechanics

- **Tiny hitbox**: Your actual collision area is just 8px radius — much smaller
  than the sprite. This is classic bullet hell design.
- **Graze system**: Flying close to bullets without getting hit earns bonus
  points. Risk vs. reward!
- **3 boss phases**: As the boss loses health, patterns become denser and faster.
- **Invincibility frames**: After a hit, you get 2 seconds of invulnerability.

---

## 🏗️ Architecture

```
max_adventure/
├── pixi.toml              # Pixi project manifest (Mojo + Python + Pygame)
├── max_adventure.py        # Main game — Pygame loop, rendering, game logic
├── physics_engine.mojo     # Mojo module — high-perf bullet/particle physics
├── tensor_ops.py           # MAX Tensor API — batch vectorized operations
└── README.md               # This file
```

### How the pieces fit together

```
┌─────────────────────────────────────────────────────────┐
│                    max_adventure.py                      │
│              (Pygame rendering + game logic)             │
│                                                         │
│  ┌───────────────────┐      ┌────────────────────────┐  │
│  │  physics_engine    │      │     tensor_ops.py      │  │
│  │     (Mojo 🔥)     │      │   (MAX Tensor API)     │  │
│  │                   │      │                        │  │
│  │ • update_bullets  │      │ • batch_update_pos     │  │
│  │ • check_collisions│      │ • batch_distance_sq    │  │
│  │ • spawn_radial    │      │ • explosion_particles  │  │
│  │ • spawn_spiral    │      │ • trail_particles      │  │
│  │ • update_particles│      │                        │  │
│  └───────────────────┘      └────────────────────────┘  │
│           │                          │                   │
│     PythonModuleBuilder        Tensor.constant()         │
│     (Mojo→Python interop)      + element-wise ops        │
└─────────────────────────────────────────────────────────┘
```

### Why Mojo for physics?

In a bullet hell game, you need to update and collision-check **hundreds of
bullets every frame** at 60fps. Pure Python would struggle with this. The Mojo
physics engine (`physics_engine.mojo`) handles:

1. **Bullet position updates** — Euler integration on flat arrays (stride-4)
2. **Collision detection** — Squared-distance checks against player hitbox
3. **Pattern generation** — Radial and spiral burst math (sin/cos)
4. **Particle effects** — Lifetime decay with drag on visual particles

All of this runs as compiled native code via Mojo, called from Python through
the `PythonModuleBuilder` interop system. The flat-array layout
(`[x,y,vx,vy,x,y,vx,vy,...]`) is cache-friendly and avoids Python object
overhead.

### MAX Tensor API usage

`tensor_ops.py` demonstrates the MAX Tensor API for batch operations:

- `Tensor.constant()` to create tensors from Python lists
- Element-wise `+`, `*` operators for vectorized position updates
- `DType.float32` for efficient GPU-ready data types
- Graceful fallback to pure Python when MAX isn't available

---

## 📦 Setup & Run

### Prerequisites

Install [pixi](https://pixi.sh/latest/) if you haven't:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Install & Play

```bash
# Clone or copy this project directory, then:
cd max_adventure

# Install all dependencies (Mojo, MAX, Python, Pygame)
pixi install

# Play the game!
pixi run play
```

### Manual setup (alternative)

```bash
pixi init max-adventure \
  -c https://conda.modular.com/max-nightly/ -c conda-forge

cd max-adventure
pixi add modular python pygame numpy

# Copy the game files into this directory, then:
pixi run python max_adventure.py
```

---

## 🔧 Technical Details

### Mojo↔Python Interop

The `physics_engine.mojo` file uses `PythonModuleBuilder` to expose functions:

```mojo
@export
fn PyInit_physics_engine() -> PythonObject:
    var m = PythonModuleBuilder("physics_engine")
    m.def_function[update_bullets]("update_bullets", ...)
    return m.finalize()
```

Python imports it transparently via `max.mojo.importer`:

```python
import max.mojo.importer
import physics_engine  # Mojo module, used like any Python module

result = physics_engine.update_bullets(flat_bullet_list, dt)
```

### Data Layout

Bullets use a flat array with stride 4: `[x, y, vx, vy, x, y, vx, vy, ...]`

This avoids the overhead of Python objects per bullet and allows Mojo to iterate
through contiguous memory efficiently. Particles use stride 5 with an additional
`lifetime` field.

### Graceful Degradation

The game runs even without Mojo/MAX installed — all physics functions have pure
Python fallbacks. The HUD shows whether you're running with `🔥 Mojo` or
`🐍 Python` physics.

---

## 📚 References

- [Get started with Mojo](https://docs.modular.com/mojo/manual/get-started/)
- [MAX Tensor fundamentals](https://docs.modular.com/max/develop/tensors)
- [MAX Data types (dtype)](https://docs.modular.com/max/develop/dtypes)
- [MAX Basic operations](https://docs.modular.com/max/develop/basic-ops)
- [Calling Mojo from Python](https://docs.modular.com/mojo/manual/python/mojo-from-python/)
- [Modular GitHub](https://github.com/modularml/modular)

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
├── tensor_ops.py           # MAX Tensor API — batch vectorized operations
└── README.md               # This file
```


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

---

## 📚 References

- [Get started with Mojo](https://docs.modular.com/mojo/manual/get-started/)
- [MAX Tensor fundamentals](https://docs.modular.com/max/develop/tensors)
- [MAX Data types (dtype)](https://docs.modular.com/max/develop/dtypes)
- [MAX Basic operations](https://docs.modular.com/max/develop/basic-ops)
- [Modular GitHub](https://github.com/modularml/modular)

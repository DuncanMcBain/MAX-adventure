#!/usr/bin/env python3
"""MAX Adventure — A Bullet Hell Game

A space bullet hell game featuring MAX the astronaut, powered by:
  • Pygame for rendering and input
  • Mojo for high-performance particle physics (via PythonModuleBuilder)
  • MAX Tensor API for batch vectorized operations

Run:  pixi run python max_adventure.py
"""

import sys
import math
import random
import time

import pygame

# ─────────────────────────────────────────────────────────────────────────────
# Try to import the Mojo physics engine (compiled via max.mojo.importer)
# Falls back to pure Python if Mojo module isn't available
# ─────────────────────────────────────────────────────────────────────────────
MOJO_PHYSICS = False
try:
    import mojo.importer
    sys.path.insert(0, "")
    import physics_engine
    MOJO_PHYSICS = True
    print("🔥 Mojo physics engine loaded!")
except ImportError:
    print("⚠  Mojo physics engine not available — using Python fallback")

from tensor_ops import (
    update_bullets,
    update_particles,
    generate_explosion_particles,
    generate_trail_particle,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def extend_particles(target: dict, source: dict):
    """Append all particles from source dict to target dict."""
    for key in ['x', 'y', 'vx', 'vy', 'life']:
        target[key].extend(source[key])

def extend_bullets(target: dict, source: dict):
    """Append all bullets from source dict to target dict."""
    for key in ['x', 'y', 'vx', 'vy']:
        target[key].extend(source[key])

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 800, 600
FPS = 60
PLAYER_SPEED = 280
PLAYER_RADIUS = 8       # Hitbox (small — this is bullet hell!)
PLAYER_DRAW_SIZE = 28   # Visual sprite size
BULLET_RADIUS = 4
GRAZE_RADIUS = 24       # Graze detection radius for score bonus

# Colors — space theme palette
C_BG           = (6, 6, 18)
C_STARS        = (80, 80, 120)
C_PLAYER       = (0, 200, 255)
C_PLAYER_VISOR = (180, 240, 255)
C_BULLET_RED   = (255, 60, 60)
C_BULLET_PINK  = (255, 100, 180)
C_BULLET_CYAN  = (0, 220, 255)
C_BULLET_GOLD  = (255, 200, 40)
C_PARTICLE     = (255, 180, 60)
C_GRAZE        = (255, 255, 100)
C_HUD          = (200, 220, 255)
C_TITLE        = (0, 200, 255)
C_ACCENT       = (255, 100, 60)

BULLET_COLORS = [C_BULLET_RED, C_BULLET_PINK, C_BULLET_CYAN, C_BULLET_GOLD]

# ═══════════════════════════════════════════════════════════════════════════════
# Sprite Drawing — Pixel-art style astronaut "MAX"
# ═══════════════════════════════════════════════════════════════════════════════

def draw_astronaut(surface: pygame.Surface, x: int, y: int, size: int = 28,
                   invincible: bool = False, frame: int = 0):
    """Draw MAX the astronaut as a pixel-art sprite."""
    s = size
    cx, cy = int(x), int(y)

    # Jetpack flame animation
    flame_offset = math.sin(frame * 0.3) * 2
    flame_h = int(s * 0.3 + flame_offset)
    flame_colors = [(255, 100, 20), (255, 180, 40), (255, 240, 100)]
    for i, fc in enumerate(flame_colors):
        fw = max(2, int(s * 0.2) - i * 2)
        fh = max(2, flame_h - i * 3)
        pygame.draw.rect(surface, fc,
                         (cx - fw // 2, cy + s // 2 + i * 2, fw, fh))

    # Body (suit)
    body_color = (200, 200, 220) if not invincible else (
        100 + int(80 * math.sin(frame * 0.5)), 200, 255
    )
    pygame.draw.rect(surface, body_color,
                     (cx - s // 4, cy - s // 6, s // 2, s // 2))

    # Backpack (jetpack)
    pygame.draw.rect(surface, (120, 120, 140),
                     (cx - s // 3, cy - s // 8, s // 8, s // 3))
    pygame.draw.rect(surface, (120, 120, 140),
                     (cx + s // 5, cy - s // 8, s // 8, s // 3))

    # Helmet (round)
    helmet_r = int(s * 0.32)
    pygame.draw.circle(surface, (220, 220, 240), (cx, cy - s // 4), helmet_r)
    # Visor
    visor_r = int(s * 0.22)
    visor_color = C_PLAYER_VISOR if not invincible else (
        int(180 + 60 * math.sin(frame * 0.4)), 240, 255
    )
    pygame.draw.circle(surface, visor_color, (cx, cy - s // 4), visor_r)
    # Visor shine
    pygame.draw.circle(surface, (255, 255, 255),
                       (cx - visor_r // 3, cy - s // 4 - visor_r // 3),
                       max(1, visor_r // 4))

    # Arms
    pygame.draw.rect(surface, body_color,
                     (cx - s // 3 - 3, cy - s // 8, 4, s // 4))
    pygame.draw.rect(surface, body_color,
                     (cx + s // 4, cy - s // 8, 4, s // 4))

    # Legs
    pygame.draw.rect(surface, body_color,
                     (cx - s // 5, cy + s // 4, 4, s // 5))
    pygame.draw.rect(surface, body_color,
                     (cx + s // 8, cy + s // 4, 4, s // 5))

    # Boots
    pygame.draw.rect(surface, (180, 80, 40),
                     (cx - s // 4, cy + s // 4 + s // 6, 6, 4))
    pygame.draw.rect(surface, (180, 80, 40),
                     (cx + s // 10, cy + s // 4 + s // 6, 6, 4))


def draw_bullet(surface: pygame.Surface, x: int, y: int, color: tuple,
                radius: int = BULLET_RADIUS):
    """Draw a glowing bullet."""
    # Outer glow
    glow_surf = pygame.Surface((radius * 6, radius * 6), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*color[:3], 40), (radius * 3, radius * 3),
                       radius * 3)
    surface.blit(glow_surf, (int(x) - radius * 3, int(y) - radius * 3))
    # Core
    pygame.draw.circle(surface, color, (int(x), int(y)), radius)
    # Bright center
    pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)),
                       max(1, radius // 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Star Field Background
# ═══════════════════════════════════════════════════════════════════════════════

class StarField:
    def __init__(self, count: int = 120):
        self.stars = []
        for _ in range(count):
            x = random.randint(0, SCREEN_W)
            y = random.randint(0, SCREEN_H)
            speed = random.uniform(10, 60)
            brightness = random.randint(40, 180)
            size = random.choice([1, 1, 1, 2])
            self.stars.append([x, y, speed, brightness, size])

    def update(self, dt: float):
        for star in self.stars:
            star[1] += star[2] * dt
            if star[1] > SCREEN_H:
                star[1] = 0
                star[0] = random.randint(0, SCREEN_W)

    def draw(self, surface: pygame.Surface):
        for x, y, _, b, sz in self.stars:
            c = (b, b, min(255, b + 40))
            if sz == 1:
                surface.set_at((int(x), int(y)), c)
            else:
                pygame.draw.circle(surface, c, (int(x), int(y)), sz)


# ═══════════════════════════════════════════════════════════════════════════════
# Boss Patterns — Bullet hell attack patterns
# ═══════════════════════════════════════════════════════════════════════════════

class BossPattern:
    """Manages boss position, health, and bullet patterns."""

    def __init__(self):
        self.x = SCREEN_W / 2
        self.y = 80
        self.health = 100
        self.max_health = 100
        self.phase = 0
        self.timer = 0.0
        self.spiral_angle = 0.0
        self.pattern_timer = 0.0
        self.move_timer = 0.0
        self.target_x = SCREEN_W / 2
        self.alive = True
        self.flash_timer = 0.0

    def update(self, dt: float) -> dict:
        """Update boss and return new bullets as SoA dict."""
        if not self.alive:
            return {'x': [], 'y': [], 'vx': [], 'vy': []}

        self.timer += dt
        self.pattern_timer += dt
        self.move_timer += dt
        self.flash_timer = max(0, self.flash_timer - dt)

        # Smooth movement
        if self.move_timer > 2.0:
            self.target_x = random.uniform(100, SCREEN_W - 100)
            self.move_timer = 0
        self.x += (self.target_x - self.x) * 2.0 * dt

        # Determine phase from health
        if self.health > 66:
            self.phase = 0
        elif self.health > 33:
            self.phase = 1
        else:
            self.phase = 2

        new_bullets = {'x': [], 'y': [], 'vx': [], 'vy': []}

        if self.phase == 0:
            # Phase 1: Slow radial bursts every 1.2s
            if self.pattern_timer > 1.2:
                self.pattern_timer = 0
                # Mojo code path remains unchanged but won't execute since MOJO_PHYSICS=False
                if MOJO_PHYSICS:
                    flat = list(physics_engine.spawn_radial_burst(
                        self.x, self.y, 120.0, 16
                    ))
                    # Convert flat to dict (won't execute)
                    for i in range(0, len(flat), 4):
                        new_bullets['x'].append(flat[i])
                        new_bullets['y'].append(flat[i+1])
                        new_bullets['vx'].append(flat[i+2])
                        new_bullets['vy'].append(flat[i+3])
                else:
                    new_bullets = self._py_radial(self.x, self.y, 120.0, 16)

        elif self.phase == 1:
            # Phase 2: Spiral + aimed shots
            if self.pattern_timer > 0.15:
                self.pattern_timer = 0
                self.spiral_angle += 0.4
                if MOJO_PHYSICS:
                    flat = list(physics_engine.spawn_spiral_burst(
                        self.x, self.y, 140.0, 3, self.spiral_angle
                    ))
                    for i in range(0, len(flat), 4):
                        new_bullets['x'].append(flat[i])
                        new_bullets['y'].append(flat[i+1])
                        new_bullets['vx'].append(flat[i+2])
                        new_bullets['vy'].append(flat[i+3])
                else:
                    new_bullets = self._py_spiral(
                        self.x, self.y, 140.0, 3, self.spiral_angle
                    )

        else:
            # Phase 3: Dense radial + fast spiral
            if self.pattern_timer > 0.08:
                self.pattern_timer = 0
                self.spiral_angle += 0.3
                if MOJO_PHYSICS:
                    spiral = list(physics_engine.spawn_spiral_burst(
                        self.x, self.y, 180.0, 5, self.spiral_angle
                    ))
                    for i in range(0, len(spiral), 4):
                        new_bullets['x'].append(spiral[i])
                        new_bullets['y'].append(spiral[i+1])
                        new_bullets['vx'].append(spiral[i+2])
                        new_bullets['vy'].append(spiral[i+3])
                else:
                    new_bullets = self._py_spiral(
                        self.x, self.y, 180.0, 5, self.spiral_angle
                    )

            # Extra radial bursts
            if int(self.timer * 10) % 20 == 0:
                if MOJO_PHYSICS:
                    radial = list(physics_engine.spawn_radial_burst(
                        self.x, self.y, 100.0, 24
                    ))
                    for i in range(0, len(radial), 4):
                        new_bullets['x'].append(radial[i])
                        new_bullets['y'].append(radial[i+1])
                        new_bullets['vx'].append(radial[i+2])
                        new_bullets['vy'].append(radial[i+3])
                else:
                    radial = self._py_radial(self.x, self.y, 100.0, 24)
                    extend_bullets(new_bullets, radial)

        return new_bullets

    def take_damage(self, amount: int = 1):
        self.health -= amount
        self.flash_timer = 0.1
        if self.health <= 0:
            self.alive = False

    def draw(self, surface: pygame.Surface, frame: int):
        if not self.alive:
            return

        cx, cy = int(self.x), int(self.y)

        # Body — menacing geometric shape
        body_color = (180, 40, 40) if self.flash_timer <= 0 else (255, 255, 255)
        pts = [
            (cx, cy - 30),
            (cx - 35, cy + 10),
            (cx - 20, cy + 25),
            (cx + 20, cy + 25),
            (cx + 35, cy + 10),
        ]
        pygame.draw.polygon(surface, body_color, pts)
        pygame.draw.polygon(surface, (255, 80, 80), pts, 2)

        # Eye
        pulse = int(6 + 3 * math.sin(frame * 0.1))
        pygame.draw.circle(surface, (255, 255, 0), (cx, cy), pulse)
        pygame.draw.circle(surface, (255, 100, 0), (cx, cy), max(2, pulse - 3))

        # Health bar
        bar_w = 70
        bar_h = 5
        ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, (80, 0, 0),
                         (cx - bar_w // 2, cy - 40, bar_w, bar_h))
        pygame.draw.rect(surface, (255, 40, 40),
                         (cx - bar_w // 2, cy - 40, int(bar_w * ratio), bar_h))

    # Pure Python fallbacks for bullet patterns
    @staticmethod
    def _py_radial(cx, cy, speed, count):
        xs, ys, vxs, vys = [], [], [], []
        for i in range(count):
            angle = 2 * math.pi * i / count
            xs.append(cx)
            ys.append(cy)
            vxs.append(math.cos(angle) * speed)
            vys.append(math.sin(angle) * speed)
        return {'x': xs, 'y': ys, 'vx': vxs, 'vy': vys}

    @staticmethod
    def _py_spiral(cx, cy, speed, count, offset):
        xs, ys, vxs, vys = [], [], [], []
        for i in range(count):
            angle = 2 * math.pi * i / count + offset
            xs.append(cx)
            ys.append(cy)
            vxs.append(math.cos(angle) * speed)
            vys.append(math.sin(angle) * speed)
        return {'x': xs, 'y': ys, 'vx': vxs, 'vy': vys}


# ═══════════════════════════════════════════════════════════════════════════════
# Game State
# ═══════════════════════════════════════════════════════════════════════════════

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("MAX Adventure — Bullet Hell")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 56)
        self.font_small = pygame.font.Font(None, 22)

        self.reset()

    def reset(self):
        # Player
        self.px = SCREEN_W / 2
        self.py = SCREEN_H - 80
        self.lives = 3
        self.score = 0
        self.graze_count = 0
        self.invincible_timer = 0.0

        # Player bullets (for shooting the boss)
        self.player_bullets = []  # [(x, y, vy), ...]
        self.shoot_timer = 0.0

        # Enemy bullets as SoA dict
        self.bullets = {'x': [], 'y': [], 'vx': [], 'vy': []}
        self.bullet_colors_idx = []  # color index per bullet

        # Particles as SoA dict
        self.particles = {'x': [], 'y': [], 'vx': [], 'vy': [], 'life': []}

        # Scene
        self.stars = StarField()
        self.boss = BossPattern()
        self.frame = 0
        self.state = "title"  # title, playing, gameover, victory
        self.shake_timer = 0.0

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)  # cap dt to avoid spiral of death
            self.frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if self.state == "title" and event.key == pygame.K_SPACE:
                        self.state = "playing"
                    elif self.state in ("gameover", "victory"):
                        if event.key == pygame.K_SPACE:
                            self.reset()
                            self.state = "playing"
                        elif event.key == pygame.K_ESCAPE:
                            running = False

            if self.state == "playing":
                self._update(dt)

            self._draw()
            pygame.display.flip()

        pygame.quit()

    # ─── Update ──────────────────────────────────────────────────────────────

    def _update(self, dt: float):
        self.stars.update(dt)
        self.shake_timer = max(0, self.shake_timer - dt)

        # Player movement (hold Shift for focus/slow)
        keys = pygame.key.get_pressed()
        speed = PLAYER_SPEED * 0.4 if keys[pygame.K_LSHIFT] else PLAYER_SPEED
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.px -= speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.px += speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.py -= speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.py += speed * dt
        self.px = max(16, min(SCREEN_W - 16, self.px))
        self.py = max(16, min(SCREEN_H - 16, self.py))

        # Player shooting
        self.shoot_timer -= dt
        if keys[pygame.K_z] or keys[pygame.K_SPACE]:
            if self.shoot_timer <= 0:
                self.player_bullets.append([self.px - 6, self.py - 15, -500])
                self.player_bullets.append([self.px + 6, self.py - 15, -500])
                self.shoot_timer = 0.1

        # Update player bullets
        new_pb = []
        for pb in self.player_bullets:
            pb[1] += pb[2] * dt
            if pb[1] > 0:
                new_pb.append(pb)
                # Check hit on boss
                if self.boss.alive:
                    dx = pb[0] - self.boss.x
                    dy = pb[1] - self.boss.y
                    if dx * dx + dy * dy < 900:  # ~30px radius
                        self.boss.take_damage(2)
                        self.score += 10
                        # Spawn hit particle
                        extend_particles(
                            self.particles,
                            generate_explosion_particles(pb[0], pb[1], 5, 80)
                        )
                        new_pb.pop()  # remove this bullet
                        continue
        self.player_bullets = new_pb

        # Boss update — get new enemy bullets
        new_enemy = self.boss.update(dt)
        if new_enemy['x']:
            num_new = len(new_enemy['x'])
            extend_bullets(self.bullets, new_enemy)
            for _ in range(num_new):
                self.bullet_colors_idx.append(
                    random.randint(0, len(BULLET_COLORS) - 1)
                )

        # ── Update enemy bullets (tensor_ops) ───────────────────────
        if self.bullets['x']:
            # Use MAX tensor ops with Python fallback
            self.bullets = update_bullets(self.bullets, dt)

        # ── Cull off-screen bullets ──────────────────────────────────────────
        culled_x, culled_y, culled_vx, culled_vy = [], [], [], []
        culled_colors = []
        margin = 40
        for i in range(len(self.bullets['x'])):
            bx = self.bullets['x'][i]
            by = self.bullets['y'][i]
            if -margin < bx < SCREEN_W + margin and -margin < by < SCREEN_H + margin:
                culled_x.append(bx)
                culled_y.append(by)
                culled_vx.append(self.bullets['vx'][i])
                culled_vy.append(self.bullets['vy'][i])
                if i < len(self.bullet_colors_idx):
                    culled_colors.append(self.bullet_colors_idx[i])
                else:
                    culled_colors.append(0)
        self.bullets = {'x': culled_x, 'y': culled_y, 'vx': culled_vx, 'vy': culled_vy}
        self.bullet_colors_idx = culled_colors

        # ── Collision detection ──────────────────────────────────────────────
        self.invincible_timer = max(0, self.invincible_timer - dt)
        if self.bullets['x']:
            hits = []
            for i in range(len(self.bullets['x'])):
                bx = self.bullets['x'][i]
                by = self.bullets['y'][i]
                dx = bx - self.px
                dy = by - self.py
                if dx*dx + dy*dy < (PLAYER_RADIUS + BULLET_RADIUS) ** 2:
                    hits.append(i)

            if hits and self.invincible_timer <= 0:
                self.lives -= 1
                self.invincible_timer = 2.0
                self.shake_timer = 0.3
                # Death explosion particles
                extend_particles(
                    self.particles,
                    generate_explosion_particles(self.px, self.py, 30, 250)
                )
                if self.lives <= 0:
                    self.state = "gameover"

            # ── Graze scoring ────────────────────────────────────────────────
            for i in range(len(self.bullets['x'])):
                bx = self.bullets['x'][i]
                by = self.bullets['y'][i]
                dx = bx - self.px
                dy = by - self.py
                dist_sq = dx*dx + dy*dy
                if dist_sq < GRAZE_RADIUS ** 2 and dist_sq >= (PLAYER_RADIUS + BULLET_RADIUS) ** 2:
                    self.graze_count += 1
                    self.score += 1

        # ── Update particles (tensor_ops) ───────────────────────────
        if self.particles['x']:
            # Use MAX tensor ops with Python fallback
            self.particles = update_particles(self.particles, dt)

        # Trail particles
        if self.frame % 3 == 0:
            extend_particles(
                self.particles,
                generate_trail_particle(self.px, self.py + 14)
            )

        # Victory condition
        if not self.boss.alive:
            self.state = "victory"
            extend_particles(
                self.particles,
                generate_explosion_particles(self.boss.x, self.boss.y, 60, 300)
            )

    # ─── Draw ────────────────────────────────────────────────────────────────

    def _draw(self):
        # Screen shake offset
        sx = random.randint(-3, 3) if self.shake_timer > 0 else 0
        sy = random.randint(-3, 3) if self.shake_timer > 0 else 0

        self.screen.fill(C_BG)
        self.stars.draw(self.screen)

        if self.state == "title":
            self._draw_title()
            return

        # Particles (behind everything)
        for i in range(len(self.particles['x'])):
            px = self.particles['x'][i] + sx
            py_coord = self.particles['y'][i] + sy
            life = self.particles['life'][i]
            alpha = min(255, int(life * 400))
            r = max(1, int(life * 6))
            color = (
                min(255, int(255 * life * 2)),
                min(255, int(160 * life * 2)),
                min(255, int(40 * life * 2)),
            )
            pygame.draw.circle(self.screen, color, (int(px), int(py_coord)), r)

        # Enemy bullets
        for i in range(len(self.bullets['x'])):
            bx = self.bullets['x'][i] + sx
            by = self.bullets['y'][i] + sy
            cidx = self.bullet_colors_idx[i] if i < len(self.bullet_colors_idx) else 0
            draw_bullet(self.screen, bx, by, BULLET_COLORS[cidx])

        # Player bullets
        for pb in self.player_bullets:
            pygame.draw.rect(self.screen, C_PLAYER,
                             (int(pb[0] + sx) - 2, int(pb[1] + sy) - 5, 4, 10))

        # Boss
        self.boss.draw(self.screen, self.frame)

        # Player
        if self.invincible_timer <= 0 or int(self.frame * 0.5) % 2 == 0:
            draw_astronaut(self.screen, self.px + sx, self.py + sy,
                           PLAYER_DRAW_SIZE,
                           invincible=self.invincible_timer > 0,
                           frame=self.frame)
            # Draw hitbox when focusing
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LSHIFT]:
                pygame.draw.circle(self.screen, (255, 255, 255, 128),
                                   (int(self.px + sx), int(self.py + sy)),
                                   PLAYER_RADIUS, 1)

        # HUD
        self._draw_hud()

        # Overlays
        if self.state == "gameover":
            self._draw_overlay("GAME OVER", "Press SPACE to retry",
                               C_ACCENT)
        elif self.state == "victory":
            self._draw_overlay("VICTORY!", f"Score: {self.score}",
                               C_TITLE)

    def _draw_hud(self):
        # Lives
        for i in range(self.lives):
            draw_astronaut(self.screen, 24 + i * 30, SCREEN_H - 24, 16)

        # Score
        score_text = self.font.render(f"Score: {self.score}", True, C_HUD)
        self.screen.blit(score_text, (SCREEN_W - score_text.get_width() - 16, 12))

        # Graze
        graze_text = self.font_small.render(f"Graze: {self.graze_count}",
                                            True, C_GRAZE)
        self.screen.blit(graze_text, (SCREEN_W - graze_text.get_width() - 16, 38))

        # Bullet count (performance indicator)
        n_bullets = len(self.bullets['x'])
        n_particles = len(self.particles['x'])
        engine = "🔥 Mojo" if MOJO_PHYSICS else "🐍 Python"
        perf_text = self.font_small.render(
            f"{engine} | Bullets: {n_bullets} | Particles: {n_particles}",
            True, (100, 100, 140)
        )
        self.screen.blit(perf_text, (12, 12))

    def _draw_title(self):
        # Title
        title = self.font_big.render("MAX ADVENTURE", True, C_TITLE)
        tx = SCREEN_W // 2 - title.get_width() // 2
        ty = SCREEN_H // 3
        self.screen.blit(title, (tx, ty))

        # Subtitle
        sub = self.font.render("A Bullet Hell Journey", True, C_HUD)
        self.screen.blit(sub, (SCREEN_W // 2 - sub.get_width() // 2, ty + 55))

        # Astronaut preview
        bob = math.sin(self.frame * 0.05) * 8
        draw_astronaut(self.screen, SCREEN_W // 2, ty + 130 + bob, 40,
                       frame=self.frame)

        # Instructions
        instructions = [
            "Arrow keys / WASD — Move",
            "SPACE / Z — Shoot",
            "L-Shift — Focus (slow + show hitbox)",
            "",
            "Press SPACE to start",
        ]
        for i, line in enumerate(instructions):
            color = C_ACCENT if "SPACE to start" in line else (140, 150, 180)
            txt = self.font_small.render(line, True, color)
            self.screen.blit(txt, (SCREEN_W // 2 - txt.get_width() // 2,
                                   ty + 200 + i * 24))

        # Powered by
        tech = self.font_small.render(
            "Powered by Mojo 🔥 + MAX Tensors + Pygame", True, (60, 60, 90)
        )
        self.screen.blit(tech, (SCREEN_W // 2 - tech.get_width() // 2,
                                SCREEN_H - 36))

    def _draw_overlay(self, title: str, subtitle: str, color: tuple):
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        t = self.font_big.render(title, True, color)
        self.screen.blit(t, (SCREEN_W // 2 - t.get_width() // 2,
                             SCREEN_H // 2 - 40))

        s = self.font.render(subtitle, True, C_HUD)
        self.screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2,
                             SCREEN_H // 2 + 20))

        hint = self.font_small.render("Press SPACE to play again / ESC to quit",
                                      True, (120, 130, 160))
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2,
                                SCREEN_H // 2 + 60))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Game().run()

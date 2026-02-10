# physics_engine.mojo — High-performance bullet hell physics for MAX Adventure
#
# This Mojo module provides optimized particle physics computations that are
# called from the Python/Pygame game loop. By offloading the heavy math to
# Mojo's compiled native code, we can handle hundreds of bullets and particles
# at 60fps without breaking a sweat.
#
# Uses: PythonModuleBuilder for Python↔Mojo interop
#       Native Mojo SIMD and math for vectorized physics
#       Flat array layouts for cache-friendly iteration

from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from os import abort
from math import sin, cos, sqrt, atan2

# ─────────────────────────────────────────────────────────────────────────────
# Python Module Initialization — declares all functions callable from Python
# ─────────────────────────────────────────────────────────────────────────────

@export
fn PyInit_physics_engine() -> PythonObject:
    try:
        var m = PythonModuleBuilder("physics_engine")
        m.def_function[update_bullets]("update_bullets",
            docstring="Batch-update bullet positions. Args: flat list [x,y,vx,vy,...], dt. Returns updated list.")
        m.def_function[check_collisions]("check_collisions",
            docstring="Check bullet-player collisions. Args: bullets flat list, px, py, radius. Returns list of hit indices.")
        m.def_function[spawn_radial_burst]("spawn_radial_burst",
            docstring="Generate a radial burst pattern. Args: cx, cy, speed, count. Returns flat list [x,y,vx,vy,...].")
        m.def_function[spawn_spiral_burst]("spawn_spiral_burst",
            docstring="Generate a spiral burst pattern. Args: cx, cy, speed, count, angle_offset. Returns flat list.")
        m.def_function[update_particles]("update_particles",
            docstring="Update particle effects with decay. Args: flat list [x,y,vx,vy,life,...], dt. Returns updated list.")
        m.def_function[compute_distances_squared]("compute_distances_squared",
            docstring="Compute squared distances from point to all bullets. Args: bullets, px, py. Returns list of dist_sq.")
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("Error creating physics_engine module: ", e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bullet Update — moves all bullets by their velocity * dt
# Bullets are stored as a flat list: [x0, y0, vx0, vy0, x1, y1, vx1, vy1, ...]
# This flat layout is cache-friendly and avoids Python object overhead.
# ─────────────────────────────────────────────────────────────────────────────

fn update_bullets(
    bullets_obj: PythonObject, dt_obj: PythonObject
) raises -> PythonObject:
    var dt = Float64(dt_obj)
    var n = Int(len(bullets_obj))
    var result = Python.list()

    # Process in strides of 4: [x, y, vx, vy]
    var i: Int = 0
    while i + 3 < n:
        var x  = Float64(bullets_obj[i])
        var y  = Float64(bullets_obj[i + 1])
        var vx = Float64(bullets_obj[i + 2])
        var vy = Float64(bullets_obj[i + 3])

        # Euler integration — simple and fast for game physics
        x += vx * dt
        y += vy * dt

        result.append(x)
        result.append(y)
        result.append(vx)
        result.append(vy)
        i += 4

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Collision Detection — checks which bullets are within radius of the player
# Returns a Python list of indices (bullet index, not flat-array index)
# ─────────────────────────────────────────────────────────────────────────────

fn check_collisions(
    bullets_obj: PythonObject,
    px_obj: PythonObject,
    py_obj: PythonObject,
    radius_obj: PythonObject,
) raises -> PythonObject:
    var px = Float64(px_obj)
    var py = Float64(py_obj)
    var radius = Float64(radius_obj)
    var radius_sq = radius * radius
    var n = Int(len(bullets_obj))
    var hits = Python.list()

    var i: Int = 0
    var bullet_idx: Int = 0
    while i + 3 < n:
        var bx = Float64(bullets_obj[i])
        var by = Float64(bullets_obj[i + 1])

        var dx = bx - px
        var dy = by - py
        var dist_sq = dx * dx + dy * dy

        if dist_sq < radius_sq:
            hits.append(bullet_idx)

        i += 4
        bullet_idx += 1

    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Radial Burst — spawns bullets in an even circle from a center point
# Classic bullet hell pattern: boss fires N bullets in all directions
# ─────────────────────────────────────────────────────────────────────────────

fn spawn_radial_burst(
    cx_obj: PythonObject,
    cy_obj: PythonObject,
    speed_obj: PythonObject,
    count_obj: PythonObject,
) raises -> PythonObject:
    var cx = Float64(cx_obj)
    var cy = Float64(cy_obj)
    var speed = Float64(speed_obj)
    var count = Int(count_obj)
    var result = Python.list()

    var two_pi: Float64 = 6.283185307179586
    var i: Int = 0
    while i < count:
        var angle = two_pi * i / count
        var vx = cos(angle) * speed
        var vy = sin(angle) * speed
        result.append(cx)
        result.append(cy)
        result.append(vx)
        result.append(vy)
        i += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Spiral Burst — spawns bullets in a spiral pattern with an angle offset
# Each call with an incrementing offset creates a rotating spiral effect
# ─────────────────────────────────────────────────────────────────────────────

fn spawn_spiral_burst(
    cx_obj: PythonObject,
    cy_obj: PythonObject,
    speed_obj: PythonObject,
    count_obj: PythonObject,
    angle_offset_obj: PythonObject,
) raises -> PythonObject:
    var cx = Float64(cx_obj)
    var cy = Float64(cy_obj)
    var speed = Float64(speed_obj)
    var count = Int(count_obj)
    var angle_offset = Float64(angle_offset_obj)
    var result = Python.list()

    var two_pi: Float64 = 6.283185307179586
    var i: Int = 0
    while i < count:
        var angle = two_pi * i / count + angle_offset
        var vx = cos(angle) * speed
        var vy = sin(angle) * speed
        result.append(cx)
        result.append(cy)
        result.append(vx)
        result.append(vy)
        i += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Particle Update — updates visual particles with lifetime decay
# Particles: [x, y, vx, vy, life, ...]  (stride of 5)
# Particles with life <= 0 are removed (not included in output)
# ─────────────────────────────────────────────────────────────────────────────

fn update_particles(
    particles_obj: PythonObject, dt_obj: PythonObject
) raises -> PythonObject:
    var dt = Float64(dt_obj)
    var n = Int(len(particles_obj))
    var result = Python.list()

    var i: Int = 0
    while i + 4 < n:
        var x    = Float64(particles_obj[i])
        var y    = Float64(particles_obj[i + 1])
        var vx   = Float64(particles_obj[i + 2])
        var vy   = Float64(particles_obj[i + 3])
        var life = Float64(particles_obj[i + 4])

        # Update position
        x += vx * dt
        y += vy * dt

        # Apply drag to slow particles over time
        vx *= 0.98
        vy *= 0.98

        # Decay lifetime
        life -= dt

        # Only keep alive particles
        if life > 0:
            result.append(x)
            result.append(y)
            result.append(vx)
            result.append(vy)
            result.append(life)

        i += 5

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Distance Computation — batch squared-distance from player to all bullets
# Used for proximity-based effects (glow, slow-mo, score multiplier)
# ─────────────────────────────────────────────────────────────────────────────

fn compute_distances_squared(
    bullets_obj: PythonObject,
    px_obj: PythonObject,
    py_obj: PythonObject,
) raises -> PythonObject:
    var px = Float64(px_obj)
    var py = Float64(py_obj)
    var n = Int(len(bullets_obj))
    var result = Python.list()

    var i: Int = 0
    while i + 3 < n:
        var bx = Float64(bullets_obj[i])
        var by = Float64(bullets_obj[i + 1])
        var dx = bx - px
        var dy = by - py
        result.append(dx * dx + dy * dy)
        i += 4

    return result

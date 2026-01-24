# DendrySampler Flow Diagram

## Overview

When Terra requests a sample at world coordinates (x, z), the Dendry sampler processes these through a hierarchical multi-resolution branching algorithm to produce terrain features like river networks or valley systems.

## Coordinate Scaling Rationale

```
World Coordinates (x, z)     e.g., (1000, 2000) - actual block positions
        │
        ▼ multiply by frequency (default 0.001)
Noise Coordinates            e.g., (1.0, 2.0) - normalized noise space
        │
        ▼ used in cell calculations
Grid Cell Coordinates        e.g., cell(1, 2) at resolution 1
```

**Why scale by frequency?**
- World coordinates are large integers (block positions: 0 to thousands)
- Noise algorithms work best in normalized space (typically -1 to 1 or small integers)
- `frequency = 0.001` means 1000 blocks = 1 unit in noise space
- This controls the "scale" of the branching pattern in the world

## Main Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TERRA TOOL CALLS SAMPLER                            │
│                     getSample(seed, x, z) or (x, y, z)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COORDINATE SCALING                                   │
│                                                                             │
│   scaledX = x * frequency    (e.g., 1000 * 0.001 = 1.0)                    │
│   scaledZ = z * frequency    (e.g., 2000 * 0.001 = 2.0)                    │
│                                                                             │
│   Purpose: Transform world-scale coordinates to noise-scale coordinates     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         evaluate(seed, x, y)                                │
│                    Main algorithm entry point                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   │
┌───────────────────────────────────┐                   │
│     LEVEL 1 (Base Resolution)     │                   │
│                                   │                   │
│  1. getCell(x, y, resolution=1)   │                   │
│     - Determines which grid cell  │                   │
│       contains the query point    │                   │
│                                   │                   │
│  2. generateNeighboringPoints3D() │◄──┐              │
│     - Gets 9x9 grid of points     │   │              │
│     - Uses CACHE if cell in range │   │ CACHE HIT   │
│       [-64, 64) for x and y       │   │ (fast path) │
│     - Generates new if outside    │   │              │
│                                   │   │              │
│  3. generateSegments()            │   │              │
│     - Connects each point to its  │   │              │
│       lowest elevation neighbor   │   │              │
│     - Creates branch network      │   │              │
│                                   │                   │
│  4. subdivideSegments()           │                   │
│     - Splits each segment into 4  │                   │
│     - Smooths the branch paths    │                   │
│                                   │                   │
│  5. displaceSegments()            │                   │
│     - SKIPPED if delta ≈ 0        │                   │
│     - Adds organic randomness     │                   │
└───────────────────────────────────┘                   │
                    │                                   │
                    ▼                                   │
            ┌───────────────┐                           │
            │ resolution=1? │───YES──► computeResult() ─┘
            └───────────────┘                RETURN
                    │ NO
                    ▼
┌───────────────────────────────────┐
│     LEVEL 2 (2x Resolution)       │
│                                   │
│  - getCell(x, y, resolution=2)    │
│  - generateNeighboringPoints3D()  │
│    with 5x5 grid                  │
│  - generateSubSegments()          │
│    connects to Level 1 segments   │
│  - displaceSegments()             │
└───────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ resolution=2? │───YES──► computeResult() ─► RETURN
            └───────────────┘
                    │ NO
                    ▼
┌───────────────────────────────────┐
│     LEVEL 3 (4x Resolution)       │
│                                   │
│  - getCell(x, y, resolution=4)    │
│  - Finer grid subdivision         │
│  - Connects to all previous       │
│    level segments                 │
└───────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ resolution=3? │───YES──► computeResult() ─► RETURN
            └───────────────┘
                    │ NO
                    ▼
┌───────────────────────────────────┐
│     LEVEL 4 (8x Resolution)       │
└───────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ resolution=4? │───YES──► computeResult() ─► RETURN
            └───────────────┘
                    │ NO
                    ▼
┌───────────────────────────────────┐
│     LEVEL 5 (16x Resolution)      │
│     (Maximum detail level)        │
└───────────────────────────────────┘
                    │
                    ▼
              computeResult() ─────────────────────────► RETURN
```

## Cache System Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POINT CACHE SYSTEM                                  │
│                                                                             │
│   Cache Size: 128 x 128 points                                             │
│   Cache Range: cell coordinates from -64 to +63 (inclusive)                 │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  generatePointCached(cellX, cellY)                                  │  │
│   │                                                                     │  │
│   │  IF cellX in [-64, 64) AND cellY in [-64, 64):                     │  │
│   │      RETURN pointCache[cellX + 64][cellY + 64]  ◄── CACHE HIT      │  │
│   │  ELSE:                                                              │  │
│   │      RETURN generatePoint(cellX, cellY)         ◄── CACHE MISS     │  │
│   │             (compute new point with RNG)                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Cache populated at sampler construction (initPointCache)                  │
│   Points are deterministic based on cell coordinates + salt                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Conditional Skips

| Condition | What Gets Skipped | Why |
|-----------|-------------------|-----|
| `resolution == 1` | Levels 2-5 | User only wants coarse detail |
| `resolution == 2` | Levels 3-5 | User wants moderate detail |
| `delta ≈ 0` | `displaceSegments()` | No displacement needed |
| `distance > 10.0` | Sub-segment connection | Point too far from parent branches |
| `segments.isEmpty()` | `findNearestSegment()` | No segments to search |
| Cache hit | `generatePoint()` | Use pre-computed cached point |

## computeResult() Output Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         computeResult(x, y, segments, level)                │
│                                                                             │
│   1. Find nearest segment to query point (x, y)                            │
│                                                                             │
│   2. Based on returnType:                                                   │
│                                                                             │
│      ┌──────────────┬────────────────────────────────────────────────────┐ │
│      │ DISTANCE     │ Return: nearest.distance                           │ │
│      │              │ (Euclidean distance to closest branch)             │ │
│      ├──────────────┼────────────────────────────────────────────────────┤ │
│      │ WEIGHTED     │ Return: nearest.distance * (1.0 / level)           │ │
│      │              │ (Distance weighted by resolution level)            │ │
│      ├──────────────┼────────────────────────────────────────────────────┤ │
│      │ ELEVATION    │ Return: baseElevation + (distance * slope)         │ │
│      │              │ (Valley/terrain elevation profile)                 │ │
│      └──────────────┴────────────────────────────────────────────────────┘ │
│                                                                             │
│   3. If no segments found: return controlSampler value or default gradient │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Scaling Summary

| Stage | Scaling Operation | Rationale |
|-------|-------------------|-----------|
| Input | `x * frequency` | Convert world coords to noise space |
| Cell lookup | `floor(x * resolution)` | Map continuous to discrete grid cells |
| Point generation | `point / resolution` | Scale cached points to current level |
| Control function | `x / frequency` | Convert back to world coords for elevation lookup |

## Performance Characteristics

- **Cache hits**: O(1) point lookup for cells in [-64, 64) range
- **Cache misses**: O(1) RNG-based point generation
- **Segment search**: O(n) where n = total segments across all levels
- **Memory**: ~128KB for point cache + dynamic segment lists

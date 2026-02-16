# DendryTerra

DendryTerra is a procedural river network generator that produces dendritic (tree-like) flow patterns. It generates hierarchical segment networks within a tiling grid and provides several output modes for terrain integration.

## Coordinate Spaces

DendryTerra operates in two coordinate spaces:

- **Sampler coordinates**: The world-space input coordinates passed to `getSample(seed, x, z)`. One unit = one block/voxel.
- **Grid coordinates**: Internal normalized coordinates where `gridCoord = samplerCoord / gridsize`. One unit = one cell width.

All slope values (`slope`, `slopeWhenStraight`, `lowestSlopeCutoff`) are measured as **elevation change per grid unit** (dz per cell width). A slope of 0.5 means 0.5 elevation change across one full cell.

IMPORTANT: As slope definitions are in units of cells, these MUST be updated if want the same real world slope limitations applied and change the grid spacing.  Keep in mind the control sampler may also be scaled after this sampler.  Consider the real world slope limitations you want when modifying slope related parameters.  If you want a real world slope of 1, you need to multiply it by the gridspacing and divide it by the actual range that a single 1 unit change of control function maps to actual world height changes.

Thus, or a real world slope of 1:  The defined slope = 1 * (2000 gridspacing) / (200 world unit height to 1 unit of y height) = 10.

## Return Types

The `return` parameter selects what value `getSample` produces. Different return types use different code paths, so not all parameters affect all return types.

| Return Type | Description | Requires |
|---|---|---|
| `DISTANCE` | Euclidean distance to nearest segment (grid coords) | — |
| `WEIGHTED` | Level-weighted distance to nearest segment | — |
| `ELEVATION` | Elevation at nearest segment point + slope contribution | — |
| `PIXEL_ELEVATION` | Cached pixel elevation with slope contribution | `cachepixels > 0` |
| `PIXEL_LEVEL` | Cached pixel resolution level | `cachepixels > 0` |
| `PIXEL_DEBUG` | Cached pixel point type visualization | `cachepixels > 0` |
| `PIXEL_RIVER_LEGACY` | River/border/outside classification (0/1/2) using pixel cache | `cachepixels > 0` |
| `PIXEL_RIVER` | BigChunk-cached distance (de-quantized to sampler units) | — |
| `PIXEL_RIVER_CTRL` | BigChunk-cached elevation (de-quantized) | — |

### PIXEL_RIVER y-input selector

When `return` is `PIXEL_RIVER`, calling the 4-argument form `getSample(seed, x, y, z)` with `y == 1.0` returns elevation instead of distance (same as `PIXEL_RIVER_CTRL`). Any other `y` value falls through to normal distance output. This allows a single sampler instance to provide both distance and elevation.

## Parameters

### Segment Generation (affects ALL return types)

These parameters control how the river network is generated. Since all return types share the same segment generation pipeline, changing any of these changes the network shape for every return type.

#### `n`
- **Config key**: `n`
- **Type**: int
- **Default**: `2`
- **Range**: 0–5
- **Effect**: Resolution level / recursion depth. Level 0 produces the base constellation (trunk rivers). Each additional level subdivides cells and adds finer tributary branches. Higher values = more detailed networks but exponentially more computation.

#### `epsilon`
- **Config key**: `epsilon`
- **Type**: double
- **Default**: `0.0`
- **Range**: [0, 0.5)
- **Effect**: Inset margin for star point placement within cells. Points are placed in the range `[epsilon, 1 - epsilon]` within each cell. A value of 0 allows points at cell edges; higher values push points toward cell centers, reducing edge-hugging segments.

#### `gridsize`
- **Config key**: `gridsize`
- **Type**: double
- **Default**: `2000.0`
- **Range**: > 0
- **Effect**: Size of one base grid cell in sampler (world) units. A gridsize of 2000 means each cell covers 2000x2000 blocks. This is the fundamental scale parameter — it determines how large river networks are relative to the world. All internal coordinates are normalized by this value.

#### `sampler` (control function)
- **Config key**: `sampler`
- **Type**: Sampler (nullable)
- **Default**: `null`
- **Effect**: Provides the base elevation landscape. Queried in sampler coordinates to determine point elevations. Rivers flow downhill according to this elevation field. When null, a default linear ramp (`x * 0.1` in grid coords) is used.

#### `salt`
- **Config key**: `salt`
- **Type**: long
- **Default**: `0`
- **Effect**: Seed modifier for deterministic random generation. Different salts produce different network layouts from the same control function.

#### `branches` / `default-branches`
- **Config key**: `branches` (Sampler, nullable), `default-branches` (int)
- **Defaults**: `null` / `1`
- **Range**: `default-branches` 1–8
- **Effect**: Number of initial star points per cell at level 0. When a `branches` sampler is provided, it is queried at each cell center (in sampler coordinates) and the result is clamped to [1, 8]. Otherwise `default-branches` is used. More branches = denser networks with more river origins per cell.

#### `curvature`
- **Config key**: `curvature`
- **Type**: double
- **Default**: `0.9`
- **Range**: [0, 1]
- **Effect**: Hermite spline curvature factor for segment subdivision. 0 = linear interpolation (straight segments, splines disabled), 1 = full spline curvature (smooth curves). Controls the visual smoothness of river bends. When set to 0, spline subdivision is completely disabled and simple linear subdivision is used instead (faster but produces straight-line segments between nodes).

#### `tangent-angle`
- **Config key**: `tangent-angle`
- **Type**: double (degrees)
- **Default**: `45.0`
- **Range**: 0–90
- **Effect**: Maximum random twist angle for spline tangents at unconnected nodes. At 0, tangents point directly along the flow/slope direction (minimal curvature). At 90, tangents can deviate up to perpendicular to the connection direction (maximum curvature). The actual twist applied is scaled by terrain slope — steep terrain uses the full angle, flat terrain reduces it (see `slope-when-straight`).

#### `tangent-strength`
- **Config key**: `tangent-strength`
- **Type**: double
- **Default**: `1.0`
- **Range**: [0, 1]
- **Effect**: Strength of spline tangent control points as a fraction of segment length. Controls how far the Hermite control point extends from the node. 0 = control point at node (effectively linear), 1 = full tangent length. Higher values create more pronounced curves.

#### `slope-when-straight`
- **Config key**: `slope-when-straight`
- **Type**: double
- **Default**: `10`
- **Units**: elevation per grid unit
- **Effect**: Terrain slope threshold that controls tangent alignment. When terrain slope >= this value, tangents align fully with the downhill gradient (rivers follow terrain faithfully). When slope is lower, tangent twist is reduced proportionally — allowing rivers to meander more on flatter terrain. Also controls elevation centroid offset in BigChunk rendering: steep segments use tighter centroid alignment.

#### `lowest-slope-cutoff`
- **Config key**: `lowest-slope-cutoff`
- **Type**: double
- **Default**: `-1`
- **Units**: normalized slope (heightDiff / dist^falloffPower, in grid coordinates)
- **Effect**: Minimum normalized slope for neighbor connection at levels 1+. During network construction, candidate connections with normalized slope below this cutoff are rejected. Positive values enforce strictly downhill flow; negative values allow slightly uphill connections (useful for flat terrain where elevation noise might cause minor reversals). Does not apply to level 0 (constellation trunks).

#### `constellation-scale`
- **Config key**: `constellation-scale`
- **Type**: int
- **Default**: `1`
- **Range**: 1–10
- **Effect**: Scale factor for constellation tiling. At scale 1, the largest inscribed square (without tilting) is 3 grid cells wide. Higher values create larger constellations that span more cells, producing longer trunk rivers. The number of neighboring constellations that must be computed increases with scale.

#### `constellation-shape`
- **Config key**: `constellation-shape`
- **Type**: enum
- **Default**: `SQUARE`
- **Options**: `SQUARE`, `HEXAGON`, `RHOMBUS`
- **Effect**: Tiling pattern for constellations. `SQUARE` uses standard 4-neighbor grid tiling. `HEXAGON` uses 6-neighbor hexagonal tiling for more uniform neighbor distances. `RHOMBUS` uses rotated square (diamond) tiling with 4 neighbors.

---

### Output Parameters (affect specific return types only)

#### `slope`
- **Config key**: `slope`
- **Type**: double
- **Default**: `0.1`
- **Affects**: `ELEVATION`, `PIXEL_ELEVATION` only
- **Does NOT affect**: `PIXEL_RIVER`, `PIXEL_RIVER_CTRL`, `DISTANCE`, `WEIGHTED`, `PIXEL_LEVEL`, `PIXEL_DEBUG`, `PIXEL_RIVER_LEGACY`
- **Effect**: Elevation contribution per unit distance from the nearest segment. In the `ELEVATION` return type, the output is `segmentElevation + distance * slope` (distance in grid units). In `PIXEL_ELEVATION`, the formula is `pixelElevation + distance * slope * gridsize` (note the gridsize multiplier — this is a known inconsistency between the two paths).

#### `riverwidth` / `default-riverwidth`
- **Config key**: `riverwidth` (Sampler, nullable), `default-riverwidth` (double)
- **Defaults**: `null` / `16.0`
- **Units**: sampler (world) units
- **Affects**: `PIXEL_RIVER_LEGACY`, `PIXEL_RIVER`, `PIXEL_RIVER_CTRL`
- **Does NOT affect**: `DISTANCE`, `WEIGHTED`, `ELEVATION`, `PIXEL_ELEVATION`, `PIXEL_LEVEL`, `PIXEL_DEBUG`
- **Effect**: Base river width before level falloff. When a `riverwidth` sampler is provided, it is queried at the point location in sampler coordinates. River width at each level = `baseWidth * 0.6^level`, with a minimum of `2 * cachepixels`. In PIXEL_RIVER/PIXEL_RIVER_CTRL, this controls how wide rivers are drawn into the BigChunk cache. Width transitions are applied at junctions where a higher-level river flows into a lower-level river.

#### `borderwidth` / `default-borderwidth`
- **Config key**: `borderwidth` (Sampler, nullable), `default-borderwidth` (double)
- **Defaults**: `null` / `20.0`
- **Units**: sampler (world) units
- **Affects**: `PIXEL_RIVER_LEGACY` only
- **Does NOT affect**: `PIXEL_RIVER`, `PIXEL_RIVER_CTRL`, and all other return types
- **Effect**: Width of the border zone around rivers. Points within `riverWidth + borderWidth` of a segment return 1 (border). Only used by the legacy river classification system.

#### `max`
- **Config key**: `max`
- **Type**: double
- **Default**: `2.0`
- **Range**: > 0
- **Affects**: `PIXEL_RIVER_CTRL` (and `PIXEL_RIVER` for elevation storage)
- **Effect**: Maximum expected elevation value from the control function. Used to quantize elevation to UInt8 (0–255) in the BigChunk cache. Elevation values above `max` will be clipped to 255. Set this to match the maximum output of your control sampler.

#### `max-dist`
- **Config key**: `max-dist`
- **Type**: double
- **Default**: `50.0`
- **Range**: > 0, must be >= `default-riverwidth + default-borderwidth`
- **Units**: sampler (world) units
- **Affects**: `PIXEL_RIVER`, `PIXEL_RIVER_CTRL`
- **Effect**: Maximum distance value for UInt8 quantization in the BigChunk cache. Also determines how far from each segment the renderer projects distance/elevation data. Points beyond `max-dist / gridsize` grid units from any segment receive zero values. Larger values capture more terrain around rivers but reduce distance precision (fewer quantization levels per unit).

#### `cachepixels`
- **Config key**: `cachepixels`
- **Type**: double
- **Default**: `0`
- **Range**: [0, gridsize], and `gridsize / cachepixels <= 65535`
- **Units**: sampler (world) units per pixel
- **Affects**: `PIXEL_ELEVATION`, `PIXEL_LEVEL`, `PIXEL_DEBUG`, `PIXEL_RIVER_LEGACY`, `PIXEL_RIVER`, `PIXEL_RIVER_CTRL`
- **Effect**: Resolution of the pixel cache. Determines the size of each cached pixel in world units. Lower values = higher resolution = more memory per cell. For PIXEL_RIVER/PIXEL_RIVER_CTRL, this sets the block size of the BigChunk grid (each block = `cachepixels x cachepixels` world units). Required to be > 0 for all PIXEL_* return types. Also sets the minimum river width floor (`2 * cachepixels`).

---

### Performance / Debug Parameters

#### `use-parallel`
- **Config key**: `use-parallel`
- **Type**: boolean
- **Default**: `true`
- **Affects**: `DISTANCE`, `WEIGHTED`, `ELEVATION` only
- **Effect**: Enable parallel stream processing for nearest-segment search. Only activates when segment count exceeds `parallel-threshold`. Has no effect on PIXEL_* return types which use their own rendering pipeline.

#### `parallel-threshold`
- **Config key**: `parallel-threshold`
- **Type**: int
- **Default**: `100`
- **Affects**: `DISTANCE`, `WEIGHTED`, `ELEVATION` only
- **Effect**: Minimum segment count before parallel streams are used in nearest-segment search.

#### `debug-timing`
- **Config key**: `debug-timing`
- **Type**: boolean
- **Default**: `false`
- **Effect**: Enable performance timing output. Logs average sample time and pixel cache hit rates periodically.

#### `debug`
- **Config key**: `debug`
- **Type**: int
- **Default**: `0`
- **Effect**: Debug visualization level. Controls what subset of the network is generated for visual debugging. Values: 0 = normal, 5 = stars only (first constellation), 6 = stars only (all), 10 = first constellation segments, 15 = all constellations trunk only (before stitching), 20 = all constellations (before stitching), 30 = all constellations (with stitching), 40 = level 1+ points as zero-length segments.

## Memory and Caching

DendryTerra uses several caching layers depending on return type:

| Cache | Size Limit | Used By |
|---|---|---|
| SegmentList LRU cache | 10 MB | `PIXEL_RIVER`, `PIXEL_RIVER_CTRL` |
| BigChunk LRU cache | 10 MB | `PIXEL_RIVER`, `PIXEL_RIVER_CTRL` |
| Pixel cache (per-cell) | 10 MB | `PIXEL_ELEVATION`, `PIXEL_LEVEL`, `PIXEL_DEBUG`, `PIXEL_RIVER_LEGACY` |

For `PIXEL_RIVER` / `PIXEL_RIVER_CTRL`, the SegmentList and BigChunk caches are only allocated when those return types are selected. Other return types skip allocation entirely.

## Architecture Overview

1. **Constellation generation**: Stars (initial points) are placed in a tiling grid pattern. Each constellation covers multiple grid cells.
2. **Slope estimation**: Each star's terrain slope is estimated from its neighbors by solving a 2x2 linear system.
3. **Network construction**: Stars are connected into segments by following downhill gradients. Level 0 creates trunk rivers; levels 1+ add tributaries.
4. **Spline subdivision**: Segments are subdivided using Hermite splines with tangents influenced by terrain slope and the `tangent-angle`/`tangent-strength` parameters.
5. **Stitching**: Segments are clipped at cell boundaries and stitched across cells for seamless tiling.
6. **Rendering**: For PIXEL_RIVER modes, segments are rasterized into 256x256 BigChunk blocks with quantized distance and elevation. Width transitions are applied at tributary junctions.

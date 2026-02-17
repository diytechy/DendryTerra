# PIXEL_RIVER Enhancement Implementation Plan

## Overview

Enhance the PIXEL_RIVER solver to:
- Fill in missing sections from segment ends (semicircle fill at endpoints with 1 connection)
- Fill in pin-holes (adjacent box "blot" filling on outward steps > 1)
- Smooth elevation changes with curved falloff toward river flow (3-layer elevation tracking with centroid distance)
- Clean transitions between segments (elevation smoothing noise at river edge)

---

## Phase 1: BigChunk Initialization Change

### 1.1: Elevation initialized to 255 instead of 0
**File**: `BigChunk.java:71-74`

```java
public BigChunkBlock() {
    this.elevation = (byte) 255;  // Max elevation initially (unset marker)
    this.distance = (byte) 255;   // Max distance initially (unset marker)
}
```

**Why**: Elevation 255 = "unset". When we check `distance < 255`, we know the box was previously written. The elevation initialization to max ensures higher-level (numerically larger) segments set elevation first, and lower-level segments can overwrite with lower values.

---

## Phase 2: Connection Count Collection

### 2.1: Add sister arrays for start/end connection counts
**File**: `DendrySampler.java` - `collectSegmentsForBigChunk`

Add two more parallel output lists. Extract connection counts from `NetworkPoint` via `SegmentList`:

```java
private void collectSegmentsForBigChunk(BigChunk chunk, double chunkSizeGrid,
                                       List<Segment3D> outSegments, List<Integer> outLevels,
                                       List<Integer> outStartConns, List<Integer> outEndConns) {
    // ... existing cell iteration ...
    for (SegmentIdx segIdx : segmentList.getSegments()) {
        // ... existing boundary check ...
        if (srtNear || endNear) {
            Segment3D seg3d = segIdx.resolve(segmentList);
            outSegments.add(seg3d);
            outLevels.add(segIdx.level);
            // NEW: Extract connection counts from endpoint NetworkPoints
            outStartConns.add(segmentList.getPoint(segIdx.srtIdx).connections);
            outEndConns.add(segmentList.getPoint(segIdx.endIdx).connections);
        }
    }
}
```

---

## Phase 3: Segment Sorting by Level

### 3.1: Sort segments highest level first in computeBigChunk
**File**: `DendrySampler.java` - `computeBigChunk`

Sort all parallel arrays by level descending (numeric level N → 0). No need to sub-sort within same level.

```java
private void computeBigChunk(BigChunk chunk) {
    double chunkSizeGrid = getBigChunkSizeGrid();

    List<Segment3D> segments = new ArrayList<>();
    List<Integer> levels = new ArrayList<>();
    List<Integer> startConns = new ArrayList<>();
    List<Integer> endConns = new ArrayList<>();

    collectSegmentsForBigChunk(chunk, chunkSizeGrid, segments, levels, startConns, endConns);

    // Sort by level descending (highest level first)
    Integer[] indices = new Integer[segments.size()];
    for (int i = 0; i < indices.length; i++) indices[i] = i;
    Arrays.sort(indices, (a, b) -> Integer.compare(levels.get(b), levels.get(a)));

    for (int idx : indices) {
        sampleSegmentAlongSpline(segments.get(idx), levels.get(idx),
            startConns.get(idx), endConns.get(idx), chunk, chunkSizeGrid);
    }
}
```

---

## Phase 4: Pre-quantize Elevation in updateBox

### 4.1: Change updateBox to accept pre-quantized elevation
**File**: `DendrySampler.java` - `updateBox`

Currently `updateBox` computes `elevation * elevQuantizeRes` internally. Change it to accept the quantized UInt8 value directly, since the caller needs to know the quantized value for elevation tracking logic.

**Current signature**:
```java
private void updateBox(BigChunk.BigChunkBlock box, double distanceGrid,
                      double elevation, double riverWidthGrid)
```

**New signature**:
```java
private void updateBox(BigChunk.BigChunkBlock box, double distanceGrid,
                      int elevationU8, double riverWidthGrid,
                      boolean blotAdjacentBoxes, int blockX, int blockY, BigChunk chunk)
```

Changes:
- `elevation` → `elevationU8` (pre-quantized)
- Remove `elevQuantizeRes` calculation from updateBox
- Add `blotAdjacentBoxes` flag for adjacent box filling
- Add `blockX`, `blockY`, `chunk` for adjacent box access

### 4.2: Add adjacent box filling (blot) with compile-time flag

```java
private static final boolean ENABLE_BLOT_FILLING = true;  // Compile-time toggle

private void updateBox(BigChunk.BigChunkBlock box, double distanceGrid,
                      int elevationU8, double riverWidthGrid,
                      boolean blotAdjacentBoxes, int blockX, int blockY, BigChunk chunk) {
    // Distance normalization (same as current)
    double normalizedDist;
    if (distanceGrid < riverWidthGrid) {
        normalizedDist = (distanceGrid / riverWidthGrid) / gridsize;
    } else {
        normalizedDist = distanceGrid - riverWidthGrid;
    }

    double distQuantizeRes = 255.0 / maxDistGrid;
    int distU8 = (int) Math.min(255, Math.max(0, normalizedDist * distQuantizeRes));

    // Elevation noise smoothing (Phase 7)
    int finalElevU8 = elevationU8;
    // ... noise logic applied here (see Phase 7) ...

    // Update this box
    applyBoxUpdate(box, distU8, finalElevU8);

    // Blot: fill 4 adjacent boxes with same values
    if (ENABLE_BLOT_FILLING && blotAdjacentBoxes) {
        int[][] neighbors = {{-1,0},{1,0},{0,-1},{0,1}};
        for (int[] d : neighbors) {
            int nx = blockX + d[0], ny = blockY + d[1];
            if (nx >= 0 && nx < 256 && ny >= 0 && ny < 256) {
                applyBoxUpdate(chunk.getBlock(nx, ny), distU8, finalElevU8);
            }
        }
    }
}

private void applyBoxUpdate(BigChunk.BigChunkBlock box, int distU8, int elevU8) {
    if (distU8 < box.getDistanceUnsigned()) {
        box.setDistanceUnsigned(distU8);
    }
    if (elevU8 < box.getElevationUnsigned()) {
        box.setElevationUnsigned(elevU8);
    }
}
```

**Note**: Elevation comparison changes from `>` to `<`. Since we initialize elevation to 255 (max) and process highest levels first, we want to keep the LOWEST elevation (closest to river). Higher-priority (higher-level) segments set their elevation first; lower-level segments can overwrite with lower values.

---

## Phase 5: Update sampleSegmentAlongSpline

### 5.1: New method signature
```java
private void sampleSegmentAlongSpline(Segment3D seg, int level,
    int startConnections, int endConnections,
    BigChunk chunk, double chunkSizeGrid)
```

### 5.2: Compute segment slope
```java
double heightChange = Math.abs(seg.end.z - seg.srt.z);
double euclideanDist = seg.length();
double segmentSlope = (euclideanDist > MathUtils.EPSILON)
    ? heightChange / euclideanDist : 0.0;
```

### 5.3: Compute elevQuantizeRes once and quantize elevation per sample
```java
double elevQuantizeRes = 255.0 / max;
double cachepixelsGrid = cachepixels / gridsize;
```

### 5.4: New sampling loop with stream tracking

State variables:
```java
// Elevation tracking (3 layers)
int outerElev = 0, innerElev = 0, centralElev = 0;
double outerRadius = 0, innerRadius = 0, centralRadius = 0;

// Stream tracking
boolean isNewStream = true;
boolean elevationChanged = false;
Vec2D prevEvalTangent = null;
Point3D prevEvalPos = null;

// Previous loop tangent (for condition B)
Vec2D prevTangent = null;
```

### 5.5: Loop body

```
for each sample i from 0 to numSamples-1:
    t = i / (numSamples - 1)

    // A. If first sample or previous was out of bounds: NEW STREAM
    if (i == 0 || wasOutOfBounds):
        point = evaluateHermiteSpline(seg, t)
        currentTangent = interpolateTangent(seg, t)
        quantizedElev = clamp(0, 255, point.z * elevQuantizeRes)

        outerElev = innerElev = centralElev = quantizedElev
        outerRadius = innerRadius = centralRadius = 0
        prevTangent = currentTangent
        prevEvalPos = point
        prevEvalTangent = currentTangent
        isNewStream = true
        elevationChanged = false

    // B. Else: CONTINUED STREAM
    else:
        // prevTangent already holds the previous loop iteration's currentTangent

    // C. Compute current sample
    point = evaluateHermiteSpline(seg, t)
    currentTangent = interpolateTangent(seg, t)

    // Boundary check
    if (!isPointNearChunk(point, chunk, chunkSizeGrid)):
        wasOutOfBounds = true
        prevTangent = currentTangent
        continue
    wasOutOfBounds = false

    // Compute potential quantized elevation
    quantizedElev = clamp(0, 255, point.z * elevQuantizeRes)
    elevationChanged = false

    if (quantizedElev != centralElev):
        outerElev = innerElev
        innerElev = centralElev
        centralElev = quantizedElev
        outerRadius = innerRadius
        innerRadius = centralRadius
        centralRadius = 1.0  // 1.0 in normalized (riverWidth) space
        elevationChanged = true

    // D. Evaluate criteria: should this sample project to boxes?
    shouldEvaluate = false
    if (isNewStream): shouldEvaluate = true
    if (angleBetween(prevEvalTangent, currentTangent) > 90°): shouldEvaluate = true
    if (distance2D(point, prevEvalPos) > 0.7 * cachepixelsGrid): shouldEvaluate = true
    if (i == numSamples - 1): shouldEvaluate = true  // last sample
    if (elevationChanged): shouldEvaluate = true

    if (!shouldEvaluate):
        prevTangent = currentTangent
        continue

    // E. Update elevation radii
    riverWidthGrid = calculateRiverWidth(level, point.x, point.y)

    if (!elevationChanged && prevEvalPos != null):
        distSinceLastEval = distance2D(point, prevEvalPos) / riverWidthGrid
        outerRadius = max(0, outerRadius - distSinceLastEval)
        innerRadius = max(0, innerRadius - distSinceLastEval)
        centralRadius = max(0, centralRadius - distSinceLastEval)

    // Saturate radii to distance from segment end (normalized by river width)
    distToEndNorm = distance2D(point, seg.end) / riverWidthGrid
    outerRadius = min(outerRadius, max(0, distToEndNorm - 1))
    innerRadius = min(innerRadius, max(0, distToEndNorm - 1))
    centralRadius = min(centralRadius, max(0, distToEndNorm - 1))

    // F. Segment fill flag
    segmentFill = false
    isStartPoint = (i == 0)
    isEndPoint = (i == numSamples - 1)
    if (isStartPoint && startConnections == 1): segmentFill = true
    if (isEndPoint && endConnections == 1): segmentFill = true

    // Project to boxes (updated method)
    projectConeToBoxes(point, currentTangent, prevEvalTangent,
        centralElev, innerElev, outerElev,
        centralRadius, innerRadius, outerRadius,
        riverWidthGrid, segmentFill, isStartPoint,
        segmentSlope, chunk, chunkSizeGrid)

    // Update state for next iteration
    prevEvalPos = point
    prevEvalTangent = currentTangent
    isNewStream = false
    prevTangent = currentTangent
```

**Key points**:
- `prevTangent` tracks the tangent from the previous LOOP iteration (for Step B)
- `prevEvalTangent` tracks the tangent from the previous EVALUATED sample (for condition D and cone filling)
- `prevEvalPos` tracks the position from the previous EVALUATED sample (for condition D and radius decrement)
- Elevation radii are all in normalized (distance/riverWidth) space, range [0, 1]

---

## Phase 6: Update projectConeToBoxes

### 6.1: New signature
```java
private void projectConeToBoxes(
    Point3D samplePoint, Vec2D currentTangent, Vec2D prevTangent,
    int centralElev, int innerElev, int outerElev,
    double centralRadius, double innerRadius, double outerRadius,
    double riverWidthGrid, boolean segmentFill, boolean isStartPoint,
    double segmentSlope, BigChunk chunk, double chunkSizeGrid)
```

### 6.2: Cone angle calculation

The cone should sweep from prevTangent to currentTangent:
```java
double coneAngle;
double bowDirection;  // center of the cone sweep (radians)

if (segmentFill) {
    // Semicircle fill for endpoints with 1 connection
    coneAngle = Math.PI;  // 180 degrees
    if (isStartPoint) {
        // Perpendicular 90° clockwise from currentTangent
        bowDirection = Math.atan2(currentTangent.y, currentTangent.x) - Math.PI / 2;
    } else {
        // Perpendicular 90° counterclockwise from currentTangent
        bowDirection = Math.atan2(currentTangent.y, currentTangent.x) + Math.PI / 2;
    }
} else {
    // Normal cone: sweep from prevTangent to currentTangent
    coneAngle = calculateConeAngle(prevTangent, currentTangent);
    bowDirection = calculateBowDirection(prevTangent, currentTangent);
}
```

Updated `calculateConeAngle` and `calculateBowDirection` to only take 2 tangents (prev + current).

### 6.3: Arc length check for large gaps

Per line 1316: If the arc swept is large (exceeds cachepixelsGrid), we need to split the projection into sub-evaluations:
```java
// Check if arc length at maxDistGrid exceeds pixelcache spacing
double maxArcLength = coneAngle * maxDistGrid;
if (maxArcLength > cachepixelsGrid) {
    // Split into multiple sub-cones from prevTangent to currentTangent
    int numSubCones = (int) Math.ceil(maxArcLength / cachepixelsGrid);
    // ... evaluate each sub-cone independently
}
```

### 6.4: Outward stepping with elevation layer selection

For each step outward:
```java
double cachepixelsGrid = cachepixels / gridsize;
int maxSteps = (int) Math.ceil(maxDistGrid / cachepixelsGrid);
double maxSlope = /* need to define - possibly from config or constant */;

for (int step = 0; step <= maxSteps; step++) {
    double distanceGrid = step * cachepixelsGrid;
    if (step > 0 && distanceGrid > maxDistGrid) break;

    // Normalized distance from river center
    double normDistFromCenter = distanceGrid / riverWidthGrid;

    // Determine which elevation to use
    int selectedElev = centralElev;  // default

    if (normDistFromCenter < 1.0) {
        // Check elevation layers from biggest radius to smallest
        double slopeFactor = Math.max(0, 1.0 - segmentSlope / maxSlope);

        // Try outer (biggest radius)
        if (outerRadius > 0) {
            double centroidDist = Math.sqrt(
                normDistFromCenter * normDistFromCenter +
                Math.pow(outerRadius * slopeFactor, 2)
            );
            if (centroidDist < 1.0) {
                selectedElev = outerElev;
                // Found match, skip others
                goto project;
            }
        }

        // Try inner
        if (innerRadius > 0) {
            double centroidDist = Math.sqrt(
                normDistFromCenter * normDistFromCenter +
                Math.pow(innerRadius * slopeFactor, 2)
            );
            if (centroidDist < 1.0) {
                selectedElev = innerElev;
                goto project;
            }
        }

        // Default: centralElev (already set)
    }

    // project:
    boolean blotAdjacentBoxes = (step > 0);  // Not first step

    // ... project to box positions along the cone ...
    // ... call updateBox with selectedElev, blotAdjacentBoxes ...
}
```

**Note**: In Java there's no `goto` - use a helper method or break-out logic instead.

### 6.5: Updated projectConeToBoxes outward loop

The existing structure projects along perpendicular and negative perpendicular. The new version should sweep through the cone angle range. For each step outward, sample positions across the cone arc:

```java
for (int step = 0; step <= maxSteps; step++) {
    double distanceGrid = step * cachepixelsGrid;
    if (step > 0 && distanceGrid > maxDistGrid) break;

    // ... elevation layer selection (above) ...

    boolean blotAdjacentBoxes = ENABLE_BLOT_FILLING && (step > 0);

    if (step == 0) {
        // Center point - just set the box at samplePoint
        int bx = gridToBlockIndex(samplePoint.x, chunk.gridOriginX, cachepixelsGrid);
        int by = gridToBlockIndex(samplePoint.y, chunk.gridOriginY, cachepixelsGrid);
        if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
            updateBox(chunk.getBlock(bx, by), 0, selectedElev, riverWidthGrid,
                     false, bx, by, chunk);
        }
    } else {
        // Arc samples at this radius
        double arcLength = coneAngle * distanceGrid;
        int numArcSamples = Math.max(2, (int) Math.ceil(arcLength / (cachepixelsGrid * 0.5)));

        // Sample across the full cone arc on both sides of perpendicular
        for (int a = 0; a < numArcSamples; a++) {
            double angleOffset = coneAngle * ((double) a / (numArcSamples - 1) - 0.5);
            double angle = bowDirection + angleOffset;

            double px = samplePoint.x + Math.cos(angle) * distanceGrid;
            double py = samplePoint.y + Math.sin(angle) * distanceGrid;

            int bx = gridToBlockIndex(px, chunk.gridOriginX, cachepixelsGrid);
            int by = gridToBlockIndex(py, chunk.gridOriginY, cachepixelsGrid);

            if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
                updateBox(chunk.getBlock(bx, by), distanceGrid, selectedElev,
                         riverWidthGrid, blotAdjacentBoxes, bx, by, chunk);
            }
        }

        // Also project on the opposite side (negative perpendicular)
        for (int a = 0; a < numArcSamples; a++) {
            double angleOffset = coneAngle * ((double) a / (numArcSamples - 1) - 0.5);
            double angle = bowDirection + Math.PI + angleOffset;

            double px = samplePoint.x + Math.cos(angle) * distanceGrid;
            double py = samplePoint.y + Math.sin(angle) * distanceGrid;

            int bx = gridToBlockIndex(px, chunk.gridOriginX, cachepixelsGrid);
            int by = gridToBlockIndex(py, chunk.gridOriginY, cachepixelsGrid);

            if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
                updateBox(chunk.getBlock(bx, by), distanceGrid, selectedElev,
                         riverWidthGrid, blotAdjacentBoxes, bx, by, chunk);
            }
        }
    }
}
```

---

## Phase 7: Elevation Smoothing Noise

### 7.1: In updateBox, add noise at river edge transitions

```java
// In updateBox, after determining finalElevU8:
int currentElev = box.getElevationUnsigned();
int currentDist = box.getDistanceUnsigned();

// Smoothing: if box was previously set (distance < 255) and new elevation is lower
// and we're at the first outward step from river edge, add random noise
if (currentElev > elevationU8 && currentDist < 255 && outwardStep == 1) {
    // Random value between new elevation and current elevation
    // No need for deterministic seeding - won't affect BigChunk tiling
    int range = currentElev - elevationU8;
    int noise = (int)(Math.random() * range);
    finalElevU8 = elevationU8 + noise;
}
```

**Note**: `outwardStep == 1` means the box is 1 cachepixel distance from the river edge (the first ring of boxes outside the river center).

---

## Resolved Decisions

### maxSlope
Use existing `slopeWhenStraight` parameter. The centroid distance formula becomes:
```java
double slopeFactor = Math.max(0, 1.0 - segmentSlope / slopeWhenStraight);
```

### Elevation comparison
Lower elevation wins (`<` comparison). Rivers carve valleys, so lower = closer to river bed.
```java
if (elevU8 < box.getElevationUnsigned()) {
    box.setElevationUnsigned(elevU8);
}
```

---

## Implementation Order

1. **Phase 1**: BigChunk initialization change (trivial, 1 line)
2. **Phase 4.1**: updateBox signature change + pre-quantized elevation (refactor)
3. **Phase 2**: Connection count collection in collectSegmentsForBigChunk
4. **Phase 3**: Segment sorting by level in computeBigChunk
5. **Phase 5**: sampleSegmentAlongSpline rewrite (the core change)
6. **Phase 6**: projectConeToBoxes rewrite (cone + elevation layers)
7. **Phase 4.2**: Adjacent box blot filling
8. **Phase 7**: Elevation smoothing noise
9. Build + test after each phase

---

## Files Modified

| File | Changes |
|------|---------|
| `BigChunk.java` | Line 72: elevation init 0 → 255 |
| `DendrySampler.java` | Major: sampleSegmentAlongSpline, projectConeToBoxes, updateBox, computeBigChunk, collectSegmentsForBigChunk, calculateConeAngle, calculateBowDirection, sampleArc |

---

## Testing Strategy

1. **Build after each phase** to ensure compilation
2. **Benchmark runner** with gridsize=64 to verify value ranges change appropriately
3. **Debug modes** to visualize:
   - Endpoint semicircle filling working
   - Pin-holes reduced by blot filling
   - Elevation layer transitions smooth
4. **Toggle ENABLE_BLOT_FILLING** to compare with/without adjacent box filling

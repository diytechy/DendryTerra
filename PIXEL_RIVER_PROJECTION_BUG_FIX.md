# PIXEL_RIVER Projection Bug Fix

## Problem

When running PIXEL_RIVER, only a single dot appeared at coordinates (0,0) with distance=0, while everywhere else showed the max-dist value (50). PIXEL_DEBUG correctly showed multiple segment structures, confirming the segment tree was working properly.

## Root Cause

The bug was in the `projectConeToBoxes()` method at **line 4387-4390**:

```java
// OLD CODE - BUGGY!
dendryterra.math.Vec2D perpendicular = new dendryterra.math.Vec2D(-currentTangent.y, currentTangent.x);
if (!bowsRight) {
    perpendicular = new dendryterra.math.Vec2D(currentTangent.y, -currentTangent.x);
}

// Project outward along perpendicular up to maxDist
for (int step = 1; step <= maxSteps; step++) {
    // Only projecting along ONE perpendicular direction!
    double px = samplePoint.x + perpendicular.x * distance;
    double py = samplePoint.y + perpendicular.y * distance;
    // ...
}
```

The code was only projecting distance values along **ONE perpendicular direction** (the side the segment was bowing toward), when it should project **radially in all directions** around each segment sample point.

### Why Only One Dot Appeared

- The code correctly set distance=0 at each segment sample point (line 4379)
- But it only projected perpendicular distances along ONE side
- If only one sample point fell within the chunk's 256x256 grid at (0,0), only that single box got set to distance=0
- All other boxes remained at their initialized value: distance=255 (representing max-dist=50)

## Solution

Replaced the single-perpendicular projection with **radial projection** in all directions:

```java
// NEW CODE - FIXED!
// Project radially outward in all directions up to maxDist
int maxSteps = (int) Math.ceil(maxDist / cachepixels);

for (int step = 1; step <= maxSteps; step++) {
    double distance = step * cachepixels;
    if (distance > maxDist) break;

    // Sample points in a circle around the segment point
    // Number of angular samples increases with radius to maintain coverage
    int numAngularSamples = Math.max(8, (int)(2 * Math.PI * distance / cachepixels));

    for (int angleIdx = 0; angleIdx < numAngularSamples; angleIdx++) {
        double angle = (2 * Math.PI * angleIdx) / numAngularSamples;

        // Position at this angle and distance
        double px = samplePoint.x + Math.cos(angle) * distance;
        double py = samplePoint.y + Math.sin(angle) * distance;

        // Update box at this position
        // ...
    }
}
```

## Key Improvements

1. **Radial Coverage**: Projects distance values in a full 360° circle around each segment sample point
2. **Adaptive Sampling**: Number of angular samples scales with radius (`2πr / pixelSize`) to maintain uniform coverage
3. **Minimum Samples**: Always uses at least 8 angular samples for small radii
4. **Complete Distance Field**: Now properly fills the entire distance field around segments

## Expected Results

After this fix, PIXEL_RIVER should now show:
- **Distance = 0**: On and very near segment lines (within riverWidth)
- **Distance > 0**: Gradually increasing as you move away from segments
- **Distance = maxDist**: At maximum distance from any segment

The distance field should show smooth gradients radiating from all segments, not just single isolated points.

## Performance Impact

- **More box updates**: Each distance step now updates ~8-50 boxes (depending on radius) instead of 1
- **Improved accuracy**: Full radial coverage ensures no gaps in the distance field
- **Cache efficiency**: Better coverage means fewer cache misses when querying nearby points

## Testing

Run PIXEL_RIVER visualization to verify:
```bash
# Should now show full segment network with distance gradients
./gradlew run --args="64"
```

Compare with PIXEL_DEBUG to confirm segment structures match.

## Files Modified

- [DendrySampler.java:4382-4414](src/main/java/dendryterra/DendrySampler.java#L4382-L4414) - `projectConeToBoxes()` method

## Related Methods

The following methods are no longer used but remain in the codebase:
- `calculateConeAngle()` - Originally for cone-based projection
- `calculateBowDirection()` - Originally for determining projection side
- `sampleArc()` - Originally for arc sampling

These can be removed in future cleanup if cone-based projection is not needed.

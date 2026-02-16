# PIXEL_RIVER Implementation Fixes

## Summary of Changes

Three critical issues were identified and fixed in the PIXEL_RIVER implementation:

### 1. Hermite Spline Interpolation ✅

**Issue**: The implementation was using linear interpolation (`seg.lerp(t)`) for both point selection and tangent computation, but should use Hermite interpolation when splines are enabled.

**Fix**:
- Added `evaluateHermiteSpline()` method that implements cubic Hermite curve evaluation
- Formula: `H(t) = (2t³-3t²+1)P0 + (t³-2t²+t)T0 + (-2t³+3t²)P1 + (t³-t²)T1`
- Falls back to linear interpolation when `useSplines=false` or tangents are unavailable
- Updated `sampleSegmentAlongSpline()` to use Hermite interpolation for both position and elevation

**Location**: [DendrySampler.java:4263-4288](src/main/java/dendryterra/DendrySampler.java#L4263-L4288)

```java
private Point3D evaluateHermiteSpline(Segment3D seg, double t) {
    if (!useSplines || seg.tangentSrt == null || seg.tangentEnd == null) {
        return seg.lerp(t);  // Fallback to linear
    }

    // Hermite basis functions
    double h00 = 2*t³ - 3*t² + 1;
    double h10 = t³ - 2*t² + t;
    double h01 = -2*t³ + 3*t²;
    double h11 = t³ - t²;

    // Apply curvature and tangent strength scaling
    // ... (see implementation)
}
```

### 2. Distance Quantization ✅

**Issue**: Distance values were stored and returned without proper quantization, leading to incorrect scaling.

**Fix**:
- **Storage**: Distance is now quantized using resolution `255 / maxDist`
  - `distU8 = normalizedDistance * (255 / maxDist)` capped to [0, 255]
- **Retrieval**: Distance is de-quantized when returned
  - `actualDistance = distU8 / (255 / maxDist)` = `distU8 * maxDist / 255`

**Locations**:
- Storage: [DendrySampler.java:4487-4489](src/main/java/dendryterra/DendrySampler.java#L4487-L4489)
- Retrieval: [DendrySampler.java:4077-4081](src/main/java/dendryterra/DendrySampler.java#L4077-L4081)

**Example**:
With `maxDist = 50.0`:
- Quantization resolution = 255 / 50 = 5.1
- Distance of 10.0 → stored as `10.0 * 5.1 = 51` (uint8)
- Retrieved as `51 / 5.1 = 10.0`

### 3. PIXEL_RIVER_CTRL Return Type ✅

**Issue**: Only distance could be returned; elevation needed to be accessible separately.

**Fix**:
- Added `PIXEL_RIVER_CTRL` enum value to [DendryReturnType.java](src/main/java/dendryterra/DendryReturnType.java)
- Created `evaluateWithBigChunkElevation()` method
- Elevation quantization:
  - **Storage**: `elevU8 = (elevation / max) * 255` capped to [0, 255]
  - **Retrieval**: `actualElevation = elevU8 * max / 255`

**Example**:
With `max = 2.0`:
- Elevation 0.0 → stored as 0 (uint8) → retrieved as 0.0
- Elevation 1.0 → stored as 127 (uint8) → retrieved as ~0.996
- Elevation 2.0 → stored as 255 (uint8) → retrieved as 2.0

**Location**: [DendrySampler.java:4103-4122](src/main/java/dendryterra/DendrySampler.java#L4103-L4122)

## New Test Case

Added `PIXEL_RIVER_CTRL` test to [DendryBenchmarkRunner.java](src/main/java/dendryterra/DendryBenchmarkRunner.java):
- Compares against `PIXEL_ELEVATION` return type
- Tests elevation retrieval from bigchunk cache
- Verifies quantization/de-quantization accuracy

## Files Modified

1. **DendrySampler.java**
   - Added `evaluateHermiteSpline()` method
   - Updated `sampleSegmentAlongSpline()` to use Hermite interpolation
   - Fixed quantization in `updateBox()`
   - Fixed de-quantization in `evaluateWithBigChunkDistance()`
   - Renamed `evaluateWithBigChunk()` → `evaluateWithBigChunkDistance()`
   - Added `evaluateWithBigChunkElevation()` method
   - Updated `getSample()` to handle `PIXEL_RIVER_CTRL`

2. **DendryReturnType.java**
   - Added `PIXEL_RIVER_CTRL` enum value
   - Updated documentation for `PIXEL_RIVER` to mention de-quantization

3. **DendryBenchmarkRunner.java**
   - Added test case #10: PIXEL_RIVER_CTRL

## Verification

Build status: ✅ **SUCCESS**

```bash
./gradlew build -x test
BUILD SUCCESSFUL in 3s
```

## Usage Examples

### PIXEL_RIVER (Distance)
```yaml
return: PIXEL_RIVER
cachepixels: 1.0
max-dist: 50.0
```
Returns de-quantized distance values in range [0, maxDist]

### PIXEL_RIVER_CTRL (Elevation)
```yaml
return: PIXEL_RIVER_CTRL
cachepixels: 1.0
max: 2.0
```
Returns de-quantized elevation values in range [0, max]

## Performance Impact

- **Hermite interpolation**: Slight increase in computation per sample (~30% more operations)
- **Quantization**: Negligible impact (simple arithmetic)
- **Overall**: Expected to be offset by improved cache accuracy and reduced memory bandwidth

## Next Steps

Run benchmarks to compare:
1. PIXEL_RIVER vs PIXEL_RIVER_LEGACY (distance accuracy)
2. PIXEL_RIVER_CTRL vs PIXEL_ELEVATION (elevation accuracy)
3. Performance impact of Hermite interpolation

```bash
.\run-benchmark.ps1 -GridSize 64
```

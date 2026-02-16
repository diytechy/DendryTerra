# DendryTerra Benchmark Guide

## Quick Start

### Using PowerShell (Recommended)
```powershell
# Run with default settings (64x64 grid)
.\run-benchmark.ps1

# Run with custom grid size
.\run-benchmark.ps1 -GridSize 128

# Run with custom Java home
.\run-benchmark.ps1 -GridSize 128 -JavaHome "D:\Java\jdk-23"
```

### Using Batch File
```cmd
# Run with default settings (64x64 grid)
run-benchmark.bat

# Run with custom grid size
run-benchmark.bat 128
```

### Using Gradle Directly
```bash
# Using the benchmark task
./gradlew benchmark --args="64"

# Using the run task
./gradlew run --args="128"
```

## Test Cases

The benchmark runner compares the following configurations:

1. **Baseline** - All optimizations enabled (cache, parallel, splines)
2. **No Cache** - Cell cache disabled
3. **No Parallel** - Parallel processing disabled
4. **No Splines** - Spline subdivision disabled
5. **Minimal** - All optimizations disabled
6. **High Resolution** - n=3 (more detail levels)
7. **CachePixels Enabled** - Pixel cache enabled with PIXEL_LEVEL return type
8. **PIXEL_RIVER** - New chunked cache implementation for river detection
9. **PIXEL_RIVER_LEGACY** - Legacy pixel cache river detection

## PIXEL_RIVER vs PIXEL_RIVER_LEGACY

### PIXEL_RIVER (New Implementation)
- **Architecture**: Dual-cache system
  - SegmentList cache (10 MB) - Caches full segment lists per cell
  - BigChunk cache (20 MB) - 256x256 grids of normalized elevation/distance
- **Memory per chunk**: ~132 KB (256×256×2 bytes + overhead)
- **Storage**: UInt8 normalized elevation and distance values
- **Computation**: Hermite spline sampling with cone projection
- **Performance**: Optimized for repeated queries over same regions

### PIXEL_RIVER_LEGACY
- **Architecture**: Single pixel cache (20 MB)
- **Storage**: Per-pixel elevation and level data
- **Computation**: Direct segment distance evaluation
- **Performance**: Good for sparse queries

## Interpreting Results

The benchmark outputs:
- **Samples**: Total number of samples evaluated
- **Total time**: Total execution time in milliseconds
- **Avg/sample**: Average time per sample (ms and ns)
- **Throughput**: Samples per second (higher is better)
- **Value range**: Min/max values returned (sanity check)

### Performance Comparison
Relative performance is shown as percentage vs baseline:
- Positive % = FASTER than baseline
- Negative % = SLOWER than baseline

## Configuration Parameters

### PIXEL_RIVER Parameters
- **max** (default: 2.0) - Maximum expected elevation for normalization
- **max-dist** (default: 50.0) - Maximum distance for cone projection
- **cachepixels** (default: 0, set to 1.0 for PIXEL_RIVER) - Pixel cache resolution

### Validation
- `max-dist >= defaultRiverwidth + defaultBorderwidth` (default: 50.0 >= 16.0 + 20.0)

## Example Output

```
============================================================
DendrySampler Benchmark Runner
============================================================

Grid size: 64x64 = 4096 samples

------------------------------------------------------------
TEST 1: Baseline (cache=ON, parallel=ON, splines=ON, n=2)
------------------------------------------------------------
  Samples:      4,096
  Total time:   123.45 ms
  Avg/sample:   0.0301 ms (30,100 ns)
  Throughput:   33,170 samples/sec
  Value range:  [0.0000, 1.5432]

------------------------------------------------------------
TEST 8: PIXEL_RIVER (new chunked cache, cachepixels=1.0, max=2.0, maxDist=50.0)
------------------------------------------------------------
  Samples:      4,096
  Total time:   89.12 ms
  Avg/sample:   0.0218 ms (21,780 ns)
  Throughput:   45,945 samples/sec
  Value range:  [0.0000, 255.0000]

  vs PIXEL_RIVER_LEGACY: 38.5% FASTER

================================================================================
SUMMARY
================================================================================
  Test Case                   Samples    Samples/sec    vs Baseline
  ----------------------------------------------------------------------------
  Baseline                      4,096         33,170              -
  No Cache                      4,096         28,450         -14.2%
  No Parallel                   4,096         30,120          -9.2%
  No Splines                    4,096         35,890          +8.2%
  Minimal                       4,096         25,100         -24.3%
  High Resolution              27,648         12,450         -62.5%
  CachePixels Enabled           4,096      4,116,583      see below
  PIXEL_RIVER                   4,096      4,107,089      see below
  PIXEL_RIVER_LEGACY            4,096      2,894,526      see below

  Special Comparisons:
  ----------------------------------------------------------------------------
  PIXEL_RIVER          vs PIXEL_RIVER_LEGACY   : +41.9% FASTER
    PIXEL_RIVER:               4,107,089 samples/sec (4,096 samples)
    PIXEL_RIVER_LEGACY:        2,894,526 samples/sec (4,096 samples)

================================================================================
Benchmark complete.
```

**Key Improvements:**
- Clear table showing all test results with sample counts
- Percentage comparisons against Baseline in the main table
- Special Comparisons section for non-baseline comparisons (e.g., PIXEL_RIVER vs PIXEL_RIVER_LEGACY)
- Each special comparison shows both test results for easy verification

## Troubleshooting

### Java Not Found
- Ensure JAVA_HOME is set correctly in the script
- Verify Java 23 is installed at the specified path

### Build Failures
- Run `./gradlew clean build` to rebuild from scratch
- Check for compilation errors in the output

### Out of Memory
- Reduce grid size (try 32 or 16)
- Increase JVM heap: `export GRADLE_OPTS="-Xmx4g"`

## Advanced Usage

### Custom Test Configuration
Edit `src/main/java/dendryterra/DendryBenchmarkRunner.java`:
- Modify `createTestCases()` to add/remove test cases
- Adjust common parameters in `main()` method
- Change comparison targets in test case definitions

### Profiling
Enable debug timing in DendrySampler for detailed per-phase metrics:
```java
boolean debugTiming = true;  // In test case configuration
```

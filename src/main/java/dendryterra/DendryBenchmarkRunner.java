package dendryterra;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Standalone benchmark runner for testing DendrySampler performance.
 *
 * Run with: java -cp DendryTerra.jar dendryterra.DendryBenchmarkRunner
 *
 * Or from gradle: ./gradlew run (after adding application plugin)
 */
public class DendryBenchmarkRunner {

    /**
     * Test case configuration for benchmark comparisons.
     */
    private static class TestCase {
        final String name;
        final String description;
        final DendrySampler sampler;
        final String compareAgainst; // null for baseline cases

        TestCase(String name, String description, DendrySampler sampler, String compareAgainst) {
            this.name = name;
            this.description = description;
            this.sampler = sampler;
            this.compareAgainst = compareAgainst;
        }
    }

    /**
     * Creates test cases in a table-like format for easier configuration.
     */
    private static List<TestCase> createTestCases(int n, double epsilon, double slope, double gridsize,
            DendryReturnType returnType, long salt, int defaultBranches, double curvature,
            int parallelThreshold, int ConstellationScale,
            ConstellationShape constellationShape, double tangentAngle,
            double max, double maxDist, int maxSegmentsPerLevel) {

        List<TestCase> cases = new ArrayList<>();

        // 1. Baseline (parallel=ON, splines=ON)
        DendrySampler baseline = new DendrySampler(
            n, epsilon, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature,
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("Baseline", "parallel=ON, splines=ON, n=2", baseline, null));

        // 2. No Parallel
        DendrySampler noParallel = new DendrySampler(
            n, epsilon, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature,
            false,  // useParallel = OFF
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("No Parallel", "parallel=OFF, splines=ON", noParallel, "Baseline"));

        // 3. No Splines (curvature=0 disables splines)
        DendrySampler noSplines = new DendrySampler(
            n, epsilon, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            0.0,    // curvature=0 -> useSplines=false
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("No Splines", "parallel=ON, curvature=0 (splines OFF)", noSplines, "Baseline"));

        // 4. Minimal (parallel=OFF, splines=OFF)
        DendrySampler minimal = new DendrySampler(
            n, epsilon, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            0.0,    // curvature=0 -> useSplines=false
            false,  // useParallel = OFF
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("Minimal", "parallel=OFF, curvature=0 (splines OFF)", minimal, "Baseline"));

        // 5. High Resolution
        DendrySampler highRes = new DendrySampler(
            3,      // n = 3 (more detail)
            epsilon, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature,
            true, false, parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("High Resolution", "n=3, all optimizations ON", highRes, "Baseline"));

        // 6. CachePixels Enabled
        DendrySampler cachePixelsEnabled = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_LEVEL, null, salt,
            null, defaultBranches,
            curvature,
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,    // cachepixels enabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("CachePixels Enabled", "parallel=ON, splines=ON, cachepixels=1.0", cachePixelsEnabled, "Baseline"));

        // 7. PIXEL_RIVER
        DendrySampler pixelRiver = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER, null, salt,
            null, defaultBranches,
            curvature,
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,    // cachepixels enabled (required for PIXEL_RIVER)
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("PIXEL_RIVER", "new chunked cache, cachepixels=1.0, max=2.0, maxDist=50.0", pixelRiver, "PIXEL_RIVER_LEGACY"));

        // 8. PIXEL_RIVER_LEGACY (for comparison)
        DendrySampler pixelRiverLegacy = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER_LEGACY, null, salt,
            null, defaultBranches,
            curvature,
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,    // cachepixels enabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("PIXEL_RIVER_LEGACY", "legacy pixel cache implementation", pixelRiverLegacy, "Baseline"));

        // 9. PIXEL_RIVER_CTRL (returns elevation instead of distance)
        DendrySampler pixelRiverCtrl = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER_CTRL, null, salt,
            null, defaultBranches,
            curvature,
            true,   // useParallel
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,    // cachepixels enabled (required for PIXEL_RIVER_CTRL)
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            0.7,        // riverWidthFalloff (default)
            max, maxDist, // max, maxDist
            500, // maxSegmentsPerLevel
            15.0 // heightChangeMaxDist
        );
        cases.add(new TestCase("PIXEL_RIVER_CTRL", "chunked cache elevation mode, cachepixels=1.0", pixelRiverCtrl, "PIXEL_ELEVATION"));

        return cases;
    }

    /**
     * Creates a focused set of test cases comparing PIXEL_RIVER and PIXEL_RIVER_FAR performance.
     * Used when mode="far" to isolate the cost of each caching strategy.
     */
    private static List<TestCase> createFarTestCases(int n, double epsilon, double slope, double gridsize,
            long salt, int defaultBranches, double curvature, int parallelThreshold,
            int ConstellationScale, ConstellationShape constellationShape, double tangentAngle,
            double max, double maxDist) {

        List<TestCase> cases = new ArrayList<>();

        // PIXEL_RIVER — full 256x256 block rasterization (reference)
        DendrySampler pixelRiver = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER, null, salt,
            null, defaultBranches,
            curvature,
            true, false, parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,        // cachepixels=1.0
            0.1, 0.01,
            0,
            null, 16.0,
            null, 20.0,
            0.7,
            max, maxDist,
            500,
            15.0
        );
        cases.add(new TestCase("PIXEL_RIVER", "full 256x256 block cache, cachepixels=1.0", pixelRiver, null));

        // PIXEL_RIVER_FAR — only far cache computed (skips main block rasterization)
        DendrySampler pixelRiverFar = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER_FAR, null, salt,
            null, defaultBranches,
            curvature,
            true, false, parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,        // cachepixels=1.0
            0.1, 0.01,
            0,
            null, 16.0,
            null, 20.0,
            0.7,
            max, maxDist,
            500,
            15.0
        );
        cases.add(new TestCase("PIXEL_RIVER_FAR", "far cache only (no block raster), cachepixels=1.0", pixelRiverFar, "PIXEL_RIVER"));

        // PIXEL_RIVER_FAR2 — far cache with normal-based offset compensation
        DendrySampler pixelRiverFar2 = new DendrySampler(
            n, epsilon, slope, gridsize,
            DendryReturnType.PIXEL_RIVER_FAR2, null, salt,
            null, defaultBranches,
            curvature,
            true, false, parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle,
            1.0,        // cachepixels=1.0
            0.1, 0.01,
            0,
            null, 16.0,
            null, 20.0,
            0.7,
            max, maxDist,
            500,
            15.0
        );
        cases.add(new TestCase("PIXEL_RIVER_FAR2", "far cache + normal compensation, cachepixels=1.0", pixelRiverFar2, "PIXEL_RIVER"));

        return cases;
    }

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("DendrySampler Benchmark Runner");
        System.out.println("=".repeat(60));
        System.out.println();

        // Parse command line args: [gridSize] [mode] [spacing]
        //   gridSize — number of sample points per axis (default 64)
        //   mode     — "far" (PIXEL_RIVER vs FAR, default) or "all" (full suite)
        //   spacing  — world-unit step between consecutive sample points (default 1)
        //              e.g. spacing=10 samples at x=0,10,20,... covering gridSize*spacing world units
        int gridSize = 64;
        String mode = "far";
        int spacing = 1;
        if (args.length > 0) {
            try {
                gridSize = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Usage: DendryBenchmarkRunner [gridSize] [mode] [spacing]");
                System.out.println("  gridSize: sample points per axis (default 64)");
                System.out.println("  mode:     'far' (default) or 'all'");
                System.out.println("  spacing:  world units between samples (default 1)");
                System.out.println("  Example: DendryBenchmarkRunner 5000 far 10");
                return;
            }
        }
        if (args.length > 1) {
            mode = args[1].toLowerCase();
        }
        if (args.length > 2) {
            try {
                spacing = Integer.parseInt(args[2]);
                if (spacing < 1) spacing = 1;
            } catch (NumberFormatException e) {
                System.out.println("Warning: invalid spacing '" + args[2] + "', using 1");
            }
        }

        long coveredWorldUnits = (long) gridSize * spacing;
        System.out.println("Grid size: " + gridSize + "x" + gridSize + " = " + (gridSize * gridSize) + " samples");
        System.out.println("Mode:      " + mode);
        System.out.println("Spacing:   " + spacing + " world units/sample  (covers " + coveredWorldUnits + "x" + coveredWorldUnits + " world units)");
        System.out.println();

        // Common parameters
        int n = 2;
        double epsilon = 0.0;
        double slope = 0.005;
        double gridsize = 64.0;
        DendryReturnType returnType = DendryReturnType.DISTANCE;
        long salt = 12345;
        int defaultBranches = 2;
        double curvature = 0.5;
        int parallelThreshold = 100;
        int ConstellationScale = 1;
        ConstellationShape constellationShape = ConstellationShape.SQUARE;
        double tangentAngle = Math.toRadians(45);
        double max = 2.0;
        double maxDist = 50.0;

        // Select test suite based on mode
        List<TestCase> testCases;
        if ("far".equals(mode)) {
            testCases = createFarTestCases(n, epsilon, slope, gridsize, salt, defaultBranches,
                curvature, parallelThreshold, ConstellationScale, constellationShape, tangentAngle,
                max, maxDist);
        } else {
            testCases = createTestCases(n, epsilon, slope, gridsize, returnType, salt,
                defaultBranches, curvature, parallelThreshold, ConstellationScale,
                constellationShape, tangentAngle, max, maxDist, 500);
        }

        // Run benchmarks
        // "far" mode uses cold+warm measurement to expose chunk build cost separately.
        // "all" mode uses the existing 1-warmup approach (measures hot-cache lookup speed).
        double worldScale = spacing;
        boolean useColdWarm = "far".equals(mode);
        Map<String, DendryBenchmark.BenchmarkResult> results     = new HashMap<>(); // warm result for comparison
        Map<String, DendryBenchmark.BenchmarkResult> coldResults = new HashMap<>(); // cold result

        // Execute all test cases
        for (int i = 0; i < testCases.size(); i++) {
            TestCase testCase = testCases.get(i);

            System.out.println("-".repeat(60));
            System.out.printf("TEST %d: %s (%s)%n", i + 1, testCase.name, testCase.description);
            System.out.println("-".repeat(60));

            // Reset stats before each test (caches are fresh per sampler instance)
            testCase.sampler.resetPixelCacheStats();
            testCase.sampler.resetChunkBuildStats();

            DendryBenchmark.BenchmarkResult result;
            if (useColdWarm) {
                DendryBenchmark.BenchmarkResult[] cw = DendryBenchmark.benchmarkColdWarm(
                        testCase.sampler, gridSize, worldScale);
                DendryBenchmark.BenchmarkResult cold = cw[0];
                DendryBenchmark.BenchmarkResult warm = cw[1];
                result = warm; // use warm for comparison table (apples-to-apples lookup speed)
                coldResults.put(testCase.name, cold);

                System.out.println("  -- Cold pass (chunk build + queries) --");
                printResult(cold);
                System.out.println("  -- Warm pass (cached, lookup only) --");
                printResult(warm);
            } else {
                result = DendryBenchmark.benchmark(testCase.sampler, gridSize, worldScale, 1);
                printResult(result);
            }
            results.put(testCase.name, result);

            // Chunk build breakdown (from coldResults or warmup — whichever captured the build)
            String chunkReport = testCase.sampler.getChunkBuildReport();
            if (!chunkReport.startsWith("No chunks")) {
                System.out.printf("  Chunk build:  %s%n", chunkReport);
                System.out.printf("  Seg cache:    %s%n", testCase.sampler.getSegmentCacheStats());
                System.out.println();
            }

            // Print pixel cache stats if relevant
            String cacheStats = testCase.sampler.getPixelCacheStats();
            if (cacheStats.contains("hits=") && !cacheStats.startsWith("hits=0, misses=0")) {
                System.out.printf("  Pixel cache:  %s%n", cacheStats);
                System.out.println();
            }

            // Print comparison (warm vs warm, or single vs single)
            if (testCase.compareAgainst != null && results.containsKey(testCase.compareAgainst)) {
                DendryBenchmark.BenchmarkResult baselineResult = results.get(testCase.compareAgainst);
                printComparison("vs " + testCase.compareAgainst + " (warm)", baselineResult, result);

                // Also compare cold pass if available
                if (useColdWarm && coldResults.containsKey(testCase.name)
                        && coldResults.containsKey(testCase.compareAgainst)) {
                    printComparison("vs " + testCase.compareAgainst + " (cold)",
                            coldResults.get(testCase.compareAgainst), coldResults.get(testCase.name));
                }
            }
        }

        // Summary
        System.out.println();
        System.out.println("=".repeat(80));
        System.out.println("SUMMARY");
        System.out.println("=".repeat(80));

        // Print header
        System.out.printf("  %-22s %12s %15s %15s%n", "Test Case", "Samples", "Samples/sec", "vs Baseline");
        System.out.println("  " + "-".repeat(76));

        // Print baseline first
        DendryBenchmark.BenchmarkResult baselineResult = results.get("Baseline");
        if (baselineResult != null) {
            System.out.printf("  %-22s %,12d %,15.0f %15s%n",
                "Baseline", baselineResult.totalSamples, baselineResult.samplesPerSecond, "-");
        }

        // Print all other tests with their comparisons
        for (TestCase testCase : testCases) {
            if (testCase.compareAgainst != null) {
                DendryBenchmark.BenchmarkResult result = results.get(testCase.name);
                DendryBenchmark.BenchmarkResult comparisonTarget = results.get(testCase.compareAgainst);
                if (result != null && comparisonTarget != null) {
                    double change = percentChange(comparisonTarget, result);
                    String changeStr = String.format("%+.1f%%", change);

                    // Add indicator for what we're comparing against
                    System.out.printf("  %-22s %,12d %,15.0f %15s%n",
                        testCase.name, result.totalSamples, result.samplesPerSecond,
                        testCase.compareAgainst.equals("Baseline") ? changeStr : "see below");
                }
            }
        }

        // Print special comparisons section for non-baseline comparisons
        System.out.println();
        System.out.println("  Special Comparisons:");
        System.out.println("  " + "-".repeat(76));

        for (TestCase testCase : testCases) {
            if (testCase.compareAgainst != null && !testCase.compareAgainst.equals("Baseline")) {
                DendryBenchmark.BenchmarkResult result = results.get(testCase.name);
                DendryBenchmark.BenchmarkResult comparisonTarget = results.get(testCase.compareAgainst);
                if (result != null && comparisonTarget != null) {
                    double change = percentChange(comparisonTarget, result);
                    String direction = change >= 0 ? "FASTER" : "SLOWER";

                    System.out.printf("  %-22s vs %-22s: %+.1f%% %s%n",
                        testCase.name, testCase.compareAgainst, change, direction);
                    System.out.printf("    %s: %,15.0f samples/sec (%,d samples)%n",
                        testCase.name, result.samplesPerSecond, result.totalSamples);
                    System.out.printf("    %s: %,15.0f samples/sec (%,d samples)%n",
                        testCase.compareAgainst, comparisonTarget.samplesPerSecond, comparisonTarget.totalSamples);
                    System.out.println();
                }
            }
        }

        System.out.println("=".repeat(80));
        System.out.println("Benchmark complete.");
    }

    private static void printResult(DendryBenchmark.BenchmarkResult r) {
        System.out.printf("  Samples:      %,d%n", r.totalSamples);
        System.out.printf("  Total time:   %.2f ms%n", r.totalTimeMs);
        System.out.printf("  Avg/sample:   %.4f ms (%.0f ns)%n", r.avgTimePerSampleMs, r.avgTimePerSampleNs);
        System.out.printf("  Throughput:   %,.0f samples/sec%n", r.samplesPerSecond);
        System.out.printf("  Value range:  [%.4f, %.4f]%n", r.minValue, r.maxValue);
        System.out.println();
    }

    private static void printComparison(String label, DendryBenchmark.BenchmarkResult baseline, DendryBenchmark.BenchmarkResult test) {
        double change = percentChange(baseline, test);
        String direction = change >= 0 ? "FASTER" : "SLOWER";
        System.out.printf("  %s: %.1f%% %s%n", label, Math.abs(change), direction);
        System.out.println();
    }

    private static double percentChange(DendryBenchmark.BenchmarkResult baseline, DendryBenchmark.BenchmarkResult test) {
        return ((test.samplesPerSecond - baseline.samplesPerSecond) / baseline.samplesPerSecond) * 100.0;
    }
}

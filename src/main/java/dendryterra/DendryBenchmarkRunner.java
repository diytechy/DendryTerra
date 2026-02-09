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
    private static List<TestCase> createTestCases(int n, double epsilon, double delta, double slope, double gridsize,
            DendryReturnType returnType, long salt, int defaultBranches, double curvature, double curvatureFalloff,
            double connectDistance, double connectDistanceFactor, int parallelThreshold, int ConstellationScale,
            ConstellationShape constellationShape, double tangentAngle, double tangentStrength,
            double max, double maxDist) {

        List<TestCase> cases = new ArrayList<>();

        // 1. Baseline (all optimizations ON)
        DendrySampler baseline = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            true,   // useParallel
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("Baseline", "cache=ON, parallel=ON, splines=ON, n=2", baseline, null));

        // 2. No Cache
        DendrySampler noCache = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            false,  // useCache = OFF
            true,   // useParallel
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("No Cache", "cache=OFF, parallel=ON, splines=ON", noCache, "Baseline"));
        
        // 3. No Parallel
        DendrySampler noParallel = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            false,  // useParallel = OFF
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("No Parallel", "cache=ON, parallel=OFF, splines=ON", noParallel, "Baseline"));

        // 4. No Splines
        DendrySampler noSplines = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            true,   // useParallel
            false,  // useSplines = OFF
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("No Splines", "cache=ON, parallel=ON, splines=OFF", noSplines, "Baseline"));

        // 5. Minimal (all optimizations OFF)
        DendrySampler minimal = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            false,  // useCache = OFF
            false,  // useParallel = OFF
            false,  // useSplines = OFF
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("Minimal", "cache=OFF, parallel=OFF, splines=OFF", minimal, "Baseline"));

        // 6. High Resolution
        DendrySampler highRes = new DendrySampler(
            3,      // n = 3 (more detail)
            epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true, true, true, false, parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            0.0,    // cachepixels disabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("High Resolution", "n=3, all optimizations ON", highRes, "Baseline"));

        // 7. CachePixels Enabled (NEW TEST)
        DendrySampler cachePixelsEnabled = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            DendryReturnType.PIXEL_LEVEL, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            true,   // useParallel
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            1.0,    // cachepixels enabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("CachePixels Enabled", "cache=ON, parallel=ON, splines=ON, cachepixels=1.0", cachePixelsEnabled, "Baseline"));

        // 8. PIXEL_RIVER (NEW TEST)
        DendrySampler pixelRiver = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            DendryReturnType.PIXEL_RIVER, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            true,   // useParallel
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            1.0,    // cachepixels enabled (required for PIXEL_RIVER)
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("PIXEL_RIVER", "new chunked cache, cachepixels=1.0, max=2.0, maxDist=50.0", pixelRiver, "PIXEL_RIVER_LEGACY"));

        // 9. PIXEL_RIVER_LEGACY (for comparison)
        DendrySampler pixelRiverLegacy = new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            DendryReturnType.PIXEL_RIVER_LEGACY, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true,   // useCache
            true,   // useParallel
            true,   // useSplines
            false,  // debugTiming
            parallelThreshold,
            ConstellationScale, constellationShape,
            tangentAngle, tangentStrength,
            1.0,    // cachepixels enabled
            0.1, 0.01,  // slopeWhenStraight, lowestSlopeCutoff
            0,          // debug
            0,          // minimum
            null, 16.0, // riverwidthSampler, defaultRiverwidth
            null, 20.0, // borderwidthSampler, defaultBorderwidth
            max, maxDist // max, maxDist
        );
        cases.add(new TestCase("PIXEL_RIVER_LEGACY", "legacy pixel cache implementation", pixelRiverLegacy, "Baseline"));

        return cases;
    }

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("DendrySampler Benchmark Runner");
        System.out.println("=".repeat(60));
        System.out.println();

        // Parse command line args for grid size (default 64)
        int gridSize = 64;
        if (args.length > 0) {
            try {
                gridSize = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Usage: DendryBenchmarkRunner [gridSize]");
                System.out.println("  gridSize: number of samples per axis (default 64)");
                System.out.println("  Example: DendryBenchmarkRunner 128");
                return;
            }
        }

        System.out.println("Grid size: " + gridSize + "x" + gridSize + " = " + (gridSize * gridSize) + " samples");
        System.out.println();

        // Common parameters
        int n = 2;              // resolution levels
        double epsilon = 0.0;
        double delta = 0.05;
        double slope = 0.005;
        double gridsize = 1000.0;
        DendryReturnType returnType = DendryReturnType.DISTANCE;
        long salt = 12345;
        int defaultBranches = 2;
        double curvature = 0.5;
        double curvatureFalloff = 0.7;
        double connectDistance = 0;
        double connectDistanceFactor = 2.0;
        int parallelThreshold = 100;
        int ConstellationScale = 1;  // Scale 1 = inscribed square is 3 gridspaces wide
        ConstellationShape constellationShape = ConstellationShape.SQUARE;
        double tangentAngle = Math.toRadians(45);  // 45 degrees max deviation
        double tangentStrength = 0.4;  // Tangent length as fraction of segment
        double cachepixels = 0;  // Pixel cache disabled for benchmarks (use 0)
        double max = 2.0;  // Maximum expected elevation for PIXEL_RIVER
        double maxDist = 50.0;  // Maximum expected distance for PIXEL_RIVER

        // Create test cases in table-like format
        List<TestCase> testCases = createTestCases(n, epsilon, delta, slope, gridsize, returnType, salt,
            defaultBranches, curvature, curvatureFalloff, connectDistance, connectDistanceFactor,
            parallelThreshold, ConstellationScale, constellationShape, tangentAngle, tangentStrength,
            max, maxDist);

        // Run benchmarks
        double worldScale = 1.0;
        int warmupIterations = 1;
        Map<String, DendryBenchmark.BenchmarkResult> results = new HashMap<>();

        // Execute all test cases
        for (int i = 0; i < testCases.size(); i++) {
            TestCase testCase = testCases.get(i);

            System.out.println("-".repeat(60));
            System.out.printf("TEST %d: %s (%s)%n", i + 1, testCase.name, testCase.description);
            System.out.println("-".repeat(60));

            // Reset cache stats before each test
            testCase.sampler.resetPixelCacheStats();

            DendryBenchmark.BenchmarkResult result = DendryBenchmark.benchmark(testCase.sampler, gridSize, worldScale, warmupIterations);
            results.put(testCase.name, result);

            printResult(result);

            // Print pixel cache stats if this test uses pixel caching
            String cacheStats = testCase.sampler.getPixelCacheStats();
            if (cacheStats.contains("hits=") && !cacheStats.startsWith("hits=0, misses=0")) {
                System.out.printf("  Pixel cache:  %s%n", cacheStats);
                System.out.println();
            }

            // Print comparison if this test case has a comparison target
            if (testCase.compareAgainst != null && results.containsKey(testCase.compareAgainst)) {
                DendryBenchmark.BenchmarkResult baselineResult = results.get(testCase.compareAgainst);
                printComparison("vs " + testCase.compareAgainst, baselineResult, result);
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
                    String comparison = testCase.compareAgainst.equals("Baseline")
                        ? changeStr
                        : String.format("%s (vs %s)", changeStr, testCase.compareAgainst);

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

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
            double connectDistance, double connectDistanceFactor, int parallelThreshold, int level0Scale,
            double tangentAngle, double tangentStrength) {
        
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            0.0     // cachepixels disabled
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
            level0Scale,
            tangentAngle, tangentStrength,
            1.0     // cachepixels enabled
        );
        cases.add(new TestCase("CachePixels Enabled", "cache=ON, parallel=ON, splines=ON, cachepixels=1.0", cachePixelsEnabled, "Baseline"));
        
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
        int level0Scale = 4;  // Level 0 cells contain 4x4 level 1 cells
        double tangentAngle = Math.toRadians(45);  // 45 degrees max deviation
        double tangentStrength = 0.4;  // Tangent length as fraction of segment
        double cachepixels = 0;  // Pixel cache disabled for benchmarks (use 0)

        // Create test cases in table-like format
        List<TestCase> testCases = createTestCases(n, epsilon, delta, slope, gridsize, returnType, salt, 
            defaultBranches, curvature, curvatureFalloff, connectDistance, connectDistanceFactor,
            parallelThreshold, level0Scale, tangentAngle, tangentStrength);

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
        System.out.println("=".repeat(60));
        System.out.println("SUMMARY (samples/sec - higher is better)");
        System.out.println("=".repeat(60));
        
        DendryBenchmark.BenchmarkResult baselineResult = results.get("Baseline");
        if (baselineResult != null) {
            System.out.printf("  %-18s %,.0f samples/sec%n", "Baseline", baselineResult.samplesPerSecond);
        }
        
        for (TestCase testCase : testCases) {
            if (testCase.compareAgainst != null) {
                DendryBenchmark.BenchmarkResult result = results.get(testCase.name);
                DendryBenchmark.BenchmarkResult baseline = results.get(testCase.compareAgainst);
                if (result != null && baseline != null) {
                    double change = percentChange(baseline, result);
                    System.out.printf("  %-18s %,.0f samples/sec (%+.1f%%)%n", 
                        testCase.name, result.samplesPerSecond, change);
                }
            }
        }
        System.out.println();
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

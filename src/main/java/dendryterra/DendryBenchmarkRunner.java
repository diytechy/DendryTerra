package dendryterra;

/**
 * Standalone benchmark runner for testing DendrySampler performance.
 *
 * Run with: java -cp DendryTerra.jar dendryterra.DendryBenchmarkRunner
 *
 * Or from gradle: ./gradlew run (after adding application plugin)
 */
public class DendryBenchmarkRunner {

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
        DendryReturnType returnType = DendryReturnType.ELEVATION;
        long salt = 12345;
        int defaultBranches = 2;
        double curvature = 0.5;
        double curvatureFalloff = 0.7;
        double connectDistance = 0;
        double connectDistanceFactor = 2.0;
        int parallelThreshold = 100;

        // Create different configurations to test
        System.out.println("Creating test configurations...");
        System.out.println();

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
            parallelThreshold
        );

        // 2. No cache
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
            parallelThreshold
        );

        // 3. No parallel
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
            parallelThreshold
        );

        // 4. No splines (linear subdivision)
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
            parallelThreshold
        );

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
            parallelThreshold
        );

        // 6. Higher resolution (n=3)
        DendrySampler highRes = new DendrySampler(
            3,      // n = 3 (more detail)
            epsilon, delta, slope, gridsize,
            returnType, null, salt,
            null, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            true, true, true, false, parallelThreshold
        );

        // Run benchmarks
        double worldScale = 10.0;
        int warmupIterations = 1;

        System.out.println("-".repeat(60));
        System.out.println("TEST 1: Baseline (cache=ON, parallel=ON, splines=ON, n=2)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r1 = DendryBenchmark.benchmark(baseline, gridSize, worldScale, warmupIterations);
        printResult(r1);

        System.out.println("-".repeat(60));
        System.out.println("TEST 2: No Cache (cache=OFF, parallel=ON, splines=ON)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r2 = DendryBenchmark.benchmark(noCache, gridSize, worldScale, warmupIterations);
        printResult(r2);
        printComparison("vs Baseline", r1, r2);

        System.out.println("-".repeat(60));
        System.out.println("TEST 3: No Parallel (cache=ON, parallel=OFF, splines=ON)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r3 = DendryBenchmark.benchmark(noParallel, gridSize, worldScale, warmupIterations);
        printResult(r3);
        printComparison("vs Baseline", r1, r3);

        System.out.println("-".repeat(60));
        System.out.println("TEST 4: No Splines (cache=ON, parallel=ON, splines=OFF)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r4 = DendryBenchmark.benchmark(noSplines, gridSize, worldScale, warmupIterations);
        printResult(r4);
        printComparison("vs Baseline", r1, r4);

        System.out.println("-".repeat(60));
        System.out.println("TEST 5: Minimal (cache=OFF, parallel=OFF, splines=OFF)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r5 = DendryBenchmark.benchmark(minimal, gridSize, worldScale, warmupIterations);
        printResult(r5);
        printComparison("vs Baseline", r1, r5);

        System.out.println("-".repeat(60));
        System.out.println("TEST 6: High Resolution (n=3, all optimizations ON)");
        System.out.println("-".repeat(60));
        DendryBenchmark.BenchmarkResult r6 = DendryBenchmark.benchmark(highRes, gridSize, worldScale, warmupIterations);
        printResult(r6);
        printComparison("vs Baseline (n=2)", r1, r6);

        // Summary
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("SUMMARY (samples/sec - higher is better)");
        System.out.println("=".repeat(60));
        System.out.printf("  Baseline (n=2):    %,.0f samples/sec%n", r1.samplesPerSecond);
        System.out.printf("  No Cache:          %,.0f samples/sec (%.1f%%)%n", r2.samplesPerSecond, percentChange(r1, r2));
        System.out.printf("  No Parallel:       %,.0f samples/sec (%.1f%%)%n", r3.samplesPerSecond, percentChange(r1, r3));
        System.out.printf("  No Splines:        %,.0f samples/sec (%.1f%%)%n", r4.samplesPerSecond, percentChange(r1, r4));
        System.out.printf("  Minimal:           %,.0f samples/sec (%.1f%%)%n", r5.samplesPerSecond, percentChange(r1, r5));
        System.out.printf("  High Res (n=3):    %,.0f samples/sec (%.1f%%)%n", r6.samplesPerSecond, percentChange(r1, r6));
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

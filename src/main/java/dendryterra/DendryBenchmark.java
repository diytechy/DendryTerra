package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Benchmark utility for testing DendrySampler performance.
 * Can be invoked programmatically or via logging to measure execution times.
 */
public final class DendryBenchmark {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendryBenchmark.class);

    private DendryBenchmark() {}

    /**
     * Run a benchmark on the given sampler.
     *
     * @param sampler The sampler to benchmark
     * @param gridSize Size of the test grid (gridSize x gridSize samples)
     * @param worldScale Scale factor for world coordinates
     * @param warmupIterations Number of warmup iterations before timing
     * @return Average time per sample in nanoseconds
     */
    public static BenchmarkResult benchmark(Sampler sampler, int gridSize, double worldScale, int warmupIterations) {
        long seed = 12345L;
        int totalSamples = gridSize * gridSize;

        // Warmup phase
        LOGGER.info("Warmup: {} iterations...", warmupIterations);
        for (int w = 0; w < warmupIterations; w++) {
            for (int x = 0; x < gridSize; x++) {
                for (int z = 0; z < gridSize; z++) {
                    sampler.getSample(seed, x * worldScale, z * worldScale);
                }
            }
        }

        // Timed phase
        LOGGER.info("Timing: {} samples...", totalSamples);
        long startTime = System.nanoTime();

        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;
        double sum = 0;

        for (int x = 0; x < gridSize; x++) {
            for (int z = 0; z < gridSize; z++) {
                double value = sampler.getSample(seed, x * worldScale, z * worldScale);
                minValue = Math.min(minValue, value);
                maxValue = Math.max(maxValue, value);
                sum += value;
            }
        }

        long endTime = System.nanoTime();
        long totalTimeNs = endTime - startTime;
        double avgTimePerSampleNs = (double) totalTimeNs / totalSamples;
        double avgTimePerSampleMs = avgTimePerSampleNs / 1_000_000.0;
        double totalTimeMs = totalTimeNs / 1_000_000.0;
        double samplesPerSecond = totalSamples / (totalTimeMs / 1000.0);

        BenchmarkResult result = new BenchmarkResult(
            totalSamples,
            totalTimeMs,
            avgTimePerSampleNs,
            avgTimePerSampleMs,
            samplesPerSecond,
            minValue,
            maxValue,
            sum / totalSamples
        );

        LOGGER.info("Benchmark complete:");
        LOGGER.info("  Total samples: {}", result.totalSamples);
        LOGGER.info("  Total time: {:.2f} ms", result.totalTimeMs);
        LOGGER.info("  Avg per sample: {:.4f} ms ({:.0f} ns)", result.avgTimePerSampleMs, result.avgTimePerSampleNs);
        LOGGER.info("  Throughput: {:.0f} samples/sec", result.samplesPerSecond);
        LOGGER.info("  Value range: [{:.4f}, {:.4f}], avg: {:.4f}", result.minValue, result.maxValue, result.avgValue);

        return result;
    }

    /**
     * Quick benchmark with default settings.
     */
    public static BenchmarkResult quickBenchmark(Sampler sampler) {
        return benchmark(sampler, 64, 10.0, 1);
    }

    /**
     * Detailed benchmark with more samples.
     */
    public static BenchmarkResult detailedBenchmark(Sampler sampler) {
        return benchmark(sampler, 256, 10.0, 3);
    }

    /**
     * Benchmark result data.
     */
    public static class BenchmarkResult {
        public final int totalSamples;
        public final double totalTimeMs;
        public final double avgTimePerSampleNs;
        public final double avgTimePerSampleMs;
        public final double samplesPerSecond;
        public final double minValue;
        public final double maxValue;
        public final double avgValue;

        public BenchmarkResult(int totalSamples, double totalTimeMs, double avgTimePerSampleNs,
                              double avgTimePerSampleMs, double samplesPerSecond,
                              double minValue, double maxValue, double avgValue) {
            this.totalSamples = totalSamples;
            this.totalTimeMs = totalTimeMs;
            this.avgTimePerSampleNs = avgTimePerSampleNs;
            this.avgTimePerSampleMs = avgTimePerSampleMs;
            this.samplesPerSecond = samplesPerSecond;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.avgValue = avgValue;
        }

        @Override
        public String toString() {
            return String.format(
                "BenchmarkResult{samples=%d, totalMs=%.2f, avgNs=%.0f, samples/sec=%.0f, range=[%.4f,%.4f]}",
                totalSamples, totalTimeMs, avgTimePerSampleNs, samplesPerSecond, minValue, maxValue
            );
        }
    }
}

package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import dendryterra.math.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Optimized with configurable caching, parallel processing, and debug timing.
 */
public class DendrySampler implements Sampler {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendrySampler.class);

    // Configuration parameters
    private final int resolution;
    private final double epsilon;
    private final double delta;
    private final double slope;
    private final double gridsize;
    private final DendryReturnType returnType;
    private final Sampler controlSampler;
    private final long salt;

    // Branch and curvature parameters
    private final Sampler branchesSampler;
    private final int defaultBranches;
    private final double curvature;
    private final double curvatureFalloff;
    private final double connectDistance;
    private final double connectDistanceFactor;

    // Performance flags
    private final boolean useCache;
    private final boolean useParallel;
    private final boolean useSplines;
    private final boolean debugTiming;
    private final int parallelThreshold;

    // Level 0 cell scale (how many level 1 cells fit in one level 0 cell per axis)
    private final int level0Scale;

    // Spline tangent parameters
    private final double tangentAngle;    // Max angle deviation (radians)
    private final double tangentStrength; // Tangent length as fraction of segment length

    // Pixel cache parameters
    private final double cachepixels;     // Pixel cache resolution (0 = disabled)
    private final int pixelGridSize;      // Number of pixels per cell axis (gridsize / cachepixels)

    // Cache configuration
    private static final int MAX_CACHE_SIZE = 16384;
    private static final int MAX_PIXEL_CACHE_BYTES = 20 * 1024 * 1024; // 20 MB max for pixel cache
    private static final int PIXEL_DATA_SIZE = 9; // 2 (x) + 2 (y) + 4 (elevation) + 1 (level) bytes

    // Lazy LRU cache (optional based on useCache flag)
    private final LoadingCache<Long, CellData> cellCache;

    // Timing statistics (only used when debugTiming is true)
    private final AtomicLong sampleCount = new AtomicLong(0);
    private final AtomicLong totalTimeNs = new AtomicLong(0);
    private volatile long lastLogTime = 0;
    private static final long LOG_INTERVAL_MS = 5000; // Log every 5 seconds

    // Pixel cache statistics (for debugging cache performance)
    private final AtomicLong pixelCacheHits = new AtomicLong(0);
    private final AtomicLong pixelCacheMisses = new AtomicLong(0);

    private static class CellData {
        final Point2D point;
        final int branchCount;

        CellData(Point2D point, int branchCount) {
            this.point = point;
            this.branchCount = branchCount;
        }
    }

    /**
     * Pixel data for a single cached point along a segment.
     * Stores position (as offset from cell origin), elevation, and resolution level.
     */
    private static class PixelData {
        final int xOffset;     // X offset from cell origin (0 to pixelGridSize-1)
        final int yOffset;     // Y offset from cell origin (0 to pixelGridSize-1)
        final float elevation; // Elevation at this point
        final byte level;      // Resolution level (0-5)

        PixelData(int xOffset, int yOffset, float elevation, byte level) {
            this.xOffset = xOffset;
            this.yOffset = yOffset;
            this.elevation = elevation;
            this.level = level;
        }
    }

    /**
     * Cached pixel grid for a single level 1 cell.
     * Stores a 2D array of pixel data indexed by pixel coordinates.
     */
    private static class CellPixelData {
        final int cellX;
        final int cellY;
        final int gridSize;
        final PixelData[][] pixels;  // [y][x] indexed
        long lastAccessTime;
        boolean populated;  // True if segments have been sampled into this cache

        CellPixelData(int cellX, int cellY, int gridSize) {
            this.cellX = cellX;
            this.cellY = cellY;
            this.gridSize = gridSize;
            this.pixels = new PixelData[gridSize][gridSize];
            this.lastAccessTime = System.nanoTime();
            this.populated = false;
        }

        void setPixel(int px, int py, float elevation, byte level) {
            if (px >= 0 && px < gridSize && py >= 0 && py < gridSize) {
                // Only set if empty or if new level is lower (more significant)
                PixelData existing = pixels[py][px];
                if (existing == null || level < existing.level) {
                    pixels[py][px] = new PixelData(px, py, elevation, level);
                }
            }
        }

        PixelData getPixel(int px, int py) {
            if (px >= 0 && px < gridSize && py >= 0 && py < gridSize) {
                lastAccessTime = System.nanoTime();
                return pixels[py][px];
            }
            return null;
        }

        int getMemorySize() {
            // Approximate memory: header + array refs + pixel objects
            int pixelCount = 0;
            for (int y = 0; y < gridSize; y++) {
                for (int x = 0; x < gridSize; x++) {
                    if (pixels[y][x] != null) pixelCount++;
                }
            }
            // PixelData object overhead (~24 bytes) + fields (4+4+4+1 = 13 bytes)
            return 64 + (gridSize * gridSize * 8) + (pixelCount * 40);
        }
    }

    // LRU pixel cache (cell key -> pixel data)
    private final Map<Long, CellPixelData> pixelCache;
    private final Object pixelCacheLock = new Object();

    public DendrySampler(int resolution, double epsilon, double delta,
                         double slope, double gridsize,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt,
                         Sampler branchesSampler, int defaultBranches,
                         double curvature, double curvatureFalloff,
                         double connectDistance, double connectDistanceFactor,
                         boolean useCache, boolean useParallel, boolean useSplines,
                         boolean debugTiming, int parallelThreshold,
                         int level0Scale,
                         double tangentAngle, double tangentStrength,
                         double cachepixels) {
        this.resolution = resolution;
        this.epsilon = epsilon;
        this.delta = delta;
        this.slope = slope;
        this.gridsize = gridsize;
        this.returnType = returnType;
        this.controlSampler = controlSampler;
        this.salt = salt;
        this.branchesSampler = branchesSampler;
        this.defaultBranches = defaultBranches;
        this.curvature = curvature;
        this.curvatureFalloff = curvatureFalloff;
        this.connectDistance = connectDistance;
        this.connectDistanceFactor = connectDistanceFactor;
        this.useCache = useCache;
        this.useParallel = useParallel;
        this.useSplines = useSplines;
        this.debugTiming = debugTiming;
        this.parallelThreshold = parallelThreshold;
        this.level0Scale = level0Scale;
        this.tangentAngle = tangentAngle;
        this.tangentStrength = tangentStrength;
        this.cachepixels = cachepixels;

        // Calculate pixel grid size
        if (cachepixels > 0) {
            this.pixelGridSize = (int) Math.ceil(gridsize / cachepixels);
        } else {
            this.pixelGridSize = 0;
        }

        // Initialize cell data cache only if enabled
        if (useCache) {
            this.cellCache = Caffeine.newBuilder()
                .maximumSize(MAX_CACHE_SIZE)
                .build(this::generateCellData);
        } else {
            this.cellCache = null;
        }

        // Initialize pixel cache if cachepixels is enabled
        if (cachepixels > 0) {
            this.pixelCache = new HashMap<>();
        } else {
            this.pixelCache = null;
        }

        if (debugTiming) {
            LOGGER.info("DendrySampler initialized with: resolution={}, gridsize={}, useCache={}, useParallel={}, useSplines={}, parallelThreshold={}, cachepixels={}, pixelGridSize={}",
                resolution, gridsize, useCache, useParallel, useSplines, parallelThreshold, cachepixels, pixelGridSize);
        }
    }

    private CellData generateCellData(Long key) {
        int cellX = unpackX(key);
        int cellY = unpackY(key);
        Point2D point = generatePoint(cellX, cellY);
        int branches = computeBranchCount(cellX, cellY);
        return new CellData(point, branches);
    }

    private static long packKey(int x, int y) {
        return ((long) x << 32) | (y & 0xFFFFFFFFL);
    }

    private static int unpackX(long key) {
        return (int) (key >> 32);
    }

    private static int unpackY(long key) {
        return (int) key;
    }

    private int computeBranchCount(int cellX, int cellY) {
        if (branchesSampler == null) {
            return defaultBranches;
        }
        double centerX = (cellX + 0.5) * gridsize;
        double centerY = (cellY + 0.5) * gridsize;
        int branches = (int) Math.round(branchesSampler.getSample(salt, centerX, centerY));
        return Math.max(1, Math.min(8, branches));
    }

    @Override
    public double getSample(long seed, double x, double z) {
        long startTime = debugTiming ? System.nanoTime() : 0;

        double result;
        double normalizedX = x / gridsize;
        double normalizedZ = z / gridsize;

        // Use pixel cache for PIXEL_ELEVATION and PIXEL_LEVEL return types
        if (usesPixelCache()) {
            result = evaluateWithPixelCache(seed, normalizedX, normalizedZ);
        } else {
            result = evaluate(seed, normalizedX, normalizedZ);
        }

        if (debugTiming) {
            long elapsed = System.nanoTime() - startTime;
            totalTimeNs.addAndGet(elapsed);
            long count = sampleCount.incrementAndGet();

            long now = System.currentTimeMillis();
            if (now - lastLogTime > LOG_INTERVAL_MS) {
                lastLogTime = now;
                double avgNs = (double) totalTimeNs.get() / count;
                double avgMs = avgNs / 1_000_000.0;
                long hits = pixelCacheHits.get();
                long misses = pixelCacheMisses.get();
                double hitRate = (hits + misses) > 0 ? (100.0 * hits / (hits + misses)) : 0;
                LOGGER.info("DendrySampler stats: {} samples, avg {:.4f} ms/sample ({:.0f} ns), pixel cache: {} hits, {} misses ({:.1f}% hit rate)",
                    count, avgMs, avgNs, hits, misses, hitRate);
            }
        }

        return result;
    }

    @Override
    public double getSample(long seed, double x, double y, double z) {
        return getSample(seed, x, z);
    }

    private double evaluate(long seed, double x, double y) {
        // Displacement factors decrease exponentially for each level
        double displacementLevel0 = delta * 2.0;
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slope constraints for sub-branch connections
        double minSlopeLevel1 = 0.5;
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 0: Cell-based network with guaranteed connectivity
        Cell cell1 = getCell(x, y, 1);

        // Step 1: Generate L0 BASE segments (no subdivision yet)
        List<Segment3D> segments0Base = generateLevel0Network(cell1);

        // Step 2: Compute L0 tangents on FULL L0 set (before pruning for determinism)
        Map<Long, NodeTangentInfo> tangentMap0 = computeNodeTangents(segments0Base);

        // Step 3: Prune L0 to 3x3 region
        List<Segment3D> segments0Pruned = pruneToQueryRegion(segments0Base, cell1);

        // Step 4: Subdivide and displace L0 using pre-computed tangents
        int level0Subdivisions = Math.max(2, defaultBranches);
        segments0Pruned = subdivideSegments(segments0Pruned, level0Subdivisions, 0, tangentMap0);
        segments0Pruned = displaceSegmentsWithSplit(segments0Pruned, displacementLevel0, 0);

        if (resolution == 0) {
            return computeResult(x, y, segments0Pruned);
        }

        // Level 1: Generate L1 from PRUNED+SUBDIVIDED L0 (only 3x3 cells needed now)
        // Since L0 is already pruned to 3x3, L1 only needs to cover 3x3 cells
        List<Segment3D> segments1Base = generateLevel1SegmentsCompact(cell1, segments0Pruned, minSlopeLevel1);

        // Compute L1 tangents
        Map<Long, NodeTangentInfo> tangentMap1 = computeNodeTangents(segments1Base);

        // Subdivide and displace L1
        int branchCount = getBranchCountForCell(cell1);
        List<Segment3D> segments1 = subdivideSegments(segments1Base, branchCount, 1, tangentMap1);
        segments1 = displaceSegmentsWithSplit(segments1, displacementLevel1, 1);

        List<Segment3D> allSegments = new ArrayList<>(segments0Pruned);
        allSegments.addAll(segments1);

        if (resolution == 1) {
            return computeResult(x, y, allSegments);
        }

        // Level 2+: Continue with existing logic
        // Level 2: 2x Resolution
        Cell cell2 = getCell(x, y, 2);
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);
        List<Segment3D> segments2 = generateSubSegments(points2, allSegments, minSlopeLevel2, 2);
        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            allSegments.addAll(segments2);
            return computeResult(x, y, allSegments);
        }

        // Level 3: 4x Resolution
        Cell cell3 = getCell(x, y, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);
        allSegments.addAll(segments2);
        List<Segment3D> segments3 = generateSubSegments(points3, allSegments, minSlopeLevel3, 3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allSegments.addAll(segments3);
            return computeResult(x, y, allSegments);
        }

        // Level 4: 8x Resolution
        Cell cell4 = getCell(x, y, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allSegments, minSlopeLevel4, 4);

        if (resolution == 4) {
            allSegments.addAll(segments4);
            return computeResult(x, y, allSegments);
        }

        // Level 5: 16x Resolution
        Cell cell5 = getCell(x, y, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allSegments, minSlopeLevel5, 5);
        allSegments.addAll(segments5);

        return computeResult(x, y, allSegments);
    }

    private static class Cell {
        final int x;
        final int y;
        final int resolution;

        Cell(int x, int y, int resolution) {
            this.x = x;
            this.y = y;
            this.resolution = resolution;
        }
    }

    private Cell getCell(double x, double y, int resolution) {
        int cellX = MathUtils.floor(x * resolution);
        int cellY = MathUtils.floor(y * resolution);
        return new Cell(cellX, cellY, resolution);
    }

    private Point2D generatePoint(int cellX, int cellY) {
        Random rng = initRandomGenerator(cellX, cellY, 1);
        double px = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        double py = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        return new Point2D(cellX + px, cellY + py);
    }

    /**
     * Get cell data - uses cache if enabled, otherwise generates directly.
     */
    private CellData getCellData(int cellX, int cellY) {
        if (useCache && cellCache != null) {
            return cellCache.get(packKey(cellX, cellY));
        }
        // Direct generation without caching
        Point2D point = generatePoint(cellX, cellY);
        int branches = computeBranchCount(cellX, cellY);
        return new CellData(point, branches);
    }

    private int getBranchCountForCell(Cell cell) {
        CellData data = getCellData(cell.x, cell.y);
        return data.branchCount;
    }

    private Random initRandomGenerator(int x, int y, int level) {
        long seed = (541L * x + 79L * y + level * 863L + salt) & 0x7FFFFFFFL;
        return new Random(seed);
    }

    private Point3D[][] generateNeighboringPoints3D(Cell cell, int size) {
        Point3D[][] points = new Point3D[size][size];
        int half = size / 2;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int px = cell.x + j - half;
                int py = cell.y + i - half;

                CellData data = getCellData(px, py);
                Point2D p2d = data.point;

                Point2D scaled = new Point2D(p2d.x / cell.resolution, p2d.y / cell.resolution);
                double elevation = evaluateControlFunction(scaled.x, scaled.y);
                points[i][j] = new Point3D(scaled.x, scaled.y, elevation);
            }
        }
        return points;
    }

    // ========== Level 0 Cell-Based Network Generation ==========

    /**
     * Get level 0 cell coordinates for a given level 1 cell.
     * Level 0 cells are level0Scale times larger than level 1 cells.
     */
    private Cell getLevel0Cell(int level1CellX, int level1CellY) {
        int l0x = Math.floorDiv(level1CellX, level0Scale);
        int l0y = Math.floorDiv(level1CellY, level0Scale);
        return new Cell(l0x, l0y, 0);
    }

    /**
     * Generate all level 0 segments for the query region.
     * This includes computing networks for all required level 0 cells,
     * stitching them together at boundaries, and pruning to the relevant area.
     */
    private List<Segment3D> generateLevel0Network(Cell queryCell1) {
        // Determine which level 0 cells we need based on query cell position
        java.util.Set<Long> requiredL0Cells = getRequiredLevel0Cells(queryCell1);

        // Generate segment network for each level 0 cell
        Map<Long, List<Segment3D>> l0CellSegments = new HashMap<>();
        Map<Long, Map<Long, Point3D>> l0CellPoints = new HashMap<>();  // level0Key -> (level1Key -> point)

        for (long l0Key : requiredL0Cells) {
            int l0x = unpackX(l0Key);
            int l0y = unpackY(l0Key);

            Map<Long, Point3D> points = generateLevel0CellPoints(l0x, l0y);
            List<Segment3D> segments = buildTreeWithinLevel0Cell(points, l0x, l0y);

            l0CellPoints.put(l0Key, points);
            l0CellSegments.put(l0Key, segments);
        }

        // Stitch adjacent level 0 cells together at their boundaries
        List<Segment3D> stitchSegments = stitchLevel0Cells(requiredL0Cells, l0CellPoints);

        // Combine all segments (no pruning here - prune after spline subdivision in evaluate())
        List<Segment3D> allSegments = new ArrayList<>();
        for (List<Segment3D> segs : l0CellSegments.values()) {
            allSegments.addAll(segs);
        }
        allSegments.addAll(stitchSegments);

        return allSegments;
    }

    /**
     * Determine which level 0 cells need to be computed for a query.
     * We check a 5x5 grid of level 1 cells around the query cell to ensure:
     * 1. All level 0 cells containing the 3x3 query region are included
     * 2. Level 0 cells needed for tangent computation at boundaries are included
     * 3. Stitching segments are properly computed for adjacent level 0 cells
     */
    private java.util.Set<Long> getRequiredLevel0Cells(Cell queryCell1) {
        java.util.Set<Long> required = new java.util.HashSet<>();

        // Check a 5x5 grid (-2 to +2) to ensure we capture adjacent L0 cells
        // needed for proper tangent computation at L0 boundaries
        for (int di = -2; di <= 2; di++) {
            for (int dj = -2; dj <= 2; dj++) {
                int l1x = queryCell1.x + dj;
                int l1y = queryCell1.y + di;
                Cell l0Cell = getLevel0Cell(l1x, l1y);
                required.add(packKey(l0Cell.x, l0Cell.y));
            }
        }

        return required;
    }

    /**
     * Generate points for all level 1 cells within a level 0 cell.
     * Returns a map from level 1 cell key to its lowest elevation point.
     */
    private Map<Long, Point3D> generateLevel0CellPoints(int l0x, int l0y) {
        Map<Long, Point3D> points = new HashMap<>();

        // Iterate over all level 1 cells in this level 0 cell
        int startX = l0x * level0Scale;
        int startY = l0y * level0Scale;

        for (int i = 0; i < level0Scale; i++) {
            for (int j = 0; j < level0Scale; j++) {
                int l1x = startX + j;
                int l1y = startY + i;

                Point3D point = findLowestPointInCell(l1x, l1y);
                points.put(packKey(l1x, l1y), point);
            }
        }

        return points;
    }

    /**
     * Find the approximate lowest elevation point within a level 1 cell by sampling.
     * Uses a 5x5 grid of sample points and selects the one with minimum elevation.
     * Adds deterministic jitter to avoid grid-aligned patterns across cells.
     */
    private Point3D findLowestPointInCell(int cellX, int cellY) {
        final int SAMPLES_PER_AXIS = 5;

        // Generate deterministic jitter for this cell
        Random rng = initRandomGenerator(cellX, cellY, 0);
        double jitterX = (rng.nextDouble() - 0.5) * 0.15;
        double jitterY = (rng.nextDouble() - 0.5) * 0.15;

        double lowestElevation = Double.MAX_VALUE;
        double lowestX = cellX + 0.5;
        double lowestY = cellY + 0.5;

        for (int si = 0; si < SAMPLES_PER_AXIS; si++) {
            for (int sj = 0; sj < SAMPLES_PER_AXIS; sj++) {
                double tx = 0.1 + 0.8 * (sj + 0.5) / SAMPLES_PER_AXIS + jitterX;
                double ty = 0.1 + 0.8 * (si + 0.5) / SAMPLES_PER_AXIS + jitterY;

                tx = Math.max(0.05, Math.min(0.95, tx));
                ty = Math.max(0.05, Math.min(0.95, ty));

                double sampleX = cellX + tx;
                double sampleY = cellY + ty;

                double elevation = evaluateControlFunction(sampleX, sampleY);

                if (elevation < lowestElevation) {
                    lowestElevation = elevation;
                    lowestX = sampleX;
                    lowestY = sampleY;
                }
            }
        }

        return new Point3D(lowestX, lowestY, lowestElevation);
    }

    /**
     * Build a tree structure connecting all points within a level 0 cell.
     * Algorithm:
     * A. Connect each node to its closest neighbor (by 2D distance)
     * B. While not all nodes form a single tree, connect disconnected components
     *    starting from lowest elevation nodes
     */
    private List<Segment3D> buildTreeWithinLevel0Cell(Map<Long, Point3D> points, int l0x, int l0y) {
        if (points.size() <= 1) return new ArrayList<>();

        List<Point3D> pointList = new ArrayList<>(points.values());
        int n = pointList.size();

        // Build all edges sorted by distance
        List<Edge> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Point3D p1 = pointList.get(i);
                Point3D p2 = pointList.get(j);
                double dist = p1.projectZ().distanceTo(p2.projectZ());
                edges.add(new Edge(i, j, dist, p1, p2));
            }
        }
        edges.sort(Comparator.comparingDouble(e -> e.distance));

        // Phase A: Connect each node to its closest neighbor
        List<Segment3D> segments = new ArrayList<>();
        int[] parent = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        // For each node, find and add its closest neighbor connection
        boolean[] hasConnection = new boolean[n];
        for (int i = 0; i < n; i++) {
            double minDist = Double.MAX_VALUE;
            int closest = -1;
            Point3D current = pointList.get(i);

            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                Point3D other = pointList.get(j);
                double dist = current.projectZ().distanceTo(other.projectZ());
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }

            if (closest >= 0) {
                int rootI = find(parent, i);
                int rootJ = find(parent, closest);

                // Add segment (avoid exact duplicates)
                if (rootI != rootJ || !hasConnection[i]) {
                    segments.add(new Segment3D(current, pointList.get(closest), 0));
                    hasConnection[i] = true;
                    hasConnection[closest] = true;
                    if (rootI != rootJ) {
                        union(parent, rank, rootI, rootJ);
                    }
                }
            }
        }

        // Phase B: Ensure all nodes are connected (form a single tree)
        // Sort nodes by elevation for deterministic tie-breaking
        List<Integer> nodesByElevation = new ArrayList<>();
        for (int i = 0; i < n; i++) nodesByElevation.add(i);
        nodesByElevation.sort(Comparator.comparingDouble(i -> pointList.get(i).z));

        // Use Kruskal's to connect any remaining disconnected components
        for (Edge edge : edges) {
            int rootA = find(parent, edge.idx1);
            int rootB = find(parent, edge.idx2);

            if (rootA != rootB) {
                segments.add(new Segment3D(edge.p1, edge.p2, 0));
                union(parent, rank, rootA, rootB);
            }
        }

        return segments;
    }

    /**
     * Stitch adjacent level 0 cells together at their boundaries.
     * For each pair of adjacent level 0 cells, find the boundary level 1 cell
     * with the lowest elevation on each side and connect them.
     */
    private List<Segment3D> stitchLevel0Cells(java.util.Set<Long> l0Cells,
                                               Map<Long, Map<Long, Point3D>> l0CellPoints) {
        List<Segment3D> stitchSegments = new ArrayList<>();
        java.util.Set<Long> processedPairs = new java.util.HashSet<>();

        for (long l0Key : l0Cells) {
            int l0x = unpackX(l0Key);
            int l0y = unpackY(l0Key);

            // Check right neighbor (+x direction)
            long rightKey = packKey(l0x + 1, l0y);
            if (l0Cells.contains(rightKey) && !processedPairs.contains(packKey(Math.min(l0x, l0x+1), Math.max(l0x, l0x+1) * 10000 + l0y))) {
                processedPairs.add(packKey(Math.min(l0x, l0x+1), Math.max(l0x, l0x+1) * 10000 + l0y));
                Segment3D stitch = stitchVerticalBoundary(l0x, l0y, l0CellPoints.get(l0Key), l0CellPoints.get(rightKey));
                if (stitch != null) stitchSegments.add(stitch);
            }

            // Check bottom neighbor (+y direction)
            long bottomKey = packKey(l0x, l0y + 1);
            if (l0Cells.contains(bottomKey) && !processedPairs.contains(packKey(l0x, Math.min(l0y, l0y+1) * 10000 + Math.max(l0y, l0y+1)))) {
                processedPairs.add(packKey(l0x, Math.min(l0y, l0y+1) * 10000 + Math.max(l0y, l0y+1)));
                Segment3D stitch = stitchHorizontalBoundary(l0x, l0y, l0CellPoints.get(l0Key), l0CellPoints.get(bottomKey));
                if (stitch != null) stitchSegments.add(stitch);
            }
        }

        return stitchSegments;
    }

    /**
     * Stitch two level 0 cells at a vertical boundary (between l0x and l0x+1).
     * Finds the lowest elevation cell on each side of the boundary and connects them.
     */
    private Segment3D stitchVerticalBoundary(int l0x, int l0y,
                                              Map<Long, Point3D> leftPoints,
                                              Map<Long, Point3D> rightPoints) {
        // Right edge of left cell: x = (l0x + 1) * level0Scale - 1
        // Left edge of right cell: x = (l0x + 1) * level0Scale
        int leftEdgeX = (l0x + 1) * level0Scale - 1;
        int rightEdgeX = (l0x + 1) * level0Scale;
        int startY = l0y * level0Scale;

        Point3D lowestLeft = null;
        Point3D lowestRight = null;
        double lowestLeftElev = Double.MAX_VALUE;
        double lowestRightElev = Double.MAX_VALUE;

        for (int i = 0; i < level0Scale; i++) {
            int y = startY + i;

            Point3D leftPoint = leftPoints.get(packKey(leftEdgeX, y));
            if (leftPoint != null && leftPoint.z < lowestLeftElev) {
                lowestLeftElev = leftPoint.z;
                lowestLeft = leftPoint;
            }

            Point3D rightPoint = rightPoints.get(packKey(rightEdgeX, y));
            if (rightPoint != null && rightPoint.z < lowestRightElev) {
                lowestRightElev = rightPoint.z;
                lowestRight = rightPoint;
            }
        }

        if (lowestLeft != null && lowestRight != null) {
            return new Segment3D(lowestLeft, lowestRight, 0);
        }
        return null;
    }

    /**
     * Stitch two level 0 cells at a horizontal boundary (between l0y and l0y+1).
     */
    private Segment3D stitchHorizontalBoundary(int l0x, int l0y,
                                                Map<Long, Point3D> topPoints,
                                                Map<Long, Point3D> bottomPoints) {
        int topEdgeY = (l0y + 1) * level0Scale - 1;
        int bottomEdgeY = (l0y + 1) * level0Scale;
        int startX = l0x * level0Scale;

        Point3D lowestTop = null;
        Point3D lowestBottom = null;
        double lowestTopElev = Double.MAX_VALUE;
        double lowestBottomElev = Double.MAX_VALUE;

        for (int i = 0; i < level0Scale; i++) {
            int x = startX + i;

            Point3D topPoint = topPoints.get(packKey(x, topEdgeY));
            if (topPoint != null && topPoint.z < lowestTopElev) {
                lowestTopElev = topPoint.z;
                lowestTop = topPoint;
            }

            Point3D bottomPoint = bottomPoints.get(packKey(x, bottomEdgeY));
            if (bottomPoint != null && bottomPoint.z < lowestBottomElev) {
                lowestBottomElev = bottomPoint.z;
                lowestBottom = bottomPoint;
            }
        }

        if (lowestTop != null && lowestBottom != null) {
            return new Segment3D(lowestTop, lowestBottom, 0);
        }
        return null;
    }

    /**
     * Prune segments to only keep those relevant to the query cell and its neighbors.
     * Keeps segments where at least one endpoint is within the 3x3 region around query cell.
     */
    private List<Segment3D> pruneToQueryRegion(List<Segment3D> segments, Cell queryCell1) {
        double minX = queryCell1.x - 1;
        double maxX = queryCell1.x + 2;
        double minY = queryCell1.y - 1;
        double maxY = queryCell1.y + 2;

        return segments.stream()
            .filter(seg -> {
                boolean aInside = isPointInBounds(seg.a, minX, minY, maxX, maxY);
                boolean bInside = isPointInBounds(seg.b, minX, minY, maxX, maxY);
                return aInside || bInside;
            })
            .collect(Collectors.toList());
    }

    /**
     * Edge class for MST algorithm.
     */
    private static class Edge {
        final int idx1, idx2;
        final double distance;
        final Point3D p1, p2;

        Edge(int idx1, int idx2, double distance, Point3D p1, Point3D p2) {
            this.idx1 = idx1;
            this.idx2 = idx2;
            this.distance = distance;
            this.p1 = p1;
            this.p2 = p2;
        }
    }

    /**
     * Pre-computed tangent information for spline control points at a node.
     * Stores tangent vectors for each connected neighbor, computed deterministically.
     */
    private static class NodeTangentInfo {
        final Point3D node;
        final Map<Long, Vec2D> tangents;  // neighborKey -> tangent vector (unit direction)

        NodeTangentInfo(Point3D node) {
            this.node = node;
            this.tangents = new HashMap<>();
        }
    }

    /**
     * Compute tangent information for all nodes in a segment list.
     * Tangents are computed deterministically based on node positions and connections,
     * independent of segment list order.
     */
    private Map<Long, NodeTangentInfo> computeNodeTangents(List<Segment3D> segments) {
        // Step 1: Build connectivity map: node -> list of connected nodes
        Map<Long, Point3D> nodePoints = new HashMap<>();
        Map<Long, List<Point3D>> connectivity = new HashMap<>();

        for (Segment3D seg : segments) {
            long keyA = pointHash(seg.a);
            long keyB = pointHash(seg.b);

            nodePoints.putIfAbsent(keyA, seg.a);
            nodePoints.putIfAbsent(keyB, seg.b);

            connectivity.computeIfAbsent(keyA, k -> new ArrayList<>()).add(seg.b);
            connectivity.computeIfAbsent(keyB, k -> new ArrayList<>()).add(seg.a);
        }

        // Step 2: For each node, compute tangent vectors
        Map<Long, NodeTangentInfo> tangentMap = new HashMap<>();

        for (Map.Entry<Long, List<Point3D>> entry : connectivity.entrySet()) {
            long nodeKey = entry.getKey();
            Point3D node = nodePoints.get(nodeKey);
            List<Point3D> neighbors = entry.getValue();

            NodeTangentInfo info = new NodeTangentInfo(node);

            if (neighbors.size() == 1) {
                // Single connection - tangent points toward neighbor
                Point3D neighbor = neighbors.get(0);
                Vec2D toNeighbor = new Vec2D(node.projectZ(), neighbor.projectZ());
                if (toNeighbor.lengthSquared() > MathUtils.EPSILON) {
                    toNeighbor = toNeighbor.normalize();
                }
                info.tangents.put(pointHash(neighbor), toNeighbor);
            } else {
                // Multiple connections - sort by angle for deterministic ordering
                neighbors.sort((a, b) -> {
                    double angleA = Math.atan2(a.y - node.y, a.x - node.x);
                    double angleB = Math.atan2(b.y - node.y, b.x - node.x);
                    return Double.compare(angleA, angleB);
                });

                int n = neighbors.size();
                for (int i = 0; i < n; i++) {
                    Point3D neighbor = neighbors.get(i);
                    long neighborKey = pointHash(neighbor);

                    // Base direction toward neighbor
                    Vec2D toNeighbor = new Vec2D(node.projectZ(), neighbor.projectZ());
                    double dist = toNeighbor.length();
                    if (dist < MathUtils.EPSILON) {
                        info.tangents.put(neighborKey, new Vec2D(1, 0));
                        continue;
                    }
                    toNeighbor = toNeighbor.normalize();

                    // For 2 connections, tangent is influenced by the opposite direction
                    // For 3+ connections, rotate tangent to avoid overlap with neighbors
                    Vec2D tangent;
                    if (n == 2) {
                        // Two connections: tangent is rotated toward the "flow" direction
                        // The tangent for going TO neighbor should come FROM the opposite neighbor
                        Point3D opposite = neighbors.get(1 - i);
                        Vec2D fromOpposite = new Vec2D(opposite.projectZ(), node.projectZ());
                        if (fromOpposite.lengthSquared() > MathUtils.EPSILON) {
                            fromOpposite = fromOpposite.normalize();
                            // Blend between direct direction and flow direction
                            tangent = blendTangent(toNeighbor, fromOpposite, tangentAngle);
                        } else {
                            tangent = toNeighbor;
                        }
                    } else {
                        // 3+ connections: rotate tangent to create separation
                        // Find the angular "gap" between this neighbor and the next
                        Point3D prevNeighbor = neighbors.get((i - 1 + n) % n);
                        Point3D nextNeighbor = neighbors.get((i + 1) % n);

                        double angleToPrev = Math.atan2(prevNeighbor.y - node.y, prevNeighbor.x - node.x);
                        double angleToNext = Math.atan2(nextNeighbor.y - node.y, nextNeighbor.x - node.x);
                        double angleToCurrent = Math.atan2(neighbor.y - node.y, neighbor.x - node.x);

                        // Calculate angular gaps
                        double gapToPrev = normalizeAngle(angleToCurrent - angleToPrev);
                        double gapToNext = normalizeAngle(angleToNext - angleToCurrent);

                        // Rotate slightly toward the larger gap to spread tangents
                        double rotationAngle = 0;
                        if (gapToPrev > gapToNext) {
                            rotationAngle = -Math.min(tangentAngle, gapToPrev * 0.3);
                        } else {
                            rotationAngle = Math.min(tangentAngle, gapToNext * 0.3);
                        }

                        tangent = rotateVec2D(toNeighbor, rotationAngle);
                    }

                    info.tangents.put(neighborKey, tangent);
                }
            }

            tangentMap.put(nodeKey, info);
        }

        return tangentMap;
    }

    /**
     * Blend two directions, rotating 'base' toward 'influence' by up to maxAngle.
     */
    private Vec2D blendTangent(Vec2D base, Vec2D influence, double maxAngle) {
        double baseAngle = Math.atan2(base.y, base.x);
        double influenceAngle = Math.atan2(influence.y, influence.x);

        double angleDiff = normalizeAngle(influenceAngle - baseAngle);

        // Clamp rotation to maxAngle
        double rotation = MathUtils.clamp(angleDiff, -maxAngle, maxAngle);

        return rotateVec2D(base, rotation);
    }

    /**
     * Rotate a 2D vector by an angle (radians).
     */
    private Vec2D rotateVec2D(Vec2D v, double angle) {
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        return new Vec2D(
            v.x * cos - v.y * sin,
            v.x * sin + v.y * cos
        );
    }

    /**
     * Normalize angle to [-PI, PI].
     */
    private double normalizeAngle(double angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /**
     * Find root with path compression (Union-Find).
     */
    private int find(int[] parent, int i) {
        if (parent[i] != i) {
            parent[i] = find(parent, parent[i]);
        }
        return parent[i];
    }

    /**
     * Union by rank (Union-Find).
     */
    private void union(int[] parent, int[] rank, int x, int y) {
        if (rank[x] < rank[y]) {
            parent[x] = y;
        } else if (rank[x] > rank[y]) {
            parent[y] = x;
        } else {
            parent[y] = x;
            rank[x]++;
        }
    }

    /**
     * Remove segments that don't connect to the inner 3x3 cell region.
     */
    private List<Segment3D> pruneDisconnectedSegments(List<Segment3D> segments,
                                                       double minX, double minY,
                                                       double maxX, double maxY) {
        return segments.stream()
            .filter(seg -> {
                // Keep if either endpoint is within inner region
                boolean aInside = isPointInBounds(seg.a, minX, minY, maxX, maxY);
                boolean bInside = isPointInBounds(seg.b, minX, minY, maxX, maxY);
                return aInside || bInside;
            })
            .collect(Collectors.toList());
    }

    private boolean isPointInBounds(Point3D p, double minX, double minY,
                                     double maxX, double maxY) {
        return p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY;
    }

    /**
     * Generate level 1 segments for 5x5 cell region around query cell.
     * The wider region ensures proper tangent computation at the 3x3 boundary.
     * Segments will be pruned to 3x3 after tangent computation.
     */
    private List<Segment3D> generateLevel1Segments(Cell queryCell, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> allSegments = new ArrayList<>();

        // Process 5x5 grid of level 1 cells to ensure proper tangent computation
        // at the edges of the 3x3 query region
        for (int di = -2; di <= 2; di++) {
            for (int dj = -2; dj <= 2; dj++) {
                Cell cell = new Cell(queryCell.x + dj, queryCell.y + di, 1);

                // Generate points for this cell's neighborhood (5x5 including neighbors)
                Point3D[][] points = generateNeighboringPoints3D(cell, 5);

                // Generate segments that connect to parent (level 0) segments
                List<Segment3D> cellSegments = generateSubSegments(points, parentSegments, minSlope, 1);
                allSegments.addAll(cellSegments);
            }
        }

        return allSegments;
    }

    /**
     * Generate level 1 segments for compact 3x3 cell region around query cell.
     * Used when parent segments (L0) are already pruned to the 3x3 region,
     * so there's no benefit to generating L1 for a wider area.
     */
    private List<Segment3D> generateLevel1SegmentsCompact(Cell queryCell, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> allSegments = new ArrayList<>();

        // Process only 3x3 grid of level 1 cells (9 cells instead of 25)
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                Cell cell = new Cell(queryCell.x + dj, queryCell.y + di, 1);

                // Generate points for this cell's neighborhood (5x5 including neighbors)
                Point3D[][] points = generateNeighboringPoints3D(cell, 5);

                // Generate segments that connect to parent (level 0) segments
                List<Segment3D> cellSegments = generateSubSegments(points, parentSegments, minSlope, 1);
                allSegments.addAll(cellSegments);
            }
        }

        return allSegments;
    }

    /**
     * Prune segments that start outside query cell and move away from it.
     */
    private List<Segment3D> pruneAwaySegments(List<Segment3D> segments, Cell queryCell) {
        // Query cell center in normalized coordinates
        double centerX = (queryCell.x + 0.5) / queryCell.resolution;
        double centerY = (queryCell.y + 0.5) / queryCell.resolution;
        Point2D center = new Point2D(centerX, centerY);

        // Query cell bounds
        double cellMinX = (double) queryCell.x / queryCell.resolution;
        double cellMaxX = (double) (queryCell.x + 1) / queryCell.resolution;
        double cellMinY = (double) queryCell.y / queryCell.resolution;
        double cellMaxY = (double) (queryCell.y + 1) / queryCell.resolution;

        return segments.stream()
            .filter(seg -> {
                Point2D a = seg.a.projectZ();
                Point2D b = seg.b.projectZ();

                // If segment starts inside query cell, keep it
                if (isInCell(a, cellMinX, cellMinY, cellMaxX, cellMaxY)) {
                    return true;
                }

                // Segment starts outside - check if it moves toward center
                double distA = a.distanceTo(center);
                double distB = b.distanceTo(center);

                // Keep if endpoint B is closer to center than A (moving toward)
                return distB < distA;
            })
            .collect(Collectors.toList());
    }

    private boolean isInCell(Point2D p, double minX, double minY,
                              double maxX, double maxY) {
        return p.x >= minX && p.x < maxX && p.y >= minY && p.y < maxY;
    }

    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            return controlSampler.getSample(salt, x * gridsize, y * gridsize);
        }
        return x * 0.1;
    }

    private List<Segment3D> generateSegments(Point3D[][] points, int level) {
        List<Segment3D> segments = new ArrayList<>();
        int size = points.length;

        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Point3D current = points[i][j];

                double lowestElevation = Double.MAX_VALUE;
                int lowestI = i;
                int lowestJ = j;

                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        Point3D neighbor = points[i + di][j + dj];
                        if (neighbor.z < lowestElevation) {
                            lowestElevation = neighbor.z;
                            lowestI = i + di;
                            lowestJ = j + dj;
                        }
                    }
                }

                Point3D lowest = points[lowestI][lowestJ];

                if (current.distanceSquaredTo(lowest) > MathUtils.EPSILON) {
                    segments.add(new Segment3D(current, lowest, level));
                }
            }
        }
        return segments;
    }

    /**
     * Subdivide segments - uses splines if enabled, otherwise linear.
     * Computes tangents internally.
     */
    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions, int level) {
        if (segments.isEmpty() || numDivisions <= 1) {
            return segments;
        }

        if (useSplines && curvature > 0) {
            return subdivideWithSplines(segments, numDivisions, level, null);
        } else {
            return subdivideLinear(segments, numDivisions, level);
        }
    }

    /**
     * Subdivide segments using pre-computed tangent map (for performance optimization).
     * When tangentMap is provided, uses it instead of computing tangents.
     */
    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions, int level,
                                               Map<Long, NodeTangentInfo> tangentMap) {
        if (segments.isEmpty() || numDivisions <= 1) {
            return segments;
        }

        if (useSplines && curvature > 0) {
            return subdivideWithSplines(segments, numDivisions, level, tangentMap);
        } else {
            return subdivideLinear(segments, numDivisions, level);
        }
    }

    /**
     * Simple linear subdivision (fast).
     */
    private List<Segment3D> subdivideLinear(List<Segment3D> segments, int numDivisions, int level) {
        List<Segment3D> subdivided = new ArrayList<>();

        for (Segment3D seg : segments) {
            Point3D prev = seg.a;
            for (int j = 1; j <= numDivisions; j++) {
                double t = (double) j / numDivisions;
                Point3D next = (j == numDivisions) ? seg.b : Point3D.lerp(seg.a, seg.b, t);
                subdivided.add(new Segment3D(prev, next, level));
                prev = next;
            }
        }
        return subdivided;
    }

    /**
     * Catmull-Rom spline subdivision using pre-computed node tangents.
     * Tangents are computed deterministically based on node positions and connections,
     * ensuring consistent results regardless of segment list order or pruning.
     *
     * @param segments The segments to subdivide
     * @param numDivisions Number of subdivisions per segment
     * @param level Resolution level (affects curvature falloff)
     * @param precomputedTangents Optional pre-computed tangent map. If null, tangents are computed internally.
     */
    private List<Segment3D> subdivideWithSplines(List<Segment3D> segments, int numDivisions, int level,
                                                  Map<Long, NodeTangentInfo> precomputedTangents) {
        double levelCurvature = curvature * Math.pow(curvatureFalloff, level - 1);
        double levelStrength = tangentStrength * Math.pow(curvatureFalloff, level - 1);
        List<Segment3D> subdivided = new ArrayList<>();

        // Use pre-computed tangents if provided, otherwise compute them
        Map<Long, NodeTangentInfo> tangentMap = (precomputedTangents != null)
            ? precomputedTangents
            : computeNodeTangents(segments);

        for (Segment3D seg : segments) {
            Point3D p1 = seg.a;
            Point3D p2 = seg.b;
            long key1 = pointHash(p1);
            long key2 = pointHash(p2);

            // Get segment length for tangent scaling
            double segLength = p1.projectZ().distanceTo(p2.projectZ());

            // Get control points from pre-computed tangents
            Point3D p0 = getControlPointFromTangent(p1, p2, tangentMap.get(key1), segLength, levelStrength, true);
            Point3D p3 = getControlPointFromTangent(p2, p1, tangentMap.get(key2), segLength, levelStrength, false);

            Point3D prev = p1;
            for (int j = 1; j <= numDivisions; j++) {
                double t = (double) j / numDivisions;
                Point3D next;

                if (j == numDivisions) {
                    next = p2;
                } else {
                    Point3D linear = Point3D.lerp(p1, p2, t);
                    Point3D spline = CatmullRomSpline.subdivide(p0, p1, p2, p3, t);
                    next = Point3D.lerp(linear, spline, levelCurvature);
                }

                subdivided.add(new Segment3D(prev, next, level));
                prev = next;
            }
        }
        return subdivided;
    }

    /**
     * Compute a Catmull-Rom control point from pre-computed tangent info.
     *
     * @param node The node point (p1 or p2)
     * @param other The other end of the segment
     * @param tangentInfo Pre-computed tangent info for this node (may be null)
     * @param segLength Length of the segment for scaling
     * @param strength Tangent strength (0-1)
     * @param isStart True if this is the start point (p0), false if end (p3)
     * @return The control point for Catmull-Rom spline
     */
    private Point3D getControlPointFromTangent(Point3D node, Point3D other, NodeTangentInfo tangentInfo,
                                                double segLength, double strength, boolean isStart) {
        if (tangentInfo == null || tangentInfo.tangents.isEmpty()) {
            // No tangent info - fall back to linear (control point at node)
            return node;
        }

        long otherKey = pointHash(other);
        Vec2D tangent = tangentInfo.tangents.get(otherKey);

        if (tangent == null) {
            // No tangent for this connection - fall back to linear
            return node;
        }

        // Control point is offset from node along the tangent direction
        // For Catmull-Rom, p0 should be "before" p1 and p3 should be "after" p2
        // We reverse the tangent direction for p0 (isStart=true)
        double offset = segLength * strength;
        double sign = isStart ? -1.0 : 1.0;

        // Interpolate elevation based on segment direction
        double elevOffset = (other.z - node.z) * strength * sign;

        return new Point3D(
            node.x + tangent.x * offset * sign,
            node.y + tangent.y * offset * sign,
            node.z + elevOffset
        );
    }

    /**
     * Hash a point's coordinates for connectivity lookup.
     * Uses fixed precision to handle floating point comparison.
     */
    private long pointHash(Point3D p) {
        // Quantize to ~0.0001 precision for reliable matching
        long hx = (long)(p.x * 10000);
        long hy = (long)(p.y * 10000);
        return (hx * 73856093L) ^ (hy * 19349663L);
    }

    /**
     * Displace segments by splitting each into two segments with a displaced midpoint.
     * Displacement is proportional to segment length for consistent curvature appearance.
     * Returns a new list (does not modify input list structure).
     * Uses absolute segment coordinates for deterministic displacement regardless of query position.
     */
    private List<Segment3D> displaceSegmentsWithSplit(List<Segment3D> segments, double displacementFactor, int level) {
        if (displacementFactor < MathUtils.EPSILON) return segments;

        List<Segment3D> result = new ArrayList<>();

        for (Segment3D seg : segments) {
            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) {
                result.add(seg);
                continue;
            }

            Vec2D perp = dir.rotateCCW90().normalize();

            // Use both endpoints for seed to ensure determinism based on segment identity
            int seedX = (int)((seg.a.x + seg.b.x) * 50);
            int seedY = (int)((seg.a.y + seg.b.y) * 50);
            Random rng = initRandomGenerator(seedX, seedY, level);
            // Displacement proportional to segment length
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor * segLength;

            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            // Split into two segments: a→mid and mid→b (preserves connectivity)
            result.add(new Segment3D(seg.a, displacedMid, seg.level));
            result.add(new Segment3D(displacedMid, seg.b, seg.level));
        }

        return result;
    }

    /**
     * Legacy displacement (modifies list in place, for chain-ordered segments).
     */
    private void displaceSegments(List<Segment3D> segments, double displacementFactor, Cell cell) {
        if (displacementFactor < MathUtils.EPSILON) return;

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) continue;

            Vec2D perp = dir.rotateCCW90().normalize();

            Random rng = initRandomGenerator((int)(seg.a.x * 100), (int)(seg.a.y * 100), cell.resolution);
            // Displacement proportional to segment length
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor * segLength;

            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            segments.set(i, new Segment3D(seg.a, displacedMid, seg.level));

            if (i + 1 < segments.size()) {
                Segment3D next = segments.get(i + 1);
                if (next.a.equals(seg.b)) {
                    segments.set(i + 1, new Segment3D(displacedMid, next.b, next.level));
                }
            }
        }
    }

    private double getConnectDistance(int level) {
        if (connectDistance > 0) {
            return connectDistance / level;
        }
        double cellSize = 1.0 / Math.pow(2, level - 1);
        return cellSize * connectDistanceFactor;
    }

    private List<Segment3D> generateSubSegments(Point3D[][] points, List<Segment3D> parentSegments,
                                                 double minSlope, int level) {
        List<Segment3D> subSegments = new ArrayList<>();
        int size = points.length;
        double maxDistance = getConnectDistance(level);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Point3D point = points[i][j];

                NearestSegmentResult nearest = findNearestSegment(point.projectZ(), parentSegments);
                if (nearest == null || nearest.distance > maxDistance) continue;

                double elevationFromControl = evaluateControlFunction(point.x, point.y);
                double elevationWithSlope = nearest.closestPoint.z + minSlope * nearest.distance;
                double elevation = Math.max(elevationWithSlope, elevationFromControl);

                Point3D adjustedPoint = new Point3D(point.x, point.y, elevation);

                if (nearest.distance > MathUtils.EPSILON) {
                    Point3D connectionPoint = new Point3D(
                        nearest.closestPoint2D.x,
                        nearest.closestPoint2D.y,
                        nearest.closestPoint.z
                    );
                    subSegments.add(new Segment3D(adjustedPoint, connectionPoint, level));
                }
            }
        }
        return subSegments;
    }

    private static class NearestSegmentResult {
        final double distance;
        final double weightedDistance;
        final Point2D closestPoint2D;
        final Point3D closestPoint;
        final Segment3D segment;

        NearestSegmentResult(double distance, double weightedDistance, Point2D closestPoint2D,
                            Point3D closestPoint, Segment3D segment) {
            this.distance = distance;
            this.weightedDistance = weightedDistance;
            this.closestPoint2D = closestPoint2D;
            this.closestPoint = closestPoint;
            this.segment = segment;
        }
    }

    /**
     * Find nearest segment - uses parallel streams if enabled and list is large enough.
     */
    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream;
        if (useParallel && segments.size() > parallelThreshold) {
            stream = segments.parallelStream();
        } else {
            stream = segments.stream();
        }

        return stream
            .map(seg -> {
                MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                double weightedDist = result.distance / seg.level;
                return new NearestSegmentResult(
                    result.distance,
                    weightedDist,
                    result.closestPoint,
                    new Point3D(result.closestPoint.x, result.closestPoint.y, z),
                    seg
                );
            })
            .min(Comparator.comparingDouble(r -> r.distance))
            .orElse(null);
    }

    private NearestSegmentResult findNearestSegmentWeighted(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream;
        if (useParallel && segments.size() > parallelThreshold) {
            stream = segments.parallelStream();
        } else {
            stream = segments.stream();
        }

        return stream
            .map(seg -> {
                MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                double weightedDist = result.distance / seg.level;
                return new NearestSegmentResult(
                    result.distance,
                    weightedDist,
                    result.closestPoint,
                    new Point3D(result.closestPoint.x, result.closestPoint.y, z),
                    seg
                );
            })
            .min(Comparator.comparingDouble(r -> r.weightedDistance))
            .orElse(null);
    }

    private double computeResult(double x, double y, List<Segment3D> segments) {
        Point2D point = new Point2D(x, y);

        switch (returnType) {
            case DISTANCE:
                NearestSegmentResult nearestDist = findNearestSegment(point, segments);
                if (nearestDist == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestDist.distance;

            case WEIGHTED:
                NearestSegmentResult nearestWeighted = findNearestSegmentWeighted(point, segments);
                if (nearestWeighted == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestWeighted.weightedDistance;

            case ELEVATION:
            default:
                NearestSegmentResult nearest = findNearestSegment(point, segments);
                if (nearest == null) {
                    return evaluateControlFunction(x, y);
                }
                double baseElevation = nearest.closestPoint.z;
                double slopeContribution = nearest.distance * slope;
                return baseElevation + slopeContribution;
        }
    }

    // ========== Pixel Cache Methods ==========

    /**
     * Get pixel cache statistics as a string for debugging.
     * Returns a summary of cache hits, misses, and hit rate.
     */
    public String getPixelCacheStats() {
        long hits = pixelCacheHits.get();
        long misses = pixelCacheMisses.get();
        long total = hits + misses;
        double hitRate = total > 0 ? (100.0 * hits / total) : 0;
        int cacheSize = pixelCache != null ? pixelCache.size() : 0;
        return String.format("hits=%d, misses=%d, hitRate=%.1f%%, cachedCells=%d",
            hits, misses, hitRate, cacheSize);
    }

    /**
     * Reset pixel cache statistics (useful between benchmark runs).
     */
    public void resetPixelCacheStats() {
        pixelCacheHits.set(0);
        pixelCacheMisses.set(0);
    }

    /**
     * Check if pixel cache is enabled and the return type uses it.
     */
    private boolean usesPixelCache() {
        return cachepixels > 0 && pixelCache != null &&
               (returnType == DendryReturnType.PIXEL_ELEVATION || returnType == DendryReturnType.PIXEL_LEVEL);
    }

    /**
     * Get or create pixel cache entry for a cell.
     * Returns null if caching is disabled.
     */
    private CellPixelData getOrCreatePixelCache(int cellX, int cellY) {
        if (pixelCache == null) return null;

        long key = packKey(cellX, cellY);
        synchronized (pixelCacheLock) {
            CellPixelData data = pixelCache.get(key);
            if (data != null) {
                data.lastAccessTime = System.nanoTime();
                return data;
            }

            // Create new cache entry
            data = new CellPixelData(cellX, cellY, pixelGridSize);

            // Check if we need to evict old entries
            evictIfNeeded(data.getMemorySize());

            pixelCache.put(key, data);
            return data;
        }
    }

    /**
     * Get cached pixel data for a cell (without creating).
     */
    private CellPixelData getCachedPixelData(int cellX, int cellY) {
        if (pixelCache == null) return null;

        long key = packKey(cellX, cellY);
        synchronized (pixelCacheLock) {
            CellPixelData data = pixelCache.get(key);
            if (data != null) {
                data.lastAccessTime = System.nanoTime();
            }
            return data;
        }
    }

    /**
     * Evict oldest cache entries if total memory exceeds limit.
     */
    private void evictIfNeeded(int additionalBytes) {
        if (pixelCache == null) return;

        int totalBytes = additionalBytes;
        for (CellPixelData data : pixelCache.values()) {
            totalBytes += data.getMemorySize();
        }

        while (totalBytes > MAX_PIXEL_CACHE_BYTES && !pixelCache.isEmpty()) {
            // Find oldest entry
            long oldestKey = -1;
            long oldestTime = Long.MAX_VALUE;

            for (Map.Entry<Long, CellPixelData> entry : pixelCache.entrySet()) {
                if (entry.getValue().lastAccessTime < oldestTime) {
                    oldestTime = entry.getValue().lastAccessTime;
                    oldestKey = entry.getKey();
                }
            }

            if (oldestKey != -1) {
                CellPixelData removed = pixelCache.remove(oldestKey);
                if (removed != null) {
                    totalBytes -= removed.getMemorySize();
                }
            } else {
                break;
            }
        }
    }

    /**
     * Convert world coordinates to pixel coordinates within a cell.
     * @param worldX World X coordinate (already divided by gridsize)
     * @param worldY World Y coordinate (already divided by gridsize)
     * @param cellX Cell X index
     * @param cellY Cell Y index
     * @return Pixel coordinates [px, py] or null if out of bounds
     */
    private int[] worldToPixel(double worldX, double worldY, int cellX, int cellY) {
        // Position within cell (0 to 1)
        double localX = worldX - cellX;
        double localY = worldY - cellY;

        // Convert to pixel coordinates
        int px = (int) Math.floor(localX * pixelGridSize);
        int py = (int) Math.floor(localY * pixelGridSize);

        if (px >= 0 && px < pixelGridSize && py >= 0 && py < pixelGridSize) {
            return new int[] { px, py };
        }
        return null;
    }

    /**
     * Sample segments into the pixel cache for a cell.
     * Walks along each segment and marks pixels it crosses.
     */
    private void sampleSegmentsToPixelCache(List<Segment3D> segments, CellPixelData cache) {
        double cellX = cache.cellX;
        double cellY = cache.cellY;

        for (Segment3D seg : segments) {
            // Get segment bounds in cell-local coordinates
            double ax = seg.a.x - cellX;
            double ay = seg.a.y - cellY;
            double bx = seg.b.x - cellX;
            double by = seg.b.y - cellY;

            // Skip if segment is entirely outside the cell
            if ((ax < 0 && bx < 0) || (ax >= 1 && bx >= 1) ||
                (ay < 0 && by < 0) || (ay >= 1 && by >= 1)) {
                continue;
            }

            // Sample along the segment at pixel resolution
            double segLength = seg.length();
            double pixelSize = 1.0 / pixelGridSize;
            int numSamples = Math.max(2, (int) Math.ceil(segLength / pixelSize * 2));

            for (int i = 0; i <= numSamples; i++) {
                double t = (double) i / numSamples;
                Point3D pt = seg.lerp(t);

                double localX = pt.x - cellX;
                double localY = pt.y - cellY;

                // Check if point is within cell
                if (localX >= 0 && localX < 1 && localY >= 0 && localY < 1) {
                    int px = (int) (localX * pixelGridSize);
                    int py = (int) (localY * pixelGridSize);

                    if (px >= 0 && px < pixelGridSize && py >= 0 && py < pixelGridSize) {
                        float elevation = (float) pt.z;
                        byte level = (byte) seg.level;
                        cache.setPixel(px, py, elevation, level);
                    }
                }
            }
        }
    }

    /**
     * Lookup pixel data for a world coordinate.
     * Returns the elevation or level based on return type.
     * Returns NaN if no pixel data is available.
     */
    private double lookupPixelValue(double worldX, double worldY) {
        // Get cell coordinates
        int cellX = MathUtils.floor(worldX);
        int cellY = MathUtils.floor(worldY);

        CellPixelData cache = getCachedPixelData(cellX, cellY);
        if (cache == null) {
            return Double.NaN;
        }

        int[] pixel = worldToPixel(worldX, worldY, cellX, cellY);
        if (pixel == null) {
            return Double.NaN;
        }

        PixelData data = cache.getPixel(pixel[0], pixel[1]);
        if (data == null) {
            return Double.NaN;
        }

        if (returnType == DendryReturnType.PIXEL_ELEVATION) {
            // Add slope contribution based on distance to nearest pixel center
            double pixelCenterX = cellX + (data.xOffset + 0.5) / pixelGridSize;
            double pixelCenterY = cellY + (data.yOffset + 0.5) / pixelGridSize;
            double dist = Math.sqrt((worldX - pixelCenterX) * (worldX - pixelCenterX) +
                                    (worldY - pixelCenterY) * (worldY - pixelCenterY));
            return data.elevation + dist * slope * gridsize;
        } else {
            // PIXEL_LEVEL
            return data.level;
        }
    }

    /**
     * Evaluate using pixel cache - checks cache first, computes if needed.
     */
    private double evaluateWithPixelCache(long seed, double x, double y) {
        // x, y are already in normalized coordinates (divided by gridsize)
        int cellX = MathUtils.floor(x);
        int cellY = MathUtils.floor(y);

        // Check if cell is already cached and populated
        CellPixelData cache = getCachedPixelData(cellX, cellY);
        if (cache != null && cache.populated) {
            // CACHE HIT: Cell is already computed - look up pixel value directly
            pixelCacheHits.incrementAndGet();
            double value = lookupPixelValue(x, y);
            // If pixel has data, return it; if empty (NaN), return -2
            // - 1 gives delta from other values.
            // This is the FIX: we don't recompute just because a pixel is empty
            return Double.isNaN(value) ? -1 : (value+1);
        }

        // CACHE MISS: Cell not computed yet - create/get cache entry and compute
        pixelCacheMisses.incrementAndGet();
        cache = getOrCreatePixelCache(cellX, cellY);

        // Compute all segments for this cell
        List<Segment3D> allSegments = computeAllSegmentsForCell(seed, cellX, cellY);

        // Sample segments into pixel cache
        if (cache != null) {
            sampleSegmentsToPixelCache(allSegments, cache);
            cache.populated = true;  // Mark as populated AFTER sampling
        }

        // Lookup the value
        double value = lookupPixelValue(x, y);
        // If pixel has data, return it; if empty (NaN), return 0
        return Double.isNaN(value) ? 0 : value;
    }

    /**
     * Compute all segments for a specific cell (used for pixel cache population).
     */
    private List<Segment3D> computeAllSegmentsForCell(long seed, int cellX, int cellY) {
        // Create a cell object for the target cell
        Cell cell1 = new Cell(cellX, cellY, 1);

        // Displacement factors
        double displacementLevel0 = delta * 2.0;
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slope constraints
        double minSlopeLevel1 = 0.5;
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 0
        List<Segment3D> segments0Base = generateLevel0Network(cell1);
        Map<Long, NodeTangentInfo> tangentMap0 = computeNodeTangents(segments0Base);
        List<Segment3D> segments0Pruned = pruneToQueryRegion(segments0Base, cell1);
        int level0Subdivisions = Math.max(2, defaultBranches);
        segments0Pruned = subdivideSegments(segments0Pruned, level0Subdivisions, 0, tangentMap0);
        segments0Pruned = displaceSegmentsWithSplit(segments0Pruned, displacementLevel0, 0);

        if (resolution == 0) {
            return segments0Pruned;
        }

        // Level 1
        List<Segment3D> segments1Base = generateLevel1SegmentsCompact(cell1, segments0Pruned, minSlopeLevel1);
        Map<Long, NodeTangentInfo> tangentMap1 = computeNodeTangents(segments1Base);
        int branchCount = getBranchCountForCell(cell1);
        List<Segment3D> segments1 = subdivideSegments(segments1Base, branchCount, 1, tangentMap1);
        segments1 = displaceSegmentsWithSplit(segments1, displacementLevel1, 1);

        List<Segment3D> allSegments = new ArrayList<>(segments0Pruned);
        allSegments.addAll(segments1);

        if (resolution == 1) {
            return allSegments;
        }

        // Level 2+
        Cell cell2 = new Cell(cellX * 2, cellY * 2, 2);
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);
        List<Segment3D> segments2 = generateSubSegments(points2, allSegments, minSlopeLevel2, 2);
        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            allSegments.addAll(segments2);
            return allSegments;
        }

        Cell cell3 = new Cell(cellX * 4, cellY * 4, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);
        allSegments.addAll(segments2);
        List<Segment3D> segments3 = generateSubSegments(points3, allSegments, minSlopeLevel3, 3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allSegments.addAll(segments3);
            return allSegments;
        }

        Cell cell4 = new Cell(cellX * 8, cellY * 8, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allSegments, minSlopeLevel4, 4);

        if (resolution == 4) {
            allSegments.addAll(segments4);
            return allSegments;
        }

        Cell cell5 = new Cell(cellX * 16, cellY * 16, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allSegments, minSlopeLevel5, 5);
        allSegments.addAll(segments5);

        return allSegments;
    }
}

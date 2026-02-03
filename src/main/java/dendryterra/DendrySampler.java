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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Optimized with configurable caching, parallel processing, and debug timing.
 */
public class DendrySampler implements Sampler {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendrySampler.class);

    /**
     * Debug parameter to return segments at different processing stages.
     * Change this value and recompile to debug segment generation.
     *
     * Values:
     *   0  - Normal operation (default)
     *   5  - Return stars as 0-length segments for FIRST constellation only
     *   6  - Return stars as 0-length segments for ALL constellations
     *  10  - Return segments for FIRST constellation only, before stitching
     *  15  - Return segments for ALL constellations, before stitching, only tree
     *  20  - Return segments for ALL constellations, before stitching
     *  30  - Return segments for all constellations INCLUDING stitching
     *  40  - Return segments after Phase A of CleanAndNetworkPoints (initial connections)
     *  50  - Return segments after Phase B of CleanAndNetworkPoints (chain connections)
     */
    private static final int SEGMENT_DEBUGGING = 0;

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

    // Constellation parameters
    private final int ConstellationScale;
    private final ConstellationShape constellationShape;

    // Hard-coded star/point spacing parameters (in cell units)
    // These are used at multiple levels with different cell sizes
    private static final double MERGE_POINT_SPACING = 2.0 / 3.0;  // Stars closer than this get merged
    private static final double MAX_POINT_SEGMENT_DISTANCE = Math.sqrt(8) + 1.0 / 3.0;  // Max distance between adjacent stars after merging

    // Divisions per level configuration
    // Each value indicates how many divisions occur at that level relative to the previous
    // Example: [3, 2, 2, 2, 2] means level 0 has 3 divisions, levels 1+ have 2 each
    private static final int[] DIVISIONS_PER_LEVEL = {1,3, 2, 2, 2, 2};

    // Pre-computed cumulative points per cell per level (calculated in static block)
    // [3, 6, 12, 24, 48] for DIVISIONS_PER_LEVEL = [3, 2, 2, 2, 2]
    private static final int[] POINTS_PER_CELL;

    static {
        POINTS_PER_CELL = new int[DIVISIONS_PER_LEVEL.length];
        POINTS_PER_CELL[0] = DIVISIONS_PER_LEVEL[0];
        for (int i = 1; i < DIVISIONS_PER_LEVEL.length; i++) {
            POINTS_PER_CELL[i] = POINTS_PER_CELL[i - 1] * DIVISIONS_PER_LEVEL[i];
        }
    }

    // Slope calculation parameters for neighbor selection
    // DistanceFalloffPower: Use dist^power in denominator to prefer tighter connections
    private static final double DISTANCE_FALLOFF_POWER = 2.0;
    // BranchEncouragementFactor: Multiply slope by this when neighbor has 2+ connections
    // to encourage attaching to existing flows
    private static final double BRANCH_ENCOURAGEMENT_FACTOR = 2.0;
    // TangentMagnitudeScale: Scale factor for tangent magnitude in Hermite spline interpolation
    // Higher values create more pronounced curvature
    private static final double TANGENT_MAGNITUDE_SCALE = 10.0;

    /**
     * Use B-spline (cubic Hermite) interpolation for pixel sampling in PIXEL_DEBUG mode.
     * When true, segments with tangent information will be sampled along the curved spline.
     * When false, uses linear interpolation between endpoints.
     */
    private static final boolean USE_BSPLINE_PIXEL_SAMPLING = true;

    /**
     * Error if any constellation segment is returned with undefined tangents.
     * When true, throws an error if any segment in the constellation has null tangentSrt or tangentEnd.
     * Useful for debugging tangent computation issues.
     */
    private static final boolean ERROR_ON_UNDEFINED_TANGENT = true;

    // Star sampling grid size (9x9 grid per cell)
    private static final int STAR_SAMPLE_GRID_SIZE = 3;
    // Star sample boundary margin (fraction of cell size)
    private static final double STAR_SAMPLE_BOUNDARY = 0.03;

    // Spline tangent parameters
    private final double tangentAngle;    // Max angle deviation (radians)
    private final double tangentStrength; // Tangent length as fraction of segment length

    // Slope-based tangent alignment parameters
    private final double slopeWhenStraight; // Slope threshold for full tangent alignment (0-1)
    private final double lowestSlopeCutoff; // Minimum slope cutoff for point rejection

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
        final byte pointType;  // -1 = segment line only, 0+ = PointType enum values (ORIGINAL, TRUNK, KNOT, LEAF)

        PixelData(int xOffset, int yOffset, float elevation, byte level) {
            this(xOffset, yOffset, elevation, level, (byte) 0);
        }

        PixelData(int xOffset, int yOffset, float elevation, byte level, byte pointType) {
            this.xOffset = xOffset;
            this.yOffset = yOffset;
            this.elevation = elevation;
            this.level = level;
            this.pointType = pointType;
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
            setPixel(px, py, elevation, level, (byte) 0);
        }

        void setPixel(int px, int py, float elevation, byte level, byte pointType) {
            if (px >= 0 && px < gridSize && py >= 0 && py < gridSize) {
                PixelData existing = pixels[py][px];
                if (existing == null) {
                    // Empty pixel - set it
                    pixels[py][px] = new PixelData(px, py, elevation, level, pointType);
                } else if (pointType > existing.pointType) {
                    // Higher priority point type (original > subdivision > segment)
                    pixels[py][px] = new PixelData(px, py, elevation, level, pointType);
                } else if (pointType == existing.pointType && level < existing.level) {
                    // Same point type but lower (more significant) level
                    pixels[py][px] = new PixelData(px, py, elevation, level, pointType);
                }
                // Otherwise, don't overwrite - existing data takes priority
            }
        }

        /**
         * Mark a point (original or subdivision) in the cache with a radius.
         * This marks a circle of pixels around the point for debug visualization.
         * @param px center pixel X
         * @param py center pixel Y
         * @param elevation elevation value
         * @param level resolution level
         * @param pointType PointType enum value (ORIGINAL=0, TRUNK=1, KNOT=2, LEAF=3)
         * @param radius number of pixels around center to mark
         */
        void markPointWithRadius(int px, int py, float elevation, byte level, byte pointType, int radius) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx * dx + dy * dy <= radius * radius) {  // Circle, not square
                        setPixel(px + dx, py + dy, elevation, level, pointType);
                    }
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
                         int ConstellationScale, ConstellationShape constellationShape,
                         double tangentAngle, double tangentStrength,
                         double cachepixels,
                         double slopeWhenStraight, double lowestSlopeCutoff) {
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
        this.ConstellationScale = ConstellationScale;
        this.constellationShape = constellationShape;
        this.tangentAngle = tangentAngle;
        this.tangentStrength = tangentStrength;
        this.cachepixels = cachepixels;
        this.slopeWhenStraight = slopeWhenStraight;
        this.lowestSlopeCutoff = lowestSlopeCutoff;

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
        Cell cell1 = getCell(x, y, 1);
        List<Segment3D> allSegments = generateAllSegments(cell1, x, y);
        return computeResult(x, y, allSegments);
    }

    /**
     * Generate all segments up to the configured resolution level.
     * This is the core segment generation logic shared by evaluate() and computeAllSegmentsForCell().
     *
     * @param cell1 The level 1 cell for the query
     * @param queryX Query X coordinate in normalized space (for higher resolution cell lookup)
     * @param queryY Query Y coordinate in normalized space (for higher resolution cell lookup)
     * @return List of all generated segments
     */
    private List<Segment3D> generateAllSegments(Cell cell1, double queryX, double queryY) {
        // Displacement factor for level 1 (levels 2+ handled in CleanAndNetworkPoints)
        double displacementLevel1 = delta;

        // Asterism (Level 0): Generate and process
        List<Segment3D> asterismBase = generateAsterism(cell1);
        List<Segment3D> asterismPruned = asterismBase;

        if (resolution == 0) {
            return asterismPruned;
        }
        List<Segment3D> allSegments = new ArrayList<>(asterismPruned);

        // Level 1+: Higher resolution refinement using loop
        // Each level generates points for the query cell and connects them using CleanAndNetworkPoints
        List<Segment3D> previousLevelSegments = new ArrayList<>(allSegments);

        for (int level = 1; level <= resolution; level++) {
            // Get the cell at this level's resolution
            int cellResolution = getCellResolutionForLevel(level);
            Cell levelCell = getCell(queryX, queryY, cellResolution);

            // Generate points for this cell at this level's density
            List<Point3D> levelPoints = generatePointsForCellAtLevel(levelCell.x, levelCell.y, level);

            // Use CleanAndNetworkPoints to create properly connected segments
            // This handles merging, cleaning, tangent computation, subdivision, and displacement
            List<Segment3D> levelSegments = CleanAndNetworkPoints(
                levelCell.x, levelCell.y, level, levelPoints, previousLevelSegments);

            if (!levelSegments.isEmpty()) {
                allSegments.addAll(levelSegments);
                // Update previous level segments for next iteration
                previousLevelSegments = new ArrayList<>(allSegments);
            }
        }

        return allSegments;
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

    /**
     * Get the grid spacing for a given level based on DIVISIONS_PER_LEVEL configuration.
     * Grid spacing = 1.0 / pointsPerCell[level]
     *
     * @param level The level (0-indexed)
     * @return Grid spacing for this level
     */
    private double getGridSpacingForLevel(int level) {
        if (level < 0) return 1.0;
        if (level >= POINTS_PER_CELL.length) {
            // Extrapolate for levels beyond configuration
            int lastIdx = POINTS_PER_CELL.length - 1;
            int extraLevels = level - lastIdx;
            int lastDivision = DIVISIONS_PER_LEVEL[lastIdx];
            return 1.0 / (POINTS_PER_CELL[lastIdx] * Math.pow(lastDivision, extraLevels));
        }
        return 1.0 / POINTS_PER_CELL[level];
    }

    /**
     * Get the number of point divisions for a given level.
     *
     * @param level The level (0-indexed)
     * @return Number of points per cell axis for this level
     */
    private int getPointsPerCellForLevel(int level) {
        if (level < 0) return 1;
        if (level >= POINTS_PER_CELL.length) {
            // Extrapolate for levels beyond configuration
            int lastIdx = POINTS_PER_CELL.length - 1;
            int extraLevels = level - lastIdx;
            int lastDivision = DIVISIONS_PER_LEVEL[lastIdx];
            return (int) (POINTS_PER_CELL[lastIdx] * Math.pow(lastDivision, extraLevels));
        }
        return POINTS_PER_CELL[level];
    }

    /**
     * Get the cell resolution multiplier for a given level.
     * This is used for getCell() to find the correct sub-cell.
     *
     * @param level The level (0-indexed, where 0 = level 1 cells)
     * @return Resolution multiplier (1 for level 0, 2 for level 1, etc.)
     */
    private int getCellResolutionForLevel(int level) {
        if (level <= 0) return 1;
        return (int) Math.pow(2, level - 1);
    }

    private Point3D[][] generateNeighboringPoints3D(Cell cell, int size) {
        Point3D[][] points = new Point3D[size][size];
        int half = size / 2;

        // First pass: create all points with position and elevation
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

        // Second pass: compute slopes for each point using 2 closest neighbors at least 30 degrees apart
        Point3D[][] pointsWithSlopes = new Point3D[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                pointsWithSlopes[i][j] = computeSlopeForPoint(points, i, j, size);
            }
        }
        return pointsWithSlopes;
    }

    /**
     * Generate points for a single cell at a given level using DIVISIONS_PER_LEVEL configuration.
     * Creates a grid of points within the cell based on getPointsPerCellForLevel().
     *
     * @param cellX Cell X coordinate in level 1 space
     * @param cellY Cell Y coordinate in level 1 space
     * @param level The resolution level (determines point density)
     * @return List of 3D points within the cell
     */
    private List<Point3D> generatePointsForCellAtLevel(int cellX, int cellY, int level) {
        List<Point3D> points = new ArrayList<>();
        int pointsPerAxis = getPointsPerCellForLevel(level);
        double gridSpacing = getGridSpacingForLevel(level);

        // Generate a grid of points within the cell [cellX, cellX+1) x [cellY, cellY+1)
        for (int i = 0; i < pointsPerAxis; i++) {
            for (int j = 0; j < pointsPerAxis; j++) {
                // Position point at center of each sub-cell with deterministic jitter
                double baseX = cellX + (j + 0.5) * gridSpacing;
                double baseY = cellY + (i + 0.5) * gridSpacing;

                // Add deterministic jitter based on position and level
                Random rng = initRandomGenerator((int)(baseX * 10000), (int)(baseY * 10000), level);
                double jitterX = (rng.nextDouble() - 0.5) * gridSpacing * 0.8;
                double jitterY = (rng.nextDouble() - 0.5) * gridSpacing * 0.8;

                double px = baseX + jitterX;
                double py = baseY + jitterY;
                double elevation = evaluateControlFunction(px, py);

                points.add(new Point3D(px, py, elevation));
            }
        }

        return points;
    }

    /** Minimum angle (in radians) between two neighbors for slope estimation. */
    private static final double MIN_NEIGHBOR_ANGLE_RAD = Math.toRadians(30);

    /**
     * Compute slope for a point using its 2 closest neighbors that are at least 30 degrees apart.
     */
    private Point3D computeSlopeForPoint(Point3D[][] points, int i, int j, int size) {
        Point3D center = points[i][j];

        // Collect all valid neighbors with their distances and angles
        List<NeighborInfo> neighbors = new ArrayList<>();
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                if (di == 0 && dj == 0) continue;
                int ni = i + di;
                int nj = j + dj;
                if (ni < 0 || ni >= size || nj < 0 || nj >= size) continue;

                Point3D neighbor = points[ni][nj];
                double dx = neighbor.x - center.x;
                double dy = neighbor.y - center.y;
                double dz = neighbor.z - center.z;
                double dist = Math.sqrt(dx * dx + dy * dy);
                double angle = Math.atan2(dy, dx);
                neighbors.add(new NeighborInfo(dx, dy, dz, dist, angle));
            }
        }

        // Sort by distance (closest first)
        neighbors.sort((a, b) -> Double.compare(a.dist, b.dist));

        // Find closest 2 neighbors that are at least 30 degrees apart
        NeighborInfo n1 = null;
        NeighborInfo n2 = null;
        for (int k = 0; k < neighbors.size() && n2 == null; k++) {
            NeighborInfo candidate = neighbors.get(k);
            if (n1 == null) {
                n1 = candidate;
            } else {
                double angleDiff = Math.abs(candidate.angle - n1.angle);
                // Normalize to [0, PI]
                if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;
                if (angleDiff >= MIN_NEIGHBOR_ANGLE_RAD) {
                    n2 = candidate;
                }
            }
        }

        // If we couldn't find 2 suitable neighbors, return point without slope
        if (n1 == null || n2 == null) {
            return center;
        }

        // Solve 2x2 linear system to find slopeX and slopeY:
        // slopeX * dx1 + slopeY * dy1 = dz1
        // slopeX * dx2 + slopeY * dy2 = dz2
        double det = n1.dx * n2.dy - n2.dx * n1.dy;
        if (Math.abs(det) < MathUtils.EPSILON) {
            // Neighbors are collinear, can't solve
            return center;
        }

        double slopeX = (n1.dz * n2.dy - n2.dz * n1.dy) / det;
        double slopeY = (n1.dx * n2.dz - n2.dx * n1.dz) / det;

        return center.withSlopes(slopeX, slopeY);
    }

    private static class NeighborInfo {
        final double dx, dy, dz;
        final double dist;
        final double angle;

        NeighborInfo(double dx, double dy, double dz, double dist, double angle) {
            this.dx = dx;
            this.dy = dy;
            this.dz = dz;
            this.dist = dist;
            this.angle = angle;
        }
    }

    // ========== Asterism (Level 0) Network Generation ==========
    //
    // Terminology:
    // - Constellation: A tileable region (square/hexagon/rhombus) containing stars
    // - Star: A key point within a constellation, found by sampling lowest elevation
    // - Asterism: The network of segments connecting stars across constellations
    // - AsterismSegment: A segment in the asterism network (level 0 segment)
    //
    // Constellation scaling:
    // - =1 means the largest inscribed square is 3 gridspaces wide
    // - This ensures only 4 closest constellations need to be solved for any query

    /**
     * Get the size of a constellation in grid units.
     * ConstellationScale=1 gives an inscribed square of 3 gridspaces.
     */
    private double getConstellationSize() {
        // For SQUARE shape: constellation size = 3 * ConstellationScale
        // This ensures inscribed square (without tilting) is 3 * ConstellationScale wide
        return 3.0 * ConstellationScale;
    }

    /**
     * Get the center of the constellation containing the given world position.
     * This is the fundamental operation - no indices, just finds the nearest grid-aligned center.
     */
    private Point2D getConstellationCenterForPoint(double worldX, double worldY) {
        double size = getConstellationSize();

        switch (constellationShape) {
            case HEXAGON: {
                // Hexagonal grid: rows spaced by size * sqrt(3)/2, odd rows offset by size/2
                double rowHeight = size * 0.866;  // sqrt(3)/2
                int row = (int) Math.floor(worldY / rowHeight);
                double rowOffsetX = (row % 2 == 0) ? 0 : size * 0.5;
                int col = (int) Math.floor((worldX - rowOffsetX) / size);
                // Center is at middle of the cell
                double centerX = (col + 0.5) * size + rowOffsetX;
                double centerY = (row + 0.5) * rowHeight;
                return new Point2D(centerX, centerY);
            }
            case RHOMBUS: {
                // Rhombus grid: rows spaced by size * sqrt(2)/2, cumulative offset
                double rowHeight = size * 0.707;  // sqrt(2)/2
                int row = (int) Math.floor(worldY / rowHeight);
                double rowOffsetX = row * size * 0.5;
                int col = (int) Math.floor((worldX - rowOffsetX) / size);
                double centerX = (col + 0.5) * size + rowOffsetX;
                double centerY = (row + 0.5) * rowHeight;
                return new Point2D(centerX, centerY);
            }
            case SQUARE:
            default: {
                // Square grid: simple aligned grid
                int col = (int) Math.floor(worldX / size);
                int row = (int) Math.floor(worldY / size);
                double centerX = (col + 0.5) * size;
                double centerY = (row + 0.5) * size;
                return new Point2D(centerX, centerY);
            }
        }
    }

    /**
     * Get the shape-specific offsets to neighboring constellation centers.
     * Returns array of [dx, dy] pairs to add to a center to get neighbor centers.
     */
    private double[][] getNeighborOffsets() {
        double size = getConstellationSize();

        switch (constellationShape) {
            case HEXAGON: {
                // Hexagon has 6 neighbors at specific offsets
                double rowHeight = size * 0.866;
                return new double[][] {
                    // Same row neighbors
                    { -size, 0 }, { size, 0 },
                    // Upper row neighbors (offset depends on current row, so include both patterns)
                    { -size * 0.5, rowHeight }, { size * 0.5, rowHeight },
                    // Lower row neighbors
                    { -size * 0.5, -rowHeight }, { size * 0.5, -rowHeight },
                    // Extended ring for safety
                    { -size * 1.5, rowHeight }, { size * 1.5, rowHeight },
                    { -size * 1.5, -rowHeight }, { size * 1.5, -rowHeight },
                    { 0, rowHeight * 2 }, { 0, -rowHeight * 2 },
                    { -size, rowHeight * 2 }, { size, rowHeight * 2 },
                    { -size, -rowHeight * 2 }, { size, -rowHeight * 2 }
                };
            }
            case RHOMBUS: {
                // Rhombus neighbors with diagonal offsets
                double rowHeight = size * 0.707;
                return new double[][] {
                    // Same row neighbors
                    { -size, 0 }, { size, 0 },
                    // Adjacent rows with offset
                    { -size * 0.5, rowHeight }, { size * 0.5, rowHeight },
                    { -size * 0.5, -rowHeight }, { size * 0.5, -rowHeight },
                    // Extended ring
                    { -size * 1.5, rowHeight }, { size * 1.5, rowHeight },
                    { -size * 1.5, -rowHeight }, { size * 1.5, -rowHeight },
                    { 0, rowHeight * 2 }, { 0, -rowHeight * 2 }
                };
            }
            case SQUARE:
            default: {
                // Square has 8 neighbors (including diagonals)
                return new double[][] {
                    { -size, -size }, { 0, -size }, { size, -size },
                    { -size, 0 },                    { size, 0 },
                    { -size, size },  { 0, size },  { size, size }
                };
            }
        }
    }

    /**
     * Create ConstellationInfo from a center position.
     * Computes the circumscribing level 1 cells based on constellation size.
     */
    private ConstellationInfo getConstellationInfoFromCenter(double centerX, double centerY) {
        double size = getConstellationSize();

        // Calculate bounding box of level 1 cells that circumscribe this constellation
        // Add margin of 1 cell on each side to ensure full coverage
        int startCellX = (int) Math.floor(centerX - size / 2.0 - 1);
        int startCellY = (int) Math.floor(centerY - size / 2.0 - 1);
        int endCellX = (int) Math.ceil(centerX + size / 2.0 + 1);
        int endCellY = (int) Math.ceil(centerY + size / 2.0 + 1);

        return new ConstellationInfo(
            centerX, centerY,
            startCellX, startCellY,
            endCellX - startCellX, endCellY - startCellY
        );
    }

    /**
     * Find the closest constellations to the query cell center.
     * Works purely with constellation centers - no indices.
     * Returns ConstellationInfo objects with center coordinates and circumscribing cell info.
     */
    private List<ConstellationInfo> findClosestConstellations(Cell queryCell1) {
        final double avgX;
        final double avgY;

        double sumX = 0;
        double sumY = 0;

        // Query cell center in world coordinates (level 1 cell coordinates)
        double queryCenterX = queryCell1.x + 0.5;
        double queryCenterY = queryCell1.y + 0.5;

        // Get center of constellation containing the query point
        Point2D baseCenter = getConstellationCenterForPoint(queryCenterX, queryCenterY);

        // Get neighbor offsets for this shape
        double[][] offsets = getNeighborOffsets();

        // Collect candidate constellations: base constellation plus all neighbors
        List<ConstellationInfo> candidates = new ArrayList<>();

        // Add the base constellation
        candidates.add(getConstellationInfoFromCenter(baseCenter.x, baseCenter.y));

        // Add all neighbors by applying offsets to base center
        for (double[] offset : offsets) {
            double neighborX = baseCenter.x + offset[0];
            double neighborY = baseCenter.y + offset[1];
            candidates.add(getConstellationInfoFromCenter(neighborX, neighborY));
        }

        // Sort by distance from query center to constellation center
        candidates.sort((a, b) -> {
            double distA = (a.centerX - queryCenterX) * (a.centerX - queryCenterX) +
                           (a.centerY - queryCenterY) * (a.centerY - queryCenterY);
            double distB = (b.centerX - queryCenterX) * (b.centerX - queryCenterX) +
                           (b.centerY - queryCenterY) * (b.centerY - queryCenterY);
            return Double.compare(distA, distB);
        });

        // Take 6 closest
        List<ConstellationInfo> closest = new ArrayList<>(candidates.subList(0, Math.min(6, candidates.size())));
        
        // Get center coordinate between the selected constellations
        for (ConstellationInfo constInfo : closest) {
            sumX += constInfo.centerX;
            sumY += constInfo.centerY;
        }

        avgX = sumX / closest.size();
        avgY = sumY / closest.size();

        // Sort in clockwise order around query center to avoid crossing stitches
        //closest.sort((a, b) -> {
        //    double angleA = Math.atan2(a.centerY - avgY, a.centerX - avgX);
        //    double angleB = Math.atan2(b.centerY - avgY, b.centerX - avgX);
        //    return Double.compare(angleA, angleB);
        //});

        return closest;
    }

    /**
     * @deprecated Use findClosestConstellations instead
     */
    private List<long[]> findFourClosestConstellations(Cell queryCell1) {
        List<ConstellationInfo> infos = findClosestConstellations(queryCell1);
        List<long[]> result = new ArrayList<>();
        for (ConstellationInfo info : infos) {
            // Use quantized center coordinates for backwards compatibility
            int qx = (int) Math.round(info.centerX * 100);
            int qy = (int) Math.round(info.centerY * 100);
            result.add(new long[] { qx, qy, 0 });
        }
        return result;
    }

    /**
     * Generate the asterism (level 0 network) for the query region.
     * Computes the 4 closest constellations, generates stars within each,
     * merges close stars, builds networks, and stitches them together.
     */
    private List<Segment3D> generateAsterism(Cell queryCell1) {
        // Find the 4 closest constellations to the query cell
        List<ConstellationInfo> closestConstellations = findClosestConstellations(queryCell1);

        // Generate asterism segments for each constellation
        Map<Long, List<Segment3D>> constellationSegments = new HashMap<>();
        Map<Long, List<Point3D>> constellationStars = new HashMap<>();

        // Initialize star segments for debugging (needs to be outside if block for scope)
        List<Segment3D> starSegments = new ArrayList<>();

        boolean firstConstellation = true;
        for (ConstellationInfo constInfo : closestConstellations) {
            long constKey = constInfo.getKey();

            // Generate stars for this constellation (with 9x9 sampling, merging, etc.)
            List<Point3D> stars = generateConstellationStars(constInfo);

            // DEBUG: Collect stars as zero-length segments for visualization
            if (SEGMENT_DEBUGGING == 5 || SEGMENT_DEBUGGING == 6) {
                for (Point3D star : stars) {
                    starSegments.add(new Segment3D(star, star, 0));
                }
            }
            if (SEGMENT_DEBUGGING == 5) {
                LOGGER.info("SEGMENT_DEBUGGING=5: Returning stars only for first constellation ({})", stars.size());
                return starSegments;
            }
            // Build network within constellation using CleanAndNetworkPoints
            // Use quantized center coordinates for cell reference
            int cellRefX = (int) Math.floor(constInfo.centerX);
            int cellRefY = (int) Math.floor(constInfo.centerY);
            List<Segment3D> segments = CleanAndNetworkPoints(cellRefX, cellRefY, 0, stars);

            constellationStars.put(constKey, stars);
            constellationSegments.put(constKey, segments);

            // DEBUG: Return after first constellation only
            if (SEGMENT_DEBUGGING == 10 && firstConstellation) {
                LOGGER.info("SEGMENT_DEBUGGING=10: Returning first constellation segments only ({} segments)", segments.size());
                return segments;
            }
            firstConstellation = false;
        }

        // DEBUG: Return all constellation segments before stitching
        if ((SEGMENT_DEBUGGING == 20)||(SEGMENT_DEBUGGING == 15)) {
            List<Segment3D> allSegments = new ArrayList<>();
            for (List<Segment3D> segs : constellationSegments.values()) {
                allSegments.addAll(segs);
            }
            LOGGER.info("SEGMENT_DEBUGGING=20or15: Returning all constellation segments before stitching ({} segments)", allSegments.size());
            return allSegments;
        }
        if (SEGMENT_DEBUGGING == 6) {
            LOGGER.info("SEGMENT_DEBUGGING=6: Returning all constellation stars: ({} stars)", starSegments.size());
            return starSegments;
        }

        // Stitch adjacent constellations together at their boundaries
        // Uses segment endpoints only (not stars) for valid connection points
        List<Segment3D> stitchSegments = stitchConstellations(closestConstellations, constellationSegments);

        // Combine all asterism segments
        List<Segment3D> allSegments = new ArrayList<>();
        for (List<Segment3D> segs : constellationSegments.values()) {
            allSegments.addAll(segs);
        }
        allSegments.addAll(stitchSegments);

        // Now shift all elevations to 0 (after stitching decisions are made)
        allSegments = shiftElevationsToZero(allSegments);

        // DEBUG: Return all segments including stitching
        if (SEGMENT_DEBUGGING == 30) {
            LOGGER.info("SEGMENT_DEBUGGING=30: Returning all segments including stitching ({} segments, {} stitch)", allSegments.size(), stitchSegments.size());
            return allSegments;
        }

        return allSegments;
    }

    /**
     * Generate stars for a constellation using ConstellationInfo.
     * Uses startCell and cellCount to iterate over circumscribing cells directly.
     * 1. Iterate over level 1 cells defined by ConstellationInfo's startCell and cellCount
     * 2. Sample 9x9 grid in each overlapping cell to find candidate stars
     * 3. Remove stars outside constellation boundary (or within 1/2 merge spacing from boundary)
     * 4. Merge stars that are closer than MERGE_POINT_SPACING
     */
    private List<Point3D> generateConstellationStars(ConstellationInfo info) {
        Point2D center = info.getCenter();
        double size = getConstellationSize();
        double halfMergeSpacing = MERGE_POINT_SPACING / 2.0;

        // Step 1: Use ConstellationInfo's startCell and cellCount to iterate over cells
        // Debug logging for constellation parameters
        if (SEGMENT_DEBUGGING > 0) {
            LOGGER.info("Constellation: center=({}, {}), size={}, ConstellationScale={}, cellRange=[{},{} +{}x{}]",
                String.format("%.1f", center.x), String.format("%.1f", center.y),
                String.format("%.1f", size), ConstellationScale,
                info.startCellX, info.startCellY, info.cellCountX, info.cellCountY);
        }

        // Step 2: Sample grid in each cell to find candidate stars
        List<Point3D> draftedStars = new ArrayList<>();
        for (int dy = 0; dy < info.cellCountY; dy++) {
            for (int dx = 0; dx < info.cellCountX; dx++) {
                int cellX = info.startCellX + dx;
                int cellY = info.startCellY + dy;

                // Check if cell overlaps with constellation before sampling
                Point2D cellCenter = new Point2D(cellX + 0.5, cellY + 0.5);
                if (cellOverlapsConstellation(cellCenter, center, size)) {
                    Point3D star = findStarInCelllLowestGrid(cellX, cellY);
                    if (star != null) {
                        draftedStars.add(star);
                    }
                }
            }
        }

        // Step 3: Remove stars outside constellation boundary (exact shape boundaries)
        List<Point3D> boundedStars = new ArrayList<>();
        for (Point3D star : draftedStars) {
            // Keep star if it's inside the exact constellation boundary
            if (isInsideConstellationBoundary(star.projectZ(), center, size)) {
                boundedStars.add(star);
            }
        }

        // Step 4: Merge stars that are closer than MERGE_POINT_SPACING
        List<Point3D> mergedStars = mergeCloseStars(boundedStars);

        // Debug logging for star counts
        if (SEGMENT_DEBUGGING > 0) {
            // Calculate bounding box of drafted stars
            double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
            double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
            for (Point3D star : draftedStars) {
                minX = Math.min(minX, star.x);
                minY = Math.min(minY, star.y);
                maxX = Math.max(maxX, star.x);
                maxY = Math.max(maxY, star.y);
            }
            LOGGER.info("  -> drafted={}, bounded={}, merged={} stars (size={}, halfMerge={}, center=({},{}), draftBounds=[{},{} to {},{}])",
                draftedStars.size(), boundedStars.size(), mergedStars.size(),
                String.format("%.3f", size), String.format("%.3f", halfMergeSpacing),
                String.format("%.2f", center.x), String.format("%.2f", center.y),
                String.format("%.2f", minX), String.format("%.2f", minY),
                String.format("%.2f", maxX), String.format("%.2f", maxY));
        }

        return mergedStars;
    }

    /**
     * Get level 1 cells that circumscribe a constellation using ConstellationInfo.
     */
    private List<int[]> getCircumscribingCells(ConstellationInfo info) {
        Point2D center = info.getCenter();
        double size = getConstellationSize();

        List<int[]> cells = new ArrayList<>();
        for (int dy = 0; dy < info.cellCountY; dy++) {
            for (int dx = 0; dx < info.cellCountX; dx++) {
                int x = info.startCellX + dx;
                int y = info.startCellY + dy;
                Point2D cellCenter = new Point2D(x + 0.5, y + 0.5);
                if (cellOverlapsConstellation(cellCenter, center, size)) {
                    cells.add(new int[] { x, y });
                }
            }
        }
        return cells;
    }

    /**
     * Check if a cell overlaps with a constellation.
     */
    private boolean cellOverlapsConstellation(Point2D cellCenter, Point2D constCenter, double constSize) {
        // For SQUARE, simple bounding box check
        double halfSize = constSize / 2.0 + 0.5; // Add half cell for overlap margin
        return Math.abs(cellCenter.x - constCenter.x) <= halfSize &&
               Math.abs(cellCenter.y - constCenter.y) <= halfSize;
    }

    /**
     * Check if a point is inside the constellation boundary (with optional margin).
     * For SQUARE, uses Chebyshev distance (max of |dx|, |dy|) instead of Euclidean.
     *
     * @param point The point to check
     * @param center The constellation center
     * @param size The constellation size
     * @param margin Distance inside the boundary to check (positive = stricter)
     * @return true if point is inside boundary minus margin
     */
    /**
     * Check if a point is inside the exact constellation boundary shape.
     * Uses precise geometric boundaries for each shape type.
     */
    private boolean isInsideConstellationBoundary(Point2D point, Point2D center, double size) {
        double dx = Math.abs(point.x - center.x);
        double dy = Math.abs(point.y - center.y);
        double halfSize = size / 2.0;

        switch (constellationShape) {
            case HEXAGON: {
                // Pointy-top hexagon with width = size
                // Circumradius R = size / sqrt(3)
                // Point is inside if:
                //   1. dx <= size/2 (within horizontal extent to left/right vertices)
                //   2. dy <= (size - dx) / sqrt(3) (within diagonal edges)
                double sqrt3 = Math.sqrt(3);
                if (dx > halfSize) return false;
                return dy <= (size - dx) / sqrt3;
            }
            case RHOMBUS: {
                // Rhombus (diamond) with diagonals along axes
                // Horizontal diagonal = size, vertical diagonal = size * sqrt(2)
                // Point is inside if: dx + dy/sqrt(2) <= halfSize
                double sqrt2 = Math.sqrt(2);
                return dx + dy / sqrt2 <= halfSize;
            }
            case SQUARE:
            default:
                // Square: use Chebyshev distance (max of |dx|, |dy|)
                return Math.max(dx, dy) <= halfSize;
        }
    }

    /**
     * Find the star (lowest elevation point) within a level 1 cell using 9x9 sampling.
     * If there are multiple points with the same lowest elevation, randomly select one.
     */
    private Point3D findStarInCelllLowestGrid(int cellX, int cellY) {
        // Generate deterministic jitter for this cell
        Random rng = initRandomGenerator(cellX, cellY, 0);
        double jitterX = rng.nextDouble()*(1-2*STAR_SAMPLE_BOUNDARY);
        double jitterY = rng.nextDouble()*(1-2*STAR_SAMPLE_BOUNDARY);

        double lowestElevation = Double.MAX_VALUE;
        List<double[]> lowestPoints = new ArrayList<>();

        for (int si = 0; si < STAR_SAMPLE_GRID_SIZE; si++) {
            for (int sj = 0; sj < STAR_SAMPLE_GRID_SIZE; sj++) {
                double tx = (sj + jitterX+STAR_SAMPLE_BOUNDARY) / STAR_SAMPLE_GRID_SIZE;
                double ty = (si + jitterY+STAR_SAMPLE_BOUNDARY) / STAR_SAMPLE_GRID_SIZE;

                tx = Math.max(0.0, Math.min(1.0, tx));
                ty = Math.max(0.0, Math.min(1.0, ty));

                double sampleX = cellX + tx;
                double sampleY = cellY + ty;

                double elevation = evaluateControlFunction(sampleX, sampleY);

                if (elevation < lowestElevation - MathUtils.EPSILON) {
                    lowestElevation = elevation;
                    lowestPoints.clear();
                    lowestPoints.add(new double[] { sampleX, sampleY, elevation });
                } else if (Math.abs(elevation - lowestElevation) < MathUtils.EPSILON) {
                    lowestPoints.add(new double[] { sampleX, sampleY, elevation });
                }
            }
        }

        if (lowestPoints.isEmpty()) {
            return new Point3D(cellX + 0.5, cellY + 0.5, evaluateControlFunction(cellX + 0.5, cellY + 0.5));
        }

        // If multiple points have same lowest elevation, randomly select one
        double[] selected;
        if (lowestPoints.size() > 1) {
            int idx = rng.nextInt(lowestPoints.size());
            selected = lowestPoints.get(idx);
        } else {
            selected = lowestPoints.get(0);
        }

        return new Point3D(selected[0], selected[1], selected[2]);
    }

    /**
     * Merge stars that are closer than MERGE_POINT_SPACING.
     * Uses shared merge function, prioritizing lower elevation stars.
     */
    private List<Point3D> mergeCloseStars(List<Point3D> stars) {
        //IMPORTANT: Suppressing merge to guarantee star availability in each cell.
        //return mergePointsByDistance(stars, MERGE_POINT_SPACING, true);
        return (stars);
    }

    /**
     * Merge points that are within mergeDistance of each other.
     * Uses an iterative epicenter algorithm:
     * 1. Find the point with the most neighbors within merge distance (epicenter)
     * 2. Merge epicenter and all its close neighbors into a single averaged point
     * 3. Evaluate control function to get new elevation
     * 4. Repeat until no points are within merge distance of each other
     *
     * @param points List of points to merge
     * @param mergeDistance Minimum distance between kept points
     * @param preferLowElevation Ignored - elevation is determined by control function
     * @return List of merged points with proper spacing
     */
    private List<Point3D> mergePointsByDistance(List<Point3D> points, double mergeDistance, boolean preferLowElevation) {
        if (points.size() <= 1) return new ArrayList<>(points);

        double mergeDistSq = mergeDistance * mergeDistance;
        List<Point3D> current = new ArrayList<>(points);

        // Iterate until no points are within merge distance
        boolean merged = true;
        while (merged && current.size() > 1) {
            merged = false;

            // Find the epicenter: point with most neighbors within merge distance
            int epicenterIdx = -1;
            int maxNeighborCount = 0;
            List<Integer> epicenterNeighbors = null;

            for (int i = 0; i < current.size(); i++) {
                Point2D pI = current.get(i).projectZ();
                List<Integer> neighbors = new ArrayList<>();

                for (int j = 0; j < current.size(); j++) {
                    if (i == j) continue;
                    double distSq = pI.distanceSquaredTo(current.get(j).projectZ());
                    if (distSq < mergeDistSq) {
                        neighbors.add(j);
                    }
                }

                if (neighbors.size() > maxNeighborCount) {
                    maxNeighborCount = neighbors.size();
                    epicenterIdx = i;
                    epicenterNeighbors = neighbors;
                }
            }

            // If no point has neighbors within merge distance, we're done
            if (maxNeighborCount == 0) {
                break;
            }

            // Merge epicenter and all its neighbors into averaged position
            double sumX = current.get(epicenterIdx).x;
            double sumY = current.get(epicenterIdx).y;
            int mergeCount = 1;

            for (int neighborIdx : epicenterNeighbors) {
                Point3D neighbor = current.get(neighborIdx);
                sumX += neighbor.x;
                sumY += neighbor.y;
                mergeCount++;
            }

            double avgX = sumX / mergeCount;
            double avgY = sumY / mergeCount;

            // Evaluate control function for new elevation
            double newElevation = evaluateControlFunction(avgX, avgY);
            Point3D mergedPoint = new Point3D(avgX, avgY, newElevation);

            // Build new list: remove epicenter and neighbors, add merged point
            Set<Integer> indicesToRemove = new HashSet<>(epicenterNeighbors);
            indicesToRemove.add(epicenterIdx);

            List<Point3D> next = new ArrayList<>();
            for (int i = 0; i < current.size(); i++) {
                if (!indicesToRemove.contains(i)) {
                    next.add(current.get(i));
                }
            }
            next.add(mergedPoint);

            current = next;
            merged = true;
        }

        return current;
    }

    // ========== CleanAndNetworkPoints Implementation ==========
    // Refactored to process segments one at a time, fully defining each before moving to next

    /**
     * Holds constellation layout information independent of shape type.
     * Provides the constellation center, and the level 1 cells that circumscribe it.
     */
    /**
     * Holds constellation layout information based on center coordinates (no indices).
     * Provides the constellation center, and the level 1 cells that circumscribe it.
     */
    private static class ConstellationInfo {
        final double centerX;      // Constellation center X coordinate
        final double centerY;      // Constellation center Y coordinate
        final int startCellX;      // First level 1 cell X that overlaps
        final int startCellY;      // First level 1 cell Y that overlaps
        final int cellCountX;      // Number of level 1 cells in X direction
        final int cellCountY;      // Number of level 1 cells in Y direction

        ConstellationInfo(double centerX, double centerY,
                          int startCellX, int startCellY, int cellCountX, int cellCountY) {
            this.centerX = centerX;
            this.centerY = centerY;
            this.startCellX = startCellX;
            this.startCellY = startCellY;
            this.cellCountX = cellCountX;
            this.cellCountY = cellCountY;
        }

        Point2D getCenter() {
            return new Point2D(centerX, centerY);
        }

        /**
         * Generate a unique key for this constellation based on center coordinates.
         * Uses quantized coordinates to ensure constellations at the same position get the same key.
         */
        long getKey() {
            // Quantize to avoid floating-point comparison issues
            // Using 1000x precision should be sufficient for constellation spacing
            long qx = Math.round(centerX * 1000);
            long qy = Math.round(centerY * 1000);
            return (qx << 32) | (qy & 0xFFFFFFFFL);
        }
    }

    /**
     * Internal class to track network node state during CleanAndNetworkPoints.
     * Mutable to allow adding subdivision points and updating tangents.
     */
    private static class NetworkNode {
        Point3D point;                     // Mutable to allow elevation adjustments
        int index;                         // Index in the node list
        final List<Integer> connections;   // Indices of connected nodes
        Vec2D tangent;                     // Computed tangent direction (null until set)
        boolean isBranchPoint;             // True if this node branches into another path
        int branchIntoNode;                // Index of node this branches into (-1 if not branching)
        boolean removed;                   // True if node was removed during cleaning
        int chainId;                       // Chain identifier for connectivity tracking
        boolean isSubdivisionPoint;        // True if this node was created by subdivision
        int sourceLevel;                   // Level of segment that created this point (for subdivision points)
        PointType pointType;               // Type of this point (ORIGINAL, TRUNK, KNOT, LEAF)

        NetworkNode(Point3D point, int index) {
            this.point = point;
            this.index = index;
            this.connections = new ArrayList<>();
            this.tangent = null;
            this.isBranchPoint = false;
            this.branchIntoNode = -1;
            this.removed = false;
            this.chainId = index;  // Initially each node is its own chain
            this.isSubdivisionPoint = false;
            this.sourceLevel = -1;
            this.pointType = PointType.ORIGINAL;  // Default for original points
        }
    }

    /**
     * Represents a fully-defined segment in the network.
     */
    private static class NetworkSegment {
        final int srtNodeIdx;
        final int endNodeIdx;
        Point3D srtPoint;
        Point3D endPoint;
        Vec2D tangentSrt;
        Vec2D tangentEnd;
        int level;
        boolean isBranch;  // True if this segment branches into the end node

        NetworkSegment(int srtIdx, int endIdx, Point3D srt, Point3D end, int level) {
            this.srtNodeIdx = srtIdx;
            this.endNodeIdx = endIdx;
            this.srtPoint = srt;
            this.endPoint = end;
            this.level = level;
            this.isBranch = false;
        }
    }

    /**
     * CleanAndNetworkPoints: Create network of segments from a list of points.
     *
     * REFACTORED Algorithm (per-segment approach):
     * 1. Clean points: merge close points, remove points near lower-level segments
     * 2. Probabilistically remove points based on branchesSampler (if not level 0)
     * 3. For each point (highest to lowest), create connection and FULLY DEFINE segment:
     *    - Connect to best neighbor
     *    - Compute tangents immediately
     *    - Subdivide segment
     *    - Displace subdivision points
     *    - Add subdivision points back to available nodes
     * 4. Connect remaining chains to root path
     *
     * @param cellX X coordinate of cell (constellation index for level 0)
     * @param cellY Y coordinate of cell
     * @param level Resolution level (0 for asterisms)
     * @param points List of points to connect
     * @return List of segments with tangent information
     */
    private List<Segment3D> CleanAndNetworkPoints(int cellX, int cellY, int level, List<Point3D> points) {
        return CleanAndNetworkPoints(cellX, cellY, level, points, new ArrayList<>());
    }

    /**
     * CleanAndNetworkPoints with previous level segments for cleaning.
     */
    private List<Segment3D> CleanAndNetworkPoints(int cellX, int cellY, int level,
                                                   List<Point3D> points,
                                                   List<Segment3D> previousLevelSegments) {
        if (points.isEmpty()) return new ArrayList<>();

        // Function setup: determine cell-specific distances using DIVISIONS_PER_LEVEL config
        double gridSpacing = getGridSpacingForLevel(level);
        double mergeDistance = MERGE_POINT_SPACING * gridSpacing;
        double maxSegmentDistance = MAX_POINT_SEGMENT_DISTANCE * gridSpacing;

        // Step 1: Clean network points - merge points within merge distance (only for level > 0)
        List<Point3D> cleanedPoints;
        if (level > 0) {
            cleanedPoints = cleanAndMergePoints(points, mergeDistance);
        } else {
            cleanedPoints = new ArrayList<>(points);
        }

        // Step 2: Remove points within merge distance of lower-level segments (if not level 0)
        if (level > 0 && !previousLevelSegments.isEmpty()) {
            cleanedPoints = removePointsNearSegments(cleanedPoints, previousLevelSegments, mergeDistance);
        }

        // Step 3: Probabilistically remove points based on branchesSampler and distance from segments
        // Points far from lower-level segments are more likely to be removed to reduce square patterns
        if (level > 0) {
            cleanedPoints = probabilisticallyRemovePoints(cleanedPoints, cellX, cellY,
                                                          previousLevelSegments, gridSpacing);
        }

        if (cleanedPoints.size() <= 1) {
            return new ArrayList<>();
        }

        // Create network nodes - these are mutable and will grow as subdivisions are added
        List<NetworkNode> nodes = new ArrayList<>();
        for (int i = 0; i < cleanedPoints.size(); i++) {
            nodes.add(new NetworkNode(cleanedPoints.get(i), i));
        }

        // Track all fully-defined segments
        List<Segment3D> allSegments = new ArrayList<>();

        // Step 4: Connect points using per-segment approach
        // Each segment is fully defined (tangents, subdivision, displacement) before next connection
        connectAndDefineSegments(nodes, allSegments, maxSegmentDistance, level, cellX, cellY, previousLevelSegments);

        // Note: For level 0 (asterisms), elevation shifting is done AFTER stitching in generateAsterism()
        // to preserve elevation information for stitch decisions

        return allSegments;
    }

    /**
     * Connect nodes and fully define each segment before moving to next.
     * This prevents overlapping connections by making subdivision points available.
     */
    private void connectAndDefineSegments(List<NetworkNode> nodes, List<Segment3D> allSegments,
                                           double maxSegmentDistance, int level, int cellX, int cellY,
                                           List<Segment3D> previousLevelSegments) {
        double maxDistSq = maxSegmentDistance * maxSegmentDistance;

        // Calculate merge distance for neighbor selection priority
        // Neighbors within merge distance are preferred (priority A in selection rules)
        double gridSpacing = getGridSpacingForLevel(level);
        double mergeDistance = MERGE_POINT_SPACING * gridSpacing;
        double mergeDistSq = mergeDistance * mergeDistance;

        // Union-Find for tracking chain connectivity
        int[] parent = new int[nodes.size() * 10];  // Extra space for subdivision points
        int[] rank = new int[nodes.size() * 10];
        for (int i = 0; i < parent.length; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        // Phase A: Create initial downstream flows
        int maxIterations = nodes.size() * nodes.size();  // Safety limit
        int iterations = 0;

        // For level 0: Build trunk continuously from highest point
        if (level == 0) {
            // Start at highest elevation point
            int currentNode = findHighestUnconnectedOriginalNode(nodes);
            if (currentNode >= 0) {
                nodes.get(currentNode).pointType = PointType.TRUNK;
            }

            // Continuously extend trunk until no more connections can be made
            while (currentNode >= 0 && iterations < maxIterations) {
                iterations++;

                // Try to extend trunk from current node (isTrunk = true)
                // Returns the neighbor index that was connected to, or -1 if no connection
                int nextNode = createAndDefineSegment(nodes, allSegments, currentNode,
                                                          maxDistSq, mergeDistSq, level, cellX, cellY,
                                                          parent, rank, previousLevelSegments,
                                                          true);  // isTrunk = true
                if (nextNode < 0) {
                    break;  // Trunk is complete - no more downhill connections
                }

                // Continue trunk from the neighbor we just connected to
                currentNode = nextNode;
                nodes.get(currentNode).pointType = PointType.TRUNK;
            }

            // DEBUG: Return after trunk creation (SEGMENT_DEBUGGING == 15)
            if (SEGMENT_DEBUGGING == 15) {
                // Add 0-length segments for unconnected points for visualization
                for (NetworkNode node : nodes) {
                    if (node.connections.isEmpty() && !node.removed) {
                        allSegments.add(new Segment3D(node.point, node.point, 0, null, null,
                                                       node.pointType, node.pointType));
                    }
                }
                LOGGER.info("SEGMENT_DEBUGGING=15: Returning after trunk ({} segments, {} nodes)", allSegments.size(), nodes.size());
                return;
            }
        }

        // For all levels: Process remaining unconnected points from highest to lowest
        while (iterations < maxIterations) {
            iterations++;

            // Find highest elevation ORIGINAL node (not subdivision point) without any connections
            int highestUnconnected = findHighestUnconnectedOriginalNode(nodes);
            if (highestUnconnected < 0) break;

            // Attempt to create and fully define a segment (isTrunk = false for branch connections)
            int neighborIdx = createAndDefineSegment(nodes, allSegments, highestUnconnected,
                                                      maxDistSq, mergeDistSq, level, cellX, cellY,
                                                      parent, rank, previousLevelSegments,
                                                      false);  // isTrunk = false
            if (neighborIdx < 0) {
                // No valid connection found - mark node so we don't try again
                NetworkNode node = nodes.get(highestUnconnected);
                if (level > 0) {
                    // At higher levels, remove nodes that can't connect
                    node.removed = true;
                }
            }
        }

        if (iterations >= maxIterations) {
            LOGGER.warn("CleanAndNetworkPoints Phase A reached max iterations ({})", maxIterations);
        }

        // DEBUG: Return after Phase A (initial connections only)
        if (SEGMENT_DEBUGGING == 40) {
            LOGGER.info("SEGMENT_DEBUGGING=40: Returning after Phase A ({} segments, {} nodes)", allSegments.size(), nodes.size());
            return;
        }

        // Phase B: Connect chains to form single connected network
        // Find chains that aren't connected to the root chain and connect them
        connectChainsToRoot(nodes, allSegments, maxDistSq, mergeDistSq, level, cellX, cellY, parent, rank, previousLevelSegments);

        // DEBUG: Return after Phase B (chain connections)
        if (SEGMENT_DEBUGGING == 50) {
            LOGGER.info("SEGMENT_DEBUGGING=50: Returning after Phase B ({} segments, {} nodes)", allSegments.size(), nodes.size());
        }

        // Mark leaf points (only connected to one other point)
        for (NetworkNode node : nodes) {
            if (!node.removed && node.connections.size() == 1 && !node.isSubdivisionPoint) {
                node.pointType = PointType.LEAF;
            }
        }

        // Check for undefined tangents if error flag is set
        if (ERROR_ON_UNDEFINED_TANGENT) {
            for (Segment3D seg : allSegments) {
                if (seg.tangentSrt == null || seg.tangentEnd == null) {
                    String msg = String.format(
                        "Segment has undefined tangent at level %d: srt=%s (tangent=%s), end=%s (tangent=%s)",
                        level, seg.srt, seg.tangentSrt, seg.end, seg.tangentEnd);
                    LOGGER.error(msg);
                    throw new IllegalStateException(msg);
                }
            }
        }
    }

    /**
     * Find the highest elevation ORIGINAL node (not a subdivision point) that has no connections yet.
     * Subdivision points should only be targets, not initiators of connections.
     */
    private int findHighestUnconnectedOriginalNode(List<NetworkNode> nodes) {
        int bestIdx = -1;
        double bestZ = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < nodes.size(); i++) {
            NetworkNode node = nodes.get(i);
            if (node.removed) continue;
            if (node.isSubdivisionPoint) continue;  // Skip subdivision points
            if (!node.connections.isEmpty()) continue;

            if (node.point.z > bestZ) {
                bestZ = node.point.z;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * Get the last connected neighbor of a node (for trunk extension).
     * Returns the most recently added connection, or -1 if no connections.
     */
    private int getLastConnectedNeighbor(List<NetworkNode> nodes, int nodeIdx) {
        NetworkNode node = nodes.get(nodeIdx);
        if (node.connections.isEmpty()) {
            return -1;
        }
        // Return the last connection added (most recent)
        return node.connections.get(node.connections.size() - 1);
    }

    /**
     * Create a connection and fully define the resulting segment.
     * This includes: connecting, computing tangents, subdividing, displacing.
     * Subdivision points are added back to the node list.
     *
     * @param mergeDistSq Squared merge distance (neighbors within this distance are preferred)
     * @param isTrunk True if building trunk (level 0), requires downhill flow
     * @return index of the neighbor that was connected to, or -1 if no connection made
     */
    private int createAndDefineSegment(List<NetworkNode> nodes, List<Segment3D> allSegments,
                                            int sourceIdx, double maxDistSq, double mergeDistSq,
                                            int level, int cellX, int cellY, int[] parent, int[] rank,
                                            List<Segment3D> previousLevelSegments,
                                            boolean isTrunk) {
        NetworkNode sourceNode = nodes.get(sourceIdx);

        // Find best neighbor: prefer within merge distance, then lowest slope
        int bestNeighbor = findBestNeighborForSegment(nodes, sourceIdx, maxDistSq, mergeDistSq, level, parent, allSegments, isTrunk);
        if (bestNeighbor < 0) {
            return -1;  // No valid neighbor found
        }

        NetworkNode targetNode = nodes.get(bestNeighbor);
        double slopeToNeighbor = calculateSlope(sourceNode.point, targetNode.point);

        // Check slope cutoff for non-asterism levels
        if (level > 0 && slopeToNeighbor > lowestSlopeCutoff && sourceNode.connections.isEmpty()) {
            sourceNode.removed = true;
            return -1;  // Slope too high
        }

        // Check if this would create a crossing segment (may need multiple redirections)
        Point2D srcPos = sourceNode.point.projectZ();
        Point2D tgtPos = targetNode.point.projectZ();
        int crossingRedirects = 0;
        final int maxCrossingRedirects = 5;  // Prevent infinite loops

        while (wouldCrossExistingSegment(srcPos, tgtPos, allSegments) && crossingRedirects < maxCrossingRedirects) {
            crossingRedirects++;
            // Find subdivision point on crossed segment instead
            int subdivisionIdx = createSubdivisionAtCrossing(nodes, allSegments, srcPos, tgtPos, level, parent, rank);
            if (subdivisionIdx >= 0) {
                bestNeighbor = subdivisionIdx;
                targetNode = nodes.get(bestNeighbor);
                tgtPos = targetNode.point.projectZ();  // Update target for next crossing check
            } else {
                return -1;  // Couldn't resolve crossing
            }
        }

        if (crossingRedirects >= maxCrossingRedirects) {
            return -1;  // Too many crossing redirects - path is blocked
        }

        // Determine if this is a branch (target already has 2+ connections)
        boolean isBranch = targetNode.connections.size() >= 2;

        // Add bidirectional connections
        if (!sourceNode.connections.contains(bestNeighbor)) {
            sourceNode.connections.add(bestNeighbor);
        }
        if (!targetNode.connections.contains(sourceIdx)) {
            targetNode.connections.add(sourceIdx);
        }

        // Update Union-Find for chain tracking
        if (sourceIdx < parent.length && bestNeighbor < parent.length) {
            int root1 = find(parent, sourceIdx);
            int root2 = find(parent, bestNeighbor);
            if (root1 != root2) {
                union(parent, rank, root1, root2);
            }
        }

        // Reduce elevation if flow would be upward
        if (slopeToNeighbor > 0) {
            adjustElevationForDownwardFlow(sourceNode, targetNode);
        }

        // Determine flow direction (srt is higher elevation)
        Point3D srtPoint, endPoint;
        Vec2D tangentSrt, tangentEnd;
        int srtIdx, endIdx;

        if (sourceNode.point.z >= targetNode.point.z) {
            srtPoint = sourceNode.point;
            endPoint = targetNode.point;
            srtIdx = sourceIdx;
            endIdx = bestNeighbor;
        } else {
            srtPoint = targetNode.point;
            endPoint = sourceNode.point;
            srtIdx = bestNeighbor;
            endIdx = sourceIdx;
        }

        // Compute tangents for both endpoints
        // forStart=true for srt (start point), forStart=false for end (end point)
        // Pass the other endpoint so tangent can be computed relative to segment direction
        tangentSrt = computeNodeTangent(nodes, srtIdx, endPoint, isBranch && srtIdx == sourceIdx, true, allSegments, cellX, cellY);
        tangentEnd = computeNodeTangent(nodes, endIdx, srtPoint, isBranch && endIdx == sourceIdx, false, allSegments, cellX, cellY);

        // Store tangents in nodes for future reference (only if not already set)
        if (nodes.get(srtIdx).tangent == null) {
            nodes.get(srtIdx).tangent = tangentSrt;
        }
        if (nodes.get(endIdx).tangent == null) {
            nodes.get(endIdx).tangent = tangentEnd;
        }

        // Create the segment
        Segment3D segment = new Segment3D(srtPoint, endPoint, level, tangentSrt, tangentEnd);

        // Subdivide and displace, adding subdivision points back to nodes
        // Pass the endpoint indices so subdivision points can be properly connected
        // Calculate divisions from segment length / merge point spacing to ensure nodes available at lower levels
        // Use next level's grid spacing for subdivision density
        double gridSpacing = getGridSpacingForLevel(level + 1);
        double mergeDistance = MERGE_POINT_SPACING * gridSpacing;
        double segmentLength = srtPoint.projectZ().distanceTo(endPoint.projectZ());
        int divisions = Math.max(1, (int) Math.ceil(segmentLength / mergeDistance));
        double jitterFactor = 0.5;
        List<Segment3D> subdivided = subdivideAndAddPoints(segment, nodes, divisions, jitterFactor, level, cellX, cellY, parent, rank, srtIdx, endIdx, mergeDistance);
        allSegments.addAll(subdivided);

        return bestNeighbor;  // Return the neighbor index for trunk continuation
    }

    /**
     * Find best neighbor for segment creation.
     * Uses normalized slope with DistanceFalloffPower and BranchEncouragementFactor.
     *
     * @param nodes List of network nodes
     * @param sourceIdx Index of source node
     * @param maxDistSq Maximum distance squared for connection
     * @param level Current resolution level
     * @param parent Union-Find parent array
     * @param existingSegments List of existing segments (unused but kept for API)
     * @param isTrunk True if building the trunk (level 0), requires negative slope
     * @return Index of best neighbor, or -1 if none found
     */
    private int findBestNeighborForSegment(List<NetworkNode> nodes, int sourceIdx,
                                            double maxDistSq, double mergeDistSq, int level,
                                            int[] parent, List<Segment3D> existingSegments,
                                            boolean isTrunk) {
        NetworkNode sourceNode = nodes.get(sourceIdx);
        Point2D sourcePos = sourceNode.point.projectZ();

        // Track best neighbor within merge distance (priority A) and overall (priority B)
        double bestSlopeWithinMerge = Double.MAX_VALUE;
        int bestNeighborWithinMerge = -1;
        double bestSlopeOverall = Double.MAX_VALUE;
        int bestNeighborOverall = -1;

        for (int i = 0; i < nodes.size(); i++) {
            if (i == sourceIdx) continue;
            NetworkNode candidate = nodes.get(i);
            if (candidate.removed) continue;
            if (sourceNode.connections.contains(i)) continue;  // Already directly connected

            // Check if already connected via path (same chain)
            if (sourceIdx < parent.length && i < parent.length) {
                if (find(parent, sourceIdx) == find(parent, i)) {
                    continue;  // Already in same chain
                }
            }

            Point2D candidatePos = candidate.point.projectZ();
            double distSq = sourcePos.distanceSquaredTo(candidatePos);

            if (distSq > maxDistSq || distSq < MathUtils.EPSILON) continue;

            // Calculate normalized slope with DistanceFalloffPower
            // normalizedSlope = heightDiff / dist^DISTANCE_FALLOFF_POWER
            double dist = Math.sqrt(distSq);
            double heightDiff = candidate.point.z - sourceNode.point.z;
            double normalizedSlope = heightDiff / Math.pow(dist, DISTANCE_FALLOFF_POWER);

            // Validity check based on context
            if (isTrunk && normalizedSlope >= 0) {
                continue;  // Trunk requires downhill flow (negative slope)
            }
            if (level > 0 && normalizedSlope > lowestSlopeCutoff) {
                continue;  // Higher levels have slope cutoff
            }

            // Apply branch encouragement factor if candidate has 2+ connections
            // (has a line passing through it - encourages attaching to existing flows)
            double effectiveSlope = normalizedSlope;
            if (candidate.connections.size() >= 2) {
                effectiveSlope *= BRANCH_ENCOURAGEMENT_FACTOR;
            }

            // Priority A: Prefer neighbors within merge distance
            if (distSq <= mergeDistSq) {
                if (effectiveSlope < bestSlopeWithinMerge) {
                    bestSlopeWithinMerge = effectiveSlope;
                    bestNeighborWithinMerge = i;
                }
            }

            // Priority B: Track overall best (fallback if none within merge distance)
            if (effectiveSlope < bestSlopeOverall) {
                bestSlopeOverall = effectiveSlope;
                bestNeighborOverall = i;
            }
        }

        // Return neighbor within merge distance if any, otherwise overall best
        return (bestNeighborWithinMerge >= 0) ? bestNeighborWithinMerge : bestNeighborOverall;
    }

    /**
     * Check if a new segment would cross any existing segment.
     */
    private boolean wouldCrossExistingSegment(Point2D srt, Point2D end, List<Segment3D> segments) {
        for (Segment3D seg : segments) {
            Point2D segSrt = seg.srt.projectZ();
            Point2D segEnd = seg.end.projectZ();

            if (segmentsIntersect(srt, end, segSrt, segEnd)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Check if two 2D line segments intersect (excluding shared endpoints).
     */
    private boolean segmentsIntersect(Point2D a1, Point2D a2, Point2D b1, Point2D b2) {
        // Skip if they share an endpoint
        if (a1.distanceSquaredTo(b1) < MathUtils.EPSILON ||
            a1.distanceSquaredTo(b2) < MathUtils.EPSILON ||
            a2.distanceSquaredTo(b1) < MathUtils.EPSILON ||
            a2.distanceSquaredTo(b2) < MathUtils.EPSILON) {
            return false;
        }

        double d1 = crossProduct(b2.x - b1.x, b2.y - b1.y, a1.x - b1.x, a1.y - b1.y);
        double d2 = crossProduct(b2.x - b1.x, b2.y - b1.y, a2.x - b1.x, a2.y - b1.y);
        double d3 = crossProduct(a2.x - a1.x, a2.y - a1.y, b1.x - a1.x, b1.y - a1.y);
        double d4 = crossProduct(a2.x - a1.x, a2.y - a1.y, b2.x - a1.x, b2.y - a1.y);

        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
            ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }

        return false;
    }

    private double crossProduct(double ax, double ay, double bx, double by) {
        return ax * by - ay * bx;
    }

    /**
     * Create a subdivision point on the crossed segment, split the segment, and add to nodes.
     * The crossed segment is replaced with two new segments: (srt -> subdivision) and (subdivision -> end).
     * The new node is properly connected to both endpoint nodes of the original segment.
     */
    private int createSubdivisionAtCrossing(List<NetworkNode> nodes, List<Segment3D> segments,
                                             Point2D newSrt, Point2D newEnd, int level,
                                             int[] parent, int[] rank) {
        // Find the segment being crossed and the intersection point
        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);
            Point2D segSrt = seg.srt.projectZ();
            Point2D segEnd = seg.end.projectZ();

            if (segmentsIntersect(newSrt, newEnd, segSrt, segEnd)) {
                // Calculate intersection point
                Point2D intersection = lineIntersection(newSrt, newEnd, segSrt, segEnd);
                if (intersection == null) continue;

                // Create 3D point at intersection (interpolate elevation)
                double t = segSrt.distanceTo(intersection) / segSrt.distanceTo(segEnd);
                double z = MathUtils.lerp(seg.srt.z, seg.end.z, t);
                Point3D newPoint = new Point3D(intersection.x, intersection.y, z);

                // Find node indices for the crossed segment's endpoints
                int srtNodeIdx = findNodeByPosition(nodes, seg.srt);
                int endNodeIdx = findNodeByPosition(nodes, seg.end);

                // Add as new node
                int newIdx = nodes.size();
                NetworkNode newNode = new NetworkNode(newPoint, newIdx);
                newNode.isSubdivisionPoint = true;
                newNode.sourceLevel = seg.level;

                // Inherit tangent from the segment at this point
                if (seg.tangentSrt != null && seg.tangentEnd != null) {
                    newNode.tangent = new Vec2D(
                        MathUtils.lerp(seg.tangentSrt.x, seg.tangentEnd.x, t),
                        MathUtils.lerp(seg.tangentSrt.y, seg.tangentEnd.y, t)
                    ).normalize();
                }

                nodes.add(newNode);

                // Connect new node to the original segment's endpoints
                if (srtNodeIdx >= 0) {
                    NetworkNode srtNode = nodes.get(srtNodeIdx);
                    // Remove old connection between srt and end
                    srtNode.connections.remove(Integer.valueOf(endNodeIdx));
                    // Add connection to new subdivision node
                    if (!srtNode.connections.contains(newIdx)) {
                        srtNode.connections.add(newIdx);
                    }
                    if (!newNode.connections.contains(srtNodeIdx)) {
                        newNode.connections.add(srtNodeIdx);
                    }
                }
                if (endNodeIdx >= 0) {
                    NetworkNode endNode = nodes.get(endNodeIdx);
                    // Remove old connection between end and srt
                    endNode.connections.remove(Integer.valueOf(srtNodeIdx));
                    // Add connection to new subdivision node
                    if (!endNode.connections.contains(newIdx)) {
                        endNode.connections.add(newIdx);
                    }
                    if (!newNode.connections.contains(endNodeIdx)) {
                        newNode.connections.add(endNodeIdx);
                    }
                }

                // Remove the original crossed segment
                segments.remove(i);

                // Add two new segments: srt->subdivision and subdivision->end
                // Interpolate tangents at the subdivision point
                Vec2D tangentAtSub = newNode.tangent;
                Segment3D seg1 = new Segment3D(seg.srt, newPoint, seg.level, seg.tangentSrt, tangentAtSub);
                Segment3D seg2 = new Segment3D(newPoint, seg.end, seg.level, tangentAtSub, seg.tangentEnd);
                segments.add(seg1);
                segments.add(seg2);

                // Update Union-Find: new node should be in same chain as original endpoints
                if (srtNodeIdx >= 0 && srtNodeIdx < parent.length && newIdx < parent.length) {
                    int rootSrt = find(parent, srtNodeIdx);
                    union(parent, rank, rootSrt, newIdx);
                }
                if (endNodeIdx >= 0 && endNodeIdx < parent.length && newIdx < parent.length) {
                    int rootEnd = find(parent, endNodeIdx);
                    int rootNew = find(parent, newIdx);
                    if (rootEnd != rootNew) {
                        union(parent, rank, rootEnd, rootNew);
                    }
                }

                return newIdx;
            }
        }

        return -1;
    }

    /**
     * Find the node index for a given 3D point by position matching.
     * Returns -1 if no matching node is found.
     */
    private int findNodeByPosition(List<NetworkNode> nodes, Point3D point) {
        double epsilon = MathUtils.EPSILON * 100;  // Allow small tolerance
        for (int i = 0; i < nodes.size(); i++) {
            NetworkNode node = nodes.get(i);
            if (node.removed) continue;
            double distSq = node.point.projectZ().distanceSquaredTo(point.projectZ());
            if (distSq < epsilon) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Calculate intersection point of two line segments.
     */
    private Point2D lineIntersection(Point2D a1, Point2D a2, Point2D b1, Point2D b2) {
        double x1 = a1.x, y1 = a1.y, x2 = a2.x, y2 = a2.y;
        double x3 = b1.x, y3 = b1.y, x4 = b2.x, y4 = b2.y;

        double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Math.abs(denom) < MathUtils.EPSILON) return null;

        double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
        return new Point2D(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
    }

    /**
     * Compute tangent for a node based on its connections and segment direction.
     * Handles tangent continuity: when connecting to an existing tangent,
     * inverts if both are used as the same type (both start or both end).
     * Tangent is computed as blend of segment direction and slope direction,
     * then clamped to ±60° of segment direction.
     *
     * @param nodes List of network nodes
     * @param nodeIdx Index of the node to compute tangent for
     * @param otherPoint The other endpoint of the segment being created (for direction reference)
     * @param isBranchEnd True if this node is a branch endpoint
     * @param forStart True if tangent is for use as segment start, false for end
     * @param allSegments List of all segments to derive tangent usage from
     * @param cellX, cellY Cell coordinates for RNG seeding
     */
    private Vec2D computeNodeTangent(List<NetworkNode> nodes, int nodeIdx, Point3D otherPoint,
                                      boolean isBranchEnd, boolean forStart,
                                      List<Segment3D> allSegments, int cellX, int cellY) {
        NetworkNode node = nodes.get(nodeIdx);

        // Compute segment direction (always points in flow direction: srt -> end)
        Vec2D segmentDir;
        if (forStart) {
            // For start point, segment direction is from this node toward otherPoint
            segmentDir = new Vec2D(node.point.projectZ(), otherPoint.projectZ());
        } else {
            // For end point, segment direction is from otherPoint toward this node
            segmentDir = new Vec2D(otherPoint.projectZ(), node.point.projectZ());
        }
        if (segmentDir.lengthSquared() > MathUtils.EPSILON) {
            segmentDir = segmentDir.normalize();
        } else {
            segmentDir = new Vec2D(1, 0);  // Fallback
        }

        // If node already has a tangent, handle continuity
        if (node.tangent != null) {
            // Derive whether the existing tangent was for start or end usage
            // by finding the segment that contains this node's point
            boolean existingWasForStart = false;
            for (Segment3D seg : allSegments) {
                if (pointsMatch(seg.srt, node.point)) {
                    existingWasForStart = true;
                    break;
                } else if (pointsMatch(seg.end, node.point)) {
                    existingWasForStart = false;
                    break;
                }
            }

            // Check if we need to invert for continuity
            // Flow-through (srt→end or end→srt): tangents should be identical
            // Same-type (srt→srt or end→end): tangent should be inverted
            Vec2D continuityTangent;
            if (existingWasForStart == forStart) {
                // Same type connection (both start or both end) - invert tangent
                continuityTangent = new Vec2D(-node.tangent.x, -node.tangent.y);
            } else {
                // Flow-through connection - use tangent as-is
                continuityTangent = node.tangent;
            }

            // Clamp continuity tangent to ±60° of segment direction
            return clampTangentToSegmentDirection(continuityTangent, segmentDir);
        }

        // Determine twist: random ±50° * max((1 - |slope|/SlopeWhenStraight), 0)
        double nodeSlope = node.point.hasSlope() ? node.point.getSlope() : 0;
        double twistFactor = Math.max(0, 1.0 - Math.abs(nodeSlope) / slopeWhenStraight);
        Random rng = initRandomGenerator((int)(node.point.x * 1000), (int)(node.point.y * 1000), 73);
        double twistAngle = (rng.nextDouble() * 2 - 1) * Math.toRadians(50) * twistFactor;

        Vec2D nominalTangent;

        if (isBranchEnd) {
            // Branch tangent: random offset of 110-170° from segment direction
            // on the same side as the flow
            double branchAngle = Math.toRadians(110 + rng.nextDouble() * 60);  // 110-170°
            if (rng.nextBoolean()) branchAngle = -branchAngle;  // Random side
            nominalTangent = rotateTangent(segmentDir, branchAngle);
        } else {
            // Blend segment direction with slope direction per NetworkingRules line 73-74
            Vec2D slopeTangent = getTangentFromPoint(node.point);

            // Average the angles of segment direction and slope tangent
            double segAngle = Math.atan2(segmentDir.y, segmentDir.x);
            double slopeAngle = Math.atan2(slopeTangent.y, slopeTangent.x);

            // Handle angle wrapping for proper averaging
            double angleDiff = slopeAngle - segAngle;
            while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
            while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

            // Average is segAngle + half the difference
            double avgAngle = segAngle + angleDiff * 0.5;

            nominalTangent = new Vec2D(Math.cos(avgAngle), Math.sin(avgAngle));
        }

        // Apply twist rotation
        Vec2D twistedTangent = rotateTangent(nominalTangent, twistAngle);

        // Clamp to ±60° of segment direction per NetworkingRules line 79
        return clampTangentToSegmentDirection(twistedTangent, segmentDir);
    }

    /**
     * Clamp a tangent vector to be within ±60° of the segment direction.
     */
    private Vec2D clampTangentToSegmentDirection(Vec2D tangent, Vec2D segmentDir) {
        double tangentAngle = Math.atan2(tangent.y, tangent.x);
        double segmentAngle = Math.atan2(segmentDir.y, segmentDir.x);

        double angleDiff = tangentAngle - segmentAngle;
        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

        double maxAngle = Math.toRadians(60);  // ±60° limit
        if (angleDiff > maxAngle) {
            double clampedAngle = segmentAngle + maxAngle;
            return new Vec2D(Math.cos(clampedAngle), Math.sin(clampedAngle));
        } else if (angleDiff < -maxAngle) {
            double clampedAngle = segmentAngle - maxAngle;
            return new Vec2D(Math.cos(clampedAngle), Math.sin(clampedAngle));
        }

        return tangent;  // Within bounds
    }

    /**
     * Adjust elevation of source node to ensure downward flow.
     */
    private void adjustElevationForDownwardFlow(NetworkNode sourceNode, NetworkNode targetNode) {
        if (sourceNode.point.z > targetNode.point.z) {
            // Flow would be upward - reduce source elevation
            double newZ = targetNode.point.z - MathUtils.EPSILON;
            sourceNode.point = new Point3D(sourceNode.point.x, sourceNode.point.y, newZ,
                                            sourceNode.point.slopeX, sourceNode.point.slopeY);
        }
    }

    /**
     * Check if two points match (within epsilon tolerance).
     */
    private static boolean pointsMatch(Point3D a, Point3D b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return (dx * dx + dy * dy + dz * dz) < MathUtils.EPSILON * MathUtils.EPSILON;
    }

    // ========== Shared Subdivision Helpers ==========

    /**
     * Result of a single subdivision interpolation: position and tangent.
     */
    private static class SubdivisionPoint {
        final Point3D position;
        final Vec2D tangent;

        SubdivisionPoint(Point3D position, Vec2D tangent) {
            this.position = position;
            this.tangent = tangent;
        }
    }

    /**
     * Compute a subdivision point along a segment at parameter t using Hermite spline interpolation.
     *
     * @param segment The segment to interpolate
     * @param t Parameter value [0, 1]
     * @param tangentScale Scale factor for tangent magnitudes in Hermite interpolation
     * @return SubdivisionPoint with interpolated position and tangent
     */
    private SubdivisionPoint computeSubdivisionPoint(Segment3D segment, double t, double tangentScale) {
        Point3D position;
        Vec2D tangent;

        if (segment.hasTangents()) {
            // Cubic Hermite spline interpolation
            double t2 = t * t;
            double t3 = t2 * t;

            // Hermite basis functions
            double h00 = 2 * t3 - 3 * t2 + 1;
            double h10 = t3 - 2 * t2 + t;
            double h01 = -2 * t3 + 3 * t2;
            double h11 = t3 - t2;

            // Position interpolation
            double x = h00 * segment.srt.x + h10 * (segment.tangentSrt != null ? segment.tangentSrt.x * tangentScale : 0)
                     + h01 * segment.end.x + h11 * (segment.tangentEnd != null ? segment.tangentEnd.x * tangentScale : 0);
            double y = h00 * segment.srt.y + h10 * (segment.tangentSrt != null ? segment.tangentSrt.y * tangentScale : 0)
                     + h01 * segment.end.y + h11 * (segment.tangentEnd != null ? segment.tangentEnd.y * tangentScale : 0);
            double z = MathUtils.lerp(segment.srt.z, segment.end.z, t);

            position = new Point3D(x, y, z);

            // Derivative of Hermite basis functions for tangent computation
            double h00d = 6 * t2 - 6 * t;
            double h10d = 3 * t2 - 4 * t + 1;
            double h01d = -6 * t2 + 6 * t;
            double h11d = 3 * t2 - 2 * t;

            double dx = h00d * segment.srt.x + h10d * (segment.tangentSrt != null ? segment.tangentSrt.x * tangentScale : 0)
                      + h01d * segment.end.x + h11d * (segment.tangentEnd != null ? segment.tangentEnd.x * tangentScale : 0);
            double dy = h00d * segment.srt.y + h10d * (segment.tangentSrt != null ? segment.tangentSrt.y * tangentScale : 0)
                      + h01d * segment.end.y + h11d * (segment.tangentEnd != null ? segment.tangentEnd.y * tangentScale : 0);

            // Normalize to get unit tangent direction
            double len = Math.sqrt(dx * dx + dy * dy);
            tangent = (len > MathUtils.EPSILON) ? new Vec2D(dx / len, dy / len) : null;
        } else {
            // Linear interpolation fallback
            position = Point3D.lerp(segment.srt, segment.end, t);

            // Tangent is just the segment direction
            double dx = segment.end.x - segment.srt.x;
            double dy = segment.end.y - segment.srt.y;
            double len = Math.sqrt(dx * dx + dy * dy);
            tangent = (len > MathUtils.EPSILON) ? new Vec2D(dx / len, dy / len) : null;
        }

        return new SubdivisionPoint(position, tangent);
    }

    /**
     * Apply jitter to a point using deterministic random based on position.
     *
     * @param point The point to jitter
     * @param maxJitter Maximum jitter distance
     * @param seedX X component for RNG seed
     * @param seedY Y component for RNG seed
     * @param seedZ Z component for RNG seed
     * @return New point with jitter applied
     */
    private Point3D applyJitter(Point3D point, double maxJitter, int seedX, int seedY, int seedZ) {
        if (maxJitter <= MathUtils.EPSILON) {
            return point;
        }
        Random rng = initRandomGenerator(seedX, seedY, seedZ);
        double jitterX = (rng.nextDouble() * 2 - 1) * maxJitter;
        double jitterY = (rng.nextDouble() * 2 - 1) * maxJitter;
        return new Point3D(point.x + jitterX, point.y + jitterY, point.z);
    }

    /**
     * Subdivide segment and add subdivision points back to nodes list.
     * Uses cubic Hermite spline interpolation when tangents are available.
     * Subdivision points are properly connected to form a chain from srtIdx to endIdx.
     *
     * @param segment The segment to subdivide
     * @param nodes The list of network nodes
     * @param divisions Number of segments to create (must be >= 1)
     * @param jitterFactor Jitter factor (0-1), where 0 = no jitter, 1 = max jitter
     * @param level The resolution level
     * @param cellX, cellY The cell coordinates for RNG seeding
     * @param parent, rank Union-Find data structures
     * @param srtIdx The node index for segment start (srt)
     * @param endIdx The node index for segment end (end)
     * @param mergeDistance The merge point spacing distance for this level
     */
    private List<Segment3D> subdivideAndAddPoints(Segment3D segment, List<NetworkNode> nodes,
                                                   int divisions, double jitterFactor,
                                                   int level, int cellX, int cellY,
                                                   int[] parent, int[] rank,
                                                   int srtIdx, int endIdx, double mergeDistance) {
        if (divisions <= 1) {
            return List.of(segment);
        }

        // Calculate maximum jitter based on segment length and number of divisions
        double segLength2D = segment.srt.projectZ().distanceTo(segment.end.projectZ());
        double maxJitter = (level > 0) ? (segLength2D / (divisions * 2.0)) * jitterFactor : 0;

        List<Segment3D> result = new ArrayList<>();
        List<Integer> chainNodeIndices = new ArrayList<>();
        chainNodeIndices.add(srtIdx);

        // Precompute tangent scale for Hermite interpolation
        double tangentScale = mergeDistance * TANGENT_MAGNITUDE_SCALE * tangentStrength;

        Point3D prev = segment.srt;
        Vec2D prevTangent = segment.tangentSrt;

        for (int i = 1; i <= divisions; i++) {
            double t = (double) i / divisions;
            Point3D next;
            Vec2D nextTangent;

            if (i == divisions) {
                // Last point is the segment endpoint
                next = segment.end;
                nextTangent = segment.tangentEnd;
            } else {
                // Use shared helper for Hermite interpolation
                SubdivisionPoint subdivPt = computeSubdivisionPoint(segment, t, tangentScale);
                next = subdivPt.position;
                nextTangent = subdivPt.tangent;

                // Apply jitter for interior points (only at level > 0 to prevent crossings)
                if (maxJitter > MathUtils.EPSILON) {
                    next = applyJitter(next,
                        maxJitter,
                        (int)(prev.x * 1000 + next.x * 500),
                        (int)(prev.y * 1000),
                        level + i);
                }

                // Create new node for this subdivision point
                int newIdx = nodes.size();
                NetworkNode newNode = new NetworkNode(next, newIdx);
                newNode.isSubdivisionPoint = true;
                newNode.sourceLevel = level;

                // Store the inherited tangent from B-spline derivative
                if (nextTangent != null) {
                    newNode.tangent = nextTangent;
                }

                nodes.add(newNode);
                chainNodeIndices.add(newIdx);
            }

            // Determine endpoint types for this sub-segment using PointType enum
            // First segment's srt keeps original type, subsequent are KNOT
            // Last segment's end keeps original type, others are KNOT
            PointType prevType = (result.isEmpty()) ? segment.srtType : PointType.KNOT;
            PointType nextType = (i == divisions) ? segment.endType : PointType.KNOT;

            // Create sub-segment with tangents for proper rendering
            result.add(new Segment3D(prev, next, segment.level, prevTangent, nextTangent, prevType, nextType));
            prev = next;
            prevTangent = nextTangent;
        }

        // Add end node to chain
        chainNodeIndices.add(endIdx);

        // Connect all nodes in the chain and update Union-Find
        for (int i = 0; i < chainNodeIndices.size() - 1; i++) {
            int nodeA = chainNodeIndices.get(i);
            int nodeB = chainNodeIndices.get(i + 1);

            NetworkNode a = nodes.get(nodeA);
            NetworkNode b = nodes.get(nodeB);

            // Add bidirectional connections (skip if already present from original connection)
            if (!a.connections.contains(nodeB)) {
                a.connections.add(nodeB);
            }
            if (!b.connections.contains(nodeA)) {
                b.connections.add(nodeA);
            }

            // Update Union-Find if both are within array bounds
            if (nodeA < parent.length && nodeB < parent.length) {
                int rootA = find(parent, nodeA);
                int rootB = find(parent, nodeB);
                if (rootA != rootB) {
                    union(parent, rank, rootA, rootB);
                }
            }
        }

        return result;
    }

    /**
     * Connect separate chains to form a single connected network.
     * At level 0 (asterisms), all stars must connect - chains are never removed.
     */
    private void connectChainsToRoot(List<NetworkNode> nodes, List<Segment3D> allSegments,
                                      double maxDistSq, double mergeDistSq, int level, int cellX, int cellY,
                                      int[] parent, int[] rank, List<Segment3D> previousLevelSegments) {
        // Find the root chain (largest chain or chain with lowest elevation point)
        int rootChain = findRootChain(nodes, parent);

        // Keep connecting until all nodes are in the same chain or no more connections possible
        boolean connectionMade;
        // At level 0, we may need more iterations since we can't remove chains
        int maxIterations = level == 0 ? nodes.size() * 5 : nodes.size() * 2;
        int iterations = 0;

        // For level 0, try with progressively larger distances if needed
        double currentMaxDistSq = maxDistSq;
        int distanceExpansions = 0;
        final int maxDistanceExpansions = 3;

        do {
            connectionMade = false;
            iterations++;

            // Find smallest non-root chain
            int smallestChain = findSmallestNonRootChain(nodes, parent, rootChain);
            if (smallestChain < 0) break;

            // Try to connect this chain to another chain
            List<Integer> chainNodes = getNodesInChain(nodes, parent, smallestChain);

            // Sort chain nodes by elevation (lowest first for escape path)
            chainNodes.sort((a, b) -> Double.compare(nodes.get(a).point.z, nodes.get(b).point.z));

            for (int nodeIdx : chainNodes) {
                int neighborIdx = createAndDefineSegment(nodes, allSegments, nodeIdx,
                                                          currentMaxDistSq, mergeDistSq, level, cellX, cellY,
                                                          parent, rank, previousLevelSegments,
                                                          false);  // isTrunk = false for chain connections
                if (neighborIdx >= 0) {
                    connectionMade = true;
                    // Update root chain
                    rootChain = findRootChain(nodes, parent);
                    break;
                }
            }

            // If chain couldn't connect
            if (!connectionMade) {
                if (level == 0 && distanceExpansions < maxDistanceExpansions) {
                    // At level 0, expand search distance and try again
                    distanceExpansions++;
                    currentMaxDistSq *= 2.0;  // Double the search distance
                    connectionMade = true;  // Force another iteration with expanded distance
                    LOGGER.debug("Level 0: Expanding search distance (expansion {})", distanceExpansions);
                } else if (level > 0) {
                    // At higher levels, remove chains that can't connect
                    for (int nodeIdx : chainNodes) {
                        nodes.get(nodeIdx).removed = true;
                    }
                } else {
                    // Level 0 but exhausted distance expansions - log warning
                    LOGGER.warn("Level 0: Chain with {} nodes could not connect after {} distance expansions",
                               chainNodes.size(), maxDistanceExpansions);
                    // Still don't remove at level 0 - just break to avoid infinite loop
                    break;
                }
            }

        } while (connectionMade && iterations < maxIterations);

        if (level == 0 && iterations >= maxIterations) {
            LOGGER.warn("Level 0: connectChainsToRoot reached max iterations ({})", maxIterations);
        }
    }

    /**
     * Find the root chain (chain with most nodes or lowest elevation).
     */
    private int findRootChain(List<NetworkNode> nodes, int[] parent) {
        Map<Integer, Integer> chainSizes = new HashMap<>();

        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).removed) continue;
            if (i >= parent.length) continue;
            int root = find(parent, i);
            chainSizes.merge(root, 1, Integer::sum);
        }

        int largestChain = -1;
        int largestSize = 0;
        for (Map.Entry<Integer, Integer> entry : chainSizes.entrySet()) {
            if (entry.getValue() > largestSize) {
                largestSize = entry.getValue();
                largestChain = entry.getKey();
            }
        }

        return largestChain;
    }

    /**
     * Find smallest chain that isn't the root chain.
     */
    private int findSmallestNonRootChain(List<NetworkNode> nodes, int[] parent, int rootChain) {
        Map<Integer, Integer> chainSizes = new HashMap<>();

        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).removed) continue;
            if (i >= parent.length) continue;
            int root = find(parent, i);
            if (root == rootChain) continue;
            chainSizes.merge(root, 1, Integer::sum);
        }

        int smallestChain = -1;
        int smallestSize = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : chainSizes.entrySet()) {
            if (entry.getValue() < smallestSize) {
                smallestSize = entry.getValue();
                smallestChain = entry.getKey();
            }
        }

        return smallestChain;
    }

    /**
     * Get all node indices in a given chain.
     */
    private List<Integer> getNodesInChain(List<NetworkNode> nodes, int[] parent, int chainRoot) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).removed) continue;
            if (i >= parent.length) continue;
            if (find(parent, i) == chainRoot) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Shift all segment elevations down so minimum is at 0 (for level 0 asterisms).
     */
    private List<Segment3D> shiftElevationsToZero(List<Segment3D> segments) {
        if (segments.isEmpty()) return segments;

        // Find minimum elevation
        double minZ = Double.MAX_VALUE;
        for (Segment3D seg : segments) {
            minZ = Math.min(minZ, Math.min(seg.srt.z, seg.end.z));
        }

        // Shift all elevations
        List<Segment3D> result = new ArrayList<>();
        for (Segment3D seg : segments) {
            Point3D newSrt = new Point3D(seg.srt.x, seg.srt.y, seg.srt.z - minZ);
            Point3D newEnd = new Point3D(seg.end.x, seg.end.y, seg.end.z - minZ);
            result.add(new Segment3D(newSrt, newEnd, seg.level, seg.tangentSrt, seg.tangentEnd));
        }

        return result;
    }

    /**
     * Clean and merge points that are within merge distance of each other.
     */
    private List<Point3D> cleanAndMergePoints(List<Point3D> points, double mergeDistance) {
        // Use shared merge function, prioritizing lower elevation points
        return mergePointsByDistance(points, mergeDistance, true);
    }

    /**
     * Remove points that are within merge distance of any lower-level segment.
     */
    private List<Point3D> removePointsNearSegments(List<Point3D> points, List<Segment3D> segments, double mergeDistance) {
        List<Point3D> result = new ArrayList<>();
        double mergeDistSq = mergeDistance * mergeDistance;

        for (Point3D point : points) {
            boolean tooClose = false;
            Point2D p2d = point.projectZ();

            for (Segment3D seg : segments) {
                // Find closest point on segment using linear interpolation
                Point2D segA = seg.srt.projectZ();
                Point2D segB = seg.end.projectZ();
                double distSq = pointToSegmentDistanceSquared(p2d, segA, segB);

                if (distSq < mergeDistSq) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                result.add(point);
            }
        }

        return result;
    }

    /**
     * Calculate squared distance from point to line segment.
     */
    private double pointToSegmentDistanceSquared(Point2D point, Point2D segA, Point2D segB) {
        double dx = segB.x - segA.x;
        double dy = segB.y - segA.y;
        double lengthSq = dx * dx + dy * dy;

        if (lengthSq < MathUtils.EPSILON) {
            // Degenerate segment (point)
            return point.distanceSquaredTo(segA);
        }

        // Project point onto line, clamped to segment
        double t = Math.max(0, Math.min(1,
            ((point.x - segA.x) * dx + (point.y - segA.y) * dy) / lengthSq));

        double projX = segA.x + t * dx;
        double projY = segA.y + t * dy;

        double pdx = point.x - projX;
        double pdy = point.y - projY;
        return pdx * pdx + pdy * pdy;
    }

    /**
     * Probabilistically remove points based on:
     * 1. branchesSampler value (1 - branchesSampler(x,y))
     * 2. Distance from lower-level segments (points farther away are more likely to be removed)
     *
     * This helps reduce the square pattern at cell edges by removing points that are
     * far from existing segments.
     *
     * @param points Points to filter
     * @param cellX Cell X coordinate
     * @param cellY Cell Y coordinate
     * @param previousLevelSegments Segments from the previous level (for distance-based removal)
     * @param gridSpacing Grid spacing for this level (used to calculate distance threshold)
     * @return Filtered list of points
     */
    private List<Point3D> probabilisticallyRemovePoints(List<Point3D> points, int cellX, int cellY,
                                                         List<Segment3D> previousLevelSegments,
                                                         double gridSpacing) {
        List<Point3D> result = new ArrayList<>();

        // Distance threshold for maximum removal probability (points beyond this are very likely removed)
        double maxDistanceForRemoval = gridSpacing * 3.0;
        double maxDistanceForRemovalSq = maxDistanceForRemoval * maxDistanceForRemoval;

        for (Point3D point : points) {
            // Base removal probability from branchesSampler
            double baseRemovalProbability = 0.0;
            if (branchesSampler != null) {
                double branchProbability = branchesSampler.getSample(salt, point.x * gridsize, point.y * gridsize);
                branchProbability = Math.max(0, Math.min(1, branchProbability / 8.0));  // Normalize to [0,1]
                baseRemovalProbability = 1.0 - branchProbability;
            }

            // Distance-based removal probability
            // Points farther from previous level segments have higher removal probability
            double distanceRemovalProbability = 0.0;
            if (!previousLevelSegments.isEmpty()) {
                double minDistSq = Double.MAX_VALUE;
                Point2D point2D = point.projectZ();

                for (Segment3D seg : previousLevelSegments) {
                    double distSq = pointToSegmentDistanceSquared(point2D, seg.srt.projectZ(), seg.end.projectZ());
                    if (distSq < minDistSq) {
                        minDistSq = distSq;
                    }
                }

                // Scale distance to [0, 1] based on maxDistanceForRemoval
                // Points at distance 0 have 0 additional removal probability
                // Points at maxDistanceForRemoval have high removal probability
                double normalizedDist = Math.min(1.0, minDistSq / maxDistanceForRemovalSq);
                distanceRemovalProbability = normalizedDist * 0.8;  // Max 80% from distance
            }

            // Combined removal probability (max of both factors)
            double totalRemovalProbability = Math.max(baseRemovalProbability, distanceRemovalProbability);

            // Deterministic removal based on point position
            Random rng = initRandomGenerator((int)(point.x * 1000), (int)(point.y * 1000), 42);
            if (rng.nextDouble() >= totalRemovalProbability) {
                result.add(point);
            }
        }
        return result;
    }

    /**
     * Probabilistically remove points based on branchesSampler only (legacy overload).
     */
    private List<Point3D> probabilisticallyRemovePoints(List<Point3D> points, int cellX, int cellY) {
        return probabilisticallyRemovePoints(points, cellX, cellY, new ArrayList<>(), 1.0);
    }

    /**
     * Connect network nodes using the specified algorithm.
     * Processes from highest to lowest elevation, applying connection rules.
     */
    private void connectNetworkNodes(List<NetworkNode> nodes, double maxSegmentDistance,
                                     int level, List<Segment3D> previousLevelSegments) {
        int n = nodes.size();
        if (n <= 1) return;

        double maxDistSq = maxSegmentDistance * maxSegmentDistance;

        // Union-Find for tracking connectivity
        int[] parent = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        // Sort nodes by elevation (highest first for Phase A)
        List<Integer> nodesByElevation = new ArrayList<>();
        for (int i = 0; i < n; i++) nodesByElevation.add(i);
        nodesByElevation.sort((a, b) -> Double.compare(nodes.get(b).point.z, nodes.get(a).point.z));

        // Phase A: Connect each node from highest to lowest, ensuring all have at least one connection
        for (int idx : nodesByElevation) {
            if (nodes.get(idx).removed) continue;

            NetworkNode node = nodes.get(idx);
            if (!node.connections.isEmpty()) continue;  // Already connected

            // Find neighbor with lowest slope within max distance
            int bestNeighbor = findBestNeighbor(nodes, idx, maxDistSq, level, parent);

            if (bestNeighbor >= 0) {
                // Check if connection would create uphill flow
                double slopeToNeighbor = calculateSlope(node.point, nodes.get(bestNeighbor).point);

                if (level > 0 && slopeToNeighbor > lowestSlopeCutoff) {
                    // Cannot achieve path to main asterism - remove this node and connected nodes
                    removeNodeAndDownstream(nodes, idx, parent);
                    continue;
                }

                // Make connection using connection rules
                makeConnection(nodes, idx, bestNeighbor, parent, rank, maxDistSq);

                // If slope is positive, reduce elevation ratiometrically
                if (slopeToNeighbor > 0) {
                    reduceElevationForUpwardFlow(nodes, idx, bestNeighbor);
                }
            }
        }

        // Phase B: Process nodes with 2 or fewer connections (lowest to highest)
        List<Integer> nodesLowToHigh = new ArrayList<>(nodesByElevation);
        java.util.Collections.reverse(nodesLowToHigh);

        for (int idx : nodesLowToHigh) {
            if (nodes.get(idx).removed) continue;

            NetworkNode node = nodes.get(idx);
            if (node.connections.size() > 2) continue;

            int bestNeighbor = findBestNeighbor(nodes, idx, maxDistSq, level, parent);
            if (bestNeighbor >= 0 && !node.connections.contains(bestNeighbor)) {
                double slopeToNeighbor = calculateSlope(node.point, nodes.get(bestNeighbor).point);

                if (level > 0 && slopeToNeighbor > lowestSlopeCutoff && node.connections.isEmpty()) {
                    removeNodeAndDownstream(nodes, idx, parent);
                    continue;
                }

                makeConnection(nodes, idx, bestNeighbor, parent, rank, maxDistSq);

                if (slopeToNeighbor > 0) {
                    reduceElevationForUpwardFlow(nodes, idx, bestNeighbor);
                }
            }
        }

        // Ensure all nodes are connected using Union-Find (form single tree)
        ensureFullConnectivity(nodes, parent, rank, maxDistSq);
    }

    /**
     * Find the best neighbor for connection (lowest slope within max distance).
     */
    private int findBestNeighbor(List<NetworkNode> nodes, int nodeIdx, double maxDistSq,
                                  int level, int[] parent) {
        NetworkNode node = nodes.get(nodeIdx);
        Point2D nodePos = node.point.projectZ();

        double bestSlope = Double.MAX_VALUE;
        int bestNeighbor = -1;

        for (int i = 0; i < nodes.size(); i++) {
            if (i == nodeIdx || nodes.get(i).removed) continue;
            if (node.connections.contains(i)) continue;  // Already connected

            NetworkNode candidate = nodes.get(i);
            Point2D candidatePos = candidate.point.projectZ();
            double distSq = nodePos.distanceSquaredTo(candidatePos);

            if (distSq > maxDistSq || distSq < MathUtils.EPSILON) continue;

            double slope = Math.abs(calculateSlope(node.point, candidate.point));
            if (slope < bestSlope) {
                bestSlope = slope;
                bestNeighbor = i;
            }
        }

        return bestNeighbor;
    }

    /**
     * Calculate slope (elevation change / horizontal distance).
     */
    private double calculateSlope(Point3D from, Point3D to) {
        double dx = to.x - from.x;
        double dy = to.y - from.y;
        double horizontalDist = Math.sqrt(dx * dx + dy * dy);
        if (horizontalDist < MathUtils.EPSILON) return 0;
        return (to.z - from.z) / horizontalDist;
    }

    /**
     * Make a connection between two nodes following the connection rules.
     */
    private void makeConnection(List<NetworkNode> nodes, int idx1, int idx2,
                                 int[] parent, int[] rank, double maxDistSq) {
        NetworkNode node1 = nodes.get(idx1);
        NetworkNode node2 = nodes.get(idx2);

        // Rule ii: If neighbor has 3 connections, subdivide closest spline
        if (node2.connections.size() >= 3) {
            // This should be rare - log for debugging
            LOGGER.debug("Node {} has 3+ connections, subdivision needed (rare case)", idx2);
            // For now, just add the connection anyway
        }

        // Rule iii: If neighbor has line passing through, mark for branch merge
        if (node2.connections.size() == 2) {
            node1.isBranchPoint = true;
            node1.branchIntoNode = idx2;
        }

        // Rule iv/v: Handle tangent setting based on neighbor's connections
        // (Tangent computation happens in computeNetworkTangents)

        // Add bidirectional connection
        if (!node1.connections.contains(idx2)) {
            node1.connections.add(idx2);
        }
        if (!node2.connections.contains(idx1)) {
            node2.connections.add(idx1);
        }

        // Update Union-Find
        int root1 = find(parent, idx1);
        int root2 = find(parent, idx2);
        if (root1 != root2) {
            union(parent, rank, root1, root2);
        }
    }

    /**
     * Remove a node and all its downstream connections.
     */
    private void removeNodeAndDownstream(List<NetworkNode> nodes, int idx, int[] parent) {
        NetworkNode node = nodes.get(idx);
        node.removed = true;

        // Remove connections to this node from other nodes
        for (int connIdx : node.connections) {
            nodes.get(connIdx).connections.remove(Integer.valueOf(idx));
        }
        node.connections.clear();
    }

    /**
     * Reduce elevation for nodes to prevent upward flow.
     */
    private void reduceElevationForUpwardFlow(List<NetworkNode> nodes, int sourceIdx, int targetIdx) {
        NetworkNode source = nodes.get(sourceIdx);
        NetworkNode target = nodes.get(targetIdx);

        // If source is higher than target, reduce source and downstream
        if (source.point.z > target.point.z) {
            double targetZ = target.point.z - MathUtils.EPSILON;
            Point3D adjustedPoint = new Point3D(source.point.x, source.point.y, targetZ,
                                                 source.point.slopeX, source.point.slopeY);
            // Note: NetworkNode.point is final, so we can't directly modify it
            // In a full implementation, we'd need to track elevation adjustments separately
        }
    }

    /**
     * Ensure all nodes are connected into a single tree.
     */
    private void ensureFullConnectivity(List<NetworkNode> nodes, int[] parent, int[] rank, double maxDistSq) {
        int n = nodes.size();

        // Build edges sorted by distance
        List<Edge> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nodes.get(i).removed) continue;
            for (int j = i + 1; j < n; j++) {
                if (nodes.get(j).removed) continue;
                double distSq = nodes.get(i).point.projectZ().distanceSquaredTo(nodes.get(j).point.projectZ());
                if (distSq <= maxDistSq) {
                    edges.add(new Edge(i, j, distSq, nodes.get(i).point, nodes.get(j).point));
                }
            }
        }
        edges.sort(Comparator.comparingDouble(e -> e.distance));

        // Use Kruskal's to connect remaining components
        for (Edge edge : edges) {
            if (nodes.get(edge.idx1).removed || nodes.get(edge.idx2).removed) continue;

            int root1 = find(parent, edge.idx1);
            int root2 = find(parent, edge.idx2);

            if (root1 != root2) {
                NetworkNode node1 = nodes.get(edge.idx1);
                NetworkNode node2 = nodes.get(edge.idx2);

                if (!node1.connections.contains(edge.idx2)) {
                    node1.connections.add(edge.idx2);
                }
                if (!node2.connections.contains(edge.idx1)) {
                    node2.connections.add(edge.idx1);
                }

                union(parent, rank, root1, root2);
            }
        }
    }

    /**
     * Compute tangents for all network nodes.
     */
    private void computeNetworkTangents(List<NetworkNode> nodes, int cellX, int cellY) {
        for (NetworkNode node : nodes) {
            if (node.removed || node.isBranchPoint) continue;  // Branch points handled separately
            if (node.connections.isEmpty()) continue;

            // Determine twist angle: random ±70° * max((1 - slope/SlopeWhenStraight), 0)
            double nodeSlope = node.point.hasSlope() ? node.point.getSlope() : 0;
            double twistFactor = Math.max(0, 1.0 - nodeSlope / slopeWhenStraight);
            Random rng = initRandomGenerator((int)(node.point.x * 1000), (int)(node.point.y * 1000), 73);
            double twistAngle = (rng.nextDouble() * 2 - 1) * Math.toRadians(70) * twistFactor;

            Vec2D nominalTangent;

            if (node.connections.size() >= 2) {
                // Connected to 2+ points: find non-branch connections
                List<Integer> nonBranchConnections = new ArrayList<>();
                for (int connIdx : node.connections) {
                    NetworkNode connNode = nodes.get(connIdx);
                    if (!connNode.isBranchPoint || connNode.branchIntoNode != node.index) {
                        nonBranchConnections.add(connIdx);
                    }
                }

                if (nonBranchConnections.size() >= 2) {
                    // Nominal tangent = direction between the two connected points
                    Point3D p1 = nodes.get(nonBranchConnections.get(0)).point;
                    Point3D p2 = nodes.get(nonBranchConnections.get(1)).point;
                    nominalTangent = new Vec2D(p1.projectZ(), p2.projectZ());
                    if (nominalTangent.lengthSquared() > MathUtils.EPSILON) {
                        nominalTangent = nominalTangent.normalize();
                    } else {
                        nominalTangent = new Vec2D(1, 0);
                    }
                } else {
                    // Use point's own tangent
                    nominalTangent = getTangentFromPoint(node.point);
                }
            } else {
                // Only one connection: use point's own tangent
                nominalTangent = getTangentFromPoint(node.point);
            }

            // Apply twist rotation
            node.tangent = rotateTangent(nominalTangent, twistAngle);
        }
    }

    /**
     * Get tangent direction from a Point3D's slope data.
     */
    private Vec2D getTangentFromPoint(Point3D point) {
        if (point.hasSlope()) {
            double tangentAngle = point.getTangent();
            return new Vec2D(Math.cos(tangentAngle), Math.sin(tangentAngle));
        }
        return new Vec2D(1, 0);  // Default
    }

    /**
     * Rotate a tangent vector by an angle.
     */
    private Vec2D rotateTangent(Vec2D tangent, double angle) {
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        return new Vec2D(
            tangent.x * cos - tangent.y * sin,
            tangent.x * sin + tangent.y * cos
        );
    }

    /**
     * Compute tangents for branch points.
     */
    private void computeBranchTangents(List<NetworkNode> nodes, int cellX, int cellY) {
        for (NetworkNode node : nodes) {
            if (node.removed || !node.isBranchPoint) continue;
            if (node.branchIntoNode < 0) continue;

            NetworkNode targetNode = nodes.get(node.branchIntoNode);
            if (targetNode.tangent == null) continue;

            // Branch tangent: random 110-170° from target's tangent
            Random rng = initRandomGenerator((int)(node.point.x * 1000), (int)(node.point.y * 1000), 137);
            double branchAngle = Math.toRadians(110 + rng.nextDouble() * 60);  // 110 to 170 degrees

            // Determine which side based on relative position
            Vec2D toNode = new Vec2D(targetNode.point.projectZ(), node.point.projectZ());
            double crossProduct = targetNode.tangent.x * toNode.y - targetNode.tangent.y * toNode.x;
            if (crossProduct < 0) {
                branchAngle = -branchAngle;
            }

            node.tangent = rotateTangent(targetNode.tangent, branchAngle);
        }
    }

    /**
     * Build Segment3D list from network nodes.
     */
    private List<Segment3D> buildSegmentsFromNetwork(List<NetworkNode> nodes, int level) {
        List<Segment3D> segments = new ArrayList<>();
        java.util.Set<Long> processedPairs = new java.util.HashSet<>();

        for (NetworkNode node : nodes) {
            if (node.removed) continue;

            for (int connIdx : node.connections) {
                // Avoid duplicate segments
                long pairKey = Math.min(node.index, connIdx) * 100000L + Math.max(node.index, connIdx);
                if (processedPairs.contains(pairKey)) continue;
                processedPairs.add(pairKey);

                NetworkNode connNode = nodes.get(connIdx);
                if (connNode.removed) continue;

                // Determine flow direction (a is start/higher, b is end/lower)
                Point3D pointA, pointB;
                Vec2D tangentA, tangentB;

                if (node.point.z >= connNode.point.z) {
                    pointA = node.point;
                    pointB = connNode.point;
                    tangentA = node.tangent;
                    tangentB = connNode.tangent;
                } else {
                    pointA = connNode.point;
                    pointB = node.point;
                    tangentA = connNode.tangent;
                    tangentB = node.tangent;
                }

                segments.add(new Segment3D(pointA, pointB, level, tangentA, tangentB));
            }
        }

        return segments;
    }

    /**
     * Subdivide and displace segments based on level.
     */
    private List<Segment3D> subdivideAndDisplaceSegments(List<Segment3D> segments, int level, int cellX, int cellY) {
        // Subdivision count per level (can be adjusted)
        int[] subdivisionsPerLevel = {2, 3, 4, 5, 6};  // Level 0-4
        int subdivisions = level < subdivisionsPerLevel.length ? subdivisionsPerLevel[level] : 6;

        List<Segment3D> result = new ArrayList<>();
        for (Segment3D seg : segments) {
            if (subdivisions <= 1) {
                result.add(seg);
            } else {
                // Subdivide using spline interpolation if tangents are available
                if (seg.hasTangents()) {
                    result.addAll(subdivideWithSpline(seg, subdivisions, cellX, cellY));
                } else {
                    // Simple linear subdivision
                    Segment3D[] subdivided = seg.subdivide(subdivisions);
                    for (Segment3D s : subdivided) {
                        result.add(s);
                    }
                }
            }
        }

        // Apply displacement to subdivided segments
        return displaceSubdividedSegments(result, level, cellX, cellY);
    }

    /**
     * Subdivide a segment using cubic Hermite spline interpolation.
     */
    private List<Segment3D> subdivideWithSpline(Segment3D seg, int subdivisions, int cellX, int cellY) {
        List<Segment3D> result = new ArrayList<>();
        Point3D prev = seg.srt;

        for (int i = 1; i <= subdivisions; i++) {
            double t = (double) i / subdivisions;
            Point3D next;

            if (i == subdivisions) {
                next = seg.end;
            } else {
                // Cubic Hermite spline interpolation
                double t2 = t * t;
                double t3 = t2 * t;
                double h00 = 2 * t3 - 3 * t2 + 1;
                double h10 = t3 - 2 * t2 + t;
                double h01 = -2 * t3 + 3 * t2;
                double h11 = t3 - t2;

                double segLength = seg.srt.projectZ().distanceTo(seg.end.projectZ());
                double tangentScale = segLength * tangentStrength;

                double x = h00 * seg.srt.x + h10 * (seg.tangentSrt != null ? seg.tangentSrt.x * tangentScale : 0)
                         + h01 * seg.end.x + h11 * (seg.tangentEnd != null ? seg.tangentEnd.x * tangentScale : 0);
                double y = h00 * seg.srt.y + h10 * (seg.tangentSrt != null ? seg.tangentSrt.y * tangentScale : 0)
                         + h01 * seg.end.y + h11 * (seg.tangentEnd != null ? seg.tangentEnd.y * tangentScale : 0);
                double z = MathUtils.lerp(seg.srt.z, seg.end.z, t);  // Linear interpolation for elevation

                next = new Point3D(x, y, z);
            }

            result.add(new Segment3D(prev, next, seg.level));
            prev = next;
        }

        return result;
    }

    /**
     * Apply small displacement to subdivided segments.
     */
    private List<Segment3D> displaceSubdividedSegments(List<Segment3D> segments, int level, int cellX, int cellY) {
        // Apply small random displacement for natural appearance
        // Displacement magnitude decreases with level
        double displacementMagnitude = 0.05 / Math.pow(2, level);

        List<Segment3D> result = new ArrayList<>();
        for (Segment3D seg : segments) {
            // Only displace interior points, not endpoints
            Random rng = initRandomGenerator((int)(seg.srt.x * 1000 + seg.srt.y * 1000), (int)(seg.end.x * 1000), level + 200);
            double dx = (rng.nextDouble() * 2 - 1) * displacementMagnitude;
            double dy = (rng.nextDouble() * 2 - 1) * displacementMagnitude;

            // Displace midpoint slightly
            Point3D midA = new Point3D(seg.srt.x + dx * 0.3, seg.srt.y + dy * 0.3, seg.srt.z);
            Point3D midB = new Point3D(seg.end.x + dx * 0.3, seg.end.y + dy * 0.3, seg.end.z);

            result.add(new Segment3D(midA, midB, seg.level, seg.tangentSrt, seg.tangentEnd));
        }

        return result;
    }

    // ========== Constellation Stitching ==========

    /**
     * Stitch constellations together by finding optimal connection points.
     *
     * Algorithm:
     * 1. For each pair of adjacent constellations, find two points within max segment distance
     *    with smallest absolute slope
     * 2. Set tangents based on connection type:
     *    - If connecting to end of line: use continuous tangent
     *    - If connecting to middle of line: use 20-80° offset from continuous tangent
     */
    /**
     * Stitch adjacent constellations together.
     * For each pair of adjacent constellations:
     * 1. Extract unique endpoints from segments
     * 2. Find up to 6 closest pairs between them
     * 3. Select the pair with lowest max elevation
     * 4. Create stitch segment with proper tangents based on connection type
     * 5. Subdivide the stitch segment using level 0 parameters
     */
    /**
     * Stitch adjacent constellations together using ConstellationInfo.
     */
    /**
     * Get the adjacency distance threshold for constellation center-to-center distance.
     * Adjacent constellations have centers within this distance (with small buffer for rounding).
     */
    private double getAdjacentConstellationThreshold() {
        double size = getConstellationSize();
        double buffer = size * 0.03;  // buffer for floating point tolerance

        switch (constellationShape) {
            case HEXAGON:
                // Hexagon neighbors are all at distance = size
                return size + buffer;
            case RHOMBUS:
                // Rhombus closest neighbors are at distance = size * sqrt(0.75) ≈ 0.866 * size
                // Same row neighbors are at distance = size
                return (size* 0.866) + buffer;
            case SQUARE:
            default:
                // Square diagonal neighbors are at distance = size * sqrt(2) ≈ 1.414 * size
                // Cardinal neighbors are at distance = size
                return size + buffer;  // Include diagonals
        }
    }

    private List<Segment3D> stitchConstellations(List<ConstellationInfo> constellations,
                                                  Map<Long, List<Segment3D>> constellationSegments) {
        List<Segment3D> stitchSegments = new ArrayList<>();

        if (constellations.size() < 2) {
            return stitchSegments;
        }

        double adjacencyThreshold = getAdjacentConstellationThreshold();
        double adjacencyThresholdSq = adjacencyThreshold * adjacencyThreshold;

        // Find all pairs of adjacent constellations (shared border)
        // Use a set of processed pair keys to avoid duplicates
        Set<Long> processedPairs = new HashSet<>();
        List<int[]> adjacentPairs = new ArrayList<>();  // [indexI, indexJ]

        for (int i = 0; i < constellations.size(); i++) {
            ConstellationInfo constI = constellations.get(i);

            for (int j = i + 1; j < constellations.size(); j++) {
                ConstellationInfo constJ = constellations.get(j);

                // Check center-to-center distance
                double dx = constI.centerX - constJ.centerX;
                double dy = constI.centerY - constJ.centerY;
                double distSq = dx * dx + dy * dy;

                if (distSq < adjacencyThresholdSq) {
                    // Create deterministic pair key (smaller key first)
                    long keyI = constI.getKey();
                    long keyJ = constJ.getKey();
                    long pairKey = (keyI < keyJ) ? (keyI ^ (keyJ << 1)) : (keyJ ^ (keyI << 1));

                    if (!processedPairs.contains(pairKey)) {
                        processedPairs.add(pairKey);
                        adjacentPairs.add(new int[]{i, j});
                    }
                }
            }
        }

        // Sort pairs deterministically by combined center coordinates for reproducible results
        adjacentPairs.sort((a, b) -> {
            ConstellationInfo aI = constellations.get(a[0]);
            ConstellationInfo aJ = constellations.get(a[1]);
            ConstellationInfo bI = constellations.get(b[0]);
            ConstellationInfo bJ = constellations.get(b[1]);

            // Compare by minimum X, then minimum Y of the pair
            double aMinX = Math.min(aI.centerX, aJ.centerX);
            double bMinX = Math.min(bI.centerX, bJ.centerX);
            if (Math.abs(aMinX - bMinX) > MathUtils.EPSILON) {
                return Double.compare(aMinX, bMinX);
            }
            double aMinY = Math.min(aI.centerY, aJ.centerY);
            double bMinY = Math.min(bI.centerY, bJ.centerY);
            return Double.compare(aMinY, bMinY);
        });

        // Process each adjacent pair
        for (int[] pair : adjacentPairs) {
            ConstellationInfo constI = constellations.get(pair[0]);
            ConstellationInfo constJ = constellations.get(pair[1]);

            long keyI = constI.getKey();
            long keyJ = constJ.getKey();

            List<Segment3D> segmentsI = constellationSegments.get(keyI);
            List<Segment3D> segmentsJ = constellationSegments.get(keyJ);

            if (segmentsI == null || segmentsI.isEmpty()) continue;
            if (segmentsJ == null || segmentsJ.isEmpty()) continue;

            // Extract unique endpoints from segments
            List<Point3D> endpointsI = extractUniqueEndpoints(segmentsI);
            List<Point3D> endpointsJ = extractUniqueEndpoints(segmentsJ);

            if (endpointsI.isEmpty() || endpointsJ.isEmpty()) continue;

            // Find up to 6 closest pairs between the two constellations
            List<double[]> candidatePairs = new ArrayList<>();  // [distSq, idxI, idxJ, maxElevation]

            for (int pi = 0; pi < endpointsI.size(); pi++) {
                Point3D pI = endpointsI.get(pi);
                for (int pj = 0; pj < endpointsJ.size(); pj++) {
                    Point3D pJ = endpointsJ.get(pj);
                    double distSq = pI.projectZ().distanceSquaredTo(pJ.projectZ());
                    if (distSq < MathUtils.EPSILON) continue;

                    double maxElevation = Math.max(pI.z, pJ.z);
                    candidatePairs.add(new double[]{distSq, pi, pj, maxElevation});
                }
            }

            // Sort by distance and take up to 6 closest
            candidatePairs.sort((a, b) -> Double.compare(a[0], b[0]));
            int pairsToConsider = Math.min(6, candidatePairs.size());

            if (pairsToConsider == 0) continue;

            // From the closest 6, select the one with lowest max elevation
            double[] bestPair = null;
            double lowestMaxElevation = Double.MAX_VALUE;

            for (int k = 0; k < pairsToConsider; k++) {
                double[] candidate = candidatePairs.get(k);
                if (candidate[3] < lowestMaxElevation) {
                    lowestMaxElevation = candidate[3];
                    bestPair = candidate;
                }
            }

            if (bestPair == null) continue;

            Point3D bestI = endpointsI.get((int) bestPair[1]);
            Point3D bestJ = endpointsJ.get((int) bestPair[2]);

            // Get tangent info including whether point is at start or end of its connected segment
            StitchTangentInfo tangentInfoI = getStitchTangentInfo(bestI, segmentsI);
            StitchTangentInfo tangentInfoJ = getStitchTangentInfo(bestJ, segmentsJ);

            // Order by elevation (srt is higher elevation)
            Point3D srtPoint, endPoint;
            StitchTangentInfo srtInfo, endInfo;
            if (bestI.z >= bestJ.z) {
                srtPoint = bestI;
                endPoint = bestJ;
                srtInfo = tangentInfoI;
                endInfo = tangentInfoJ;
            } else {
                srtPoint = bestJ;
                endPoint = bestI;
                srtInfo = tangentInfoJ;
                endInfo = tangentInfoI;
            }

            // Apply tangent alignment rules for leaf connections:
            // - If stitch point type MATCHES leaf type (both start or both end) → INVERT
            // - If stitch point type is OPPOSITE of leaf type → use directly
            Vec2D tangentSrt = srtInfo.tangent;
            Vec2D tangentEnd = endInfo.tangent;

            // For srtPoint: stitch uses it as START
            // If leaf also has it as START (isLeafStart=true), types match → invert
            if (srtInfo.isLeaf && srtInfo.isLeafStart && tangentSrt != null) {
                tangentSrt = new Vec2D(-tangentSrt.x, -tangentSrt.y);
            }
            // If leaf has it as END (isLeafStart=false), types are opposite → use directly (no change)

            // For endPoint: stitch uses it as END
            // If leaf also has it as END (isLeafStart=false), types match → invert
            if (endInfo.isLeaf && !endInfo.isLeafStart && tangentEnd != null) {
                tangentEnd = new Vec2D(-tangentEnd.x, -tangentEnd.y);
            }
            // If leaf has it as START (isLeafStart=true), types are opposite → use directly (no change)

            // Create the stitch segment at level 0
            Segment3D stitchSegment = new Segment3D(srtPoint, endPoint, 0, tangentSrt, tangentEnd);

            // Subdivide using same methodology as createAndDefineSegment for level 0
            double gridSpacing = getGridSpacingForLevel(1);  // Level 1 grid spacing for stitches.
            double mergeDistance = MERGE_POINT_SPACING * gridSpacing;
            double segmentLength = srtPoint.projectZ().distanceTo(endPoint.projectZ());
            int divisions = Math.max(1, (int) Math.floor(segmentLength / mergeDistance));

            if (divisions > 1) {
                // Subdivide using level 0 tangent magnitude
                List<Segment3D> subdivided = subdivideStitchSegment(stitchSegment, divisions, mergeDistance);
                stitchSegments.addAll(subdivided);
            } else {
                stitchSegments.add(stitchSegment);
            }
        }

        return stitchSegments;
    }

    /**
     * Extract unique endpoints from a list of segments.
     */
    private List<Point3D> extractUniqueEndpoints(List<Segment3D> segments) {
        List<Point3D> endpoints = new ArrayList<>();

        for (Segment3D seg : segments) {
            // Check if srt point is already in list
            boolean srtFound = false;
            for (Point3D existing : endpoints) {
                if (pointsMatch(existing, seg.srt)) {
                    srtFound = true;
                    break;
                }
            }
            if (!srtFound) {
                endpoints.add(seg.srt);
            }

            // Check if end point is already in list
            boolean endFound = false;
            for (Point3D existing : endpoints) {
                if (pointsMatch(existing, seg.end)) {
                    endFound = true;
                    break;
                }
            }
            if (!endFound) {
                endpoints.add(seg.end);
            }
        }

        return endpoints;
    }

    /**
     * Holds tangent information for a stitch connection point.
     */
    private static class StitchTangentInfo {
        final Vec2D tangent;       // The tangent at this point
        final boolean isLeaf;      // True if this is a leaf node (single connection)
        final boolean isLeafStart; // If leaf, true if point is the START of its segment

        StitchTangentInfo(Vec2D tangent, boolean isLeaf, boolean isLeafStart) {
            this.tangent = tangent;
            this.isLeaf = isLeaf;
            this.isLeafStart = isLeafStart;
        }
    }

    /**
     * Get tangent information for a stitch point based on its segment connections.
     * Returns the raw tangent from the connected segment plus metadata about the connection type.
     * The caller applies inversion based on whether stitch point type matches leaf type.
     */
    private StitchTangentInfo getStitchTangentInfo(Point3D point, List<Segment3D> segments) {
        // Find segments connected to this point
        List<Segment3D> connectedSegments = new ArrayList<>();
        List<Boolean> isStartPoint = new ArrayList<>();

        for (Segment3D seg : segments) {
            if (pointsMatch(seg.srt, point)) {
                connectedSegments.add(seg);
                isStartPoint.add(true);
            } else if (pointsMatch(seg.end, point)) {
                connectedSegments.add(seg);
                isStartPoint.add(false);
            }
        }

        if (connectedSegments.isEmpty()) {
            // No connections - use slope-based tangent, not a leaf
            return new StitchTangentInfo(getTangentFromPoint(point), false, false);
        }

        if (connectedSegments.size() == 1) {
            // Leaf node (single connection)
            Segment3D seg = connectedSegments.get(0);
            boolean isSrt = isStartPoint.get(0);

            // Get the tangent from the connected segment
            Vec2D existingTangent = isSrt ? seg.tangentSrt : seg.tangentEnd;

            if (existingTangent != null) {
                // Return raw tangent with leaf info - caller will apply inversion logic
                return new StitchTangentInfo(existingTangent, true, isSrt);
            }

            // Fallback: direction from other endpoint to this point
            Point3D other = isSrt ? seg.end : seg.srt;
            Vec2D tangent = new Vec2D(other.projectZ(), point.projectZ());
            if (tangent.lengthSquared() > MathUtils.EPSILON) {
                return new StitchTangentInfo(tangent.normalize(), true, isSrt);
            }
            return new StitchTangentInfo(getTangentFromPoint(point), true, isSrt);
        } else {
            // Middle node (2+ connections): not a leaf, use existing tangent
            Segment3D seg = connectedSegments.get(0);
            boolean isSrt = isStartPoint.get(0);
            Vec2D existingTangent = isSrt ? seg.tangentSrt : seg.tangentEnd;

            if (existingTangent != null) {
                return new StitchTangentInfo(existingTangent, false, false);
            }
        }

        // Default fallback
        return new StitchTangentInfo(getTangentFromPoint(point), false, false);
    }

    /**
     * Subdivide a stitch segment using Hermite spline interpolation.
     * Uses shared subdivision helpers - similar to subdivideAndAddPoints but without node tracking.
     */
    private List<Segment3D> subdivideStitchSegment(Segment3D segment, int divisions, double mergeDistance) {
        if (divisions <= 1) {
            return List.of(segment);
        }

        List<Segment3D> result = new ArrayList<>();
        double tangentScale = mergeDistance * TANGENT_MAGNITUDE_SCALE * tangentStrength;

        Point3D prev = segment.srt;
        Vec2D prevTangent = segment.tangentSrt;

        for (int i = 1; i <= divisions; i++) {
            double t = (double) i / divisions;
            Point3D next;
            Vec2D nextTangent;

            if (i == divisions) {
                // Last point is the segment endpoint
                next = segment.end;
                nextTangent = segment.tangentEnd;
            } else {
                // Use shared helper for Hermite interpolation
                SubdivisionPoint subdivPt = computeSubdivisionPoint(segment, t, tangentScale);
                next = subdivPt.position;
                nextTangent = (subdivPt.tangent != null) ? subdivPt.tangent : prevTangent;
            }

            // Create subdivision segment with appropriate point types
            PointType srtType = (i == 1) ? segment.srtType : PointType.KNOT;
            PointType endType = (i == divisions) ? segment.endType : PointType.KNOT;
            result.add(new Segment3D(prev, next, 0, prevTangent, nextTangent, srtType, endType));

            prev = next;
            prevTangent = nextTangent;
        }

        return result;
    }

    // ========== Legacy constellation methods (kept for compatibility) ==========

    /**
     * @deprecated Use getConstellationCenterForPoint instead
     */
    private Cell getConstellation(int level1CellX, int level1CellY) {
        Point2D center = getConstellationCenterForPoint(level1CellX + 0.5, level1CellY + 0.5);
        // Return quantized center for backwards compatibility
        int qx = (int) Math.round(center.x * 100);
        int qy = (int) Math.round(center.y * 100);
        return new Cell(qx, qy, 0);
    }

    /**
     * @deprecated Use findClosestConstellations instead
     */
    private java.util.Set<Long> getRequiredConstellations(Cell queryCell1) {
        List<ConstellationInfo> closest = findClosestConstellations(queryCell1);
        java.util.Set<Long> result = new java.util.HashSet<>();
        for (ConstellationInfo c : closest) {
            result.add(c.getKey());
        }
        return result;
    }

    /**
     * @deprecated Use generateConstellationStars(ConstellationInfo) instead
     */
    private Map<Long, Point3D> generateConstellationStarsLegacy(int cellX, int cellY) {
        Point2D center = getConstellationCenterForPoint(cellX + 0.5, cellY + 0.5);
        ConstellationInfo info = getConstellationInfoFromCenter(center.x, center.y);
        List<Point3D> stars = generateConstellationStars(info);
        Map<Long, Point3D> result = new HashMap<>();
        for (int i = 0; i < stars.size(); i++) {
            Point3D star = stars.get(i);
            result.put(packKey((int)(star.x * 100), (int)(star.y * 100)), star);
        }
        return result;
    }

    /**
     * @deprecated Use findStarInCelllLowestGrid instead
     */
    private Point3D findStarInCell(int cellX, int cellY) {
        return findStarInCelllLowestGrid(cellX, cellY);
    }

    /**
     * Build a tree structure (asterism) connecting all stars within a constellation.
     * Algorithm:
     * A. Connect each star to its closest neighbor (by 2D distance squared)
     * B. While not all stars form a single tree, connect disconnected components
     *    starting from lowest elevation stars
     */
    private List<Segment3D> buildTreeWithinConstellation(Map<Long, Point3D> stars, int constX, int constY) {
        if (stars.size() <= 1) return new ArrayList<>();

        List<Point3D> starList = new ArrayList<>(stars.values());
        int n = starList.size();

        // Build all edges sorted by distance squared (favors shorter connections more strongly)
        List<Edge> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Point3D star1 = starList.get(i);
                Point3D star2 = starList.get(j);
                double distSq = star1.projectZ().distanceSquaredTo(star2.projectZ());
                edges.add(new Edge(i, j, distSq, star1, star2));
            }
        }
        edges.sort(Comparator.comparingDouble(e -> e.distance));

        // Phase A: Connect each star to its closest neighbor
        List<Segment3D> asterismSegments = new ArrayList<>();
        int[] parent = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }

        // For each star, find and add its closest neighbor connection (by distance squared)
        boolean[] hasConnection = new boolean[n];
        for (int i = 0; i < n; i++) {
            double minDistSq = Double.MAX_VALUE;
            int closest = -1;
            Point3D current = starList.get(i);

            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                Point3D other = starList.get(j);
                double distSq = current.projectZ().distanceSquaredTo(other.projectZ());
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    closest = j;
                }
            }

            if (closest >= 0) {
                int rootI = find(parent, i);
                int rootJ = find(parent, closest);

                // Add asterism segment (avoid exact duplicates)
                if (rootI != rootJ || !hasConnection[i]) {
                    asterismSegments.add(new Segment3D(current, starList.get(closest), 0));
                    hasConnection[i] = true;
                    hasConnection[closest] = true;
                    if (rootI != rootJ) {
                        union(parent, rank, rootI, rootJ);
                    }
                }
            }
        }

        // Phase B: Ensure all stars are connected (form a single tree)
        // Sort stars by elevation for deterministic tie-breaking
        List<Integer> starsByElevation = new ArrayList<>();
        for (int i = 0; i < n; i++) starsByElevation.add(i);
        starsByElevation.sort(Comparator.comparingDouble(i -> starList.get(i).z));

        // Use Kruskal's to connect any remaining disconnected star components
        for (Edge edge : edges) {
            int rootA = find(parent, edge.idx1);
            int rootB = find(parent, edge.idx2);

            if (rootA != rootB) {
                asterismSegments.add(new Segment3D(edge.p1, edge.p2, 0));
                union(parent, rank, rootA, rootB);
            }
        }

        return asterismSegments;
    }

    /**
     * Stitch adjacent constellations together at their boundaries.
     * For each pair of adjacent constellations, find the boundary star
     * with the lowest elevation on each side and connect them.
     */
    private List<Segment3D> stitchConstellations(java.util.Set<Long> constellations,
                                                  Map<Long, Map<Long, Point3D>> constellationStars) {
        List<Segment3D> stitchSegments = new ArrayList<>();
        java.util.Set<Long> processedPairs = new java.util.HashSet<>();

        for (long constKey : constellations) {
            int constX = unpackX(constKey);
            int constY = unpackY(constKey);

            // Check right neighbor (+x direction)
            long rightKey = packKey(constX + 1, constY);
            if (constellations.contains(rightKey) && !processedPairs.contains(packKey(Math.min(constX, constX+1), Math.max(constX, constX+1) * 10000 + constY))) {
                processedPairs.add(packKey(Math.min(constX, constX+1), Math.max(constX, constX+1) * 10000 + constY));
                Segment3D stitch = stitchConstellationsVertical(constX, constY, constellationStars.get(constKey), constellationStars.get(rightKey));
                if (stitch != null) stitchSegments.add(stitch);
            }

            // Check bottom neighbor (+y direction)
            long bottomKey = packKey(constX, constY + 1);
            if (constellations.contains(bottomKey) && !processedPairs.contains(packKey(constX, Math.min(constY, constY+1) * 10000 + Math.max(constY, constY+1)))) {
                processedPairs.add(packKey(constX, Math.min(constY, constY+1) * 10000 + Math.max(constY, constY+1)));
                Segment3D stitch = stitchConstellationsHorizontal(constX, constY, constellationStars.get(constKey), constellationStars.get(bottomKey));
                if (stitch != null) stitchSegments.add(stitch);
            }
        }

        return stitchSegments;
    }

    /**
     * Stitch two constellations at a vertical boundary (between constX and constX+1).
     * Finds the lowest elevation star on each side of the boundary and connects them.
     */
    private Segment3D stitchConstellationsVertical(int constX, int constY,
                                                    Map<Long, Point3D> leftStars,
                                                    Map<Long, Point3D> rightStars) {
        // Right edge of left constellation: x = (constX + 1) * ConstellationScale - 1
        // Left edge of right constellation: x = (constX + 1) * ConstellationScale
        int leftEdgeX = (constX + 1) * ConstellationScale - 1;
        int rightEdgeX = (constX + 1) * ConstellationScale;
        int startY = constY * ConstellationScale;

        Point3D lowestLeftStar = null;
        Point3D lowestRightStar = null;
        double lowestLeftElev = Double.MAX_VALUE;
        double lowestRightElev = Double.MAX_VALUE;

        for (int i = 0; i < ConstellationScale; i++) {
            int y = startY + i;

            Point3D leftStar = leftStars.get(packKey(leftEdgeX, y));
            if (leftStar != null && leftStar.z < lowestLeftElev) {
                lowestLeftElev = leftStar.z;
                lowestLeftStar = leftStar;
            }

            Point3D rightStar = rightStars.get(packKey(rightEdgeX, y));
            if (rightStar != null && rightStar.z < lowestRightElev) {
                lowestRightElev = rightStar.z;
                lowestRightStar = rightStar;
            }
        }

        if (lowestLeftStar != null && lowestRightStar != null) {
            return new Segment3D(lowestLeftStar, lowestRightStar, 0);
        }
        return null;
    }

    /**
     * Stitch two constellations at a horizontal boundary (between constY and constY+1).
     */
    private Segment3D stitchConstellationsHorizontal(int constX, int constY,
                                                      Map<Long, Point3D> topStars,
                                                      Map<Long, Point3D> bottomStars) {
        int topEdgeY = (constY + 1) * ConstellationScale - 1;
        int bottomEdgeY = (constY + 1) * ConstellationScale;
        int startX = constX * ConstellationScale;

        Point3D lowestTopStar = null;
        Point3D lowestBottomStar = null;
        double lowestTopElev = Double.MAX_VALUE;
        double lowestBottomElev = Double.MAX_VALUE;

        for (int i = 0; i < ConstellationScale; i++) {
            int x = startX + i;

            Point3D topStar = topStars.get(packKey(x, topEdgeY));
            if (topStar != null && topStar.z < lowestTopElev) {
                lowestTopElev = topStar.z;
                lowestTopStar = topStar;
            }

            Point3D bottomStar = bottomStars.get(packKey(x, bottomEdgeY));
            if (bottomStar != null && bottomStar.z < lowestBottomElev) {
                lowestBottomElev = bottomStar.z;
                lowestBottomStar = bottomStar;
            }
        }

        if (lowestTopStar != null && lowestBottomStar != null) {
            return new Segment3D(lowestTopStar, lowestBottomStar, 0);
        }
        return null;
    }

    /**
     * Prune asterism segments to only keep those relevant to the query cell and its neighbors.
     * Keeps segments where at least one endpoint is within the 3x3 region around query cell.
     */
    private List<Segment3D> pruneToQueryRegion(List<Segment3D> segments, Cell queryCell1) {
        double minX = queryCell1.x - 1;
        double maxX = queryCell1.x + 2;
        double minY = queryCell1.y - 1;
        double maxY = queryCell1.y + 2;

        return segments.stream()
            .filter(seg -> {
                boolean aInside = isPointInBounds(seg.srt, minX, minY, maxX, maxY);
                boolean bInside = isPointInBounds(seg.end, minX, minY, maxX, maxY);
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
     * Pre-computed tangent information for spline control points at a star/node.
     * Stores tangent vectors for each connected neighbor, computed deterministically.
     * Used for smooth asterism segment curves at star junctions.
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
     * Compute tangent information for all stars/nodes in a segment list.
     * Tangents are computed deterministically based on star positions and connections,
     * independent of segment list order. Used for smooth asterism curves.
     */
    private Map<Long, NodeTangentInfo> computeNodeTangents(List<Segment3D> segments) {
        // Step 1: Build connectivity map: node -> list of connected nodes
        Map<Long, Point3D> nodePoints = new HashMap<>();
        Map<Long, List<Point3D>> connectivity = new HashMap<>();

        for (Segment3D seg : segments) {
            long keyA = pointHash(seg.srt);
            long keyB = pointHash(seg.end);

            nodePoints.putIfAbsent(keyA, seg.srt);
            nodePoints.putIfAbsent(keyB, seg.end);

            connectivity.computeIfAbsent(keyA, k -> new ArrayList<>()).add(seg.end);
            connectivity.computeIfAbsent(keyB, k -> new ArrayList<>()).add(seg.srt);
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

                    // For 2 connections, use the flow-through direction for smooth curves
                    // For 3+ connections, rotate tangent to avoid overlap with neighbors
                    Vec2D tangent;
                    if (n == 2) {
                        // Two connections: tangent follows flow direction directly
                        // For smooth curves, the tangent should be along the path from
                        // opposite → node → neighbor (the "flow" direction)
                        Point3D opposite = neighbors.get(1 - i);
                        Vec2D flowDir = new Vec2D(opposite.projectZ(), neighbor.projectZ());
                        if (flowDir.lengthSquared() > MathUtils.EPSILON) {
                            tangent = flowDir.normalize();
                        } else {
                            tangent = toNeighbor;
                        }
                    } else {
                        // 3+ connections: identify the "flow pair" (most opposite neighbors)
                        // and give them continuous tangents, spread others

                        // Find the most opposite pair (closest to 180° apart)
                        int flowIdx1 = -1, flowIdx2 = -1;
                        double maxOpposition = 0;
                        for (int a = 0; a < n; a++) {
                            for (int b = a + 1; b < n; b++) {
                                double angleA = Math.atan2(neighbors.get(a).y - node.y, neighbors.get(a).x - node.x);
                                double angleB = Math.atan2(neighbors.get(b).y - node.y, neighbors.get(b).x - node.x);
                                double opposition = Math.abs(normalizeAngle(angleA - angleB));
                                if (opposition > maxOpposition) {
                                    maxOpposition = opposition;
                                    flowIdx1 = a;
                                    flowIdx2 = b;
                                }
                            }
                        }

                        // Check if current neighbor is part of the flow pair
                        boolean isFlowPair = (i == flowIdx1 || i == flowIdx2) && maxOpposition > Math.PI * 0.6;

                        if (isFlowPair) {
                            // Flow pair: use opposite tangent method (like 2-connection case)
                            int oppositeIdx = (i == flowIdx1) ? flowIdx2 : flowIdx1;
                            Point3D opposite = neighbors.get(oppositeIdx);
                            Vec2D fromOpposite = new Vec2D(opposite.projectZ(), node.projectZ());
                            if (fromOpposite.lengthSquared() > MathUtils.EPSILON) {
                                fromOpposite = fromOpposite.normalize();
                                tangent = blendTangent(toNeighbor, fromOpposite, tangentAngle);
                            } else {
                                tangent = toNeighbor;
                            }
                        } else {
                            // Non-flow connection: rotate tangent to create separation
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
     * Remove asterism segments that don't connect to the inner 3x3 cell region.
     */
    private List<Segment3D> pruneDisconnectedSegments(List<Segment3D> segments,
                                                       double minX, double minY,
                                                       double maxX, double maxY) {
        return segments.stream()
            .filter(seg -> {
                // Keep if either endpoint is within inner region
                boolean aInside = isPointInBounds(seg.srt, minX, minY, maxX, maxY);
                boolean bInside = isPointInBounds(seg.end, minX, minY, maxX, maxY);
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
                Point2D a = seg.srt.projectZ();
                Point2D b = seg.end.projectZ();

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
            Point3D prev = seg.srt;
            for (int j = 1; j <= numDivisions; j++) {
                double t = (double) j / numDivisions;
                Point3D next = (j == numDivisions) ? seg.end : Point3D.lerp(seg.srt, seg.end, t);
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
            Point3D p1 = seg.srt;
            Point3D p2 = seg.end;
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
        Vec2D tangent = null;

        // Try to get pre-computed tangent
        if (tangentInfo != null && !tangentInfo.tangents.isEmpty()) {
            long otherKey = pointHash(other);
            tangent = tangentInfo.tangents.get(otherKey);
        }

        // Fallback: use segment direction as tangent if lookup failed
        if (tangent == null) {
            Vec2D segDir = new Vec2D(node.projectZ(), other.projectZ());
            if (segDir.lengthSquared() > MathUtils.EPSILON) {
                tangent = segDir.normalize();
            } else {
                // Degenerate case - return offset point in arbitrary direction
                return new Point3D(node.x + 0.01, node.y, node.z);
            }
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
        // Quantize to ~0.001 precision with rounding for reliable matching
        // Use Math.round instead of cast to handle floating point edge cases
        long hx = Math.round(p.x * 1000);
        long hy = Math.round(p.y * 1000);
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
            Vec2D dir = new Vec2D(seg.srt.projectZ(), seg.end.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) {
                result.add(seg);
                continue;
            }

            Vec2D perp = dir.rotateCCW90().normalize();

            // Use both endpoints for seed to ensure determinism based on segment identity
            int seedX = (int)((seg.srt.x + seg.end.x) * 50);
            int seedY = (int)((seg.srt.y + seg.end.y) * 50);
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
            result.add(new Segment3D(seg.srt, displacedMid, seg.level));
            result.add(new Segment3D(displacedMid, seg.end, seg.level));
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

            Vec2D dir = new Vec2D(seg.srt.projectZ(), seg.end.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) continue;

            Vec2D perp = dir.rotateCCW90().normalize();

            Random rng = initRandomGenerator((int)(seg.srt.x * 100), (int)(seg.srt.y * 100), cell.resolution);
            // Displacement proportional to segment length
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor * segLength;

            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            segments.set(i, new Segment3D(seg.srt, displacedMid, seg.level));

            if (i + 1 < segments.size()) {
                Segment3D next = segments.get(i + 1);
                if (next.srt.equals(seg.end)) {
                    segments.set(i + 1, new Segment3D(displacedMid, next.end, next.level));
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
                double z = MathUtils.lerp(seg.srt.z, seg.end.z, result.parameter);
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
                double z = MathUtils.lerp(seg.srt.z, seg.end.z, result.parameter);
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
               (returnType == DendryReturnType.PIXEL_ELEVATION ||
                returnType == DendryReturnType.PIXEL_LEVEL ||
                returnType == DendryReturnType.PIXEL_DEBUG);
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
     * For PIXEL_DEBUG mode, also marks endpoint positions with appropriate point types.
     */
    private void sampleSegmentsToPixelCache(List<Segment3D> segments, CellPixelData cache) {
        double cellX = cache.cellX;
        double cellY = cache.cellY;
        boolean isDebugMode = (returnType == DendryReturnType.PIXEL_DEBUG);

        for (Segment3D seg : segments) {
            // Get segment bounds in cell-local coordinates
            double ax = seg.srt.x - cellX;
            double ay = seg.srt.y - cellY;
            double bx = seg.end.x - cellX;
            double by = seg.end.y - cellY;

            // Skip if segment is entirely outside the cell (with margin for point radius)
            double margin = isDebugMode ? 3.0 / pixelGridSize : 0;
            if ((ax < -margin && bx < -margin) || (ax >= 1 + margin && bx >= 1 + margin) ||
                (ay < -margin && by < -margin) || (ay >= 1 + margin && by >= 1 + margin)) {
                continue;
            }

            // Sample along the segment at pixel resolution (for segment line)
            double segLength = seg.length();
            double pixelSize = 1.0 / pixelGridSize;
            int numSamples = Math.max(2, (int) Math.ceil(segLength / pixelSize * 2));

            // Determine if we should use B-spline interpolation
            boolean useSpline = USE_BSPLINE_PIXEL_SAMPLING && seg.hasTangents();

            // Precompute tangent scale for Hermite interpolation
            double tangentScale = useSpline ? segLength * tangentStrength : 0;

            for (int i = 0; i <= numSamples; i++) {
                double t = (double) i / numSamples;
                Point3D pt;

                if (useSpline) {
                    // Cubic Hermite spline interpolation
                    double t2 = t * t;
                    double t3 = t2 * t;
                    double h00 = 2 * t3 - 3 * t2 + 1;
                    double h10 = t3 - 2 * t2 + t;
                    double h01 = -2 * t3 + 3 * t2;
                    double h11 = t3 - t2;

                    double x = h00 * seg.srt.x + h10 * (seg.tangentSrt != null ? seg.tangentSrt.x * tangentScale : 0)
                             + h01 * seg.end.x + h11 * (seg.tangentEnd != null ? seg.tangentEnd.x * tangentScale : 0);
                    double y = h00 * seg.srt.y + h10 * (seg.tangentSrt != null ? seg.tangentSrt.y * tangentScale : 0)
                             + h01 * seg.end.y + h11 * (seg.tangentEnd != null ? seg.tangentEnd.y * tangentScale : 0);
                    double z = MathUtils.lerp(seg.srt.z, seg.end.z, t);

                    pt = new Point3D(x, y, z);
                } else {
                    // Linear interpolation
                    pt = seg.lerp(t);
                }

                double localX = pt.x - cellX;
                double localY = pt.y - cellY;

                // Check if point is within cell
                if (localX >= 0 && localX < 1 && localY >= 0 && localY < 1) {
                    int px = (int) (localX * pixelGridSize);
                    int py = (int) (localY * pixelGridSize);

                    if (px >= 0 && px < pixelGridSize && py >= 0 && py < pixelGridSize) {
                        float elevation = (float) pt.z;
                        byte level = (byte) seg.level;
                        // pointType -1 = segment line only (no special point)
                        cache.setPixel(px, py, elevation, level, (byte) -1);
                    }
                }
            }

            // In debug mode, mark endpoints with their point types and radius
            if (isDebugMode) {
                // Mark start point
                if (ax >= -0.1 && ax < 1.1 && ay >= -0.1 && ay < 1.1) {
                    int px = (int) (ax * pixelGridSize);
                    int py = (int) (ay * pixelGridSize);
                    float elevation = (float) seg.srt.z;
                    byte level = (byte) seg.level;
                    // pointType from PointType enum value
                    byte pointType = (byte) seg.srtType.getValue();
                    cache.markPointWithRadius(px, py, elevation, level, pointType, 2);
                }

                // Mark end point
                if (bx >= -0.1 && bx < 1.1 && by >= -0.1 && by < 1.1) {
                    int px = (int) (bx * pixelGridSize);
                    int py = (int) (by * pixelGridSize);
                    float elevation = (float) seg.end.z;
                    byte level = (byte) seg.level;
                    // pointType from PointType enum value
                    byte pointType = (byte) seg.endType.getValue();
                    cache.markPointWithRadius(px, py, elevation, level, pointType, 2);
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
        } else if (returnType == DendryReturnType.PIXEL_DEBUG) {
            // Debug mode: return point type as value
            // PointType enum: ORIGINAL=0, TRUNK=1, KNOT=2, LEAF=3, segment line=-1
            // We return higher values for more "special" point types:
            //   4 = LEAF (branch endpoint)
            //   3 = KNOT (subdivision point)
            //   2 = TRUNK (main flow path)
            //   1 = ORIGINAL (initial star/point)
            //   0 = segment line only (pointType=-1)
            //  -1 = no data
            if (data.pointType == PointType.LEAF.getValue()) {
                return 4;  // Leaf point
            } else if (data.pointType == PointType.KNOT.getValue()) {
                return 3;  // Subdivision/knot point
            } else if (data.pointType == PointType.TRUNK.getValue()) {
                return 2;  // Trunk point
            } else if (data.pointType == PointType.ORIGINAL.getValue()) {
                return 1;  // Original star point
            } else {
                return 0;  // Segment line only (pointType=-1)
            }
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
     * Uses cell center coordinates for higher resolution cell lookup.
     */
    private List<Segment3D> computeAllSegmentsForCell(long seed, int cellX, int cellY) {
        Cell cell1 = new Cell(cellX, cellY, 1);
        // Use cell center for consistent higher resolution cell lookup
        double queryCenterX = cellX + 0.5;
        double queryCenterY = cellY + 0.5;
        return generateAllSegments(cell1, queryCenterX, queryCenterY);
    }
}

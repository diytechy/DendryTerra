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
     * Configurable via constructor parameter (default: 0).
     *
     * Values:
     *   0  - Normal operation (default)
     *   5  - Return stars as 0-length segments for FIRST constellation only
     *   6  - Return stars as 0-length segments for ALL constellations
     *  10  - Return segments for FIRST constellation only, before stitching
     *  15  - Return segments for ALL constellations, before stitching, only tree
     *  20  - Return segments for ALL constellations, before stitching
     *  30  - Return segments for all constellations INCLUDING stitching
     *  40  - Return level 1+ points as 0-length segments to check distribution before connection
     */
    private final int debug;

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
                         double slopeWhenStraight, double lowestSlopeCutoff,
                         int debug) {
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
        this.debug = debug;

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
        SegmentList allSegments = generateAllSegments(cell1, x, y);
        // Convert to List<Segment3D> for backward compatibility with computeResult
        return computeResult(x, y, allSegments.toSegment3DList());
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
    private SegmentList generateAllSegments(Cell cell1, double queryX, double queryY) {
        // Displacement factor for level 1 (levels 2+ handled in CleanAndNetworkPoints)
        double displacementLevel1 = delta;

        // Asterism (Level 0): Generate and process
        SegmentList asterismBase = generateAsterism(cell1);
        // Prune asterism to query cell - clips segments at cell boundary with EDGE points
        SegmentList asterismPruned = pruneSegmentsToCell(asterismBase, cell1);

        if (resolution == 0) {
            return asterismPruned;
        }

        // Level 1+: Higher resolution refinement using loop
        // Each level generates points for the query cell and connects them using CleanAndNetworkPointsV2
        SegmentList previousSegList = asterismPruned;

        for (int level = 1; level <= resolution; level++) {
            // Get the cell at this level's resolution
            int cellResolution = getCellResolutionForLevel(level);
            Cell levelCell = getCell(queryX, queryY, cellResolution);

            // Generate points for this cell at this level's density
            List<Point3D> levelPoints = generatePointsForCellAtLevel(levelCell.x, levelCell.y, level);

            // Use CleanAndNetworkPointsV2 to create properly connected segments
            // This handles merging, cleaning, tangent computation, subdivision, and displacement
            SegmentList levelSegList = CleanAndNetworkPointsV2(
                levelCell.x, levelCell.y, level, levelPoints, previousSegList);

            // Update previousSegList for next iteration
            previousSegList = levelSegList;
        }

        return previousSegList;
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

    /**
     * Prune segments to a cell boundary.
     * - Keeps segments that have at least one endpoint within the cell
     * - Clips segments that cross the boundary, marking clipped points as EDGE
     * - Removes segments entirely outside the cell
     *
     * @param segments List of segments to prune
     * @param cell The cell to prune to (level 1 cell)
     * @return List of pruned segments with EDGE points at boundaries
     */
    private SegmentList pruneSegmentsToCell(SegmentList segmentList, Cell cell) {
        SegmentList result = new SegmentList(segmentList.getConfig());

        // Map from original point index to new point index in result
        Map<Integer, Integer> pointIndexMap = new HashMap<>();

        // Cell bounds (level 1 cells are 1x1 in normalized space)
        double minX = cell.x;
        double maxX = cell.x + 1.0;
        double minY = cell.y;
        double maxY = cell.y + 1.0;

        for (Segment3D seg : segmentList.getSegments()) {
            Point3D srtPos = seg.getSrt(segmentList);
            Point3D endPos = seg.getEnd(segmentList);

            boolean srtInside = isPointInCell(srtPos, minX, maxX, minY, maxY);
            boolean endInside = isPointInCell(endPos, minX, maxX, minY, maxY);

            if (srtInside && endInside) {
                // Both ends inside - copy points and segment
                int newSrtIdx = copyPointToResult(segmentList, seg.srtIdx, result, pointIndexMap);
                int newEndIdx = copyPointToResult(segmentList, seg.endIdx, result, pointIndexMap);
                result.addBasicSegment(newSrtIdx, newEndIdx, seg.level, seg.tangentSrt, seg.tangentEnd);
            } else if (srtInside || endInside) {
                // One end inside, one outside - clip at boundary
                clipSegmentToCell(segmentList, seg, result, pointIndexMap, minX, maxX, minY, maxY, srtInside);
            }
            // If neither end is inside, check if segment crosses the cell
            else if (segmentCrossesCell(segmentList, seg, minX, maxX, minY, maxY)) {
                // Segment crosses through cell - clip both ends
                clipSegmentBothEnds(segmentList, seg, result, minX, maxX, minY, maxY);
            }
            // Otherwise segment is entirely outside - skip it
        }

        // Post-processing: Mark points near cell edges as EDGE if we have enough segments
        // This prevents higher-level segments from connecting near boundaries
        if (result.getSegmentCount() > 4) {
            //markEdgePointsNearBoundary(result, minX, maxX, minY, maxY, 2);
        }

        return result;
    }

    /**
     * Mark points closest to each cell edge as EDGE type.
     * This prevents segment creation near edges which can cause visual discontinuities.
     *
     * @param segList The segment list to modify
     * @param minX Cell minimum X
     * @param maxX Cell maximum X
     * @param minY Cell minimum Y
     * @param maxY Cell maximum Y
     * @param countPerEdge Number of points to mark per edge
     */
    private void markEdgePointsNearBoundary(SegmentList segList, double minX, double maxX,
                                             double minY, double maxY, int countPerEdge) {
        List<NetworkPoint> points = segList.getPoints();

        // For each edge, find the N closest non-EDGE points and mark them as EDGE
        // Left edge (closest to minX)
        markClosestPointsToEdge(segList, points, p -> p.position.x - minX, countPerEdge);
        // Right edge (closest to maxX)
        markClosestPointsToEdge(segList, points, p -> maxX - p.position.x, countPerEdge);
        // Bottom edge (closest to minY)
        markClosestPointsToEdge(segList, points, p -> p.position.y - minY, countPerEdge);
        // Top edge (closest to maxY)
        markClosestPointsToEdge(segList, points, p -> maxY - p.position.y, countPerEdge);
    }

    /**
     * Mark the N points closest to an edge (as measured by distanceFunc) as EDGE type.
     * Only considers points that aren't already EDGE type.
     */
    private void markClosestPointsToEdge(SegmentList segList, List<NetworkPoint> points,
                                          java.util.function.Function<NetworkPoint, Double> distanceFunc,
                                          int count) {
        // Collect non-EDGE points with their distances
        List<int[]> candidates = new ArrayList<>();  // [index, distance * 10000]
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;
            double dist = distanceFunc.apply(p);
            if (dist >= 0) {
                candidates.add(new int[]{i, (int)(dist * 10000)});
            }
        }

        // Sort by distance (ascending)
        candidates.sort((a, b) -> Integer.compare(a[1], b[1]));

        // Mark the closest N points as EDGE
        for (int i = 0; i < Math.min(count, candidates.size()); i++) {
            int idx = candidates.get(i)[0];
            NetworkPoint p = points.get(idx);
            segList.updatePoint(idx, p.withPointType(PointType.EDGE));
        }
    }

    /**
     * Copy a point from source SegmentList to result, using pointIndexMap to avoid duplicates.
     * @return The index in the result SegmentList
     */
    private int copyPointToResult(SegmentList source, int sourceIdx, SegmentList result, Map<Integer, Integer> pointIndexMap) {
        Integer existingIdx = pointIndexMap.get(sourceIdx);
        if (existingIdx != null) {
            return existingIdx;
        }

        NetworkPoint pt = source.getPoint(sourceIdx);
        int newIdx = result.addPoint(pt.position, pt.pointType, pt.level);
        pointIndexMap.put(sourceIdx, newIdx);
        return newIdx;
    }

    /**
     * Check if a point is inside a cell (inclusive of boundaries).
     */
    private boolean isPointInCell(Point3D point, double minX, double maxX, double minY, double maxY) {
        return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY;
    }

    /**
     * Clip a segment where one end is inside the cell.
     * The outside end is clipped to the cell boundary and marked as EDGE.
     */
    private void clipSegmentToCell(SegmentList source, Segment3D seg, SegmentList result, Map<Integer, Integer> pointIndexMap,
                                    double minX, double maxX, double minY, double maxY, boolean srtInside) {
        Point3D srtPos = seg.getSrt(source);
        Point3D endPos = seg.getEnd(source);
        PointType srtType = seg.getSrtType(source);
        PointType endType = seg.getEndType(source);

        Point3D insidePoint = srtInside ? srtPos : endPos;
        Point3D outsidePoint = srtInside ? endPos : srtPos;
        PointType insideType = srtInside ? srtType : endType;
        Vec2D insideTangent = srtInside ? seg.tangentSrt : seg.tangentEnd;
        Vec2D outsideTangent = srtInside ? seg.tangentEnd : seg.tangentSrt;

        // Find intersection with cell boundary
        Point3D clippedPoint = findCellBoundaryIntersection(insidePoint, outsidePoint, minX, maxX, minY, maxY);
        if (clippedPoint == null) {
            return;
        }

        // Create clipped segment with EDGE type at boundary
        if (srtInside) {
            int insideIdx = copyPointToResult(source, seg.srtIdx, result, pointIndexMap);
            int clipIdx = result.addPoint(clippedPoint, PointType.EDGE, seg.level);
            result.addBasicSegment(insideIdx, clipIdx, seg.level, seg.tangentSrt, outsideTangent);
        } else {
            int clipIdx = result.addPoint(clippedPoint, PointType.EDGE, seg.level);
            int insideIdx = copyPointToResult(source, seg.endIdx, result, pointIndexMap);
            result.addBasicSegment(clipIdx, insideIdx, seg.level, outsideTangent, seg.tangentEnd);
        }
    }

    /**
     * Clip a segment where both ends are outside but the segment crosses through the cell.
     * Creates a midpoint KNOT so higher levels can attach to orphaned EDGE-EDGE segments.
     */
    private void clipSegmentBothEnds(SegmentList source, Segment3D seg, SegmentList result,
                                      double minX, double maxX, double minY, double maxY) {
        Point3D srtPos = seg.getSrt(source);
        Point3D endPos = seg.getEnd(source);

        // Find both intersection points
        Point3D entry = findCellBoundaryIntersection(srtPos, endPos, minX, maxX, minY, maxY);
        Point3D exit = findCellBoundaryIntersection(endPos, srtPos, minX, maxX, minY, maxY);

        if (entry == null || exit == null) {
            return;
        }

        // Create midpoint KNOT so higher levels can attach to this segment
        Point3D midpoint = new Point3D(
            (entry.x + exit.x) / 2.0,
            (entry.y + exit.y) / 2.0,
            (entry.z + exit.z) / 2.0
        );

        // Compute midpoint tangent by interpolating between entry and exit tangents
        Vec2D midTangent = null;
        if (seg.tangentSrt != null && seg.tangentEnd != null) {
            midTangent = new Vec2D(
                (seg.tangentSrt.x + seg.tangentEnd.x) / 2.0,
                (seg.tangentSrt.y + seg.tangentEnd.y) / 2.0
            ).normalize();
        } else if (seg.tangentSrt != null) {
            midTangent = seg.tangentSrt;
        } else if (seg.tangentEnd != null) {
            midTangent = seg.tangentEnd;
        }

        // Add points: EDGE -> KNOT -> EDGE
        int entryIdx = result.addPoint(entry, PointType.EDGE, seg.level);
        int midIdx = result.addPoint(midpoint, PointType.KNOT, seg.level);
        int exitIdx = result.addPoint(exit, PointType.EDGE, seg.level);

        // Create two segments through the midpoint
        result.addBasicSegment(entryIdx, midIdx, seg.level, seg.tangentSrt, midTangent);
        result.addBasicSegment(midIdx, exitIdx, seg.level, midTangent, seg.tangentEnd);
    }

    /**
     * Check if a segment crosses through a cell (both ends outside but line passes through).
     */
    private boolean segmentCrossesCell(SegmentList source, Segment3D seg, double minX, double maxX, double minY, double maxY) {
        Point3D srtPos = seg.getSrt(source);
        Point3D endPos = seg.getEnd(source);

        // Use line-box intersection test
        Point2D p1 = srtPos.projectZ();
        Point2D p2 = endPos.projectZ();

        // Check intersection with each edge of the cell
        return lineIntersectsBox(p1.x, p1.y, p2.x, p2.y, minX, maxX, minY, maxY);
    }

    /**
     * Check if a line segment intersects a box (axis-aligned).
     */
    private boolean lineIntersectsBox(double x1, double y1, double x2, double y2,
                                       double minX, double maxX, double minY, double maxY) {
        // Liang-Barsky algorithm
        double dx = x2 - x1;
        double dy = y2 - y1;

        double[] p = {-dx, dx, -dy, dy};
        double[] q = {x1 - minX, maxX - x1, y1 - minY, maxY - y1};

        double tMin = 0.0;
        double tMax = 1.0;

        for (int i = 0; i < 4; i++) {
            if (Math.abs(p[i]) < MathUtils.EPSILON) {
                if (q[i] < 0) return false;
            } else {
                double t = q[i] / p[i];
                if (p[i] < 0) {
                    tMin = Math.max(tMin, t);
                } else {
                    tMax = Math.min(tMax, t);
                }
            }
        }

        return tMin <= tMax;
    }

    /**
     * Find intersection point of a line segment with the cell boundary.
     * Returns the intersection point closest to 'from' that enters the cell.
     */
    private Point3D findCellBoundaryIntersection(Point3D from, Point3D to,
                                                  double minX, double maxX, double minY, double maxY) {
        double dx = to.x - from.x;
        double dy = to.y - from.y;
        double dz = to.z - from.z;

        double tMin = 0.0;
        double tMax = 1.0;

        // Check each boundary
        double[] tValues = new double[4];
        int tCount = 0;

        // Left boundary (x = minX)
        if (Math.abs(dx) > MathUtils.EPSILON) {
            double t = (minX - from.x) / dx;
            if (t > 0 && t < 1) {
                double y = from.y + t * dy;
                if (y >= minY && y <= maxY) {
                    tValues[tCount++] = t;
                }
            }
        }

        // Right boundary (x = maxX)
        if (Math.abs(dx) > MathUtils.EPSILON) {
            double t = (maxX - from.x) / dx;
            if (t > 0 && t < 1) {
                double y = from.y + t * dy;
                if (y >= minY && y <= maxY) {
                    tValues[tCount++] = t;
                }
            }
        }

        // Bottom boundary (y = minY)
        if (Math.abs(dy) > MathUtils.EPSILON) {
            double t = (minY - from.y) / dy;
            if (t > 0 && t < 1) {
                double x = from.x + t * dx;
                if (x >= minX && x <= maxX) {
                    tValues[tCount++] = t;
                }
            }
        }

        // Top boundary (y = maxY)
        if (Math.abs(dy) > MathUtils.EPSILON) {
            double t = (maxY - from.y) / dy;
            if (t > 0 && t < 1) {
                double x = from.x + t * dx;
                if (x >= minX && x <= maxX) {
                    tValues[tCount++] = t;
                }
            }
        }

        if (tCount == 0) {
            return null;
        }

        // Find the smallest t (closest intersection to 'from')
        double tBest = tValues[0];
        for (int i = 1; i < tCount; i++) {
            if (tValues[i] < tBest) {
                tBest = tValues[i];
            }
        }

        // Compute intersection point
        double x = from.x + tBest * dx;
        double y = from.y + tBest * dy;
        double z = from.z + tBest * dz;

        return new Point3D(x, y, z);
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

        // Convert cell indices to world coordinates
        // For level 1: resolution=1, cells are 1x1 in world space
        // For level 2: resolution=2, cells are 0.5x0.5 in world space, etc.
        double cellResolution = getCellResolutionForLevel(level);
        double worldCellStartX = (double) cellX / cellResolution;
        double worldCellStartY = (double) cellY / cellResolution;
        double cellWorldSize = 1.0 / cellResolution;

        // Grid spacing in world coordinates (spacing between point centers within this cell)
        double worldGridSpacing = cellWorldSize / pointsPerAxis;

        // Generate a grid of points within the cell
        for (int i = 0; i < pointsPerAxis; i++) {
            for (int j = 0; j < pointsPerAxis; j++) {
                // Position point at center of each sub-cell with deterministic jitter
                double baseX = worldCellStartX + (j + 0.5) * worldGridSpacing;
                double baseY = worldCellStartY + (i + 0.5) * worldGridSpacing;

                // Add deterministic jitter based on position and level
                Random rng = initRandomGenerator((int)(baseX * 10000), (int)(baseY * 10000), level);
                double jitterX = (rng.nextDouble() - 0.5) * worldGridSpacing * 0.8;
                double jitterY = (rng.nextDouble() - 0.5) * worldGridSpacing * 0.8;

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

    /**
     * Compute slopes for a list of points based on their neighbors.
     * Unlike grid-based slope computation, this considers all other points as potential neighbors.
     * Uses the same algorithm: find 2 closest neighbors at least 30 degrees apart.
     */
    private List<Point3D> computeSlopesForPoints(List<Point3D> points) {
        if (points.size() < 3) {
            // Not enough points to compute meaningful slopes
            return new ArrayList<>(points);
        }

        List<Point3D> result = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            Point3D center = points.get(i);

            // Collect all neighbors with their distances and angles
            List<NeighborInfo> neighbors = new ArrayList<>();
            for (int j = 0; j < points.size(); j++) {
                if (i == j) continue;
                Point3D neighbor = points.get(j);
                double dx = neighbor.x - center.x;
                double dy = neighbor.y - center.y;
                double dz = neighbor.z - center.z;
                double dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < MathUtils.EPSILON) continue; // Skip coincident points
                double angle = Math.atan2(dy, dx);
                neighbors.add(new NeighborInfo(dx, dy, dz, dist, angle));
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

            // If we couldn't find 2 suitable neighbors, keep point without slope
            if (n1 == null || n2 == null) {
                result.add(center);
                continue;
            }

            // Solve 2x2 linear system to find slopeX and slopeY
            double det = n1.dx * n2.dy - n2.dx * n1.dy;
            if (Math.abs(det) < MathUtils.EPSILON) {
                // Neighbors are collinear, can't solve
                result.add(center);
                continue;
            }

            double slopeX = (n1.dz * n2.dy - n2.dz * n1.dy) / det;
            double slopeY = (n1.dx * n2.dz - n2.dx * n1.dz) / det;

            result.add(center.withSlopes(slopeX, slopeY));
        }

        return result;
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
    private SegmentList generateAsterism(Cell queryCell1) {
        // Find the 4 closest constellations to the query cell
        List<ConstellationInfo> closestConstellations = findClosestConstellations(queryCell1);

        // Generate asterism SegmentLists for each constellation
        Map<Long, SegmentList> constellationSegmentLists = new HashMap<>();
        Map<Long, List<Point3D>> constellationStars = new HashMap<>();

        // Initialize star segment list for debugging (needs to be outside if block for scope)
        SegmentList starSegmentList = new SegmentList(new SegmentListConfig(salt));

        boolean firstConstellation = true;
        for (ConstellationInfo constInfo : closestConstellations) {
            long constKey = constInfo.getKey();

            // Generate stars for this constellation (with 9x9 sampling, merging, etc.)
            List<Point3D> stars = generateConstellationStars(constInfo);

            // DEBUG: Collect stars as zero-length segments for visualization
            if (debug == 5 || debug == 6) {
                for (Point3D star : stars) {
                    int starIdx = starSegmentList.addPoint(star, PointType.ORIGINAL, 0);
                    starSegmentList.addBasicSegment(starIdx, starIdx, 0, null, null);
                }
            }
            if (debug == 5) {
                LOGGER.info("debug=5: Returning stars only for first constellation ({})", stars.size());
                return starSegmentList;
            }
            // Build network within constellation using CleanAndNetworkPointsV2
            // Use quantized center coordinates for cell reference
            int cellRefX = (int) Math.floor(constInfo.centerX);
            int cellRefY = (int) Math.floor(constInfo.centerY);
            SegmentList segList = CleanAndNetworkPointsV2(cellRefX, cellRefY, 0, stars, null);

            constellationStars.put(constKey, stars);
            constellationSegmentLists.put(constKey, segList);

            // DEBUG: Return after first constellation only
            if (debug == 10 && firstConstellation) {
                LOGGER.info("debug=10: Returning first constellation segments only ({} segments)", segList.getSegmentCount());
                return segList;
            }
            firstConstellation = false;
        }

        // DEBUG: Return all constellation segments before stitching
        if ((debug == 20)||(debug == 15)) {
            SegmentList combined = combineConstellationSegmentLists(constellationSegmentLists);
            LOGGER.info("debug=20or15: Returning all constellation segments before stitching ({} segments)", combined.getSegmentCount());
            return combined;
        }
        if (debug == 6) {
            LOGGER.info("debug=6: Returning all constellation stars: ({} stars)", starSegmentList.getPointCount());
            return starSegmentList;
        }

        // Combine all constellation SegmentLists into a single combined SegmentList
        SegmentList combinedSegList = combineConstellationSegmentLists(constellationSegmentLists);

        // Stitch adjacent constellations together using point indices in the combined SegmentList
        int stitchCount = stitchConstellationsV2(closestConstellations, constellationSegmentLists, combinedSegList);

        // Normalize elevations after stitching
        combinedSegList.normalizeElevations();

        // DEBUG: Return all segments including stitching
        if (debug == 30) {
            LOGGER.info("debug=30: Returning all segments including stitching ({} segments, {} stitch)", combinedSegList.getSegmentCount(), stitchCount);
            return combinedSegList;
        }

        return combinedSegList;
    }

    /**
     * Combine multiple constellation SegmentLists into a single SegmentList.
     * Points are de-duplicated by position, segments are copied directly.
     */
    private SegmentList combineConstellationSegmentLists(Map<Long, SegmentList> constellationSegmentLists) {
        // Use the config from the first constellation (they should all be the same)
        SegmentListConfig config = null;
        for (SegmentList segList : constellationSegmentLists.values()) {
            config = segList.getConfig();
            break;
        }
        SegmentList combined = new SegmentList(config != null ? config : new SegmentListConfig(salt));

        // Map from quantized position to new index in combined list
        Map<Long, Integer> positionToIndex = new HashMap<>();

        // First pass: add all unique points from all constellations
        for (SegmentList segList : constellationSegmentLists.values()) {
            for (int i = 0; i < segList.getPointCount(); i++) {
                NetworkPoint pt = segList.getPoint(i);
                long posKey = quantizePositionForCombine(pt.position);

                if (!positionToIndex.containsKey(posKey)) {
                    int newIdx = combined.addPoint(pt.position, pt.pointType, pt.level);
                    positionToIndex.put(posKey, newIdx);
                }
            }
        }

        // Second pass: add all segments using position map for index lookup
        for (SegmentList segList : constellationSegmentLists.values()) {
            for (Segment3D seg : segList.getSegments()) {
                // Resolve Point3D positions from indices
                Point3D srtPos = seg.getSrt(segList);
                Point3D endPos = seg.getEnd(segList);

                long srtKey = quantizePositionForCombine(srtPos);
                long endKey = quantizePositionForCombine(endPos);

                Integer srtIdx = positionToIndex.get(srtKey);
                Integer endIdx = positionToIndex.get(endKey);

                if (srtIdx != null && endIdx != null) {
                    combined.addBasicSegment(srtIdx, endIdx, seg.level, seg.tangentSrt, seg.tangentEnd);
                }
            }
        }

        return combined;
    }

    /**
     * Quantize position for combining SegmentLists (high precision).
     */
    private long quantizePositionForCombine(Point3D pos) {
        long qx = Math.round(pos.x * 100000);
        long qy = Math.round(pos.y * 100000);
        return (qx << 32) | (qy & 0xFFFFFFFFL);
    }

    /**
     * Stitch constellations together using point indices in the combined SegmentList.
     * Returns the number of stitch segments created.
     */
    private int stitchConstellationsV2(List<ConstellationInfo> constellations,
                                        Map<Long, SegmentList> constellationSegmentLists,
                                        SegmentList combinedSegList) {
        int stitchCount = 0;

        if (constellations.size() < 2) {
            return stitchCount;
        }

        double adjacencyThreshold = getAdjacentConstellationThreshold();
        double adjacencyThresholdSq = adjacencyThreshold * adjacencyThreshold;

        // Compute max segment length for stitching (using level 1 grid spacing)
        double gridSpacing = getGridSpacingForLevel(1);
        double maxSegmentLength = MERGE_POINT_SPACING * gridSpacing;

        // Find all pairs of adjacent constellations
        Set<Long> processedPairs = new HashSet<>();
        List<int[]> adjacentPairs = new ArrayList<>();

        for (int i = 0; i < constellations.size(); i++) {
            ConstellationInfo constI = constellations.get(i);
            for (int j = i + 1; j < constellations.size(); j++) {
                ConstellationInfo constJ = constellations.get(j);

                double dx = constI.centerX - constJ.centerX;
                double dy = constI.centerY - constJ.centerY;
                double distSq = dx * dx + dy * dy;

                if (distSq < adjacencyThresholdSq) {
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

        // Sort pairs deterministically
        adjacentPairs.sort((a, b) -> {
            ConstellationInfo aI = constellations.get(a[0]);
            ConstellationInfo aJ = constellations.get(a[1]);
            ConstellationInfo bI = constellations.get(b[0]);
            ConstellationInfo bJ = constellations.get(b[1]);
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

            SegmentList segListI = constellationSegmentLists.get(constI.getKey());
            SegmentList segListJ = constellationSegmentLists.get(constJ.getKey());

            if (segListI == null || segListI.isEmpty()) continue;
            if (segListJ == null || segListJ.isEmpty()) continue;

            // Extract unique endpoints from each constellation's segments
            List<Point3D> endpointsI = extractUniqueEndpointsFromSegmentList(segListI);
            List<Point3D> endpointsJ = extractUniqueEndpointsFromSegmentList(segListJ);

            if (endpointsI.isEmpty() || endpointsJ.isEmpty()) continue;

            // Find best pair to connect (closest distance, then lowest max elevation)
            List<double[]> candidatePairs = new ArrayList<>();

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

            candidatePairs.sort((a, b) -> Double.compare(a[0], b[0]));
            int pairsToConsider = Math.min(6, candidatePairs.size());

            if (pairsToConsider == 0) continue;

            // Select the one with lowest max elevation from closest 6
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

            // Find indices in combined SegmentList
            int idxI = combinedSegList.findPointByPosition(bestI, MathUtils.EPSILON);
            int idxJ = combinedSegList.findPointByPosition(bestJ, MathUtils.EPSILON);

            if (idxI < 0 || idxJ < 0) {
                LOGGER.warn("Stitch point not found in combined SegmentList: I={}, J={}", idxI, idxJ);
                continue;
            }

            // Create stitch segment using addSegmentWithDivisions
            combinedSegList.addSegmentWithDivisions(idxI, idxJ, 0, maxSegmentLength);
            stitchCount++;
        }

        return stitchCount;
    }

    /**
     * Extract unique endpoints from a SegmentList's segments.
     * Since SegmentList already maintains unique NetworkPoints, just return their positions.
     */
    private List<Point3D> extractUniqueEndpointsFromSegmentList(SegmentList segList) {
        List<Point3D> endpoints = new ArrayList<>();
        for (NetworkPoint np : segList.getPoints()) {
            endpoints.add(np.position);
        }
        return endpoints;
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
        if (debug > 0) {
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

        // Step 5: Compute slopes for each star based on neighboring stars
        List<Point3D> starsWithSlopes = computeSlopesForPoints(mergedStars);

        // Debug logging for star counts
        if (debug > 0) {
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
                draftedStars.size(), boundedStars.size(), starsWithSlopes.size(),
                String.format("%.3f", size), String.format("%.3f", halfMergeSpacing),
                String.format("%.2f", center.x), String.format("%.2f", center.y),
                String.format("%.2f", minX), String.format("%.2f", minY),
                String.format("%.2f", maxX), String.format("%.2f", maxY));
        }

        return starsWithSlopes;
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


    // ========== CleanAndNetworkPoints V2 Implementation ==========
    // New implementation using SegmentList and UnconnectedPoints
    // Simplified algorithm: single tree growth, no chain tracking needed

    /**
     * V2: Create network of segments from a list of points.
     * Uses SegmentList and UnconnectedPoints for cleaner architecture.
     *
     * @param cellX X coordinate of cell (constellation index for level 0)
     * @param cellY Y coordinate of cell
     * @param level Resolution level (0 for asterisms)
     * @param points List of points to connect
     * @return SegmentList containing connected points and segments
     */
    private SegmentList CleanAndNetworkPointsV2(int cellX, int cellY, int level, List<Point3D> points) {
        return CleanAndNetworkPointsV2(cellX, cellY, level, points, null);
    }

    /**
     * V2: CleanAndNetworkPoints with previous level SegmentList.
     *
     * Algorithm:
     * 1. Clean points (merge, remove near segments, probabilistic removal)
     * 2. Create UnconnectedPoints from cleaned points
     * 3. For level 0: Build trunk from highest point
     * 4. Grow network: Connect points from UnconnectedPoints to SegmentList
     * 5. Mark LEAF points
     */
    private SegmentList CleanAndNetworkPointsV2(int cellX, int cellY, int level,
                                                 List<Point3D> points,
                                                 SegmentList previousLevelSegments) {
        // Create SegmentList with full configuration
        SegmentListConfig config = new SegmentListConfig(salt)
            .withSplines(useSplines)
            .withCurvature(curvature)
            .withTangentStrength(tangentStrength)
            .withMaxTwistAngle(tangentAngle)
            .withSlopeWithoutTwist(slopeWhenStraight);
        SegmentList result = new SegmentList(config);
        if (points.isEmpty()) return result;

        // Function setup: determine cell-specific distances
        double gridSpacing = getGridSpacingForLevel(level);
        double mergeDistance = MERGE_POINT_SPACING * getGridSpacingForLevel(level+1);
        double maxSegmentDistance = MAX_POINT_SEGMENT_DISTANCE * gridSpacing;

        // DEBUG 40: Track point counts at each stage for the highest level
        int draftedCount = points.size();
        int afterMergeCount = 0;
        int afterNearSegmentsCount = 0;
        int afterProbabilisticCount = 0;
        boolean isDebug40TargetLevel = (debug == 40 && level > 0 && level == resolution);

        // Step 1: Clean network points - merge points within merge distance (only for level > 0)
        List<Point3D> cleanedPoints;
        if (level > 0) {
            cleanedPoints = cleanAndMergePoints(points, mergeDistance);
        } else {
            cleanedPoints = new ArrayList<>(points);
        }
        afterMergeCount = cleanedPoints.size();

        // Step 2: Remove points within merge distance of lower-level segments (if not level 0)
        if (level > 0 && previousLevelSegments != null && !previousLevelSegments.isEmpty()) {
            cleanedPoints = removePointsNearSegments(cleanedPoints, previousLevelSegments, mergeDistance);
        }
        afterNearSegmentsCount = cleanedPoints.size();

        // Step 3: Probabilistically remove points based on branchesSampler and distance from segments
        if (level > 0 && previousLevelSegments != null) {
            cleanedPoints = probabilisticallyRemovePoints(cleanedPoints,
                                                          previousLevelSegments, gridSpacing);
        }
        afterProbabilisticCount = cleanedPoints.size();

        if (cleanedPoints.isEmpty()) return result;

        // Step 4: Create UnconnectedPoints from cleaned points
        UnconnectedPoints unconnected = UnconnectedPoints.fromPoints(cleanedPoints, PointType.ORIGINAL, level);

        // DEBUG 40: Show points as 0-length segments ONLY for the highest level (resolution)
        // For lower levels, fully connect them so we can see the segment tree up to this point
        if (isDebug40TargetLevel) {
            // Copy previous level segments first (these are already connected)
            if (previousLevelSegments != null) {
                result = previousLevelSegments.copy();
            }
            // Add all cleaned level points as 0-length segments for visualization
            for (Point3D pt : cleanedPoints) {
                int idx = result.addPoint(pt, PointType.ORIGINAL, level);
                result.addBasicSegment(idx, idx, level, null, null);
            }
            LOGGER.info("debug=40: Level {} (resolution={}) - Point counts: drafted={}, afterMerge={}, afterNearSegments={}, afterProbabilistic={}",
                       level, resolution, draftedCount, afterMergeCount, afterNearSegmentsCount, afterProbabilisticCount);
            return result;
        }

        // Step 5: Initialize SegmentList
        // For level 0: start empty, build trunk first
        // For level 1+: copy previous level segments (points already connected)
        if (level > 0 && previousLevelSegments != null) {
            result = previousLevelSegments.copy();
        }

        // Step 6: Connect points using simplified algorithm
        connectAndDefineSegmentsV2(unconnected, result, maxSegmentDistance, mergeDistance, level, cellX, cellY);

        // Step 7: Mark LEAF points
        result.markLeafPoints();

        return result;
    }

    /**
     * V2: Connect points from UnconnectedPoints to SegmentList.
     * Simplified algorithm: always grow from existing segments (single tree).
     */
    private void connectAndDefineSegmentsV2(UnconnectedPoints unconnected, SegmentList segList,
                                             double maxSegmentDistance, double mergeDistance,
                                             int level, int cellX, int cellY) {
        double maxDistSq = maxSegmentDistance; // * maxSegmentDistance;
        double mergeDistSq = mergeDistance; // * mergeDistance;

        // Phase A: For level 0, build trunk from highest point
        if (level == 0) {
            buildTrunkV2(unconnected, segList, maxSegmentDistance, mergeDistance, level, cellX, cellY);

            // DEBUG: Return after trunk creation
            if (debug == 15) {
                // Add 0-length segments for unconnected points for visualization
                unconnected.forEach(p -> {
                    int idx = segList.addPoint(p.position, p.pointType, level);
                    // 0-length segment for visualization
                    segList.addBasicSegment(idx, idx, level, null, null);
                });
                LOGGER.info("debug=15: V2 returning after trunk ({} segments, {} points)",
                           segList.getSegmentCount(), segList.getPointCount());
                return;
            }
        }

        // Phase B: Grow network from existing segments
        // Pre-sort unconnected points by distance to existing segment list (computed once)
        // This prevents new connections from affecting sort order and reduces crossing segments

        // Step 1: Compute initial distances for all unconnected points to segment list
        List<int[]> sortedByDistance = new ArrayList<>();
        List<Integer> remaining = unconnected.getRemainingIndices();

        for (int unconnIdx : remaining) {
            NetworkPoint unconnPt = unconnected.getPoint(unconnIdx);
            if (unconnPt.pointType == PointType.EDGE) continue;

            // Find minimum distance to any point in segment list
            double minDistSq = Double.MAX_VALUE;
            Point2D unconnPos = unconnPt.position.projectZ();

            for (int i = 0; i < segList.getPointCount(); i++) {
                NetworkPoint segPt = segList.getPoint(i);
                if (segPt.pointType == PointType.EDGE) continue;

                double distSq = unconnPos.distanceSquaredTo(segPt.position.projectZ());
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                }
            }

            // Only include points within max segment distance
            if (minDistSq <= maxDistSq) {
                // Store as [unconnIdx, distance * 1000000 as int for sorting]
                sortedByDistance.add(new int[]{unconnIdx, (int)(minDistSq * 1000000)});
            }
        }

        // Step 2: Sort by initial distance (closest first)
        sortedByDistance.sort((a, b) -> Integer.compare(a[1], b[1]));

        // Step 3: Process points in sorted order (closest to furthest from initial segment list)
        for (int[] entry : sortedByDistance) {
            int unconnIdx = entry[0];

            // Skip if already removed (shouldn't happen but safety check)
            if (unconnected.isRemoved(unconnIdx)) continue;

            NetworkPoint unconnPt = unconnected.getPoint(unconnIdx);

            // Find best neighbor in SegmentList for this point
            // This looks at the CURRENT segment list, which may include newly added points
            int neighborIdx = findBestNeighborV2(unconnPt, segList, maxDistSq, mergeDistSq, level);

            if (neighborIdx < 0) {
                // No valid neighbor - mark as removed (can't connect)
                unconnected.markRemoved(unconnIdx);
                continue;
            }

            // Create connection: move point to SegmentList and create segment
            // The newly added point becomes available for subsequent iterations
            createSegmentV2(unconnected, unconnIdx, segList, neighborIdx, level, mergeDistance);
        }
    }

    /**
     * V2: Build trunk for level 0 asterisms.
     * Starts at lowest point and extends uphill until no more connections possible.
     */
    private void buildTrunkV2(UnconnectedPoints unconnected, SegmentList segList,
                               double maxSegmentDistance, double mergeDistance, int level, int cellX, int cellY) {
        double maxDistSq = maxSegmentDistance; // * maxSegmentDistance;

        // Find lowest unconnected point to start trunk
        int startIdx = unconnected.findLowestUnconnected();
        if (startIdx < 0) return;

        // Get first trunk point
        NetworkPoint startPt = unconnected.removeAndGet(startIdx);
        NetworkPoint trunkStartPt = new NetworkPoint(startPt.position, -1, PointType.TRUNK, level);

        int maxIterations = unconnected.totalSize() + 1;
        int iterations = 0;
        boolean firstSegment = true;
        int currentIdx = -1;

        // Extend trunk uphill
        while (iterations < maxIterations) {
            iterations++;

            // For first segment, find neighbor from unconnected points based on startPt position
            // For subsequent, use currentIdx in segList
            int nextUnconnIdx;
            if (firstSegment) {
                nextUnconnIdx = findBestTrunkNeighborFromPoint(unconnected, trunkStartPt.position, maxDistSq, level);
            } else {
                nextUnconnIdx = findBestTrunkNeighborV2(unconnected, segList, currentIdx, maxDistSq, level);
            }

            if (nextUnconnIdx < 0) {
                // No more uphill connections
                if (firstSegment) {
                    // No neighbor found for first point - just add it alone
                    segList.addPoint(trunkStartPt.position, PointType.TRUNK, level);
                }
                break;
            }

            // Get next trunk point
            NetworkPoint nextPt = unconnected.removeAndGet(nextUnconnIdx);
            NetworkPoint trunkNextPt = new NetworkPoint(nextPt.position, -1, PointType.TRUNK, level);

            if (firstSegment) {
                // First segment: use addSegmentWithDivisions with two new NetworkPoints
                // Returns the index of trunkNextPt (the uphill endpoint for continuation)
                currentIdx = segList.addSegmentWithDivisions(trunkStartPt, trunkNextPt, level, mergeDistance);
                firstSegment = false;
            } else {
                // Subsequent segments: use addSegmentWithDivisions with new NetworkPoint and existing index
                // Returns the index of trunkNextPt (the new point for continuation)
                currentIdx = segList.addSegmentWithDivisions(trunkNextPt, currentIdx, level, mergeDistance);
            }
        }
    }

    /**
     * Find best trunk neighbor from a position (for first trunk segment).
     * Looks for uphill neighbors (positive slope).
     */
    private int findBestTrunkNeighborFromPoint(UnconnectedPoints unconnected, Point3D currentPos,
                                                double maxDistSq, int level) {
        Point2D currentPos2D = currentPos.projectZ();

        double bestSlope = 0;  // Must be positive (uphill)
        int bestUnconnIdx = -1;

        List<Integer> remaining = unconnected.getRemainingIndices();
        for (int unconnIdx : remaining) {
            NetworkPoint candidate = unconnected.getPoint(unconnIdx);
            if (candidate.pointType == PointType.EDGE) continue;

            Point2D candidatePos = candidate.position.projectZ();
            double distSq = currentPos2D.distanceSquaredTo(candidatePos);

            if (distSq > maxDistSq || distSq < MathUtils.EPSILON) continue;

            // Calculate slope
            double dist = Math.sqrt(distSq);
            double heightDiff = candidate.position.z - currentPos.z;
            double normalizedSlope = heightDiff / Math.pow(dist, DISTANCE_FALLOFF_POWER);

            // Trunk requires uphill (positive slope)
            if (normalizedSlope <= 0) continue;

            if (normalizedSlope > bestSlope) {
                bestSlope = normalizedSlope;
                bestUnconnIdx = unconnIdx;
            }
        }

        return bestUnconnIdx;
    }

    /**
     * V2: Find best neighbor in SegmentList for a point.
     * Uses priority rules: merge distance first, then lowest slope.
     */
    private int findBestNeighborV2(NetworkPoint sourcePt, SegmentList segList,
                                    double maxDistSq, double mergeDistSq, int level) {
        Point2D sourcePos = sourcePt.position.projectZ();

        double bestSlopeWithinMerge = Double.MAX_VALUE;
        int bestNeighborWithinMerge = -1;
        double bestSlopeOverall = Double.MAX_VALUE;
        int bestNeighborOverall = -1;

        for (int i = 0; i < segList.getPointCount(); i++) {
            NetworkPoint candidate = segList.getPoint(i);
            if (candidate.pointType == PointType.EDGE) continue;

            // Skip if already has 5+ connections (full)
            if (candidate.connections >= 5) continue;

            Point2D candidatePos = candidate.position.projectZ();
            double distSq = sourcePos.distanceSquaredTo(candidatePos);

            if (distSq > maxDistSq || distSq < MathUtils.EPSILON) continue;

            // Calculate normalized slope with DistanceFalloffPower
            double dist = Math.sqrt(distSq);
            double heightDiff = sourcePt.position.z - candidate.position.z;
            double normalizedSlope = heightDiff / Math.pow(dist, DISTANCE_FALLOFF_POWER);

            // Level 1+ has slope cutoff
            if (level > 0 && normalizedSlope > lowestSlopeCutoff) {
                continue;
            }

            // Apply branch encouragement if candidate has 2+ connections
            double effectiveSlope = normalizedSlope;
            if (candidate.connections >= 2) {
                effectiveSlope *= BRANCH_ENCOURAGEMENT_FACTOR;
            }

            // Priority A: Prefer neighbors within merge distance
            if (distSq <= mergeDistSq) {
                if (effectiveSlope < bestSlopeWithinMerge) {
                    bestSlopeWithinMerge = effectiveSlope;
                    bestNeighborWithinMerge = i;
                }
            }

            // Priority B: Track overall best
            if (effectiveSlope < bestSlopeOverall) {
                bestSlopeOverall = effectiveSlope;
                bestNeighborOverall = i;
            }
        }

        return (bestNeighborWithinMerge >= 0) ? bestNeighborWithinMerge : bestNeighborOverall;
    }

    /**
     * V2: Find best trunk neighbor (must be uphill).
     */
    private int findBestTrunkNeighborV2(UnconnectedPoints unconnected, SegmentList segList,
                                         int currentSegListIdx, double maxDistSq, int level) {
        NetworkPoint currentPt = segList.getPoint(currentSegListIdx);
        Point2D currentPos = currentPt.position.projectZ();

        double bestSlope = 0;  // Must be positive (uphill)
        int bestUnconnIdx = -1;

        List<Integer> remaining = unconnected.getRemainingIndices();
        for (int unconnIdx : remaining) {
            NetworkPoint candidate = unconnected.getPoint(unconnIdx);
            if (candidate.pointType == PointType.EDGE) continue;

            Point2D candidatePos = candidate.position.projectZ();
            double distSq = currentPos.distanceSquaredTo(candidatePos);

            if (distSq > maxDistSq || distSq < MathUtils.EPSILON) continue;

            // Calculate slope
            double dist = Math.sqrt(distSq);
            double heightDiff = candidate.position.z - currentPt.position.z;
            double normalizedSlope = heightDiff / Math.pow(dist, DISTANCE_FALLOFF_POWER);

            // Trunk requires uphill (positive slope)
            if (normalizedSlope <= 0) continue;

            if (normalizedSlope > bestSlope) {
                bestSlope = normalizedSlope;
                bestUnconnIdx = unconnIdx;
            }
        }

        return bestUnconnIdx;
    }

    /**
     * V2: Create a segment connecting an unconnected point to a SegmentList point.
     * Uses SegmentList.addSegment which handles point addition, tangent computation, and subdivision.
     */
    private void createSegmentV2(UnconnectedPoints unconnected, int unconnIdx,
                                  SegmentList segList, int neighborIdx,
                                  int level, double mergeDistance) {
        // Get the unconnected point
        NetworkPoint unconnPt = unconnected.removeAndGet(unconnIdx);

        // Use addSegmentWithDivisions which adds the point, computes tangents, and subdivides as needed
        segList.addSegmentWithDivisions(unconnPt, neighborIdx, level, mergeDistance);
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
    private List<Point3D> removePointsNearSegments(List<Point3D> points, SegmentList segmentList, double mergeDistance) {
        List<Point3D> result = new ArrayList<>();
        double mergeDistSq = mergeDistance * mergeDistance;

        for (Point3D point : points) {
            boolean tooClose = false;
            Point2D p2d = point.projectZ();

            for (Segment3D seg : segmentList.getSegments()) {
                // Find closest point on segment using linear interpolation
                Point3D srtPos = seg.getSrt(segmentList);
                Point3D endPos = seg.getEnd(segmentList);
                Point2D segA = srtPos.projectZ();
                Point2D segB = endPos.projectZ();
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
    private List<Point3D> probabilisticallyRemovePoints(List<Point3D> points,
                                                         SegmentList previousLevelSegments,
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

                for (Segment3D seg : previousLevelSegments.getSegments()) {
                    Point3D srtPos = seg.getSrt(previousLevelSegments);
                    Point3D endPos = seg.getEnd(previousLevelSegments);
                    double distSq = pointToSegmentDistanceSquared(point2D, srtPos.projectZ(), endPos.projectZ());
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
    private List<Point3D> probabilisticallyRemovePoints(List<Point3D> points) {
        return probabilisticallyRemovePoints(points, new SegmentList(new SegmentListConfig(salt)), 1.0);
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
     * Check if two points match (within epsilon tolerance).
     */
    private static boolean pointsMatch(Point3D a, Point3D b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return (dx * dx + dy * dy + dz * dz) < MathUtils.EPSILON * MathUtils.EPSILON;
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
    private void sampleSegmentsToPixelCache(SegmentList segmentList, CellPixelData cache) {
        double cellX = cache.cellX;
        double cellY = cache.cellY;
        boolean isDebugMode = (returnType == DendryReturnType.PIXEL_DEBUG);

        for (Segment3D seg : segmentList.getSegments()) {
            // Resolve segment endpoints from SegmentList
            Point3D srtPos = seg.getSrt(segmentList);
            Point3D endPos = seg.getEnd(segmentList);
            PointType srtType = seg.getSrtType(segmentList);
            PointType endType = seg.getEndType(segmentList);

            // Get segment bounds in cell-local coordinates
            double ax = srtPos.x - cellX;
            double ay = srtPos.y - cellY;
            double bx = endPos.x - cellX;
            double by = endPos.y - cellY;

            // Skip if segment is entirely outside the cell (with margin for point radius)
            double margin = isDebugMode ? 3.0 / pixelGridSize : 0;
            if ((ax < -margin && bx < -margin) || (ax >= 1 + margin && bx >= 1 + margin) ||
                (ay < -margin && by < -margin) || (ay >= 1 + margin && by >= 1 + margin)) {
                continue;
            }

            // Sample along the segment at pixel resolution (for segment line)
            double segLength = seg.length(segmentList);
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

                    double x = h00 * srtPos.x + h10 * (seg.tangentSrt != null ? seg.tangentSrt.x * tangentScale : 0)
                             + h01 * endPos.x + h11 * (seg.tangentEnd != null ? seg.tangentEnd.x * tangentScale : 0);
                    double y = h00 * srtPos.y + h10 * (seg.tangentSrt != null ? seg.tangentSrt.y * tangentScale : 0)
                             + h01 * endPos.y + h11 * (seg.tangentEnd != null ? seg.tangentEnd.y * tangentScale : 0);
                    double z = MathUtils.lerp(srtPos.z, endPos.z, t);

                    pt = new Point3D(x, y, z);
                } else {
                    // Linear interpolation
                    pt = seg.lerp(segmentList, t);
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
                        // Use value 0 for segment lines so point types (KNOT=0+, TRUNK=2, etc.) take priority
                        cache.setPixel(px, py, elevation, level, (byte) 0);
                    }
                }
            }

            // In debug mode, mark endpoints with their point types and radius
            if (isDebugMode) {
                // Mark start point
                if (ax >= -0.1 && ax < 1.1 && ay >= -0.1 && ay < 1.1) {
                    int px = (int) (ax * pixelGridSize);
                    int py = (int) (ay * pixelGridSize);
                    float elevation = (float) srtPos.z;
                    byte level = (byte) seg.level;
                    // pointType from PointType enum value
                    byte pointType = (byte) srtType.getValue();
                    cache.markPointWithRadius(px, py, elevation, level, pointType, 2);
                }

                // Mark end point
                if (bx >= -0.1 && bx < 1.1 && by >= -0.1 && by < 1.1) {
                    int px = (int) (bx * pixelGridSize);
                    int py = (int) (by * pixelGridSize);
                    float elevation = (float) endPos.z;
                    byte level = (byte) seg.level;
                    // pointType from PointType enum value
                    byte pointType = (byte) endType.getValue();
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
            // Direct mapping: return pointType
            return data.pointType;
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
            return Double.isNaN(value) ? -1 : (value);
        }

        // CACHE MISS: Cell not computed yet - create/get cache entry and compute
        pixelCacheMisses.incrementAndGet();
        cache = getOrCreatePixelCache(cellX, cellY);

        // Compute all segments for this cell
        SegmentList allSegments = computeAllSegmentsForCell(seed, cellX, cellY);

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
    private SegmentList computeAllSegmentsForCell(long seed, int cellX, int cellY) {
        Cell cell1 = new Cell(cellX, cellY, 1);
        // Use cell center for consistent higher resolution cell lookup
        double queryCenterX = cellX + 0.5;
        double queryCenterY = cellY + 0.5;
        return generateAllSegments(cell1, queryCenterX, queryCenterY);
    }
}

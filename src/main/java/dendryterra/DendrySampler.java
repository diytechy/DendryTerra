package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
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
    private final int maxSegmentsPerLevel;

    // Configuration parameters
    private final int resolution;
    private final double epsilon;
    private final double slope;
    private final double gridsize;
    private final DendryReturnType returnType;
    private final Sampler controlSampler;
    private final long salt;

    // Branch and curvature parameters
    private final Sampler branchesSampler;
    private final int defaultBranches;
    private final double curvature;

    // Performance flags
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
    private static final double DISTANCE_FALLOFF_POWER = 3.0;
    // BranchEncouragementFactor: Multiply slope by this when neighbor has 2+ connections
    // to encourage attaching to existing flows
    private static final double BRANCH_ENCOURAGEMENT_FACTOR = 2.0;
    // Penalty multiplier on distSq for same-level candidates in distance-preferred scoring.
    // Higher values make the algorithm strongly prefer connecting to lower-level segments.
    // 4.0 means a same-level candidate must be 2x closer than a lower-level candidate.
    private static final double SAME_LEVEL_DISTANCE_PENALTY = 1.5;
    // Power exponent for slope influence in blended scoring (0 = pure distance, 1 = full slope influence).
    // Lower values compress slope's dynamic range, making distance dominate connection preference.
    private static final double SLOPE_INFLUENCE = 0.5;
    /**
     * Use B-spline (cubic Hermite) interpolation for pixel sampling in PIXEL_DEBUG mode.
     * When true, segments with tangent information will be sampled along the curved spline.
     * When false, uses linear interpolation between endpoints.
     */
    private static final boolean USE_BSPLINE_PIXEL_SAMPLING = true;

    /**
     * When true, segmentFill (semicircle fill) is enabled for ALL segment start/end points,
     * not just endpoints with exactly 1 connection.
     */
    private static final boolean ENABLE_SEGMENT_FILL_ALL = true;

    // Star sampling grid size (currently 3x3 grid per cell)
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

    // River width parameters
    private final Sampler riverwidthSampler;    // Sampler for river width at a point
    private final double defaultRiverwidth;      // Default river width when no sampler
    private final Sampler borderwidthSampler;   // Sampler for border width around rivers
    private final double defaultBorderwidth;    // Default border width when no sampler
    private static final double RIVER_WIDTH_FALLOFF = 0.6;  // River width reduction per level

    // PIXEL_RIVER parameters
    private final double max;         // Maximum expected elevation for normalization
    private final double maxDistGrid; // Maximum distance in grid coordinates (maxDist / gridsize)

    // Cache configuration
    private static final int MAX_PIXEL_CACHE_BYTES = 20 * 1024 * 1024; // 20 MB max for pixel cache

    // PIXEL_RIVER caches (10 MB each, only allocated for PIXEL_RIVER/PIXEL_RIVER_CTRL)
    private final SegmentListCache segmentListCache;
    private final BigChunkCache bigChunkCache;

    // Timing statistics (only used when debugTiming is true)
    private final AtomicLong sampleCount = new AtomicLong(0);
    private final AtomicLong totalTimeNs = new AtomicLong(0);
    private volatile long lastLogTime = 0;
    private static final long LOG_INTERVAL_MS = 5000; // Log every 5 seconds

    // Pixel cache statistics (for debugging cache performance)
    private final AtomicLong pixelCacheHits = new AtomicLong(0);
    private final AtomicLong pixelCacheMisses = new AtomicLong(0);

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

    public DendrySampler(int resolution, double epsilon,
                         double slope, double gridsize,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt,
                         Sampler branchesSampler, int defaultBranches,
                         double curvature,
                         boolean useParallel,
                         boolean debugTiming, int parallelThreshold,
                         int ConstellationScale, ConstellationShape constellationShape,
                         double tangentAngle, double tangentStrength,
                         double cachepixels,
                         double slopeWhenStraight, double lowestSlopeCutoff,
                         int debug,
                         Sampler riverwidthSampler, double defaultRiverwidth,
                         Sampler borderwidthSampler, double defaultBorderwidth,
                         double max, double maxDist,
                         int maxSegmentsPerLevel) {
        this.resolution = resolution;
        this.epsilon = epsilon;
        this.slope = slope;
        this.gridsize = gridsize;
        this.returnType = returnType;
        this.controlSampler = controlSampler;
        this.salt = salt;
        this.branchesSampler = branchesSampler;
        this.defaultBranches = defaultBranches;
        this.curvature = curvature;
        this.useParallel = useParallel;
        this.useSplines = curvature != 0;
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
        this.riverwidthSampler = riverwidthSampler;
        this.defaultRiverwidth = defaultRiverwidth;
        this.borderwidthSampler = borderwidthSampler;
        this.defaultBorderwidth = defaultBorderwidth;
        this.max = max;
        this.maxSegmentsPerLevel = maxSegmentsPerLevel;
        this.maxDistGrid = maxDist / gridsize;  // Convert from sampler to grid coordinates

        // Calculate pixel grid size
        if (cachepixels > 0) {
            this.pixelGridSize = (int) Math.ceil(gridsize / cachepixels);
        } else {
            this.pixelGridSize = 0;
        }

        // Initialize pixel cache only for return types that use cell-based pixel caching
        if (cachepixels > 0 && (returnType == DendryReturnType.PIXEL_LEVEL
                || returnType == DendryReturnType.PIXEL_ELEVATION)) {
            this.pixelCache = new HashMap<>();
        } else {
            this.pixelCache = null;
        }

        // Initialize PIXEL_RIVER caches only when using BigChunk-based return types
        if (returnType == DendryReturnType.PIXEL_RIVER || returnType == DendryReturnType.PIXEL_RIVER_CTRL) {
            this.segmentListCache = new SegmentListCache();
            this.bigChunkCache = new BigChunkCache();
        } else {
            this.segmentListCache = null;
            this.bigChunkCache = null;
        }

        if (debugTiming) {
            LOGGER.info("DendrySampler initialized with: resolution={}, gridsize={}, useParallel={}, useSplines={}, parallelThreshold={}, cachepixels={}, pixelGridSize={}",
                resolution, gridsize, useParallel, useSplines, parallelThreshold, cachepixels, pixelGridSize);
        }
    }

    private static long packKey(int x, int y) {
        return ((long) x << 32) | (y & 0xFFFFFFFFL);
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

        // Use bigchunk cache for PIXEL_RIVER and PIXEL_RIVER_CTRL return types
        if (returnType == DendryReturnType.PIXEL_RIVER) {
            result = evaluateWithBigChunkDistance(normalizedX, normalizedZ);
        } else if (returnType == DendryReturnType.PIXEL_RIVER_CTRL) {
            result = evaluateWithBigChunkElevation(normalizedX, normalizedZ);
        }
        // Use pixel cache for PIXEL_ELEVATION and PIXEL_LEVEL return types
        else if (usesPixelCache()) {
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
        // When returnType is PIXEL_RIVER and y == 1.0, return elevation instead of distance
        if (returnType == DendryReturnType.PIXEL_RIVER && y == 1.0) {
            return evaluateWithBigChunkElevation(x / gridsize, z / gridsize);
        }
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
        // Asterism (Level 0): Generate and process
        SegmentList asterismBase = generateAsterism(cell1);
        // Prune asterism to query cell - clips segments at cell boundary with EDGE points
        SegmentList asterismPruned = pruneSegmentsToCell(asterismBase, cell1);

        if (resolution == 0) {
            return asterismPruned;
        }

        // Level 1+: Higher resolution refinement using loop
        // Each level generates points spanning the entire world cell (cell1) at increasing density
        SegmentList previousSegList = asterismPruned;

        for (int level = 1; level <= resolution; level++) {
            // Generate points for this level spanning the entire world cell
            List<Point3D> levelPoints = generatePointsForWorldCell(cell1.x, cell1.y, level);

            // Use CleanAndNetworkPointsV2 to create properly connected segments
            // This handles merging, cleaning, tangent computation, subdivision, and displacement
            SegmentList levelSegList = CleanAndNetworkPointsV2(
                cell1.x, cell1.y, level, levelPoints, previousSegList);

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

        for (SegmentIdx seg : segmentList.getSegments()) {
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

        // Post-processing: Propagate EDGE type one step inward from boundary points
        // This prevents higher-level segments from connecting near boundaries
        if (result.getSegmentCount() > 4) {
            propagateEdgeTypeOneStep(result);
        }

        return result;
    }

    /**
     * Propagate EDGE type one step inward from existing EDGE points.
     * For each segment where one endpoint is EDGE (with 1 connection) and the other is non-EDGE,
     * mark the non-EDGE point as EDGE only if it has another connected segment with two non-EDGE points.
     * This ensures we don't orphan segments while preventing connections near boundaries.
     */
    private void propagateEdgeTypeOneStep(SegmentList segList) {
        List<SegmentIdx> segments = segList.getSegments();
        List<NetworkPoint> points = segList.getPoints();

        // Collect points to mark as EDGE (don't modify while iterating)
        Set<Integer> pointsToMark = new HashSet<>();

        for (SegmentIdx seg : segments) {
            NetworkPoint srtPt = points.get(seg.srtIdx);
            NetworkPoint endPt = points.get(seg.endIdx);

            // Check if exactly one endpoint is EDGE with only 1 connection
            int edgeIdx = -1;
            int nonEdgeIdx = -1;

            if (srtPt.pointType == PointType.EDGE && srtPt.connections == 1 &&
                endPt.pointType != PointType.EDGE) {
                edgeIdx = seg.srtIdx;
                nonEdgeIdx = seg.endIdx;
            } else if (endPt.pointType == PointType.EDGE && endPt.connections == 1 &&
                       srtPt.pointType != PointType.EDGE) {
                edgeIdx = seg.endIdx;
                nonEdgeIdx = seg.srtIdx;
            }

            if (nonEdgeIdx < 0) continue;

            // Check if the non-EDGE point has another segment with two non-EDGE points
            NetworkPoint nonEdgePt = points.get(nonEdgeIdx);
            if (nonEdgePt.connections < 2) continue;  // Must have at least 2 connections

            boolean hasNonEdgeSegment = false;
            for (SegmentIdx otherSeg : segments) {
                if (otherSeg == seg) continue;

                // Check if this segment connects to the non-EDGE point
                int otherIdx = -1;
                if (otherSeg.srtIdx == nonEdgeIdx) {
                    otherIdx = otherSeg.endIdx;
                } else if (otherSeg.endIdx == nonEdgeIdx) {
                    otherIdx = otherSeg.srtIdx;
                }

                if (otherIdx < 0) continue;

                // Check if the other endpoint is non-EDGE
                NetworkPoint otherPt = points.get(otherIdx);
                if (otherPt.pointType != PointType.EDGE) {
                    hasNonEdgeSegment = true;
                    break;
                }
            }

            if (hasNonEdgeSegment) {
                pointsToMark.add(nonEdgeIdx);
            }
        }

        // Mark collected points as EDGE
        for (int idx : pointsToMark) {
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
    private void clipSegmentToCell(SegmentList source, SegmentIdx seg, SegmentList result, Map<Integer, Integer> pointIndexMap,
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
    private void clipSegmentBothEnds(SegmentList source, SegmentIdx seg, SegmentList result,
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
    private boolean segmentCrossesCell(SegmentList source, SegmentIdx seg, double minX, double maxX, double minY, double maxY) {
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


    /**
     * Generate points for a single cell at a given level using DIVISIONS_PER_LEVEL configuration.
     * Creates a grid of points within the cell based on getPointsPerCellForLevel().
     *
     * @param worldCellX World cell X coordinate (level 1 cell)
     * @param worldCellY World cell Y coordinate (level 1 cell)
     * @param level The resolution level (determines point density via POINTS_PER_CELL[level])
     * @return List of 3D points spanning the entire world cell
     */
    private List<Point3D> generatePointsForWorldCell(int worldCellX, int worldCellY, int level) {
        List<Point3D> points = new ArrayList<>();
        int pointsPerAxis = getPointsPerCellForLevel(level);

        // Grid spacing: world cell is 1x1, divide by points per axis
        double gridSpacing = 1.0 / pointsPerAxis;

        // Generate pointsPerAxis^2 points spanning the entire world cell [worldCellX, worldCellX+1)
        for (int i = 0; i < pointsPerAxis; i++) {
            for (int j = 0; j < pointsPerAxis; j++) {
                double cellMinX = worldCellX + j * gridSpacing;
                double cellMinY = worldCellY + i * gridSpacing;

                // Deterministic jitter offset for the probe grid
                Random rng = initRandomGenerator((int)(cellMinX * 10000), (int)(cellMinY * 10000), level);
                double jitterX = rng.nextDouble() * 0.5;  // [0, 0.5) offset for probe grid
                double jitterY = rng.nextDouble() * 0.5;

                // Probe 4x4 grid within sub-cell, find lowest elevation
                double boundary = 0.05;  // 5% inset from sub-cell edges
                double lowestElev = Double.MAX_VALUE;
                double bestX = cellMinX + 0.5 * gridSpacing;
                double bestY = cellMinY + 0.5 * gridSpacing;

                int probeGrid = 4;
                for (int pi = 0; pi < probeGrid; pi++) {
                    for (int pj = 0; pj < probeGrid; pj++) {
                        double tx = boundary + (pj + jitterX) / probeGrid * (1.0 - 2 * boundary);
                        double ty = boundary + (pi + jitterY) / probeGrid * (1.0 - 2 * boundary);
                        double px = cellMinX + tx * gridSpacing;
                        double py = cellMinY + ty * gridSpacing;
                        double elev = evaluateControlFunction(px, py);
                        if (elev < lowestElev) {
                            lowestElev = elev;
                            bestX = px;
                            bestY = py;
                        }
                    }
                }

                points.add(new Point3D(bestX, bestY, lowestElev));
            }
        }

        return points;
    }

    /** Minimum angle (in radians) between two neighbors for slope estimation. */
    private static final double MIN_NEIGHBOR_ANGLE_RAD = Math.toRadians(30);


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
            for (SegmentIdx seg : segList.getSegments()) {
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

        // Step 4: Compute slopes for each star based on neighboring stars
        List<Point3D> starsWithSlopes = computeSlopesForPoints(boundedStars);

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
     * Merge points that are within mergeDistance of each other.
     * Uses an iterative epicenter algorithm:
     * 1. Find the point with the most neighbors within merge distance (epicenter)
     * 2. Merge epicenter and all its close neighbors into a single averaged point
     * 3. Evaluate control function to get new elevation
     * 4. Repeat until no points are within merge distance of each other
     *
     * @param points List of points to merge
     * @param mergeDistance Minimum distance between kept points
     * @return List of merged points with proper spacing
     */
    private List<Point3D> mergePointsByDistance(List<Point3D> points, double mergeDistance) {
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
        double maxSegmentDistance = Double.MAX_VALUE;
        if(level >0){
            maxSegmentDistance = MAX_POINT_SEGMENT_DISTANCE * gridSpacing;
        }
        else{
            mergeDistance = mergeDistance/2;
        }

        // DEBUG 40: Track point counts at each stage for the highest level
        int draftedCount = points.size();
        int afterMergeCount = 0;
        int afterNearSegmentsCount = 0;
        int afterProbabilisticCount = 0;
        boolean isDebug40TargetLevel = (debug == 40 && level > 0 && level == resolution);

        // Step 1: Clean network points - merge points within merge distance (only for level > 0)
        List<Point3D> cleanedPoints;
        if (level > 0) {
            cleanedPoints = mergePointsByDistance(points, mergeDistance);
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
        double maxDist = maxSegmentDistance;
        double mergeDist = mergeDistance;

        // Phase A: For level 0, build trunk from lowest point
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
        // Note: We include ALL points in the sort, not just those within maxDistSq.
        // This allows chain growth - points far from the initial trunk can still connect
        // to newly added points. The actual distance check happens in findBestNeighborV2.
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

            // Include all points in sort (removed the maxDistSq filter)
            // Store as [unconnIdx, distance * 1000000 as int for sorting]
            sortedByDistance.add(new int[]{unconnIdx, (int)(minDistSq * 1000000)});
        }

        // Step 2: Sort by initial distance (closest first)
        sortedByDistance.sort((a, b) -> Integer.compare(a[1], b[1]));

        // Step 3: Process points in sorted order (closest to furthest from initial segment list)
        // Track segment budget for highest level when maxSegmentsPerLevel is set
        boolean budgetLimited = (level == resolution && maxSegmentsPerLevel < Integer.MAX_VALUE);
        int initialSegCount = budgetLimited ? segList.getSegmentCount() : 0;

        for (int[] entry : sortedByDistance) {
            int unconnIdx = entry[0];

            // Skip if already removed (shouldn't happen but safety check)
            if (unconnected.isRemoved(unconnIdx)) continue;

            // Check budget before attempting connection
            if (budgetLimited) {
                int segmentsCreated = segList.getSegmentCount() - initialSegCount;
                if (segmentsCreated >= maxSegmentsPerLevel) break;
            }

            NetworkPoint unconnPt = unconnected.getPoint(unconnIdx);

            // Find best neighbor in SegmentList for this point
            int neighborIdx = findBestNeighborV2(unconnPt, segList, maxDist, mergeDist, level);

            if (neighborIdx < 0) {
                unconnected.markRemoved(unconnIdx);
                continue;
            }

            // Save state for potential rollback if crossing detected
            int pointsBefore = segList.getPointCount();
            int segsBefore = segList.getSegmentCount();

            // Create connection: move point to SegmentList and create segment
            unconnected.markRemoved(unconnIdx);
            if (budgetLimited) {
                int segBudget = maxSegmentsPerLevel - (segList.getSegmentCount() - initialSegCount);
                segList.addSegmentWithDivisions(unconnPt, neighborIdx, level, mergeDistance, segBudget);
            } else {
                segList.addSegmentWithDivisions(unconnPt, neighborIdx, level, mergeDistance);
            }

            // Check for crossings between newly added segments and pre-existing ones
            boolean hasCrossing = false;
            List<SegmentIdx> allSegs = segList.getSegments();
            for (int i = segsBefore; i < segList.getSegmentCount() && !hasCrossing; i++) {
                SegmentIdx newSeg = allSegs.get(i);
                Point2D newA = newSeg.getSrt(segList).projectZ();
                Point2D newB = newSeg.getEnd(segList).projectZ();
                for (int j = 0; j < segsBefore; j++) {
                    SegmentIdx existing = allSegs.get(j);
                    Point2D exA = existing.getSrt(segList).projectZ();
                    Point2D exB = existing.getEnd(segList).projectZ();
                    if (segmentsIntersect(newA, newB, exA, exB)) {
                        hasCrossing = true;
                        break;
                    }
                }
            }
            if (hasCrossing) {
                segList.rollback(pointsBefore, segsBefore);
            }
        }
    }

    /**
     * V2: Build trunk for level 0 asterisms.
     * Starts at lowest point and extends uphill until no more connections possible.
     */
    private void buildTrunkV2(UnconnectedPoints unconnected, SegmentList segList,
                               double maxSegmentDistance, double mergeDistance, int level, int cellX, int cellY) {
        double maxDist = maxSegmentDistance;

        // Find lowest unconnected point to start trunk
        int startIdx = unconnected.findLowestUnconnected();
        if (startIdx < 0) return;

        // Get first trunk point and add it to the segment list
        NetworkPoint startPt = unconnected.removeAndGet(startIdx);
        NetworkPoint trunkStartPt = new NetworkPoint(startPt.position, -1, PointType.TRUNK, level);

        int maxIterations = unconnected.totalSize() + 1;
        int iterations = 0;
        boolean firstSegment = true;

        // Track which point indices are eligible for branching (upper half of latest segment)
        List<Integer> searchableIndices = new ArrayList<>();

        // Extend trunk uphill — each iteration can branch from the upper half of the latest segment
        while (iterations < maxIterations) {
            iterations++;

            int bestUnconnIdx = -1;
            int bestTrunkIdx = -1;
            double bestSlope = 0;

            if (firstSegment) {
                // Only one candidate trunk point: the start point (not yet in segList)
                bestUnconnIdx = findBestTrunkNeighbor(unconnected, trunkStartPt.position, maxDist, level);
                bestTrunkIdx = -1; // sentinel: use trunkStartPt
            } else {
                // Search only from the upper-half points of the latest segment
                for (int idx : searchableIndices) {
                    NetworkPoint trunkPt = segList.getPoint(idx);

                    int candidateIdx = findBestTrunkNeighbor(unconnected, trunkPt.position, maxDist, level);
                    if (candidateIdx < 0) continue;

                    NetworkPoint candidate = unconnected.getPoint(candidateIdx);
                    double dist = trunkPt.position.projectZ().distanceTo(candidate.position.projectZ());
                    if (dist < MathUtils.EPSILON) continue;
                    double slope = -calculateNormalizedSlope(trunkPt.position.z, candidate.position.z, dist, level);
                    if (slope > bestSlope) {
                        bestSlope = slope;
                        bestUnconnIdx = candidateIdx;
                        bestTrunkIdx = idx;
                    }
                }
            }

            if (bestUnconnIdx < 0) {
                if (firstSegment) {
                    segList.addPoint(trunkStartPt.position, PointType.TRUNK, level);
                }
                break;
            }

            // Get next trunk point
            NetworkPoint nextPt = unconnected.removeAndGet(bestUnconnIdx);
            NetworkPoint trunkNextPt = new NetworkPoint(nextPt.position, -1, PointType.TRUNK, level);

            // Record point count before adding so we can identify new points
            int pointsBefore = segList.getPointCount();

            if (firstSegment) {
                segList.addSegmentWithDivisions(trunkStartPt, trunkNextPt, level, mergeDistance);
                firstSegment = false;
            } else {
                segList.addSegmentWithDivisions(trunkNextPt, bestTrunkIdx, level, mergeDistance);
            }

            // Determine searchable indices for next iteration: upper half of newly created points
            // New points are in range [pointsBefore, segList.getPointCount())
            // The endpoint (trunkNextPt, the uphill end) is always included.
            // "Upper half" = the points closer to the end (higher indices in the subdivision chain)
            int pointsAfter = segList.getPointCount();
            int newPointCount = pointsAfter - pointsBefore;
            int upperHalfStart = pointsBefore + (newPointCount / 2);  // skip lower half

            searchableIndices.clear();
            for (int i = upperHalfStart; i < pointsAfter; i++) {
                searchableIndices.add(i);
            }
        }

        // Force all trunk points to elevation 0 for flattened path preferences
        segList.forceAllPointElevations(0);
    }

    /**
     * Calculate normalized slope between two points with distance falloff.
     *
     * @param sourceZ Source point elevation
     * @param candidateZ Candidate point elevation
     * @param dist 2D distance between points
     * @param level Current level
     * @return Normalized slope (sourceZ - candidateZ) / dist^DISTANCE_FALLOFF_POWER
     */
    private double calculateNormalizedSlope(double sourceZ, double candidateZ, double dist, int level) {
        double heightDiff = sourceZ - candidateZ;
        return heightDiff / Math.pow(dist, DISTANCE_FALLOFF_POWER);
    }

    /**
     * Find best uphill trunk neighbor from unconnected points.
     * Searches for the steepest uphill candidate within maxDist.
     */
    private int findBestTrunkNeighbor(UnconnectedPoints unconnected, Point3D currentPos,
                                       double maxDist, int level) {
        Point2D currentPos2D = currentPos.projectZ();
        double bestSlope = 0;  // Must be positive (uphill)
        int bestUnconnIdx = -1;

        List<Integer> remaining = unconnected.getRemainingIndices();
        for (int unconnIdx : remaining) {
            NetworkPoint candidate = unconnected.getPoint(unconnIdx);
            if (candidate.pointType == PointType.EDGE) continue;

            Point2D candidatePos = candidate.position.projectZ();
            double dist = currentPos2D.distanceTo(candidatePos);
            if (dist > maxDist || dist < MathUtils.EPSILON) continue;
            // For trunk, we want uphill: candidate - current (inverted from normal)
            double normalizedSlope = -calculateNormalizedSlope(currentPos.z, candidate.position.z, dist, level);
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
     * Uses two priority tiers:
     *   A. Merge distance: closest candidate within merge range
     *   B. Blended score: dist * levelPenalty / slopeFactor (lower = better)
     *      - Prefers closer candidates (small dist)
     *      - Prefers steeper downhill candidates (large slopeFactor)
     *      - Prefers lower-level candidates (levelPenalty 1.0 vs SAME_LEVEL_DISTANCE_PENALTY)
     *      - Branch encouragement boosts slope for already-connected candidates
     */
    private int findBestNeighborV2(NetworkPoint sourcePt, SegmentList segList,
                                    double maxDist, double mergeDist, int level) {
        Point2D sourcePos = sourcePt.position.projectZ();

        // Priority A: closest within merge distance
        double bestMergeDist = Double.MAX_VALUE;
        int bestNeighborWithinMerge = -1;

        // Priority B: blended distance + slope + level score (lower = better)
        double bestBlendedScore = Double.MAX_VALUE;
        int bestBlendedNeighbor = -1;

        for (int i = 0; i < segList.getPointCount(); i++) {
            NetworkPoint candidate = segList.getPoint(i);
            if (candidate.pointType == PointType.EDGE) continue;
            if (candidate.connections >= 5) continue;

            Point2D candidatePos = candidate.position.projectZ();
            double dist = sourcePos.distanceTo(candidatePos);
            if (dist > maxDist || dist < MathUtils.EPSILON) continue;

            double normalizedSlope = calculateNormalizedSlope(
                sourcePt.position.z, candidate.position.z, dist, level);

            // Level 1+ has slope cutoff
            if (level > 0 && normalizedSlope < lowestSlopeCutoff) continue;

            // Priority A: merge distance
            if (dist <= mergeDist) {
                if (dist < bestMergeDist) {
                    bestMergeDist = dist;
                    bestNeighborWithinMerge = i;
                }
            }

            // Priority B: blended score
            // slopeFactor: how far above the cutoff this candidate is (always positive)
            double slopeFactor = normalizedSlope - lowestSlopeCutoff + MathUtils.EPSILON;
            if (candidate.connections >= 1) {
                slopeFactor *= BRANCH_ENCOURAGEMENT_FACTOR;
            }
            double levelPenalty = (candidate.level < level) ? 1.0 : SAME_LEVEL_DISTANCE_PENALTY;
            double blendedScore = dist * levelPenalty / Math.pow(slopeFactor, SLOPE_INFLUENCE);
            if (blendedScore < bestBlendedScore) {
                bestBlendedScore = blendedScore;
                bestBlendedNeighbor = i;
            }
        }

        // Priority: merge > blended
        if (bestNeighborWithinMerge >= 0) return bestNeighborWithinMerge;
        return bestBlendedNeighbor;
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

            for (SegmentIdx seg : segmentList.getSegments()) {
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
     * Test if two line segments intersect in 2D (excluding shared endpoints).
     * Uses parametric intersection with epsilon margins to avoid false positives
     * at endpoints where segments naturally meet.
     */
    private boolean segmentsIntersect(Point2D a1, Point2D a2, Point2D b1, Point2D b2) {
        double d1x = a2.x - a1.x, d1y = a2.y - a1.y;
        double d2x = b2.x - b1.x, d2y = b2.y - b1.y;
        double cross = d1x * d2y - d1y * d2x;
        if (Math.abs(cross) < MathUtils.EPSILON) return false; // parallel
        double t = ((b1.x - a1.x) * d2y - (b1.y - a1.y) * d2x) / cross;
        double u = ((b1.x - a1.x) * d1y - (b1.y - a1.y) * d1x) / cross;
        return t > 0.001 && t < 0.999 && u > 0.001 && u < 0.999;
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

                for (SegmentIdx seg : previousLevelSegments.getSegments()) {
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
     * Calculate river width for a given level.
     * River width decreases by RIVER_WIDTH_FALLOFF (0.6) per level.
     * @param level The segment level (0-5)
     * @param x World X coordinate for sampling
     * @param y World Y coordinate for sampling
     * @return River width in world units
     */
    /**
     * Calculate river width at a given position and level.
     * @param level River level (0 = main river, higher = tributaries)
     * @param gridX Grid X coordinate (normalized)
     * @param gridY Grid Y coordinate (normalized)
     * @return River width in grid coordinates
     */
    private double calculateRiverWidth(int level, double gridX, double gridY) {
        // Convert grid coordinates to sampler coordinates for the width sampler
        double samplerX = gridX * gridsize;
        double samplerY = gridY * gridsize;

        // Get base width in sampler coordinates
        double baseWidthSampler = (riverwidthSampler != null)
            ? riverwidthSampler.getSample(salt, samplerX, 0, samplerY)
            : defaultRiverwidth;

        // Convert to grid coordinates and apply level falloff
        double baseWidthGrid = baseWidthSampler / gridsize;
        return baseWidthGrid * Math.pow(RIVER_WIDTH_FALLOFF, level);
    }


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





    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            return controlSampler.getSample(salt, x * gridsize, y * gridsize);
        }
        return x * 0.1;
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
     * Find nearest segment by a given metric - uses parallel streams if enabled and list is large enough.
     */
    private NearestSegmentResult findNearestSegmentBy(Point2D point, List<Segment3D> segments,
                                                       java.util.function.ToDoubleFunction<NearestSegmentResult> metric) {
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
                double weightedDist = result.distance;
                return new NearestSegmentResult(
                    result.distance,
                    weightedDist,
                    result.closestPoint,
                    new Point3D(result.closestPoint.x, result.closestPoint.y, z),
                    seg
                );
            })
            .min(Comparator.comparingDouble(metric))
            .orElse(null);
    }

    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        return findNearestSegmentBy(point, segments, r -> r.distance);
    }

    private NearestSegmentResult findNearestSegmentWeighted(Point2D point, List<Segment3D> segments) {
        return findNearestSegmentBy(point, segments, r -> r.weightedDistance);
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
                returnType == DendryReturnType.PIXEL_DEBUG ||
                returnType == DendryReturnType.PIXEL_RIVER_LEGACY);
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

        for (SegmentIdx seg : segmentList.getSegments()) {
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
        } else if (returnType == DendryReturnType.PIXEL_RIVER_LEGACY) {
            // River mode: we're on a segment, so distance is 0 -> return 0 (river)
            return 0;
        } else {
            // PIXEL_LEVEL
            return data.level;
        }
    }

    /**
     * Get river width at a point, applying level-based falloff.
     * @param worldX World X coordinate (normalized)
     * @param worldY World Y coordinate (normalized)
     * @param level Segment level (0-5)
     * @return River width in normalized coordinates
     */
    private double getRiverWidth(double worldX, double worldY, int level) {
        double baseWidth;
        if (riverwidthSampler != null) {
            baseWidth = riverwidthSampler.getSample(salt, worldX * gridsize, worldY * gridsize);
        } else {
            baseWidth = defaultRiverwidth;
        }
        // Apply level-based falloff: width * (0.6^level)
        double levelWidth = baseWidth * Math.pow(RIVER_WIDTH_FALLOFF, level);
        // Minimum width is 2x pixel resolution
        double minWidth = 2.0 * cachepixels;
        return Math.max(levelWidth, minWidth) / gridsize;  // Normalize to cell units
    }

    /**
     * Get border width at a point.
     * @param worldX World X coordinate (normalized)
     * @param worldY World Y coordinate (normalized)
     * @return Border width in normalized coordinates
     */
    private double getBorderWidth(double worldX, double worldY) {
        double width;
        if (borderwidthSampler != null) {
            width = borderwidthSampler.getSample(salt, worldX * gridsize, worldY * gridsize);
        } else {
            width = defaultBorderwidth;
        }
        return width / gridsize;  // Normalize to cell units
    }

    /**
     * Find the nearest segment pixel within a search radius.
     * Searches the current cell and adjacent cells if needed.
     * @param worldX World X coordinate (normalized)
     * @param worldY World Y coordinate (normalized)
     * @param maxSearchRadius Maximum search radius in normalized coordinates
     * @return Array of [distance, level] or null if no segment found
     */
    private double[] findNearestSegmentPixel(double worldX, double worldY, double maxSearchRadius) {
        int cellX = MathUtils.floor(worldX);
        int cellY = MathUtils.floor(worldY);

        // Convert search radius to pixels
        int maxPixelRadius = (int) Math.ceil(maxSearchRadius * pixelGridSize) + 1;

        // Get local pixel coordinates
        double localX = worldX - cellX;
        double localY = worldY - cellY;
        int centerPx = (int) (localX * pixelGridSize);
        int centerPy = (int) (localY * pixelGridSize);

        double bestDistSq = Double.MAX_VALUE;
        int bestLevel = -1;

        // Search in expanding rings
        for (int radius = 0; radius <= maxPixelRadius; radius++) {
            // If we've found a segment within current radius, stop
            if (bestDistSq <= radius * radius) {
                break;
            }

            // Search ring at this radius
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    // Only process pixels on the ring edge (optimization)
                    if (Math.abs(dx) != radius && Math.abs(dy) != radius) {
                        continue;
                    }

                    int px = centerPx + dx;
                    int py = centerPy + dy;

                    // Determine which cell this pixel is in
                    int checkCellX = cellX;
                    int checkCellY = cellY;
                    int checkPx = px;
                    int checkPy = py;

                    // Handle crossing into adjacent cells
                    if (px < 0) {
                        checkCellX = cellX - 1;
                        checkPx = px + pixelGridSize;
                    } else if (px >= pixelGridSize) {
                        checkCellX = cellX + 1;
                        checkPx = px - pixelGridSize;
                    }
                    if (py < 0) {
                        checkCellY = cellY - 1;
                        checkPy = py + pixelGridSize;
                    } else if (py >= pixelGridSize) {
                        checkCellY = cellY + 1;
                        checkPy = py - pixelGridSize;
                    }

                    // Get cache for this cell
                    CellPixelData cache = getCachedPixelData(checkCellX, checkCellY);
                    if (cache == null || !cache.populated) {
                        continue;
                    }

                    PixelData data = cache.getPixel(checkPx, checkPy);
                    if (data != null) {
                        // Calculate distance in normalized coordinates
                        double pixelWorldX = checkCellX + (checkPx + 0.5) / pixelGridSize;
                        double pixelWorldY = checkCellY + (checkPy + 0.5) / pixelGridSize;
                        double distSq = (worldX - pixelWorldX) * (worldX - pixelWorldX) +
                                       (worldY - pixelWorldY) * (worldY - pixelWorldY);

                        if (distSq < bestDistSq) {
                            bestDistSq = distSq;
                            bestLevel = data.level;
                        }
                    }
                }
            }
        }

        if (bestLevel >= 0) {
            return new double[] { Math.sqrt(bestDistSq), bestLevel };
        }
        return null;
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
            // If pixel has data, return it
            if (!Double.isNaN(value)) {
                return value;
            }
            // Pixel is empty - handle based on return type
            if (returnType == DendryReturnType.PIXEL_RIVER_LEGACY) {
                return evaluateRiverDistance(x, y);
            }
            return -1;
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
        // If pixel has data, return it
        if (!Double.isNaN(value)) {
            return value;
        }
        // Pixel is empty - handle based on return type
        if (returnType == DendryReturnType.PIXEL_RIVER_LEGACY) {
            return evaluateRiverDistance(x, y);
        }
        return 0;
    }

    /**
     * Evaluate river/border classification for a point not directly on a segment.
     * Searches for nearest segment and compares distance to river/border widths.
     * @param x Normalized X coordinate
     * @param y Normalized Y coordinate
     * @return 0 = river, 1 = border, 2 = outside
     */
    private double evaluateRiverDistance(double x, double y) {
        // Get max search radius (river width + border width at level 0)
        double maxRiverWidth = getRiverWidth(x, y, 0);
        double borderWidth = getBorderWidth(x, y);
        double maxSearchRadius = maxRiverWidth + borderWidth;

        // Find nearest segment pixel
        double[] nearest = findNearestSegmentPixel(x, y, maxSearchRadius);
        if (nearest == null) {
            // No segment found within search radius - outside river/border
            return 2;
        }

        double distance = nearest[0];
        int level = (int) nearest[1];

        // Get river width for this level
        double riverWidth = getRiverWidth(x, y, level);

        // Classify based on distance
        if (distance < riverWidth) {
            return 0;  // Within river
        } else if (distance < riverWidth + borderWidth) {
            return 1;  // Within border
        } else {
            return 2;  // Outside
        }
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

    // ========== PIXEL_RIVER Implementation ==========

    /**
     * Evaluate using the BigChunk cache system for PIXEL_RIVER return type.
     * Returns the de-quantized distance value from the cached bigchunk.
     * @param gridX Grid X coordinate (normalized)
     * @param gridY Grid Y coordinate (normalized)
     * @return Distance value in sampler coordinates
     */
    /**
     * Look up the BigChunk block at the given grid coordinates.
     */
    private BigChunk.BigChunkBlock getBigChunkBlock(double gridX, double gridY) {
        BigChunk chunk = getOrCreateBigChunk(gridX, gridY);
        double cachepixelsGrid = cachepixels / gridsize;
        int blockX = Math.max(0, Math.min(255, gridToBlockIndex(gridX, chunk.gridOriginX, cachepixelsGrid)));
        int blockY = Math.max(0, Math.min(255, gridToBlockIndex(gridY, chunk.gridOriginY, cachepixelsGrid)));
        return chunk.getBlock(blockX, blockY);
    }

    private double evaluateWithBigChunkDistance(double gridX, double gridY) {
        BigChunk.BigChunkBlock block = getBigChunkBlock(gridX, gridY);
        double distQuantizeRes = 255.0 / maxDistGrid;
        return (block.getDistanceUnsigned() / distQuantizeRes) * gridsize;
    }

    private double evaluateWithBigChunkElevation(double gridX, double gridY) {
        BigChunk.BigChunkBlock block = getBigChunkBlock(gridX, gridY);
        double elevQuantizeRes = 255.0 / max;
        return block.getElevationUnsigned() / elevQuantizeRes;
    }

    /**
     * Get the size of a bigchunk in grid coordinate units.
     * A bigchunk is 256 blocks, each block is cachepixels sized.
     * cachepixels is in sampler coordinates, so we convert to grid coordinates.
     */
    private double getBigChunkSizeGrid() {
        double cachepixelsGrid = cachepixels / gridsize;
        return cachepixelsGrid * 256;
    }

    /**
     * Convert grid coordinates to chunk coordinates.
     * @param gridCoord Grid X or Y coordinate (sampler coord / gridsize)
     * @return Integer chunk coordinate
     */
    private int gridToChunkCoord(double gridCoord) {
        double chunkSizeGrid = getBigChunkSizeGrid();
        return (int) Math.floor(gridCoord / chunkSizeGrid);
    }

    /**
     * Convert chunk coordinates to grid coordinates (chunk origin).
     * @param chunkCoord Integer chunk coordinate
     * @return Grid coordinate of chunk origin (in normalized space)
     */
    private double chunkToGridCoord(int chunkCoord) {
        double chunkSizeGrid = getBigChunkSizeGrid();
        return chunkCoord * chunkSizeGrid;
    }

    /**
     * Get or create a BigChunk for the specified grid coordinates.
     * @param gridX Grid X coordinate (sampler X / gridsize)
     * @param gridY Grid Y coordinate (sampler Y / gridsize)
     * @return BigChunk containing the specified grid position
     */
    private BigChunk getOrCreateBigChunk(double gridX, double gridY) {
        // Convert grid coordinates to chunk coordinates
        int chunkX = gridToChunkCoord(gridX);
        int chunkY = gridToChunkCoord(gridY);

        // Get grid origin of this chunk
        double gridOriginX = chunkToGridCoord(chunkX);
        double gridOriginY = chunkToGridCoord(chunkY);

        // Get or create chunk from cache
        BigChunk chunk = bigChunkCache.getOrCreate(chunkX, chunkY, gridOriginX, gridOriginY);

        // Compute if not already done
        if (!chunk.computed) {
            computeBigChunk(chunk);
            chunk.computed = true;
        }

        return chunk;
    }

    /**
     * Compute all block values for a BigChunk.
     * This is the main PIXEL_RIVER computation method.
     * All coordinates are in grid coordinate space.
     */
    private void computeBigChunk(BigChunk chunk) {
        double chunkSizeGrid = getBigChunkSizeGrid();

        // A. Collect all segments from nearby cells that could influence this chunk
        List<Segment3D> segments = new ArrayList<>();
        List<Integer> levels = new ArrayList<>();
        List<Integer> startConns = new ArrayList<>();
        List<Integer> endConns = new ArrayList<>();
        List<Integer> endFlowLevels = new ArrayList<>();

        collectSegmentsForBigChunk(chunk, chunkSizeGrid, segments, levels, startConns, endConns, endFlowLevels);

        // B. Sort by level descending (highest level segments processed first)
        Integer[] sortedIndices = new Integer[segments.size()];
        for (int i = 0; i < sortedIndices.length; i++) sortedIndices[i] = i;
        java.util.Arrays.sort(sortedIndices, (a, b) -> Integer.compare(levels.get(b), levels.get(a)));

        // C. Process each segment in sorted order
        for (int idx : sortedIndices) {
            sampleSegmentAlongSpline(segments.get(idx), levels.get(idx),
                startConns.get(idx), endConns.get(idx), endFlowLevels.get(idx),
                chunk, chunkSizeGrid);
        }
    }

    /**
     * Collect all segments from nearby cells that could influence the bigchunk.
     * Segments are filtered to only include those within maxDistGrid of the chunk boundary.
     * All coordinates are in grid coordinate space.
     * @param chunk The BigChunk to collect segments for (chunk.gridOriginX/Y are in grid coordinates)
     * @param chunkSizeGrid Size of chunk in grid coordinates
     * @param outSegments Output list for segments
     * @param outLevels Output list for segment levels (parallel to outSegments)
     * @param outStartConns Output list for start endpoint connection counts
     * @param outEndConns Output list for end endpoint connection counts
     * @param outEndFlowLevel Output list: minimum level of a lower-level segment connected at end point, or -1 if none
     */
    private void collectSegmentsForBigChunk(BigChunk chunk, double chunkSizeGrid,
                                           List<Segment3D> outSegments, List<Integer> outLevels,
                                           List<Integer> outStartConns, List<Integer> outEndConns,
                                           List<Integer> outEndFlowLevel) {
        // Determine the range of cells that could contain relevant segments
        // Grid coordinates are already normalized (sampler / gridsize), so cell boundaries are at integer values
        double minGridX = chunk.gridOriginX - maxDistGrid;
        double maxGridX = chunk.gridOriginX + chunkSizeGrid + maxDistGrid;
        double minGridY = chunk.gridOriginY - maxDistGrid;
        double maxGridY = chunk.gridOriginY + chunkSizeGrid + maxDistGrid;

        // Convert to cell coordinates (cells are at integer boundaries in grid space)
        int minCellX = (int) Math.floor(minGridX);
        int maxCellX = (int) Math.floor(maxGridX);
        int minCellY = (int) Math.floor(minGridY);
        int maxCellY = (int) Math.floor(maxGridY);

        // B. For each cell, get or compute SegmentList
        for (int cellY = minCellY; cellY <= maxCellY; cellY++) {
            for (int cellX = minCellX; cellX <= maxCellX; cellX++) {
                // Cache key is in sampler coordinates for compatibility with existing cache
                double cellSamplerX = cellX * gridsize;
                double cellSamplerY = cellY * gridsize;

                SegmentList segmentList = segmentListCache.get(cellSamplerX, cellSamplerY);

                // If not in cache, compute it
                if (segmentList == null) {
                    segmentList = computeAllSegmentsForCell(salt, cellX, cellY);
                    segmentListCache.put(cellSamplerX, cellSamplerY, segmentList);
                }

                // Convert SegmentList to Segment3D and filter by distance
                for (SegmentIdx segIdx : segmentList.getSegments()) {
                    // Check if at least one endpoint is within maxDistGrid of chunk boundary
                    // Segments are in grid coordinates
                    Point3D srt = segIdx.getSrt(segmentList);
                    Point3D end = segIdx.getEnd(segmentList);

                    boolean srtNear = isPointNearChunk(srt, chunk, chunkSizeGrid);
                    boolean endNear = isPointNearChunk(end, chunk, chunkSizeGrid);

                    // If neither endpoint is near, check midpoint for segments crossing at corners
                    if (!srtNear && !endNear) {
                        Point3D mid = new Point3D(
                            (srt.x + end.x) * 0.5,
                            (srt.y + end.y) * 0.5,
                            (srt.z + end.z) * 0.5);
                        if (isPointNearChunk(mid, chunk, chunkSizeGrid)) {
                            srtNear = true; // treat as near so we enter the block below
                        }
                    }

                    if (srtNear || endNear) {
                        // Convert to Segment3D and add to list
                        Segment3D seg3d = segIdx.resolve(segmentList);
                        outSegments.add(seg3d);
                        outLevels.add(segIdx.level);
                        // Extract connection counts from endpoint NetworkPoints
                        outStartConns.add(segmentList.getPoint(segIdx.srtIdx).connections);
                        outEndConns.add(segmentList.getPoint(segIdx.endIdx).connections);

                        // Check if end point connects to a lower-level (wider) segment
                        int endFlowLevel = -1;
                        for (SegmentIdx otherSeg : segmentList.getSegments()) {
                            if (otherSeg == segIdx) continue;
                            if (otherSeg.level < segIdx.level
                                    && (otherSeg.srtIdx == segIdx.endIdx || otherSeg.endIdx == segIdx.endIdx)) {
                                if (endFlowLevel < 0 || otherSeg.level < endFlowLevel) {
                                    endFlowLevel = otherSeg.level;
                                }
                            }
                        }
                        outEndFlowLevel.add(endFlowLevel);
                    }
                }
            }
        }
    }

    /**
     * Check if a point is within maxDistGrid of a chunk boundary.
     * Uses the minimum of the per-axis distances to the chunk border,
     * so a point only needs to be close on ONE axis to be retained.
     * All coordinates are in grid coordinate space.
     * @param point Point in grid coordinates
     * @param chunk BigChunk with gridOriginX/Y in grid coordinates
     * @param chunkSizeGrid Size of chunk in grid coordinates
     */
    private boolean isPointNearChunk(Point3D point, BigChunk chunk, double chunkSizeGrid) {
        double chunkMaxX = chunk.gridOriginX + chunkSizeGrid;
        double chunkMaxY = chunk.gridOriginY + chunkSizeGrid;

        // Per-axis distance to chunk border (0 if inside chunk on that axis)
        double distX = (point.x < chunk.gridOriginX) ? chunk.gridOriginX - point.x
                     : (point.x > chunkMaxX) ? point.x - chunkMaxX
                     : 0;
        double distY = (point.y < chunk.gridOriginY) ? chunk.gridOriginY - point.y
                     : (point.y > chunkMaxY) ? point.y - chunkMaxY
                     : 0;

        // Keep the point if the minimum axis distance is within maxDistGrid
        return Math.min(distX, distY) <= maxDistGrid;
    }

    /**
     * Sample a segment along its hermite spline and project influence onto bigchunk blocks.
     * Uses 3-layer elevation tracking and adaptive sample evaluation criteria.
     * All coordinates are in grid coordinate space.
     * @param seg Segment in grid coordinates
     * @param level River level
     * @param startConnections Number of connections at start endpoint
     * @param endConnections Number of connections at end endpoint
     * @param endFlowLevel Level of lower-level segment connected at end point, or -1 if none
     * @param chunk BigChunk to project onto
     * @param chunkSizeGrid Size of chunk in grid coordinates
     */
    private void sampleSegmentAlongSpline(Segment3D seg, int level,
                                          int startConnections, int endConnections,
                                          int endFlowLevel,
                                          BigChunk chunk, double chunkSizeGrid) {
        // Pre-compute constants
        double segmentLength = seg.length();
        double cachepixelsGrid = cachepixels / gridsize;
        double elevQuantizeRes = 255.0 / max;
        int numSamples = (int) Math.ceil((segmentLength / cachepixelsGrid) * 2.11);
        if (numSamples < 2) numSamples = 2;

        // Segment slope: abs(height change / euclidean distance)
        double heightChange = Math.abs(seg.end.z - seg.srt.z);
        double segmentSlope = (segmentLength > MathUtils.EPSILON)
            ? heightChange / segmentLength : 0.0;

        // Evaluation distance threshold: 70% of cachepixels in grid coordinates
        double evalDistThreshold = 0.7 * cachepixelsGrid;

        // === Elevation tracking state (3 layers) ===
        int outerElev = 0, innerElev = 0, centralElev = 0;
        double outerRadius = 0, innerRadius = 0, centralRadius = 0;

        // === Stream tracking state ===
        boolean isNewStream = true;
        boolean wasOutOfBounds = true;
        dendryterra.math.Vec2D prevEvalTangent = null;
        Point3D prevEvalPos = null;
        dendryterra.math.Vec2D prevLoopTangent = null;

        // Sample along the segment
        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / (numSamples - 1);

            // Interpolate position and tangent
            Point3D samplePoint = evaluateHermiteSpline(seg, t);
            dendryterra.math.Vec2D currentTangent = interpolateTangent(seg, t);

            // === Step A: New stream initialization ===
            if (i == 0 || wasOutOfBounds) {
                int quantizedElev = (int) Math.min(255, Math.max(0, samplePoint.z * elevQuantizeRes));
                outerElev = innerElev = centralElev = quantizedElev;
                outerRadius = innerRadius = centralRadius = 0;
                prevLoopTangent = currentTangent;
                prevEvalPos = samplePoint;
                prevEvalTangent = currentTangent;
                isNewStream = true;
            }
            // === Step B: Continued stream ===
            // prevLoopTangent already holds previous loop iteration's currentTangent

            // === Step C: Boundary check ===
            if (!isPointNearChunk(samplePoint, chunk, chunkSizeGrid)) {
                wasOutOfBounds = true;
                prevLoopTangent = currentTangent;
                continue;
            }
            wasOutOfBounds = false;

            // Compute potential quantized elevation
            int potentialElev = (int) Math.min(255, Math.max(0, samplePoint.z * elevQuantizeRes));
            boolean elevationChanged = false;

            // Check if quantized elevation changed
            if (potentialElev != centralElev) {
                outerElev = innerElev;
                innerElev = centralElev;
                centralElev = potentialElev;
                outerRadius = innerRadius;
                innerRadius = centralRadius;
                centralRadius = 1.0;  // 1.0 in normalized (river-width) space
                elevationChanged = true;
            }

            // === Step D: Should this sample be evaluated/projected? ===
            boolean shouldEvaluate = false;

            if (isNewStream) {
                shouldEvaluate = true;
            } else {
                // Check tangent angle difference > 90 degrees
                if (prevEvalTangent != null) {
                    double dotProduct = prevEvalTangent.x * currentTangent.x
                                      + prevEvalTangent.y * currentTangent.y;
                    dotProduct = Math.max(-1.0, Math.min(1.0, dotProduct));
                    double angleDeg = Math.toDegrees(Math.acos(dotProduct));
                    if (angleDeg > 90.0) {
                        shouldEvaluate = true;
                    }
                }

                // Check distance from previous evaluated position
                if (prevEvalPos != null) {
                    double dx = samplePoint.x - prevEvalPos.x;
                    double dy = samplePoint.y - prevEvalPos.y;
                    double dist2D = Math.sqrt(dx * dx + dy * dy);
                    if (dist2D > evalDistThreshold) {
                        shouldEvaluate = true;
                    }
                }

                // Last sample in segment
                if (i == numSamples - 1) {
                    shouldEvaluate = true;
                }

                // Elevation changed
                if (elevationChanged) {
                    shouldEvaluate = true;
                }
            }

            if (!shouldEvaluate) {
                prevLoopTangent = currentTangent;
                continue;
            }

            // === Step E: Update elevation radii ===
            double riverWidthGrid = calculateRiverWidth(level, samplePoint.x, samplePoint.y);

            // Width transition: linearly widen to match lower-level river at endpoint
            if (endFlowLevel >= 0 && endFlowLevel < level && endConnections == 1) {
                double targetWidthGrid = calculateRiverWidth(endFlowLevel, samplePoint.x, samplePoint.y);
                riverWidthGrid = riverWidthGrid + (targetWidthGrid - riverWidthGrid) * t;
            }

            if (!elevationChanged && prevEvalPos != null && riverWidthGrid > 0) {
                double dx = samplePoint.x - prevEvalPos.x;
                double dy = samplePoint.y - prevEvalPos.y;
                double distSinceLastEval = Math.sqrt(dx * dx + dy * dy) / riverWidthGrid;
                outerRadius = Math.max(0, outerRadius - distSinceLastEval);
                innerRadius = Math.max(0, innerRadius - distSinceLastEval);
                centralRadius = Math.max(0, centralRadius - distSinceLastEval);
            }

            // Saturate radii to distance from segment end (normalized by river width)
            if (riverWidthGrid > 0) {
                double dxEnd = samplePoint.x - seg.end.x;
                double dyEnd = samplePoint.y - seg.end.y;
                double distToEndNorm = Math.sqrt(dxEnd * dxEnd + dyEnd * dyEnd) / riverWidthGrid;
                double maxRadius = Math.max(0, distToEndNorm - 1);
                outerRadius = Math.min(outerRadius, maxRadius);
                innerRadius = Math.min(innerRadius, maxRadius);
                centralRadius = Math.min(centralRadius, maxRadius);
            }

            // === Step F: Segment fill flag ===
            boolean segmentFill = false;
            boolean isStartPoint = (i == 0);
            boolean isEndPoint = (i == numSamples - 1);
            if (ENABLE_SEGMENT_FILL_ALL) {
                if (isStartPoint || isEndPoint) segmentFill = true;
            } else {
                if (isStartPoint && startConnections == 1) segmentFill = true;
                if (isEndPoint && endConnections == 1) segmentFill = true;
            }

            // === Project to boxes ===
            projectConeToBoxes(samplePoint, currentTangent, prevEvalTangent,
                centralElev, innerElev, outerElev,
                centralRadius, innerRadius, outerRadius,
                riverWidthGrid, segmentFill, isStartPoint,
                segmentSlope, chunk, chunkSizeGrid);

            // Update state for next iteration
            prevEvalPos = samplePoint;
            prevEvalTangent = currentTangent;
            isNewStream = false;
            prevLoopTangent = currentTangent;
        }
    }

    /**
     * Evaluate point on Hermite spline at parameter t.
     * Uses cubic Hermite interpolation: H(t) = (2t³-3t²+1)P0 + (t³-2t²+t)T0 + (-2t³+3t²)P1 + (t³-t²)T1
     */
    private Point3D evaluateHermiteSpline(Segment3D seg, double t) {
        if (!useSplines || seg.tangentSrt == null || seg.tangentEnd == null) {
            // Fall back to linear interpolation
            return seg.lerp(t);
        }

        double t2 = t * t;
        double t3 = t2 * t;

        // Hermite basis functions
        double h00 = 2*t3 - 3*t2 + 1;   // (2t³ - 3t² + 1)
        double h10 = t3 - 2*t2 + t;     // (t³ - 2t² + t)
        double h01 = -2*t3 + 3*t2;      // (-2t³ + 3t²)
        double h11 = t3 - t2;            // (t³ - t²)

        // Tangent vectors scaled by segment length and tangent strength
        double segLen = seg.length();
        double tangentScale = segLen * tangentStrength;

        // Apply curvature scaling
        double effectiveCurvature = curvature;

        // Tangent vectors in 3D (tangents are 2D, z component from elevation difference)
        double tx0 = seg.tangentSrt.x * tangentScale * effectiveCurvature;
        double ty0 = seg.tangentSrt.y * tangentScale * effectiveCurvature;
        double tz0 = (seg.end.z - seg.srt.z) * effectiveCurvature;

        double tx1 = seg.tangentEnd.x * tangentScale * effectiveCurvature;
        double ty1 = seg.tangentEnd.y * tangentScale * effectiveCurvature;
        double tz1 = (seg.end.z - seg.srt.z) * effectiveCurvature;

        // Hermite interpolation
        double x = h00 * seg.srt.x + h10 * tx0 + h01 * seg.end.x + h11 * tx1;
        double y = h00 * seg.srt.y + h10 * ty0 + h01 * seg.end.y + h11 * ty1;
        double z = h00 * seg.srt.z + h10 * tz0 + h01 * seg.end.z + h11 * tz1;

        return new Point3D(x, y, z);
    }

    /**
     * Interpolate tangent at parameter t along the segment.
     */
    private dendryterra.math.Vec2D interpolateTangent(Segment3D seg, double t) {
        if (seg.tangentSrt != null && seg.tangentEnd != null && useSplines) {
            // True Hermite spline derivative: H'(t) = dh00*P0 + dh10*T0*s + dh01*P1 + dh11*T1*s
            double segLen = seg.srt.projectZ().distanceTo(seg.end.projectZ());
            double s = segLen * tangentStrength * curvature;
            double t2 = t * t;

            double dh00 = 6 * t2 - 6 * t;
            double dh10 = 3 * t2 - 4 * t + 1;
            double dh01 = -6 * t2 + 6 * t;
            double dh11 = 3 * t2 - 2 * t;

            double dx = dh00 * seg.srt.x + dh10 * seg.tangentSrt.x * s
                       + dh01 * seg.end.x + dh11 * seg.tangentEnd.x * s;
            double dy = dh00 * seg.srt.y + dh10 * seg.tangentSrt.y * s
                       + dh01 * seg.end.y + dh11 * seg.tangentEnd.y * s;

            double len = Math.sqrt(dx * dx + dy * dy);
            if (len > MathUtils.EPSILON) {
                return new dendryterra.math.Vec2D(dx / len, dy / len);
            }
        }
        // Fallback: use single tangent or segment direction
        if (seg.tangentSrt != null && seg.tangentEnd == null) {
            return seg.tangentSrt;
        } else if (seg.tangentEnd != null && seg.tangentSrt == null) {
            return seg.tangentEnd;
        }
        // No tangents or Hermite derivative was zero - use segment direction
        double dx = seg.end.x - seg.srt.x;
        double dy = seg.end.y - seg.srt.y;
        double len = Math.sqrt(dx * dx + dy * dy);
        if (len > 0) {
            return new dendryterra.math.Vec2D(dx / len, dy / len);
        } else {
            return new dendryterra.math.Vec2D(1, 0);  // Default
        }
    }

    /**
     * Project a cone of influence from a sample point onto bigchunk boxes.
     * All coordinates and distances are in grid coordinate space.
     * @param samplePoint Sample position in grid coordinates
     * @param elevation Elevation value
     * @param riverWidthGrid River width in grid coordinates
     * @param chunkSizeGrid Chunk size in grid coordinates
     */
    /**
     * Project a cone of influence from a sample point onto bigchunk boxes.
     * All coordinates and distances are in grid coordinate space.
     * @param samplePoint Sample position in grid coordinates
     * @param currentTangent Current tangent direction at sample point
     * @param prevTangent Previous evaluated tangent direction
     * @param centralElev Pre-quantized central elevation (UInt8)
     * @param innerElev Pre-quantized inner elevation (UInt8)
     * @param outerElev Pre-quantized outer elevation (UInt8)
     * @param centralRadius Central elevation radius (normalized by river width, 0-1)
     * @param innerRadius Inner elevation radius (normalized by river width, 0-1)
     * @param outerRadius Outer elevation radius (normalized by river width, 0-1)
     * @param riverWidthGrid River width in grid coordinates
     * @param segmentFill If true, fill a 180° semicircle at segment endpoint
     * @param isStartPoint True if this is the start of the segment (affects semicircle direction)
     * @param segmentSlope Segment slope (abs(height change / euclidean distance))
     * @param chunk BigChunk to project onto
     * @param chunkSizeGrid Chunk size in grid coordinates
     */
    private void projectConeToBoxes(Point3D samplePoint, dendryterra.math.Vec2D currentTangent,
                                   dendryterra.math.Vec2D prevTangent,
                                   int centralElev, int innerElev, int outerElev,
                                   double centralRadius, double innerRadius, double outerRadius,
                                   double riverWidthGrid, boolean segmentFill, boolean isStartPoint,
                                   double segmentSlope, BigChunk chunk, double chunkSizeGrid) {
        double cachepixelsGrid = cachepixels / gridsize;

        // Determine cone angle and bow direction
        double coneAngle;
        double bowDirectionRad;  // Center angle of the cone sweep (radians)

        if (segmentFill) {
            // Semicircle fill for endpoints with 1 connection
            coneAngle = Math.PI;  // 180 degrees
            if (isStartPoint) {
                // Perpendicular 90° clockwise from currentTangent (faces backward from segment)
                bowDirectionRad = Math.atan2(currentTangent.y, currentTangent.x) - Math.PI / 2;
            } else {
                // Perpendicular 90° counterclockwise from currentTangent (faces forward from segment)
                bowDirectionRad = Math.atan2(currentTangent.y, currentTangent.x) + Math.PI / 2;
            }
        } else {
            // Normal cone: sweep from prevTangent to currentTangent
            coneAngle = calculateConeAngle(prevTangent, currentTangent);
            bowDirectionRad = calculateBowDirectionRad(prevTangent, currentTangent);
        }

        // Slope factor for elevation centroid distance
        double slopeFactor = Math.max(0, 1.0 - segmentSlope / slopeWhenStraight);

        // Project outward from sample point
        int maxSteps = (int) Math.ceil(maxDistGrid / cachepixelsGrid);
        for (int step = 0; step <= maxSteps; step++) {
            double distanceGrid = step * cachepixelsGrid;
            if (step > 0 && distanceGrid > maxDistGrid) break;

            // Normalized distance from river center (in river-width units)
            double normDistFromCenter = (riverWidthGrid > 0) ? distanceGrid / riverWidthGrid : 1.0;

            // Determine which elevation to use based on centroid distances
            int selectedElev = centralElev;  // Default
            if (normDistFromCenter < 1.0) {
                // Check elevation layers from biggest radius to smallest
                if (outerRadius > 0) {
                    double centroidDist = Math.sqrt(
                        normDistFromCenter * normDistFromCenter +
                        Math.pow(outerRadius * slopeFactor, 2)
                    );
                    if (centroidDist < 1.0) {
                        selectedElev = outerElev;
                    }
                }
                if (selectedElev == centralElev && innerRadius > 0) {
                    double centroidDist = Math.sqrt(
                        normDistFromCenter * normDistFromCenter +
                        Math.pow(innerRadius * slopeFactor, 2)
                    );
                    if (centroidDist < 1.0) {
                        selectedElev = innerElev;
                    }
                }
                // Otherwise centralElev (already set)
            }

            boolean blotAdjacentBoxes = ENABLE_BLOT_FILLING && (step > 0);

            if (step == 0) {
                // Center point - set the box at samplePoint
                int bx = gridToBlockIndex(samplePoint.x, chunk.gridOriginX, cachepixelsGrid);
                int by = gridToBlockIndex(samplePoint.y, chunk.gridOriginY, cachepixelsGrid);
                if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
                    updateBox(chunk.getBlock(bx, by), 0.0, selectedElev, riverWidthGrid,
                             false, bx, by, chunk, step);
                }
            } else {
                // Arc samples at this radius - both sides (positive and opposite)
                double arcLength = coneAngle * distanceGrid;
                int numArcSamples = Math.max(2, (int) Math.ceil(arcLength / (cachepixelsGrid * 0.5)));

                for (int side = 0; side < 2; side++) {
                    double sideOffset = side * Math.PI;
                    for (int a = 0; a < numArcSamples; a++) {
                        double angleOffset = coneAngle * ((double) a / (numArcSamples - 1) - 0.5);
                        double angle = bowDirectionRad + sideOffset + angleOffset;

                        double px = samplePoint.x + Math.cos(angle) * distanceGrid;
                        double py = samplePoint.y + Math.sin(angle) * distanceGrid;

                        int bx = gridToBlockIndex(px, chunk.gridOriginX, cachepixelsGrid);
                        int by = gridToBlockIndex(py, chunk.gridOriginY, cachepixelsGrid);

                        if (bx >= 0 && bx < 256 && by >= 0 && by < 256) {
                            updateBox(chunk.getBlock(bx, by), distanceGrid, selectedElev,
                                     riverWidthGrid, blotAdjacentBoxes, bx, by, chunk, step);
                        }
                    }
                }
            }
        }
    }

    /**
     * Convert grid coordinate to block index within a chunk.
     * @param gridCoord Grid coordinate (in normalized space)
     * @param chunkGridOrigin Grid origin of the chunk
     * @param blockSizeGrid Size of one block in grid coordinates (cachepixels / gridsize)
     * @return Block index (0-255)
     */
    private int gridToBlockIndex(double gridCoord, double chunkGridOrigin, double blockSizeGrid) {
        return (int) Math.floor((gridCoord - chunkGridOrigin) / blockSizeGrid);
    }

    /**
     * Calculate the cone angle between previous and current tangent vectors.
     * Returns the absolute angle change in radians.
     */
    private double calculateConeAngle(dendryterra.math.Vec2D prev, dendryterra.math.Vec2D current) {
        double angle = Math.atan2(current.y, current.x) - Math.atan2(prev.y, prev.x);
        angle = normalizeAngle(angle);
        return Math.abs(angle);
    }

    /**
     * Calculate the bow direction (center angle of cone sweep) between previous and current tangent.
     * Returns the perpendicular direction in radians, on the side the curve bows toward.
     */
    private double calculateBowDirectionRad(dendryterra.math.Vec2D prev, dendryterra.math.Vec2D current) {
        // Cross product to determine which side the curve bows toward
        double cross = prev.x * current.y - prev.y * current.x;

        // Perpendicular to current tangent, on the bowing side
        if (cross > 0) {
            // Bows right: perpendicular is (-y, x)
            return Math.atan2(current.x, -current.y);
        } else {
            // Bows left: perpendicular is (y, -x)
            return Math.atan2(-current.x, current.y);
        }
    }

    /**
     * Normalize angle to range [-PI, PI].
     */
    private double normalizeAngle(double angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /** Compile-time toggle for adjacent box blot filling */
    private static final boolean ENABLE_BLOT_FILLING = true;

    /**
     * Update a bigchunk block with new distance and elevation values.
     * All distances are in grid coordinate space.
     * @param box Block to update
     * @param distanceGrid Distance in grid coordinates
     * @param elevationU8 Pre-quantized elevation value (0-255)
     * @param riverWidthGrid River width in grid coordinates
     * @param blotAdjacentBoxes If true, also fill 4 adjacent boxes with same values
     * @param blockX Block X index within chunk (for adjacent access)
     * @param blockY Block Y index within chunk (for adjacent access)
     * @param chunk BigChunk for adjacent box access
     * @param outwardStep The outward step index (0 = center, 1 = first ring, etc.)
     */
    private void updateBox(BigChunk.BigChunkBlock box, double distanceGrid,
                          int elevationU8, double riverWidthGrid,
                          boolean blotAdjacentBoxes, int blockX, int blockY,
                          BigChunk chunk, int outwardStep) {
        // Compute normalized distance
        double normalizedDist;
        if (distanceGrid < riverWidthGrid) {
            // River is ratiometric, inside river is less than 1.
            // Divided by gridsize since this is the ratio in grid units.
            normalizedDist = (distanceGrid / riverWidthGrid) / gridsize;
        } else {
            // Outside river is absolute distance to river's edge.
            normalizedDist = distanceGrid - riverWidthGrid;
        }

        // Quantize distance to uint8
        double distQuantizeRes = 255.0 / maxDistGrid;
        int distU8 = (int) Math.min(255, Math.max(0, normalizedDist * distQuantizeRes));

        // Apply elevation smoothing noise at river edge transitions
        int finalElevU8 = elevationU8;
        int currentElev = box.getElevationUnsigned();
        int currentDist = box.getDistanceUnsigned();

        // If box was previously set (distance < 255), new elevation is lower,
        // and we're at the first outward step from river edge, add random noise
        if (currentElev > elevationU8 && currentDist < 255 && outwardStep == 1) {
            int range = currentElev - elevationU8;
            int noise = (int)(Math.random() * range);
            finalElevU8 = elevationU8 + noise;
        }

        // Update this box
        applyBoxUpdate(box, distU8, finalElevU8);

        // Blot: fill 4 adjacent boxes with same values (pin-hole filling)
        if (ENABLE_BLOT_FILLING && blotAdjacentBoxes) {
            int[][] neighbors = {{-1,0},{1,0},{0,-1},{0,1}};
            for (int[] d : neighbors) {
                int nx = blockX + d[0], ny = blockY + d[1];
                if (nx >= 0 && nx < 256 && ny >= 0 && ny < 256) {
                    applyBoxUpdate(chunk.getBlock(nx, ny), distU8, finalElevU8);
                }
            }
        }
    }

    /**
     * Apply distance and elevation updates to a single box.
     * Distance: lower wins (closer to river).
     * Elevation: lower wins (river valley).
     */
    private void applyBoxUpdate(BigChunk.BigChunkBlock box, int distU8, int elevU8) {
        if (distU8 < box.getDistanceUnsigned()) {
            box.setDistanceUnsigned(distU8);
        }
        if (elevU8 < box.getElevationUnsigned()) {
            box.setElevationUnsigned(elevU8);
        }
    }
}

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

    // Cache configuration
    private static final int MAX_CACHE_SIZE = 16384;

    // Lazy LRU cache (optional based on useCache flag)
    private final LoadingCache<Long, CellData> cellCache;

    // Timing statistics (only used when debugTiming is true)
    private final AtomicLong sampleCount = new AtomicLong(0);
    private final AtomicLong totalTimeNs = new AtomicLong(0);
    private volatile long lastLogTime = 0;
    private static final long LOG_INTERVAL_MS = 5000; // Log every 5 seconds

    private static class CellData {
        final Point2D point;
        final int branchCount;

        CellData(Point2D point, int branchCount) {
            this.point = point;
            this.branchCount = branchCount;
        }
    }

    public DendrySampler(int resolution, double epsilon, double delta,
                         double slope, double gridsize,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt,
                         Sampler branchesSampler, int defaultBranches,
                         double curvature, double curvatureFalloff,
                         double connectDistance, double connectDistanceFactor,
                         boolean useCache, boolean useParallel, boolean useSplines,
                         boolean debugTiming, int parallelThreshold) {
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

        // Initialize cache only if enabled
        if (useCache) {
            this.cellCache = Caffeine.newBuilder()
                .maximumSize(MAX_CACHE_SIZE)
                .build(this::generateCellData);
        } else {
            this.cellCache = null;
        }

        if (debugTiming) {
            LOGGER.info("DendrySampler initialized with: resolution={}, gridsize={}, useCache={}, useParallel={}, useSplines={}, parallelThreshold={}",
                resolution, gridsize, useCache, useParallel, useSplines, parallelThreshold);
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

        double result = evaluate(seed, x / gridsize, z / gridsize);

        if (debugTiming) {
            long elapsed = System.nanoTime() - startTime;
            totalTimeNs.addAndGet(elapsed);
            long count = sampleCount.incrementAndGet();

            long now = System.currentTimeMillis();
            if (now - lastLogTime > LOG_INTERVAL_MS) {
                lastLogTime = now;
                double avgNs = (double) totalTimeNs.get() / count;
                double avgMs = avgNs / 1_000_000.0;
                LOGGER.info("DendrySampler stats: {} samples, avg {:.4f} ms/sample ({:.0f} ns)",
                    count, avgMs, avgNs);
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
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slope constraints for sub-branch connections
        double minSlopeLevel1 = 0.5;
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 0: Root network spanning 5x5 level 1 cells
        Cell cell1 = getCell(x, y, 1);
        Point3D[][] level0Points = generateLevel0Points(cell1);
        List<Segment3D> segments0 = generateLevel0Segments(level0Points);

        // Prune level 0 segments not connected to inner 3x3
        double innerMinX = cell1.x - 1;
        double innerMaxX = cell1.x + 2;
        double innerMinY = cell1.y - 1;
        double innerMaxY = cell1.y + 2;
        segments0 = pruneDisconnectedSegments(segments0, innerMinX, innerMinY, innerMaxX, innerMaxY);

        if (resolution == 0) {
            return computeResult(x, y, segments0);
        }

        // Level 1: Process 3x3 cells around query cell
        List<Segment3D> segments1 = generateLevel1Segments(cell1, segments0, minSlopeLevel1);
        segments1 = pruneAwaySegments(segments1, cell1);

        int branchCount = getBranchCountForCell(cell1);
        segments1 = subdivideSegments(segments1, branchCount, 1);
        displaceSegments(segments1, displacementLevel1, cell1);

        List<Segment3D> allSegments = new ArrayList<>(segments0);
        allSegments.addAll(segments1);

        if (resolution == 1) {
            return computeResult(x, y, allSegments);
        }

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

    /**
     * Generate level 0 points: one per level 1 cell in a 5x5 grid.
     * All points have elevation 0 (flat network layer).
     */
    private Point3D[][] generateLevel0Points(Cell cell1) {
        Point3D[][] points = new Point3D[5][5];

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                int cellX = cell1.x + j - 2;  // -2 to +2 offset
                int cellY = cell1.y + i - 2;

                // Get the jittered point for this cell (reuse existing logic)
                CellData data = getCellData(cellX, cellY);
                Point2D p2d = data.point;

                // Level 0 points are at elevation 0
                points[i][j] = new Point3D(p2d.x, p2d.y, 0.0);
            }
        }
        return points;
    }

    /**
     * Generate level 0 segments by connecting each point to nearest neighbor.
     * Uses 2D distance, not elevation-based flow.
     */
    private List<Segment3D> generateLevel0Segments(Point3D[][] points) {
        List<Segment3D> segments = new ArrayList<>();
        int size = points.length;  // 5

        // Process inner 3x3 (indices 1-3)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Point3D current = points[i][j];

                // Find nearest neighbor by 2D distance
                double minDist = Double.MAX_VALUE;
                Point3D nearest = null;

                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;

                        Point3D neighbor = points[i + di][j + dj];
                        double dist = current.projectZ().distanceTo(neighbor.projectZ());

                        if (dist < minDist) {
                            minDist = dist;
                            nearest = neighbor;
                        }
                    }
                }

                if (nearest != null && minDist > MathUtils.EPSILON) {
                    segments.add(new Segment3D(current, nearest, 0));  // level = 0
                }
            }
        }
        return segments;
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
     * Generate level 1 segments for 3x3 cell region around query cell.
     */
    private List<Segment3D> generateLevel1Segments(Cell queryCell, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> allSegments = new ArrayList<>();

        // Process 3x3 grid of level 1 cells
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
     */
    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions, int level) {
        if (segments.isEmpty() || numDivisions <= 1) {
            return segments;
        }

        if (useSplines && curvature > 0) {
            return subdivideWithSplines(segments, numDivisions, level);
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
     * Catmull-Rom spline subdivision (smoother but slower).
     */
    private List<Segment3D> subdivideWithSplines(List<Segment3D> segments, int numDivisions, int level) {
        double levelCurvature = curvature * Math.pow(curvatureFalloff, level - 1);
        List<Segment3D> subdivided = new ArrayList<>();

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            Point3D p0 = (i > 0) ? segments.get(i - 1).a : seg.a;
            Point3D p1 = seg.a;
            Point3D p2 = seg.b;
            Point3D p3 = (i < segments.size() - 1) ? segments.get(i + 1).b : seg.b;

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

    private void displaceSegments(List<Segment3D> segments, double displacementFactor, Cell cell) {
        if (displacementFactor < MathUtils.EPSILON) return;

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            Vec2D perp = dir.rotateCCW90().normalize();

            Random rng = initRandomGenerator((int)(seg.a.x * 100), (int)(seg.a.y * 100), cell.resolution);
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor;

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
}

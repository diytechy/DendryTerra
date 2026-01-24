package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import dendryterra.math.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Optimized with lazy LRU caching, parallel processing, and configurable parameters.
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

    // New configuration parameters
    private final Sampler branchesSampler;
    private final int defaultBranches;
    private final double curvature;
    private final double curvatureFalloff;
    private final double connectDistance;
    private final double connectDistanceFactor;

    // Cache configuration
    private static final int MAX_CACHE_SIZE = 16384;
    private static final int PARALLEL_THRESHOLD = 100;

    /**
     * Lazy LRU cache for cell data (points and branch counts).
     * Replaces the eager 128x128 fixed array with on-demand generation and eviction.
     */
    private final LoadingCache<Long, CellData> cellCache;

    /**
     * Cached cell data including point position and branch count.
     */
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
                         double connectDistance, double connectDistanceFactor) {
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

        // Initialize lazy LRU cache
        this.cellCache = Caffeine.newBuilder()
            .maximumSize(MAX_CACHE_SIZE)
            .build(this::generateCellData);
    }

    /**
     * Generate cell data on demand (called by cache on miss).
     */
    private CellData generateCellData(Long key) {
        int cellX = unpackX(key);
        int cellY = unpackY(key);
        Point2D point = generatePoint(cellX, cellY);
        int branches = getBranchCount(cellX, cellY);
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

    /**
     * Get branch count for a cell, using branches sampler if available.
     */
    private int getBranchCount(int cellX, int cellY) {
        if (branchesSampler == null) {
            return defaultBranches;
        }
        // Query at cell center in world coordinates
        double centerX = (cellX + 0.5) * gridsize;
        double centerY = (cellY + 0.5) * gridsize;
        int branches = (int) Math.round(branchesSampler.getSample(salt, centerX, centerY));
        return Math.max(1, Math.min(8, branches));
    }

    @Override
    public double getSample(long seed, double x, double z) {
        // Convert world coordinates to noise space by dividing by gridsize
        return evaluate(seed, x / gridsize, z / gridsize);
    }

    @Override
    public double getSample(long seed, double x, double y, double z) {
        return evaluate(seed, x / gridsize, z / gridsize);
    }

    private double evaluate(long seed, double x, double y) {
        // Displacement factors decrease exponentially for each level
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slope constraints for sub-branch connections
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 1: Base Resolution
        Cell cell1 = getCell(x, y, 1);
        Point3D[][] points1 = generateNeighboringPoints3D(cell1, 9);
        List<Segment3D> segments1 = generateSegments(points1, 1);
        segments1 = subdivideSegmentsWithSpline(segments1, getBranchCountForCell(cell1), 1);
        displaceSegments(segments1, displacementLevel1, cell1);

        if (resolution == 1) {
            return computeResult(x, y, segments1);
        }

        // Level 2: 2x Resolution
        Cell cell2 = getCell(x, y, 2);
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);
        List<Segment3D> segments2 = generateSubSegments(points2, segments1, minSlopeLevel2, 2);
        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            List<Segment3D> allSegments = new ArrayList<>(segments1);
            allSegments.addAll(segments2);
            return computeResult(x, y, allSegments);
        }

        // Level 3: 4x Resolution
        Cell cell3 = getCell(x, y, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);
        List<Segment3D> allPreviousSegments = new ArrayList<>(segments1);
        allPreviousSegments.addAll(segments2);
        List<Segment3D> segments3 = generateSubSegments(points3, allPreviousSegments, minSlopeLevel3, 3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allPreviousSegments.addAll(segments3);
            return computeResult(x, y, allPreviousSegments);
        }

        // Level 4: 8x Resolution
        Cell cell4 = getCell(x, y, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allPreviousSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allPreviousSegments, minSlopeLevel4, 4);

        if (resolution == 4) {
            allPreviousSegments.addAll(segments4);
            return computeResult(x, y, allPreviousSegments);
        }

        // Level 5: 16x Resolution
        Cell cell5 = getCell(x, y, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allPreviousSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allPreviousSegments, minSlopeLevel5, 5);
        allPreviousSegments.addAll(segments5);

        return computeResult(x, y, allPreviousSegments);
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
     * Get cached cell data (lazy LRU).
     */
    private CellData getCellData(int cellX, int cellY) {
        return cellCache.get(packKey(cellX, cellY));
    }

    /**
     * Get branch count for the cell containing this Cell object.
     */
    private int getBranchCountForCell(Cell cell) {
        // For base level cells, get from cache
        CellData data = getCellData(cell.x, cell.y);
        return data.branchCount;
    }

    private Random initRandomGenerator(int x, int y, int level) {
        long seed = (541L * x + 79L * y + level * 1000L + salt) & 0x7FFFFFFFL;
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

                // Scale by resolution to map cell-space to noise-space
                Point2D scaled = new Point2D(p2d.x / cell.resolution, p2d.y / cell.resolution);
                double elevation = evaluateControlFunction(scaled.x, scaled.y);
                points[i][j] = new Point3D(scaled.x, scaled.y, elevation);
            }
        }
        return points;
    }

    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            // Convert noise coordinates back to world coordinates
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
     * Subdivide segments using Catmull-Rom splines with curvature blending.
     */
    private List<Segment3D> subdivideSegmentsWithSpline(List<Segment3D> segments, int numDivisions, int level) {
        if (segments.isEmpty() || numDivisions <= 1) {
            return segments;
        }

        // Calculate level-adjusted curvature
        double levelCurvature = curvature * Math.pow(curvatureFalloff, level - 1);

        List<Segment3D> subdivided = new ArrayList<>();

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            // Get control points (predecessor and successor)
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
                    // Blend between linear and spline based on curvature
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

    /**
     * Get the maximum connection distance for a given level.
     */
    private double getConnectDistance(int level) {
        if (connectDistance > 0) {
            return connectDistance / level;
        }
        // Auto-calculate based on grid size and level
        double cellSize = 1.0 / Math.pow(2, level - 1);  // In noise space
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
     * Find nearest segment using parallel streams for large lists.
     * Returns 2D distance (Z coordinate ignored in distance calculation).
     */
    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream = (segments.size() > PARALLEL_THRESHOLD)
            ? segments.parallelStream()
            : segments.stream();

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

    /**
     * Find nearest segment by weighted distance (for WEIGHTED return type).
     */
    private NearestSegmentResult findNearestSegmentWeighted(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream = (segments.size() > PARALLEL_THRESHOLD)
            ? segments.parallelStream()
            : segments.stream();

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
                // Return 2D distance to nearest segment
                NearestSegmentResult nearestDist = findNearestSegment(point, segments);
                if (nearestDist == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestDist.distance;

            case WEIGHTED:
                // Return minimum weighted distance (distance / segment.level)
                NearestSegmentResult nearestWeighted = findNearestSegmentWeighted(point, segments);
                if (nearestWeighted == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestWeighted.weightedDistance;

            case ELEVATION:
            default:
                // Return elevation-based result
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

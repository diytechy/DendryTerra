package com.dfsek.terra.addons.dendry;

import com.dfsek.seismic.type.sampler.Sampler;
import com.dfsek.terra.addons.dendry.math.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Based on the algorithm described in Gaillard et al. (I3D 2019).
 */
public class DendrySampler implements Sampler {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendrySampler.class);

    private final int resolution;
    private final double epsilon;
    private final double delta;
    private final double slope;
    private final double frequency;
    private final DendryReturnType returnType;
    private final Sampler controlSampler;
    private final long salt;

    private static final int SUBDIVISIONS = 4;
    private static final int CACHE_SIZE = 128;

    // Point cache for performance
    private final Point2D[][] pointCache;

    public DendrySampler(int resolution, double epsilon, double delta,
                         double slope, double frequency,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt) {
        this.resolution = resolution;
        this.epsilon = epsilon;
        this.delta = delta;
        this.slope = slope;
        this.frequency = frequency;
        this.returnType = returnType;
        this.controlSampler = controlSampler;
        this.salt = salt;

        // Initialize point cache
        this.pointCache = new Point2D[CACHE_SIZE][CACHE_SIZE];
        initPointCache();
    }

    private void initPointCache() {
        int halfCache = CACHE_SIZE / 2;
        for (int x = -halfCache; x < halfCache; x++) {
            for (int y = -halfCache; y < halfCache; y++) {
                pointCache[x + halfCache][y + halfCache] = generatePoint(x, y);
            }
        }
    }

    @Override
    public double getSample(long seed, double x, double z) {
        return evaluate(seed, x * frequency, z * frequency);
    }

    @Override
    public double getSample(long seed, double x, double y, double z) {
        return evaluate(seed, x * frequency, z * frequency);
    }

    private double evaluate(long seed, double x, double y) {
        // Displacement factors for each level
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slopes for terrain generation
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 1: Base resolution
        Cell cell1 = getCell(x, y, 1);
        Point3D[][] points1 = generateNeighboringPoints3D(cell1, 9);
        List<Segment3D> segments1 = generateSegments(points1);
        segments1 = subdivideSegments(segments1, SUBDIVISIONS);
        displaceSegments(segments1, displacementLevel1, cell1);

        if (resolution == 1) {
            return computeResult(x, y, segments1, 1);
        }

        // Level 2: 2x resolution
        Cell cell2 = getCell(x, y, 2);
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);
        List<Segment3D> segments2 = generateSubSegments(points2, segments1, minSlopeLevel2);
        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            List<Segment3D> allSegments = new ArrayList<>(segments1);
            allSegments.addAll(segments2);
            return computeResult(x, y, allSegments, 2);
        }

        // Level 3: 4x resolution
        Cell cell3 = getCell(x, y, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);
        List<Segment3D> allPreviousSegments = new ArrayList<>(segments1);
        allPreviousSegments.addAll(segments2);
        List<Segment3D> segments3 = generateSubSegments(points3, allPreviousSegments, minSlopeLevel3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allPreviousSegments.addAll(segments3);
            return computeResult(x, y, allPreviousSegments, 3);
        }

        // Level 4: 8x resolution
        Cell cell4 = getCell(x, y, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allPreviousSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allPreviousSegments, minSlopeLevel4);

        if (resolution == 4) {
            allPreviousSegments.addAll(segments4);
            return computeResult(x, y, allPreviousSegments, 4);
        }

        // Level 5: 16x resolution
        Cell cell5 = getCell(x, y, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allPreviousSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allPreviousSegments, minSlopeLevel5);
        allPreviousSegments.addAll(segments5);

        return computeResult(x, y, allPreviousSegments, 5);
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
        int cellX = (int) Math.floor(x * resolution);
        int cellY = (int) Math.floor(y * resolution);
        return new Cell(cellX, cellY, resolution);
    }

    private Point2D generatePoint(int cellX, int cellY) {
        Random rng = initRandomGenerator(cellX, cellY, 1);
        double px = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        double py = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        return new Point2D(cellX + px, cellY + py);
    }

    private Point2D generatePointCached(int cellX, int cellY) {
        int halfCache = CACHE_SIZE / 2;
        if (cellX >= -halfCache && cellX < halfCache && cellY >= -halfCache && cellY < halfCache) {
            return pointCache[cellX + halfCache][cellY + halfCache];
        }
        return generatePoint(cellX, cellY);
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

                Point2D p2d = generatePointCached(px, py);
                // Scale by resolution
                Point2D scaled = new Point2D(p2d.x / cell.resolution, p2d.y / cell.resolution);
                double elevation = evaluateControlFunction(scaled.x, scaled.y);
                points[i][j] = new Point3D(scaled.x, scaled.y, elevation);
            }
        }
        return points;
    }

    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            // Use the control sampler to get base elevation
            return controlSampler.getSample(salt, x / frequency, y / frequency);
        }
        // Default: simple gradient
        return x * 0.1;
    }

    private List<Segment3D> generateSegments(Point3D[][] points) {
        List<Segment3D> segments = new ArrayList<>();
        int size = points.length;

        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Point3D current = points[i][j];

                // Find lowest neighbor
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

                // Create segment from current to lowest (if different)
                if (current.distanceSquaredTo(lowest) > MathUtils.EPSILON) {
                    segments.add(new Segment3D(current, lowest));
                }
            }
        }
        return segments;
    }

    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions) {
        List<Segment3D> subdivided = new ArrayList<>();

        for (Segment3D seg : segments) {
            // For simplicity, just linearly subdivide
            // A full implementation would use Catmull-Rom with predecessor/successor segments
            Point3D prev = seg.a;
            for (int i = 1; i <= numDivisions; i++) {
                double t = (double) i / numDivisions;
                Point3D next = (i == numDivisions) ? seg.b : Point3D.lerp(seg.a, seg.b, t);
                subdivided.add(new Segment3D(prev, next));
                prev = next;
            }
        }
        return subdivided;
    }

    private void displaceSegments(List<Segment3D> segments, double displacementFactor, Cell cell) {
        if (displacementFactor < MathUtils.EPSILON) return;

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            // Get perpendicular direction
            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            Vec2D perp = dir.rotateCCW90().normalize();

            // Generate random displacement
            Random rng = initRandomGenerator((int)(seg.a.x * 100), (int)(seg.a.y * 100), cell.resolution);
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor;

            // Displace the midpoint (simplified - in full impl would displace all intermediate points)
            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            // Replace with two segments through displaced midpoint
            segments.set(i, new Segment3D(seg.a, displacedMid));
            if (i + 1 < segments.size()) {
                // Adjust next segment if it connects
                Segment3D next = segments.get(i + 1);
                if (next.a.equals(seg.b)) {
                    segments.set(i + 1, new Segment3D(displacedMid, next.b));
                }
            }
        }
    }

    private List<Segment3D> generateSubSegments(Point3D[][] points, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> subSegments = new ArrayList<>();
        int size = points.length;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Point3D point = points[i][j];

                // Find nearest segment in parent level
                NearestSegmentResult nearest = findNearestSegment(point.projectZ(), parentSegments);
                if (nearest == null || nearest.distance > 10.0) continue;

                // Calculate elevation with slope constraint
                double elevationFromControl = evaluateControlFunction(point.x, point.y);
                double elevationWithSlope = nearest.closestPoint.z + minSlope * nearest.distance;
                double elevation = Math.max(elevationWithSlope, elevationFromControl);

                Point3D adjustedPoint = new Point3D(point.x, point.y, elevation);

                // Connect to nearest segment
                if (nearest.distance > MathUtils.EPSILON) {
                    Point3D connectionPoint = new Point3D(
                        nearest.closestPoint2D.x,
                        nearest.closestPoint2D.y,
                        nearest.closestPoint.z
                    );
                    subSegments.add(new Segment3D(adjustedPoint, connectionPoint));
                }
            }
        }
        return subSegments;
    }

    private static class NearestSegmentResult {
        final double distance;
        final Point2D closestPoint2D;
        final Point3D closestPoint;
        final Segment3D segment;

        NearestSegmentResult(double distance, Point2D closestPoint2D, Point3D closestPoint, Segment3D segment) {
            this.distance = distance;
            this.closestPoint2D = closestPoint2D;
            this.closestPoint = closestPoint;
            this.segment = segment;
        }
    }

    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        double minDist = Double.MAX_VALUE;
        Point2D closestPoint2D = null;
        Point3D closestPoint3D = null;
        Segment3D nearestSeg = null;

        for (Segment3D seg : segments) {
            MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);
            if (result.distance < minDist) {
                minDist = result.distance;
                closestPoint2D = result.closestPoint;
                // Interpolate Z coordinate
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                closestPoint3D = new Point3D(result.closestPoint.x, result.closestPoint.y, z);
                nearestSeg = seg;
            }
        }

        return new NearestSegmentResult(minDist, closestPoint2D, closestPoint3D, nearestSeg);
    }

    private double computeResult(double x, double y, List<Segment3D> segments, int level) {
        Point2D point = new Point2D(x, y);
        NearestSegmentResult nearest = findNearestSegment(point, segments);

        if (nearest == null) {
            return evaluateControlFunction(x, y);
        }

        switch (returnType) {
            case DISTANCE:
                return nearest.distance;

            case WEIGHTED:
                // Weight by level - higher levels have less weight
                return nearest.distance * (1.0 / level);

            case ELEVATION:
            default:
                // Return the elevation at the nearest point, with slope-based interpolation
                double baseElevation = nearest.closestPoint.z;
                double slopeContribution = nearest.distance * slope;
                return baseElevation + slopeContribution;
        }
    }
}

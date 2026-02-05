package dendryterra;

import dendryterra.math.*;
import java.util.*;

/**
 * Container for connected network of points and segments.
 * The single source of truth for network topology.
 *
 * Points are stored as NetworkPoint objects with unique indices.
 * Segments reference points by index (srtIdx, endIdx).
 * Connection counts are automatically maintained when segments are added/removed.
 */
public class SegmentList {
    private final List<NetworkPoint> points;
    private final List<Segment3D> segments;
    private int nextIndex;  // For generating unique indices
    private SegmentListConfig config;  // Global configuration

    public SegmentList() {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.nextIndex = 0;
        this.config = new SegmentListConfig();
    }
    
    public SegmentList(long salt) {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.nextIndex = 0;
        this.config = new SegmentListConfig(salt);
    }
    
    public SegmentList(SegmentListConfig config) {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.nextIndex = 0;
        this.config = config;
    }

    // ========== Point Operations ==========
    
    /**
     * Get the global configuration.
     */
    public SegmentListConfig getConfig() {
        return config;
    }
    
    /**
     * Set the global configuration.
     */
    public void setConfig(SegmentListConfig config) {
        this.config = config;
    }
    
    /**
     * Get global salt value used for deterministic randomness.
     */
    public long getSalt() {
        return config.salt;
    }
    
    /**
     * Set global salt value used for deterministic randomness.
     */
    public void setSalt(long salt) {
        config.salt = salt;
    }

    /**
     * Add a new point with the given position, type, and level.
     * @return The index assigned to this point
     */
    public int addPoint(Point3D position, PointType type, int level) {
        int idx = nextIndex++;
        points.add(new NetworkPoint(position, idx, type, level));
        return idx;
    }

    /**
     * Add an existing NetworkPoint, re-indexing it for this list.
     * @return The new index assigned to this point
     */
    public int addPoint(NetworkPoint point) {
        int idx = nextIndex++;
        NetworkPoint reindexed = point.withIndex(idx);
        points.add(reindexed);
        return idx;
    }

    /**
     * Get a point by its index.
     */
    public NetworkPoint getPoint(int index) {
        return points.get(index);
    }

    /**
     * Update a point at the given index.
     */
    public void updatePoint(int index, NetworkPoint newPoint) {
        points.set(index, newPoint);
    }

    /**
     * Get the total number of points.
     */
    public int getPointCount() {
        return points.size();
    }

    /**
     * Get an unmodifiable view of all points.
     */
    public List<NetworkPoint> getPoints() {
        return Collections.unmodifiableList(points);
    }

    // ========== Segment Operations ==========

    /**
     * Add a segment and update connection counts on its endpoints.
     * Note: Currently accepts Segment3D with Point3D srt/end.
     * After Segment3D is modified to use indices, this will accept index-based segments.
     */
    public void addSegment(Segment3D segment) {
        segments.add(segment);
        // Connection count updates will be handled after Segment3D transition
        // For now, we'll update counts based on finding points by position
        updateConnectionCountsForSegment(segment, 1);
    }

    /**
     * Add a segment directly using point indices.
     * Creates a Segment3D internally.
     */
    public void addSegment(int srtIdx, int endIdx, int level,
                           Vec2D tangentSrt, Vec2D tangentEnd) {
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);

        // Create Segment3D (using current Point3D-based format)
        Segment3D segment = new Segment3D(
            srt.position, end.position, level,
            tangentSrt, tangentEnd,
            srt.pointType, end.pointType
        );
        segments.add(segment);

        // Update connection counts
        points.set(srtIdx, srt.incrementConnections());
        points.set(endIdx, end.incrementConnections());
    }

    /**
     * Add a segment using new network point to known existing point in segment.
     * Uses global configuration parameters.
     */
    public void addSegment(NetworkPoint srtNetPnt, int endIdx, int level, double maxSegmentLength) {
        // Add the start point to get its index
        int srtIdx = addPoint(srtNetPnt);
        
        // Call full implementation using global config
        addSegmentWithDivisions(srtIdx, endIdx, level, maxSegmentLength);
    }
    
    /**
     * Add a segment with full implementation using global configuration.
     * This is the main implementation that creates multiple connected segments from a single call.
     */
    public void addSegmentWithDivisions(int A, int B, int level, double maxSegmentLength) {
        // Fetch points once and cache hash codes to avoid duplicate calculations
        NetworkPoint ptA = points.get(A);
        NetworkPoint ptB = points.get(B);
        int hashA = ptA.position.hashCode();
        int hashB = ptB.position.hashCode();

        // Determine start/end ordering for consistent spline generation
        int srtIdx, endIdx;
        if (ptA.connections > 0 && ptB.connections == 0) {
            // Side B has no connections, it should be the start for consistency
            srtIdx = B;
            endIdx = A;
        } else if (ptB.connections > 0 && ptA.connections == 0) {
            // Side A has no connections, it should be the start for consistency
            srtIdx = A;
            endIdx = B;
        } else {
            // Both sides have connections or both have none, order by hash for consistency
            if (hashA < hashB) {
                srtIdx = A;
                endIdx = B;
            } else {
                srtIdx = B;
                endIdx = A;
            }
        }

        NetworkPoint srt = (srtIdx == A) ? ptA : ptB;
        NetworkPoint end = (endIdx == A) ? ptA : ptB;
        int hashSrt = (srtIdx == A) ? hashA : hashB;
        int hashEnd = (endIdx == A) ? hashA : hashB;

        // Instantiate a random generator for consistent "random" artifacts per segment
        long seed = (541L * (hashSrt + hashEnd) + config.salt) & 0x7FFFFFFFL;
        Random rng = new Random(seed);

        // Step 1: Compute tangents based on connection patterns using global config
        Vec2D[] tangents = computeTangentsForConnection(srtIdx, endIdx, rng);
        Vec2D tangentSrt = tangents[0];
        Vec2D tangentEnd = tangents[1];

        
        // Step 2: Subdivide long segments if needed using provided maxSegmentLength
        double distance = srt.position.distanceTo(end.position);
        int numDivisions = (int) Math.ceil(distance / maxSegmentLength);
        if (numDivisions <= 1) {
            // Single segment - add directly
            addSegment(srtIdx, endIdx, level, tangentSrt, tangentEnd);
        } else {
            // Multiple segments - pass computed tangents and RNG to avoid recomputation
            createSubdividedSegments(srtIdx, endIdx, level, numDivisions, tangentSrt, tangentEnd, rng);
        }
    }
    
    /**
     * Add a segment to with two new network points, only used for trunk initialization.
     */
    public void addSegmentWithDivisions(NetworkPoint srtNetPnt, NetworkPoint endNetPnt, int level, double maxSegmentLength) {

        int endIdx = addPoint(endNetPnt);
        addSegment(srtNetPnt, endIdx, level, maxSegmentLength);
    }
    
    /**
     * Helper method to rotate a vector by an angle.
     */
    private Vec2D rotateVector(Vec2D v, double angle) {
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        return new Vec2D(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
    }
    
    /**
     * Helper method to create a vector from an angle.
     */
    private Vec2D createVectorFromAngle(double angle) {
        return new Vec2D(Math.cos(angle), Math.sin(angle));
    }
    
    /**
     * Compute tangents for a connection based on existing connectivity patterns.
     * Uses global configuration parameters.
     */
    private Vec2D[] computeTangentsForConnection(int srtIdx, int endIdx, Random NoiseGen) {
        
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);
        
        // For straight segments or when splines are disabled, use simple direction
        if (!config.useSplines || config.curvature <= 0) {
            Vec2D direction = new Vec2D(srt.position.projectZ(), end.position.projectZ());
            if (direction.lengthSquared() > MathUtils.EPSILON) {
                direction = direction.normalize();
            }
            return new Vec2D[] { direction, direction };
        }
        
        // Compute tangents based on connection patterns
        Vec2D tangentSrt = computePointTangent(srtIdx, endIdx, true,NoiseGen);
        Vec2D tangentEnd = computePointTangent(endIdx, srtIdx, false,NoiseGen);
        
        return new Vec2D[] { tangentSrt, tangentEnd };
    }
    
    /**
     * Compute tangent for a specific point based on its connections.
     * Uses global configuration parameters.
     */
    private Vec2D computePointTangent(int pointIdx, int targetIdx, boolean isStart, Random NoiseGen) {
        
        NetworkPoint point = points.get(pointIdx);
        NetworkPoint target = points.get(targetIdx);
        //Initialize point variables that determine tangent and twist.
        Vec2D angle = new Vec2D(0,0);
        double slope = 0;
        // Base direction toward target
        Vec2D toTarget = new Vec2D(point.position.projectZ(), target.position.projectZ());
        if (toTarget.lengthSquared() < MathUtils.EPSILON) {
            return new Vec2D(1, 0); // Default direction
        }
        toTarget = toTarget.normalize();
        // Get slope tangent trajectory based on point:
        if (isStart) {
            angle = point.position.getTangentVector();
            slope = Math.abs(point.position.getSlope());
        }
        else {
            angle = target.position.getTangentVector().negate();
            slope = Math.abs(point.position.getSlope());
        }
        // Adjust based on connection count
        if (point.connections == 0) {
            // No existing connections - use flow direction with deterministic twist
            double twist = (NoiseGen.nextFloat() * 2.0 * Math.min(slope/config.SlopeWithoutTwist, 1.0) - 1.0) * config.maxTwistAngle;
            return rotateVector(angle, twist);
        } else if (point.connections == 1) {
            // One existing connection - check if aligned with target
            Vec2D existingDir = getExistingConnectionDirection(pointIdx);
            if (existingDir != null) {
                double alignment = toTarget.dot(existingDir);
                if (alignment > 0.7) {
                    // Aligned - use existing for continuity
                    return isStart ? existingDir : existingDir.negate();
                }
            }
            // Not aligned - use direction to target with small offset
            double offset = (Math.random() - 0.5) * config.SlopeWithoutTwist * 0.5;
            return rotateVector(toTarget, offset);
        } else {
            // Multiple connections - compute offset tangent
            Vec2D existingDir = getExistingConnectionDirection(pointIdx);
            if (existingDir != null) {
                // Offset 20-70 degrees on the side of the target
                double angleToTarget = Math.atan2(toTarget.y, toTarget.x);
                double offsetAngle = config.SlopeWithoutTwist * (0.4 + Math.random() * 0.3); // 20-70 degrees
                return createVectorFromAngle(angleToTarget + offsetAngle);
            }
            return toTarget;
        }
    }
    
    /**
     * Get the direction of existing connections from a point.
     */
    private Vec2D getExistingConnectionDirection(int pointIdx) {
        NetworkPoint point = points.get(pointIdx);
        Point3D pointPos = point.position;
        
        // Find segments connected to this point
        for (Segment3D seg : segments) {
            if (seg.srt.distanceSquaredTo(pointPos) < MathUtils.EPSILON) {
                return new Vec2D(pointPos.projectZ(), seg.end.projectZ()).normalize();
            }
            if (seg.end.distanceSquaredTo(pointPos) < MathUtils.EPSILON) {
                return new Vec2D(pointPos.projectZ(), seg.srt.projectZ()).normalize();
            }
        }
        return null;
    }
    
    /**
     * Create subdivided segments between two points using spline or linear interpolation.
     * Uses global configuration parameters.
     *
     * @param srtIdx Start point index
     * @param endIdx End point index
     * @param level Resolution level
     * @param numDivisions Number of divisions to create
     * @param tangentSrt Pre-computed tangent at start point
     * @param tangentEnd Pre-computed tangent at end point
     * @param rng Random number generator for deterministic jitter
     */
    private void createSubdividedSegments(int srtIdx, int endIdx, int level, int numDivisions,
                                          Vec2D tangentSrt, Vec2D tangentEnd, Random rng) {
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);

        // Create intermediate points
        int prevIdx = srtIdx;

        for (int i = 1; i < numDivisions; i++) {
            double t = (double) i / numDivisions;
            Point3D intermediatePoint;

            if (config.useSplines && config.curvature > 0) {
                // Use cubic Hermite spline interpolation with jitter
                intermediatePoint = interpolateHermiteSpline(srt.position, end.position,
                                                           tangentSrt, tangentEnd, t, config.tangentStrength, rng);
            } else {
                // Use linear interpolation with jitter
                intermediatePoint = interpolateLinearWithJitter(srt.position, end.position, t, rng);
            }

            // Add intermediate point
            int intermediateIdx = addPoint(intermediatePoint, PointType.KNOT, level);

            // Create segment from previous to intermediate
            addSegment(prevIdx, intermediateIdx, level,
                      prevIdx == srtIdx ? tangentSrt : null,
                      null);

            prevIdx = intermediateIdx;
        }

        // Create final segment to end point
        addSegment(prevIdx, endIdx, level,
                  null,
                  tangentEnd);
    }
    
    /**
     * Cubic Hermite spline interpolation with deterministic jitter.
     * References subdivideWithSpline logic from DendrySampler.
     *
     * @param srt Start point position
     * @param end End point position
     * @param tangentSrt Tangent at start point
     * @param tangentEnd Tangent at end point
     * @param t Interpolation parameter [0, 1]
     * @param tangentStrength Scale factor for tangent magnitudes
     * @param rng Random number generator for jitter
     * @return Interpolated point with jitter applied
     */
    private Point3D interpolateHermiteSpline(Point3D srt, Point3D end, Vec2D tangentSrt, Vec2D tangentEnd,
                                             double t, double tangentStrength, Random rng) {
        double t2 = t * t;
        double t3 = t2 * t;
        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double segLength = srt.projectZ().distanceTo(end.projectZ());
        double tangentScale = segLength * tangentStrength;

        double x = h00 * srt.x + h10 * (tangentSrt != null ? tangentSrt.x * tangentScale : 0)
                 + h01 * end.x + h11 * (tangentEnd != null ? tangentEnd.x * tangentScale : 0);
        double y = h00 * srt.y + h10 * (tangentSrt != null ? tangentSrt.y * tangentScale : 0)
                 + h01 * end.y + h11 * (tangentEnd != null ? tangentEnd.y * tangentScale : 0);
        double z = MathUtils.lerp(srt.z, end.z, t);  // Linear interpolation for elevation

        // Add small deterministic jitter for natural appearance (intermediate points only)
        double jitterMagnitude = segLength * 0.02; // 2% of segment length
        double jitterX = (rng.nextDouble() - 0.5) * jitterMagnitude;
        double jitterY = (rng.nextDouble() - 0.5) * jitterMagnitude;

        return new Point3D(x + jitterX, y + jitterY, z);
    }
    
    /**
     * Linear interpolation with deterministic jitter.
     * References jitter application logic from DendrySampler.
     */
    private Point3D interpolateLinearWithJitter(Point3D srt, Point3D end, double t, Random rng) {
        // Linear interpolation
        double x = MathUtils.lerp(srt.x, end.x, t);
        double y = MathUtils.lerp(srt.y, end.y, t);
        double z = MathUtils.lerp(srt.z, end.z, t);
        
        // Add small deterministic jitter for natural appearance
        double jitterMagnitude = 0.01; // Small jitter
        double jitterX = (rng.nextDouble() - 0.5) * jitterMagnitude;
        double jitterY = (rng.nextDouble() - 0.5) * jitterMagnitude;
        
        return new Point3D(x + jitterX, y + jitterY, z);
    }

    /**
     * Remove a segment by index and decrement connection counts.
     */
    public void removeSegment(int segmentIndex) {
        Segment3D seg = segments.get(segmentIndex);
        updateConnectionCountsForSegment(seg, -1);
        segments.remove(segmentIndex);
    }

    /**
     * Get a segment by its index.
     */
    public Segment3D getSegment(int index) {
        return segments.get(index);
    }

    /**
     * Get an unmodifiable view of all segments.
     */
    public List<Segment3D> getSegments() {
        return Collections.unmodifiableList(segments);
    }

    /**
     * Get the total number of segments.
     */
    public int getSegmentCount() {
        return segments.size();
    }

    // ========== Query Operations ==========

    /**
     * Find a point by its 3D position within epsilon tolerance.
     * @return Point index, or -1 if not found
     */
    public int findPointByPosition(Point3D pos, double epsilon) {
        double epsSq = epsilon * epsilon;
        for (int i = 0; i < points.size(); i++) {
            if (points.get(i).position.distanceSquaredTo(pos) < epsSq) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find all points within the given horizontal (XZ) distance of a position.
     * Excludes EDGE points.
     */
    public List<Integer> findPointsWithinDistance(Point3D pos, double maxDist) {
        List<Integer> result = new ArrayList<>();
        double maxDistSq = maxDist * maxDist;
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;
            double distSq = p.position.projectZ().distanceSquaredTo(pos.projectZ());
            if (distSq <= maxDistSq) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Check if two points are already connected by a segment.
     * Uses position matching since current Segment3D uses Point3D.
     */
    public boolean areConnected(int idx1, int idx2) {
        Point3D pos1 = points.get(idx1).position;
        Point3D pos2 = points.get(idx2).position;
        double epsilon = 1e-9;

        for (Segment3D seg : segments) {
            boolean match1 = seg.srt.distanceSquaredTo(pos1) < epsilon &&
                             seg.end.distanceSquaredTo(pos2) < epsilon;
            boolean match2 = seg.srt.distanceSquaredTo(pos2) < epsilon &&
                             seg.end.distanceSquaredTo(pos1) < epsilon;
            if (match1 || match2) {
                return true;
            }
        }
        return false;
    }

    /**
     * Find the closest point in this SegmentList to the given position.
     * Excludes EDGE points.
     * @return Point index, or -1 if no points exist
     */
    public int findClosestPoint(Point3D pos) {
        int bestIdx = -1;
        double bestDistSq = Double.MAX_VALUE;

        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;

            double distSq = p.position.projectZ().distanceSquaredTo(pos.projectZ());
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // ========== Utility Operations ==========

    /**
     * Mark all points with exactly 1 connection as LEAF type.
     */
    public void markLeafPoints() {
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.connections == 1 && p.pointType != PointType.EDGE) {
                points.set(i, p.withPointType(PointType.LEAF));
            }
        }
    }

    /**
     * Shift all point elevations down so minimum is at 0.
     * Used for asterism normalization.
     */
    public void normalizeElevations() {
        if (points.isEmpty()) return;

        // Find minimum elevation
        double minZ = Double.MAX_VALUE;
        for (NetworkPoint p : points) {
            minZ = Math.min(minZ, p.position.z);
        }

        if (minZ == 0) return;  // Already normalized

        // Shift all points
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            Point3D newPos = new Point3D(p.position.x, p.position.y, p.position.z - minZ);
            points.set(i, p.withPosition(newPos));
        }

        // Also shift segments (they contain Point3D)
        List<Segment3D> shiftedSegments = new ArrayList<>();
        for (Segment3D seg : segments) {
            Point3D newSrt = new Point3D(seg.srt.x, seg.srt.y, seg.srt.z - minZ);
            Point3D newEnd = new Point3D(seg.end.x, seg.end.y, seg.end.z - minZ);
            shiftedSegments.add(new Segment3D(newSrt, newEnd, seg.level,
                                               seg.tangentSrt, seg.tangentEnd,
                                               seg.srtType, seg.endType));
        }
        segments.clear();
        segments.addAll(shiftedSegments);
    }

    /**
     * Create a copy of this SegmentList.
     */
    public SegmentList copy() {
        SegmentList copy = new SegmentList(this.config);
        copy.nextIndex = this.nextIndex;
        for (NetworkPoint p : this.points) {
            copy.points.add(p);  // NetworkPoint is immutable
        }
        for (Segment3D s : this.segments) {
            copy.segments.add(s);  // Segment3D is immutable
        }
        return copy;
    }

    /**
     * Check if this SegmentList is empty.
     */
    public boolean isEmpty() {
        return points.isEmpty();
    }

    // ========== Internal Helpers ==========

    /**
     * Update connection counts for points referenced by a segment.
     * @param delta +1 for adding, -1 for removing
     */
    private void updateConnectionCountsForSegment(Segment3D segment, int delta) {
        // Find point indices by position matching
        int srtIdx = findPointByPosition(segment.srt, 1e-9);
        int endIdx = findPointByPosition(segment.end, 1e-9);

        if (srtIdx >= 0) {
            NetworkPoint p = points.get(srtIdx);
            points.set(srtIdx, p.withConnections(Math.max(0, p.connections + delta)));
        }
        if (endIdx >= 0) {
            NetworkPoint p = points.get(endIdx);
            points.set(endIdx, p.withConnections(Math.max(0, p.connections + delta)));
        }
    }

    // ========== Conversion Methods ==========

    /**
     * Convert to old format: List<Segment3D> with Point3D srt/end.
     * Used for backward compatibility with existing code that expects Point3D-based segments.
     */
    public List<Segment3D> toSegment3DList() {
        List<Segment3D> result = new ArrayList<>();

        for (Segment3D seg : segments) {
            if (seg.isPointBased()) {
                // Already has Point3D - use as-is
                result.add(seg);
            } else {
                // Index-based - resolve to Point3D
                NetworkPoint srtPt = points.get(seg.srtIdx);
                NetworkPoint endPt = points.get(seg.endIdx);

                result.add(new Segment3D(
                    srtPt.position, endPt.position, seg.level,
                    seg.tangentSrt, seg.tangentEnd,
                    srtPt.pointType, endPt.pointType
                ));
            }
        }

        return result;
    }

    /**
     * Create a SegmentList from an existing List<Segment3D>.
     * Used to import legacy segments into the new structure.
     */
    public static SegmentList fromSegment3DList(List<Segment3D> segments, int level) {
        return fromSegment3DList(segments, level, 12345); // Default salt
    }
    
    /**
     * Create a SegmentList from an existing List<Segment3D> with specified salt.
     * Used to import legacy segments into the new structure.
     */
    public static SegmentList fromSegment3DList(List<Segment3D> segments, int level, long salt) {
        SegmentList result = new SegmentList(salt);

        // Build a map of positions to point indices to avoid duplicates
        Map<Long, Integer> positionToIndex = new HashMap<>();

        for (Segment3D seg : segments) {
            if (!seg.isPointBased()) {
                throw new IllegalArgumentException("Cannot import index-based segments");
            }

            // Get or create point for srt
            int srtIdx = getOrCreatePointIndex(result, positionToIndex, seg.srt, seg.srtType, level);

            // Get or create point for end
            int endIdx = getOrCreatePointIndex(result, positionToIndex, seg.end, seg.endType, level);

            // Add segment using indices
            result.addSegment(srtIdx, endIdx, level, seg.tangentSrt, seg.tangentEnd);
        }

        return result;
    }

    /**
     * Helper to get or create a point index for a position.
     */
    private static int getOrCreatePointIndex(SegmentList list, Map<Long, Integer> positionToIndex,
                                              Point3D pos, PointType type, int level) {
        // Quantize position for lookup
        long key = quantizePosition(pos);

        Integer existingIdx = positionToIndex.get(key);
        if (existingIdx != null) {
            return existingIdx;
        }

        // Create new point
        int idx = list.addPoint(pos, type, level);
        positionToIndex.put(key, idx);
        return idx;
    }

    /**
     * Quantize a position to a long key for deduplication.
     */
    private static long quantizePosition(Point3D pos) {
        // Use high precision to avoid false matches
        long qx = Math.round(pos.x * 100000);
        long qy = Math.round(pos.y * 100000);
        return (qx << 32) | (qy & 0xFFFFFFFFL);
    }

    @Override
    public String toString() {
        return String.format("SegmentList(points=%d, segments=%d)", points.size(), segments.size());
    }
}

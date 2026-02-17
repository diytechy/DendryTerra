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
    private final List<SegmentIdx> segments;
    private final Map<Integer, List<SegmentConnection>> pointToSegments;  // Maps point index to connected segments
    private int nextIndex;  // For generating unique indices
    private SegmentListConfig config;  // Global configuration

    /**
     * Tracks a segment connection to a point, storing which end of the segment connects.
     */
    private static class SegmentConnection {
        final int segmentIndex;
        final boolean isStart;  // true if point is at segment start, false if at end

        SegmentConnection(int segmentIndex, boolean isStart) {
            this.segmentIndex = segmentIndex;
            this.isStart = isStart;
        }
    }

    public SegmentList() {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.pointToSegments = new HashMap<>();
        this.nextIndex = 0;
        this.config = new SegmentListConfig();
    }

    public SegmentList(long salt) {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.pointToSegments = new HashMap<>();
        this.nextIndex = 0;
        this.config = new SegmentListConfig(salt);
    }

    public SegmentList(SegmentListConfig config) {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.pointToSegments = new HashMap<>();
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
     * Force all points in the segment list to a specific elevation.
     * Useful for flattening level 0 constellations to change path preferences.
     *
     * @param elevation The elevation to set for all points
     */
    public void forceAllPointElevations(double elevation) {
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            Point3D newPosition = new Point3D(p.position.x, p.position.y, elevation);
            points.set(i, p.withPosition(newPosition));
        }
    }

    /**
     * Ensure water flows downhill by propagating elevation reduction when a new point
     * connects at a lower elevation than the existing network.
     *
     * When sourceIdx has lower elevation than targetIdx, all points connected to targetIdx
     * (in the chain toward level 0) are reduced ratiometrically by (sourceZ / targetZ).
     *
     * @param sourceIdx The newly added point index
     * @param targetIdx The existing point it's connecting to
     * @param level The current level (only applies at level > 0)
     */
    public void ensureDownhillFlow(int sourceIdx, int targetIdx, int level) {
        if (level == 0) return; // Level 0 handled separately

        NetworkPoint source = points.get(sourceIdx);
        NetworkPoint target = points.get(targetIdx);

        double sourceZ = source.position.z;
        double targetZ = target.position.z;

        // Only adjust if source is lower than target (water would flow uphill)
        if (sourceZ >= targetZ || targetZ <= MathUtils.EPSILON) return;

        // Calculate reduction ratio
        double ratio = sourceZ / targetZ;

        // Propagate reduction through the chain toward level 0
        Set<Integer> visited = new HashSet<>();
        propagateElevationReduction(targetIdx, ratio, visited);
    }

    /**
     * Recursively propagate elevation reduction through connected points.
     * Only follows connections to points at the same or lower level (toward asterism).
     *
     * @param pointIdx Current point to reduce
     * @param ratio Reduction ratio to apply (new_elevation = old_elevation * ratio)
     * @param visited Set of already visited point indices (to avoid cycles)
     */
    private void propagateElevationReduction(int pointIdx, double ratio, Set<Integer> visited) {
        if (visited.contains(pointIdx)) return;
        visited.add(pointIdx);

        NetworkPoint p = points.get(pointIdx);

        // Reduce this point's elevation
        double newZ = p.position.z * ratio;
        Point3D newPosition = new Point3D(p.position.x, p.position.y, newZ);
        points.set(pointIdx, p.withPosition(newPosition));

        // Find connected points through segments
        List<SegmentConnection> connections = pointToSegments.get(pointIdx);
        if (connections == null) return;

        for (SegmentConnection conn : connections) {
            SegmentIdx seg = segments.get(conn.segmentIndex);

            // Get the other endpoint
            int otherIdx = conn.isStart ? seg.endIdx : seg.srtIdx;

            // Only propagate to points at same or lower level (toward level 0)
            NetworkPoint other = points.get(otherIdx);
            if (other.level <= p.level) {
                propagateElevationReduction(otherIdx, ratio, visited);
            }
        }
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
     * Add a segment directly using point indices.
     * Creates a SegmentIdx internally.
     */
    public void addBasicSegment(int srtIdx, int endIdx, int level,
                           Vec2D tangentSrt, Vec2D tangentEnd) {
        // Create index-based SegmentIdx
        SegmentIdx segment = new SegmentIdx(
            srtIdx, endIdx, level,
            tangentSrt, tangentEnd
        );
        int segmentIndex = segments.size();
        segments.add(segment);

        // Track segment connections by point index
        pointToSegments.computeIfAbsent(srtIdx, k -> new ArrayList<>())
            .add(new SegmentConnection(segmentIndex, true));
        pointToSegments.computeIfAbsent(endIdx, k -> new ArrayList<>())
            .add(new SegmentConnection(segmentIndex, false));

        // Update connection counts
        NetworkPoint srtPt = points.get(srtIdx);
        NetworkPoint endPt = points.get(endIdx);
        points.set(srtIdx, srtPt.incrementConnections());
        points.set(endIdx, endPt.incrementConnections());
    }

    /**
     * Add a segment using new network point to known existing point in segment.
     * Uses global configuration parameters.
     * @return The index of the newly added point (srtNetPnt)
     */
    public int addSegmentWithDivisions(NetworkPoint srtNetPnt, int endIdx, int level, double maxSegmentLength) {
        return addSegmentWithDivisions(srtNetPnt, endIdx, level, maxSegmentLength, Integer.MAX_VALUE);
    }

    /**
     * Add a segment using new network point to known existing point in segment.
     * @param maxSegments Maximum number of segments to create (for budget-limited subdivision).
     *                    When limited, only the segments closest to endIdx are created.
     * @return The index of the newly added point (srtNetPnt)
     */
    public int addSegmentWithDivisions(NetworkPoint srtNetPnt, int endIdx, int level, double maxSegmentLength, int maxSegments) {
        // Add the start point to get its index
        int srtIdx = addPoint(srtNetPnt);

        // Ensure water flows downhill: if new point is lower than existing,
        // propagate elevation reduction through the connected chain toward level 0
        ensureDownhillFlow(srtIdx, endIdx, level);

        // Call full implementation using global config
        addSegmentWithDivisions(srtIdx, endIdx, level, maxSegmentLength, maxSegments);

        // Return the index of the new point
        return srtIdx;
    }
    
    public void addSegmentWithDivisions(int A, int B, int level, double maxSegmentLength) {
        addSegmentWithDivisions(A, B, level, maxSegmentLength, Integer.MAX_VALUE);
    }

    /**
     * Add a segment with full implementation using global configuration.
     * This is the main implementation that creates multiple connected segments from a single call.
     * @param maxSegments Maximum segments to create. When limited, keeps segments closest to end.
     */
    public void addSegmentWithDivisions(int A, int B, int level, double maxSegmentLength, int maxSegments) {
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

        // Step 1b: Bound tangent magnitudes to maxSegmentLength to prevent excessive curves
        double distance = srt.position.distanceTo(end.position);
        tangentSrt = scaleTangentMagnitude(tangentSrt, distance);
        tangentEnd = scaleTangentMagnitude(tangentEnd, distance);

        // Step 1c: Clamp tangent components near cell boundaries to prevent spline overshoot
        tangentSrt = clampTangentToCellBoundary(tangentSrt, srt.position, distance);
        tangentEnd = clampTangentToCellBoundary(tangentEnd, end.position, distance);

        // Step 2: Subdivide long segments if needed using provided maxSegmentLength
        int numDivisions = (int) Math.ceil(distance / maxSegmentLength);
        // TODO: Allow subdivision at higher levels after confirming basic segment shape is appropriate.
        //if (level>0){
        //    numDivisions = 0;
        //}
        if (numDivisions <= 1) {
            // Single segment - add directly
            addBasicSegment(srtIdx, endIdx, level, tangentSrt, tangentEnd);
        } else {
            // Multiple segments - pass computed tangents and RNG to avoid recomputation
            createSubdividedSegments(srtIdx, endIdx, level, maxSegmentLength, distance, tangentSrt, tangentEnd, rng, maxSegments);
        }
    }

    /**
     * Add a segment with two new network points, only used for trunk initialization.
     * @return The index of endNetPnt (the "far end" for trunk continuation)
     */
    public int addSegmentWithDivisions(NetworkPoint srtNetPnt, NetworkPoint endNetPnt, int level, double maxSegmentLength) {
        // Add end point first to get its index (this is the continuation point for trunk)
        int endIdx = addPoint(endNetPnt);
        // Add start point and create segment(s)
        addSegmentWithDivisions(srtNetPnt, endIdx, level, maxSegmentLength);
        // Return the end point index (the trunk continuation point)
        return endIdx;
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
     * Scale a tangent vector to match the target magnitude.
     * Preserves direction while setting length to targetMagnitude.
     *
     * @param tangent The tangent vector to scale
     * @param targetMagnitude Desired magnitude
     * @return Scaled tangent vector
     */
    private Vec2D scaleTangentMagnitude(Vec2D tangent, double targetMagnitude) {
        if (tangent == null) return null;

        double magnitude = tangent.length();
        if (magnitude < MathUtils.EPSILON) {
            return tangent;
        }

        double scale = targetMagnitude / magnitude;
        return new Vec2D(tangent.x * scale, tangent.y * scale);
    }

    /**
     * Clamp tangent components to prevent Hermite spline from crossing cell boundaries.
     * For a cubic Hermite spline, the maximum overshoot from a tangent is approximately
     * 0.25 * tangent_component * segmentLength * tangentStrength.
     *
     * @param tangent The tangent vector to clamp
     * @param position The point position (to compute distance to cell edges)
     * @param segmentLength Length of the segment
     * @return Clamped tangent vector
     */
    private Vec2D clampTangentToCellBoundary(Vec2D tangent, Point3D position, double segmentLength) {
        if (tangent == null || position == null) return tangent;

        // Get local position within the cell [0, 1) x [0, 1)
        double localX = position.x - Math.floor(position.x);
        double localY = position.y - Math.floor(position.y);

        // Distance to each cell edge
        double distToLeft = localX;
        double distToRight = 1.0 - localX;
        double distToBottom = localY;
        double distToTop = 1.0 - localY;

        // Maximum allowed tangent component = distToEdge / (overshootFactor * segmentLength * tangentStrength)
        // Using 0.25 as conservative overshoot factor for cubic Hermite
        double overshootFactor = 0.25 * segmentLength * config.tangentStrength;
        if (overshootFactor < MathUtils.EPSILON) return tangent;

        double clampedX = tangent.x;
        double clampedY = tangent.y;

        // Clamp positive x component (pointing right)
        if (tangent.x > 0) {
            double maxX = distToRight / overshootFactor;
            clampedX = Math.min(tangent.x, maxX);
        }
        // Clamp negative x component (pointing left)
        else if (tangent.x < 0) {
            double maxNegX = distToLeft / overshootFactor;
            clampedX = Math.max(tangent.x, -maxNegX);
        }

        // Clamp positive y component (pointing up/top)
        if (tangent.y > 0) {
            double maxY = distToTop / overshootFactor;
            clampedY = Math.min(tangent.y, maxY);
        }
        // Clamp negative y component (pointing down/bottom)
        else if (tangent.y < 0) {
            double maxNegY = distToBottom / overshootFactor;
            clampedY = Math.max(tangent.y, -maxNegY);
        }

        // Only create new vector if clamping occurred
        if (clampedX != tangent.x || clampedY != tangent.y) {
            return new Vec2D(clampedX, clampedY);
        }
        return tangent;
    }

    /**
     * Clamp a tangent vector to be within maxAngle of a reference direction.
     * This prevents extreme tangent angles that can cause problematic spline behavior.
     *
     * @param tangent The tangent vector to clamp
     * @param direction The reference direction (typically start-to-end vector)
     * @param maxAngleRadians Maximum allowed angle in radians (e.g., 60 degrees = 1.047 radians)
     * @return Clamped tangent vector
     */
    private Vec2D clampTangentAngle(Vec2D tangent, Vec2D direction, double maxAngleRadians) {
        if (tangent == null || direction == null) {
            return tangent;
        }

        // Normalize both vectors for angle calculation
        Vec2D tangentNorm = tangent.normalize();
        Vec2D directionNorm = direction.normalize();

        // Compute angle between tangent and direction using dot product
        double dotProduct = tangentNorm.x * directionNorm.x + tangentNorm.y * directionNorm.y;
        // Clamp dot product to [-1, 1] to handle floating point errors
        dotProduct = Math.max(-1.0, Math.min(1.0, dotProduct));
        double angle = Math.acos(dotProduct);

        // If angle is within limit, return original tangent
        if (angle <= maxAngleRadians) {
            return tangent;
        }

        // Need to clamp: rotate tangent toward direction
        // Determine rotation direction using cross product (z-component in 2D)
        double cross = tangentNorm.x * directionNorm.y - tangentNorm.y * directionNorm.x;
        double rotationSign = (cross >= 0) ? 1.0 : -1.0;

        // Rotate direction by maxAngle in the appropriate direction
        double clampedAngle = rotationSign * maxAngleRadians;
        Vec2D clampedDirection = rotateVector(directionNorm, clampedAngle);

        // Scale back to original tangent magnitude
        double tangentLength = tangent.length();
        return new Vec2D(clampedDirection.x * tangentLength, clampedDirection.y * tangentLength);
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
        Vec2D segTangent = new Vec2D(0,0);
        double slope = 0;
        // Get the tangent along the segment creation direction.
        if (isStart) {
            segTangent = new Vec2D(point.position.projectZ(), target.position.projectZ());
        }
        else{
            segTangent = new Vec2D( target.position.projectZ(),point.position.projectZ());
        }
        if (segTangent.lengthSquared() < MathUtils.EPSILON) {
            return new Vec2D(1, 0); // Default direction
        }
        segTangent = segTangent.normalize();
        // Adjust based on connection count
        if (point.connections == 0) {
            // Get slope tangent trajectory based on point, also invert the target direction if we are solving for the end condition.
            if (isStart) {
                angle = point.position.getTangentVector();
                slope = Math.abs(point.position.getSlope());
            }
            else {
                Vec2D targetTangent = target.position.getTangentVector();
                angle = (targetTangent != null) ? targetTangent.negate() : null;
                slope = Math.abs(point.position.getSlope());
            }

            // Fallback to segment tangent if slope-based tangent is not available
            if (angle == null) {
                angle = segTangent;
                slope = 0.0;
            }

            // No existing connections - use flow direction with deterministic twist
            double twist = (NoiseGen.nextFloat() * 2.0 * Math.min(slope/config.SlopeWithoutTwist, 1.0) - 1.0) * config.maxTwistAngle;
            Vec2D tangent = rotateVector(angle, twist);

            // Clamp tangent to be within 60 degrees of the target direction toward the end point.
            // This prevents extreme tangent angles that cause problematic spline behavior
            double maxTangentAngle = Math.PI / 3.0; // 60 degrees in radians
            return clampTangentAngle(tangent, segTangent, maxTangentAngle);
        } else {
            Vec2D continuousTangent = getContinuousTangent(pointIdx, isStart);
            if (continuousTangent != null) {
                // One existing connection - use tangent continuity for smooth splines
                if (point.connections == 1) {
                    return continuousTangent;
                }
                // Multiple connections, take random angle between continuous tangent and direction to target
                else {
                        // Pick a random angle between continuousTangent and toTarget
                        double angleContinuous = Math.atan2(continuousTangent.y, continuousTangent.x);
                        double angleTarget = Math.atan2(segTangent.y, segTangent.x);

                        // Calculate the angular difference (taking the shorter path)
                        double angleDiff = angleTarget - angleContinuous;
                        // Normalize to [-PI, PI]
                        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
                        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

                        // Pick a random interpolation factor [0, 1] and interpolate the angle
                        double interpolationFactor = NoiseGen.nextFloat();
                        double resultAngle = angleContinuous + angleDiff * interpolationFactor;

                        // Create vector at the interpolated angle with the magnitude of continuousTangent
                        double magnitude = continuousTangent.length();
                        return new Vec2D(Math.cos(resultAngle) * magnitude, Math.sin(resultAngle) * magnitude);
                    }

                }
            // Fallback - use direction to target with small offset
            // double offset = (NoiseGen.nextFloat() - 0.5) * config.SlopeWithoutTwist * 0.5;
            return segTangent;
        }
    }
    
    /**
     * Get the continuous tangent for C1 spline continuity when connecting to an existing segment.
     * Properly handles all connection cases:
     * - existing END → new START: same direction (forward chain)
     * - existing START → new END: same direction (backward chain)
     * - existing END → new END: negate (meeting point)
     * - existing START → new START: negate (splitting point)
     *
     * @param pointIdx The point index where connection occurs
     * @param isNewSegmentStart True if this is the START of the new segment being created
     * @return The properly oriented tangent for C1 continuity, or null if no connection
     */
    private Vec2D getContinuousTangent(int pointIdx, boolean isNewSegmentStart) {
        List<SegmentConnection> connections = pointToSegments.get(pointIdx);
        if (connections == null || connections.isEmpty()) {
            return null;
        }

        // Use the first connected segment
        SegmentConnection conn = connections.get(0);
        SegmentIdx seg = segments.get(conn.segmentIndex);

        // Get the raw tangent from the existing segment
        Vec2D existingTangent;
        if (conn.isStart) {
            // Point is at existing segment's START
            existingTangent = seg.tangentSrt;
            if (existingTangent == null) {
                // Fallback: direction from start toward end
                System.err.println("Warning: No tangent available for segment start point index " + pointIdx);
                NetworkPoint point = points.get(pointIdx);
                Point3D endPos = seg.getEnd(this);
                existingTangent = new Vec2D(point.position.projectZ(), endPos.projectZ()).normalize();
            }
        } else {
            // Point is at existing segment's END
            existingTangent = seg.tangentEnd;
            if (existingTangent == null) {
                // Fallback: direction from start toward end (same as tangent direction)
                System.err.println("Warning: No tangent available for segment end point index " + pointIdx);
                NetworkPoint point = points.get(pointIdx);
                Point3D srtPos = seg.getSrt(this);
                existingTangent = new Vec2D(srtPos.projectZ(), point.position.projectZ()).normalize();
            }
        }

        // Determine if we need to negate based on connection types:
        // Same-side connections (start-start or end-end): negate for opposite flow
        // Opposite-side connections (start-end or end-start): same direction for continuous flow
        boolean sameType = (conn.isStart == isNewSegmentStart);

        if (sameType) {
            // start→start or end→end: negate for proper flow direction
            return existingTangent.negate();
        } else {
            // end→start or start→end: same direction for continuous flow
            return existingTangent;
        }
    }

    /**
     * Create subdivided segments between two points using spline or linear interpolation.
     * Uses global configuration parameters.
     *
     * Note: Hermite splines use uniform parameter spacing (t), not uniform arc length.
     * Strong tangents or tight curves can cause points to bunch up in certain areas.
     * Points that end up too close together (< 0.5% of original segment length) are skipped.
     *
     * @param srtIdx Start point index
     * @param endIdx End point index
     * @param level Resolution level
     * @param numDivisions Number of divisions to create
     * @param tangentSrt Pre-computed tangent at start point
     * @param tangentEnd Pre-computed tangent at end point
     * @param rng Random number generator for deterministic jitter
     */
    /**
     * @param maxSegments Maximum number of segments to create. When limited, only the last
     *                    maxSegments segments (closest to endIdx) are materialized.
     *                    RNG is still advanced for all divisions to keep determinism.
     */
    private void createSubdividedSegments(int srtIdx, int endIdx, int level, double maxSegmentLength,
                                          double distance, Vec2D tangentSrt, Vec2D tangentEnd, Random rng,
                                          int maxSegments) {

        int numDivisions = (int) Math.ceil(distance / maxSegmentLength);
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);

        // Determine intermediate point type based on endpoint types
        PointType intermediateType = (srt.pointType == PointType.TRUNK && end.pointType == PointType.TRUNK)
            ? PointType.TRUNK
            : PointType.KNOT;

        // Minimum distance threshold: % of intended segment length
        double minDistanceThreshold = maxSegmentLength * 0.2;

        // Pre-compute ALL intermediate positions and tangents (to keep RNG deterministic)
        List<Point3D> interPositions = new ArrayList<>();
        List<Vec2D> interTangents = new ArrayList<>();

        for (int i = 1; i < numDivisions; i++) {
            double t = (double) i / numDivisions;
            double jitterX = rng.nextDouble() - 0.5;
            double jitterY = rng.nextDouble() - 0.5;

            double rawMagnitude = Math.sqrt(jitterX * jitterX + jitterY * jitterY);
            double jitterMagnitude = rawMagnitude / Math.sqrt(0.5);

            double maxJitter = maxSegmentLength * 0.5 * Math.pow(config.jitterReductionBase, level);
            double scaledMagnitude = Math.min(rawMagnitude * maxSegmentLength, maxJitter);

            if (rawMagnitude > MathUtils.EPSILON) {
                jitterX = (jitterX / rawMagnitude) * scaledMagnitude;
                jitterY = (jitterY / rawMagnitude) * scaledMagnitude;
            } else {
                jitterX = 0;
                jitterY = 0;
            }

            Point3D intermediatePoint;
            if (config.useSplines && config.curvature > 0) {
                intermediatePoint = interpolateHermiteSpline(srt.position, end.position,
                                                           tangentSrt, tangentEnd, t, config.tangentStrength, jitterX, jitterY);
            } else {
                intermediatePoint = interpolateLinearWithJitter(srt.position, end.position, t, jitterX, jitterY);
            }

            // Distance check (currently always passes since distanceToSel = MAX_VALUE)
            double distanceToSel = Double.MAX_VALUE;
            if (distanceToSel < minDistanceThreshold) {
                continue;
            }

            Vec2D intermediateTangent;
            if (config.useSplines && config.curvature > 0 && tangentSrt != null && tangentEnd != null) {
                intermediateTangent = computeHermiteTangent(srt.position, end.position,
                                                           tangentSrt, tangentEnd, t, config.tangentStrength);
                intermediateTangent = applyTangentTwist(intermediateTangent, jitterMagnitude,
                                                       config.maxIntermediateTwistAngle, rng);
            } else {
                Vec2D baseDirection = new Vec2D(srt.position.projectZ(), end.position.projectZ()).normalize();
                intermediateTangent = applyTangentTwist(baseDirection, jitterMagnitude,
                                                       config.maxIntermediateTwistAngle, rng);
            }
            intermediateTangent = scaleTangentMagnitude(intermediateTangent,
                ((maxSegmentLength * 8) / Math.pow(config.tangentReductionBase, level)));

            interPositions.add(intermediatePoint);
            interTangents.add(intermediateTangent);
        }

        // Total segments = interPositions.size() + 1 (for the final segment to end)
        int totalSegments = interPositions.size() + 1;

        // Determine which segments to materialize (closest to end = highest indices)
        int startFrom = Math.max(0, totalSegments - maxSegments);

        // Build the chain, only materializing segments from startFrom onward
        int prevIdx;
        Vec2D prevTangent;

        if (startFrom == 0) {
            // Creating all segments from the original start
            prevIdx = srtIdx;
            prevTangent = tangentSrt;
        } else {
            // Skip early segments; start from an intermediate point
            // The point at index (startFrom - 1) becomes the new chain start
            int newStartInterIdx = startFrom - 1;
            prevIdx = addPoint(interPositions.get(newStartInterIdx), intermediateType, level);
            prevTangent = interTangents.get(newStartInterIdx);
        }

        // Create intermediate segments from startFrom onward
        for (int i = startFrom; i < interPositions.size(); i++) {
            int intermediateIdx = addPoint(interPositions.get(i), intermediateType, level);
            addBasicSegment(prevIdx, intermediateIdx, level, prevTangent, interTangents.get(i));
            prevIdx = intermediateIdx;
            prevTangent = interTangents.get(i);
        }

        // Create final segment to end point
        addBasicSegment(prevIdx, endIdx, level, prevTangent, tangentEnd);
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
                                             double t, double tangentStrength, double jitterX, double jitterY) {
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

        return new Point3D(x + jitterX, y + jitterY, z);
    }
    
    /**
     * Linear interpolation with deterministic jitter.
     * References jitter application logic from DendrySampler.
     */
    private Point3D interpolateLinearWithJitter(Point3D srt, Point3D end, double t, double jitterX, double jitterY) {
        // Linear interpolation
        double x = MathUtils.lerp(srt.x, end.x, t);
        double y = MathUtils.lerp(srt.y, end.y, t);
        double z = MathUtils.lerp(srt.z, end.z, t);

        return new Point3D(x + jitterX, y + jitterY, z);
    }

    /**
     * Compute the tangent direction at parameter t along a Hermite spline.
     * Returns the derivative of the spline, which gives the direction at that point.
     */
    private Vec2D computeHermiteTangent(Point3D srt, Point3D end, Vec2D tangentSrt, Vec2D tangentEnd,
                                        double t, double tangentStrength) {
        double segLength = srt.projectZ().distanceTo(end.projectZ());
        double tangentScale = segLength * tangentStrength;

        // Hermite basis function derivatives
        double t2 = t * t;
        double h00_prime = 6 * t2 - 6 * t;
        double h10_prime = 3 * t2 - 4 * t + 1;
        double h01_prime = -6 * t2 + 6 * t;
        double h11_prime = 3 * t2 - 2 * t;

        double dx = h00_prime * srt.x + h10_prime * (tangentSrt != null ? tangentSrt.x * tangentScale : 0)
                  + h01_prime * end.x + h11_prime * (tangentEnd != null ? tangentEnd.x * tangentScale : 0);
        double dy = h00_prime * srt.y + h10_prime * (tangentSrt != null ? tangentSrt.y * tangentScale : 0)
                  + h01_prime * end.y + h11_prime * (tangentEnd != null ? tangentEnd.y * tangentScale : 0);

        return new Vec2D(dx, dy).normalize();
    }

    /**
     * Apply a random twist (rotation) to a tangent vector for intermediate points.
     * The twist amount is scaled based on jitter magnitude - more jitter means less twist.
     * Used with config.maxIntermediateTwistAngle for intermediate subdivision points.
     *
     * @param tangent Base tangent direction
     * @param jitterMagnitude Magnitude of position jitter applied
     * @param maxTwist Maximum twist angle in radians (use config.maxIntermediateTwistAngle)
     * @param rng Random number generator
     * @return Twisted tangent vector
     */
    private Vec2D applyTangentTwist(Vec2D tangent, double jitterMagnitude, double maxTwist, Random rng) {
        // Compute twist reduction factor based on jitter
        // More jitter (more displacement) -> less twist
        // Use exponential decay: twist scales down as jitter increases
        //double jitterRatio = Math.min(1.0, jitterMagnitude / 0.02); // Normalize to typical jitter scale
        //double twistScale = Math.exp(-2.0 * jitterRatio); // Exponential decay
        double twistScale = (1.0 - jitterMagnitude); // Linear decay alternative

        // Random twist angle in range [-maxTwist, +maxTwist], scaled by twist reduction
        double twistAngle = (rng.nextDouble() * 2.0 - 1.0) * maxTwist * twistScale;

        // Apply rotation
        double cos = Math.cos(twistAngle);
        double sin = Math.sin(twistAngle);
        double newX = tangent.x * cos - tangent.y * sin;
        double newY = tangent.x * sin + tangent.y * cos;

        return new Vec2D(newX, newY);
    }

    /**
     * Get an unmodifiable view of all segments.
     */
    public List<SegmentIdx> getSegments() {
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

        // Note: Segments are index-based and automatically reflect updated point positions
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
        for (SegmentIdx s : this.segments) {
            copy.segments.add(s);  // SegmentIdx is immutable
        }
        // Copy point-to-segment mapping
        for (Map.Entry<Integer, List<SegmentConnection>> entry : this.pointToSegments.entrySet()) {
            copy.pointToSegments.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        return copy;
    }

    /**
     * Rollback the SegmentList to a previous state by removing all points and segments
     * added after the saved counts. Used by crossing detection to undo a connection
     * that produced crossing segments.
     *
     * @param savedPointCount Number of points before the addition
     * @param savedSegmentCount Number of segments before the addition
     */
    public void rollback(int savedPointCount, int savedSegmentCount) {
        // Remove segments added after savedSegmentCount and clean up their pointToSegments entries
        for (int i = segments.size() - 1; i >= savedSegmentCount; i--) {
            SegmentIdx seg = segments.get(i);

            // Remove from pointToSegments map
            removeSegmentConnection(seg.srtIdx, i);
            removeSegmentConnection(seg.endIdx, i);

            // Decrement connection counts on endpoints (only if point still exists)
            if (seg.srtIdx < savedPointCount) {
                NetworkPoint srtPt = points.get(seg.srtIdx);
                points.set(seg.srtIdx, srtPt.withConnections(Math.max(0, srtPt.connections - 1)));
            }
            if (seg.endIdx < savedPointCount) {
                NetworkPoint endPt = points.get(seg.endIdx);
                points.set(seg.endIdx, endPt.withConnections(Math.max(0, endPt.connections - 1)));
            }

            segments.remove(i);
        }

        // Remove points added after savedPointCount and clean up their map entries
        for (int i = points.size() - 1; i >= savedPointCount; i--) {
            pointToSegments.remove(i);
            points.remove(i);
        }

        // Reset nextIndex
        nextIndex = savedPointCount;
    }

    /**
     * Remove a specific segment connection entry from a point's connection list.
     */
    private void removeSegmentConnection(int pointIdx, int segmentIndex) {
        List<SegmentConnection> connections = pointToSegments.get(pointIdx);
        if (connections != null) {
            connections.removeIf(conn -> conn.segmentIndex == segmentIndex);
            if (connections.isEmpty()) {
                pointToSegments.remove(pointIdx);
            }
        }
    }

    /**
     * Check if this SegmentList is empty.
     */
    public boolean isEmpty() {
        return points.isEmpty();
    }

    // ========== Conversion Methods ==========

    /**
     * Convert to List<Segment3D> for evaluation.
     * Resolves indices to Point3D positions.
     */
    public List<Segment3D> toSegment3DList() {
        List<Segment3D> result = new ArrayList<>();

        for (SegmentIdx seg : segments) {
            // Resolve indices to Point3D
            NetworkPoint srtPt = points.get(seg.srtIdx);
            NetworkPoint endPt = points.get(seg.endIdx);

            result.add(new Segment3D(
                srtPt.position,
                endPt.position,
                seg.tangentSrt,
                seg.tangentEnd
            ));
        }

        return result;
    }

    @Override
    public String toString() {
        return String.format("SegmentList(points=%d, segments=%d)", points.size(), segments.size());
    }
}

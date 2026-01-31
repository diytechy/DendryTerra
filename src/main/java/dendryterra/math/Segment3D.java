package dendryterra.math;

/**
 * Immutable 3D line segment with elevation, resolution level, and endpoint tangents.
 * Tangents describe the curve direction at each endpoint for spline interpolation.
 * Flow direction: srt (start) -> end (end point).
 */
public final class Segment3D {
    public final Point3D srt;  // Start point of flow
    public final Point3D end;  // End point of flow
    public final int level;    // Resolution level that created this segment (0-5)

    // Tangent vectors at endpoints (for spline interpolation)
    // These are 2D direction vectors in the x,y plane
    public final Vec2D tangentSrt;  // Tangent direction at start point (may be null)
    public final Vec2D tangentEnd;  // Tangent direction at end point (may be null)

    // Endpoint type for debug visualization and tracking point provenance
    public final PointType srtType;
    public final PointType endType;

    /**
     * Create a segment with specified resolution level, tangents, and endpoint types.
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd,
                     PointType srtType, PointType endType) {
        this.srt = srt;
        this.end = end;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        this.srtType = srtType;
        this.endType = endType;
    }

    /**
     * Create a segment with specified resolution level, tangents, and boolean endpoint types.
     * @deprecated Use constructor with PointType instead
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd,
                     boolean srtIsOriginal, boolean endIsOriginal) {
        this(srt, end, level, tangentSrt, tangentEnd,
             srtIsOriginal ? PointType.ORIGINAL : PointType.KNOT,
             endIsOriginal ? PointType.ORIGINAL : PointType.KNOT);
    }

    /**
     * Create a segment with specified resolution level and tangents.
     * Defaults to both endpoints being ORIGINAL type (backward compatible).
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd) {
        this(srt, end, level, tangentSrt, tangentEnd, PointType.ORIGINAL, PointType.ORIGINAL);
    }

    /**
     * Create a segment with specified resolution level (no tangents).
     */
    public Segment3D(Point3D srt, Point3D end, int level) {
        this(srt, end, level, null, null);
    }

    /**
     * Create a segment with default level 1 (backward compatible).
     */
    public Segment3D(Point3D srt, Point3D end) {
        this(srt, end, 1, null, null);
    }

    /**
     * Create a new segment with the same endpoints and level but different tangents.
     */
    public Segment3D withTangents(Vec2D tangentSrt, Vec2D tangentEnd) {
        return new Segment3D(this.srt, this.end, this.level, tangentSrt, tangentEnd,
                             this.srtType, this.endType);
    }

    /**
     * Create a new segment with specified endpoint types.
     */
    public Segment3D withEndpointTypes(PointType srtType, PointType endType) {
        return new Segment3D(this.srt, this.end, this.level, this.tangentSrt, this.tangentEnd,
                             srtType, endType);
    }

    /**
     * Create a new segment with specified endpoint types (boolean version for compatibility).
     * @deprecated Use withEndpointTypes(PointType, PointType) instead
     */
    public Segment3D withEndpointTypes(boolean srtIsOriginal, boolean endIsOriginal) {
        return withEndpointTypes(
            srtIsOriginal ? PointType.ORIGINAL : PointType.KNOT,
            endIsOriginal ? PointType.ORIGINAL : PointType.KNOT);
    }

    /**
     * Check if start point is an original point.
     * @deprecated Use srtType field directly
     */
    public boolean isSrtOriginal() {
        return srtType == PointType.ORIGINAL;
    }

    /**
     * Check if end point is an original point.
     * @deprecated Use endType field directly
     */
    public boolean isEndOriginal() {
        return endType == PointType.ORIGINAL;
    }

    /**
     * Check if this segment has tangent information.
     */
    public boolean hasTangents() {
        return tangentSrt != null || tangentEnd != null;
    }

    public double lengthSquared() {
        return srt.distanceSquaredTo(end);
    }

    public double length() {
        return srt.distanceTo(end);
    }

    public Point3D midpoint() {
        return new Point3D(
            (srt.x + end.x) / 2.0,
            (srt.y + end.y) / 2.0,
            (srt.z + end.z) / 2.0
        );
    }

    /**
     * Interpolate along the segment.
     * @param t parameter from 0 (at srt) to 1 (at end)
     */
    public Point3D lerp(double t) {
        return Point3D.lerp(srt, end, t);
    }

    /**
     * Project to 2D by dropping z coordinates.
     */
    public Segment2D projectZ() {
        return new Segment2D(srt.projectZ(), end.projectZ());
    }

    /**
     * Subdivide this segment into n equal segments using linear interpolation.
     * Note: For B-spline subdivision with jitter, use DendrySampler.subdivideSegment() instead.
     *
     * @param n number of segments to create
     * @return array of n segments
     */
    public Segment3D[] subdivide(int n) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");

        Segment3D[] segments = new Segment3D[n];
        Point3D prev = srt;

        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / n;
            Point3D next = (i == n - 1) ? end : Point3D.lerp(srt, end, t);

            // First segment: srt keeps original type, interior points become KNOT
            // Last segment: end keeps original type
            PointType prevType = (i == 0) ? this.srtType : PointType.KNOT;
            PointType nextType = (i == n - 1) ? this.endType : PointType.KNOT;

            segments[i] = new Segment3D(prev, next, this.level, null, null, prevType, nextType);
            prev = next;
        }

        return segments;
    }

    /**
     * Get n-1 interior points that divide this segment into n equal parts.
     */
    public Point3D[] subdivideInPoints(int n) {
        Point3D[] points = new Point3D[n - 1];
        for (int i = 0; i < n - 1; i++) {
            double t = (double)(i + 1) / n;
            points[i] = Point3D.lerp(srt, end, t);
        }
        return points;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Segment3D(");
        sb.append(srt).append(" -> ").append(end);
        sb.append(", level=").append(level);
        if (hasTangents()) {
            sb.append(", tangentSrt=").append(tangentSrt);
            sb.append(", tangentEnd=").append(tangentEnd);
        }
        sb.append(", srtType=").append(srtType);
        sb.append(", endType=").append(endType);
        sb.append(")");
        return sb.toString();
    }
}

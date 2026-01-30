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

    /**
     * Create a segment with specified resolution level and tangents.
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd) {
        this.srt = srt;
        this.end = end;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
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
        return new Segment3D(this.srt, this.end, this.level, tangentSrt, tangentEnd);
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
     * Subdivide this segment into n equal segments, preserving the level.
     */
    public Segment3D[] subdivide(int n) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");

        Segment3D[] segments = new Segment3D[n];
        Point3D prev = srt;

        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / n;
            Point3D next = (i == n - 1) ? end : Point3D.lerp(srt, end, t);
            segments[i] = new Segment3D(prev, next, this.level);
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
        if (hasTangents()) {
            return "Segment3D(" + srt + " -> " + end + ", level=" + level +
                   ", tangentSrt=" + tangentSrt + ", tangentEnd=" + tangentEnd + ")";
        }
        return "Segment3D(" + srt + " -> " + end + ", level=" + level + ")";
    }
}

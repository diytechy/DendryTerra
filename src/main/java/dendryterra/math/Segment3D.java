package dendryterra.math;

/**
 * Lightweight 3D line segment for evaluation and computation.
 * Contains only the essential fields needed for segment operations.
 * Immutable.
 */
public final class Segment3D {
    public final Point3D srt;  // Start point of the segment
    public final Point3D end;  // End point of the segment
    public final Vec2D tangentSrt;  // Tangent direction at start point (may be null)
    public final Vec2D tangentEnd;  // Tangent direction at end point (may be null)

    /**
     * Create a segment with specified endpoints and tangents.
     */
    public Segment3D(Point3D srt, Point3D end, Vec2D tangentSrt, Vec2D tangentEnd) {
        this.srt = srt;
        this.end = end;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
    }

    /**
     * Create a segment with specified endpoints (no tangents).
     */
    public Segment3D(Point3D srt, Point3D end) {
        this(srt, end, null, null);
    }

    /**
     * Create a new segment with different tangents.
     */
    public Segment3D withTangents(Vec2D tangentSrt, Vec2D tangentEnd) {
        return new Segment3D(this.srt, this.end, tangentSrt, tangentEnd);
    }

    /**
     * Check if this segment has tangent information.
     */
    public boolean hasTangents() {
        return tangentSrt != null || tangentEnd != null;
    }

    /**
     * Get squared length of this segment.
     */
    public double lengthSquared() {
        return srt.distanceSquaredTo(end);
    }

    /**
     * Get length of this segment.
     */
    public double length() {
        return srt.distanceTo(end);
    }

    /**
     * Get midpoint of this segment.
     */
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

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Segment3D(");
        sb.append(srt).append(" -> ").append(end);
        if (hasTangents()) {
            sb.append(", tangentSrt=").append(tangentSrt);
            sb.append(", tangentEnd=").append(tangentEnd);
        }
        sb.append(")");
        return sb.toString();
    }
}

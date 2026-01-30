package dendryterra.math;

/**
 * Immutable 2D line segment.
 * Flow direction: srt (start) -> end (end point).
 */
public final class Segment2D {
    public final Point2D srt;  // Start point of flow
    public final Point2D end;  // End point of flow

    public Segment2D(Point2D srt, Point2D end) {
        this.srt = srt;
        this.end = end;
    }

    public double lengthSquared() {
        return srt.distanceSquaredTo(end);
    }

    public double length() {
        return srt.distanceTo(end);
    }

    public Point2D midpoint() {
        return new Point2D(
            (srt.x + end.x) / 2.0,
            (srt.y + end.y) / 2.0
        );
    }

    /**
     * Interpolate along the segment.
     * @param t parameter from 0 (at srt) to 1 (at end)
     */
    public Point2D lerp(double t) {
        return Point2D.lerp(srt, end, t);
    }

    /**
     * Get the direction vector from srt to end.
     */
    public Vec2D direction() {
        return new Vec2D(srt, end);
    }

    @Override
    public String toString() {
        return "Segment2D(" + srt + " -> " + end + ")";
    }
}

package com.dfsek.terra.addons.dendry.math;

/**
 * Immutable 2D line segment.
 */
public final class Segment2D {
    public final Point2D a;
    public final Point2D b;

    public Segment2D(Point2D a, Point2D b) {
        this.a = a;
        this.b = b;
    }

    public double lengthSquared() {
        return a.distanceSquaredTo(b);
    }

    public double length() {
        return a.distanceTo(b);
    }

    public Point2D midpoint() {
        return new Point2D(
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0
        );
    }

    /**
     * Interpolate along the segment.
     * @param t parameter from 0 (at a) to 1 (at b)
     */
    public Point2D lerp(double t) {
        return Point2D.lerp(a, b, t);
    }

    /**
     * Get the direction vector from a to b.
     */
    public Vec2D direction() {
        return new Vec2D(a, b);
    }

    @Override
    public String toString() {
        return "Segment2D(" + a + " -> " + b + ")";
    }
}

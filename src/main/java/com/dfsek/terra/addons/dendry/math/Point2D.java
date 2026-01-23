package com.dfsek.terra.addons.dendry.math;

/**
 * Immutable 2D point with basic operations.
 */
public final class Point2D {
    public final double x;
    public final double y;

    public Point2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Point2D add(Point2D other) {
        return new Point2D(x + other.x, y + other.y);
    }

    public Point2D add(Vec2D v) {
        return new Point2D(x + v.x, y + v.y);
    }

    public Point2D subtract(Point2D other) {
        return new Point2D(x - other.x, y - other.y);
    }

    public Point2D subtract(Vec2D v) {
        return new Point2D(x - v.x, y - v.y);
    }

    public Point2D scale(double s) {
        return new Point2D(x * s, y * s);
    }

    public Point2D negate() {
        return new Point2D(-x, -y);
    }

    public double distanceSquaredTo(Point2D other) {
        double dx = x - other.x;
        double dy = y - other.y;
        return dx * dx + dy * dy;
    }

    public double distanceTo(Point2D other) {
        return Math.sqrt(distanceSquaredTo(other));
    }

    public static Point2D lerp(Point2D a, Point2D b, double t) {
        return new Point2D(
            MathUtils.lerp(a.x, b.x, t),
            MathUtils.lerp(a.y, b.y, t)
        );
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Point2D other)) return false;
        return Math.abs(x - other.x) < MathUtils.EPSILON
            && Math.abs(y - other.y) < MathUtils.EPSILON;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(x) * 31 + Double.hashCode(y);
    }

    @Override
    public String toString() {
        return "Point2D(" + x + ", " + y + ")";
    }
}

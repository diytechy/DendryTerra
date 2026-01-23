package com.dfsek.terra.addons.dendry.math;

/**
 * Immutable 3D line segment with elevation.
 */
public final class Segment3D {
    public final Point3D a;
    public final Point3D b;

    public Segment3D(Point3D a, Point3D b) {
        this.a = a;
        this.b = b;
    }

    public double lengthSquared() {
        return a.distanceSquaredTo(b);
    }

    public double length() {
        return a.distanceTo(b);
    }

    public Point3D midpoint() {
        return new Point3D(
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0,
            (a.z + b.z) / 2.0
        );
    }

    /**
     * Interpolate along the segment.
     * @param t parameter from 0 (at a) to 1 (at b)
     */
    public Point3D lerp(double t) {
        return Point3D.lerp(a, b, t);
    }

    /**
     * Project to 2D by dropping z coordinates.
     */
    public Segment2D projectZ() {
        return new Segment2D(a.projectZ(), b.projectZ());
    }

    /**
     * Subdivide this segment into n equal segments.
     */
    public Segment3D[] subdivide(int n) {
        if (n <= 0) throw new IllegalArgumentException("n must be positive");

        Segment3D[] segments = new Segment3D[n];
        Point3D prev = a;

        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / n;
            Point3D next = (i == n - 1) ? b : Point3D.lerp(a, b, t);
            segments[i] = new Segment3D(prev, next);
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
            points[i] = Point3D.lerp(a, b, t);
        }
        return points;
    }

    @Override
    public String toString() {
        return "Segment3D(" + a + " -> " + b + ")";
    }
}

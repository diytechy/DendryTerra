package com.dfsek.terra.addons.dendry.math;

/**
 * Catmull-Rom spline interpolation using chordal parameterization.
 * Used for smooth subdivision of segments in Dendry noise generation.
 */
public final class CatmullRomSpline {

    private CatmullRomSpline() {}

    /**
     * Evaluate Catmull-Rom spline at parameter t.
     * The spline passes through p1 and p2, with p0 and p3 as control points.
     *
     * @param p0 Control point before the segment
     * @param p1 Start of the segment
     * @param p2 End of the segment
     * @param p3 Control point after the segment
     * @param t  Parameter value (t1 to t2 range based on chordal distance)
     * @return Interpolated point
     */
    public static Point2D evaluate(Point2D p0, Point2D p1, Point2D p2, Point2D p3, double t) {
        double t0 = 0.0;
        double t1 = p0.distanceTo(p1) + t0;
        double t2 = p1.distanceTo(p2) + t1;
        double t3 = p2.distanceTo(p3) + t2;

        // Handle degenerate cases
        if (t1 <= t0 + MathUtils.EPSILON || t2 <= t1 + MathUtils.EPSILON || t3 <= t2 + MathUtils.EPSILON) {
            return Point2D.lerp(p1, p2, MathUtils.clamp((t - t1) / Math.max(t2 - t1, MathUtils.EPSILON), 0, 1));
        }

        Point2D a1 = p0.scale((t1 - t) / (t1 - t0)).add(p1.scale((t - t0) / (t1 - t0)));
        Point2D a2 = p1.scale((t2 - t) / (t2 - t1)).add(p2.scale((t - t1) / (t2 - t1)));
        Point2D a3 = p2.scale((t3 - t) / (t3 - t2)).add(p3.scale((t - t2) / (t3 - t2)));

        Point2D b1 = a1.scale((t2 - t) / (t2 - t0)).add(a2.scale((t - t0) / (t2 - t0)));
        Point2D b2 = a2.scale((t3 - t) / (t3 - t1)).add(a3.scale((t - t1) / (t3 - t1)));

        return b1.scale((t2 - t) / (t2 - t1)).add(b2.scale((t - t1) / (t2 - t1)));
    }

    /**
     * Evaluate Catmull-Rom spline at parameter t for 3D points.
     */
    public static Point3D evaluate(Point3D p0, Point3D p1, Point3D p2, Point3D p3, double t) {
        double t0 = 0.0;
        double t1 = p0.distanceTo(p1) + t0;
        double t2 = p1.distanceTo(p2) + t1;
        double t3 = p2.distanceTo(p3) + t2;

        // Handle degenerate cases
        if (t1 <= t0 + MathUtils.EPSILON || t2 <= t1 + MathUtils.EPSILON || t3 <= t2 + MathUtils.EPSILON) {
            return Point3D.lerp(p1, p2, MathUtils.clamp((t - t1) / Math.max(t2 - t1, MathUtils.EPSILON), 0, 1));
        }

        Point3D a1 = p0.scale((t1 - t) / (t1 - t0)).add(p1.scale((t - t0) / (t1 - t0)));
        Point3D a2 = p1.scale((t2 - t) / (t2 - t1)).add(p2.scale((t - t1) / (t2 - t1)));
        Point3D a3 = p2.scale((t3 - t) / (t3 - t2)).add(p3.scale((t - t2) / (t3 - t2)));

        Point3D b1 = a1.scale((t2 - t) / (t2 - t0)).add(a2.scale((t - t0) / (t2 - t0)));
        Point3D b2 = a2.scale((t3 - t) / (t3 - t1)).add(a3.scale((t - t1) / (t3 - t1)));

        return b1.scale((t2 - t) / (t2 - t1)).add(b2.scale((t - t1) / (t2 - t1)));
    }

    /**
     * Subdivide the spline segment between p1 and p2 into n intermediate points.
     *
     * @param p0 Control point before the segment
     * @param p1 Start of the segment
     * @param p2 End of the segment
     * @param p3 Control point after the segment
     * @param x  Normalized parameter [0, 1] to evaluate between p1 and p2
     * @return Interpolated point
     */
    public static Point3D subdivide(Point3D p0, Point3D p1, Point3D p2, Point3D p3, double x) {
        double t0 = 0.0;
        double t1 = p0.distanceTo(p1) + t0;
        double t2 = p1.distanceTo(p2) + t1;
        double t3 = p2.distanceTo(p3) + t2;

        // Handle degenerate cases
        if (t1 <= t0 + MathUtils.EPSILON || t2 <= t1 + MathUtils.EPSILON || t3 <= t2 + MathUtils.EPSILON) {
            return Point3D.lerp(p1, p2, x);
        }

        // Evaluate the spline between p1 and p2
        double t = MathUtils.lerp(t1, t2, x);

        Point3D a1 = p0.scale((t1 - t) / (t1 - t0)).add(p1.scale((t - t0) / (t1 - t0)));
        Point3D a2 = p1.scale((t2 - t) / (t2 - t1)).add(p2.scale((t - t1) / (t2 - t1)));
        Point3D a3 = p2.scale((t3 - t) / (t3 - t2)).add(p3.scale((t - t2) / (t3 - t2)));

        Point3D b1 = a1.scale((t2 - t) / (t2 - t0)).add(a2.scale((t - t0) / (t2 - t0)));
        Point3D b2 = a2.scale((t3 - t) / (t3 - t1)).add(a3.scale((t - t1) / (t3 - t1)));

        return b1.scale((t2 - t) / (t2 - t1)).add(b2.scale((t - t1) / (t2 - t1)));
    }

    /**
     * Subdivide the segment between p1 and p2 into n intermediate points using spline interpolation.
     *
     * @param p0 Control point before the segment
     * @param p1 Start of the segment
     * @param p2 End of the segment
     * @param p3 Control point after the segment
     * @param n  Number of intermediate points to generate
     * @return Array of n intermediate points
     */
    public static Point3D[] subdivideToPoints(Point3D p0, Point3D p1, Point3D p2, Point3D p3, int n) {
        Point3D[] points = new Point3D[n];
        for (int i = 0; i < n; i++) {
            double x = (double)(i + 1) / (n + 1);
            points[i] = subdivide(p0, p1, p2, p3, x);
        }
        return points;
    }

    /**
     * Subdivide a segment into multiple segments using spline interpolation.
     *
     * @param p0          Control point before the segment
     * @param segmentStart Start of the segment (p1)
     * @param segmentEnd   End of the segment (p2)
     * @param p3          Control point after the segment
     * @param numSegments Number of segments to create
     * @return Array of segments
     */
    public static Segment3D[] subdivideToSegments(Point3D p0, Point3D segmentStart, Point3D segmentEnd, Point3D p3, int numSegments) {
        if (numSegments <= 0) {
            throw new IllegalArgumentException("numSegments must be positive");
        }

        Segment3D[] segments = new Segment3D[numSegments];

        if (numSegments == 1) {
            segments[0] = new Segment3D(segmentStart, segmentEnd);
            return segments;
        }

        Point3D prev = segmentStart;
        for (int i = 0; i < numSegments; i++) {
            Point3D next;
            if (i == numSegments - 1) {
                next = segmentEnd;
            } else {
                double x = (double)(i + 1) / numSegments;
                next = subdivide(p0, segmentStart, segmentEnd, p3, x);
            }
            segments[i] = new Segment3D(prev, next);
            prev = next;
        }

        return segments;
    }
}

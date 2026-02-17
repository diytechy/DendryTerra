package dendryterra.math;

import com.dfsek.seismic.math.numericanalysis.interpolation.InterpolationFunctions;
import com.dfsek.seismic.math.floatingpoint.FloatingPointFunctions;

/**
 * Mathematical utility functions for Dendry noise generation.
 * Uses Seismic library functions where available for optimization.
 */
public final class MathUtils {

    public static final double EPSILON = 1e-9;

    private MathUtils() {}

    /**
     * Linear interpolation between a and b.
     * Delegates to Seismic's optimized implementation.
     */
    public static double lerp(double a, double b, double t) {
        return InterpolationFunctions.lerp(a, b, t);
    }

    /**
     * Fast floor operation using Seismic.
     */
    public static int floor(double x) {
        return FloatingPointFunctions.floor(x);
    }

    /**
     * Clamp value to [min, max] range.
     */
    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    /**
     * Project point p onto line defined by points a and b.
     * Returns parameter u where the projection point is a + u*(b-a).
     * u < 0 means projection is before a, u > 1 means after b.
     */
    public static double pointLineProjection(Point2D p, Point2D a, Point2D b) {
        Vec2D ap = new Vec2D(a, p);
        Vec2D ab = new Vec2D(a, b);

        double abLenSq = ab.lengthSquared();
        if (abLenSq <= EPSILON) {
            return 0.0;
        }

        return ap.dot(ab) / abLenSq;
    }

    /**
     * Calculate distance from point p to line segment [a, b].
     * Returns the closest point on the segment.
     */
    public static DistanceResult distanceToLineSegment(Point2D p, Point2D a, Point2D b) {
        Vec2D ab = new Vec2D(a, b);
        double u = pointLineProjection(p, a, b);

        if (u < 0.0) {
            return new DistanceResult(p.distanceTo(a), a, 0.0);
        }
        if (u > 1.0) {
            return new DistanceResult(p.distanceTo(b), b, 1.0);
        }

        Point2D closest = a.add(ab.scale(u));
        return new DistanceResult(p.distanceTo(closest), closest, u);
    }

    /**
     * Calculate distance from point p to a 3D segment (projected to 2D).
     */
    public static DistanceResult distanceToLineSegment(Point2D p, Segment3D seg) {
        return distanceToLineSegment(p, seg.srt.projectZ(), seg.end.projectZ());
    }

    /**
     * Result of a distance calculation, including the closest point.
     */
    public static final class DistanceResult {
        public final double distance;
        public final Point2D closestPoint;
        public final double parameter;

        public DistanceResult(double distance, Point2D closestPoint, double parameter) {
            this.distance = distance;
            this.closestPoint = closestPoint;
            this.parameter = parameter;
        }
    }
}

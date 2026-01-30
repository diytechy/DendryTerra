package dendryterra.math;

/**
 * Immutable 3D point with basic operations.
 * In Dendry context: x,y are horizontal position, z is elevation.
 * Optional slope properties (slopeX, slopeY) represent elevation gradient.
 */
public final class Point3D {
    public final double x;
    public final double y;
    public final double z;

    /** Slope component in x direction (dz/dx). NaN if not computed. */
    public final double slopeX;
    /** Slope component in y direction (dz/dy). NaN if not computed. */
    public final double slopeY;

    public Point3D(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.slopeX = Double.NaN;
        this.slopeY = Double.NaN;
    }

    public Point3D(double x, double y, double z, double slopeX, double slopeY) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.slopeX = slopeX;
        this.slopeY = slopeY;
    }

    public Point3D(Point2D point, double z) {
        this.x = point.x;
        this.y = point.y;
        this.z = z;
        this.slopeX = Double.NaN;
        this.slopeY = Double.NaN;
    }

    /**
     * Create a new Point3D with slope values.
     */
    public Point3D withSlopes(double slopeX, double slopeY) {
        return new Point3D(this.x, this.y, this.z, slopeX, slopeY);
    }

    /**
     * Check if this point has computed slope values.
     */
    public boolean hasSlope() {
        return !Double.isNaN(slopeX) && !Double.isNaN(slopeY);
    }

    /**
     * Get the tangent angle (direction of steepest descent) in radians.
     * Calculated as the angle of the negative gradient vector.
     * Returns NaN if slopes are not computed.
     */
    public double getTangent() {
        if (!hasSlope()) return Double.NaN;
        // Gradient points uphill; tangent for flow is downhill (negative gradient)
        return Math.atan2(-slopeY, -slopeX);
    }

    /**
     * Get the slope magnitude along the tangent direction.
     * This is the steepness in the direction of steepest descent.
     * Returns NaN if slopes are not computed.
     */
    public double getSlope() {
        if (!hasSlope()) return Double.NaN;
        return Math.sqrt(slopeX * slopeX + slopeY * slopeY);
    }

    public Point3D add(Point3D other) {
        return new Point3D(x + other.x, y + other.y, z + other.z);
    }

    public Point3D subtract(Point3D other) {
        return new Point3D(x - other.x, y - other.y, z - other.z);
    }

    public Point3D scale(double s) {
        return new Point3D(x * s, y * s, z * s);
    }

    public Point3D negate() {
        return new Point3D(-x, -y, -z);
    }

    public double distanceSquaredTo(Point3D other) {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    public double distanceTo(Point3D other) {
        return Math.sqrt(distanceSquaredTo(other));
    }

    /**
     * Project to 2D by dropping the z coordinate.
     */
    public Point2D projectZ() {
        return new Point2D(x, y);
    }

    public static Point3D lerp(Point3D a, Point3D b, double t) {
        return new Point3D(
            MathUtils.lerp(a.x, b.x, t),
            MathUtils.lerp(a.y, b.y, t),
            MathUtils.lerp(a.z, b.z, t)
        );
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Point3D other)) return false;
        return Math.abs(x - other.x) < MathUtils.EPSILON
            && Math.abs(y - other.y) < MathUtils.EPSILON
            && Math.abs(z - other.z) < MathUtils.EPSILON;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(x) * 961 + Double.hashCode(y) * 31 + Double.hashCode(z);
    }

    @Override
    public String toString() {
        return "Point3D(" + x + ", " + y + ", " + z + ")";
    }
}

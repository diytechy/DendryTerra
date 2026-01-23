package dendryterra.math;

/**
 * Immutable 2D vector with direction operations.
 */
public final class Vec2D {
    public final double x;
    public final double y;

    public Vec2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Create a vector from point a to point b.
     */
    public Vec2D(Point2D a, Point2D b) {
        this.x = b.x - a.x;
        this.y = b.y - a.y;
    }

    public Vec2D add(Vec2D other) {
        return new Vec2D(x + other.x, y + other.y);
    }

    public Vec2D subtract(Vec2D other) {
        return new Vec2D(x - other.x, y - other.y);
    }

    public Vec2D scale(double s) {
        return new Vec2D(x * s, y * s);
    }

    public Vec2D negate() {
        return new Vec2D(-x, -y);
    }

    public double lengthSquared() {
        return x * x + y * y;
    }

    public double length() {
        return Math.sqrt(lengthSquared());
    }

    public double dot(Vec2D other) {
        return x * other.x + y * other.y;
    }

    /**
     * 2D cross product (returns scalar).
     */
    public double cross(Vec2D other) {
        return x * other.y - y * other.x;
    }

    public Vec2D normalize() {
        double len = length();
        if (len < MathUtils.EPSILON) {
            return new Vec2D(0, 0);
        }
        return new Vec2D(x / len, y / len);
    }

    /**
     * Rotate 90 degrees counter-clockwise.
     */
    public Vec2D rotateCCW90() {
        return new Vec2D(-y, x);
    }

    /**
     * Rotate 90 degrees clockwise.
     */
    public Vec2D rotateCW90() {
        return new Vec2D(y, -x);
    }

    /**
     * Calculate angle between this vector and another.
     */
    public double angleTo(Vec2D other) {
        double denom = Math.sqrt(lengthSquared() * other.lengthSquared());
        if (denom < MathUtils.EPSILON) return 0;
        return Math.acos(MathUtils.clamp(dot(other) / denom, -1.0, 1.0));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Vec2D other)) return false;
        return Math.abs(x - other.x) < MathUtils.EPSILON
            && Math.abs(y - other.y) < MathUtils.EPSILON;
    }

    @Override
    public int hashCode() {
        return Double.hashCode(x) * 31 + Double.hashCode(y);
    }

    @Override
    public String toString() {
        return "Vec2D(" + x + ", " + y + ")";
    }
}

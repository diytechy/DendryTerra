package dendryterra.math;

/**
 * A point in the network with connectivity metadata.
 * Immutable - use builder methods to create modified copies.
 *
 * Replaces the old NetworkNode class with a cleaner design where:
 * - Points carry their own metadata (connections count, type, level)
 * - No mutable state for graph traversal
 * - Index is unique within a SegmentList
 * - Tangent information is stored on segments, not points
 */
public final class NetworkPoint {
    /** 3D position of this point */
    public final Point3D position;

    /** Unique index within the SegmentList (resets to 0 per coordinate query) */
    public final int index;

    /** Type of this point (ORIGINAL, TRUNK, KNOT, LEAF, EDGE) */
    public final PointType pointType;

    /** Level at which this point was created (0 for asterisms, 1+ for higher levels) */
    public final int level;

    /** Number of segments connected to this point (0-3+) */
    public final int connections;

    /**
     * Create a new NetworkPoint with default connections (0).
     */
    public NetworkPoint(Point3D position, int index, PointType pointType, int level) {
        this(position, index, pointType, level, 0);
    }

    /**
     * Full constructor with all fields.
     */
    public NetworkPoint(Point3D position, int index, PointType pointType,
                        int level, int connections) {
        this.position = position;
        this.index = index;
        this.pointType = pointType;
        this.level = level;
        this.connections = connections;
    }

    /**
     * Create a copy with a new position.
     */
    public NetworkPoint withPosition(Point3D newPos) {
        return new NetworkPoint(newPos, index, pointType, level, connections);
    }

    /**
     * Create a copy with a new point type.
     */
    public NetworkPoint withPointType(PointType type) {
        return new NetworkPoint(position, index, type, level, connections);
    }

    /**
     * Create a copy with a new connection count.
     */
    public NetworkPoint withConnections(int count) {
        return new NetworkPoint(position, index, pointType, level, count);
    }

    /**
     * Create a copy with connection count incremented by 1.
     */
    public NetworkPoint incrementConnections() {
        return withConnections(connections + 1);
    }

    /**
     * Create a copy with a new index (used when transferring between lists).
     */
    public NetworkPoint withIndex(int newIndex) {
        return new NetworkPoint(position, newIndex, pointType, level, connections);
    }

    /**
     * Check if this is a branch point (has 2+ connections, meaning flow passes through).
     */
    public boolean isBranchPoint() {
        return connections >= 2;
    }

    /**
     * Check if this is a leaf point (has exactly 1 connection, end of a branch).
     */
    public boolean isLeaf() {
        return connections == 1;
    }

    @Override
    public String toString() {
        return String.format("NetworkPoint(idx=%d, pos=%s, type=%s, level=%d, conn=%d)",
                             index, position, pointType, level, connections);
    }
}

package dendryterra.math;

/**
 * Enum representing the type/origin of a point in the network.
 * Used for debug visualization and tracking point provenance.
 */
public enum PointType {

    /**
     * Point was created as a part of segment subdivision (knot point).
     */
    KNOT(0),

    /**
     * Point was part of the original trunk creation (main flow path).
     */
    TRUNK(2),

    /**
     * Point exists at the end of a branch on its level.
     */
    LEAF(3),
    /**
     * Point was originally created as a star or initial network point.
     */
    ORIGINAL(4),

    /**
     * Point was created by clipping a segment at a cell boundary.
     * Edge points should not be connected to by subsequent levels.
     */
    EDGE(5);

    private final int value;

    PointType(int value) {
        this.value = value;
    }

    /**
     * Get the numeric value for this point type.
     * Used for pixel cache storage and debug visualization.
     */
    public int getValue() {
        return value;
    }

    /**
     * Get PointType from numeric value.
     */
    public static PointType fromValue(int value) {
        for (PointType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        return ORIGINAL; // Default fallback
    }
}

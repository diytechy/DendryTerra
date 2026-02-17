package dendryterra;

/**
 * Defines the tileable shape for constellations.
 * Each shape tiles the plane differently and affects how constellations
 * are positioned and which neighbors need to be computed.
 */
public enum ConstellationShape {
    /**
     * Square tiling - simplest form, 4 neighbors per constellation.
     * Inscribed square width = 3 * ConstellationScale gridspaces.
     */
    SQUARE,

    /**
     * Perfect hexagon tiling - 6 neighbors per constellation.
     * Provides more uniform distance to neighbors.
     */
    HEXAGON,

    /**
     * Rhombus (diamond) tiling - 4 neighbors per constellation.
     * Rotated square pattern.
     */
    RHOMBUS
}

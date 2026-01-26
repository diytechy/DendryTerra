package dendryterra;

/**
 * Enum representing the different return value types for the Dendry sampler.
 */
public enum DendryReturnType {
    /**
     * Returns the Euclidean distance to the closest branch.
     */
    DISTANCE,

    /**
     * Returns the weighted distance to the "closest" branch,
     * where distance is weighted according to branch level.
     * Higher resolution levels have less weight.
     */
    WEIGHTED,

    /**
     * Returns the elevation difference from the root branch.
     */
    ELEVATION,

    /**
     * Returns the elevation from cached pixel data.
     * Requires cachepixels > 0 to be set.
     * Uses pre-computed pixel grid for faster lookups after initial cell computation.
     */
    PIXEL_ELEVATION,

    /**
     * Returns the resolution level from cached pixel data.
     * Requires cachepixels > 0 to be set.
     * Uses pre-computed pixel grid for faster lookups after initial cell computation.
     */
    PIXEL_LEVEL
}

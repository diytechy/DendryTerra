package com.dfsek.terra.addons.dendry;

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
    ELEVATION
}

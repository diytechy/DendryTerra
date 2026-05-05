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
    PIXEL_LEVEL,

    /**
     * Debug mode for visualizing point types in cached pixel data.
     * Requires cachepixels > 0 to be set.
     * Returns:
     *   3 - Within 2 pixels of an original star/point
     *   2 - Within 2 pixels of a subdivision point
     *   1 - On a segment
     *  -1 - Empty (no data)
     */
    PIXEL_DEBUG,

    /**
     * Legacy river detection mode using cached pixel data.
     * Requires cachepixels > 0 to be set.
     * Uses riverwidth and borderwidth samplers to determine thresholds.
     * River width per level = riverwidth * (0.6^level), minimum 2x pixel resolution.
     * Returns:
     *   0 - Within river (distance to segment < river width for that level)
     *   1 - Within border (distance to segment < river width + border width)
     *   2 - Outside river and border
     */
    PIXEL_RIVER_LEGACY,

    /**
     * Optimized river detection mode using chunked caching system.
     * Each chunk is 256x256 blocks, where each block stores normalized elevation and distance.
     * Uses bigchunk cache (20 MB) and segment list cache (20 MB) for performance.
     * Returns de-quantized distance as a double value.
     */
    PIXEL_RIVER,

    /**
     * Optimized elevation/control function mode using chunked caching system.
     * Uses the same bigchunk cache as PIXEL_RIVER but returns elevation instead of distance.
     * Elevation is quantized to UInt8 based on the 'max' parameter (0 = 0.0, 255 = max).
     * Returns de-quantized elevation as a double value.
     */
    PIXEL_RIVER_CTRL,

    /**
     * Reports approximate far distance to the nearest river segment centerline in world units.
     * Uses a 4x4 directional distance cache per BigChunk (reduced from 8x8 for better fidelity).
     * The return value is quantized to multiples of cachepixels (distU8 * cachepixels).
     * Returns maxDist when no segment is within range.
     * Also accessible via PIXEL_RIVER with y in (-2, 0).
     */
    PIXEL_RIVER_FAR,

    /**
     * Reports far distance to the nearest river segment with normal-based offset compensation.
     * Uses the same 4x4 distance+normal cache per BigChunk for sub-cell accuracy by projecting
     * the query offset from the cell center onto the stored perpendicular normal direction.
     * Also accessible via PIXEL_RIVER with y in [-3, -2).
     */
    PIXEL_RIVER_FAR2,

    /**
     * Reports approximate far distance normalized to the same [-1, 1] scale as PIXEL_RIVER.
     * Uses the 4x4 far cache distance and approximates river width by sampling the
     * river-width sampler at the query location (level 0, no falloff applied).
     * Returns -1 at the estimated river center, 0 at the estimated flow edge,
     * and values &gt; 1 for distances beyond the border-width from the flow edge.
     * Also accessible via PIXEL_RIVER with y &lt;= -3.
     */
    PIXEL_RIVER_FAR_NORM
}

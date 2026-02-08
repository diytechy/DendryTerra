package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import com.dfsek.tectonic.api.config.template.ValidatedConfigTemplate;
import com.dfsek.tectonic.api.config.template.annotations.Default;
import com.dfsek.tectonic.api.config.template.annotations.Value;
import com.dfsek.tectonic.api.config.template.object.ObjectTemplate;
import com.dfsek.tectonic.api.exception.ValidationException;

import com.dfsek.terra.api.config.meta.Meta;

/**
 * Configuration template for the Dendry noise sampler.
 */
public class DendryTemplate implements ValidatedConfigTemplate, ObjectTemplate<Sampler> {

    @Value("n")
    @Default
    private @Meta int n = 2;

    @Value("epsilon")
    @Default
    private @Meta double epsilon = 0.0;

    @Value("delta")
    @Default
    private @Meta double delta = 0.05;

    @Value("slope")
    @Default
    private @Meta double slope = 0.005;

    /**
     * Grid cell size in world units. Replaces the old 'frequency' parameter.
     * A gridsize of 1000 means each base grid cell covers 1000x1000 blocks.
     */
    @Value("gridsize")
    @Default
    private @Meta double gridsize = 1000.0;

    @Value("return")
    @Default
    private @Meta DendryReturnType returnType = DendryReturnType.ELEVATION;

    /**
     * Control sampler for base elevation.
     */
    @Value("sampler")
    @Default
    private @Meta Sampler controlSampler = null;

    @Value("salt")
    @Default
    private @Meta long salt = 0;

    /**
     * Optional sampler to determine branch count per cell.
     * Queried at cell center; output clamped to [1, 8].
     */
    @Value("branches")
    @Default
    private @Meta Sampler branchesSampler = null;

    /**
     * Default branch count when no branches sampler is provided.
     */
    @Value("default-branches")
    @Default
    private @Meta int defaultBranches = 2;

    /**
     * Curvature factor for Catmull-Rom spline subdivision.
     * 0 = linear interpolation, 1 = full spline curvature.
     */
    @Value("curvature")
    @Default
    private @Meta double curvature = 0.5;

    /**
     * Curvature falloff per level. Each level's curvature is multiplied by this.
     * Lower values = less curvature at finer detail levels.
     */
    @Value("curvature-falloff")
    @Default
    private @Meta double curvatureFalloff = 0.7;

    /**
     * Maximum distance for sub-segment connection.
     * 0 = auto-calculate based on grid size.
     */
    @Value("connect-distance")
    @Default
    private @Meta double connectDistance = 0;

    /**
     * Multiplier for auto-calculated connect distance.
     * connectDistance = cellSize * connectDistanceFactor
     */
    @Value("connect-distance-factor")
    @Default
    private @Meta double connectDistanceFactor = 2.0;

    // ========== Performance Tuning Flags ==========

    /**
     * Enable LRU caching of cell data.
     * Disable to test performance without caching overhead.
     */
    @Value("use-cache")
    @Default
    private @Meta boolean useCache = false;

    /**
     * Enable parallel stream processing for large segment lists.
     * Disable to test sequential processing performance.
     */
    @Value("use-parallel")
    @Default
    private @Meta boolean useParallel = true;

    /**
     * Enable Catmull-Rom spline subdivision.
     * Disable to use simple linear subdivision (faster).
     */
    @Value("use-splines")
    @Default
    private @Meta boolean useSplines = true;

    /**
     * Enable debug timing output to console.
     * Shows per-phase execution times for profiling.
     */
    @Value("debug-timing")
    @Default
    private @Meta boolean debugTiming = false;

    /**
     * Threshold for parallel processing (segment count).
     * Only use parallel streams when segment count exceeds this.
     */
    @Value("parallel-threshold")
    @Default
    private @Meta int parallelThreshold = 100;

    /**
     * Scale factor for constellations.
     * ConstellationScale of 1 means the largest inscribed square (without tilting) is 3 gridspaces wide.
     * This ensures when tiled, only the 4 closest constellations need to be solved.
     * Range: 1-10, default 1.
     */
    @Value("constellation-scale")
    @Default
    private @Meta int ConstellationScale = 1;

    /**
     * Shape of constellation tiling pattern.
     * SQUARE: Standard square grid tiling
     * HEXAGON: Hexagonal tiling for more uniform neighbor distances
     * RHOMBUS: Diamond/rotated square tiling
     */
    @Value("constellation-shape")
    @Default
    private @Meta ConstellationShape constellationShape = ConstellationShape.SQUARE;

    /**
     * Maximum angle deviation for spline tangents at nodes (in degrees).
     * 0 = tangents point directly at connected nodes (linear-ish curves)
     * 90 = tangents can be perpendicular to the connection direction (maximum curvature)
     * Range: 0-90, default 45.
     */
    @Value("tangent-angle")
    @Default
    private @Meta double tangentAngle = 45.0;

    /**
     * Strength of spline tangents as a fraction of segment length.
     * Controls how far the control point is from the node.
     * 0 = control point at node (linear), 1 = control point at full tangent length.
     * Higher values create more pronounced curves but risk overlap.
     * Range: 0-1, default 0.4.
     */
    @Value("tangent-strength")
    @Default
    private @Meta double tangentStrength = 1.0;

    /**
     * Pixel cache resolution for faster repeated queries.
     * When > 0, caches segment data as a pixel grid for each cell.
     * Range: 0 (disabled) to gridsize.
     * Lower values = higher resolution cache = more memory per cell.
     * Use with PIXEL_ELEVATION or PIXEL_LEVEL return types.
     */
    @Value("cachepixels")
    @Default
    private @Meta double cachepixels = 0;

    /**
     * Slope threshold for fully aligning tangents with the gradient.
     * When the terrain slope is at or above this value, point tangents will
     * align fully with the downhill direction without variance.
     * Range: 0-1, default 0.1.
     */
    @Value("slope-when-straight")
    @Default
    private @Meta double slopeWhenStraight = 0.1;

    /**
     * Minimum slope cutoff for point rejection.
     * Points with slope below this threshold may be rejected or adjusted.
     * Positive values indicate minimum downward grade required for flow.
     * If flow would go uphill but slope is still > 0, lower level segments
     * will be reduced in elevation to maintain downhill flow.
     * Default: 0.01.
     */
    @Value("lowest-slope-cutoff")
    @Default
    private @Meta double lowestSlopeCutoff = -0.01;

    /**
     * Debug level for segment visualization.
     * 0 = normal operation (default)
     * 5 = stars only for first constellation
     * 6 = stars only for all constellations
     * 10 = first constellation segments only
     * 15 = all constellations before stitching (trunk only)
     * 20 = all constellations before stitching
     * 30 = all constellations including stitching
     * 40 = level 1+ points as 0-length segments (distribution check)
     */
    @Value("debug")
    @Default
    private @Meta int debug = 0;

    /**
     * Minimum elevation for level 0 (constellation) points.
     * At level 0, all candidate points are assigned this elevation instead of
     * the control function value. This flattens the elevation landscape at
     * level 0, changing path preferences from "downhill flow" to distance-based.
     * Default: 0 (disabled - uses control function elevation).
     * When set, level 0 points use this value as their elevation.
     */
    @Value("minimum")
    @Default
    private @Meta double minimum = 0;

    @Override
    public boolean validate() throws ValidationException {
        if (n < 0 || n > 5) {
            throw new ValidationException("n must be between 0 and 5, got: " + n);
        }
        if (epsilon < 0 || epsilon >= 0.5) {
            throw new ValidationException("epsilon must be in range [0, 0.5), got: " + epsilon);
        }
        if (delta < 0) {
            throw new ValidationException("delta must be non-negative, got: " + delta);
        }
        if (gridsize <= 0) {
            throw new ValidationException("gridsize must be positive, got: " + gridsize);
        }
        if (defaultBranches < 1 || defaultBranches > 8) {
            throw new ValidationException("default-branches must be between 1 and 8, got: " + defaultBranches);
        }
        if (curvature < 0 || curvature > 1) {
            throw new ValidationException("curvature must be in range [0, 1], got: " + curvature);
        }
        if (curvatureFalloff < 0 || curvatureFalloff > 1) {
            throw new ValidationException("curvature-falloff must be in range [0, 1], got: " + curvatureFalloff);
        }
        if (connectDistance < 0) {
            throw new ValidationException("connect-distance must be non-negative, got: " + connectDistance);
        }
        if (connectDistanceFactor <= 0) {
            throw new ValidationException("connect-distance-factor must be positive, got: " + connectDistanceFactor);
        }
        if (ConstellationScale < 1 || ConstellationScale > 10) {
            throw new ValidationException("constellation-scale must be between 1 and 10, got: " + ConstellationScale);
        }
        if (tangentAngle < 0 || tangentAngle > 90) {
            throw new ValidationException("tangent-angle must be between 0 and 90, got: " + tangentAngle);
        }
        if (tangentStrength < 0 || tangentStrength > 1) {
            throw new ValidationException("tangent-strength must be between 0 and 1, got: " + tangentStrength);
        }
        if (cachepixels < 0) {
            throw new ValidationException("cachepixels must be non-negative, got: " + cachepixels);
        }
        if (cachepixels > gridsize) {
            throw new ValidationException("cachepixels must not exceed gridsize, got: " + cachepixels + " > " + gridsize);
        }
        if (cachepixels > 0 && gridsize / cachepixels > 65535) {
            throw new ValidationException("gridsize/cachepixels exceeds UInt16 max (65535), got: " + (gridsize / cachepixels));
        }
        if (slopeWhenStraight < 0 || slopeWhenStraight > 1) {
            throw new ValidationException("slope-when-straight must be in range [0, 1], got: " + slopeWhenStraight);
        }
        return true;
    }

    @Override
    public Sampler get() {
        return new DendrySampler(
            n, epsilon, delta, slope, gridsize,
            returnType, controlSampler, salt,
            branchesSampler, defaultBranches,
            curvature, curvatureFalloff,
            connectDistance, connectDistanceFactor,
            useCache, useParallel, useSplines,
            debugTiming, parallelThreshold,
            ConstellationScale, constellationShape,
            Math.toRadians(tangentAngle), tangentStrength,
            cachepixels,
            slopeWhenStraight, lowestSlopeCutoff,
            debug, minimum
        );
    }
}

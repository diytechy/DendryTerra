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
     * Duplicate branch suppression level for level 0 network.
     * Controls how far to look for duplicate paths between nodes.
     * Level 0 grid size = 5 + 2 * this value (e.g., 1 = 7x7, 2 = 9x9).
     * Higher values = more thorough duplicate removal but slower.
     */
    @Value("duplicate-branch-suppression")
    @Default
    private @Meta int duplicateBranchSuppression = 1;

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
        if (duplicateBranchSuppression < 0 || duplicateBranchSuppression > 4) {
            throw new ValidationException("duplicate-branch-suppression must be between 0 and 4, got: " + duplicateBranchSuppression);
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
            duplicateBranchSuppression
        );
    }
}

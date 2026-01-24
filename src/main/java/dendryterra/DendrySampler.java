package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import dendryterra.math.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Based on the algorithm described in Gaillard et al. (I3D 2019).
 *
 * <h2>Algorithm Overview</h2>
 * This sampler generates natural branching patterns (river networks, valley systems,
 * root structures) by creating hierarchical segment networks across multiple resolution
 * levels. Each level adds finer detail while maintaining connectivity to parent branches.
 *
 * <h2>Coordinate System</h2>
 * <pre>
 * World Coordinates (x, z)     Large integers (e.g., 1000, 2000) - actual block positions
 *         |
 *         v  multiply by frequency (default 0.001)
 * Noise Coordinates            Small floats (e.g., 1.0, 2.0) - normalized noise space
 *         |
 *         v  floor(x * resolution)
 * Grid Cell Coordinates        Discrete integers identifying which cell contains the point
 * </pre>
 *
 * <h2>Why Scale by Frequency?</h2>
 * <ul>
 *   <li>World coordinates are large integers (block positions: 0 to thousands)</li>
 *   <li>Noise algorithms work best in normalized space (typically small values)</li>
 *   <li>frequency = 0.001 means 1000 blocks = 1 unit in noise space</li>
 *   <li>This controls the "scale" of branching patterns in the world</li>
 * </ul>
 */
public class DendrySampler implements Sampler {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendrySampler.class);

    // Configuration parameters
    private final int resolution;           // Number of resolution levels (1-5)
    private final double epsilon;           // Point bias within cells [0, 0.5)
    private final double delta;             // Base displacement amount for organic curves
    private final double slope;             // Valley/cliff slope factor for elevation
    private final double frequency;         // World-to-noise coordinate scaling factor
    private final DendryReturnType returnType;  // What value to return (DISTANCE, WEIGHTED, ELEVATION)
    private final Sampler controlSampler;   // Optional sampler for base elevation control
    private final long salt;                // Seed modifier for deterministic randomness

    // Algorithm constants
    private static final int SUBDIVISIONS = 4;   // Number of times to subdivide each segment
    private static final int CACHE_SIZE = 128;   // Dimensions of the point cache (128x128)

    /**
     * Point cache for performance optimization.
     *
     * <h3>Cache Strategy</h3>
     * Pre-computing points for commonly accessed cells avoids repeated RNG calls.
     * The cache covers cell coordinates from -64 to +63 in both dimensions.
     *
     * <h3>Cache Indexing</h3>
     * <pre>
     * Cell (-64, -64) -> pointCache[0][0]
     * Cell (0, 0)     -> pointCache[64][64]
     * Cell (63, 63)   -> pointCache[127][127]
     * </pre>
     *
     * Points outside this range are computed on-demand (cache miss).
     */
    private final Point2D[][] pointCache;

    public DendrySampler(int resolution, double epsilon, double delta,
                         double slope, double frequency,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt) {
        this.resolution = resolution;
        this.epsilon = epsilon;
        this.delta = delta;
        this.slope = slope;
        this.frequency = frequency;
        this.returnType = returnType;
        this.controlSampler = controlSampler;
        this.salt = salt;

        // Initialize point cache at construction time
        // This is a one-time cost that speeds up all subsequent samples
        this.pointCache = new Point2D[CACHE_SIZE][CACHE_SIZE];
        initPointCache();
    }

    /**
     * Populates the point cache with pre-computed points.
     *
     * Called once during sampler construction. Each cell in the [-64, 64) range
     * gets a deterministic point based on its coordinates and the salt value.
     */
    private void initPointCache() {
        int halfCache = CACHE_SIZE / 2;  // 64
        for (int x = -halfCache; x < halfCache; x++) {
            for (int y = -halfCache; y < halfCache; y++) {
                // Store at positive array indices: cell -64 -> index 0, cell 0 -> index 64
                pointCache[x + halfCache][y + halfCache] = generatePoint(x, y);
            }
        }
    }

    /**
     * 2D sampling entry point called by Terra.
     *
     * <h3>Coordinate Scaling Rationale</h3>
     * World coordinates (x, z) are multiplied by frequency to transform them
     * into noise space. This allows the branching pattern scale to be controlled:
     * <ul>
     *   <li>Lower frequency (0.0001) = larger patterns, fewer branches per area</li>
     *   <li>Higher frequency (0.01) = smaller patterns, more branches per area</li>
     * </ul>
     *
     * @param seed World seed (combined with salt for determinism)
     * @param x World X coordinate (block position)
     * @param z World Z coordinate (block position)
     * @return Sample value based on returnType (distance, weighted, or elevation)
     */
    @Override
    public double getSample(long seed, double x, double z) {
        // SCALING: Convert world coordinates to noise space
        // Example: x=1000, frequency=0.001 -> scaledX=1.0
        return evaluate(seed, x * frequency, z * frequency);
    }

    /**
     * 3D sampling entry point called by Terra.
     *
     * For terrain generation, we use the X-Z plane (horizontal), ignoring Y (vertical).
     * The algorithm produces a 2D pattern that defines terrain features.
     *
     * @param seed World seed
     * @param x World X coordinate
     * @param y World Y coordinate (ignored - vertical axis)
     * @param z World Z coordinate
     * @return Sample value based on returnType
     */
    @Override
    public double getSample(long seed, double x, double y, double z) {
        // Y is ignored - we only use the horizontal plane for branching patterns
        return evaluate(seed, x * frequency, z * frequency);
    }

    /**
     * Main algorithm entry point - processes the hierarchical branching at all resolution levels.
     *
     * <h3>Algorithm Flow</h3>
     * <ol>
     *   <li>Level 1: Generate base grid (9x9 points), create segments, subdivide, displace</li>
     *   <li>Level 2-5: Generate finer grids (5x5 points), connect to parent segments</li>
     *   <li>Early exit if resolution setting is reached</li>
     *   <li>Compute final result based on distance to nearest segment</li>
     * </ol>
     *
     * <h3>Displacement Scaling</h3>
     * Each level has progressively smaller displacement to maintain smooth transitions:
     * <pre>
     * Level 1: delta
     * Level 2: delta / 4
     * Level 3: delta / 16
     * </pre>
     *
     * @param seed World seed (unused in current implementation - salt used instead)
     * @param x Scaled X coordinate (already in noise space)
     * @param y Scaled Y coordinate (already in noise space, represents Z in world)
     * @return Computed sample value
     */
    private double evaluate(long seed, double x, double y) {
        // Displacement factors decrease exponentially for each level
        // This creates natural-looking transitions between detail levels
        double displacementLevel1 = delta;           // Full displacement at coarsest level
        double displacementLevel2 = displacementLevel1 / 4.0;   // 1/4 of base
        double displacementLevel3 = displacementLevel2 / 4.0;   // 1/16 of base

        // Minimum slope constraints ensure sub-branches flow "downhill" toward parent branches
        // Higher values = steeper required slope = branches connect more steeply
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // ═══════════════════════════════════════════════════════════════════════
        // LEVEL 1: Base Resolution (coarsest detail)
        // ═══════════════════════════════════════════════════════════════════════

        // Determine which grid cell contains the query point at resolution 1
        // getCell maps continuous coordinates to discrete cell indices
        Cell cell1 = getCell(x, y, 1);

        // Generate a 9x9 grid of points centered on the cell
        // Uses cache for cells in [-64, 64) range, generates on-demand otherwise
        Point3D[][] points1 = generateNeighboringPoints3D(cell1, 9);

        // Create segments by connecting each point to its lowest-elevation neighbor
        // This naturally creates drainage/flow patterns toward low areas
        List<Segment3D> segments1 = generateSegments(points1);

        // Subdivide each segment into SUBDIVISIONS (4) smaller segments
        // This smooths the branch paths for more natural appearance
        segments1 = subdivideSegments(segments1, SUBDIVISIONS);

        // Add organic randomness by displacing segment midpoints perpendicular to direction
        // SKIP CONDITION: If delta ≈ 0, no displacement is applied
        displaceSegments(segments1, displacementLevel1, cell1);

        // EARLY EXIT: If user only wants resolution level 1, return now
        if (resolution == 1) {
            return computeResult(x, y, segments1, 1);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // LEVEL 2: 2x Resolution (finer detail)
        // ═══════════════════════════════════════════════════════════════════════

        // Cell size is halved (resolution doubles) for finer grid
        Cell cell2 = getCell(x, y, 2);

        // Smaller grid (5x5) because we're adding detail, not replacing
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);

        // Connect level 2 points to level 1 segments (parent branches)
        // SKIP CONDITION: Points farther than 10.0 units from any segment are skipped
        List<Segment3D> segments2 = generateSubSegments(points2, segments1, minSlopeLevel2);

        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            // Combine all segments for final distance calculation
            List<Segment3D> allSegments = new ArrayList<>(segments1);
            allSegments.addAll(segments2);
            return computeResult(x, y, allSegments, 2);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // LEVEL 3: 4x Resolution
        // ═══════════════════════════════════════════════════════════════════════
        Cell cell3 = getCell(x, y, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);

        // Accumulate all previous segments for connectivity
        List<Segment3D> allPreviousSegments = new ArrayList<>(segments1);
        allPreviousSegments.addAll(segments2);

        List<Segment3D> segments3 = generateSubSegments(points3, allPreviousSegments, minSlopeLevel3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allPreviousSegments.addAll(segments3);
            return computeResult(x, y, allPreviousSegments, 3);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // LEVEL 4: 8x Resolution
        // ═══════════════════════════════════════════════════════════════════════
        Cell cell4 = getCell(x, y, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allPreviousSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allPreviousSegments, minSlopeLevel4);
        // Note: No displacement at level 4 (displacement becomes negligible)

        if (resolution == 4) {
            allPreviousSegments.addAll(segments4);
            return computeResult(x, y, allPreviousSegments, 4);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // LEVEL 5: 16x Resolution (finest detail)
        // ═══════════════════════════════════════════════════════════════════════
        Cell cell5 = getCell(x, y, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allPreviousSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allPreviousSegments, minSlopeLevel5);
        allPreviousSegments.addAll(segments5);

        return computeResult(x, y, allPreviousSegments, 5);
    }

    /**
     * Represents a discrete grid cell at a specific resolution level.
     *
     * <h3>Cell Coordinates</h3>
     * Cell coordinates are computed by: floor(noiseCoord * resolution)
     * <pre>
     * Resolution 1: cell(1,2) covers noise coords [1.0-2.0) x [2.0-3.0)
     * Resolution 2: cell(2,4) covers noise coords [1.0-1.5) x [2.0-2.5)
     * Resolution 4: cell(4,8) covers noise coords [1.0-1.25) x [2.0-2.25)
     * </pre>
     */
    private static class Cell {
        final int x;          // Cell X index
        final int y;          // Cell Y index
        final int resolution; // Resolution multiplier (1, 2, 4, 8, or 16)

        Cell(int x, int y, int resolution) {
            this.x = x;
            this.y = y;
            this.resolution = resolution;
        }
    }

    /**
     * Maps continuous noise coordinates to a discrete grid cell.
     *
     * <h3>Scaling Rationale</h3>
     * Multiplying by resolution before flooring effectively subdivides the space:
     * <ul>
     *   <li>Resolution 1: 1 cell per unit of noise space</li>
     *   <li>Resolution 2: 2 cells per unit (cells are half the size)</li>
     *   <li>Resolution 4: 4 cells per unit (cells are quarter the size)</li>
     * </ul>
     *
     * @param x Noise-space X coordinate
     * @param y Noise-space Y coordinate
     * @param resolution Resolution multiplier
     * @return Cell containing the point
     */
    private Cell getCell(double x, double y, int resolution) {
        // SCALING: Multiply coordinates by resolution to get finer grid at higher levels
        int cellX = (int) Math.floor(x * resolution);
        int cellY = (int) Math.floor(y * resolution);
        return new Cell(cellX, cellY, resolution);
    }

    /**
     * Generates a deterministic point within a grid cell.
     *
     * <h3>Point Placement</h3>
     * Points are placed randomly within the cell, but biased away from edges by epsilon.
     * <pre>
     * epsilon = 0.0: point can be anywhere in [0, 1)
     * epsilon = 0.1: point is in [0.1, 0.9) - avoids edges
     * epsilon = 0.4: point is in [0.4, 0.6) - clustered near center
     * </pre>
     *
     * @param cellX Cell X index
     * @param cellY Cell Y index
     * @return Point2D in absolute coordinates (cellX + offset, cellY + offset)
     */
    private Point2D generatePoint(int cellX, int cellY) {
        // Create deterministic RNG based on cell coordinates
        Random rng = initRandomGenerator(cellX, cellY, 1);

        // Generate position within cell, biased by epsilon
        // Range: [epsilon, 1.0 - epsilon)
        double px = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        double py = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);

        // Return absolute position (cell origin + offset)
        return new Point2D(cellX + px, cellY + py);
    }

    /**
     * Retrieves a point from cache if available, otherwise generates it.
     *
     * <h3>Cache Hit Condition</h3>
     * Point is cached if: cellX in [-64, 64) AND cellY in [-64, 64)
     *
     * <h3>Performance Impact</h3>
     * <ul>
     *   <li>Cache hit: O(1) array lookup</li>
     *   <li>Cache miss: O(1) RNG computation (but slower than lookup)</li>
     * </ul>
     *
     * @param cellX Cell X index
     * @param cellY Cell Y index
     * @return Cached or newly generated point
     */
    private Point2D generatePointCached(int cellX, int cellY) {
        int halfCache = CACHE_SIZE / 2;  // 64

        // CACHE HIT CONDITION: Both coordinates within cached range
        if (cellX >= -halfCache && cellX < halfCache && cellY >= -halfCache && cellY < halfCache) {
            return pointCache[cellX + halfCache][cellY + halfCache];
        }

        // CACHE MISS: Generate point on demand
        return generatePoint(cellX, cellY);
    }

    /**
     * Creates a deterministic random number generator for a specific cell.
     *
     * <h3>Seed Computation</h3>
     * The seed combines cell coordinates, level, and user-provided salt to ensure:
     * <ul>
     *   <li>Same cell always produces same random values</li>
     *   <li>Different cells produce different values</li>
     *   <li>Different salt values produce entirely different patterns</li>
     * </ul>
     *
     * @param x Cell X coordinate
     * @param y Cell Y coordinate
     * @param level Resolution level
     * @return Seeded Random instance
     */
    private Random initRandomGenerator(int x, int y, int level) {
        // Combine coordinates with primes to reduce correlation
        // The mask ensures positive seed values
        long seed = (541L * x + 79L * y + level * 1000L + salt) & 0x7FFFFFFFL;
        return new Random(seed);
    }

    /**
     * Generates a grid of 3D points (x, y, elevation) around a cell.
     *
     * <h3>Grid Layout</h3>
     * For a 9x9 grid centered on cell (cx, cy):
     * <pre>
     * [cx-4,cy-4] [cx-3,cy-4] ... [cx+4,cy-4]
     * [cx-4,cy-3] [cx-3,cy-3] ... [cx+4,cy-3]
     *     ...         ...     ...     ...
     * [cx-4,cy+4] [cx-3,cy+4] ... [cx+4,cy+4]
     * </pre>
     *
     * <h3>Scaling by Resolution</h3>
     * Points are divided by resolution to convert from cell-space to noise-space:
     * <pre>
     * Resolution 1: point at cell (5, 10) -> noise coords (5.x, 10.y)
     * Resolution 2: point at cell (5, 10) -> noise coords (2.5x, 5.0y)
     * </pre>
     * This ensures finer resolution levels have proportionally smaller features.
     *
     * @param cell Center cell
     * @param size Grid dimension (9 for level 1, 5 for levels 2-5)
     * @return 2D array of Point3D with position and elevation
     */
    private Point3D[][] generateNeighboringPoints3D(Cell cell, int size) {
        Point3D[][] points = new Point3D[size][size];
        int half = size / 2;  // Center offset

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Calculate cell coordinates relative to center
                int px = cell.x + j - half;
                int py = cell.y + i - half;

                // Get cached or generated 2D point
                Point2D p2d = generatePointCached(px, py);

                // SCALING: Divide by resolution to map cell-space to noise-space
                // Higher resolution = smaller coordinate values = finer detail
                Point2D scaled = new Point2D(p2d.x / cell.resolution, p2d.y / cell.resolution);

                // Get elevation from control function (or default gradient)
                double elevation = evaluateControlFunction(scaled.x, scaled.y);

                points[i][j] = new Point3D(scaled.x, scaled.y, elevation);
            }
        }
        return points;
    }

    /**
     * Evaluates the control function to get base elevation at a point.
     *
     * <h3>Scaling Back to World Coordinates</h3>
     * The control sampler expects world coordinates, but we have noise coordinates.
     * We divide by frequency to convert back:
     * <pre>
     * noiseCoord = 1.0, frequency = 0.001
     * worldCoord = 1.0 / 0.001 = 1000
     * </pre>
     *
     * @param x Noise-space X coordinate
     * @param y Noise-space Y coordinate
     * @return Elevation value
     */
    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            // SCALING: Convert noise coordinates back to world coordinates
            // for the control sampler which operates in world space
            return controlSampler.getSample(salt, x / frequency, y / frequency);
        }
        // Default: simple gradient (creates downhill flow toward negative X)
        return x * 0.1;
    }

    /**
     * Creates segments connecting each point to its lowest-elevation neighbor.
     *
     * <h3>Branch Formation</h3>
     * This creates natural drainage patterns where "water" flows downhill.
     * Each point connects to whichever neighbor has the lowest elevation,
     * forming tree-like branching structures.
     *
     * <h3>Skip Condition</h3>
     * Segments are only created if the current point differs from the lowest
     * neighbor (avoids zero-length segments at local minima).
     *
     * @param points Grid of 3D points with elevations
     * @return List of segments forming the branch network
     */
    private List<Segment3D> generateSegments(Point3D[][] points) {
        List<Segment3D> segments = new ArrayList<>();
        int size = points.length;

        // Skip edge points (need neighbors on all sides)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Point3D current = points[i][j];

                // Find lowest neighbor in 3x3 neighborhood
                double lowestElevation = Double.MAX_VALUE;
                int lowestI = i;
                int lowestJ = j;

                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        Point3D neighbor = points[i + di][j + dj];
                        if (neighbor.z < lowestElevation) {
                            lowestElevation = neighbor.z;
                            lowestI = i + di;
                            lowestJ = j + dj;
                        }
                    }
                }

                Point3D lowest = points[lowestI][lowestJ];

                // SKIP CONDITION: Don't create zero-length segments
                // (occurs when current point is the local minimum)
                if (current.distanceSquaredTo(lowest) > MathUtils.EPSILON) {
                    segments.add(new Segment3D(current, lowest));
                }
            }
        }
        return segments;
    }

    /**
     * Subdivides segments into smaller pieces for smoother paths.
     *
     * <h3>Subdivision Process</h3>
     * Each segment is split into numDivisions equal parts:
     * <pre>
     * Original: A ────────────────────── B
     * Subdivided (4): A ─── p1 ─── p2 ─── p3 ─── B
     * </pre>
     *
     * Note: This is linear interpolation. A full implementation would use
     * Catmull-Rom splines with predecessor/successor segments for C1 continuity.
     *
     * @param segments Original segments
     * @param numDivisions Number of subdivisions per segment
     * @return New list with subdivided segments
     */
    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions) {
        List<Segment3D> subdivided = new ArrayList<>();

        for (Segment3D seg : segments) {
            Point3D prev = seg.a;
            for (int i = 1; i <= numDivisions; i++) {
                double t = (double) i / numDivisions;
                // Linear interpolation (could be replaced with Catmull-Rom)
                Point3D next = (i == numDivisions) ? seg.b : Point3D.lerp(seg.a, seg.b, t);
                subdivided.add(new Segment3D(prev, next));
                prev = next;
            }
        }
        return subdivided;
    }

    /**
     * Displaces segment midpoints perpendicular to their direction for organic appearance.
     *
     * <h3>Skip Condition</h3>
     * If displacementFactor ≈ 0 (less than EPSILON), no displacement is applied.
     * This allows users to disable displacement by setting delta=0.
     *
     * <h3>Displacement Direction</h3>
     * Displacement is perpendicular to the segment direction, creating S-curves:
     * <pre>
     * Before: A ────────── B
     * After:  A ────╱╲──── B  (midpoint displaced sideways)
     * </pre>
     *
     * @param segments Segments to displace (modified in place)
     * @param displacementFactor Maximum displacement distance
     * @param cell Current cell (used for RNG seeding)
     */
    private void displaceSegments(List<Segment3D> segments, double displacementFactor, Cell cell) {
        // SKIP CONDITION: No displacement if factor is negligible
        if (displacementFactor < MathUtils.EPSILON) return;

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            // Get perpendicular direction (rotate segment direction 90 degrees)
            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            Vec2D perp = dir.rotateCCW90().normalize();

            // Generate random displacement amount in range [-factor, +factor]
            Random rng = initRandomGenerator((int)(seg.a.x * 100), (int)(seg.a.y * 100), cell.resolution);
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor;

            // Displace the midpoint perpendicular to segment direction
            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z  // Elevation unchanged
            );

            // Replace segment with one ending at displaced midpoint
            segments.set(i, new Segment3D(seg.a, displacedMid));

            // Adjust next segment to start from displaced midpoint if connected
            if (i + 1 < segments.size()) {
                Segment3D next = segments.get(i + 1);
                if (next.a.equals(seg.b)) {
                    segments.set(i + 1, new Segment3D(displacedMid, next.b));
                }
            }
        }
    }

    /**
     * Generates sub-segments that connect finer-level points to parent-level segments.
     *
     * <h3>Connection Strategy</h3>
     * Each point at the finer level finds the nearest segment from the parent level
     * and creates a connection to it. This ensures hierarchical connectivity.
     *
     * <h3>Skip Conditions</h3>
     * <ul>
     *   <li>Points farther than 10.0 units from any parent segment are skipped</li>
     *   <li>Points at distance ≈ 0 (already on a segment) are skipped</li>
     * </ul>
     *
     * <h3>Elevation Constraint</h3>
     * Sub-branch elevation = max(controlElevation, parentElevation + slope * distance)
     * This ensures water flows "downhill" toward the parent branch.
     *
     * @param points Grid of points at the finer resolution level
     * @param parentSegments Segments from all coarser levels
     * @param minSlope Minimum slope constraint
     * @return New segments connecting fine points to parent segments
     */
    private List<Segment3D> generateSubSegments(Point3D[][] points, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> subSegments = new ArrayList<>();
        int size = points.length;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Point3D point = points[i][j];

                // Find nearest segment in parent level
                NearestSegmentResult nearest = findNearestSegment(point.projectZ(), parentSegments);

                // SKIP CONDITION: Point too far from any parent segment
                if (nearest == null || nearest.distance > 10.0) continue;

                // Calculate elevation with slope constraint
                // Ensures "tributaries" flow downhill toward main branches
                double elevationFromControl = evaluateControlFunction(point.x, point.y);
                double elevationWithSlope = nearest.closestPoint.z + minSlope * nearest.distance;
                double elevation = Math.max(elevationWithSlope, elevationFromControl);

                Point3D adjustedPoint = new Point3D(point.x, point.y, elevation);

                // SKIP CONDITION: Already on the segment (no connection needed)
                if (nearest.distance > MathUtils.EPSILON) {
                    Point3D connectionPoint = new Point3D(
                        nearest.closestPoint2D.x,
                        nearest.closestPoint2D.y,
                        nearest.closestPoint.z
                    );
                    subSegments.add(new Segment3D(adjustedPoint, connectionPoint));
                }
            }
        }
        return subSegments;
    }

    /**
     * Result of finding the nearest segment to a query point.
     */
    private static class NearestSegmentResult {
        final double distance;           // Euclidean distance to nearest point on segment
        final Point2D closestPoint2D;    // Closest point on segment (2D)
        final Point3D closestPoint;      // Closest point with interpolated elevation
        final Segment3D segment;         // The nearest segment

        NearestSegmentResult(double distance, Point2D closestPoint2D, Point3D closestPoint, Segment3D segment) {
            this.distance = distance;
            this.closestPoint2D = closestPoint2D;
            this.closestPoint = closestPoint;
            this.segment = segment;
        }
    }

    /**
     * Finds the nearest segment to a query point.
     *
     * <h3>Skip Condition</h3>
     * Returns null if the segment list is empty.
     *
     * <h3>Performance</h3>
     * O(n) where n = total number of segments across all levels.
     * For large resolution values, this could be optimized with spatial indexing.
     *
     * @param point Query point (2D)
     * @param segments List of segments to search
     * @return NearestSegmentResult or null if no segments
     */
    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        // SKIP CONDITION: No segments to search
        if (segments.isEmpty()) return null;

        double minDist = Double.MAX_VALUE;
        Point2D closestPoint2D = null;
        Point3D closestPoint3D = null;
        Segment3D nearestSeg = null;

        for (Segment3D seg : segments) {
            // Find closest point on this segment
            MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);

            if (result.distance < minDist) {
                minDist = result.distance;
                closestPoint2D = result.closestPoint;

                // Interpolate Z (elevation) along the segment using parameter t
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                closestPoint3D = new Point3D(result.closestPoint.x, result.closestPoint.y, z);
                nearestSeg = seg;
            }
        }

        return new NearestSegmentResult(minDist, closestPoint2D, closestPoint3D, nearestSeg);
    }

    /**
     * Computes the final output value based on the return type setting.
     *
     * <h3>Return Types</h3>
     * <table>
     *   <tr><td>DISTANCE</td><td>Raw Euclidean distance to nearest segment</td></tr>
     *   <tr><td>WEIGHTED</td><td>Distance weighted by 1/level (finer levels have less weight)</td></tr>
     *   <tr><td>ELEVATION</td><td>Base elevation + (distance * slope) for valley profiles</td></tr>
     * </table>
     *
     * <h3>Fallback</h3>
     * If no segments are found, returns the control function value (or default gradient).
     *
     * @param x Query point X (noise space)
     * @param y Query point Y (noise space)
     * @param segments All segments from all processed levels
     * @param level Current resolution level (1-5)
     * @return Computed sample value
     */
    private double computeResult(double x, double y, List<Segment3D> segments, int level) {
        Point2D point = new Point2D(x, y);
        NearestSegmentResult nearest = findNearestSegment(point, segments);

        // FALLBACK: No segments found, use control function
        if (nearest == null) {
            return evaluateControlFunction(x, y);
        }

        switch (returnType) {
            case DISTANCE:
                // Raw distance - useful for creating hard-edged features
                return nearest.distance;

            case WEIGHTED:
                // Distance weighted by level - higher levels contribute less
                // Useful for smooth blending between detail levels
                return nearest.distance * (1.0 / level);

            case ELEVATION:
            default:
                // Valley/terrain profile:
                // Base elevation at branch + slope contribution from distance
                // Creates V-shaped or U-shaped valley profiles depending on slope
                double baseElevation = nearest.closestPoint.z;
                double slopeContribution = nearest.distance * slope;
                return baseElevation + slopeContribution;
        }
    }
}

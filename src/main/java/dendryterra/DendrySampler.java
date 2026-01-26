package dendryterra;

import com.dfsek.seismic.type.sampler.Sampler;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import dendryterra.math.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Dendry noise sampler implementing hierarchical multi-resolution branching.
 * Optimized with configurable caching, parallel processing, and debug timing.
 */
public class DendrySampler implements Sampler {
    private static final Logger LOGGER = LoggerFactory.getLogger(DendrySampler.class);

    // Configuration parameters
    private final int resolution;
    private final double epsilon;
    private final double delta;
    private final double slope;
    private final double gridsize;
    private final DendryReturnType returnType;
    private final Sampler controlSampler;
    private final long salt;

    // Branch and curvature parameters
    private final Sampler branchesSampler;
    private final int defaultBranches;
    private final double curvature;
    private final double curvatureFalloff;
    private final double connectDistance;
    private final double connectDistanceFactor;

    // Performance flags
    private final boolean useCache;
    private final boolean useParallel;
    private final boolean useSplines;
    private final boolean debugTiming;
    private final int parallelThreshold;

    // Duplicate branch suppression (controls level 0 grid size and path pruning)
    private final int duplicateBranchSuppression;

    // Cache configuration
    private static final int MAX_CACHE_SIZE = 16384;

    // Lazy LRU cache (optional based on useCache flag)
    private final LoadingCache<Long, CellData> cellCache;

    // Timing statistics (only used when debugTiming is true)
    private final AtomicLong sampleCount = new AtomicLong(0);
    private final AtomicLong totalTimeNs = new AtomicLong(0);
    private volatile long lastLogTime = 0;
    private static final long LOG_INTERVAL_MS = 5000; // Log every 5 seconds

    private static class CellData {
        final Point2D point;
        final int branchCount;

        CellData(Point2D point, int branchCount) {
            this.point = point;
            this.branchCount = branchCount;
        }
    }

    public DendrySampler(int resolution, double epsilon, double delta,
                         double slope, double gridsize,
                         DendryReturnType returnType,
                         Sampler controlSampler, long salt,
                         Sampler branchesSampler, int defaultBranches,
                         double curvature, double curvatureFalloff,
                         double connectDistance, double connectDistanceFactor,
                         boolean useCache, boolean useParallel, boolean useSplines,
                         boolean debugTiming, int parallelThreshold,
                         int duplicateBranchSuppression) {
        this.resolution = resolution;
        this.epsilon = epsilon;
        this.delta = delta;
        this.slope = slope;
        this.gridsize = gridsize;
        this.returnType = returnType;
        this.controlSampler = controlSampler;
        this.salt = salt;
        this.branchesSampler = branchesSampler;
        this.defaultBranches = defaultBranches;
        this.curvature = curvature;
        this.curvatureFalloff = curvatureFalloff;
        this.connectDistance = connectDistance;
        this.connectDistanceFactor = connectDistanceFactor;
        this.useCache = useCache;
        this.useParallel = useParallel;
        this.useSplines = useSplines;
        this.debugTiming = debugTiming;
        this.parallelThreshold = parallelThreshold;
        this.duplicateBranchSuppression = duplicateBranchSuppression;

        // Initialize cache only if enabled
        if (useCache) {
            this.cellCache = Caffeine.newBuilder()
                .maximumSize(MAX_CACHE_SIZE)
                .build(this::generateCellData);
        } else {
            this.cellCache = null;
        }

        if (debugTiming) {
            LOGGER.info("DendrySampler initialized with: resolution={}, gridsize={}, useCache={}, useParallel={}, useSplines={}, parallelThreshold={}",
                resolution, gridsize, useCache, useParallel, useSplines, parallelThreshold);
        }
    }

    private CellData generateCellData(Long key) {
        int cellX = unpackX(key);
        int cellY = unpackY(key);
        Point2D point = generatePoint(cellX, cellY);
        int branches = computeBranchCount(cellX, cellY);
        return new CellData(point, branches);
    }

    private static long packKey(int x, int y) {
        return ((long) x << 32) | (y & 0xFFFFFFFFL);
    }

    private static int unpackX(long key) {
        return (int) (key >> 32);
    }

    private static int unpackY(long key) {
        return (int) key;
    }

    private int computeBranchCount(int cellX, int cellY) {
        if (branchesSampler == null) {
            return defaultBranches;
        }
        double centerX = (cellX + 0.5) * gridsize;
        double centerY = (cellY + 0.5) * gridsize;
        int branches = (int) Math.round(branchesSampler.getSample(salt, centerX, centerY));
        return Math.max(1, Math.min(8, branches));
    }

    @Override
    public double getSample(long seed, double x, double z) {
        long startTime = debugTiming ? System.nanoTime() : 0;

        double result = evaluate(seed, x / gridsize, z / gridsize);

        if (debugTiming) {
            long elapsed = System.nanoTime() - startTime;
            totalTimeNs.addAndGet(elapsed);
            long count = sampleCount.incrementAndGet();

            long now = System.currentTimeMillis();
            if (now - lastLogTime > LOG_INTERVAL_MS) {
                lastLogTime = now;
                double avgNs = (double) totalTimeNs.get() / count;
                double avgMs = avgNs / 1_000_000.0;
                LOGGER.info("DendrySampler stats: {} samples, avg {:.4f} ms/sample ({:.0f} ns)",
                    count, avgMs, avgNs);
            }
        }

        return result;
    }

    @Override
    public double getSample(long seed, double x, double y, double z) {
        return getSample(seed, x, z);
    }

    private double evaluate(long seed, double x, double y) {
        // Displacement factors decrease exponentially for each level
        double displacementLevel1 = delta;
        double displacementLevel2 = displacementLevel1 / 4.0;
        double displacementLevel3 = displacementLevel2 / 4.0;

        // Minimum slope constraints for sub-branch connections
        double minSlopeLevel1 = 0.5;
        double minSlopeLevel2 = 0.9;
        double minSlopeLevel3 = 0.18;
        double minSlopeLevel4 = 0.38;
        double minSlopeLevel5 = 1.0;

        // Level 0: Root network spanning (5 + 2*suppression) level 1 cells
        Cell cell1 = getCell(x, y, 1);
        Point3D[][] level0Points = generateLevel0Points(cell1);
        List<Segment3D> segments0 = generateLevel0Segments(level0Points);

        // Subdivide and displace level 0 for curvature (larger displacement than level 1)
        double displacementLevel0 = delta * 2.0;
        int level0Subdivisions = Math.max(2, defaultBranches);  // at least 2 subdivisions
        segments0 = subdivideSegments(segments0, level0Subdivisions, 0);
        // Use split-based displacement for tree structures (preserves all connections)
        // Pass level 0 for deterministic seeding based on absolute segment coordinates
        segments0 = displaceSegmentsWithSplit(segments0, displacementLevel0, 0);

        // Prune level 0 segments not connected to inner 3x3 region around query cell
        // The inner region is always 3x3 regardless of outer grid size
        double innerMinX = cell1.x - 1;
        double innerMaxX = cell1.x + 2;
        double innerMinY = cell1.y - 1;
        double innerMaxY = cell1.y + 2;
        segments0 = pruneDisconnectedSegments(segments0, innerMinX, innerMinY, innerMaxX, innerMaxY);

        if (resolution == 0) {
            return computeResult(x, y, segments0);
        }

        // Level 1: Process 3x3 cells around query cell
        List<Segment3D> segments1 = generateLevel1Segments(cell1, segments0, minSlopeLevel1);
        segments1 = pruneAwaySegments(segments1, cell1);

        int branchCount = getBranchCountForCell(cell1);
        segments1 = subdivideSegments(segments1, branchCount, 1);
        displaceSegments(segments1, displacementLevel1, cell1);

        List<Segment3D> allSegments = new ArrayList<>(segments0);
        allSegments.addAll(segments1);

        if (resolution == 1) {
            return computeResult(x, y, allSegments);
        }

        // Level 2: 2x Resolution
        Cell cell2 = getCell(x, y, 2);
        Point3D[][] points2 = generateNeighboringPoints3D(cell2, 5);
        List<Segment3D> segments2 = generateSubSegments(points2, allSegments, minSlopeLevel2, 2);
        displaceSegments(segments2, displacementLevel2, cell2);

        if (resolution == 2) {
            allSegments.addAll(segments2);
            return computeResult(x, y, allSegments);
        }

        // Level 3: 4x Resolution
        Cell cell3 = getCell(x, y, 4);
        Point3D[][] points3 = generateNeighboringPoints3D(cell3, 5);
        allSegments.addAll(segments2);
        List<Segment3D> segments3 = generateSubSegments(points3, allSegments, minSlopeLevel3, 3);
        displaceSegments(segments3, displacementLevel3, cell3);

        if (resolution == 3) {
            allSegments.addAll(segments3);
            return computeResult(x, y, allSegments);
        }

        // Level 4: 8x Resolution
        Cell cell4 = getCell(x, y, 8);
        Point3D[][] points4 = generateNeighboringPoints3D(cell4, 5);
        allSegments.addAll(segments3);
        List<Segment3D> segments4 = generateSubSegments(points4, allSegments, minSlopeLevel4, 4);

        if (resolution == 4) {
            allSegments.addAll(segments4);
            return computeResult(x, y, allSegments);
        }

        // Level 5: 16x Resolution
        Cell cell5 = getCell(x, y, 16);
        Point3D[][] points5 = generateNeighboringPoints3D(cell5, 5);
        allSegments.addAll(segments4);
        List<Segment3D> segments5 = generateSubSegments(points5, allSegments, minSlopeLevel5, 5);
        allSegments.addAll(segments5);

        return computeResult(x, y, allSegments);
    }

    private static class Cell {
        final int x;
        final int y;
        final int resolution;

        Cell(int x, int y, int resolution) {
            this.x = x;
            this.y = y;
            this.resolution = resolution;
        }
    }

    private Cell getCell(double x, double y, int resolution) {
        int cellX = MathUtils.floor(x * resolution);
        int cellY = MathUtils.floor(y * resolution);
        return new Cell(cellX, cellY, resolution);
    }

    private Point2D generatePoint(int cellX, int cellY) {
        Random rng = initRandomGenerator(cellX, cellY, 1);
        double px = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        double py = epsilon + rng.nextDouble() * (1.0 - 2.0 * epsilon);
        return new Point2D(cellX + px, cellY + py);
    }

    /**
     * Get cell data - uses cache if enabled, otherwise generates directly.
     */
    private CellData getCellData(int cellX, int cellY) {
        if (useCache && cellCache != null) {
            return cellCache.get(packKey(cellX, cellY));
        }
        // Direct generation without caching
        Point2D point = generatePoint(cellX, cellY);
        int branches = computeBranchCount(cellX, cellY);
        return new CellData(point, branches);
    }

    private int getBranchCountForCell(Cell cell) {
        CellData data = getCellData(cell.x, cell.y);
        return data.branchCount;
    }

    private Random initRandomGenerator(int x, int y, int level) {
        long seed = (541L * x + 79L * y + level * 863L + salt) & 0x7FFFFFFFL;
        return new Random(seed);
    }

    private Point3D[][] generateNeighboringPoints3D(Cell cell, int size) {
        Point3D[][] points = new Point3D[size][size];
        int half = size / 2;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int px = cell.x + j - half;
                int py = cell.y + i - half;

                CellData data = getCellData(px, py);
                Point2D p2d = data.point;

                Point2D scaled = new Point2D(p2d.x / cell.resolution, p2d.y / cell.resolution);
                double elevation = evaluateControlFunction(scaled.x, scaled.y);
                points[i][j] = new Point3D(scaled.x, scaled.y, elevation);
            }
        }
        return points;
    }

    /**
     * Get the level 0 grid size based on duplicate branch suppression.
     * gridSize = 5 + 2 * duplicateBranchSuppression
     * (e.g., 0 = 5x5, 1 = 7x7, 2 = 9x9)
     */
    private int getLevel0GridSize() {
        return 5 + 2 * duplicateBranchSuppression;
    }

    /**
     * Generate level 0 points: one per level 1 cell.
     * Grid size is determined by duplicateBranchSuppression config.
     * Points are placed at the approximate lowest elevation within each cell
     * (found by sampling multiple points and selecting the minimum).
     */
    private Point3D[][] generateLevel0Points(Cell cell1) {
        int size = getLevel0GridSize();
        int half = size / 2;
        Point3D[][] points = new Point3D[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int cellX = cell1.x + j - half;
                int cellY = cell1.y + i - half;

                // Find approximate lowest point in this cell
                Point3D lowestPoint = findLowestPointInCell(cellX, cellY);
                points[i][j] = lowestPoint;
            }
        }
        return points;
    }

    /**
     * Find the approximate lowest elevation point within a cell by sampling.
     * Uses a 5x5 grid of sample points and selects the one with minimum elevation.
     * Adds deterministic jitter to avoid grid-aligned patterns across cells.
     */
    private Point3D findLowestPointInCell(int cellX, int cellY) {
        final int SAMPLES_PER_AXIS = 5;  // 5x5 = 25 samples

        // Generate deterministic jitter for this cell (same offset for all samples)
        Random rng = initRandomGenerator(cellX, cellY, 0);
        double jitterX = (rng.nextDouble() - 0.5) * 0.15;  // +/- 0.075 cell units
        double jitterY = (rng.nextDouble() - 0.5) * 0.15;

        double lowestElevation = Double.MAX_VALUE;
        double lowestX = cellX + 0.5;
        double lowestY = cellY + 0.5;

        // Sample on a jittered grid within the cell
        for (int si = 0; si < SAMPLES_PER_AXIS; si++) {
            for (int sj = 0; sj < SAMPLES_PER_AXIS; sj++) {
                // Position within cell [0.1, 0.9] to stay away from edges, plus jitter
                double tx = 0.1 + 0.8 * (sj + 0.5) / SAMPLES_PER_AXIS + jitterX;
                double ty = 0.1 + 0.8 * (si + 0.5) / SAMPLES_PER_AXIS + jitterY;

                // Clamp to stay within cell bounds [0.05, 0.95]
                tx = Math.max(0.05, Math.min(0.95, tx));
                ty = Math.max(0.05, Math.min(0.95, ty));

                double sampleX = cellX + tx;
                double sampleY = cellY + ty;

                double elevation = evaluateControlFunction(sampleX, sampleY);

                if (elevation < lowestElevation) {
                    lowestElevation = elevation;
                    lowestX = sampleX;
                    lowestY = sampleY;
                }
            }
        }

        return new Point3D(lowestX, lowestY, lowestElevation);
    }

    /**
     * Generate level 0 segments using elevation-based connections.
     * Each point connects to at least 2 neighbors, preferring lower elevation.
     * This creates a deterministic, tileable network.
     */
    private List<Segment3D> generateLevel0Segments(Point3D[][] points) {
        int size = points.length;
        List<Segment3D> segments = new ArrayList<>();

        // Track unique connections to avoid duplicates (use canonical ordering)
        java.util.Set<Long> addedConnections = new java.util.HashSet<>();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Point3D current = points[i][j];

                // Collect all neighbors with their properties
                List<NeighborInfo> neighbors = new ArrayList<>();

                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni < 0 || ni >= size || nj < 0 || nj >= size) continue;

                        Point3D neighbor = points[ni][nj];
                        double dist2D = current.projectZ().distanceTo(neighbor.projectZ());
                        double elevDiff = neighbor.z - current.z;  // negative = lower

                        neighbors.add(new NeighborInfo(neighbor, ni, nj, dist2D, elevDiff));
                    }
                }

                // Sort: prefer lower elevation, then by 2D distance
                neighbors.sort((a, b) -> {
                    // First: prefer lower or equal elevation
                    boolean aLowerOrEqual = a.elevDiff <= 0;
                    boolean bLowerOrEqual = b.elevDiff <= 0;

                    if (aLowerOrEqual && !bLowerOrEqual) return -1;
                    if (!aLowerOrEqual && bLowerOrEqual) return 1;

                    // Among same category, sort by 2D distance
                    return Double.compare(a.dist2D, b.dist2D);
                });

                // Connect to at least 2 neighbors
                int connectCount = 0;
                for (NeighborInfo n : neighbors) {
                    if (connectCount >= 2) break;

                    // Create canonical connection key (smaller index first)
                    long key = createConnectionKey(i, j, n.gridI, n.gridJ, size);
                    if (!addedConnections.contains(key)) {
                        addedConnections.add(key);
                        segments.add(new Segment3D(current, n.point, 0));
                    }
                    connectCount++;
                }
            }
        }

        // Prune duplicate short paths
        segments = pruneDuplicatePaths(segments, points);

        return segments;
    }

    /**
     * Neighbor info for sorting during connection selection.
     */
    private static class NeighborInfo {
        final Point3D point;
        final int gridI, gridJ;
        final double dist2D;
        final double elevDiff;

        NeighborInfo(Point3D point, int gridI, int gridJ, double dist2D, double elevDiff) {
            this.point = point;
            this.gridI = gridI;
            this.gridJ = gridJ;
            this.dist2D = dist2D;
            this.elevDiff = elevDiff;
        }
    }

    /**
     * Create a canonical key for a connection between two grid positions.
     * Ensures (i1,j1)-(i2,j2) and (i2,j2)-(i1,j1) map to the same key.
     */
    private long createConnectionKey(int i1, int j1, int i2, int j2, int size) {
        int idx1 = i1 * size + j1;
        int idx2 = i2 * size + j2;
        if (idx1 > idx2) {
            int tmp = idx1;
            idx1 = idx2;
            idx2 = tmp;
        }
        return ((long) idx1 << 32) | (idx2 & 0xFFFFFFFFL);
    }

    /**
     * Prune duplicate paths between nodes.
     * If two nodes are connected by multiple chains of length <= (1 + duplicateBranchSuppression),
     * keep only the shortest chain. If same length, use deterministic selection.
     */
    private List<Segment3D> pruneDuplicatePaths(List<Segment3D> segments, Point3D[][] points) {
        if (duplicateBranchSuppression == 0) {
            return segments;  // No pruning for level 0
        }

        int maxPathLength = 1 + duplicateBranchSuppression;

        // Build adjacency map from segments
        Map<Long, List<Segment3D>> adjacency = new HashMap<>();
        for (Segment3D seg : segments) {
            long keyA = pointHash(seg.a);
            long keyB = pointHash(seg.b);
            adjacency.computeIfAbsent(keyA, k -> new ArrayList<>()).add(seg);
            adjacency.computeIfAbsent(keyB, k -> new ArrayList<>()).add(seg);
        }

        // Find all pairs of nodes connected by short paths
        java.util.Set<Segment3D> toRemove = new java.util.HashSet<>();

        for (Segment3D seg : segments) {
            if (toRemove.contains(seg)) continue;

            // For each segment, look for alternate paths between its endpoints
            List<List<Segment3D>> alternatePaths = findAlternatePaths(
                seg.a, seg.b, seg, adjacency, maxPathLength
            );

            if (!alternatePaths.isEmpty()) {
                // We have the direct segment (length 1) and alternate paths
                // Keep the shortest path; if tie, use deterministic selection

                // Direct segment is always length 1
                List<Segment3D> directPath = List.of(seg);
                List<List<Segment3D>> allPaths = new ArrayList<>();
                allPaths.add(directPath);
                allPaths.addAll(alternatePaths);

                // Sort by length, then by deterministic hash
                allPaths.sort((a, b) -> {
                    int lenCompare = Integer.compare(a.size(), b.size());
                    if (lenCompare != 0) return lenCompare;
                    // Same length: use deterministic hash
                    return Long.compare(pathHash(a), pathHash(b));
                });

                // Keep the first path, mark others for removal
                List<Segment3D> keepPath = allPaths.get(0);
                for (int i = 1; i < allPaths.size(); i++) {
                    for (Segment3D s : allPaths.get(i)) {
                        // Only remove if it's a short segment and not in keepPath
                        if (!keepPath.contains(s)) {
                            toRemove.add(s);
                        }
                    }
                }
            }
        }

        // Return filtered list
        return segments.stream()
            .filter(s -> !toRemove.contains(s))
            .collect(Collectors.toList());
    }

    /**
     * Find alternate paths between two points, excluding a specific segment.
     */
    private List<List<Segment3D>> findAlternatePaths(Point3D start, Point3D end,
                                                      Segment3D excludeSeg,
                                                      Map<Long, List<Segment3D>> adjacency,
                                                      int maxLength) {
        List<List<Segment3D>> results = new ArrayList<>();
        long endKey = pointHash(end);

        // BFS to find paths up to maxLength
        java.util.Queue<PathState> queue = new java.util.LinkedList<>();
        queue.add(new PathState(start, new ArrayList<>(), new java.util.HashSet<>()));

        while (!queue.isEmpty()) {
            PathState state = queue.poll();

            if (state.path.size() >= maxLength) continue;

            long currentKey = pointHash(state.current);
            List<Segment3D> neighbors = adjacency.get(currentKey);
            if (neighbors == null) continue;

            for (Segment3D seg : neighbors) {
                if (seg == excludeSeg) continue;
                if (state.visited.contains(seg)) continue;

                Point3D next = pointsMatch(seg.a, state.current) ? seg.b : seg.a;
                long nextKey = pointHash(next);

                List<Segment3D> newPath = new ArrayList<>(state.path);
                newPath.add(seg);

                if (nextKey == endKey || pointsMatch(next, end)) {
                    // Found a path to end
                    results.add(newPath);
                } else if (newPath.size() < maxLength) {
                    java.util.Set<Segment3D> newVisited = new java.util.HashSet<>(state.visited);
                    newVisited.add(seg);
                    queue.add(new PathState(next, newPath, newVisited));
                }
            }
        }

        return results;
    }

    /**
     * State for path-finding BFS.
     */
    private static class PathState {
        final Point3D current;
        final List<Segment3D> path;
        final java.util.Set<Segment3D> visited;

        PathState(Point3D current, List<Segment3D> path, java.util.Set<Segment3D> visited) {
            this.current = current;
            this.path = path;
            this.visited = visited;
        }
    }

    /**
     * Compute a deterministic hash for a path (for tie-breaking).
     */
    private long pathHash(List<Segment3D> path) {
        long hash = 0;
        for (Segment3D seg : path) {
            // Use coordinates to create deterministic hash
            hash ^= Double.doubleToLongBits(seg.a.x + seg.b.x) * 73856093L;
            hash ^= Double.doubleToLongBits(seg.a.y + seg.b.y) * 19349663L;
        }
        return hash;
    }

    /**
     * Edge class for MST algorithm.
     */
    private static class Edge {
        final int idx1, idx2;
        final double distance;
        final Point3D p1, p2;

        Edge(int idx1, int idx2, double distance, Point3D p1, Point3D p2) {
            this.idx1 = idx1;
            this.idx2 = idx2;
            this.distance = distance;
            this.p1 = p1;
            this.p2 = p2;
        }
    }

    /**
     * Find root with path compression (Union-Find).
     */
    private int find(int[] parent, int i) {
        if (parent[i] != i) {
            parent[i] = find(parent, parent[i]);
        }
        return parent[i];
    }

    /**
     * Union by rank (Union-Find).
     */
    private void union(int[] parent, int[] rank, int x, int y) {
        if (rank[x] < rank[y]) {
            parent[x] = y;
        } else if (rank[x] > rank[y]) {
            parent[y] = x;
        } else {
            parent[y] = x;
            rank[x]++;
        }
    }

    /**
     * Remove segments that don't connect to the inner 3x3 cell region.
     */
    private List<Segment3D> pruneDisconnectedSegments(List<Segment3D> segments,
                                                       double minX, double minY,
                                                       double maxX, double maxY) {
        return segments.stream()
            .filter(seg -> {
                // Keep if either endpoint is within inner region
                boolean aInside = isPointInBounds(seg.a, minX, minY, maxX, maxY);
                boolean bInside = isPointInBounds(seg.b, minX, minY, maxX, maxY);
                return aInside || bInside;
            })
            .collect(Collectors.toList());
    }

    private boolean isPointInBounds(Point3D p, double minX, double minY,
                                     double maxX, double maxY) {
        return p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY;
    }

    /**
     * Generate level 1 segments for 3x3 cell region around query cell.
     */
    private List<Segment3D> generateLevel1Segments(Cell queryCell, List<Segment3D> parentSegments, double minSlope) {
        List<Segment3D> allSegments = new ArrayList<>();

        // Process 3x3 grid of level 1 cells
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                Cell cell = new Cell(queryCell.x + dj, queryCell.y + di, 1);

                // Generate points for this cell's neighborhood (5x5 including neighbors)
                Point3D[][] points = generateNeighboringPoints3D(cell, 5);

                // Generate segments that connect to parent (level 0) segments
                List<Segment3D> cellSegments = generateSubSegments(points, parentSegments, minSlope, 1);
                allSegments.addAll(cellSegments);
            }
        }

        return allSegments;
    }

    /**
     * Prune segments that start outside query cell and move away from it.
     */
    private List<Segment3D> pruneAwaySegments(List<Segment3D> segments, Cell queryCell) {
        // Query cell center in normalized coordinates
        double centerX = (queryCell.x + 0.5) / queryCell.resolution;
        double centerY = (queryCell.y + 0.5) / queryCell.resolution;
        Point2D center = new Point2D(centerX, centerY);

        // Query cell bounds
        double cellMinX = (double) queryCell.x / queryCell.resolution;
        double cellMaxX = (double) (queryCell.x + 1) / queryCell.resolution;
        double cellMinY = (double) queryCell.y / queryCell.resolution;
        double cellMaxY = (double) (queryCell.y + 1) / queryCell.resolution;

        return segments.stream()
            .filter(seg -> {
                Point2D a = seg.a.projectZ();
                Point2D b = seg.b.projectZ();

                // If segment starts inside query cell, keep it
                if (isInCell(a, cellMinX, cellMinY, cellMaxX, cellMaxY)) {
                    return true;
                }

                // Segment starts outside - check if it moves toward center
                double distA = a.distanceTo(center);
                double distB = b.distanceTo(center);

                // Keep if endpoint B is closer to center than A (moving toward)
                return distB < distA;
            })
            .collect(Collectors.toList());
    }

    private boolean isInCell(Point2D p, double minX, double minY,
                              double maxX, double maxY) {
        return p.x >= minX && p.x < maxX && p.y >= minY && p.y < maxY;
    }

    private double evaluateControlFunction(double x, double y) {
        if (controlSampler != null) {
            return controlSampler.getSample(salt, x * gridsize, y * gridsize);
        }
        return x * 0.1;
    }

    private List<Segment3D> generateSegments(Point3D[][] points, int level) {
        List<Segment3D> segments = new ArrayList<>();
        int size = points.length;

        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                Point3D current = points[i][j];

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

                if (current.distanceSquaredTo(lowest) > MathUtils.EPSILON) {
                    segments.add(new Segment3D(current, lowest, level));
                }
            }
        }
        return segments;
    }

    /**
     * Subdivide segments - uses splines if enabled, otherwise linear.
     */
    private List<Segment3D> subdivideSegments(List<Segment3D> segments, int numDivisions, int level) {
        if (segments.isEmpty() || numDivisions <= 1) {
            return segments;
        }

        if (useSplines && curvature > 0) {
            return subdivideWithSplines(segments, numDivisions, level);
        } else {
            return subdivideLinear(segments, numDivisions, level);
        }
    }

    /**
     * Simple linear subdivision (fast).
     */
    private List<Segment3D> subdivideLinear(List<Segment3D> segments, int numDivisions, int level) {
        List<Segment3D> subdivided = new ArrayList<>();

        for (Segment3D seg : segments) {
            Point3D prev = seg.a;
            for (int j = 1; j <= numDivisions; j++) {
                double t = (double) j / numDivisions;
                Point3D next = (j == numDivisions) ? seg.b : Point3D.lerp(seg.a, seg.b, t);
                subdivided.add(new Segment3D(prev, next, level));
                prev = next;
            }
        }
        return subdivided;
    }

    /**
     * Catmull-Rom spline subdivision (smoother but slower).
     * Uses connectivity-based control point lookup to ensure deterministic results
     * regardless of segment list order (important for MST/tree structures).
     */
    private List<Segment3D> subdivideWithSplines(List<Segment3D> segments, int numDivisions, int level) {
        double levelCurvature = curvature * Math.pow(curvatureFalloff, level - 1);
        List<Segment3D> subdivided = new ArrayList<>();

        // Build connectivity map: point -> list of segments connected to that point
        Map<Long, List<Segment3D>> connectivityMap = buildConnectivityMap(segments);

        for (Segment3D seg : segments) {
            Point3D p1 = seg.a;
            Point3D p2 = seg.b;

            // Find control point p0: the "other end" of a segment connected to p1 (not p2)
            Point3D p0 = findConnectedControlPoint(p1, p2, seg, connectivityMap);

            // Find control point p3: the "other end" of a segment connected to p2 (not p1)
            Point3D p3 = findConnectedControlPoint(p2, p1, seg, connectivityMap);

            Point3D prev = p1;
            for (int j = 1; j <= numDivisions; j++) {
                double t = (double) j / numDivisions;
                Point3D next;

                if (j == numDivisions) {
                    next = p2;
                } else {
                    Point3D linear = Point3D.lerp(p1, p2, t);
                    Point3D spline = CatmullRomSpline.subdivide(p0, p1, p2, p3, t);
                    next = Point3D.lerp(linear, spline, levelCurvature);
                }

                subdivided.add(new Segment3D(prev, next, level));
                prev = next;
            }
        }
        return subdivided;
    }

    /**
     * Build a map from point hash to segments connected at that point.
     * Uses a spatial hash of point coordinates for lookup.
     */
    private Map<Long, List<Segment3D>> buildConnectivityMap(List<Segment3D> segments) {
        Map<Long, List<Segment3D>> map = new HashMap<>();

        for (Segment3D seg : segments) {
            long keyA = pointHash(seg.a);
            long keyB = pointHash(seg.b);

            map.computeIfAbsent(keyA, k -> new ArrayList<>()).add(seg);
            map.computeIfAbsent(keyB, k -> new ArrayList<>()).add(seg);
        }

        return map;
    }

    /**
     * Hash a point's coordinates for connectivity lookup.
     * Uses fixed precision to handle floating point comparison.
     */
    private long pointHash(Point3D p) {
        // Quantize to ~0.0001 precision for reliable matching
        long hx = (long)(p.x * 10000);
        long hy = (long)(p.y * 10000);
        return (hx * 73856093L) ^ (hy * 19349663L);
    }

    /**
     * Find a control point for spline interpolation by looking at connectivity.
     * Given a point 'anchor' and the 'other' end of the current segment,
     * find another segment connected to 'anchor' and return its far endpoint.
     *
     * @param anchor The point we're looking for connections at
     * @param other The other end of the current segment (to exclude)
     * @param currentSeg The current segment (to exclude from search)
     * @param connectivityMap Map of point hash -> connected segments
     * @return The control point, or 'anchor' itself if no other connection found
     */
    private Point3D findConnectedControlPoint(Point3D anchor, Point3D other, Segment3D currentSeg,
                                               Map<Long, List<Segment3D>> connectivityMap) {
        long key = pointHash(anchor);
        List<Segment3D> connected = connectivityMap.get(key);

        if (connected == null || connected.size() <= 1) {
            // No other segments connected - use anchor as control point (linear at endpoint)
            return anchor;
        }

        // Find a segment other than currentSeg connected at anchor
        for (Segment3D seg : connected) {
            if (seg == currentSeg) continue;

            // Determine which end of this segment is at 'anchor' and return the other end
            if (pointsMatch(seg.a, anchor)) {
                return seg.b;
            } else if (pointsMatch(seg.b, anchor)) {
                return seg.a;
            }
        }

        // No other connection found
        return anchor;
    }

    /**
     * Check if two points match within epsilon tolerance.
     */
    private boolean pointsMatch(Point3D a, Point3D b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return (dx * dx + dy * dy) < MathUtils.EPSILON * MathUtils.EPSILON * 10000;
    }

    /**
     * Displace segments by splitting each into two segments with a displaced midpoint.
     * Displacement is proportional to segment length for consistent curvature appearance.
     * Returns a new list (does not modify input list structure).
     * Uses absolute segment coordinates for deterministic displacement regardless of query position.
     */
    private List<Segment3D> displaceSegmentsWithSplit(List<Segment3D> segments, double displacementFactor, int level) {
        if (displacementFactor < MathUtils.EPSILON) return segments;

        List<Segment3D> result = new ArrayList<>();

        for (Segment3D seg : segments) {
            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) {
                result.add(seg);
                continue;
            }

            Vec2D perp = dir.rotateCCW90().normalize();

            // Use both endpoints for seed to ensure determinism based on segment identity
            int seedX = (int)((seg.a.x + seg.b.x) * 50);
            int seedY = (int)((seg.a.y + seg.b.y) * 50);
            Random rng = initRandomGenerator(seedX, seedY, level);
            // Displacement proportional to segment length
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor * segLength;

            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            // Split into two segments: a→mid and mid→b (preserves connectivity)
            result.add(new Segment3D(seg.a, displacedMid, seg.level));
            result.add(new Segment3D(displacedMid, seg.b, seg.level));
        }

        return result;
    }

    /**
     * Legacy displacement (modifies list in place, for chain-ordered segments).
     */
    private void displaceSegments(List<Segment3D> segments, double displacementFactor, Cell cell) {
        if (displacementFactor < MathUtils.EPSILON) return;

        for (int i = 0; i < segments.size(); i++) {
            Segment3D seg = segments.get(i);

            Vec2D dir = new Vec2D(seg.a.projectZ(), seg.b.projectZ());
            double segLength = dir.length();

            if (segLength < MathUtils.EPSILON) continue;

            Vec2D perp = dir.rotateCCW90().normalize();

            Random rng = initRandomGenerator((int)(seg.a.x * 100), (int)(seg.a.y * 100), cell.resolution);
            // Displacement proportional to segment length
            double displacement = (rng.nextDouble() * 2.0 - 1.0) * displacementFactor * segLength;

            Point3D mid = seg.midpoint();
            Point3D displacedMid = new Point3D(
                mid.x + perp.x * displacement,
                mid.y + perp.y * displacement,
                mid.z
            );

            segments.set(i, new Segment3D(seg.a, displacedMid, seg.level));

            if (i + 1 < segments.size()) {
                Segment3D next = segments.get(i + 1);
                if (next.a.equals(seg.b)) {
                    segments.set(i + 1, new Segment3D(displacedMid, next.b, next.level));
                }
            }
        }
    }

    private double getConnectDistance(int level) {
        if (connectDistance > 0) {
            return connectDistance / level;
        }
        double cellSize = 1.0 / Math.pow(2, level - 1);
        return cellSize * connectDistanceFactor;
    }

    private List<Segment3D> generateSubSegments(Point3D[][] points, List<Segment3D> parentSegments,
                                                 double minSlope, int level) {
        List<Segment3D> subSegments = new ArrayList<>();
        int size = points.length;
        double maxDistance = getConnectDistance(level);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                Point3D point = points[i][j];

                NearestSegmentResult nearest = findNearestSegment(point.projectZ(), parentSegments);
                if (nearest == null || nearest.distance > maxDistance) continue;

                double elevationFromControl = evaluateControlFunction(point.x, point.y);
                double elevationWithSlope = nearest.closestPoint.z + minSlope * nearest.distance;
                double elevation = Math.max(elevationWithSlope, elevationFromControl);

                Point3D adjustedPoint = new Point3D(point.x, point.y, elevation);

                if (nearest.distance > MathUtils.EPSILON) {
                    Point3D connectionPoint = new Point3D(
                        nearest.closestPoint2D.x,
                        nearest.closestPoint2D.y,
                        nearest.closestPoint.z
                    );
                    subSegments.add(new Segment3D(adjustedPoint, connectionPoint, level));
                }
            }
        }
        return subSegments;
    }

    private static class NearestSegmentResult {
        final double distance;
        final double weightedDistance;
        final Point2D closestPoint2D;
        final Point3D closestPoint;
        final Segment3D segment;

        NearestSegmentResult(double distance, double weightedDistance, Point2D closestPoint2D,
                            Point3D closestPoint, Segment3D segment) {
            this.distance = distance;
            this.weightedDistance = weightedDistance;
            this.closestPoint2D = closestPoint2D;
            this.closestPoint = closestPoint;
            this.segment = segment;
        }
    }

    /**
     * Find nearest segment - uses parallel streams if enabled and list is large enough.
     */
    private NearestSegmentResult findNearestSegment(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream;
        if (useParallel && segments.size() > parallelThreshold) {
            stream = segments.parallelStream();
        } else {
            stream = segments.stream();
        }

        return stream
            .map(seg -> {
                MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                double weightedDist = result.distance / seg.level;
                return new NearestSegmentResult(
                    result.distance,
                    weightedDist,
                    result.closestPoint,
                    new Point3D(result.closestPoint.x, result.closestPoint.y, z),
                    seg
                );
            })
            .min(Comparator.comparingDouble(r -> r.distance))
            .orElse(null);
    }

    private NearestSegmentResult findNearestSegmentWeighted(Point2D point, List<Segment3D> segments) {
        if (segments.isEmpty()) return null;

        Stream<Segment3D> stream;
        if (useParallel && segments.size() > parallelThreshold) {
            stream = segments.parallelStream();
        } else {
            stream = segments.stream();
        }

        return stream
            .map(seg -> {
                MathUtils.DistanceResult result = MathUtils.distanceToLineSegment(point, seg);
                double z = MathUtils.lerp(seg.a.z, seg.b.z, result.parameter);
                double weightedDist = result.distance / seg.level;
                return new NearestSegmentResult(
                    result.distance,
                    weightedDist,
                    result.closestPoint,
                    new Point3D(result.closestPoint.x, result.closestPoint.y, z),
                    seg
                );
            })
            .min(Comparator.comparingDouble(r -> r.weightedDistance))
            .orElse(null);
    }

    private double computeResult(double x, double y, List<Segment3D> segments) {
        Point2D point = new Point2D(x, y);

        switch (returnType) {
            case DISTANCE:
                NearestSegmentResult nearestDist = findNearestSegment(point, segments);
                if (nearestDist == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestDist.distance;

            case WEIGHTED:
                NearestSegmentResult nearestWeighted = findNearestSegmentWeighted(point, segments);
                if (nearestWeighted == null) {
                    return evaluateControlFunction(x, y);
                }
                return nearestWeighted.weightedDistance;

            case ELEVATION:
            default:
                NearestSegmentResult nearest = findNearestSegment(point, segments);
                if (nearest == null) {
                    return evaluateControlFunction(x, y);
                }
                double baseElevation = nearest.closestPoint.z;
                double slopeContribution = nearest.distance * slope;
                return baseElevation + slopeContribution;
        }
    }
}

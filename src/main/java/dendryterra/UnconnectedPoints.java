package dendryterra;

import dendryterra.math.*;
import java.util.*;
import java.util.function.Consumer;

/**
 * Holds points generated for the current level awaiting connection.
 * Points are removed as they're linked into the SegmentList.
 *
 * This class tracks a point cloud that will be consumed during networking.
 * Points can only connect TO points already in the SegmentList (except for
 * the first trunk segment at level 0).
 */
public class UnconnectedPoints {
    private final List<NetworkPoint> points;
    private final Set<Integer> removedIndices;
    private int nextLocalIndex;

    public UnconnectedPoints() {
        this.points = new ArrayList<>();
        this.removedIndices = new HashSet<>();
        this.nextLocalIndex = 0;
    }

    /**
     * Create UnconnectedPoints from a list of Point3D positions.
     * All points are assigned the given type and level.
     */
    public static UnconnectedPoints fromPoints(List<Point3D> positions, PointType type, int level) {
        UnconnectedPoints result = new UnconnectedPoints();
        for (Point3D pos : positions) {
            result.addPoint(pos, type, level);
        }
        return result;
    }

    // ========== Add Operations ==========

    /**
     * Add a new point to the unconnected pool.
     * @return The local index assigned to this point
     */
    public int addPoint(Point3D position, PointType type, int level) {
        int localIdx = nextLocalIndex++;
        points.add(new NetworkPoint(position, localIdx, type, level));
        return localIdx;
    }

    /**
     * Add an existing NetworkPoint to the unconnected pool.
     * The point is re-indexed with a local index.
     * @return The new local index assigned to this point
     */
    public int addPoint(NetworkPoint point) {
        int localIdx = nextLocalIndex++;
        points.add(point.withIndex(localIdx));
        return localIdx;
    }

    // ========== Access Operations ==========

    /**
     * Get a point by its local index.
     */
    public NetworkPoint getPoint(int localIndex) {
        return points.get(localIndex);
    }

    /**
     * Mark a point as removed (consumed by the segment list).
     */
    public void markRemoved(int localIndex) {
        removedIndices.add(localIndex);
    }

    /**
     * Check if a point has been removed.
     */
    public boolean isRemoved(int localIndex) {
        return removedIndices.contains(localIndex);
    }

    /**
     * Get a point and mark it as removed (for transfer to SegmentList).
     */
    public NetworkPoint removeAndGet(int localIndex) {
        NetworkPoint p = points.get(localIndex);
        removedIndices.add(localIndex);
        return p;
    }

    // ========== Query Operations ==========

    /**
     * Find the highest unconnected point (for trunk building at level 0).
     * Skips EDGE points.
     * @return Local index of highest point, or -1 if none remain
     */
    public int findHighestUnconnected() {
        int bestIdx = -1;
        double bestZ = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < points.size(); i++) {
            if (removedIndices.contains(i)) continue;
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;

            if (p.position.z > bestZ) {
                bestZ = p.position.z;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Find all points within the given horizontal distance of any point in the SegmentList.
     * Returns local indices of matching unconnected points.
     */
    public List<Integer> findPointsNearSegmentList(SegmentList segList, double maxDist) {
        if (segList.isEmpty()) {
            return Collections.emptyList();
        }

        List<Integer> result = new ArrayList<>();
        double maxDistSq = maxDist * maxDist;

        for (int i = 0; i < points.size(); i++) {
            if (removedIndices.contains(i)) continue;
            NetworkPoint unconnPt = points.get(i);
            if (unconnPt.pointType == PointType.EDGE) continue;

            // Check distance to any SegmentList point
            for (NetworkPoint segPt : segList.getPoints()) {
                if (segPt.pointType == PointType.EDGE) continue;
                double distSq = unconnPt.position.projectZ()
                    .distanceSquaredTo(segPt.position.projectZ());
                if (distSq <= maxDistSq) {
                    result.add(i);
                    break;  // Found a match, no need to check more SegmentList points
                }
            }
        }
        return result;
    }

    /**
     * Find the unconnected point closest to any point in the SegmentList.
     * Returns a pair of (unconnected local index, segmentList point index).
     * @return int array [unconnectedIdx, segListIdx], or null if none found
     */
    public int[] findClosestToSegmentList(SegmentList segList, double maxDist) {
        if (segList.isEmpty() || isEmpty()) {
            return null;
        }

        int bestUnconnIdx = -1;
        int bestSegIdx = -1;
        double bestDistSq = maxDist * maxDist;

        for (int i = 0; i < points.size(); i++) {
            if (removedIndices.contains(i)) continue;
            NetworkPoint unconnPt = points.get(i);
            if (unconnPt.pointType == PointType.EDGE) continue;

            List<NetworkPoint> segPoints = segList.getPoints();
            for (int j = 0; j < segPoints.size(); j++) {
                NetworkPoint segPt = segPoints.get(j);
                if (segPt.pointType == PointType.EDGE) continue;

                double distSq = unconnPt.position.projectZ()
                    .distanceSquaredTo(segPt.position.projectZ());
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestUnconnIdx = i;
                    bestSegIdx = j;
                }
            }
        }

        if (bestUnconnIdx >= 0) {
            return new int[] { bestUnconnIdx, bestSegIdx };
        }
        return null;
    }

    /**
     * Get all remaining (non-removed) point indices.
     */
    public List<Integer> getRemainingIndices() {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            if (!removedIndices.contains(i)) {
                result.add(i);
            }
        }
        return result;
    }

    // ========== Size Operations ==========

    /**
     * Get the number of remaining (non-removed) points.
     */
    public int size() {
        return points.size() - removedIndices.size();
    }

    /**
     * Check if all points have been removed.
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Get the total number of points (including removed).
     */
    public int totalSize() {
        return points.size();
    }

    // ========== Iteration ==========

    /**
     * Iterate over remaining (non-removed) points.
     */
    public void forEach(Consumer<NetworkPoint> action) {
        for (int i = 0; i < points.size(); i++) {
            if (!removedIndices.contains(i)) {
                action.accept(points.get(i));
            }
        }
    }

    /**
     * Iterate over remaining points with their indices.
     */
    public void forEachIndexed(IndexedPointConsumer action) {
        for (int i = 0; i < points.size(); i++) {
            if (!removedIndices.contains(i)) {
                action.accept(i, points.get(i));
            }
        }
    }

    /**
     * Functional interface for indexed iteration.
     */
    @FunctionalInterface
    public interface IndexedPointConsumer {
        void accept(int index, NetworkPoint point);
    }

    @Override
    public String toString() {
        return String.format("UnconnectedPoints(total=%d, remaining=%d)",
                             points.size(), size());
    }
}

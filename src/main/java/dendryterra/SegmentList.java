package dendryterra;

import dendryterra.math.*;
import java.util.*;

/**
 * Container for connected network of points and segments.
 * The single source of truth for network topology.
 *
 * Points are stored as NetworkPoint objects with unique indices.
 * Segments reference points by index (srtIdx, endIdx).
 * Connection counts are automatically maintained when segments are added/removed.
 */
public class SegmentList {
    private final List<NetworkPoint> points;
    private final List<Segment3D> segments;
    private int nextIndex;  // For generating unique indices

    public SegmentList() {
        this.points = new ArrayList<>();
        this.segments = new ArrayList<>();
        this.nextIndex = 0;
    }

    // ========== Point Operations ==========

    /**
     * Add a new point with the given position, type, and level.
     * @return The index assigned to this point
     */
    public int addPoint(Point3D position, PointType type, int level) {
        int idx = nextIndex++;
        points.add(new NetworkPoint(position, idx, type, level));
        return idx;
    }

    /**
     * Add an existing NetworkPoint, re-indexing it for this list.
     * @return The new index assigned to this point
     */
    public int addPoint(NetworkPoint point) {
        int idx = nextIndex++;
        NetworkPoint reindexed = point.withIndex(idx);
        points.add(reindexed);
        return idx;
    }

    /**
     * Get a point by its index.
     */
    public NetworkPoint getPoint(int index) {
        return points.get(index);
    }

    /**
     * Update a point at the given index.
     */
    public void updatePoint(int index, NetworkPoint newPoint) {
        points.set(index, newPoint);
    }

    /**
     * Get the total number of points.
     */
    public int getPointCount() {
        return points.size();
    }

    /**
     * Get an unmodifiable view of all points.
     */
    public List<NetworkPoint> getPoints() {
        return Collections.unmodifiableList(points);
    }

    // ========== Segment Operations ==========

    /**
     * Add a segment and update connection counts on its endpoints.
     * Note: Currently accepts Segment3D with Point3D srt/end.
     * After Segment3D is modified to use indices, this will accept index-based segments.
     */
    public void addSegment(Segment3D segment) {
        segments.add(segment);
        // Connection count updates will be handled after Segment3D transition
        // For now, we'll update counts based on finding points by position
        updateConnectionCountsForSegment(segment, 1);
    }

    /**
     * Add a segment directly using point indices.
     * Creates a Segment3D internally.
     */
    public void addSegment(int srtIdx, int endIdx, int level,
                           Vec2D tangentSrt, Vec2D tangentEnd) {
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);

        // Create Segment3D (using current Point3D-based format)
        Segment3D segment = new Segment3D(
            srt.position, end.position, level,
            tangentSrt, tangentEnd,
            srt.pointType, end.pointType
        );
        segments.add(segment);

        // Update connection counts
        points.set(srtIdx, srt.incrementConnections());
        points.set(endIdx, end.incrementConnections());
    }

    /**
     * Remove a segment by index and decrement connection counts.
     */
    public void removeSegment(int segmentIndex) {
        Segment3D seg = segments.get(segmentIndex);
        updateConnectionCountsForSegment(seg, -1);
        segments.remove(segmentIndex);
    }

    /**
     * Get a segment by its index.
     */
    public Segment3D getSegment(int index) {
        return segments.get(index);
    }

    /**
     * Get an unmodifiable view of all segments.
     */
    public List<Segment3D> getSegments() {
        return Collections.unmodifiableList(segments);
    }

    /**
     * Get the total number of segments.
     */
    public int getSegmentCount() {
        return segments.size();
    }

    // ========== Query Operations ==========

    /**
     * Find a point by its 3D position within epsilon tolerance.
     * @return Point index, or -1 if not found
     */
    public int findPointByPosition(Point3D pos, double epsilon) {
        double epsSq = epsilon * epsilon;
        for (int i = 0; i < points.size(); i++) {
            if (points.get(i).position.distanceSquaredTo(pos) < epsSq) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Find all points within the given horizontal (XZ) distance of a position.
     * Excludes EDGE points.
     */
    public List<Integer> findPointsWithinDistance(Point3D pos, double maxDist) {
        List<Integer> result = new ArrayList<>();
        double maxDistSq = maxDist * maxDist;
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;
            double distSq = p.position.projectZ().distanceSquaredTo(pos.projectZ());
            if (distSq <= maxDistSq) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Check if two points are already connected by a segment.
     * Uses position matching since current Segment3D uses Point3D.
     */
    public boolean areConnected(int idx1, int idx2) {
        Point3D pos1 = points.get(idx1).position;
        Point3D pos2 = points.get(idx2).position;
        double epsilon = 1e-9;

        for (Segment3D seg : segments) {
            boolean match1 = seg.srt.distanceSquaredTo(pos1) < epsilon &&
                             seg.end.distanceSquaredTo(pos2) < epsilon;
            boolean match2 = seg.srt.distanceSquaredTo(pos2) < epsilon &&
                             seg.end.distanceSquaredTo(pos1) < epsilon;
            if (match1 || match2) {
                return true;
            }
        }
        return false;
    }

    /**
     * Find the closest point in this SegmentList to the given position.
     * Excludes EDGE points.
     * @return Point index, or -1 if no points exist
     */
    public int findClosestPoint(Point3D pos) {
        int bestIdx = -1;
        double bestDistSq = Double.MAX_VALUE;

        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.pointType == PointType.EDGE) continue;

            double distSq = p.position.projectZ().distanceSquaredTo(pos.projectZ());
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // ========== Utility Operations ==========

    /**
     * Mark all points with exactly 1 connection as LEAF type.
     */
    public void markLeafPoints() {
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            if (p.connections == 1 && p.pointType != PointType.EDGE) {
                points.set(i, p.withPointType(PointType.LEAF));
            }
        }
    }

    /**
     * Shift all point elevations down so minimum is at 0.
     * Used for asterism normalization.
     */
    public void normalizeElevations() {
        if (points.isEmpty()) return;

        // Find minimum elevation
        double minZ = Double.MAX_VALUE;
        for (NetworkPoint p : points) {
            minZ = Math.min(minZ, p.position.z);
        }

        if (minZ == 0) return;  // Already normalized

        // Shift all points
        for (int i = 0; i < points.size(); i++) {
            NetworkPoint p = points.get(i);
            Point3D newPos = new Point3D(p.position.x, p.position.y, p.position.z - minZ);
            points.set(i, p.withPosition(newPos));
        }

        // Also shift segments (they contain Point3D)
        List<Segment3D> shiftedSegments = new ArrayList<>();
        for (Segment3D seg : segments) {
            Point3D newSrt = new Point3D(seg.srt.x, seg.srt.y, seg.srt.z - minZ);
            Point3D newEnd = new Point3D(seg.end.x, seg.end.y, seg.end.z - minZ);
            shiftedSegments.add(new Segment3D(newSrt, newEnd, seg.level,
                                               seg.tangentSrt, seg.tangentEnd,
                                               seg.srtType, seg.endType));
        }
        segments.clear();
        segments.addAll(shiftedSegments);
    }

    /**
     * Create a copy of this SegmentList.
     */
    public SegmentList copy() {
        SegmentList copy = new SegmentList();
        copy.nextIndex = this.nextIndex;
        for (NetworkPoint p : this.points) {
            copy.points.add(p);  // NetworkPoint is immutable
        }
        for (Segment3D s : this.segments) {
            copy.segments.add(s);  // Segment3D is immutable
        }
        return copy;
    }

    /**
     * Check if this SegmentList is empty.
     */
    public boolean isEmpty() {
        return points.isEmpty();
    }

    // ========== Internal Helpers ==========

    /**
     * Update connection counts for points referenced by a segment.
     * @param delta +1 for adding, -1 for removing
     */
    private void updateConnectionCountsForSegment(Segment3D segment, int delta) {
        // Find point indices by position matching
        int srtIdx = findPointByPosition(segment.srt, 1e-9);
        int endIdx = findPointByPosition(segment.end, 1e-9);

        if (srtIdx >= 0) {
            NetworkPoint p = points.get(srtIdx);
            points.set(srtIdx, p.withConnections(Math.max(0, p.connections + delta)));
        }
        if (endIdx >= 0) {
            NetworkPoint p = points.get(endIdx);
            points.set(endIdx, p.withConnections(Math.max(0, p.connections + delta)));
        }
    }

    @Override
    public String toString() {
        return String.format("SegmentList(points=%d, segments=%d)", points.size(), segments.size());
    }
}

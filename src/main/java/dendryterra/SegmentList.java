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
                //    int idx = segList.addPoint(p.position, p.pointType, level);
                //    // 0-length segment for visualization
                //    segList.addSegment(idx, idx, level, null, null);


    /**
     * Add a segment directly using point indices.
     * Creates a Segment3D internally.
     */
    public void addSegmentWithDivisions(int srtIdx, int endIdx, int level, double maxSegmentDistance) {
        //Hardcoded jitter factor:
        double jitter = 0.5;
        //Get points, the point tangent and slope information is available from points
        NetworkPoint srt = points.get(srtIdx);
        NetworkPoint end = points.get(endIdx);

        // Placeholders for tangents for initial segment, to be divided.
        // If splines with curves are being used (useSplines && curvature > 0):
            // Derive if the point is connected as it affects tangent calculation.
            // Connections = 0 -> No connection, compute nominal from slope an flow path + twist amount, and clamp (clampTangentToSegmentDirection).
            // Connections = 1 -> Tangent already exists, to give continuity use existing tangent if the paths align (start to end connection or end to start connection), otherwise the tangent needs to be flipped / inverted.
            // Connections = 2+ -> Multiple connections, compute offset tangent from existing tangent direction. Find the connecting point with the matching point (If this is a start point, find the segments already connected to this point as a start, or vice-versa for end point), and offset the tangent angle 20 to 70 degrees on the side of the end index 
        // Otherwise, for straight segments, tangents are null.
        Vec2D tangentSrt = null;
        Vec2D tangentEnd = null;

        // Placeholder to subdivide long segments, points created here should likely be created with a for loop, but other methods could be used.
        // If distance between srt and end > maxSegmentDistance, computer the number of samples to create, create points in between using linear interpolation, and then add deterministic jitter between segments.
        double distance = srt.position.distanceTo(end.position);
        int numDivisions = (int)Math.ceil(distance / maxSegmentDistance);
        // Use shared helper for Hermite interpolation if splines are being used, else just use straight linear interpolation.
        int srtIdxSub = srtIdx;
        int endIdxSub = endIdx;
        Vec2D tangentSrtSub = null;
        Vec2D tangentEndSub = null;


        // Placeholder (likely for loop) to add each subdivided segment back into the main segment group.
        addSegment(srtIdxSub, endIdxSub, level,tangentSrtSub,tangentEndSub);
    }
    /**
     * Add a segment using new network point to known existing point in segment.
     * Automatically computes tangent and segmentation.
     */
    public void addSegmentWithDivisions(NetworkPoint srtNetPnt, int endIdx, int level, double maxSegmentDistance) {
    
        int srtIdx = addPoint(srtNetPnt);
        addSegmentWithDivisions(srtIdx, endIdx, level, maxSegmentDistance);
    }
    
    /**
     * Add a segment to with two new network points, only used for trunk initialization.
     */
    public void addSegmentWithDivisions(NetworkPoint srtNetPnt, NetworkPoint endNetPnt, int level, double maxSegmentDistance) {

        int endIdx = addPoint(endNetPnt);
        addSegmentWithDivisions(srtNetPnt, endIdx, level, maxSegmentDistance);
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

    // ========== Conversion Methods ==========

    /**
     * Convert to old format: List<Segment3D> with Point3D srt/end.
     * Used for backward compatibility with existing code that expects Point3D-based segments.
     */
    public List<Segment3D> toSegment3DList() {
        List<Segment3D> result = new ArrayList<>();

        for (Segment3D seg : segments) {
            if (seg.isPointBased()) {
                // Already has Point3D - use as-is
                result.add(seg);
            } else {
                // Index-based - resolve to Point3D
                NetworkPoint srtPt = points.get(seg.srtIdx);
                NetworkPoint endPt = points.get(seg.endIdx);

                result.add(new Segment3D(
                    srtPt.position, endPt.position, seg.level,
                    seg.tangentSrt, seg.tangentEnd,
                    srtPt.pointType, endPt.pointType
                ));
            }
        }

        return result;
    }

    /**
     * Create a SegmentList from an existing List<Segment3D>.
     * Used to import legacy segments into the new structure.
     */
    public static SegmentList fromSegment3DList(List<Segment3D> segments, int level) {
        SegmentList result = new SegmentList();

        // Build a map of positions to point indices to avoid duplicates
        Map<Long, Integer> positionToIndex = new HashMap<>();

        for (Segment3D seg : segments) {
            if (!seg.isPointBased()) {
                throw new IllegalArgumentException("Cannot import index-based segments");
            }

            // Get or create point for srt
            int srtIdx = getOrCreatePointIndex(result, positionToIndex, seg.srt, seg.srtType, level);

            // Get or create point for end
            int endIdx = getOrCreatePointIndex(result, positionToIndex, seg.end, seg.endType, level);

            // Add segment using indices
            result.addSegment(srtIdx, endIdx, level, seg.tangentSrt, seg.tangentEnd);
        }

        return result;
    }

    /**
     * Helper to get or create a point index for a position.
     */
    private static int getOrCreatePointIndex(SegmentList list, Map<Long, Integer> positionToIndex,
                                              Point3D pos, PointType type, int level) {
        // Quantize position for lookup
        long key = quantizePosition(pos);

        Integer existingIdx = positionToIndex.get(key);
        if (existingIdx != null) {
            return existingIdx;
        }

        // Create new point
        int idx = list.addPoint(pos, type, level);
        positionToIndex.put(key, idx);
        return idx;
    }

    /**
     * Quantize a position to a long key for deduplication.
     */
    private static long quantizePosition(Point3D pos) {
        // Use high precision to avoid false matches
        long qx = Math.round(pos.x * 100000);
        long qy = Math.round(pos.y * 100000);
        return (qx << 32) | (qy & 0xFFFFFFFFL);
    }

    @Override
    public String toString() {
        return String.format("SegmentList(points=%d, segments=%d)", points.size(), segments.size());
    }
}

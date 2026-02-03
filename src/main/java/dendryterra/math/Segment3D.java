package dendryterra.math;

import dendryterra.SegmentList;

/**
 * Immutable 3D line segment with elevation, resolution level, and endpoint tangents.
 * Tangents describe the curve direction at each endpoint for spline interpolation.
 * Flow direction: srt (start) -> end (end point).
 *
 * Supports two modes:
 * 1. Point3D-based: srt/end contain actual Point3D objects (legacy, for backward compatibility)
 * 2. Index-based: srtIdx/endIdx reference points in a SegmentList (new, preferred)
 *
 * During the transition period, both are supported. New code should use index-based segments.
 */
public final class Segment3D {
    // Legacy Point3D-based fields (kept for backward compatibility)
    public final Point3D srt;  // Start point of flow (null for index-based segments)
    public final Point3D end;  // End point of flow (null for index-based segments)

    // NEW: Index-based endpoint references (-1 if using Point3D mode)
    public final int srtIdx;  // Index into SegmentList.points
    public final int endIdx;  // Index into SegmentList.points

    public final int level;    // Resolution level that created this segment (0-5)

    // Tangent vectors at endpoints (for spline interpolation)
    // These are 2D direction vectors in the x,y plane
    public final Vec2D tangentSrt;  // Tangent direction at start point (may be null)
    public final Vec2D tangentEnd;  // Tangent direction at end point (may be null)

    // Endpoint type for debug visualization and tracking point provenance
    // (will be deprecated - use NetworkPoint.pointType instead for index-based segments)
    public final PointType srtType;
    public final PointType endType;

    // ========== Index-based Constructors (NEW) ==========

    /**
     * Create an index-based segment referencing points in a SegmentList.
     * This is the preferred constructor for new code.
     */
    public Segment3D(int srtIdx, int endIdx, int level,
                     Vec2D tangentSrt, Vec2D tangentEnd) {
        this.srtIdx = srtIdx;
        this.endIdx = endIdx;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        // Point3D fields are null for index-based segments
        this.srt = null;
        this.end = null;
        // Types will be derived from SegmentList for index-based segments
        this.srtType = null;
        this.endType = null;
    }

    /**
     * Create an index-based segment with explicit endpoint types.
     * (For cases where types need to be cached rather than looked up)
     */
    public Segment3D(int srtIdx, int endIdx, int level,
                     Vec2D tangentSrt, Vec2D tangentEnd,
                     PointType srtType, PointType endType) {
        this.srtIdx = srtIdx;
        this.endIdx = endIdx;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        this.srt = null;
        this.end = null;
        this.srtType = srtType;
        this.endType = endType;
    }

    // ========== Point3D-based Constructors (Legacy) ==========

    /**
     * Create a segment with specified resolution level, tangents, and endpoint types.
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd,
                     PointType srtType, PointType endType) {
        this.srt = srt;
        this.end = end;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        this.srtType = srtType;
        this.endType = endType;
        // Index fields are -1 for Point3D-based segments
        this.srtIdx = -1;
        this.endIdx = -1;
    }

    /**
     * Create a segment with specified resolution level, tangents, and boolean endpoint types.
     * @deprecated Use constructor with PointType instead
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd,
                     boolean srtIsOriginal, boolean endIsOriginal) {
        this(srt, end, level, tangentSrt, tangentEnd,
             srtIsOriginal ? PointType.ORIGINAL : PointType.KNOT,
             endIsOriginal ? PointType.ORIGINAL : PointType.KNOT);
    }

    /**
     * Create a segment with specified resolution level and tangents.
     * Defaults to both endpoints being ORIGINAL type (backward compatible).
     */
    public Segment3D(Point3D srt, Point3D end, int level, Vec2D tangentSrt, Vec2D tangentEnd) {
        this(srt, end, level, tangentSrt, tangentEnd, PointType.ORIGINAL, PointType.ORIGINAL);
    }

    /**
     * Create a segment with specified resolution level (no tangents).
     */
    public Segment3D(Point3D srt, Point3D end, int level) {
        this(srt, end, level, null, null);
    }

    /**
     * Create a segment with default level 1 (backward compatible).
     */
    public Segment3D(Point3D srt, Point3D end) {
        this(srt, end, 1, null, null);
    }

    // ========== Mode Detection ==========

    /**
     * Check if this is an index-based segment.
     */
    public boolean isIndexBased() {
        return srtIdx >= 0 && endIdx >= 0;
    }

    /**
     * Check if this is a Point3D-based segment.
     */
    public boolean isPointBased() {
        return srt != null && end != null;
    }

    // ========== Resolution Methods (for index-based segments) ==========

    /**
     * Get the start point position by resolving the index through a SegmentList.
     * For Point3D-based segments, returns srt directly.
     */
    public Point3D getSrt(SegmentList list) {
        if (isPointBased()) {
            return srt;
        }
        return list.getPoint(srtIdx).position;
    }

    /**
     * Get the end point position by resolving the index through a SegmentList.
     * For Point3D-based segments, returns end directly.
     */
    public Point3D getEnd(SegmentList list) {
        if (isPointBased()) {
            return end;
        }
        return list.getPoint(endIdx).position;
    }

    /**
     * Get the start point type by resolving through a SegmentList.
     * For Point3D-based segments, returns srtType directly.
     */
    public PointType getSrtType(SegmentList list) {
        if (isPointBased() || srtType != null) {
            return srtType;
        }
        return list.getPoint(srtIdx).pointType;
    }

    /**
     * Get the end point type by resolving through a SegmentList.
     * For Point3D-based segments, returns endType directly.
     */
    public PointType getEndType(SegmentList list) {
        if (isPointBased() || endType != null) {
            return endType;
        }
        return list.getPoint(endIdx).pointType;
    }

    // ========== Builder Methods ==========

    /**
     * Create a new segment with the same endpoints and level but different tangents.
     */
    public Segment3D withTangents(Vec2D tangentSrt, Vec2D tangentEnd) {
        if (isIndexBased()) {
            return new Segment3D(this.srtIdx, this.endIdx, this.level, tangentSrt, tangentEnd,
                                 this.srtType, this.endType);
        }
        return new Segment3D(this.srt, this.end, this.level, tangentSrt, tangentEnd,
                             this.srtType, this.endType);
    }

    /**
     * Create a new segment with specified endpoint types.
     */
    public Segment3D withEndpointTypes(PointType srtType, PointType endType) {
        if (isIndexBased()) {
            return new Segment3D(this.srtIdx, this.endIdx, this.level,
                                 this.tangentSrt, this.tangentEnd, srtType, endType);
        }
        return new Segment3D(this.srt, this.end, this.level, this.tangentSrt, this.tangentEnd,
                             srtType, endType);
    }

    /**
     * Create a new segment with specified endpoint types (boolean version for compatibility).
     * @deprecated Use withEndpointTypes(PointType, PointType) instead
     */
    public Segment3D withEndpointTypes(boolean srtIsOriginal, boolean endIsOriginal) {
        return withEndpointTypes(
            srtIsOriginal ? PointType.ORIGINAL : PointType.KNOT,
            endIsOriginal ? PointType.ORIGINAL : PointType.KNOT);
    }

    // ========== Deprecated Accessors ==========

    /**
     * Check if start point is an original point.
     * @deprecated Use srtType field directly or getSrtType(SegmentList)
     */
    public boolean isSrtOriginal() {
        return srtType == PointType.ORIGINAL;
    }

    /**
     * Check if end point is an original point.
     * @deprecated Use endType field directly or getEndType(SegmentList)
     */
    public boolean isEndOriginal() {
        return endType == PointType.ORIGINAL;
    }

    // ========== Geometry Methods ==========

    /**
     * Check if this segment has tangent information.
     */
    public boolean hasTangents() {
        return tangentSrt != null || tangentEnd != null;
    }

    /**
     * Get squared length of this segment.
     * Requires Point3D-based segment or SegmentList for index-based.
     */
    public double lengthSquared() {
        if (!isPointBased()) {
            throw new IllegalStateException("Use lengthSquared(SegmentList) for index-based segments");
        }
        return srt.distanceSquaredTo(end);
    }

    /**
     * Get squared length of this segment, resolving indices if needed.
     */
    public double lengthSquared(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return s.distanceSquaredTo(e);
    }

    /**
     * Get length of this segment.
     * Requires Point3D-based segment or SegmentList for index-based.
     */
    public double length() {
        if (!isPointBased()) {
            throw new IllegalStateException("Use length(SegmentList) for index-based segments");
        }
        return srt.distanceTo(end);
    }

    /**
     * Get length of this segment, resolving indices if needed.
     */
    public double length(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return s.distanceTo(e);
    }

    /**
     * Get midpoint of this segment.
     * Requires Point3D-based segment or SegmentList for index-based.
     */
    public Point3D midpoint() {
        if (!isPointBased()) {
            throw new IllegalStateException("Use midpoint(SegmentList) for index-based segments");
        }
        return new Point3D(
            (srt.x + end.x) / 2.0,
            (srt.y + end.y) / 2.0,
            (srt.z + end.z) / 2.0
        );
    }

    /**
     * Get midpoint of this segment, resolving indices if needed.
     */
    public Point3D midpoint(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return new Point3D(
            (s.x + e.x) / 2.0,
            (s.y + e.y) / 2.0,
            (s.z + e.z) / 2.0
        );
    }

    /**
     * Interpolate along the segment.
     * @param t parameter from 0 (at srt) to 1 (at end)
     * Requires Point3D-based segment or SegmentList for index-based.
     */
    public Point3D lerp(double t) {
        if (!isPointBased()) {
            throw new IllegalStateException("Use lerp(SegmentList, t) for index-based segments");
        }
        return Point3D.lerp(srt, end, t);
    }

    /**
     * Interpolate along the segment, resolving indices if needed.
     * @param t parameter from 0 (at srt) to 1 (at end)
     */
    public Point3D lerp(SegmentList list, double t) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return Point3D.lerp(s, e, t);
    }

    /**
     * Project to 2D by dropping z coordinates.
     * Requires Point3D-based segment.
     */
    public Segment2D projectZ() {
        if (!isPointBased()) {
            throw new IllegalStateException("Use projectZ(SegmentList) for index-based segments");
        }
        return new Segment2D(srt.projectZ(), end.projectZ());
    }

    /**
     * Project to 2D by dropping z coordinates, resolving indices if needed.
     */
    public Segment2D projectZ(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return new Segment2D(s.projectZ(), e.projectZ());
    }

    /**
     * Subdivide this segment into n equal segments using linear interpolation.
     * Note: For B-spline subdivision with jitter, use DendrySampler.subdivideSegment() instead.
     * Requires Point3D-based segment.
     *
     * @param n number of segments to create
     * @return array of n segments
     */
    public Segment3D[] subdivide(int n) {
        if (!isPointBased()) {
            throw new IllegalStateException("subdivide() not supported for index-based segments");
        }
        if (n <= 0) throw new IllegalArgumentException("n must be positive");

        Segment3D[] segments = new Segment3D[n];
        Point3D prev = srt;

        for (int i = 0; i < n; i++) {
            double t = (double)(i + 1) / n;
            Point3D next = (i == n - 1) ? end : Point3D.lerp(srt, end, t);

            // First segment: srt keeps original type, interior points become KNOT
            // Last segment: end keeps original type
            PointType prevType = (i == 0) ? this.srtType : PointType.KNOT;
            PointType nextType = (i == n - 1) ? this.endType : PointType.KNOT;

            segments[i] = new Segment3D(prev, next, this.level, null, null, prevType, nextType);
            prev = next;
        }

        return segments;
    }

    /**
     * Get n-1 interior points that divide this segment into n equal parts.
     * Requires Point3D-based segment.
     */
    public Point3D[] subdivideInPoints(int n) {
        if (!isPointBased()) {
            throw new IllegalStateException("subdivideInPoints() not supported for index-based segments");
        }
        Point3D[] points = new Point3D[n - 1];
        for (int i = 0; i < n - 1; i++) {
            double t = (double)(i + 1) / n;
            points[i] = Point3D.lerp(srt, end, t);
        }
        return points;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Segment3D(");
        if (isIndexBased()) {
            sb.append("idx:").append(srtIdx).append("->").append(endIdx);
        } else {
            sb.append(srt).append(" -> ").append(end);
        }
        sb.append(", level=").append(level);
        if (hasTangents()) {
            sb.append(", tangentSrt=").append(tangentSrt);
            sb.append(", tangentEnd=").append(tangentEnd);
        }
        if (srtType != null) {
            sb.append(", srtType=").append(srtType);
        }
        if (endType != null) {
            sb.append(", endType=").append(endType);
        }
        sb.append(")");
        return sb.toString();
    }
}

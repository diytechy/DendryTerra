package dendryterra.math;

import dendryterra.SegmentList;

/**
 * Index-based 3D line segment for storage in SegmentList.
 * References endpoints by index to reduce memory consumption.
 * Immutable.
 */
public final class SegmentIdx {
    // Index-based endpoint references
    public final int srtIdx;  // Index into SegmentList.points
    public final int endIdx;  // Index into SegmentList.points

    public final int level;    // Resolution level that created this segment (0-5)

    // Tangent vectors at endpoints (for spline interpolation)
    // These are 2D direction vectors in the x,y plane
    public final Vec2D tangentSrt;  // Tangent direction at start point (may be null)
    public final Vec2D tangentEnd;  // Tangent direction at end point (may be null)

    // Endpoint type for debug visualization and tracking point provenance
    // (optional - can be null and derived from NetworkPoint.pointType instead)
    public final PointType srtType;
    public final PointType endType;

    // ========== Constructors ==========

    /**
     * Create an index-based segment referencing points in a SegmentList.
     * This is the primary constructor.
     */
    public SegmentIdx(int srtIdx, int endIdx, int level,
                     Vec2D tangentSrt, Vec2D tangentEnd) {
        this.srtIdx = srtIdx;
        this.endIdx = endIdx;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        this.srtType = null;
        this.endType = null;
    }

    /**
     * Create an index-based segment with explicit endpoint types.
     * (For cases where types need to be cached rather than looked up)
     */
    public SegmentIdx(int srtIdx, int endIdx, int level,
                     Vec2D tangentSrt, Vec2D tangentEnd,
                     PointType srtType, PointType endType) {
        this.srtIdx = srtIdx;
        this.endIdx = endIdx;
        this.level = level;
        this.tangentSrt = tangentSrt;
        this.tangentEnd = tangentEnd;
        this.srtType = srtType;
        this.endType = endType;
    }

    // ========== Resolution Methods ==========

    /**
     * Get the start point position by resolving the index through a SegmentList.
     */
    public Point3D getSrt(SegmentList list) {
        return list.getPoint(srtIdx).position;
    }

    /**
     * Get the end point position by resolving the index through a SegmentList.
     */
    public Point3D getEnd(SegmentList list) {
        return list.getPoint(endIdx).position;
    }

    /**
     * Get the start point type by resolving through a SegmentList.
     */
    public PointType getSrtType(SegmentList list) {
        if (srtType != null) {
            return srtType;
        }
        return list.getPoint(srtIdx).pointType;
    }

    /**
     * Get the end point type by resolving through a SegmentList.
     */
    public PointType getEndType(SegmentList list) {
        if (endType != null) {
            return endType;
        }
        return list.getPoint(endIdx).pointType;
    }

    /**
     * Resolve to a lightweight Segment3D for evaluation.
     */
    public Segment3D resolve(SegmentList list) {
        return new Segment3D(
            getSrt(list),
            getEnd(list),
            tangentSrt,
            tangentEnd
        );
    }

    // ========== Builder Methods ==========

    /**
     * Create a new segment with the same endpoints and level but different tangents.
     */
    public SegmentIdx withTangents(Vec2D tangentSrt, Vec2D tangentEnd) {
        return new SegmentIdx(this.srtIdx, this.endIdx, this.level, tangentSrt, tangentEnd,
                             this.srtType, this.endType);
    }

    /**
     * Create a new segment with specified endpoint types.
     */
    public SegmentIdx withEndpointTypes(PointType srtType, PointType endType) {
        return new SegmentIdx(this.srtIdx, this.endIdx, this.level,
                             this.tangentSrt, this.tangentEnd, srtType, endType);
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
     */
    public double lengthSquared(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return s.distanceSquaredTo(e);
    }

    /**
     * Get length of this segment.
     */
    public double length(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return s.distanceTo(e);
    }

    /**
     * Get midpoint of this segment.
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
     */
    public Point3D lerp(SegmentList list, double t) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return Point3D.lerp(s, e, t);
    }

    /**
     * Project to 2D by dropping z coordinates.
     */
    public Segment2D projectZ(SegmentList list) {
        Point3D s = getSrt(list);
        Point3D e = getEnd(list);
        return new Segment2D(s.projectZ(), e.projectZ());
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("SegmentIdx(");
        sb.append("idx:").append(srtIdx).append("->").append(endIdx);
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

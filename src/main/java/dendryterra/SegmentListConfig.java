package dendryterra;

/**
 * Configuration class for SegmentList parameters that don't change frequently.
 * This avoids passing the same parameters through multiple function calls.
 */
public class SegmentListConfig {
    public long salt = 12345;
    public boolean useSplines = false;
    public double curvature = 0.0;
    public double curvatureFalloff = 0.0;
    public double tangentStrength = 1.0;
    public double tangentAngle = 0.5;
    public double maxSegmentDistance = 1.0;
    
    public SegmentListConfig() {}
    
    public SegmentListConfig(long salt) {
        this.salt = salt;
    }
    
    public SegmentListConfig withSalt(long salt) {
        this.salt = salt;
        return this;
    }
    
    public SegmentListConfig withSplines(boolean useSplines) {
        this.useSplines = useSplines;
        return this;
    }
    
    public SegmentListConfig withCurvature(double curvature) {
        this.curvature = curvature;
        return this;
    }
    
    public SegmentListConfig withTangentStrength(double tangentStrength) {
        this.tangentStrength = tangentStrength;
        return this;
    }
    
    public SegmentListConfig withMaxSegmentDistance(double maxSegmentDistance) {
        this.maxSegmentDistance = maxSegmentDistance;
        return this;
    }
}

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
    public double maxTwistAngle = 1.0; //  +/- degrees in radians when slope is 0, affects random rotation of origin points.
    public double maxIntermediateTwistAngle = 0.2; // +/- degrees in radians, for intermediate points, reduced by jitter magnitude
    public double SlopeWithoutTwist = 0.5;
    
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
    
    public SegmentListConfig withMaxTwistAngle(double maxTwistAngle) {
        this.maxTwistAngle = maxTwistAngle;
        return this;
    }

    public SegmentListConfig withMaxIntermediateTwistAngle(double maxIntermediateTwistAngle) {
        this.maxIntermediateTwistAngle = maxIntermediateTwistAngle;
        return this;
    }

    public SegmentListConfig withSlopeWithoutTwist(double SlopeWithoutTwist) {
        this.SlopeWithoutTwist = SlopeWithoutTwist;
        return this;
    }
}

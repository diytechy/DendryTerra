package dendryterra;

/**
 * A 256x256 grid of blocks for optimized river distance/elevation caching.
 * Each block represents a pixel-cache sized square and stores normalized
 * elevation and distance values as UInt8 (0-255).
 *
 * Memory usage: 256*256*2 bytes + overhead ≈ 132 KB per chunk.
 */
public class BigChunk {
    /** Grid X coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginX;

    /** Grid Y coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginY;

    /** 256x256 grid of blocks */
    public final BigChunkBlock[][] blocks;

    /** Whether this chunk has been fully computed (volatile for thread-safe double-check) */
    public volatile boolean computed;

    /** LRU counter for cache eviction */
    public int lruCounter;

    /**
     * Create a new BigChunk at the specified grid coordinates.
     * Grid coordinates are in normalized space (sampler coordinates / gridsize).
     * All blocks are initialized with elevation=255, distance=255.
     */
    public BigChunk(double gridOriginX, double gridOriginY) {
        this.gridOriginX = gridOriginX;
        this.gridOriginY = gridOriginY;
        this.blocks = new BigChunkBlock[256][256];

        // Initialize all blocks
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                blocks[i][j] = new BigChunkBlock();
            }
        }

        this.computed = false;
        this.lruCounter = 0;
    }

    /**
     * Get the block at the specified grid coordinates within this chunk.
     * @param gridX X coordinate within chunk (0-255)
     * @param gridY Y coordinate within chunk (0-255)
     * @return The block at that position
     */
    public BigChunkBlock getBlock(int gridX, int gridY) {
        return blocks[gridX][gridY];
    }

    /**
     * A single block within a BigChunk, storing normalized elevation and distance.
     */
    public static class BigChunkBlock {
        /** Normalized elevation (0-255) */
        public byte elevation;

        /** Normalized distance (0-255) */
        public byte distance;

        /**
         * Create a new block with default values:
         * elevation = 255 (unset), distance = 255 (maximum)
         */
        public BigChunkBlock() {
            this.elevation = (byte) 255;  // Max elevation initially (unset marker)
            this.distance = (byte) 255;   // Max distance initially
        }

        /**
         * Get elevation as unsigned value (0-255).
         */
        public int getElevationUnsigned() {
            return Byte.toUnsignedInt(elevation);
        }

        /**
         * Get distance as unsigned value (0-255).
         */
        public int getDistanceUnsigned() {
            return Byte.toUnsignedInt(distance);
        }

        /**
         * Set elevation from unsigned value (0-255).
         */
        public void setElevationUnsigned(int value) {
            this.elevation = (byte) Math.min(255, Math.max(0, value));
        }

        /**
         * Set distance from unsigned value (0-255).
         */
        public void setDistanceUnsigned(int value) {
            this.distance = (byte) Math.min(255, Math.max(0, value));
        }
    }
}

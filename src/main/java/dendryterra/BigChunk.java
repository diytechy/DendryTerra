package dendryterra;

/**
 * A 256x256 grid of blocks for optimized river distance/elevation caching.
 * Each block represents a pixel-cache sized square and stores normalized
 * elevation and distance values as UInt8 (0-255).
 *
 * Also contains a 4x4 far-distance cache for coarse directional distance queries.
 *
 * Memory usage: 256*256*3 bytes + 4*4*4 bytes + overhead ≈ 196 KB per chunk.
 */
public class BigChunk {
    /** Grid X coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginX;

    /** Grid Y coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginY;

    /** 256x256 grid of blocks */
    public final BigChunkBlock[][] blocks;

    /** 4x4 far-distance cache for coarse directional distance queries */
    public final FarCacheCell[][] farCache;

    /** Whether this chunk has been fully computed (volatile for thread-safe double-check) */
    public volatile boolean computed;

    /** LRU counter for cache eviction */
    public int lruCounter;

    /**
     * Create a new BigChunk at the specified grid coordinates.
     * Grid coordinates are in normalized space (sampler coordinates / gridsize).
     * All blocks are initialized with elevation=255, distance=255.
     * All far cache cells are initialized with distances=255 (max/unset).
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

        // Initialize 4x4 far cache
        this.farCache = new FarCacheCell[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                farCache[i][j] = new FarCacheCell();
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
     * A single block within a BigChunk, storing normalized elevation, distance,
     * and packed level/distance-to-change nibbles.
     */
    public static class BigChunkBlock {
        /** Normalized elevation (0-255) */
        public byte elevation;

        /** Normalized distance (0-255) */
        public byte distance;

        /**
         * Packed level and distance-to-change nibbles.
         * High 4 bits: segment level (0-14, 15 = not available)
         * Low 4 bits: quantized distance to next elevation change (0-14, 15 = undefined/too far)
         */
        public byte levelAndDistChange;

        /**
         * Create a new block with default values:
         * elevation = 255 (unset), distance = 255 (maximum),
         * levelAndDistChange = 0xFF (both nibbles = 15 = unset)
         */
        public BigChunkBlock() {
            this.elevation = (byte) 255;  // Max elevation initially (unset marker)
            this.distance = (byte) 255;   // Max distance initially
            this.levelAndDistChange = (byte) 0xFF;  // Both nibbles = 15 (unset)
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

        /** Get segment level from high nibble (0-15, 15 = not available). */
        public int getLevelNibble() {
            return (levelAndDistChange >> 4) & 0x0F;
        }

        /** Set segment level in high nibble (0-15), preserving low nibble. */
        public void setLevelNibble(int value) {
            int clamped = Math.min(15, Math.max(0, value));
            this.levelAndDistChange = (byte) ((clamped << 4) | (this.levelAndDistChange & 0x0F));
        }

        /** Get quantized distance-to-change from low nibble (0-15, 15 = undefined/too far). */
        public int getDistChangeNibble() {
            return this.levelAndDistChange & 0x0F;
        }

        /** Set quantized distance-to-change in low nibble (0-15), preserving high nibble. */
        public void setDistChangeNibble(int value) {
            int clamped = Math.min(15, Math.max(0, value));
            this.levelAndDistChange = (byte) ((this.levelAndDistChange & 0xF0) | clamped);
        }
    }

    /**
     * A cell in the 8x8 far-distance cache.
     * Stores the closest Euclidean distance to any segment (UInt8, quantized)
     * and a normal angle (UInt8, 0-255 maps to 0-2π) pointing away from the nearest river.
     * 255 distance = maxDist (no segment found), 0 = segment at cell center.
     */
    public static class FarCacheCell {
        public byte distance;  // Quantized closest distance to any segment
        public byte normal;    // Quantized angle (0-255 → 0-2π) pointing AWAY from nearest river

        public FarCacheCell() {
            distance = (byte) 255;  // max distance (unset)
            normal = 0;
        }

        public int getDistanceUnsigned() { return Byte.toUnsignedInt(distance); }
        public void setDistanceUnsigned(int v) { distance = (byte) Math.min(255, Math.max(0, v)); }

        public int getNormalUnsigned() { return Byte.toUnsignedInt(normal); }
        public void setNormalUnsigned(int v) { normal = (byte) Math.min(255, Math.max(0, v)); }

        /** Decode normal to radians (0-2π). */
        public double getNormalRadians() { return Byte.toUnsignedInt(normal) / 255.0 * 2.0 * Math.PI; }
    }
}

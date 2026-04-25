package dendryterra;

/**
 * A 256x256 grid of blocks for optimized river distance/elevation caching,
 * combined with a 64x64 far-distance cache for coarse distance queries.
 *
 * The two caches are computed and allocated lazily and independently:
 *   - farCache  (64x64, 8 KB):   allocated at construction; computed on first far-type query.
 *   - blocks (256x256, 196 KB): allocated and computed on first main-block query.
 *
 * This allows PIXEL_RIVER_FAR* queries to avoid the expensive 256x256 rasterization,
 * and PIXEL_RIVER queries to avoid filling the far cache when y > 0.
 *
 * Thread safety: both farCacheComputed and blocksComputed are volatile; each computation
 * is guarded by a double-checked synchronized block on the BigChunk instance.
 */
public class BigChunk {
    /** Grid X coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginX;

    /** Grid Y coordinate of this chunk's origin (in normalized grid space) */
    public final double gridOriginY;

    /**
     * 256x256 grid of high-resolution blocks.
     * Null until the first main-block query triggers lazy allocation and computation.
     * Each block covers cachepixels × cachepixels world units.
     */
    public BigChunkBlock[][] blocks;

    /**
     * 64x64 coarse far-distance cache.
     * Each cell covers 4 × 4 cachepixel blocks (4 cachepixels per side).
     * Array size = 256 / 4 = 64 per dimension.
     * Allocated at construction; computed lazily on first far-type query.
     */
    public final FarCacheCell[][] farCache;

    /** Set to true once the 64x64 far cache has been fully computed. */
    public volatile boolean farCacheComputed;

    /** Set to true once the 256x256 block grid has been fully computed. */
    public volatile boolean blocksComputed;

    /** LRU counter for cache eviction */
    public int lruCounter;

    /**
     * Create a new BigChunk at the specified grid coordinates.
     * The 64x64 farCache is allocated immediately (8 KB, all cells at max/unset distance).
     * The 256x256 blocks grid is NOT allocated here; it is allocated lazily when first needed.
     */
    public BigChunk(double gridOriginX, double gridOriginY) {
        this.gridOriginX = gridOriginX;
        this.gridOriginY = gridOriginY;
        this.blocks = null;  // allocated lazily

        // Initialize 64x64 far cache (4-pixel cell size → 256/4 = 64 cells per axis)
        this.farCache = new FarCacheCell[64][64];
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                farCache[i][j] = new FarCacheCell();
            }
        }

        this.farCacheComputed = false;
        this.blocksComputed = false;
        this.lruCounter = 0;
    }

    /**
     * Allocate the 256x256 blocks grid, initializing every block to default (unset) values.
     * Must be called inside a synchronized block before computeMainBlockPhases is invoked.
     */
    public void allocateBlocks() {
        BigChunkBlock[][] b = new BigChunkBlock[256][256];
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                b[i][j] = new BigChunkBlock();
            }
        }
        this.blocks = b;
    }

    /**
     * Get the block at the specified coordinates within this chunk.
     * Requires that blocksComputed == true (blocks must have been allocated and computed).
     * @param gridX X index within chunk (0-255)
     * @param gridY Y index within chunk (0-255)
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

        /** Get elevation as unsigned value (0-255). */
        public int getElevationUnsigned() {
            return Byte.toUnsignedInt(elevation);
        }

        /** Get distance as unsigned value (0-255). */
        public int getDistanceUnsigned() {
            return Byte.toUnsignedInt(distance);
        }

        /** Set elevation from unsigned value (0-255). */
        public void setElevationUnsigned(int value) {
            this.elevation = (byte) Math.min(255, Math.max(0, value));
        }

        /** Set distance from unsigned value (0-255). */
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
     * A cell in the 64x64 far-distance cache.
     * Each cell covers 4x4 cachepixel blocks (4 cachepixels per axis).
     * Stores the closest Euclidean distance to any segment (UInt8, quantized in units of
     * cachepixels) and a normal angle (UInt8, 0-255 → 0-2π) pointing away from the river.
     * distance=255 means no segment found within range; distance=0 means segment at cell center.
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

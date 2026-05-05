package dendryterra;

import java.util.Arrays;

/**
 * Storage for one BigChunk, backed by flat byte arrays instead of individual Java objects.
 *
 * Object-per-block layout (old): 65,536 BigChunkBlock objects × 16 bytes = ~1,024 KB for blocks.
 * Flat array layout (new): 3 byte[] of 65,536 bytes each = 192 KB for blocks — a 5x reduction.
 *
 * Block arrays (lazily allocated on first main-block query, null until then):
 *   blockElevation[by * 256 + bx]      UInt8 elevation (0 = min, 255 = max/unset)
 *   blockDistance[by * 256 + bx]       UInt8 distance  (0-127 inside, 128-255 outside, 255 = unset)
 *   blockLevelDistChange[by * 256 + bx] packed nibbles: high=level(0-14,15=unset), low=distChange
 *
 * Far cache arrays (always allocated, filled lazily on first far-type query):
 *   farDistance[ci * 64 + cj]  UInt8 quantized distance to nearest segment (255 = unset)
 *   farNormal[ci * 64 + cj]    UInt8 angle (0-255 → 0-2π) pointing away from nearest river
 *
 * Memory per chunk:
 *   Far only (PIXEL_RIVER_FAR):  2 × 4,096 bytes =  8 KB
 *   Full (PIXEL_RIVER):          3 × 65,536 bytes + 8 KB = 200 KB
 */
public class BigChunk {
    public static final int BLOCK_GRID = 256;
    public static final int FAR_GRID   = 64;

    /** Grid origin of this chunk in normalized grid space. */
    public final double gridOriginX;
    public final double gridOriginY;

    /** Flat block arrays — null until allocateBlocks() is called. */
    public byte[] blockElevation;
    public byte[] blockDistance;
    public byte[] blockLevelDistChange;

    /** Flat far-cache arrays — always allocated at construction. */
    public final byte[] farDistance;
    public final byte[] farNormal;

    public volatile boolean farCacheComputed;
    public volatile boolean blocksComputed;
    public int lruCounter;

    public BigChunk(double gridOriginX, double gridOriginY) {
        this.gridOriginX = gridOriginX;
        this.gridOriginY = gridOriginY;

        // Far cache always allocated; blocks are lazy
        int farSize = FAR_GRID * FAR_GRID;
        this.farDistance = new byte[farSize];
        this.farNormal   = new byte[farSize];
        Arrays.fill(farDistance, (byte) 255); // 255 = unset (max distance)

        this.farCacheComputed = false;
        this.blocksComputed   = false;
        this.lruCounter       = 0;
    }

    /**
     * Allocate and initialise the three block arrays.
     * Must be called inside a synchronized block before computeMainBlockPhases.
     */
    public void allocateBlocks() {
        int n = BLOCK_GRID * BLOCK_GRID;
        byte[] elev = new byte[n];
        byte[] dist = new byte[n];
        byte[] lvl  = new byte[n];
        Arrays.fill(elev, (byte) 255);  // 255 = unset elevation
        Arrays.fill(dist, (byte) 255);  // 255 = max distance (unset)
        Arrays.fill(lvl,  (byte) 0xFF); // both nibbles = 15 = unset
        this.blockElevation        = elev;
        this.blockDistance         = dist;
        this.blockLevelDistChange  = lvl;
    }

    // -------------------------------------------------------------------------
    // Block accessors  (bx = 0-255, by = 0-255, index = by*256 + bx)
    // -------------------------------------------------------------------------

    public int getBlockDistance(int bx, int by) {
        return Byte.toUnsignedInt(blockDistance[by * BLOCK_GRID + bx]);
    }
    public void setBlockDistance(int bx, int by, int value) {
        blockDistance[by * BLOCK_GRID + bx] = (byte) Math.min(255, Math.max(0, value));
    }

    public int getBlockElevation(int bx, int by) {
        return Byte.toUnsignedInt(blockElevation[by * BLOCK_GRID + bx]);
    }
    public void setBlockElevation(int bx, int by, int value) {
        blockElevation[by * BLOCK_GRID + bx] = (byte) Math.min(255, Math.max(0, value));
    }

    public int getBlockLevelNibble(int bx, int by) {
        return (blockLevelDistChange[by * BLOCK_GRID + bx] >> 4) & 0x0F;
    }
    public void setBlockLevelNibble(int bx, int by, int value) {
        int idx = by * BLOCK_GRID + bx;
        int clamped = Math.min(15, Math.max(0, value));
        blockLevelDistChange[idx] = (byte) ((clamped << 4) | (blockLevelDistChange[idx] & 0x0F));
    }

    public int getBlockDistChangeNibble(int bx, int by) {
        return blockLevelDistChange[by * BLOCK_GRID + bx] & 0x0F;
    }
    public void setBlockDistChangeNibble(int bx, int by, int value) {
        int idx = by * BLOCK_GRID + bx;
        int clamped = Math.min(15, Math.max(0, value));
        blockLevelDistChange[idx] = (byte) ((blockLevelDistChange[idx] & 0xF0) | clamped);
    }

    // -------------------------------------------------------------------------
    // Far-cache accessors  (ci = row 0-63, cj = col 0-63, index = ci*64 + cj)
    // -------------------------------------------------------------------------

    public int getFarDistance(int ci, int cj) {
        return Byte.toUnsignedInt(farDistance[ci * FAR_GRID + cj]);
    }
    public void setFarDistance(int ci, int cj, int value) {
        farDistance[ci * FAR_GRID + cj] = (byte) Math.min(255, Math.max(0, value));
    }

    public int getFarNormal(int ci, int cj) {
        return Byte.toUnsignedInt(farNormal[ci * FAR_GRID + cj]);
    }
    public void setFarNormal(int ci, int cj, int value) {
        farNormal[ci * FAR_GRID + cj] = (byte) Math.min(255, Math.max(0, value));
    }

    /** Decode stored normal byte to radians (0-2π). */
    public double getFarNormalRadians(int ci, int cj) {
        return Byte.toUnsignedInt(farNormal[ci * FAR_GRID + cj]) / 255.0 * 2.0 * Math.PI;
    }
}

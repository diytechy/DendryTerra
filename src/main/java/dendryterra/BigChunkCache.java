package dendryterra;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe LRU cache for BigChunk instances.
 *
 * Memory estimates per chunk (JVM 64-bit with compressed refs):
 *   farCache[64][64]: 4,096 FarCacheCell objects × ~16 bytes = ~64 KB
 *   blocks[256][256]: 65,536 BigChunkBlock objects × ~16 bytes = ~1,024 KB (only when allocated)
 *
 * PIXEL_RIVER_FAR chunks have blocks=null, so they consume ~64 KB each.
 * PIXEL_RIVER chunks have blocks allocated, so they consume ~1,088 KB each.
 * The cache limit is enforced by chunk count, not exact byte tracking.
 */
public class BigChunkCache {
    // Estimated JVM memory per chunk layer (object array of N items × ~16 bytes/object)
    private static final long FAR_CACHE_BYTES  =  64L * 1024; // 4096 FarCacheCell objects
    private static final long BLOCKS_BYTES     = 1024L * 1024; // 65536 BigChunkBlock objects

    // Cap at ~150 FAR-only chunks (~9.6 MB) or ~9 full chunks (~9.8 MB)
    private static final int MAX_CHUNKS = 150;

    /** Map from chunk coordinates to BigChunk instances */
    private final ConcurrentHashMap<ChunkKey, BigChunk> cache;

    /** LRU counter for cache eviction (incremented on each access) */
    private final AtomicInteger lruCounter;

    /** Running estimate of total allocated memory across all cached chunks */
    private final AtomicLong estimatedMemoryBytes = new AtomicLong(0);

    /** Count of chunks evicted since creation */
    private final AtomicLong evictions = new AtomicLong(0);

    public BigChunkCache() {
        this.cache = new ConcurrentHashMap<>();
        this.lruCounter = new AtomicInteger(0);
    }

    /** Estimated JVM heap bytes consumed by one BigChunk based on its allocation state. */
    private static long estimateChunkBytes(BigChunk chunk) {
        return FAR_CACHE_BYTES + (chunk.blocks != null ? BLOCKS_BYTES : 0);
    }

    /**
     * Get or create a BigChunk at the specified chunk coordinates.
     * Thread-safe: uses computeIfAbsent for atomic insertion and synchronized eviction.
     * @param chunkX Integer chunk X coordinate
     * @param chunkY Integer chunk Y coordinate
     * @param gridOriginX Grid X coordinate of chunk origin (normalized space)
     * @param gridOriginY Grid Y coordinate of chunk origin (normalized space)
     * @return The BigChunk (may be newly created or from cache)
     */
    public BigChunk getOrCreate(int chunkX, int chunkY, double gridOriginX, double gridOriginY) {
        ChunkKey key = new ChunkKey(chunkX, chunkY);
        BigChunk existing = cache.get(key);

        if (existing != null) {
            existing.lruCounter = lruCounter.incrementAndGet();
            return existing;
        }

        // Synchronize creation + eviction to prevent race conditions
        synchronized (this) {
            // Double-check after acquiring lock
            existing = cache.get(key);
            if (existing != null) {
                existing.lruCounter = lruCounter.incrementAndGet();
                return existing;
            }

            BigChunk chunk = new BigChunk(gridOriginX, gridOriginY);
            chunk.lruCounter = lruCounter.incrementAndGet();

            if (cache.size() >= MAX_CHUNKS) {
                evictOldest();
            }

            cache.put(key, chunk);
            // Note: chunk.blocks is null at this point; memory is updated lazily.
            // We add only the far-cache cost now; block cost is added when blocks are allocated.
            estimatedMemoryBytes.addAndGet(FAR_CACHE_BYTES);
            return chunk;
        }
    }

    /**
     * Evict the least recently used chunk from the cache.
     * Must be called while holding the synchronized lock.
     */
    private void evictOldest() {
        if (cache.isEmpty()) {
            return;
        }

        ChunkKey oldestKey = null;
        int oldestLru = Integer.MAX_VALUE;

        for (Map.Entry<ChunkKey, BigChunk> entry : cache.entrySet()) {
            if (entry.getValue().lruCounter < oldestLru) {
                oldestLru = entry.getValue().lruCounter;
                oldestKey = entry.getKey();
            }
        }

        if (oldestKey != null) {
            BigChunk removed = cache.remove(oldestKey);
            if (removed != null) {
                estimatedMemoryBytes.addAndGet(-estimateChunkBytes(removed));
                evictions.incrementAndGet();
            }
        }
    }

    /**
     * Notify the cache that a chunk's blocks array has just been allocated.
     * Call this after chunk.allocateBlocks() so the memory estimate stays accurate.
     */
    public void notifyBlocksAllocated() {
        estimatedMemoryBytes.addAndGet(BLOCKS_BYTES);
    }

    /**
     * Return a stats string reporting chunk count, estimated heap usage, and evictions.
     * Memory figures are estimates based on JVM object sizing (64-bit, compressed refs).
     */
    public String getStats() {
        int chunks = cache.size();
        long memKB = estimatedMemoryBytes.get() / 1024;
        long evict = evictions.get();
        // Count how many chunks have blocks allocated (PIXEL_RIVER) vs far-only (PIXEL_RIVER_FAR)
        int withBlocks = 0;
        for (BigChunk c : cache.values()) if (c.blocks != null) withBlocks++;
        return String.format("chunks=%d (%d w/ blocks, %d far-only), est. heap=%dKB, evictions=%d",
                chunks, withBlocks, chunks - withBlocks, memKB, evict);
    }

    /**
     * Key for identifying chunks by integer coordinates.
     */
    private static class ChunkKey {
        final int chunkX;
        final int chunkY;

        ChunkKey(int chunkX, int chunkY) {
            this.chunkX = chunkX;
            this.chunkY = chunkY;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof ChunkKey)) return false;
            ChunkKey key = (ChunkKey) o;
            return chunkX == key.chunkX && chunkY == key.chunkY;
        }

        @Override
        public int hashCode() {
            return Objects.hash(chunkX, chunkY);
        }

        @Override
        public String toString() {
            return String.format("ChunkKey(%d, %d)", chunkX, chunkY);
        }
    }
}

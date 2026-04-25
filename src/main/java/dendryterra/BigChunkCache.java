package dendryterra;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Thread-safe LRU cache for BigChunk instances with memory-based eviction.
 *
 * Memory estimates per chunk (JVM 64-bit, compressed refs, ~16 bytes/object):
 *   farCache[64][64]: 4,096 FarCacheCell objects  → ~64 KB  (always allocated)
 *   blocks[256][256]: 65,536 BigChunkBlock objects → ~1,024 KB  (allocated lazily)
 *
 * PIXEL_RIVER_FAR chunks keep blocks=null: ~64 KB each → ~312 FAR-only chunks per 20 MB.
 * PIXEL_RIVER chunks allocate blocks: ~1,088 KB each → ~18 full chunks per 20 MB.
 * Eviction is triggered whenever estimated total exceeds MAX_MEMORY — both at chunk
 * creation (far-cache cost) and when blocks are lazily allocated (block cost).
 */
public class BigChunkCache {
    private static final long MAX_MEMORY = 20L * 1024 * 1024; // 20 MB hard limit

    // Estimated JVM heap bytes per chunk layer (object arrays of N items × ~16 bytes/item)
    static final long FAR_CACHE_BYTES = 64L  * 1024; // 4,096 FarCacheCell objects
    static final long BLOCKS_BYTES    = 1024L * 1024; // 65,536 BigChunkBlock objects

    /** Map from chunk coordinates to BigChunk instances */
    private final ConcurrentHashMap<ChunkKey, BigChunk> cache;

    /** LRU counter for cache eviction (incremented on each access) */
    private final AtomicInteger lruCounter;

    /** Running estimate of total heap bytes used by all cached chunks. */
    private long estimatedMemoryBytes = 0;

    /** Count of chunks evicted since creation. */
    private long evictions = 0;

    public BigChunkCache() {
        this.cache = new ConcurrentHashMap<>();
        this.lruCounter = new AtomicInteger(0);
    }

    /** Estimated heap bytes for one BigChunk based on what is currently allocated. */
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

            // Evict until adding this chunk's far-cache cost fits within MAX_MEMORY
            while (estimatedMemoryBytes + FAR_CACHE_BYTES > MAX_MEMORY && !cache.isEmpty()) {
                evictOldest();
            }

            cache.put(key, chunk);
            estimatedMemoryBytes += FAR_CACHE_BYTES;
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
                estimatedMemoryBytes -= estimateChunkBytes(removed);
                evictions++;
            }
        }
    }

    /**
     * Notify the cache that a chunk's blocks array has just been lazily allocated.
     * Triggers eviction of other chunks if adding the block cost exceeds MAX_MEMORY.
     * The chunk whose blocks were just allocated is never itself evicted here.
     */
    public synchronized void notifyBlocksAllocated(BigChunk allocatedChunk) {
        estimatedMemoryBytes += BLOCKS_BYTES;
        // Evict older chunks (not the one we just allocated) until within limit
        while (estimatedMemoryBytes > MAX_MEMORY && cache.size() > 1) {
            evictOldestExcept(allocatedChunk);
        }
    }

    /**
     * Evict the LRU chunk from the cache, skipping the given chunk.
     * Must be called while holding the synchronized lock.
     */
    private void evictOldestExcept(BigChunk skip) {
        ChunkKey oldestKey = null;
        int oldestLru = Integer.MAX_VALUE;
        for (Map.Entry<ChunkKey, BigChunk> entry : cache.entrySet()) {
            if (entry.getValue() != skip && entry.getValue().lruCounter < oldestLru) {
                oldestLru = entry.getValue().lruCounter;
                oldestKey = entry.getKey();
            }
        }
        if (oldestKey != null) {
            BigChunk removed = cache.remove(oldestKey);
            if (removed != null) {
                estimatedMemoryBytes -= estimateChunkBytes(removed);
                evictions++;
            }
        }
    }

    /**
     * Return a stats string: chunk count, blocks-vs-far split, estimated heap, evictions.
     * Memory is estimated based on JVM object sizing (64-bit, compressed refs).
     * MAX_MEMORY limit = 20 MB; PIXEL_RIVER chunks are ~1,088 KB each (far+blocks),
     * PIXEL_RIVER_FAR chunks are ~64 KB each (far only).
     */
    public synchronized String getStats() {
        int withBlocks = 0;
        for (BigChunk c : cache.values()) if (c.blocks != null) withBlocks++;
        int total = cache.size();
        return String.format(
                "chunks=%d (%d w/blocks ~%dMB, %d far-only ~%dMB), est.total=%dKB / %dKB limit, evictions=%d",
                total, withBlocks, withBlocks * BLOCKS_BYTES / (1024 * 1024),
                total - withBlocks, (total - withBlocks) * FAR_CACHE_BYTES / (1024 * 1024),
                estimatedMemoryBytes / 1024, MAX_MEMORY / 1024, evictions);
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

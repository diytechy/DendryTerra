package dendryterra;

import java.util.*;

/**
 * LRU cache for BigChunk instances.
 * Stores up to 10 MB of bigchunks (~75 chunks at 132 KB each).
 */
public class BigChunkCache {
    private static final long MAX_MEMORY = 10 * 1024 * 1024; // 10 MB
    private static final long CHUNK_SIZE = 132 * 1024; // ~132 KB per chunk
    private static final int MAX_CHUNKS = (int)(MAX_MEMORY / CHUNK_SIZE); // ~150 chunks

    /** Map from chunk coordinates to BigChunk instances */
    private final Map<ChunkKey, BigChunk> cache;

    /** LRU counter for cache eviction (incremented on each access) */
    private int lruCounter;

    /** Cache statistics */
    private long hits;
    private long misses;

    public BigChunkCache() {
        this.cache = new HashMap<>();
        this.lruCounter = 0;
        this.hits = 0;
        this.misses = 0;
    }

    /**
     * Get or create a BigChunk at the specified chunk coordinates.
     * @param chunkX Integer chunk X coordinate
     * @param chunkY Integer chunk Y coordinate
     * @param gridOriginX Grid X coordinate of chunk origin (normalized space)
     * @param gridOriginY Grid Y coordinate of chunk origin (normalized space)
     * @return The BigChunk (may be newly created or from cache)
     */
    public BigChunk getOrCreate(int chunkX, int chunkY, double gridOriginX, double gridOriginY) {
        ChunkKey key = new ChunkKey(chunkX, chunkY);
        BigChunk chunk = cache.get(key);

        if (chunk != null) {
            // Cache hit
            hits++;
            chunk.lruCounter = ++lruCounter;
            return chunk;
        }

        // Cache miss - create new chunk
        misses++;
        chunk = new BigChunk(gridOriginX, gridOriginY);
        chunk.lruCounter = ++lruCounter;

        // Evict oldest if cache is full
        if (cache.size() >= MAX_CHUNKS) {
            evictOldest();
        }

        cache.put(key, chunk);
        return chunk;
    }

    /**
     * Evict the least recently used chunk from the cache.
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
            cache.remove(oldestKey);
        }
    }

    /**
     * Clear all cached chunks.
     */
    public void clear() {
        cache.clear();
        lruCounter = 0;
    }

    /**
     * Get cache statistics as a formatted string.
     */
    public String getStats() {
        long estimatedMemoryKB = (cache.size() * CHUNK_SIZE) / 1024;
        return String.format("bigchunks=%d, hits=%d, misses=%d, memory=%d KB",
            cache.size(), hits, misses, estimatedMemoryKB);
    }

    /**
     * Reset cache statistics.
     */
    public void resetStats() {
        hits = 0;
        misses = 0;
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

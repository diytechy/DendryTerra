package dendryterra;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * LRU cache for SegmentList instances from generateAllSegments.
 * Stores up to 10 MB of segment lists to avoid regenerating asterisms and segments.
 * Thread-safe: uses ConcurrentHashMap for reads and synchronized for mutations.
 */
public class SegmentListCache {
    private static final long MAX_MEMORY = 10 * 1024 * 1024; // 10 MB

    /** Estimated bytes per point (position + metadata) */
    private static final int BYTES_PER_POINT = 64;

    /** Estimated bytes per segment (indices + tangents + metadata) */
    private static final int BYTES_PER_SEGMENT = 48;

    /** Map from cell coordinates to cached segment lists */
    private final ConcurrentHashMap<CellKey, CachedSegmentList> cache;

    /** Current estimated memory usage in bytes */
    private long currentMemory;

    /** LRU counter for cache eviction */
    private int lruCounter;

    public SegmentListCache() {
        this.cache = new ConcurrentHashMap<>();
        this.currentMemory = 0;
        this.lruCounter = 0;
    }

    /**
     * Get a cached SegmentList for the specified cell coordinates.
     * @param cellX Cell X coordinate
     * @param cellY Cell Y coordinate
     * @return The cached SegmentList, or null if not in cache
     */
    public SegmentList get(double cellX, double cellY) {
        CellKey key = new CellKey(cellX, cellY);
        CachedSegmentList cached = cache.get(key);

        if (cached != null) {
            cached.lruCounter = lruCounter; // Approximate LRU, exact ordering not critical
            return cached.segmentList;
        }

        return null;
    }

    /**
     * Store a SegmentList in the cache.
     * @param cellX Cell X coordinate
     * @param cellY Cell Y coordinate
     * @param segmentList The SegmentList to cache
     */
    public synchronized void put(double cellX, double cellY, SegmentList segmentList) {
        CellKey key = new CellKey(cellX, cellY);

        // Check if already inserted by another thread
        if (cache.containsKey(key)) {
            return;
        }

        // Calculate memory usage for this segment list
        long memorySize = estimateMemory(segmentList);

        // Evict old entries if needed to make room
        while (currentMemory + memorySize > MAX_MEMORY && !cache.isEmpty()) {
            evictOldest();
        }

        // Add to cache if it fits
        if (currentMemory + memorySize <= MAX_MEMORY) {
            CachedSegmentList cached = new CachedSegmentList(segmentList, memorySize, ++lruCounter);
            cache.put(key, cached);
            currentMemory += memorySize;
        }
    }

    /**
     * Estimate memory usage of a SegmentList.
     */
    private long estimateMemory(SegmentList segmentList) {
        long pointMemory = segmentList.getPointCount() * BYTES_PER_POINT;
        long segmentMemory = segmentList.getSegmentCount() * BYTES_PER_SEGMENT;
        return pointMemory + segmentMemory + 64; // +64 for object overhead
    }

    /**
     * Evict the least recently used segment list from the cache.
     * Must be called while holding the synchronized lock.
     */
    private void evictOldest() {
        if (cache.isEmpty()) {
            return;
        }

        CellKey oldestKey = null;
        int oldestLru = Integer.MAX_VALUE;

        for (Map.Entry<CellKey, CachedSegmentList> entry : cache.entrySet()) {
            if (entry.getValue().lruCounter < oldestLru) {
                oldestLru = entry.getValue().lruCounter;
                oldestKey = entry.getKey();
            }
        }

        if (oldestKey != null) {
            CachedSegmentList removed = cache.remove(oldestKey);
            if (removed != null) {
                currentMemory -= removed.memorySize;
            }
        }
    }

    /**
     * Wrapper for cached SegmentList with metadata.
     */
    private static class CachedSegmentList {
        final SegmentList segmentList;
        final long memorySize;
        volatile int lruCounter;

        CachedSegmentList(SegmentList segmentList, long memorySize, int lruCounter) {
            this.segmentList = segmentList;
            this.memorySize = memorySize;
            this.lruCounter = lruCounter;
        }
    }

    /**
     * Key for identifying cells by coordinates.
     */
    private static class CellKey {
        final double cellX;
        final double cellY;

        CellKey(double cellX, double cellY) {
            this.cellX = cellX;
            this.cellY = cellY;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof CellKey)) return false;
            CellKey key = (CellKey) o;
            return Double.compare(key.cellX, cellX) == 0 &&
                   Double.compare(key.cellY, cellY) == 0;
        }

        @Override
        public int hashCode() {
            return Objects.hash(cellX, cellY);
        }

        @Override
        public String toString() {
            return String.format("CellKey(%.2f, %.2f)", cellX, cellY);
        }
    }
}

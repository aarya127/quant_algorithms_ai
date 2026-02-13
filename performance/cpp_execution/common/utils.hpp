#pragma once

#include <chrono>
#include <cstdint>

namespace quant {
namespace common {

// High-resolution timestamp
inline uint64_t get_timestamp_ns() {
    return std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
}

// Aligned memory allocation for cache optimization
template<typename T>
T* aligned_alloc(size_t alignment, size_t count) {
    return static_cast<T*>(::aligned_alloc(alignment, sizeof(T) * count));
}

// Cache line size (typically 64 bytes)
constexpr size_t CACHE_LINE_SIZE = 64;

// Padding to prevent false sharing
template<typename T>
struct alignas(CACHE_LINE_SIZE) CacheAligned {
    T value;
};

} // namespace common
} // namespace quant

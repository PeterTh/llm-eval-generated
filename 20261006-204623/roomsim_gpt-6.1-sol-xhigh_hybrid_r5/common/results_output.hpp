#pragma once

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <vector>

#include "hash.hpp"

// Kahan summation for numerically stable sum
template <typename T>
T kahan_sum(const std::vector<T>& data) {
    T sum = 0.0;
    T c = 0.0;
    for (const auto& val : data) {
        T y = val - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

// Print results for floating-point arrays
template <typename T>
void print_results(const std::vector<T>& data, const char* name = "Result") {
    if (data.empty()) {
        printf("=== RESULTS ===\n");
        printf("Name: %s\n", name);
        printf("Elements: 0\n");
        printf("=== END RESULTS ===\n");
        return;
    }

    const size_t N = data.size();
    
    // Compute statistics
    T sum = kahan_sum(data);
    T min_val = data[0];
    T max_val = data[0];
    for (const auto& val : data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // Compute hash
    hash h;
    for (const auto& val : data) {
        h.add(val);
    }
    
    // Sample indices: 0, N/4, N/2, 3N/4, N-1
    size_t indices[5] = {0, N/4, N/2, 3*N/4, N-1};
    
    printf("=== RESULTS ===\n");
    printf("Name: %s\n", name);
    printf("Elements: %zu\n", N);
    printf("Sum: %.17e\n", static_cast<double>(sum));
    printf("Min: %.17e\n", static_cast<double>(min_val));
    printf("Max: %.17e\n", static_cast<double>(max_val));
    for (int i = 0; i < 5; ++i) {
        printf("Sample[%zu]: %.17e\n", indices[i], static_cast<double>(data[indices[i]]));
    }
    printf("Hash: %016" PRIx64 "\n", h.get());
    printf("=== END RESULTS ===\n");
}

// Print results for integer arrays (uses exact hash comparison)
template <typename T>
void print_results_int(const std::vector<T>& data, const char* name = "Result") {
    if (data.empty()) {
        printf("=== RESULTS ===\n");
        printf("Name: %s\n", name);
        printf("Elements: 0\n");
        printf("=== END RESULTS ===\n");
        return;
    }

    const size_t N = data.size();
    
    // Compute hash
    hash h;
    for (const auto& val : data) {
        h.add(val);
    }
    
    // Compute min/max
    T min_val = data[0];
    T max_val = data[0];
    for (const auto& val : data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // Sample indices: 0, N/4, N/2, 3N/4, N-1
    size_t indices[5] = {0, N/4, N/2, 3*N/4, N-1};
    
    printf("=== RESULTS ===\n");
    printf("Name: %s\n", name);
    printf("Elements: %zu\n", N);
    printf("Min: %lld\n", static_cast<long long>(min_val));
    printf("Max: %lld\n", static_cast<long long>(max_val));
    for (int i = 0; i < 5; ++i) {
        printf("Sample[%zu]: %lld\n", indices[i], static_cast<long long>(data[indices[i]]));
    }
    printf("Hash: %016" PRIx64 "\n", h.get());
    printf("=== END RESULTS ===\n");
}

#pragma once

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <vector>

#include "hash.hpp"

// Add one value to a Kahan sum.
template <typename T>
void kahan_add(T val, T& sum, T& correction) {
    T y = val - correction;
    T t = sum + y;
    correction = (t - sum) - y;
    sum = t;
}

template <typename T>
T kahan_sum(const std::vector<T>& data) {
    T sum = 0.0;
    T c = 0.0;
    for (const auto& val : data) {
        kahan_add(val, sum, c);
    }
    return sum;
}

// Summarize floating-point results supplied in consecutive blocks.
template <typename T>
class FloatingPointResults {
public:
    explicit FloatingPointResults(size_t count)
        : count_(count),
          indices_{0, count / 4, count / 2, 3 * (count / 4) + 3 * (count % 4) / 4,
                   count == 0 ? 0 : count - 1} {}

    void add(const T* data, size_t count) {
        if (count == 0) {
            return;
        }
        if (offset_ == 0) {
            min_ = max_ = data[0];
        }
        for (size_t i = 0; i < count; ++i) {
            const T val = data[i];
            kahan_add(val, sum_, correction_);
            min_ = std::min(min_, val);
            max_ = std::max(max_, val);
            hash_.add(val);
        }
        for (size_t i = 0; i < indices_.size(); ++i) {
            if (indices_[i] >= offset_ && indices_[i] - offset_ < count) {
                samples_[i] = data[indices_[i] - offset_];
            }
        }
        offset_ += count;
    }

    void print(const char* name) const {
        printf("=== RESULTS ===\n");
        printf("Name: %s\n", name);
        printf("Elements: %zu\n", count_);
        if (count_ != 0) {
            printf("Sum: %.17e\n", static_cast<double>(sum_));
            printf("Min: %.17e\n", static_cast<double>(min_));
            printf("Max: %.17e\n", static_cast<double>(max_));
            for (size_t i = 0; i < indices_.size(); ++i) {
                printf("Sample[%zu]: %.17e\n", indices_[i], static_cast<double>(samples_[i]));
            }
            printf("Hash: %016" PRIx64 "\n", hash_.get());
        }
        printf("=== END RESULTS ===\n");
    }

private:
    size_t count_;
    std::array<size_t, 5> indices_;
    std::array<T, 5> samples_{};
    size_t offset_ = 0;
    T sum_ = 0;
    T correction_ = 0;
    T min_ = 0;
    T max_ = 0;
    hash hash_;
};

template <typename T>
void print_results(const std::vector<T>& data, const char* name = "Result") {
    FloatingPointResults<T> results(data.size());
    results.add(data.data(), data.size());
    results.print(name);
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

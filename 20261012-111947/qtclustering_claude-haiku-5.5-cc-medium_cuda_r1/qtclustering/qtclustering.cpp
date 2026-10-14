// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The candidate cluster grown from each unclustered seed is independent of
// the others within a round, so every seed is grown concurrently on the GPU
// (one thread block per seed). Within a block, candidates are scanned in
// parallel and the closest one is chosen with a block-wide reduction that
// breaks ties by lowest index, matching the sequential semantics exactly.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block used for growing one candidate cluster
static const int GROW_BLOCK_SIZE = 256;

// Upper bound on the device scratch used for per-seed distance arrays
static const size_t SCRATCH_BUDGET_BYTES = size_t(512) << 20;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Distance on the device. Explicit round-to-nearest intrinsics prevent FMA
// contraction so results match the host arithmetic bit for bit.
__device__ inline double deviceDistance(const Point& p1, const Point& p2) {
    const double dx = __dsub_rn(p1.x, p2.x);
    const double dy = __dsub_rn(p1.y, p2.y);
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

// Grows the candidate cluster for each seed in seeds[0..numSeeds) concurrently.
// Block b grows the cluster for seeds[b]; it reads the round's clustered
// flags and writes the cluster cardinality to cardinality[b].
//
// For each candidate c, maxDist[c] holds the maximum distance from c to the
// current cluster members (the diameter the cluster would have with c added).
// Candidates that are already clustered or in this cluster are pinned to +inf
// so they can never be selected. Because a max only grows, the incremental
// update yields exactly the same values as recomputing over all members.
//
// If membersOut is non-null, the block also writes its members in insertion
// order (membersOut[0..cardinality)). This is only used for the winning seed.
__global__ void growCandidateClusters(const Point* __restrict__ points,
                                      const uint8_t* __restrict__ clustered,
                                      const int* __restrict__ seeds,
                                      const int numSeeds,
                                      const int pointCount,
                                      const double threshold,
                                      double* __restrict__ maxDistScratch,
                                      int* __restrict__ cardinality,
                                      int* __restrict__ membersOut) {
    __shared__ double sVal[GROW_BLOCK_SIZE];
    __shared__ int sIdx[GROW_BLOCK_SIZE];
    __shared__ int sBest;

    const int block = blockIdx.x;
    if (block >= numSeeds) return;

    const int tid = threadIdx.x;
    const int seed = seeds[block];
    const double inf = static_cast<double>(INFINITY);
    double* maxDist = maxDistScratch + static_cast<size_t>(block) * pointCount;

    for (int c = tid; c < pointCount; c += GROW_BLOCK_SIZE) {
        maxDist[c] = (clustered[c] || c == seed) ? inf : 0.0;
    }
    __syncthreads();

    int count = 1;
    int current = seed;
    if (membersOut && tid == 0) membersOut[0] = seed;

    while (count < pointCount) {
        // Fold the newest member into every candidate's max distance
        const Point& cur = points[current];
        for (int c = tid; c < pointCount; c += GROW_BLOCK_SIZE) {
            const double d = deviceDistance(points[c], cur);
            if (maxDist[c] < d) maxDist[c] = d;
        }

        // Local best: smallest max distance below threshold, lowest index on ties
        double bestVal = inf;
        int bestIdx = -1;
        for (int c = tid; c < pointCount; c += GROW_BLOCK_SIZE) {
            const double v = maxDist[c];
            if (v < threshold && (bestIdx < 0 || v < bestVal)) {
                bestVal = v;
                bestIdx = c;
            }
        }
        sVal[tid] = bestVal;
        sIdx[tid] = bestIdx;
        __syncthreads();

        // Block reduction; threads hold ascending index ranges, so a strict
        // comparison keeps the lowest index among equal values
        for (int s = GROW_BLOCK_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const int other = sIdx[tid + s];
                if (other >= 0 && (sIdx[tid] < 0 || sVal[tid + s] < sVal[tid] ||
                                   (sVal[tid + s] == sVal[tid] && other < sIdx[tid]))) {
                    sVal[tid] = sVal[tid + s];
                    sIdx[tid] = other;
                }
            }
            __syncthreads();
        }

        if (tid == 0) sBest = sIdx[0];
        __syncthreads();

        const int best = sBest;
        if (best < 0) break; // No more points can be added

        if (tid == 0) {
            maxDist[best] = inf; // Now a member; never selected again
            if (membersOut) membersOut[count] = best;
        }
        ++count;
        current = best;
        __syncthreads();
    }

    if (tid == 0) cardinality[block] = count;
}

// Owns a device allocation and releases it on scope exit
template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }

  private:
    T* ptr_ = nullptr;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    if (N == 0) return clusters;

    // Device buffers are sized for the worst case and reused across rounds
    const size_t maxBatch = std::max<size_t>(
        1, std::min<size_t>(N, SCRATCH_BUDGET_BYTES / (sizeof(double) * N)));
    DeviceBuffer<Point> d_points(N);
    DeviceBuffer<uint8_t> d_clustered(N);
    DeviceBuffer<int> d_seeds(N);
    DeviceBuffer<double> d_maxDist(maxBatch * N);
    DeviceBuffer<int> d_cardinality(maxBatch);
    DeviceBuffer<int> d_members(N);

    CUDA_CHECK(cudaMemcpy(d_points.get(), points.data(), N * sizeof(Point),
                          cudaMemcpyHostToDevice));

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int numSeeds = static_cast<int>(unclustered_indices.size());
        int max_cardinality = -1;
        int best_seed_pos = -1;

        CUDA_CHECK(cudaMemcpy(d_clustered.get(), clustered.data(), N * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds.get(), unclustered_indices.data(),
                              numSeeds * sizeof(int), cudaMemcpyHostToDevice));

        // Grow a candidate cluster from every unclustered seed, in batches that
        // fit the scratch budget. Scanning in order with a strict comparison
        // selects the first seed that reaches the maximum cardinality.
        std::vector<int> cardinalities(numSeeds);
        for (int start = 0; start < numSeeds; start += static_cast<int>(maxBatch)) {
            const int batch = std::min<int>(static_cast<int>(maxBatch), numSeeds - start);
            growCandidateClusters<<<batch, GROW_BLOCK_SIZE>>>(
                d_points.get(), d_clustered.get(), d_seeds.get() + start, batch, N,
                threshold, d_maxDist.get(), d_cardinality.get(), nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(cardinalities.data() + start, d_cardinality.get(),
                                  batch * sizeof(int), cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < numSeeds; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed_pos = i;
            }
        }

        // If we found a cluster, add it
        if (best_seed_pos >= 0 && max_cardinality > 0) {
            const int best_seed = unclustered_indices[best_seed_pos];

            // Re-grow the winning seed to recover its members in insertion order.
            // The computation is deterministic, so this matches the selection pass.
            growCandidateClusters<<<1, GROW_BLOCK_SIZE>>>(
                d_points.get(), d_clustered.get(), d_seeds.get() + best_seed_pos, 1, N,
                threshold, d_maxDist.get(), d_cardinality.get(), d_members.get());
            CUDA_CHECK(cudaGetLastError());

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.resize(max_cardinality);
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members.get(),
                                  max_cardinality * sizeof(int), cudaMemcpyDeviceToHost));
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Create the CUDA context up front so one-time driver setup is not timed
    CUDA_CHECK(cudaFree(nullptr));

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 : 
        static_cast<double>(total_clustered) / clusters.size();
    
    printf("Points clustered: %d / %d (%.1f%%)\n", 
           total_clustered, num_points, 
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }
    
    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

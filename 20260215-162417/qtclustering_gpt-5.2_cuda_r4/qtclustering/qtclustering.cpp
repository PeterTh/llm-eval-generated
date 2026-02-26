// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cstdint>
#include <climits>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

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

namespace cuda_qt {
static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::abort();
    }
}
} // namespace cuda_qt

#define CUDA_CHECK(x) cuda_qt::cudaCheck((x), __FILE__, __LINE__)

namespace cuda_qt {
struct Context {
    int N = 0;
    double* d_x = nullptr;
    double* d_y = nullptr;
    uint8_t* d_clustered = nullptr;
    uint8_t* d_in_cluster = nullptr;
    double* d_maxDistSq = nullptr;
    double* d_bestVal = nullptr;
    int* d_bestIdx = nullptr;
};

__device__ __forceinline__ double distSq(const double* x, const double* y, int a, int b) {
    const double dx = x[a] - x[b];
    const double dy = y[a] - y[b];
    return dx * dx + dy * dy;
}

__device__ __forceinline__ void atomicMinDouble(double* addr, double val) {
    // Distances are non-negative; IEEE-754 positive double bit ordering matches numerical ordering.
    auto* ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ull;
    while (__longlong_as_double(old) > val) {
        const unsigned long long assumed = old;
        old = atomicCAS(ull, assumed, __double_as_longlong(val));
        if (old == assumed) break;
    }
}

__global__ void init_state(const double* x, const double* y, int N, int seed,
                          uint8_t* in_cluster, double* maxDistSq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    in_cluster[i] = static_cast<uint8_t>(i == seed);
    maxDistSq[i] = distSq(x, y, i, seed);
}

__global__ void reset_best(double* bestVal, int* bestIdx) {
    *bestVal = DBL_MAX;
    *bestIdx = INT_MAX;
}

__global__ void update_and_find_best_val(const double* x, const double* y, int N, int new_member,
                                        double thresholdSq,
                                        const uint8_t* clustered, const uint8_t* in_cluster,
                                        double* maxDistSq, double* bestVal) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (clustered[i] || in_cluster[i]) return;

    const double d2 = distSq(x, y, i, new_member);
    const double prev = maxDistSq[i];
    const double val = (d2 > prev) ? d2 : prev;
    maxDistSq[i] = val;

    if (val < thresholdSq) {
        atomicMinDouble(bestVal, val);
    }
}

__global__ void find_best_idx(int N, double thresholdSq,
                             const uint8_t* clustered, const uint8_t* in_cluster,
                             const double* maxDistSq, const double* bestVal, int* bestIdx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (clustered[i] || in_cluster[i]) return;

    const double v = maxDistSq[i];
    const double bv = *bestVal;
    if (v < thresholdSq && v == bv) {
        atomicMin(bestIdx, i);
    }
}

__global__ void mark_in_cluster(uint8_t* in_cluster, int idx) {
    in_cluster[idx] = 1;
}

static void initContext(Context& ctx, const std::vector<Point>& points) {
    ctx.N = static_cast<int>(points.size());
    std::vector<double> h_x(ctx.N), h_y(ctx.N);
    for (int i = 0; i < ctx.N; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
    }

    CUDA_CHECK(cudaMalloc(&ctx.d_x, ctx.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_y, ctx.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_clustered, ctx.N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_in_cluster, ctx.N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&ctx.d_maxDistSq, ctx.N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_bestVal, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_bestIdx, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(ctx.d_x, h_x.data(), ctx.N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_y, h_y.data(), ctx.N * sizeof(double), cudaMemcpyHostToDevice));
}

static void freeContext(Context& ctx) {
    cudaFree(ctx.d_x);
    cudaFree(ctx.d_y);
    cudaFree(ctx.d_clustered);
    cudaFree(ctx.d_in_cluster);
    cudaFree(ctx.d_maxDistSq);
    cudaFree(ctx.d_bestVal);
    cudaFree(ctx.d_bestIdx);
    ctx = Context{};
}

static void updateClustered(Context& ctx, const std::vector<uint8_t>& clustered) {
    CUDA_CHECK(cudaMemcpy(ctx.d_clustered, clustered.data(), ctx.N * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
}

// Generate a candidate cluster starting from a seed point using CUDA.
// Returns the cardinality (size) of the cluster.
[[maybe_unused]] static int generateCandidateCluster(const int seed_point,
                                   Context& ctx,
                                   const double threshold,
                                   const int point_count,
                                   std::vector<int>* cluster_members = nullptr) {
    const double thresholdSq = threshold * threshold;
    const int N = point_count;

    const dim3 block(256);
    const dim3 grid((N + block.x - 1) / block.x);

    init_state<<<grid, block>>>(ctx.d_x, ctx.d_y, N, seed_point, ctx.d_in_cluster, ctx.d_maxDistSq);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> members;
    members.reserve(N);
    members.push_back(seed_point);

    int new_member = seed_point;
    while (static_cast<int>(members.size()) < N) {
        reset_best<<<1, 1>>>(ctx.d_bestVal, ctx.d_bestIdx);
        CUDA_CHECK(cudaGetLastError());

        update_and_find_best_val<<<grid, block>>>(ctx.d_x, ctx.d_y, N, new_member, thresholdSq,
                                                 ctx.d_clustered, ctx.d_in_cluster,
                                                 ctx.d_maxDistSq, ctx.d_bestVal);
        CUDA_CHECK(cudaGetLastError());

        find_best_idx<<<grid, block>>>(N, thresholdSq, ctx.d_clustered, ctx.d_in_cluster,
                                       ctx.d_maxDistSq, ctx.d_bestVal, ctx.d_bestIdx);
        CUDA_CHECK(cudaGetLastError());

        int bestIdx = INT_MAX;
        double bestVal = DBL_MAX;
        CUDA_CHECK(cudaMemcpy(&bestIdx, ctx.d_bestIdx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bestVal, ctx.d_bestVal, sizeof(double), cudaMemcpyDeviceToHost));

        if (bestIdx == INT_MAX || !(bestVal < thresholdSq)) break;

        members.push_back(bestIdx);
        mark_in_cluster<<<1, 1>>>(ctx.d_in_cluster, bestIdx);
        CUDA_CHECK(cudaGetLastError());
        new_member = bestIdx;
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

constexpr int SEED_BLOCK = 256;

__device__ __forceinline__ bool better_pair(double v, int idx, double bestV, int bestIdx) {
    return (v < bestV) || (v == bestV && idx < bestIdx);
}

__global__ void seed_cardinality_kernel(const double* x, const double* y, int N,
                                       const uint8_t* clustered, const int* seeds,
                                       double thresholdSq, int* outCard) {
    const int seed = seeds[blockIdx.x];
    const int tid = threadIdx.x;

    if (clustered[seed]) {
        if (tid == 0) outCard[blockIdx.x] = 0;
        return;
    }

    extern __shared__ unsigned char smem[];
    double* maxDistSq = reinterpret_cast<double*>(smem);
    uint8_t* in_cluster = reinterpret_cast<uint8_t*>(maxDistSq + N);

    __shared__ double redVal[SEED_BLOCK];
    __shared__ int redIdx[SEED_BLOCK];

    for (int i = tid; i < N; i += SEED_BLOCK) {
        in_cluster[i] = static_cast<uint8_t>(i == seed);
        maxDistSq[i] = distSq(x, y, i, seed);
    }
    __syncthreads();

    __shared__ int s_count;
    __shared__ int s_new_member;
    __shared__ int s_bestIdx;

    if (tid == 0) {
        s_count = 1;
        s_new_member = seed;
        s_bestIdx = INT_MAX;
    }
    __syncthreads();

    while (s_count < N) {
        double bestV = DBL_MAX;
        int bestIdx = INT_MAX;

        for (int i = tid; i < N; i += SEED_BLOCK) {
            if (clustered[i] || in_cluster[i]) continue;
            const double v = maxDistSq[i];
            if (v < thresholdSq && better_pair(v, i, bestV, bestIdx)) {
                bestV = v;
                bestIdx = i;
            }
        }

        redVal[tid] = bestV;
        redIdx[tid] = bestIdx;
        __syncthreads();

        for (int offset = SEED_BLOCK / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double ov = redVal[tid + offset];
                const int oi = redIdx[tid + offset];
                if (oi != INT_MAX && better_pair(ov, oi, redVal[tid], redIdx[tid])) {
                    redVal[tid] = ov;
                    redIdx[tid] = oi;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_bestIdx = redIdx[0];
            if (s_bestIdx != INT_MAX) {
                in_cluster[s_bestIdx] = 1;
                s_new_member = s_bestIdx;
                s_count++;
            }
        }
        __syncthreads();

        if (s_bestIdx == INT_MAX) break;

        const int new_member = s_new_member;
        for (int i = tid; i < N; i += SEED_BLOCK) {
            if (clustered[i] || in_cluster[i]) continue;
            const double d2 = distSq(x, y, i, new_member);
            const double prev = maxDistSq[i];
            maxDistSq[i] = (d2 > prev) ? d2 : prev;
        }
        __syncthreads();
    }

    if (tid == 0) outCard[blockIdx.x] = s_count;
}

__global__ void build_members_kernel(const double* x, const double* y, int N,
                                    const uint8_t* clustered, int seed,
                                    double thresholdSq, int* outMembers, int* outCount) {
    const int tid = threadIdx.x;

    if (clustered[seed]) {
        if (tid == 0) *outCount = 0;
        return;
    }

    extern __shared__ unsigned char smem[];
    double* maxDistSq = reinterpret_cast<double*>(smem);
    uint8_t* in_cluster = reinterpret_cast<uint8_t*>(maxDistSq + N);

    __shared__ double redVal[SEED_BLOCK];
    __shared__ int redIdx[SEED_BLOCK];

    for (int i = tid; i < N; i += SEED_BLOCK) {
        in_cluster[i] = static_cast<uint8_t>(i == seed);
        maxDistSq[i] = distSq(x, y, i, seed);
    }

    if (tid == 0) {
        outMembers[0] = seed;
        *outCount = 1;
    }
    __syncthreads();

    __shared__ int s_count;
    __shared__ int s_new_member;
    __shared__ int s_bestIdx;

    if (tid == 0) {
        s_count = 1;
        s_new_member = seed;
        s_bestIdx = INT_MAX;
    }
    __syncthreads();

    while (s_count < N) {
        double bestV = DBL_MAX;
        int bestIdx = INT_MAX;

        for (int i = tid; i < N; i += SEED_BLOCK) {
            if (clustered[i] || in_cluster[i]) continue;
            const double v = maxDistSq[i];
            if (v < thresholdSq && better_pair(v, i, bestV, bestIdx)) {
                bestV = v;
                bestIdx = i;
            }
        }

        redVal[tid] = bestV;
        redIdx[tid] = bestIdx;
        __syncthreads();

        for (int offset = SEED_BLOCK / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double ov = redVal[tid + offset];
                const int oi = redIdx[tid + offset];
                if (oi != INT_MAX && better_pair(ov, oi, redVal[tid], redIdx[tid])) {
                    redVal[tid] = ov;
                    redIdx[tid] = oi;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_bestIdx = redIdx[0];
            if (s_bestIdx != INT_MAX) {
                in_cluster[s_bestIdx] = 1;
                s_new_member = s_bestIdx;
                outMembers[s_count] = s_bestIdx;
                s_count++;
                *outCount = s_count;
            }
        }
        __syncthreads();

        if (s_bestIdx == INT_MAX) break;

        const int new_member = s_new_member;
        for (int i = tid; i < N; i += SEED_BLOCK) {
            if (clustered[i] || in_cluster[i]) continue;
            const double d2 = distSq(x, y, i, new_member);
            const double prev = maxDistSq[i];
            maxDistSq[i] = (d2 > prev) ? d2 : prev;
        }
        __syncthreads();
    }
}

static size_t sharedBytesForN(int N) {
    return static_cast<size_t>(N) * (sizeof(double) + sizeof(uint8_t));
}
} // namespace cuda_qt

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    cuda_qt::Context ctx;
    cuda_qt::initContext(ctx, points);

    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        cuda_qt::updateClustered(ctx, clustered);

        const int numSeeds = static_cast<int>(unclustered_indices.size());
        const double thresholdSq = threshold * threshold;

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              static_cast<size_t>(numSeeds) * sizeof(int),
                              cudaMemcpyHostToDevice));

        const size_t sharedBytes = cuda_qt::sharedBytesForN(N);
        cuda_qt::seed_cardinality_kernel<<<numSeeds, cuda_qt::SEED_BLOCK, sharedBytes>>>(
            ctx.d_x, ctx.d_y, N, ctx.d_clustered, d_seeds, thresholdSq, d_card);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> h_card(numSeeds);
        CUDA_CHECK(cudaMemcpy(h_card.data(), d_card,
                              static_cast<size_t>(numSeeds) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < numSeeds; ++i) {
            const int card = h_card[i];
            if (card > max_cardinality) {
                max_cardinality = card;
                best_seed = unclustered_indices[static_cast<size_t>(i)];
            }
        }

        std::vector<int> best_cluster_members;
        if (best_seed >= 0 && max_cardinality > 0) {
            cuda_qt::build_members_kernel<<<1, cuda_qt::SEED_BLOCK, sharedBytes>>>(
                ctx.d_x, ctx.d_y, N, ctx.d_clustered, best_seed, thresholdSq, d_members, d_count);
            CUDA_CHECK(cudaGetLastError());

            int count = 0;
            CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
            best_cluster_members.resize(static_cast<size_t>(count));
            if (count > 0) {
                CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), d_members,
                                      static_cast<size_t>(count) * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
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

    cudaFree(d_seeds);
    cudaFree(d_card);
    cudaFree(d_members);
    cudaFree(d_count);

    cuda_qt::freeContext(ctx);
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

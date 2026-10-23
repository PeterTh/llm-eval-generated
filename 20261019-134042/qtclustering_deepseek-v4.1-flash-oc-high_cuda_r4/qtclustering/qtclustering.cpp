// QT Clustering Benchmark - CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// All candidate clusters of a clustering round are evaluated in parallel on
// the GPU (one CUDA block per seed point).

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static const int QT_BLOCK_THREADS = 256;

__device__ __forceinline__ double pointDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Greedily grow one candidate cluster.  For every not-yet-clustered point we
// keep the largest distance to any member of the growing cluster (md).  The
// next point added is the one with the smallest such value, provided it is
// below the threshold.  This is exactly the reference sequencing rule, but
// the per-candidate maxima are maintained incrementally so each expansion
// step costs O(point_count) instead of O(point_count * cluster_size).
template <bool USE_MATRIX>
__device__ int greedyCandidate(const Point* __restrict__ points,
                               const double* __restrict__ dist_matrix,
                               const unsigned char* __restrict__ clustered,
                               const int seed,
                               const int point_count,
                               const double threshold,
                               double* md,
                               int* __restrict__ members_out) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ double s_val[QT_BLOCK_THREADS];
    __shared__ int s_idx[QT_BLOCK_THREADS];
    __shared__ int s_next;

    for (int c = tid; c < point_count; c += nthreads) {
        if (clustered[c]) {
            md[c] = DBL_MAX;
        } else if constexpr (USE_MATRIX) {
            md[c] = dist_matrix[static_cast<size_t>(seed) * point_count + c];
        } else {
            md[c] = pointDistance(points[c], points[seed]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        md[seed] = DBL_MAX;
        if (members_out) members_out[0] = seed;
    }
    __syncthreads();

    int count = 1;
    while (count < point_count) {
        // Per-thread minimum (ties resolve to the lowest index).
        double best_value = DBL_MAX;
        int best_index = -1;
        for (int c = tid; c < point_count; c += nthreads) {
            const double v = md[c];
            if (v < best_value || (v == best_value && c < best_index)) {
                best_value = v;
                best_index = c;
            }
        }

        s_val[tid] = best_value;
        s_idx[tid] = best_index;
        __syncthreads();

        // Block-wide minimum.  The comparison rule makes the result
        // independent of the reduction order and matches the sequential
        // first-minimum tie break.
        for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const double v1 = s_val[tid];
                const double v2 = s_val[tid + stride];
                const int i1 = s_idx[tid];
                const int i2 = s_idx[tid + stride];
                if (v2 < v1 || (v2 == v1 && i2 < i1)) {
                    s_val[tid] = v2;
                    s_idx[tid] = i2;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_next = (s_val[0] < threshold) ? s_idx[0] : -1;
        }
        __syncthreads();

        const int next = s_next;
        if (next < 0) break;  // No remaining point keeps the diameter below threshold.

        ++count;
        if (members_out && tid == 0) members_out[count - 1] = next;

        if (tid == 0) md[next] = DBL_MAX;
        __syncthreads();

        // Fold the newly added member into every remaining candidate's maximum.
        for (int c = tid; c < point_count; c += nthreads) {
            const double current = md[c];
            if (current < DBL_MAX) {
                double d;
                if constexpr (USE_MATRIX) {
                    d = dist_matrix[static_cast<size_t>(next) * point_count + c];
                } else {
                    d = pointDistance(points[c], points[next]);
                }
                md[c] = (d > current) ? d : current;
            }
        }
        __syncthreads();
    }

    return count;
}

template <bool USE_MATRIX>
__global__ void candidateSizesSharedKernel(
    const Point* __restrict__ points,
    const double* __restrict__ dist_matrix,
    const unsigned char* __restrict__ clustered,
    const int* __restrict__ seeds,
    const int num_seeds,
    const int point_count,
    const double threshold,
    int* __restrict__ sizes) {
    extern __shared__ double shared_md[];
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        sizes[s] = greedyCandidate<USE_MATRIX>(
            points, dist_matrix, clustered, seeds[s], point_count, threshold,
            shared_md, nullptr);
    }
}

template <bool USE_MATRIX>
__global__ void candidateSizesGlobalKernel(
    const Point* __restrict__ points,
    const double* __restrict__ dist_matrix,
    const unsigned char* __restrict__ clustered,
    const int* __restrict__ seeds,
    const int num_seeds,
    const int point_count,
    const double threshold,
    double* __restrict__ scratch,
    int* __restrict__ sizes) {
    double* md = scratch + static_cast<size_t>(blockIdx.x) * point_count;
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        sizes[s] = greedyCandidate<USE_MATRIX>(
            points, dist_matrix, clustered, seeds[s], point_count, threshold,
            md, nullptr);
    }
}

template <bool USE_MATRIX>
__global__ void candidateMembersSharedKernel(
    const Point* __restrict__ points,
    const double* __restrict__ dist_matrix,
    const unsigned char* __restrict__ clustered,
    const int seed,
    const int point_count,
    const double threshold,
    int* __restrict__ members) {
    extern __shared__ double shared_md[];
    greedyCandidate<USE_MATRIX>(points, dist_matrix, clustered, seed,
                                point_count, threshold, shared_md, members);
}

template <bool USE_MATRIX>
__global__ void candidateMembersGlobalKernel(
    const Point* __restrict__ points,
    const double* __restrict__ dist_matrix,
    const unsigned char* __restrict__ clustered,
    const int seed,
    const int point_count,
    const double threshold,
    double* __restrict__ scratch,
    int* __restrict__ members) {
    greedyCandidate<USE_MATRIX>(points, dist_matrix, clustered, seed,
                                point_count, threshold, scratch, members);
}

__global__ void buildDistanceMatrixKernel(const Point* __restrict__ points,
                                          double* __restrict__ matrix,
                                          const int point_count) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < point_count && j < point_count) {
        matrix[static_cast<size_t>(i) * point_count + j] =
            pointDistance(points[i], points[j]);
    }
}

// Main QT clustering algorithm (GPU accelerated).  All seeds of every
// clustering round are evaluated concurrently on the device; the host only
// performs the reduction over candidate cardinalities and bookkeeping.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    std::vector<Cluster> clusters;

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_sizes = nullptr;
    int* d_members = nullptr;
    double* d_scratch = nullptr;
    double* d_dist_matrix = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N,
                          cudaMemcpyHostToDevice));

    const size_t shared_bytes = static_cast<size_t>(N) * sizeof(double);
    const size_t static_shared =
        static_cast<size_t>(QT_BLOCK_THREADS) * sizeof(double) +
        static_cast<size_t>(QT_BLOCK_THREADS) * sizeof(int) + sizeof(int);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const size_t max_dynamic_shared =
        prop.sharedMemPerBlockOptin > static_shared
            ? prop.sharedMemPerBlockOptin - static_shared
            : 0;
    // Keep the per-candidate state in fast shared memory whenever the whole
    // point set fits in one block's opt-in shared allocation.  Very large
    // point sets fall back to the global-memory variant.
    const bool use_shared = shared_bytes <= max_dynamic_shared;
    if (use_shared) {
        CUDA_CHECK(cudaFuncSetAttribute(
            candidateSizesSharedKernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            candidateSizesSharedKernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            candidateMembersSharedKernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            candidateMembersSharedKernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Precompute all pairwise distances when the matrix fits comfortably in
    // device memory.  This removes sqrt() from the inner loop and makes the
    // expansion step bandwidth efficient (rows are accessed contiguously).
    bool use_matrix = false;
    const size_t matrix_bytes =
        static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(double);
    const size_t matrix_budget =
        std::min<size_t>(2ull * 1024ull * 1024ull * 1024ull, free_mem / 2);
    if (matrix_bytes > 0 && matrix_bytes <= matrix_budget) {
        if (cudaMalloc(&d_dist_matrix, matrix_bytes) == cudaSuccess) {
            const dim3 block(16, 16);
            const dim3 grid(static_cast<unsigned>((N + 15) / 16),
                            static_cast<unsigned>((N + 15) / 16));
            buildDistanceMatrixKernel<<<grid, block>>>(d_points, d_dist_matrix, N);
            CUDA_CHECK(cudaGetLastError());
            use_matrix = true;
        } else {
            (void)cudaGetLastError();
            d_dist_matrix = nullptr;
        }
    }

    size_t max_scratch_blocks = 0;
    if (!use_shared) {
        const size_t scratch_budget =
            std::min<size_t>(2ull * 1024ull * 1024ull * 1024ull, free_mem / 2);
        size_t blocks = shared_bytes > 0 ? scratch_budget / shared_bytes : 1;
        if (blocks < 1) blocks = 1;
        max_scratch_blocks = std::min<size_t>(static_cast<size_t>(N), blocks);
        CUDA_CHECK(cudaMalloc(&d_scratch, max_scratch_blocks * shared_bytes));
    }

    std::vector<unsigned char> h_clustered(N, 0);
    std::vector<int> h_sizes;
    std::vector<int> h_members(N);

    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        for (int i = 0; i < N; ++i) {
            h_clustered[i] = clustered[i] ? 1 : 0;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                              sizeof(unsigned char) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              sizeof(int) * num_seeds, cudaMemcpyHostToDevice));

        if (use_shared) {
            if (use_matrix) {
                candidateSizesSharedKernel<true>
                    <<<num_seeds, QT_BLOCK_THREADS, shared_bytes>>>(
                        d_points, d_dist_matrix, d_clustered, d_seeds,
                        num_seeds, N, threshold, d_sizes);
            } else {
                candidateSizesSharedKernel<false>
                    <<<num_seeds, QT_BLOCK_THREADS, shared_bytes>>>(
                        d_points, d_dist_matrix, d_clustered, d_seeds,
                        num_seeds, N, threshold, d_sizes);
            }
        } else {
            const int blocks = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(num_seeds), max_scratch_blocks));
            if (use_matrix) {
                candidateSizesGlobalKernel<true>
                    <<<blocks, QT_BLOCK_THREADS>>>(
                        d_points, d_dist_matrix, d_clustered, d_seeds,
                        num_seeds, N, threshold, d_scratch, d_sizes);
            } else {
                candidateSizesGlobalKernel<false>
                    <<<blocks, QT_BLOCK_THREADS>>>(
                        d_points, d_dist_matrix, d_clustered, d_seeds,
                        num_seeds, N, threshold, d_scratch, d_sizes);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        h_sizes.resize(num_seeds);
        CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_sizes,
                              sizeof(int) * num_seeds, cudaMemcpyDeviceToHost));

        int best_slot = -1;
        int best_cardinality = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_sizes[i] > best_cardinality) {
                best_cardinality = h_sizes[i];
                best_slot = i;
            }
        }
        if (best_slot < 0 || best_cardinality <= 0) break;

        const int best_seed = unclustered_indices[best_slot];

        // Rebuild the winning cluster to obtain its member list.
        if (use_shared) {
            if (use_matrix) {
                candidateMembersSharedKernel<true>
                    <<<1, QT_BLOCK_THREADS, shared_bytes>>>(
                        d_points, d_dist_matrix, d_clustered, best_seed, N,
                        threshold, d_members);
            } else {
                candidateMembersSharedKernel<false>
                    <<<1, QT_BLOCK_THREADS, shared_bytes>>>(
                        d_points, d_dist_matrix, d_clustered, best_seed, N,
                        threshold, d_members);
            }
        } else {
            if (use_matrix) {
                candidateMembersGlobalKernel<true>
                    <<<1, QT_BLOCK_THREADS>>>(
                        d_points, d_dist_matrix, d_clustered, best_seed, N,
                        threshold, d_scratch, d_members);
            } else {
                candidateMembersGlobalKernel<false>
                    <<<1, QT_BLOCK_THREADS>>>(
                        d_points, d_dist_matrix, d_clustered, best_seed, N,
                        threshold, d_scratch, d_members);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_members.data(), d_members,
                              sizeof(int) * best_cardinality,
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(h_members.begin(),
                               h_members.begin() + best_cardinality);
        clusters.push_back(std::move(cluster));

        for (int i = 0; i < best_cardinality; ++i) {
            clustered[h_members[i]] = true;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    if (d_dist_matrix) CUDA_CHECK(cudaFree(d_dist_matrix));
    if (d_scratch) CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));

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

    // Initialize the CUDA context before timing so the measurement covers the
    // clustering work rather than one-time driver/runtime setup.
    CUDA_CHECK(cudaSetDevice(0));

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

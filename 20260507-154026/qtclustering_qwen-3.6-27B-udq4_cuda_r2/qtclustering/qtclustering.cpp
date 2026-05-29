// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// GPU parallelization strategy:
//   - Data generation remains sequential (deterministic PRNG for reproducibility)
//   - All seeds are evaluated in parallel: one CUDA block per seed
//   - Within each block, threads cooperatively scan candidate points and
//     perform a shared-memory reduction to find the closest point
//   - The main clustering loop (sequential across iterations) runs on the host

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = call;                                                 \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Host / Device data structures
// ---------------------------------------------------------------------------

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ---------------------------------------------------------------------------
// CUDA device code
// ---------------------------------------------------------------------------

__device__ inline double device_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel: evaluate candidate clusters for multiple seeds in parallel.
//
// One block per seed.  Threads within a block cooperate to find the closest
// point by scanning candidates in a grid-stride loop and reducing with a
// tree reduction over shared memory.
//
// Shared memory layout (dynamic):
//   [in_cluster_bits : num_bitwords * 4]
//   [members         : point_count * 4]
//   [tbd             : blockDim.x * 8]
//   [tbi             : blockDim.x * 4]
//   [shared_mc       : 4]
//
// in_cluster_bits is a uint32_t bitmask so that the per-block shared-memory
// footprint stays well within the 48 KB limit even for large point counts.
__global__ void evaluateSeedsKernel(
    const Point*   points,
    const uint8_t* clustered,
    const int*     seeds,
    int            num_seeds,
    double         threshold,
    int            point_count,
    int*           out_cardinalities,
    int*           out_members)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int seed_point = seeds[seed_idx];

    // ---- shared memory pointers (properly aligned) ----
    extern __shared__ char shared[];

    int num_bitwords = (point_count + 31) / 32;
    uint32_t* in_cluster_bits = (uint32_t*)shared;

    // members starts after in_cluster_bits (4-byte aligned)
    int members_off = num_bitwords * 4;
    int* members = (int*)(shared + members_off);

    // tbd (double*) needs 8-byte alignment — pad members_off to 8-byte boundary
    int tbd_off = (members_off + point_count * 4 + 7) & ~7;
    double* tbd = (double*)(shared + tbd_off);

    // tbi (int*) needs 4-byte alignment — already guaranteed after doubles
    int tbi_off = tbd_off + blockDim.x * 8;
    int* tbi = (int*)(shared + tbi_off);

    // shared_mc (int*) needs 4-byte alignment
    int mc_off = tbi_off + blockDim.x * 4;
    int* shared_mc = (int*)(shared + mc_off);

    // ---- initialise ----
    for (int i = threadIdx.x; i < num_bitwords; i += blockDim.x)
        in_cluster_bits[i] = 0;
    __syncthreads();

    if (threadIdx.x == 0) *shared_mc = 0;
    __syncthreads();

    // add seed
    if (threadIdx.x == 0) {
        in_cluster_bits[seed_point >> 5] |= (1u << (seed_point & 31));
        members[0] = seed_point;
        *shared_mc = 1;
    }
    __syncthreads();

    // ---- main loop: iteratively add closest points ----
    while (*shared_mc < point_count) {
        // each thread scans a subset of candidates
        int    local_closest  = -1;
        double local_min_dist = 1e300;

        for (int cand = threadIdx.x; cand < point_count; cand += blockDim.x) {
            if (clustered[cand]) continue;
            if ((in_cluster_bits[cand >> 5] >> (cand & 31)) & 1u) continue;

            double max_dist = 0.0;
            int mc = *shared_mc;
            for (int m = 0; m < mc; ++m) {
                double d = device_distance(points[cand], points[members[m]]);
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < local_min_dist) {
                local_min_dist = max_dist;
                local_closest  = cand;
            }
        }

        // store local results in shared memory
        tbd[threadIdx.x] = local_min_dist;
        tbi[threadIdx.x] = local_closest;
        __syncthreads();

        // tree reduction: find candidate with minimum max-distance
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                int    i1 = tbi[threadIdx.x];
                int    i2 = tbi[threadIdx.x + stride];
                double d1 = tbd[threadIdx.x];
                double d2 = tbd[threadIdx.x + stride];

                // lexicographic comparison: prefer smaller distance, then
                // smaller index (matches the original sequential tie-breaking)
                if (i2 >= 0 && (i1 < 0 || d2 < d1 || (d2 == d1 && i2 < i1))) {
                    tbi[threadIdx.x] = i2;
                    tbd[threadIdx.x] = d2;
                }
            }
            __syncthreads();
        }

        // all threads read the winner
        int closest = tbi[0];
        __syncthreads();

        // add the closest point to the cluster (thread 0 only)
        if (closest >= 0) {
            if (threadIdx.x == 0) {
                in_cluster_bits[closest >> 5] |= (1u << (closest & 31));
                members[*shared_mc] = closest;
                (*shared_mc)++;
            }
        }
        __syncthreads();

        if (closest < 0) break;
    }

    // write results
    if (threadIdx.x == 0) {
        out_cardinalities[seed_idx] = *shared_mc;
        int* my_members = out_members + seed_idx * point_count;
        for (int i = 0; i < *shared_mc; ++i)
            my_members[i] = members[i];
    }
}

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------

// Generate synthetic 2D point data in clusters (sequential for reproducibility)
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42)
{
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Main QT clustering algorithm — CUDA parallel version
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold)
{
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i)
        unclustered_indices.push_back(i);

    // ---- allocate device memory ----
    Point* d_points = nullptr;
    uint8_t* d_clustered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point),
                          cudaMemcpyHostToDevice));
    // initial clustered array is all false
    CUDA_CHECK(cudaMemset(d_clustered, 0, N * sizeof(uint8_t)));

    // ---- determine block size and shared-memory size ----
    int block_size = 256;
    int num_bitwords = (N + 31) / 32;

    auto calc_shared_size = [num_bitwords, N](int bs) -> size_t {
        size_t members_off = static_cast<size_t>(num_bitwords) * 4;
        size_t tbd_off = (members_off + static_cast<size_t>(N) * 4 + 7) & ~7ULL;
        return tbd_off +
               static_cast<size_t>(bs) * 8 +   // tbd
               static_cast<size_t>(bs) * 4 +   // tbi
               4;                              // shared_mc
    };

    size_t shared_mem_size = calc_shared_size(block_size);

    // query device and shrink block size if shared memory would overflow
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t max_shared = prop.sharedMemPerBlock;

    while (shared_mem_size > max_shared && block_size > 64) {
        block_size >>= 1;
        shared_mem_size = calc_shared_size(block_size);
    }

    // ---- main clustering loop (sequential across iterations) ----
    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());

        // allocate output buffers on device
        int* d_cardinalities = nullptr;
        int* d_members = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cardinalities, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_members,
                              static_cast<size_t>(num_seeds) * N * sizeof(int)));

        // copy seeds to device
        int* d_seeds = nullptr;
        CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                              num_seeds * sizeof(int), cudaMemcpyHostToDevice));

        // launch: one block per seed
        evaluateSeedsKernel<<<num_seeds, block_size, shared_mem_size>>>(
            d_points, d_clustered, d_seeds, num_seeds, threshold, N,
            d_cardinalities, d_members);
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy cardinalities back
        std::vector<int> h_cardinalities(num_seeds);
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                              num_seeds * sizeof(int), cudaMemcpyDeviceToHost));

        // find best seed (maximum cardinality, first-wins tie-breaking)
        int max_card = -1;
        int best_seed_idx = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cardinalities[i] > max_card) {
                max_card = h_cardinalities[i];
                best_seed_idx = i;
            }
        }

        // copy best members back
        std::vector<int> best_members(max_card);
        if (best_seed_idx >= 0 && max_card > 0) {
            CUDA_CHECK(cudaMemcpy(best_members.data(),
                                  d_members + best_seed_idx * N,
                                  max_card * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        // free per-iteration device memory
        CUDA_CHECK(cudaFree(d_cardinalities));
        CUDA_CHECK(cudaFree(d_members));
        CUDA_CHECK(cudaFree(d_seeds));

        // add cluster to result
        if (best_seed_idx >= 0 && max_card > 0) {
            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_seed_idx];
            cluster.members = std::move(best_members);
            clusters.push_back(cluster);

            // update clustered array
            for (int m : clusters.back().members)
                clustered[m] = true;
            // convert std::vector<bool> to uint8_t for cudaMemcpy
            std::vector<uint8_t> clustered_bytes(N);
            for (int i = 0; i < N; ++i)
                clustered_bytes[i] = clustered[i] ? 1 : 0;
            CUDA_CHECK(cudaMemcpy(d_clustered, clustered_bytes.data(),
                                  N * sizeof(uint8_t), cudaMemcpyHostToDevice));

            // remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(),
                               unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    // free device memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));

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

// QT Clustering Benchmark - CUDA Version
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

static const int BLOCK_THREADS = 256;
static const int NUM_WARPS = BLOCK_THREADS / 32;

// Candidate selection key: smaller value wins, ties go to the lower index
// (matches the strict '<' scan in ascending index order of the original).
__device__ __forceinline__ bool better(double v, int i, double bv, int bi) {
    return v < bv || (v == bv && i < bi);
}

// One thread block grows complete candidate clusters for seeds taken from a
// dynamic work queue. For each candidate point it maintains the maximum
// distance to the current cluster members, so each growth step is one O(N)
// pass (update with the newly added member + arg-min reduction).
// maxd[c] == +inf marks points that are clustered or already in the cluster.
// If members_out != nullptr, the member list (in insertion order) is recorded.
__global__ void growClustersKernel(const double2* __restrict__ points,
                                   const char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int num_seeds,
                                   int* __restrict__ work_counter,
                                   int* __restrict__ cardinality,
                                   double* __restrict__ scratch,
                                   int* __restrict__ members_out,
                                   const double threshold,
                                   const int N) {
    __shared__ double s_val[2][NUM_WARPS];
    __shared__ int s_idx[2][NUM_WARPS];
    __shared__ int s_work;

    double* maxd = scratch + static_cast<size_t>(blockIdx.x) * N;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const double inf = __longlong_as_double(0x7ff0000000000000LL);

    while (true) {
        __syncthreads();
        if (tid == 0) s_work = atomicAdd(work_counter, 1);
        __syncthreads();
        const int w = s_work;
        if (w >= num_seeds) break;

        const int seed = seeds[w];
        for (int c = tid; c < N; c += BLOCK_THREADS) {
            maxd[c] = clustered[c] ? inf : 0.0;
        }

        int member = seed;
        int count = 1;
        if (members_out && tid == 0) members_out[0] = seed;
        int buf = 0;

        while (count < N) {
            const double2 pm = points[member];
            double best_v = inf;
            int best_i = -1;
            for (int c = tid; c < N; c += BLOCK_THREADS) {
                double d = maxd[c];
                if (c == member) {
                    d = inf;
                } else if (d != inf) {
                    const double2 pc = __ldg(&points[c]);
                    const double dx = pc.x - pm.x;
                    const double dy = pc.y - pm.y;
                    d = fmax(d, sqrt(dx * dx + dy * dy));
                }
                maxd[c] = d;
                if (d < threshold && d < best_v) {
                    best_v = d;
                    best_i = c;
                }
            }
            // warp reduction
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, best_v, off);
                const int oi = __shfl_down_sync(0xffffffffu, best_i, off);
                if (oi >= 0 && (best_i < 0 || better(ov, oi, best_v, best_i))) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (lane == 0) {
                s_val[buf][warp] = best_v;
                s_idx[buf][warp] = best_i;
            }
            __syncthreads();
            double gv = inf;
            int gi = -1;
            #pragma unroll
            for (int k = 0; k < NUM_WARPS; ++k) {
                const int oi = s_idx[buf][k];
                const double ov = s_val[buf][k];
                if (oi >= 0 && (gi < 0 || better(ov, oi, gv, gi))) {
                    gv = ov;
                    gi = oi;
                }
            }
            buf ^= 1;
            if (gi < 0) break;
            member = gi;
            if (members_out && tid == 0) members_out[count] = member;
            ++count;
        }
        if (tid == 0) cardinality[w] = count;
    }
}

__global__ void markClusteredKernel(char* clustered, const int* members, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) clustered[members[i]] = 1;
}

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int num_sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Per-block scratch: N doubles. Limit to ~60% of free memory.
    size_t max_blocks_mem = static_cast<size_t>(free_mem * 0.6) / (sizeof(double) * static_cast<size_t>(N));
    int max_blocks = std::min<size_t>(static_cast<size_t>(num_sms) * 6, max_blocks_mem);
    if (max_blocks < 1) max_blocks = 1;

    double2* d_points;
    char* d_clustered;
    int *d_seeds, *d_card, *d_counter, *d_members;
    double* d_scratch;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(double) * static_cast<size_t>(max_blocks) * N));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));

    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;
    std::vector<int> card;
    std::vector<int> members;
    std::vector<char> is_clustered(N, 0);

    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());
        int best_w = 0;
        int max_card = -1;

        if (U == 1) {
            best_w = 0;
            max_card = 1;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_seeds, unclustered.data(), sizeof(int) * U,
                                       cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));
            const int blocks = std::min(U, max_blocks);
            growClustersKernel<<<blocks, BLOCK_THREADS>>>(
                d_points, d_clustered, d_seeds, U, d_counter, d_card, d_scratch,
                nullptr, threshold, N);
            CUDA_CHECK(cudaGetLastError());
            card.resize(U);
            CUDA_CHECK(cudaMemcpy(card.data(), d_card, sizeof(int) * U, cudaMemcpyDeviceToHost));
            // First seed (in list order) with maximal cardinality wins
            for (int w = 0; w < U; ++w) {
                if (card[w] > max_card) {
                    max_card = card[w];
                    best_w = w;
                }
            }
        }

        const int best_seed = unclustered[best_w];

        // Re-grow the winning cluster, this time recording its members
        CUDA_CHECK(cudaMemcpyAsync(d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int)));
        growClustersKernel<<<1, BLOCK_THREADS>>>(
            d_points, d_clustered, d_seeds, 1, d_counter, d_card, d_scratch,
            d_members, threshold, N);
        CUDA_CHECK(cudaGetLastError());
        members.resize(max_card);
        CUDA_CHECK(cudaMemcpy(members.data(), d_members, sizeof(int) * max_card,
                              cudaMemcpyDeviceToHost));
        markClusteredKernel<<<(max_card + 255) / 256, 256>>>(d_clustered, d_members, max_card);
        CUDA_CHECK(cudaGetLastError());

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = members;
        clusters.push_back(std::move(cluster));

        for (int m : members) is_clustered[m] = 1;
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&is_clustered](int idx) { return is_clustered[idx] != 0; }),
            unclustered.end());
    }

    cudaFree(d_points);
    cudaFree(d_clustered);
    cudaFree(d_seeds);
    cudaFree(d_card);
    cudaFree(d_counter);
    cudaFree(d_members);
    cudaFree(d_scratch);
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
    
    // Initialize the CUDA context before timing
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

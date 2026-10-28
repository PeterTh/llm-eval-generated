// QT Clustering Benchmark - CUDA GPU-Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * Every candidate seed of a round grows its candidate cluster in its own
//     CUDA thread block. Seeds are processed in memory-bounded batches.
//   * For each candidate point, the maximum distance to the current cluster
//     members (its "diameter increase") is maintained incrementally: adding a
//     member costs one new distance per live candidate. max() is exact, so
//     this gives the same values as recomputing the max over all members.
//   * A candidate whose diameter increase reaches the threshold can never be
//     selected again (the value only grows), so it is dropped from the
//     per-seed candidate list. The lists shrink quickly as clusters grow.
//   * The closest candidate is selected with a block-wide argmin on the key
//     (distance, point index), which reproduces the sequential tie-breaking
//     (smallest distance, lowest index among equals). List order is therefore
//     irrelevant to the result.
//   * The best seed across batches is chosen on the host with a strict '>'
//     in the original seed order, matching the sequential semantics.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Threads per block for the cluster-growing kernel (multiple of 32)
static const int BLOCK_THREADS = 256;
static const int MAX_WARPS = BLOCK_THREADS / 32;

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

// Calculate Euclidean distance between two points (host reference, used by validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Device distance. Evaluates exactly sqrt(fma(dx, dx, round(dy*dy))), which is
// the contraction the host compiler applies to the expression above, so that
// GPU distances are bit-identical to the sequential reference.
__device__ __forceinline__ double deviceDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
}

__device__ __forceinline__ double deviceInf() {
    return __longlong_as_double(0x7FF0000000000000LL);
}

// Keep (v, i) if it is lexicographically smaller: smaller value, or equal value and smaller index
__device__ __forceinline__ void keepMin(double& bv, int& bi, const double v, const int i) {
    if (v < bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// Block-wide argmin over (value, index) pairs. Every thread receives the same
// winning index (INT_MAX if no entry was valid). Uses double-buffered shared
// memory selected by 'parity' so that only one barrier is needed per call.
__device__ int blockArgMin(double v, int i, const int parity,
                           double (*wv)[MAX_WARPS], int (*wi)[MAX_WARPS]) {
    const unsigned full = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(full, v, off);
        const int oi = __shfl_down_sync(full, i, off);
        keepMin(v, i, ov, oi);
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        wv[parity][warp] = v;
        wi[parity][warp] = i;
    }
    __syncthreads();

    const int nWarps = blockDim.x >> 5;
    double bv = deviceInf();
    int bi = INT_MAX;
    for (int w = 0; w < nWarps; ++w) {
        keepMin(bv, bi, wv[parity][w], wi[parity][w]);
    }
    return bi;
}

// One block per seed: grows the candidate cluster starting from seeds[blockIdx.x]
// and writes its cardinality and member sequence (insertion order).
//
// Per seed b, the live candidate list is double-buffered:
//   candIdx[(2b + k)*N + e] : point index of live candidate e (buffer k)
//   candVal[(2b + k)*N + e] : its current max distance to the cluster members (< threshold)
// Live candidates are unclustered, not in the cluster, and have value < threshold.
//   members[b*N + k] : k-th member of the candidate cluster for seed b
__global__ void growCandidateClusters(const Point* __restrict__ points,
                                      const int N,
                                      const double threshold,
                                      const int* __restrict__ unclustered,
                                      const int numUnclustered,
                                      const int* __restrict__ seeds,
                                      int* __restrict__ candIdx,
                                      double* __restrict__ candVal,
                                      int* __restrict__ members,
                                      int* __restrict__ cardinality) {
    __shared__ double wv[2][MAX_WARPS];
    __shared__ int wi[2][MAX_WARPS];
    // Output-size counters, rotated over three slots. Slot ph is reset one round
    // before it is reused, so no thread can increment it while it is being reset.
    __shared__ int shCnt[3];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int seed = seeds[b];
    const double INF = deviceInf();

    const Point seedPt = points[seed];
    int* mem = members + static_cast<size_t>(b) * N;
    int* idxBuf[2] = {candIdx + (static_cast<size_t>(2 * b) * N),
                      candIdx + (static_cast<size_t>(2 * b + 1) * N)};
    double* valBuf[2] = {candVal + (static_cast<size_t>(2 * b) * N),
                         candVal + (static_cast<size_t>(2 * b + 1) * N)};

    if (tid == 0) {
        shCnt[0] = 0;
        shCnt[1] = 0;
        shCnt[2] = 0;
        mem[0] = seed;
    }
    __syncthreads();

    int parity = 0;  // selects the wv/wi buffer used by the current barrier round
    int ph = 0;      // selects the shCnt slot used by the current round
    int cur = 0;     // selects the live-list buffer holding the current list
    int count = 1;   // cluster cardinality so far

    // Initial live list: all unclustered points (except the seed) within threshold of the seed
    int n;
    int sel;
    {
        int* outIdx = idxBuf[cur];
        double* outVal = valBuf[cur];
        double lv = INF;
        int li = INT_MAX;
        for (int j = tid; j < numUnclustered; j += stride) {
            const int c = unclustered[j];
            if (c == seed) continue;
            const double d = deviceDistance(points[c], seedPt);
            if (d < threshold) {
                const int pos = atomicAdd(&shCnt[ph], 1);
                outIdx[pos] = c;
                outVal[pos] = d;
                keepMin(lv, li, d, c);
            }
        }
        sel = blockArgMin(lv, li, parity, wv, wi);
        n = shCnt[ph];
        if (tid == 0) shCnt[(ph + 2) % 3] = 0;  // slot of the previous round, no longer read
        parity ^= 1;
        ph = (ph + 1) % 3;
    }

    while (sel != INT_MAX) {
        if (tid == 0) mem[count] = sel;
        ++count;

        // Add 'sel' to the cluster: bring every live candidate's max distance up to date
        // and keep only those still below the threshold. 'sel' itself is dropped.
        const int* inIdx = idxBuf[cur];
        const double* inVal = valBuf[cur];
        int* outIdx = idxBuf[cur ^ 1];
        double* outVal = valBuf[cur ^ 1];
        const Point selPt = points[sel];

        double lv = INF;
        int li = INT_MAX;
        for (int e = tid; e < n; e += stride) {
            const int c = inIdx[e];
            if (c == sel) continue;
            const double dist = deviceDistance(points[c], selPt);
            const double old = inVal[e];
            const double d = (dist > old) ? dist : old;  // max(current, new distance)
            if (d < threshold) {
                const int pos = atomicAdd(&shCnt[ph], 1);
                outIdx[pos] = c;
                outVal[pos] = d;
                keepMin(lv, li, d, c);
            }
        }
        sel = blockArgMin(lv, li, parity, wv, wi);
        n = shCnt[ph];
        if (tid == 0) shCnt[(ph + 2) % 3] = 0;  // slot of the previous round, no longer read
        parity ^= 1;
        ph = (ph + 1) % 3;
        cur ^= 1;
    }

    if (tid == 0) cardinality[b] = count;
}

// Main QT clustering algorithm (GPU-parallel)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // Size seed batches so that per-seed device storage fits comfortably in free
    // device memory: two live-list buffers (int + double per entry) and member list.
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t bytesPerSeed = static_cast<size_t>(N) *
                                (2 * (sizeof(int) + sizeof(double)) + sizeof(int));
    size_t batchCapSz = (freeBytes / 2) / bytesPerSeed;
    if (batchCapSz < 1) batchCapSz = 1;
    const int batchCap = static_cast<int>(std::min<size_t>(batchCapSz, static_cast<size_t>(N)));

    Point* d_points = nullptr;
    int* d_unclustered = nullptr;
    int* d_seeds = nullptr;
    int* d_candIdx = nullptr;
    double* d_candVal = nullptr;
    int* d_members = nullptr;
    int* d_card = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_unclustered, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * batchCap));
    CUDA_CHECK(cudaMalloc(&d_candIdx, sizeof(int) * static_cast<size_t>(batchCap) * 2 * N));
    CUDA_CHECK(cudaMalloc(&d_candVal, sizeof(double) * static_cast<size_t>(batchCap) * 2 * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * static_cast<size_t>(batchCap) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * batchCap));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    std::vector<int> h_card(batchCap);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());
        CUDA_CHECK(cudaMemcpy(d_unclustered, unclustered_indices.data(), sizeof(int) * U,
                              cudaMemcpyHostToDevice));

        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        // Grow a candidate cluster from every unclustered seed, in batches.
        // Strict '>' in seed order keeps the first maximal seed, as in the sequential code.
        for (int start = 0; start < U; start += batchCap) {
            const int cnt = std::min(batchCap, U - start);

            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data() + start,
                                  sizeof(int) * cnt, cudaMemcpyHostToDevice));

            growCandidateClusters<<<cnt, BLOCK_THREADS>>>(d_points, N, threshold,
                                                          d_unclustered, U, d_seeds,
                                                          d_candIdx, d_candVal,
                                                          d_members, d_card);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(h_card.data(), d_card, sizeof(int) * cnt,
                                  cudaMemcpyDeviceToHost));

            int local_b = -1;
            int local_card = -1;
            for (int b = 0; b < cnt; ++b) {
                if (h_card[b] > local_card) {
                    local_card = h_card[b];
                    local_b = b;
                }
            }

            if (local_card > max_cardinality) {
                max_cardinality = local_card;
                best_seed = unclustered_indices[start + local_b];
                best_cluster_members.resize(local_card);
                CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                      d_members + static_cast<size_t>(local_b) * N,
                                      sizeof(int) * local_card, cudaMemcpyDeviceToHost));
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

            // Remove clustered points from unclustered list (order preserved)
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

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_unclustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_candIdx));
    CUDA_CHECK(cudaFree(d_candVal));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_card));

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
    
    // Initialize the CUDA runtime/context before timing so that one-time driver
    // startup cost is not attributed to the clustering computation.
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

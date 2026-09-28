// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Every round, one candidate cluster is grown per remaining unclustered
//    seed. Each candidate growth is assigned to one CUDA thread block; the
//    threads of the block cooperatively maintain, for every candidate point,
//    the maximum distance to the cluster members so far (incremental update),
//    and perform a block-wide argmin reduction to select the next member.
//  - Seeds are distributed across all available GPUs and processed in
//    batches sized to fit device memory.
//  - Tie-breaking in all reductions matches the sequential algorithm
//    exactly (smallest distance, then smallest point index; largest
//    cardinality, then earliest seed in the unclustered list).

#include <algorithm>
#include <chrono>
#include <cfloat>
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

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

__device__ inline double devDistance(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

static const int BLOCK_SIZE = 256;

// Grow one candidate cluster per block, starting from seeds[blockIdx.x].
// For every point, maxd holds the maximum distance to the members added so
// far (DBL_MAX marks points that are unavailable: already globally
// clustered, or already members of this candidate cluster). Each step the
// block selects the point with the smallest such maximum distance below the
// threshold (ties resolved towards the smaller point index, matching the
// sequential scan order) and adds it to the cluster.
// Writes the resulting cardinality; if members != nullptr (single-seed
// launch) it also records the member indices in insertion order.
__global__ void growClustersKernel(const double2* __restrict__ points,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds,
                                   const int num_seeds,
                                   const int N,
                                   const double threshold,
                                   double* __restrict__ maxd_all,
                                   int* __restrict__ cardinality,
                                   int* __restrict__ members) {
    const int s = blockIdx.x;
    if (s >= num_seeds) return;
    const int seed = seeds[s];
    const int tid = threadIdx.x;
    double* maxd = maxd_all + static_cast<size_t>(s) * N;

    __shared__ double s_val[BLOCK_SIZE];
    __shared__ int s_idx[BLOCK_SIZE];
    __shared__ int s_chosen;
    __shared__ double2 s_newP;

    const double2 seedP = points[seed];
    for (int c = tid; c < N; c += BLOCK_SIZE) {
        maxd[c] = (clustered[c] || c == seed) ? DBL_MAX
                                              : devDistance(points[c], seedP);
    }
    if (members && tid == 0) members[0] = seed;
    __syncthreads();

    int count = 1;
    while (count < N) {
        // Block-wide argmin over available points with maxd < threshold
        double best = DBL_MAX;
        int best_idx = -1;
        for (int c = tid; c < N; c += BLOCK_SIZE) {
            const double v = maxd[c];
            if (v < threshold && v < best) {
                best = v;
                best_idx = c;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int off = BLOCK_SIZE / 2; off > 0; off >>= 1) {
            if (tid < off) {
                const double ov = s_val[tid + off];
                const int oi = s_idx[tid + off];
                if (oi >= 0 &&
                    (ov < s_val[tid] ||
                     (ov == s_val[tid] && (s_idx[tid] < 0 || oi < s_idx[tid])))) {
                    s_val[tid] = ov;
                    s_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_chosen = s_idx[0];
            if (s_chosen >= 0) s_newP = points[s_chosen];
        }
        __syncthreads();

        const int chosen = s_chosen;
        if (chosen < 0) break;  // No more points can be added
        if (members && tid == 0) members[count] = chosen;
        count++;

        // Fold the new member's distances into the running maxima
        const double2 np = s_newP;
        for (int c = tid; c < N; c += BLOCK_SIZE) {
            const double v = maxd[c];
            if (c == chosen) {
                maxd[c] = DBL_MAX;
            } else if (v != DBL_MAX) {
                const double d = devDistance(points[c], np);
                if (d > v) maxd[c] = d;
            }
        }
        __syncthreads();
    }

    if (tid == 0) cardinality[s] = count;
}

// Per-GPU device state
struct GpuState {
    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;  // only allocated on device 0
    double* d_maxd = nullptr;
    int batch = 0;  // max seeds processed per kernel launch
};

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices (kept sorted ascending throughout)
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(1);
    }

    std::vector<GpuState> gpus(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        GpuState& st = gpus[g];
        CUDA_CHECK(cudaMalloc(&st.d_points, sizeof(double2) * N));
        CUDA_CHECK(cudaMemcpy(st.d_points, points.data(), sizeof(double2) * N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&st.d_clustered, N));
        CUDA_CHECK(cudaMemset(st.d_clustered, 0, N));
        CUDA_CHECK(cudaMalloc(&st.d_seeds, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&st.d_card, sizeof(int) * N));
        if (g == 0) {
            CUDA_CHECK(cudaMalloc(&st.d_members, sizeof(int) * N));
        }

        // Size the per-seed working set (one row of N doubles per seed in
        // flight) to fit in 80% of the currently free device memory.
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t budget = static_cast<size_t>(free_mem * 0.8);
        size_t row_bytes = sizeof(double) * static_cast<size_t>(N);
        size_t max_rows = budget / row_bytes;
        if (max_rows < 1) max_rows = 1;
        st.batch = static_cast<int>(std::min<size_t>(max_rows, N));
        CUDA_CHECK(cudaMalloc(&st.d_maxd, row_bytes * st.batch));
    }

    std::vector<int> cardinalities;
    cardinalities.reserve(N);

    // Main clustering loop: each round grows one candidate cluster per
    // remaining seed (in parallel across blocks and GPUs), keeps the largest
    // one, and retires its members.
    while (!unclustered_indices.empty()) {
        const int M = static_cast<int>(unclustered_indices.size());
        // Use fewer GPUs when there is too little work to amortize the
        // per-device launch/sync overhead.
        int use_gpus = std::min(num_gpus, std::max(1, M / 256));
        cardinalities.assign(M, 0);

        // Partition seeds contiguously across GPUs (preserves global order)
        const int per_gpu = (M + use_gpus - 1) / use_gpus;
        for (int g = 0; g < use_gpus; ++g) {
            const int start = g * per_gpu;
            const int cnt = std::min(per_gpu, M - start);
            if (cnt <= 0) continue;
            GpuState& st = gpus[g];
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpyAsync(st.d_seeds,
                                       unclustered_indices.data() + start,
                                       sizeof(int) * cnt,
                                       cudaMemcpyHostToDevice));
            for (int off = 0; off < cnt; off += st.batch) {
                const int chunk = std::min(st.batch, cnt - off);
                growClustersKernel<<<chunk, BLOCK_SIZE>>>(
                    st.d_points, st.d_clustered, st.d_seeds + off, chunk, N,
                    threshold, st.d_maxd, st.d_card + off, nullptr);
            }
        }
        for (int g = 0; g < use_gpus; ++g) {
            const int start = g * per_gpu;
            const int cnt = std::min(per_gpu, M - start);
            if (cnt <= 0) continue;
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(cardinalities.data() + start,
                                  gpus[g].d_card, sizeof(int) * cnt,
                                  cudaMemcpyDeviceToHost));
        }

        // Select the best seed: maximum cardinality, ties broken towards the
        // earliest position in the (ascending) unclustered list, matching
        // the sequential scan with a strict '>' comparison.
        int best_pos = 0;
        for (int i = 1; i < M; ++i) {
            if (cardinalities[i] > cardinalities[best_pos]) best_pos = i;
        }
        const int max_cardinality = cardinalities[best_pos];
        const int best_seed = unclustered_indices[best_pos];

        if (max_cardinality <= 1) {
            // Every remaining point is farther than the threshold from all
            // others; the sequential algorithm would emit them as singleton
            // clusters one round at a time, in ascending index order.
            for (int idx : unclustered_indices) {
                Cluster cluster;
                cluster.seed_point = idx;
                cluster.members.push_back(idx);
                clusters.push_back(cluster);
                clustered[idx] = 1;
            }
            unclustered_indices.clear();
            break;
        }

        // Re-grow the winning cluster on device 0, recording its members
        {
            GpuState& st = gpus[0];
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(st.d_seeds, &best_seed, sizeof(int),
                                  cudaMemcpyHostToDevice));
            growClustersKernel<<<1, BLOCK_SIZE>>>(
                st.d_points, st.d_clustered, st.d_seeds, 1, N, threshold,
                st.d_maxd, st.d_card, st.d_members);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(max_cardinality);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), gpus[0].d_members,
                              sizeof(int) * max_cardinality,
                              cudaMemcpyDeviceToHost));

        // Mark all members as clustered
        for (int m : cluster.members) {
            clustered[m] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list (order preserved)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );

        // Refresh the clustered flags on every device
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(gpus[g].d_clustered, clustered.data(), N,
                                  cudaMemcpyHostToDevice));
        }
    }

    for (int g = 0; g < num_gpus; ++g) {
        cudaSetDevice(g);
        cudaFree(gpus[g].d_points);
        cudaFree(gpus[g].d_clustered);
        cudaFree(gpus[g].d_seeds);
        cudaFree(gpus[g].d_card);
        cudaFree(gpus[g].d_maxd);
        if (gpus[g].d_members) cudaFree(gpus[g].d_members);
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

    // Initialize CUDA contexts on all devices before timing
    {
        int num_gpus = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        for (int g = 0; g < num_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(nullptr));
        }
        CUDA_CHECK(cudaSetDevice(0));
    }

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

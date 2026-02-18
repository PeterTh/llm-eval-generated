// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering parallelized with CUDA.
// Each unclustered seed is evaluated in parallel (one CUDA block per seed).
// Threads within a block cooperatively find the closest point via reduction.

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

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

// CPU distance for validation
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static const int CAND_BLOCK = 256;

// Tie-breaking: prefer smaller squared distance, then smaller index
__device__ bool isBetter(double d_new, int idx_new, double d_old, int idx_old) {
    if (idx_new < 0) return false;
    if (idx_old < 0) return true;
    if (d_new < d_old) return true;
    if (d_new == d_old && idx_new < idx_old) return true;
    return false;
}

// Each block builds a candidate cluster for one seed.
// Threads cooperatively find the closest point via parallel reduction.
__global__ void generateCandidatesKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ clustered,
    const int* __restrict__ seeds,
    int num_seeds,
    int N,
    double threshold_sq,
    int* __restrict__ cardinalities,
    int* __restrict__ all_members,
    unsigned char* __restrict__ in_cluster_ws)
{
    extern __shared__ char smem[];
    double* s_dist = reinterpret_cast<double*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_dist + blockDim.x);

    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int seed = seeds[sid];
    long long off = static_cast<long long>(sid) * N;
    int* my_mem = all_members + off;
    unsigned char* my_in = in_cluster_ws + off;

    for (int i = threadIdx.x; i < N; i += blockDim.x)
        my_in[i] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        my_in[seed] = 1;
        my_mem[0] = seed;
    }
    __syncthreads();

    int mcnt = 1;

    while (mcnt < N) {
        double best_d = 1.0e30;
        int best_c = -1;

        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (clustered[c] || my_in[c]) continue;

            double cx = px[c], cy = py[c];
            double max_dsq = 0.0;
            bool ok = true;

            for (int m = 0; m < mcnt; m++) {
                int member = my_mem[m];
                double dx = cx - px[member];
                double dy = cy - py[member];
                double dsq = dx * dx + dy * dy;
                if (dsq >= threshold_sq) { ok = false; break; }
                if (dsq > max_dsq) max_dsq = dsq;
            }

            if (ok && isBetter(max_dsq, c, best_d, best_c)) {
                best_d = max_dsq;
                best_c = c;
            }
        }

        s_dist[threadIdx.x] = best_d;
        s_idx[threadIdx.x] = best_c;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (isBetter(s_dist[threadIdx.x + s], s_idx[threadIdx.x + s],
                             s_dist[threadIdx.x], s_idx[threadIdx.x])) {
                    s_dist[threadIdx.x] = s_dist[threadIdx.x + s];
                    s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
                }
            }
            __syncthreads();
        }

        if (s_idx[0] < 0) break;

        if (threadIdx.x == 0) {
            my_in[s_idx[0]] = 1;
            my_mem[mcnt] = s_idx[0];
        }
        __syncthreads();
        mcnt++;
    }

    if (threadIdx.x == 0)
        cardinalities[sid] = mcnt;
}

// Main QT clustering algorithm using CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered_h(N, 0);
    std::vector<Cluster> clusters;

    // Structure-of-Arrays layout for GPU
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) { px[i] = points[i].x; py[i] = points[i].y; }

    // Allocate device memory for point coordinates
    double *d_px, *d_py;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate clustered flags on device
    int *d_clustered;
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered_h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine batch size from available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    long long per_seed = static_cast<long long>(N) * (sizeof(int) + sizeof(unsigned char));
    long long overhead = static_cast<long long>(N) * sizeof(int) * 2 + (1LL << 26);
    long long avail = static_cast<long long>(free_mem) - overhead;
    int max_batch = (avail > per_seed)
        ? static_cast<int>(std::min(static_cast<long long>(N), avail / per_seed))
        : 1;

    int *d_seeds, *d_card, *d_members;
    unsigned char *d_in;
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<long long>(max_batch) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<long long>(max_batch) * N));

    double thr_sq = threshold * threshold;
    size_t smem_size = CAND_BLOCK * sizeof(double) + CAND_BLOCK * sizeof(int);

    std::vector<int> unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; i++) unclustered.push_back(i);
    std::vector<int> h_card(N);

    while (!unclustered.empty()) {
        int ns = static_cast<int>(unclustered.size());
        int best_card = -1, best_seed = -1;

        // Evaluate all seeds in batches
        for (int bs = 0; bs < ns; bs += max_batch) {
            int bsz = std::min(max_batch, ns - bs);
            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data() + bs,
                                  bsz * sizeof(int), cudaMemcpyHostToDevice));

            generateCandidatesKernel<<<bsz, CAND_BLOCK, smem_size>>>(
                d_px, d_py, d_clustered, d_seeds, bsz, N, thr_sq,
                d_card, d_members, d_in);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_card.data(), d_card,
                                  bsz * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < bsz; i++) {
                int seed_val = unclustered[bs + i];
                if (h_card[i] > best_card ||
                    (h_card[i] == best_card && seed_val < best_seed)) {
                    best_card = h_card[i];
                    best_seed = seed_val;
                }
            }
        }

        if (best_card <= 0 || best_seed < 0) break;

        // Re-run the single best seed to retrieve its member list
        CUDA_CHECK(cudaMemcpy(d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
        generateCandidatesKernel<<<1, CAND_BLOCK, smem_size>>>(
            d_px, d_py, d_clustered, d_seeds, 1, N, thr_sq,
            d_card, d_members, d_in);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> members(best_card);
        CUDA_CHECK(cudaMemcpy(members.data(), d_members,
                              best_card * sizeof(int), cudaMemcpyDeviceToHost));

        Cluster cl;
        cl.seed_point = best_seed;
        cl.members = members;
        clusters.push_back(cl);

        for (int m : members) clustered_h[m] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered_h.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered_h](int idx) { return clustered_h[idx]; }),
            unclustered.end());
    }

    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_in));

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

// QT Clustering Benchmark - CUDA GPU Parallel Version
//
// Parallelization: precompute N×N distance matrix on GPU, then for each
// clustering iteration build candidate clusters in parallel (one thread
// block per seed, threads cooperate via parallel reduction on findClosestPoint).

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

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

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

// Host-side distance for validation
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------- CUDA Kernels ----------

__global__ void distanceMatrixKernel(const double* __restrict__ px,
                                     const double* __restrict__ py,
                                     double* __restrict__ dist,
                                     int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        double dx = px[i] - px[j];
        double dy = py[i] - py[j];
        dist[(long long)i * N + j] = sqrt(dx * dx + dy * dy);
    }
}

constexpr int CLUSTER_BS = 256;

// One block per seed; threads cooperate on findClosestPoint via reduction
__global__ void buildCandidatesKernel(
    const double* __restrict__ dist_matrix,
    const int* __restrict__ seeds, int num_seeds,
    const int* __restrict__ clustered, int N, double threshold,
    int* __restrict__ cardinalities,
    int* __restrict__ members_buf,
    int* __restrict__ in_cluster_buf)
{
    int bid = blockIdx.x;
    if (bid >= num_seeds) return;

    int tid = threadIdx.x;
    int bs  = blockDim.x;
    int seed = seeds[bid];

    int* members  = members_buf    + (long long)bid * N;
    int* in_clust = in_cluster_buf + (long long)bid * N;

    for (int i = tid; i < N; i += bs) in_clust[i] = 0;
    __syncthreads();

    if (tid == 0) {
        members[0] = seed;
        in_clust[seed] = 1;
    }
    __syncthreads();

    int cur_size = 1;

    __shared__ double s_dist[CLUSTER_BS];
    __shared__ int    s_idx[CLUSTER_BS];

    while (cur_size < N) {
        double local_min = 1e308;
        int    local_pt  = -1;

        for (int c = tid; c < N; c += bs) {
            if (clustered[c] || in_clust[c]) continue;

            double max_d = 0.0;
            for (int m = 0; m < cur_size; m++) {
                double d = dist_matrix[(long long)members[m] * N + c];
                if (d > max_d) max_d = d;
            }

            if (max_d < threshold && max_d < local_min) {
                local_min = max_d;
                local_pt  = c;
            }
        }

        s_dist[tid] = local_min;
        s_idx[tid]  = local_pt;
        __syncthreads();

        // Reduction: minimum distance, ties broken by lower candidate index
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) {
                bool take = false;
                if (s_idx[tid] < 0 && s_idx[tid + s] >= 0) {
                    take = true;
                } else if (s_idx[tid] >= 0 && s_idx[tid + s] >= 0) {
                    if (s_dist[tid + s] < s_dist[tid])
                        take = true;
                    else if (s_dist[tid + s] == s_dist[tid] &&
                             s_idx[tid + s] < s_idx[tid])
                        take = true;
                }
                if (take) {
                    s_dist[tid] = s_dist[tid + s];
                    s_idx[tid]  = s_idx[tid + s];
                }
            }
            __syncthreads();
        }

        if (s_idx[0] < 0) break;

        if (tid == 0) {
            members[cur_size]    = s_idx[0];
            in_clust[s_idx[0]]  = 1;
        }
        __syncthreads();
        cur_size++;
    }

    if (tid == 0) cardinalities[bid] = cur_size;
}

// ---------- Host QT Clustering ----------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());

    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) { px[i] = points[i].x; py[i] = points[i].y; }

    // Allocate & compute distance matrix
    double *d_px, *d_py, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, (long long)N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        distanceMatrixKernel<<<grid, block>>>(d_px, d_py, d_dist, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));

    // Clustering state
    std::vector<int> h_clustered(N, 0);
    std::vector<int> unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; i++) unclustered.push_back(i);

    int *d_clustered;
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Workspace sizing – use up to 80% of free GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    long long per_seed = 2LL * N * static_cast<long long>(sizeof(int));
    int max_batch = static_cast<int>(std::min((long long)(free_mem * 0.8) / std::max(per_seed, 1LL),
                                              (long long)N));
    if (max_batch > 65535) max_batch = 65535;
    if (max_batch < 1)     max_batch = 1;

    int *d_seeds, *d_card, *d_members, *d_inclust;
    CUDA_CHECK(cudaMalloc(&d_seeds,   max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card,    max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, (long long)max_batch * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inclust, (long long)max_batch * N * sizeof(int)));

    std::vector<int> h_card(max_batch);
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        int num_unc = static_cast<int>(unclustered.size());
        int best_idx  = -1;
        int best_card = -1;

        for (int bstart = 0; bstart < num_unc; bstart += max_batch) {
            int bsize = std::min(max_batch, num_unc - bstart);
            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data() + bstart,
                                  bsize * sizeof(int), cudaMemcpyHostToDevice));

            buildCandidatesKernel<<<bsize, CLUSTER_BS>>>(
                d_dist, d_seeds, bsize, d_clustered, N, threshold,
                d_card, d_members, d_inclust);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_card.data(), d_card,
                                  bsize * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < bsize; i++) {
                if (h_card[i] > best_card) {
                    best_card = h_card[i];
                    best_idx  = bstart + i;
                }
            }
        }

        if (best_idx < 0 || best_card <= 0) break;

        int best_seed = unclustered[best_idx];
        std::vector<int> best_members(best_card);

        // If single batch, members are still on GPU
        if (num_unc <= max_batch) {
            CUDA_CHECK(cudaMemcpy(best_members.data(),
                                  d_members + (long long)best_idx * N,
                                  best_card * sizeof(int), cudaMemcpyDeviceToHost));
        } else {
            // Re-run single seed to retrieve members
            CUDA_CHECK(cudaMemcpy(d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
            buildCandidatesKernel<<<1, CLUSTER_BS>>>(
                d_dist, d_seeds, 1, d_clustered, N, threshold,
                d_card, d_members, d_inclust);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(best_members.data(), d_members,
                                  best_card * sizeof(int), cudaMemcpyDeviceToHost));
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members    = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) h_clustered[m] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&h_clustered](int idx) { return h_clustered[idx]; }),
            unclustered.end());
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_inclust));

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

// QT Clustering Benchmark - CUDA Parallel Version

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

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Compute pairwise squared distances
__global__ void compute_distances_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int N,
    double* __restrict__ dist_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int row = i / N;
        int col = i % N;
        double dx = px[row] - px[col];
        double dy = py[row] - py[col];
        dist_sq[i] = dx * dx + dy * dy;
    }
}

// Evaluate all seeds in parallel - one block per seed
__global__ void evaluate_seeds_kernel(
    const double* __restrict__ dist_sq,
    const int* __restrict__ seeds,
    const int* __restrict__ clustered,
    const int num_seeds,
    const int point_count,
    const double threshold_sq,
    int* __restrict__ sizes,
    char* __restrict__ in_cluster_global,
    int* __restrict__ members_global
) {
    extern __shared__ char smem[];

    const int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    const int seed = seeds[seed_idx];
    const int nwarps = blockDim.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int wid = threadIdx.x >> 5;

    int* s_members = (int*)smem;
    double* s_red_val = (double*)(smem + point_count * sizeof(int));
    int* s_red_idx = (int*)(s_red_val + nwarps);

    char* my_ic = in_cluster_global + (long long)seed_idx * point_count;
    int* my_members = members_global + (long long)seed_idx * point_count;

    // Initialize in_cluster
    for (int i = threadIdx.x; i < point_count; i += blockDim.x)
        my_ic[i] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        my_ic[seed] = 1;
        s_members[0] = seed;
        my_members[0] = seed;
    }
    __syncthreads();

    int cluster_size = 1;

    // Iteratively add closest point
    while (cluster_size < point_count) {
        double best_val = 1e300;
        int best_cand = point_count;

        // Each thread evaluates a subset of candidates
        for (int cand = threadIdx.x; cand < point_count; cand += blockDim.x) {
            if (clustered[cand] || my_ic[cand]) continue;

            double max_d2 = 0.0;
            
            // Compute max distance to all cluster members
            for (int m = 0; m < cluster_size; ++m) {
                const int mi = s_members[m];
                const double d2 = dist_sq[(long long)cand * point_count + mi];
                max_d2 = fmax(max_d2, d2);
            }

            // Strict < ensures tie-breaking by smallest candidate index
            if (max_d2 < threshold_sq && max_d2 < best_val) {
                best_val = max_d2;
                best_cand = cand;
            }
        }

        // Block-level argmin reduction
        double wv = best_val;
        int wi = best_cand;
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            double ov = __shfl_down_sync(0xffffffff, wv, off);
            int oi = __shfl_down_sync(0xffffffff, wi, off);
            if (ov < wv || (ov == wv && oi < wi)) {
                wv = ov;
                wi = oi;
            }
        }

        if (lane == 0) {
            s_red_val[wid] = wv;
            s_red_idx[wid] = wi;
        }
        __syncthreads();

        int winner = -1;
        if (wid == 0) {
            double lv = (threadIdx.x < nwarps) ? s_red_val[threadIdx.x] : 1e300;
            int li = (threadIdx.x < nwarps) ? s_red_idx[threadIdx.x] : point_count;
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                double ov = __shfl_down_sync(0xffffffff, lv, off);
                int oi = __shfl_down_sync(0xffffffff, li, off);
                if (ov < lv || (ov == lv && oi < li)) {
                    lv = ov;
                    li = oi;
                }
            }
            if (threadIdx.x == 0) {
                winner = (lv < 1e299 && li < point_count) ? li : -1;
                s_red_idx[0] = winner;
            }
        }
        __syncthreads();
        winner = s_red_idx[0];

        if (winner < 0) break;

        if (threadIdx.x == 0) {
            my_ic[winner] = 1;
            s_members[cluster_size] = winner;
            my_members[cluster_size] = winner;
        }
        __syncthreads();
        cluster_size++;
    }

    if (threadIdx.x == 0)
        sizes[seed_idx] = cluster_size;
}

// Mark cluster members as clustered
__global__ void mark_clustered_kernel(
    int* __restrict__ clustered,
    const int* __restrict__ members,
    const int size
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        clustered[members[i]] = 1;
    }
}

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

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;

    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Allocate GPU memory
    double* d_px = nullptr;
    double* d_py = nullptr;
    double* d_dist_sq = nullptr;
    int* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_sizes = nullptr;
    char* d_in_cluster = nullptr;
    int* d_members = nullptr;

    cudaMalloc(&d_px, N * sizeof(double));
    cudaMalloc(&d_py, N * sizeof(double));
    cudaMalloc(&d_dist_sq, (long long)N * N * sizeof(double));
    cudaMalloc(&d_clustered, N * sizeof(int));
    cudaMalloc(&d_seeds, N * sizeof(int));
    cudaMalloc(&d_sizes, N * sizeof(int));
    cudaMalloc(&d_in_cluster, (long long)N * N * sizeof(char));
    cudaMalloc(&d_members, (long long)N * N * sizeof(int));

    // Copy point data
    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }
    cudaMemcpy(d_px, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Compute distance matrix
    int dist_threads = 256;
    int dist_blocks = ((long long)N * N + dist_threads - 1) / dist_threads;
    if (dist_blocks > 4096) dist_blocks = 4096;
    compute_distances_kernel<<<dist_blocks, dist_threads>>>(d_px, d_py, N, d_dist_sq);

    cudaMemset(d_clustered, 0, N * sizeof(int));

    const int block_size = BLOCK_SIZE;
    const int nwarps = block_size / WARP_SIZE;
    size_t eval_smem = N * sizeof(int) + nwarps * sizeof(double) + nwarps * sizeof(int);

    cudaFuncSetAttribute(evaluate_seeds_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         eval_smem);

    std::vector<int> h_sizes(N);
    std::vector<int> h_seeds(N);
    std::vector<int> h_members((long long)N * N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Copy seeds to GPU
        for (int i = 0; i < num_seeds; ++i) {
            h_seeds[i] = unclustered_indices[i];
        }
        cudaMemcpy(d_seeds, h_seeds.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel: one block per seed
        evaluate_seeds_kernel<<<num_seeds, block_size, eval_smem>>>(
            d_dist_sq, d_seeds, d_clustered,
            num_seeds, N, threshold_sq,
            d_sizes, d_in_cluster, d_members
        );

        // Read back sizes
        cudaMemcpy(h_sizes.data(), d_sizes, num_seeds * sizeof(int), cudaMemcpyDeviceToHost);

        // Find best seed (max cardinality, tie-break by smallest seed index)
        int max_cardinality = -1;
        int best_seed_idx = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_sizes[i] > max_cardinality) {
                max_cardinality = h_sizes[i];
                best_seed_idx = i;
            }
        }

        if (max_cardinality <= 0) break;

        // Read members of best cluster
        cudaMemcpy(h_members.data(),
                   d_members + (long long)best_seed_idx * N,
                   max_cardinality * sizeof(int),
                   cudaMemcpyDeviceToHost);

        // Build cluster
        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_seed_idx];
        cluster.members.resize(max_cardinality);
        for (int i = 0; i < max_cardinality; ++i) {
            cluster.members[i] = h_members[i];
            clustered[h_members[i]] = true;
        }
        clusters.push_back(cluster);

        // Mark members as clustered on GPU
        int mark_blocks = (max_cardinality + block_size - 1) / block_size;
        mark_clustered_kernel<<<mark_blocks, block_size>>>(
            d_clustered,
            d_members + (long long)best_seed_idx * N,
            max_cardinality
        );

        // Update unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    // Free GPU memory
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_dist_sq);
    cudaFree(d_clustered);
    cudaFree(d_seeds);
    cudaFree(d_sizes);
    cudaFree(d_in_cluster);
    cudaFree(d_members);

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

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

    printf("QT Clustering Benchmark (CUDA)\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());

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

    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n",
           clusters_per_sec, points_per_sec);

    if (printResults) {
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

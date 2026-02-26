// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering parallelized with CUDA.
// Each seed's candidate cluster is built by one thread block.
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

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

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

// Each block builds a candidate cluster for one seed point.
// Threads cooperatively find the closest point at each step via parallel reduction.
__global__ __launch_bounds__(256)
void buildCandidateClustersKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ clustered,
    const int* __restrict__ seeds,
    int num_seeds,
    int N,
    double threshold,
    int* __restrict__ cardinalities,
    int* __restrict__ all_members,
    int* __restrict__ in_cluster_flags)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int seed = seeds[sid];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int* my_icf = in_cluster_flags + (long long)sid * N;
    int* my_mem = all_members + (long long)sid * N;

    extern __shared__ char smem[];
    double* s_dist = (double*)smem;
    int* s_point = (int*)(s_dist + bs);
    __shared__ int cluster_sz;
    __shared__ int best_pt;

    for (int i = tid; i < N; i += bs) my_icf[i] = 0;
    __syncthreads();
    if (tid == 0) {
        my_icf[seed] = 1;
        my_mem[0] = seed;
        cluster_sz = 1;
    }
    __syncthreads();

    for (int iter = 1; iter < N; iter++) {
        int csz = cluster_sz;

        double my_best_d = 1e30;
        int my_best_p = -1;

        for (int c = tid; c < N; c += bs) {
            if (clustered[c] || my_icf[c]) continue;

            double cx = px[c], cy = py[c];
            double max_d = 0.0;
            bool ok = true;

            for (int m = 0; m < csz; m++) {
                int member = my_mem[m];
                double dx = cx - px[member];
                double dy = cy - py[member];
                double d = sqrt(dx * dx + dy * dy);
                if (d >= threshold) { ok = false; break; }
                if (d > max_d) max_d = d;
            }

            if (ok && max_d < threshold) {
                if (max_d < my_best_d || (max_d == my_best_d && (my_best_p < 0 || c < my_best_p))) {
                    my_best_d = max_d;
                    my_best_p = c;
                }
            }
        }

        s_dist[tid] = my_best_d;
        s_point[tid] = my_best_p;
        __syncthreads();

        // Tree reduction: find candidate with minimum max-distance, ties broken by lowest index
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) {
                bool take = false;
                if (s_point[tid + s] >= 0) {
                    if (s_point[tid] < 0) take = true;
                    else if (s_dist[tid + s] < s_dist[tid]) take = true;
                    else if (s_dist[tid + s] == s_dist[tid] && s_point[tid + s] < s_point[tid]) take = true;
                }
                if (take) {
                    s_dist[tid] = s_dist[tid + s];
                    s_point[tid] = s_point[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) best_pt = s_point[0];
        __syncthreads();

        if (best_pt < 0) break;

        if (tid == 0) {
            my_icf[best_pt] = 1;
            my_mem[csz] = best_pt;
            cluster_sz = csz + 1;
        }
        __syncthreads();
    }

    if (tid == 0) cardinalities[sid] = cluster_sz;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    if (N == 0) return {};

    // Structure-of-arrays for GPU
    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; i++) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
    }

    double *d_px, *d_py;
    int *d_clustered, *d_seeds, *d_cardinalities, *d_members, *d_icf;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));

    // Determine workspace batch size from available GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t per_seed = 2 * (size_t)N * sizeof(int);
    size_t reserve = 128ULL * 1024 * 1024; // keep 128MB free
    size_t available = (free_mem > reserve) ? (free_mem - reserve) : (64ULL * 1024 * 1024);
    int max_batch = (int)std::min((size_t)N, available / per_seed);
    max_batch = std::max(max_batch, 1);

    CUDA_CHECK(cudaMalloc(&d_cardinalities, max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, (size_t)max_batch * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_icf, (size_t)max_batch * N * sizeof(int)));

    std::vector<int> h_clustered(N, 0);
    CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> unclustered;
    for (int i = 0; i < N; i++) unclustered.push_back(i);

    std::vector<Cluster> clusters;
    size_t smem_size = BLOCK_SIZE * (sizeof(double) + sizeof(int));

    while (!unclustered.empty()) {
        int K = static_cast<int>(unclustered.size());
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        int global_best_card = 0;
        int global_best_abs_idx = -1;
        std::vector<int> global_best_members;

        for (int batch_start = 0; batch_start < K; batch_start += max_batch) {
            int this_batch = std::min(max_batch, K - batch_start);

            buildCandidateClustersKernel<<<this_batch, BLOCK_SIZE, smem_size>>>(
                d_px, d_py, d_clustered, d_seeds + batch_start,
                this_batch, N, threshold,
                d_cardinalities, d_members, d_icf
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<int> h_card(this_batch);
            CUDA_CHECK(cudaMemcpy(h_card.data(), d_cardinalities,
                                  this_batch * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < this_batch; i++) {
                if (h_card[i] > global_best_card) {
                    global_best_card = h_card[i];
                    global_best_abs_idx = batch_start + i;
                    global_best_members.resize(h_card[i]);
                    CUDA_CHECK(cudaMemcpy(global_best_members.data(),
                                          d_members + (size_t)i * N,
                                          h_card[i] * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
            }
        }

        if (global_best_abs_idx < 0 || global_best_card <= 0) break;

        Cluster cluster;
        cluster.seed_point = unclustered[global_best_abs_idx];
        cluster.members = global_best_members;
        clusters.push_back(cluster);

        for (int m : global_best_members) h_clustered[m] = 1;

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                          [&h_clustered](int idx) { return h_clustered[idx] != 0; }),
            unclustered.end()
        );
    }

    cudaFree(d_px); cudaFree(d_py);
    cudaFree(d_clustered); cudaFree(d_seeds);
    cudaFree(d_cardinalities); cudaFree(d_members);
    cudaFree(d_icf);

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

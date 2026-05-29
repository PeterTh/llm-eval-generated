// QT Clustering Benchmark - CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// CUDA parallelization strategy:
// - Kernel 1 (maxDistKernel): parallel max-distance computation across all
//   candidate points for each cluster-growth iteration
// - Kernel 2 (findBestCandidateKernel): parallel reduction to find the
//   candidate with minimum max-distance (best next point to add)
// - Pre-allocated device buffers to minimize malloc/memcpy overhead
// - Data residency on GPU to minimize host-device transfers

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

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));      \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ============================================================
// CUDA Kernels
// ============================================================

// Kernel: compute max distance from each candidate to all cluster members.
// Each thread handles one candidate point. Writes max_dist per candidate.
// Candidates that are already clustered or in-cluster get -1.0.
__global__ void maxDistKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int*    __restrict__ cluster_members,
    const int     cluster_size,
    const int*    __restrict__ clustered,
    const int*    __restrict__ in_cluster,
    double*       __restrict__ max_dists,
    const int     N)
{
    const int cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= N) return;

    if (clustered[cand] || in_cluster[cand]) {
        max_dists[cand] = -1.0;
        return;
    }

    double max_dist = 0.0;
    const double cx = px[cand];
    const double cy = py[cand];
    for (int i = 0; i < cluster_size; ++i) {
        const int m = cluster_members[i];
        const double dx = cx - px[m];
        const double dy = cy - py[m];
        const double d = sqrt(dx * dx + dy * dy);
        if (d > max_dist) max_dist = d;
    }
    max_dists[cand] = max_dist;
}

// Kernel: find the candidate with minimum max_dist that is under threshold.
// Each block does a local reduction; block leaders write results atomically.
__global__ void findBestCandidateKernel(
    const double* __restrict__ max_dists,
    const int     N,
    const double  threshold,
    int*          best_index,
    double*       best_diameter)
{
    extern __shared__ char shared_mem[];
    double* s_dists = reinterpret_cast<double*>(shared_mem);
    int*    s_indices = reinterpret_cast<int*>(s_dists + blockDim.x);

    const int tid = threadIdx.x;

    double my_dist = -1.0;
    int    my_idx  = -1;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += gridDim.x * blockDim.x) {
        double d = max_dists[idx];
        if (d >= 0.0 && d < threshold) {
            if (my_dist < 0.0 || d < my_dist) {
                my_dist = d;
                my_idx  = idx;
            }
        }
    }

    s_dists[tid] = my_dist;
    s_indices[tid] = my_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other_d = s_dists[tid + s];
            int    other_i = s_indices[tid + s];
            if (other_d > 0.0 && (my_dist < 0.0 || other_d < my_dist)) {
                my_dist = other_d;
                my_idx  = other_i;
            }
        }
        s_dists[tid] = my_dist;
        s_indices[tid] = my_idx;
        __syncthreads();
    }

    if (tid == 0) {
        if (my_dist > 0.0) {
            double cur = *best_diameter;
            while (true) {
                if (cur < 0.0 || my_dist < cur) {
                    if (atomicCAS((unsigned long long*)best_diameter,
                                  __double_as_longlong(cur),
                                  __double_as_longlong(my_dist)) ==
                        __double_as_longlong(cur)) {
                        atomicExch(best_index, my_idx);
                        break;
                    }
                    cur = *best_diameter;
                } else {
                    break;
                }
            }
        }
    }
}

// Fused kernel: compute max distances AND find best candidate in one launch.
// This eliminates the intermediate max_dists buffer and one kernel launch.
__global__ void fusedFindClosestKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int*    __restrict__ cluster_members,
    const int     cluster_size,
    const int*    __restrict__ clustered,
    const int*    __restrict__ in_cluster,
    const int     N,
    const double  threshold,
    int*          best_index,
    double*       best_diameter)
{
    extern __shared__ char shared_mem[];
    double* s_dists = reinterpret_cast<double*>(shared_mem);
    int*    s_indices = reinterpret_cast<int*>(s_dists + blockDim.x);

    const int tid = threadIdx.x;

    double my_dist = -1.0;
    int    my_idx  = -1;

    // Each thread computes max_dist for its assigned candidates
    for (int cand = blockIdx.x * blockDim.x + tid; cand < N; cand += gridDim.x * blockDim.x) {
        if (clustered[cand] || in_cluster[cand]) continue;

        double max_dist = 0.0;
        const double cx = px[cand];
        const double cy = py[cand];
        for (int i = 0; i < cluster_size; ++i) {
            const int m = cluster_members[i];
            const double dx = cx - px[m];
            const double dy = cy - py[m];
            const double d = sqrt(dx * dx + dy * dy);
            if (d > max_dist) max_dist = d;
        }

        if (max_dist < threshold) {
            if (my_dist < 0.0 || max_dist < my_dist) {
                my_dist = max_dist;
                my_idx  = cand;
            }
        }
    }

    s_dists[tid] = my_dist;
    s_indices[tid] = my_idx;
    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other_d = s_dists[tid + s];
            int    other_i = s_indices[tid + s];
            if (other_d > 0.0 && (my_dist < 0.0 || other_d < my_dist)) {
                my_dist = other_d;
                my_idx  = other_i;
            }
        }
        s_dists[tid] = my_dist;
        s_indices[tid] = my_idx;
        __syncthreads();
    }

    // Block leader writes result atomically
    if (tid == 0) {
        if (my_dist > 0.0) {
            double cur = *best_diameter;
            while (true) {
                if (cur < 0.0 || my_dist < cur) {
                    if (atomicCAS((unsigned long long*)best_diameter,
                                  __double_as_longlong(cur),
                                  __double_as_longlong(my_dist)) ==
                        __double_as_longlong(cur)) {
                        atomicExch(best_index, my_idx);
                        break;
                    }
                    cur = *best_diameter;
                } else {
                    break;
                }
            }
        }
    }
}

// ============================================================
// Host functions
// ============================================================

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

// Find the closest unclustered point using CUDA (fused kernel)
int findClosestPointCUDA(
    const double* d_px, const double* d_py,
    const int* d_cluster_members, const int cluster_size,
    const int* d_clustered, const int* d_in_cluster,
    const double threshold, const int N,
    int* d_best_index, double* d_best_diameter)
{
    const int block_size = 256;
    const int grid_size = std::min((N + block_size - 1) / block_size, 1024);
    const int shared_mem_size = block_size * (sizeof(double) + sizeof(int));

    const double init_diameter = -1.0;
    CUDA_CHECK(cudaMemcpy(d_best_diameter, &init_diameter, sizeof(double), cudaMemcpyHostToDevice));
    const int init_index = -1;
    CUDA_CHECK(cudaMemcpy(d_best_index, &init_index, sizeof(int), cudaMemcpyHostToDevice));

    fusedFindClosestKernel<<<grid_size, block_size, shared_mem_size>>>(
        d_px, d_py,
        d_cluster_members, cluster_size,
        d_clustered, d_in_cluster,
        N, threshold,
        d_best_index, d_best_diameter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int result = -1;
    CUDA_CHECK(cudaMemcpy(&result, d_best_index, sizeof(int), cudaMemcpyDeviceToHost));

    return result;
}

// Generate a candidate cluster starting from a seed point using CUDA
int generateCandidateClusterCUDA(
    const double* d_px, const double* d_py,
    const int* d_clustered,
    const int seed_point, const double threshold, const int N,
    int* d_in_cluster, int* d_cluster_members,
    int* d_best_index, double* d_best_diameter,
    std::vector<int>* cluster_members)
{
    std::vector<int> in_cluster_int(N, 0);
    in_cluster_int[seed_point] = 1;
    
    CUDA_CHECK(cudaMemcpy(d_in_cluster, in_cluster_int.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<int> members;
    members.reserve(N);
    members.push_back(seed_point);

    CUDA_CHECK(cudaMemcpy(d_cluster_members, members.data(), sizeof(int), cudaMemcpyHostToDevice));

    while (static_cast<int>(members.size()) < N) {
        const int closest = findClosestPointCUDA(
            d_px, d_py,
            d_cluster_members, static_cast<int>(members.size()),
            d_clustered, d_in_cluster,
            threshold, N,
            d_best_index, d_best_diameter);

        if (closest < 0) break;

        members.push_back(closest);
        in_cluster_int[closest] = 1;
        
        CUDA_CHECK(cudaMemcpy(d_in_cluster + closest, &in_cluster_int[closest], sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cluster_members, members.data(), members.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm - CUDA parallelized
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }
    
    // Pre-allocate all device buffers once
    double* d_px = nullptr;
    double* d_py = nullptr;
    int*    d_clustered = nullptr;
    int*    d_in_cluster = nullptr;
    int*    d_cluster_members = nullptr;
    int*    d_best_index = nullptr;
    double* d_best_diameter = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_index, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_diameter, sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    std::vector<bool> h_clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (h_clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateClusterCUDA(
                d_px, d_py, d_clustered,
                seed, threshold, N,
                d_in_cluster, d_cluster_members,
                d_best_index, d_best_diameter,
                &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                h_clustered[best_cluster_members[i]] = true;
            }
            
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                const int idx = best_cluster_members[i];
                const int val = 1;
                CUDA_CHECK(cudaMemcpy(d_clustered + idx, &val, sizeof(int), cudaMemcpyHostToDevice));
            }
            
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&h_clustered](int idx) { return h_clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_cluster_members));
    CUDA_CHECK(cudaFree(d_best_index));
    CUDA_CHECK(cudaFree(d_best_diameter));
    
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
    
    printf("QT Clustering Benchmark\n");
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

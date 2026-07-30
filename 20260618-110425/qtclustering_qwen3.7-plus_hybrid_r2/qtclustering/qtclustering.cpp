// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
// - MPI: Distribute seed evaluations across processes
// - OpenMP: Parallelize seed evaluation loop within each process
// - CUDA: Accelerate distance computations in findClosestPoint

#include <mpi.h>
#include <omp.h>
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

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// Per-thread CUDA resources to avoid race conditions
struct ThreadCudaRes {
    double* d_points_x;
    double* d_points_y;
    int* d_cluster_members;
    int* d_clustered;
    int* d_in_cluster;
    double* d_best_diameters;
    int* d_best_candidates;
    int* h_cluster_members;
    int* h_clustered;
    int* h_in_cluster;
    double* h_best_diameters;
    int* h_best_candidates;
    cudaStream_t stream;
    bool initialized;
    ThreadCudaRes() : initialized(false) {}
};

// Global read-only point data on GPU (shared, uploaded once)
static double* g_d_points_x = nullptr;
static double* g_d_points_y = nullptr;
static int g_num_points = 0;

// CUDA kernel for finding closest point to cluster
__global__ void findClosestPointKernel(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int* __restrict__ cluster_members,
    const int* __restrict__ clustered,
    const int* __restrict__ in_cluster,
    int num_candidates,
    int num_members,
    double threshold,
    double* __restrict__ best_diameters,
    int* __restrict__ best_candidates
) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates) return;
    
    if (clustered[candidate] || in_cluster[candidate]) {
        best_diameters[candidate] = 1e308;
        best_candidates[candidate] = -1;
        return;
    }
    
    double cand_x = points_x[candidate];
    double cand_y = points_y[candidate];
    double max_dist = 0.0;
    
    for (int i = 0; i < num_members; ++i) {
        int member = cluster_members[i];
        double dx = cand_x - points_x[member];
        double dy = cand_y - points_y[member];
        double dist = sqrt(dx * dx + dy * dy);
        max_dist = fmax(max_dist, dist);
    }
    
    if (max_dist < threshold) {
        best_diameters[candidate] = max_dist;
        best_candidates[candidate] = candidate;
    } else {
        best_diameters[candidate] = 1e308;
        best_candidates[candidate] = -1;
    }
}

void initThreadCuda(ThreadCudaRes& res, int num_points) {
    if (res.initialized) return;
    
    // Points are shared - use global GPU buffers
    res.d_points_x = g_d_points_x;
    res.d_points_y = g_d_points_y;
    
    CUDA_CHECK(cudaMalloc(&res.d_cluster_members, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&res.d_clustered, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&res.d_in_cluster, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&res.d_best_diameters, num_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&res.d_best_candidates, num_points * sizeof(int)));
    
    CUDA_CHECK(cudaMallocHost(&res.h_cluster_members, num_points * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&res.h_clustered, num_points * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&res.h_in_cluster, num_points * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&res.h_best_diameters, num_points * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&res.h_best_candidates, num_points * sizeof(int)));
    
    CUDA_CHECK(cudaStreamCreate(&res.stream));
    res.initialized = true;
}

void cleanupThreadCuda(ThreadCudaRes& res) {
    if (!res.initialized) return;
    cudaFree(res.d_cluster_members);
    cudaFree(res.d_clustered);
    cudaFree(res.d_in_cluster);
    cudaFree(res.d_best_diameters);
    cudaFree(res.d_best_candidates);
    cudaFreeHost(res.h_cluster_members);
    cudaFreeHost(res.h_clustered);
    cudaFreeHost(res.h_in_cluster);
    cudaFreeHost(res.h_best_diameters);
    cudaFreeHost(res.h_best_candidates);
    cudaStreamDestroy(res.stream);
    res.initialized = false;
}

void initGlobalGpuPoints(const std::vector<Point>& points) {
    int N = static_cast<int>(points.size());
    g_num_points = N;
    
    std::vector<double> hx(N), hy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
    }
    
    CUDA_CHECK(cudaMalloc(&g_d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_d_points_y, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(g_d_points_x, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_d_points_y, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
}

void cleanupGlobalGpuPoints() {
    if (g_d_points_x) { cudaFree(g_d_points_x); g_d_points_x = nullptr; }
    if (g_d_points_y) { cudaFree(g_d_points_y); g_d_points_y = nullptr; }
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
        
        if (group_cnt > (N - count)) group_cnt = N - count;
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            
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

// Find closest point using CUDA with per-thread resources
int findClosestPointCuda(ThreadCudaRes& res,
                         const std::vector<int>& cluster_members,
                         const int* clustered,
                         const int* in_cluster,
                         const double threshold,
                         const int point_count) {
    int num_members = static_cast<int>(cluster_members.size());
    
    // Copy data to pinned host buffers
    memcpy(res.h_cluster_members, cluster_members.data(), num_members * sizeof(int));
    memcpy(res.h_clustered, clustered, point_count * sizeof(int));
    memcpy(res.h_in_cluster, in_cluster, point_count * sizeof(int));
    
    // Async copy to device
    CUDA_CHECK(cudaMemcpyAsync(res.d_cluster_members, res.h_cluster_members,
                               num_members * sizeof(int), cudaMemcpyHostToDevice, res.stream));
    CUDA_CHECK(cudaMemcpyAsync(res.d_clustered, res.h_clustered,
                               point_count * sizeof(int), cudaMemcpyHostToDevice, res.stream));
    CUDA_CHECK(cudaMemcpyAsync(res.d_in_cluster, res.h_in_cluster,
                               point_count * sizeof(int), cudaMemcpyHostToDevice, res.stream));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (point_count + blockSize - 1) / blockSize;
    
    findClosestPointKernel<<<numBlocks, blockSize, 0, res.stream>>>(
        res.d_points_x, res.d_points_y,
        res.d_cluster_members, res.d_clustered, res.d_in_cluster,
        point_count, num_members, threshold,
        res.d_best_diameters, res.d_best_candidates
    );
    
    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(res.h_best_diameters, res.d_best_diameters,
                               point_count * sizeof(double), cudaMemcpyDeviceToHost, res.stream));
    CUDA_CHECK(cudaMemcpyAsync(res.h_best_candidates, res.d_best_candidates,
                               point_count * sizeof(int), cudaMemcpyDeviceToHost, res.stream));
    CUDA_CHECK(cudaStreamSynchronize(res.stream));
    
    // Find best candidate
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    for (int i = 0; i < point_count; ++i) {
        if (res.h_best_candidates[i] >= 0 && res.h_best_diameters[i] < min_diameter) {
            min_diameter = res.h_best_diameters[i];
            closest_point = res.h_best_candidates[i];
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
int generateCandidateCluster(ThreadCudaRes& res,
                              const int seed_point,
                              const int* clustered,
                              const double threshold,
                              const int point_count,
                              std::vector<int>& out_members) {
    std::vector<int> in_cluster(point_count, 0);
    std::vector<int> members;
    members.reserve(64);
    
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);
    
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointCuda(res, members, clustered,
                                                 in_cluster.data(), threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }
    
    out_members = std::move(members);
    return static_cast<int>(out_members.size());
}

// Main QT clustering algorithm with hybrid parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Upload points to GPU
    initGlobalGpuPoints(points);
    
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Per-thread CUDA resources
    int max_threads = omp_get_max_threads();
    std::vector<ThreadCudaRes> thread_res(max_threads);
    
    // Initialize all thread resources
    for (int t = 0; t < max_threads; ++t) {
        initThreadCuda(thread_res[t], N);
    }
    
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            ThreadCudaRes& my_res = thread_res[tid];
            
            int thread_max_cardinality = -1;
            int thread_best_seed = -1;
            std::vector<int> thread_best_cluster_members;
            
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                
                std::vector<int> candidate_members;
                const int cardinality = generateCandidateCluster(
                    my_res, seed, clustered.data(), threshold, N, candidate_members);
                
                if (cardinality > thread_max_cardinality) {
                    thread_max_cardinality = cardinality;
                    thread_best_seed = seed;
                    thread_best_cluster_members = std::move(candidate_members);
                }
            }
            
            #pragma omp critical
            {
                if (thread_max_cardinality > local_max_cardinality) {
                    local_max_cardinality = thread_max_cardinality;
                    local_best_seed = thread_best_seed;
                    local_best_cluster_members = std::move(thread_best_cluster_members);
                }
            }
        }
        
        // MPI reduction
        int global_max_cardinality;
        MPI_Allreduce(&local_max_cardinality, &global_max_cardinality, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if (global_max_cardinality <= 0) break;
        
        int has_best = (local_max_cardinality == global_max_cardinality &&
                        local_best_seed >= 0) ? mpi_rank : -1;
        int best_rank;
        MPI_Allreduce(&has_best, &best_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        int broadcast_data[2] = {global_max_cardinality, local_best_seed};
        MPI_Bcast(broadcast_data, 2, MPI_INT, best_rank, MPI_COMM_WORLD);
        
        int final_cardinality = broadcast_data[0];
        int final_seed = broadcast_data[1];
        
        std::vector<int> best_members(final_cardinality);
        if (mpi_rank == best_rank) {
            best_members = local_best_cluster_members;
        }
        MPI_Bcast(best_members.data(), final_cardinality, MPI_INT, best_rank, MPI_COMM_WORLD);
        
        Cluster cluster;
        cluster.seed_point = final_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(std::move(cluster));
        
        for (int member : clusters.back().members) {
            clustered[member] = 1;
        }
        
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }
    
    // Cleanup thread resources
    for (int t = 0; t < max_threads; ++t) {
        cleanupThreadCuda(thread_res[t]);
    }
    
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
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    cleanupGlobalGpuPoints();
    
    if (mpi_rank == 0) {
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
               total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        const double time_sec = cluster_time.count() / 1000.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters.size() / time_sec, num_points / time_sec);
        
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
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }
    
    MPI_Finalize();
    return 0;
}

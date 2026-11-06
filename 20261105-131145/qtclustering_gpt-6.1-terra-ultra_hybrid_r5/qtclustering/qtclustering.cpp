// QT Clustering Benchmark - MPI, OpenMP and CUDA implementation
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
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
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
        if (N <= 30) group_cnt = 1;
        
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

// Each block grows one seed's cluster.  The maximum distance for each
// candidate is updated only for the newly added member; all previous distances
// are already represented in scratch.  Reductions break ties by point index.
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Match the host compiler's native contraction of dx*dx + dy*dy.  Explicit
// rounded operations keep the non-FMA path independent of CUDA defaults.
__device__ double deviceDistance(Point candidate, Point member, bool fused) {
    const double dx = candidate.x - member.x;
    const double dy = candidate.y - member.y;
    const double yy = __dmul_rn(dy, dy);
    const double sum = fused ? __fma_rn(dx, dx, yy) : __dadd_rn(__dmul_rn(dx, dx), yy);
    return sqrt(sum);
}

__global__ void distanceKernel(const Point* points, int n, double* distances, bool fused) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    for (; index < total; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const Point candidate = points[index % n];
        const Point member = points[index / n];
        distances[index] = deviceDistance(candidate, member, fused);
    }
}

__global__ void candidateKernel(const Point* points, const double* distances, const unsigned char* clustered,
                                int n, double threshold, bool fused, const int* seeds,
                                double* scratch, int* counts, int* members) {
    const int slot = blockIdx.x;
    const int tid = threadIdx.x;
    double* maxima = scratch + static_cast<size_t>(slot) * n;
    __shared__ double best_dist[BLOCK_SIZE];
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ int last, size;
    if (tid == 0) {
        last = seeds[slot];
        size = 1;
        if (members) members[0] = last;
    }
    __syncthreads();

    for (int step = 0; step < n; ++step) {
        const int added = last;
        const Point p = points[added];
        double local_best = DBL_MAX;
        int local_index = INT_MAX;
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            double max_dist = step == 0 ? 0.0 : maxima[i];
            if (clustered[i] || i == added || max_dist == DBL_MAX) {
                maxima[i] = DBL_MAX;
                continue;
            }
            const double d = distances ? distances[static_cast<size_t>(added) * n + i]
                                       : deviceDistance(points[i], p, fused);
            // std::max(a, b) returns a if b is NaN.
            if (max_dist < d) max_dist = d;
            maxima[i] = max_dist >= threshold ? DBL_MAX : max_dist;
            if (max_dist < threshold &&
                (max_dist < local_best || (max_dist == local_best && i < local_index))) {
                local_best = max_dist;
                local_index = i;
            }
        }
        best_dist[tid] = local_best;
        best_index[tid] = local_index;
        __syncthreads();
        for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                double other = best_dist[tid + offset];
                int index = best_index[tid + offset];
                if (other < best_dist[tid] ||
                    (other == best_dist[tid] && index < best_index[tid])) {
                    best_dist[tid] = other;
                    best_index[tid] = index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            last = best_index[0];
            if (last != INT_MAX) {
                if (members) members[size] = last;
                ++size;
            }
        }
        __syncthreads();
        if (last == INT_MAX || size == n) break;
    }
    if (tid == 0) counts[slot] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    int fused = 1;
#if defined(__x86_64__) || defined(__i386__)
    if (rank == 0) fused = __builtin_cpu_supports("fma") != 0;
#endif
    MPI_Bcast(&fused, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    Point* d_points;
    unsigned char* d_clustered;
    int *d_seeds, *d_counts, *d_members;
    double *d_scratch, *d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    // Bound per-rank scratch space so large inputs do not require N squared
    // device memory.  The same buffers are reused for all seed batches.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t budget = std::min<size_t>(256ull * 1024 * 1024, free_bytes / 4);
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    if (matrix_bytes <= std::min<size_t>(128ull * 1024 * 1024, free_bytes / 4)) {
        CUDA_CHECK(cudaMalloc(&d_distances, matrix_bytes));
        const int blocks = static_cast<int>(std::min<size_t>(65535, (static_cast<size_t>(n) * n + 255) / 256));
        distanceKernel<<<blocks, 256>>>(d_points, n, d_distances, fused != 0);
        CUDA_CHECK(cudaGetLastError());
    }
    size_t batch_size = std::max<size_t>(1, budget / (static_cast<size_t>(n) * sizeof(double)));
    batch_size = std::min<size_t>(batch_size, (static_cast<size_t>(n) + ranks - 1) / ranks);
    CUDA_CHECK(cudaMalloc(&d_scratch, batch_size * static_cast<size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_seeds, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, batch_size * sizeof(int)));
    const size_t owned_count = rank < n ? (static_cast<size_t>(n) - 1 - rank) / ranks + 1 : 0;
    std::vector<int> owned_seeds(owned_count);
    const int seed_threads = std::min(omp_get_max_threads(),
                                      std::max(1, static_cast<int>(owned_count / 2048)));
    #pragma omp parallel for schedule(static) num_threads(seed_threads)
    for (size_t i = 0; i < owned_count; ++i)
        owned_seeds[i] = static_cast<int>(rank + i * ranks);
    std::vector<int> seeds, counts(batch_size), selected(n);
    seeds.reserve(owned_count);
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(n), cudaMemcpyHostToDevice));
        seeds.clear();
        for (int seed : owned_seeds) {
            if (!clustered[seed]) seeds.push_back(seed);
        }
        struct { int count; int seed; } local = {-1, INT_MAX}, global;
        for (size_t start = 0; start < seeds.size(); start += batch_size) {
            const int batch = static_cast<int>(std::min(batch_size, seeds.size() - start));
            CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data() + start, static_cast<size_t>(batch) * sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<batch, BLOCK_SIZE>>>(d_points, d_distances, d_clustered, n, threshold, fused != 0,
                                                    d_seeds, d_scratch, d_counts, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, static_cast<size_t>(batch) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batch; ++i) {
                int seed = seeds[start + i];
                if (counts[i] > local.count || (counts[i] == local.count && seed < local.seed))
                    local = {counts[i], seed};
            }
        }
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Once the largest candidate has one point, removing more points
        // cannot make any larger cluster.  Emit the remaining seeds in order.
        if (global.count == 1) {
            for (int i = 0; i < n; ++i) {
                if (!clustered[i]) clusters.push_back(Cluster{{i}, i});
            }
            break;
        }
        const int owner = global.seed % ranks;
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(d_seeds, &global.seed, sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<1, BLOCK_SIZE>>>(d_points, d_distances, d_clustered, n, threshold, fused != 0,
                                               d_seeds, d_scratch, d_counts, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(selected.data(), d_members, static_cast<size_t>(global.count) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(selected.data(), global.count, MPI_INT, owner, MPI_COMM_WORLD);
        Cluster cluster;
        cluster.seed_point = global.seed;
        cluster.members.assign(selected.begin(), selected.begin() + global.count);
        clusters.push_back(std::move(cluster));
        #pragma omp parallel for schedule(static) if(global.count >= 4096)
        for (int i = 0; i < global.count; ++i) clustered[selected[i]] = 1;
        remaining -= global.count;
    }
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    std::vector<double> diameters(clusters.size());
    #pragma omp parallel for schedule(dynamic) if(clusters.size() >= 128)
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
        
        diameters[c] = max_diameter;
    }
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    size_t bytes = static_cast<size_t>(num_points) * sizeof(Point);
    char* point_bytes = reinterpret_cast<char*>(points.data());
    while (bytes > 0) {
        const int chunk = static_cast<int>(std::min<size_t>(bytes, INT_MAX));
        MPI_Bcast(point_bytes, chunk, MPI_BYTE, 0, MPI_COMM_WORLD);
        point_bytes += chunk;
        bytes -= chunk;
    }
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    int exit_code = 0;
    if (rank == 0) {
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
        
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exit_code = valid ? 0 : 1;
    }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

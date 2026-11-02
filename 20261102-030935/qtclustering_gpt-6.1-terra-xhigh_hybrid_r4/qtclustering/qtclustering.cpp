// QT Clustering Benchmark - MPI, OpenMP, and CUDA version
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
#include <math_constants.h>

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

// Each block constructs one candidate.  Its row of maxima is updated only
// with the most recently added member, preserving the original greedy order.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

constexpr int BLOCK_SIZE = 256;

__global__ void computeDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int a = static_cast<int>(i / n), b = static_cast<int>(i % n);
        double dx = points[a].x - points[b].x;
        double dy = points[a].y - points[b].y;
        distances[i] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void candidateClusters(const Point* points, const double* distances,
                                  const unsigned char* clustered, const int* seeds,
                                  int n, double threshold, double* maxima,
                                  int* sizes, int* members) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[row];
    double* max_dist = maxima + static_cast<size_t>(row) * n;
    __shared__ double best_dist[BLOCK_SIZE];
    __shared__ int best_index[BLOCK_SIZE];
    __shared__ int last, count;
    if (tid == 0) {
        last = seed;
        count = 1;
        if (members) members[0] = seed;
    }
    for (int i = tid; i < n; i += BLOCK_SIZE)
        max_dist[i] = (clustered[i] || i == seed) ? CUDART_INF : 0.0;
    __syncthreads();

    while (true) {
        double best = DBL_MAX;
        int index = INT_MAX;
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            double previous = max_dist[i];
            if (previous >= threshold || previous == CUDART_INF) continue;
            double dist;
            if (distances) {
                dist = distances[static_cast<size_t>(last) * n + i];
            } else {
                double dx = points[i].x - points[last].x;
                double dy = points[i].y - points[last].y;
                dist = sqrt(dx * dx + dy * dy);
            }
            double current = previous > dist ? previous : dist;
            max_dist[i] = current;
            if (current < threshold && (current < best || (current == best && i < index))) {
                best = current;
                index = i;
            }
        }
        best_dist[tid] = best;
        best_index[tid] = index;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
            if (tid < stride) {
                double other = best_dist[tid + stride];
                int other_index = best_index[tid + stride];
                if (other < best_dist[tid] ||
                    (other == best_dist[tid] && other_index < best_index[tid])) {
                    best_dist[tid] = other;
                    best_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        if (best_index[0] == INT_MAX) break;
        if (tid == 0) {
            last = best_index[0];
            max_dist[last] = CUDART_INF;
            if (members) members[count] = last;
            ++count;
        }
        __syncthreads();
    }
    if (tid == 0) sizes[row] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    Point* d_points = nullptr;
    double *d_distances = nullptr, *d_maxima = nullptr;
    unsigned char* d_clustered = nullptr;
    int *d_seeds = nullptr, *d_sizes = nullptr, *d_members = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * static_cast<size_t>(n), cudaMemcpyHostToDevice));

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    // Keep memory for working rows; calculate distances on demand when a full
    // distance matrix would consume too much device memory.
    if (matrix_bytes <= free_bytes / 3) {
        CUDA_CHECK(cudaMalloc(&d_distances, matrix_bytes));
        const size_t blocks = (static_cast<size_t>(n) * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        computeDistances<<<static_cast<unsigned int>(std::min<size_t>(blocks, 65535)), BLOCK_SIZE>>>(d_points, d_distances, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t row_bytes = sizeof(double) * static_cast<size_t>(n);
    const size_t local_capacity = (static_cast<size_t>(n) + ranks - 1) / ranks;
    const size_t budget = std::min<size_t>(free_bytes / 2, size_t(512) * 1024 * 1024);
    if (budget < row_bytes) {
        fprintf(stderr, "Not enough GPU memory for a candidate cluster\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int batch_capacity = static_cast<int>(std::max<size_t>(1, std::min(local_capacity, budget / row_bytes)));
    CUDA_CHECK(cudaMalloc(&d_maxima, row_bytes * batch_capacity));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * static_cast<size_t>(batch_capacity)));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(int) * static_cast<size_t>(batch_capacity)));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> local_seeds, sizes(batch_capacity), winning_members(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        local_seeds.clear();
        for (int seed = rank; seed < n; seed += ranks)
            if (!clustered[seed]) local_seeds.push_back(seed);
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(n), cudaMemcpyHostToDevice));
        struct { int size; int seed; } local_best{0, INT_MAX}, global_best;
        for (size_t offset = 0; offset < local_seeds.size(); offset += batch_capacity) {
            int count = static_cast<int>(std::min<size_t>(batch_capacity, local_seeds.size() - offset));
            CUDA_CHECK(cudaMemcpy(d_seeds, local_seeds.data() + offset, sizeof(int) * static_cast<size_t>(count), cudaMemcpyHostToDevice));
            candidateClusters<<<count, BLOCK_SIZE>>>(d_points, d_distances, d_clustered, d_seeds, n, threshold, d_maxima, d_sizes, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, sizeof(int) * static_cast<size_t>(count), cudaMemcpyDeviceToHost));
            // Reduction is deterministic even when candidate sizes tie.
            const int threads = std::min(omp_get_max_threads(), std::max(1, count / 256));
            #pragma omp parallel num_threads(threads)
            {
                int thread_size = 0, thread_seed = INT_MAX;
                #pragma omp for nowait
                for (int i = 0; i < count; ++i) {
                    int seed = local_seeds[offset + i];
                    if (sizes[i] > thread_size || (sizes[i] == thread_size && seed < thread_seed)) {
                        thread_size = sizes[i];
                        thread_seed = seed;
                    }
                }
                #pragma omp critical
                if (thread_size > local_best.size || (thread_size == local_best.size && thread_seed < local_best.seed)) {
                    local_best.size = thread_size;
                    local_best.seed = thread_seed;
                }
            }
        }
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best.size <= 0) break;
        const int owner = global_best.seed % ranks;
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(d_seeds, &global_best.seed, sizeof(int), cudaMemcpyHostToDevice));
            candidateClusters<<<1, BLOCK_SIZE>>>(d_points, d_distances, d_clustered, d_seeds, n, threshold, d_maxima, d_sizes, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(winning_members.data(), d_members, sizeof(int) * static_cast<size_t>(global_best.size), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(winning_members.data(), global_best.size, MPI_INT, owner, MPI_COMM_WORLD);
        Cluster cluster;
        cluster.seed_point = global_best.seed;
        cluster.members.assign(winning_members.begin(), winning_members.begin() + global_best.size);
        for (int member : cluster.members) clustered[member] = 1;
        remaining -= global_best.size;
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaFree(d_points));
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_maxima));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_sizes));
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct MPIFinalizer { ~MPIFinalizer() { MPI_Finalize(); } } mpi_finalizer;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                              num_points, threshold);
        return 1;
    }
    
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA device available for rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two doubles");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    const long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    cluster_time = std::chrono::milliseconds(max_cluster_time);

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

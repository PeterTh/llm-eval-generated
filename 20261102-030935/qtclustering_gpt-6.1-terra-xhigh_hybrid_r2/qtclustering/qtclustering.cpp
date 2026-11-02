// QT Clustering Benchmark - MPI, OpenMP, and CUDA implementation
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
#include <climits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <omp.h>
#include <vector>

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

// All CUDA failures are fatal across the MPI job, including failures on ranks
// that are not responsible for printing the results.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Distances are reused by every seed and every clustering round. Compute the
// same double-precision distance as the sequential host implementation.
__device__ double pointDistance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
#ifdef HOST_HAS_FMA
    return sqrt(fma(dx, dx, dy * dy));
#else
    return sqrt(dx * dx + dy * dy);
#endif
}

__global__ void calculateDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t pos = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pos < total; pos += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int i = static_cast<int>(pos / n);
        int j = static_cast<int>(pos % n);
        distances[pos] = pointDistance(points[i], points[j]);
    }
}

// One block builds one candidate cluster. The running maximum distance for
// each point replaces repeated scans of all previously selected members.
// Selection is ordered by (diameter, point index), exactly as in the original.
__global__ void evaluateSeeds(const Point* points, const double* distances, const unsigned char* clustered,
                              const int* seeds, int count, int n, double threshold,
                              double* maxima, int* sizes, int* output_members) {
    const int block = blockIdx.x;
    if (block >= count) return;
    const int tid = threadIdx.x;
    const int seed = seeds[block];
    double* row = maxima + static_cast<size_t>(block) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int last, cardinality;
    if (tid == 0) {
        last = seed;
        cardinality = 1;
        if (output_members) output_members[0] = seed;
    }
    for (int i = tid; i < n; i += blockDim.x)
        row[i] = (clustered[i] || i == seed) ? CUDART_INF : 0.0;
    __syncthreads();

    while (true) {
        double local_best = CUDART_INF;
        int local_index = INT_MAX;
        const int member = last;
        for (int i = tid; i < n; i += blockDim.x) {
            double value = row[i];
            if (value < threshold) {
                const double d = distances ? distances[static_cast<size_t>(i) * n + member]
                                           : pointDistance(points[i], points[member]);
                if (d > value) value = d;
                row[i] = value;
                if (value < threshold &&
                    (value < local_best || (value == local_best && i < local_index))) {
                    local_best = value;
                    local_index = i;
                }
            }
        }
        best_distance[tid] = local_best;
        best_index[tid] = local_index;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset /= 2) {
            if (tid < offset &&
                (best_distance[tid + offset] < best_distance[tid] ||
                 (best_distance[tid + offset] == best_distance[tid] &&
                  best_index[tid + offset] < best_index[tid]))) {
                best_distance[tid] = best_distance[tid + offset];
                best_index[tid] = best_index[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            last = best_index[0];
            if (last != INT_MAX) {
                row[last] = CUDART_INF;
                if (output_members) output_members[cardinality] = last;
                ++cardinality;
            }
        }
        __syncthreads();
        if (last == INT_MAX) break;
    }
    if (tid == 0) sizes[block] = cardinality;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_size = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    const size_t elements = static_cast<size_t>(n) * n;
    const bool matrix_size_valid = elements <= std::numeric_limits<size_t>::max() / sizeof(double);
    const size_t matrix_bytes = matrix_size_valid ? elements * sizeof(double) : 0;
    const int max_local_seeds = n / ranks + (n % ranks != 0);
    Point* device_points = nullptr;
    double *device_distances = nullptr, *device_maxima = nullptr;
    unsigned char* device_clustered = nullptr;
    int *device_seeds = nullptr, *device_sizes = nullptr, *device_members = nullptr;
    CUDA_CHECK(cudaMalloc(&device_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&device_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&device_seeds, static_cast<size_t>(max_local_seeds) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_sizes, static_cast<size_t>(max_local_seeds) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_members, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));
    // Reserve room for other CUDA allocations. If the full distance table does
    // not fit comfortably, calculate distances on demand and process seeds in
    // batches, so GPU memory does not impose a quadratic input-size limit.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const int device_index = local_rank % device_count;
    const int sharing_ranks = local_size / device_count + (device_index < local_size % device_count);
    size_t budget = (free_bytes / static_cast<size_t>(sharing_ranks)) / 4 * 3;
    if (matrix_size_valid && matrix_bytes <= budget / 2) {
        CUDA_CHECK(cudaMalloc(&device_distances, matrix_bytes));
        budget -= matrix_bytes;
        const size_t blocks_needed = (static_cast<size_t>(n) * n + 255) / 256;
        calculateDistances<<<static_cast<int>(std::min<size_t>(blocks_needed, 65535)), 256>>>(device_points, device_distances, n);
        CUDA_CHECK(cudaGetLastError());
    }
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    const int batch_size = static_cast<int>(std::min<size_t>(max_local_seeds, budget / row_bytes));
    if (batch_size < 1) {
        fprintf(stderr, "Rank %d: insufficient GPU memory for one candidate\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaMalloc(&device_maxima, static_cast<size_t>(batch_size) * row_bytes));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> remaining(n), sizes(max_local_seeds), members(n);
    for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;
    while (!remaining.empty()) {
        const int total = static_cast<int>(remaining.size());
        const int begin = static_cast<int>(static_cast<long long>(total) * rank / ranks);
        const int end = static_cast<int>(static_cast<long long>(total) * (rank + 1) / ranks);
        const int count = end - begin;
        CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        if (count) {
            CUDA_CHECK(cudaMemcpy(device_seeds, remaining.data() + begin, static_cast<size_t>(count) * sizeof(int), cudaMemcpyHostToDevice));
            for (int offset = 0; offset < count; offset += batch_size) {
                const int batch = std::min(batch_size, count - offset);
                evaluateSeeds<<<batch, 256>>>(device_points, device_distances, device_clustered,
                                              device_seeds + offset, batch, n, threshold,
                                              device_maxima, device_sizes + offset, nullptr);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaMemcpy(sizes.data(), device_sizes, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        int max_size = -1;
        #pragma omp parallel for reduction(max:max_size) if(count >= 256)
        for (int i = 0; i < count; ++i) max_size = std::max(max_size, sizes[i]);
        int best_seed = INT_MAX;
        #pragma omp parallel for reduction(min:best_seed) if(count >= 256)
        for (int i = 0; i < count; ++i)
            if (sizes[i] == max_size) best_seed = std::min(best_seed, remaining[begin + i]);
        struct { int size, seed; } local_best{max_size, best_seed}, global_best;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        const int position = static_cast<int>(std::lower_bound(remaining.begin(), remaining.end(), global_best.seed) - remaining.begin());
        // Invert floor(total * rank / ranks), including ranks with zero seeds.
        const int owner = static_cast<int>(((static_cast<long long>(position) + 1) * ranks - 1) / total);
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(device_seeds, &global_best.seed, sizeof(int), cudaMemcpyHostToDevice));
            evaluateSeeds<<<1, 256>>>(device_points, device_distances, device_clustered, device_seeds, 1, n, threshold,
                                      device_maxima, device_sizes, device_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(members.data(), device_members, static_cast<size_t>(global_best.size) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), global_best.size, MPI_INT, owner, MPI_COMM_WORLD);
        Cluster cluster;
        cluster.seed_point = global_best.seed;
        cluster.members.assign(members.begin(), members.begin() + global_best.size);
        for (int member : cluster.members) clustered[member] = 1;
        clusters.push_back(std::move(cluster));
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                        [&](int i) { return clustered[i] != 0; }), remaining.end());
    }
    CUDA_CHECK(cudaFree(device_points));
    CUDA_CHECK(cudaFree(device_distances));
    CUDA_CHECK(cudaFree(device_maxima));
    CUDA_CHECK(cudaFree(device_clustered));
    CUDA_CHECK(cudaFree(device_seeds));
    CUDA_CHECK(cudaFree(device_sizes));
    CUDA_CHECK(cudaFree(device_members));
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain only two doubles");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}

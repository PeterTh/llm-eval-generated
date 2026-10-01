// QT Clustering Benchmark - Simplified Sequential Version
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
#include <cstdint>
#include <climits>
#include <cfloat>
#include <vector>
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
        // The original expression is always zero for N <= 30.
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

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One block grows one seed's cluster.  The per-candidate diameter is updated
// incrementally when a point joins, so each distance is computed only once.
__global__ void growSeeds(const Point* points, const unsigned char* clustered,
                          double* diameters, int* cardinalities, int* members,
                          int N, int first_seed, int seed_count, double threshold) {
    const int slot = blockIdx.x;
    if (slot >= seed_count) return;
    const int seed = first_seed + slot;
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[slot] = 0;
        return;
    }

    double* maxima = diameters + static_cast<size_t>(slot) * N;
    __shared__ double best_dist[256];
    __shared__ int best_idx[256];
    __shared__ int last_member;
    __shared__ int count;
    if (threadIdx.x == 0) {
        last_member = seed;
        count = 1;
        if (members) members[0] = seed;
    }
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        maxima[i] = (clustered[i] || i == seed) ? -1.0 : 0.0;
    __syncthreads();

    while (true) {
        const Point last = points[last_member];
        double local_dist = DBL_MAX;
        int local_idx = INT_MAX;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            double value = maxima[i];
            if (value < 0.0) continue;
            const double dx = points[i].x - last.x;
            const double dy = points[i].y - last.y;
            const double dist = sqrt(dx * dx + dy * dy);
            value = fmax(value, dist);
            // The maximum can only increase as the cluster grows.
            maxima[i] = value < threshold ? value : -1.0;
            if (value < threshold &&
                (value < local_dist || (value == local_dist && i < local_idx))) {
                local_dist = value;
                local_idx = i;
            }
        }
        best_dist[threadIdx.x] = local_dist;
        best_idx[threadIdx.x] = local_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                const int other = threadIdx.x + stride;
                if (best_dist[other] < best_dist[threadIdx.x] ||
                    (best_dist[other] == best_dist[threadIdx.x] && best_idx[other] < best_idx[threadIdx.x])) {
                    best_dist[threadIdx.x] = best_dist[other];
                    best_idx[threadIdx.x] = best_idx[other];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int next = best_idx[0];
            if (next != INT_MAX) {
                maxima[next] = -1.0;
                last_member = next;
                if (members) members[count] = next;
                ++count;
            }
        }
        __syncthreads();
        if (best_idx[0] == INT_MAX) break;
    }
    if (threadIdx.x == 0) cardinalities[slot] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int rank, int ranks) {
    const int N = static_cast<int>(points.size());
    const int first = static_cast<int>((static_cast<int64_t>(N) * rank) / ranks);
    const int end = static_cast<int>((static_cast<int64_t>(N) * (rank + 1)) / ranks);
    const int local_count = end - first;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    double* d_diameters = nullptr;
    int* d_counts = nullptr;
    int* d_members = nullptr;
    size_t free_bytes = 0, total_bytes = 0;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t bytes_per_seed = static_cast<size_t>(N) * sizeof(double);
    const size_t batch_limit = std::max<size_t>(1, free_bytes / 4 / bytes_per_seed);
    const int batch_size = static_cast<int>(std::min<size_t>(std::max(1, local_count),
                                      std::min<size_t>(4096, batch_limit)));
    cudaCheck(cudaMalloc(&d_points, static_cast<size_t>(N) * sizeof(Point)));
    cudaCheck(cudaMalloc(&d_clustered, N));
    cudaCheck(cudaMalloc(&d_diameters, static_cast<size_t>(batch_size) * bytes_per_seed));
    cudaCheck(cudaMalloc(&d_counts, static_cast<size_t>(batch_size) * sizeof(int)));
    cudaCheck(cudaMalloc(&d_members, static_cast<size_t>(N) * sizeof(int)));
    cudaCheck(cudaMemcpy(d_points, points.data(), static_cast<size_t>(N) * sizeof(Point), cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> counts(local_count);
    std::vector<Cluster> clusters;
    int remaining = N;
    while (remaining > 0) {
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        for (int offset = 0; offset < local_count; offset += batch_size) {
            const int batch = std::min(batch_size, local_count - offset);
            growSeeds<<<batch, 256>>>(d_points, d_clustered, d_diameters,
                                      d_counts, nullptr, N, first + offset, batch, threshold);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(counts.data() + offset, d_counts,
                                 static_cast<size_t>(batch) * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // Higher cardinality wins; for equal cardinality the smallest seed wins.
        unsigned long long local_best = 0;
        #pragma omp parallel for reduction(max:local_best) schedule(static)
        for (int i = 0; i < local_count; ++i) {
            if (counts[i] > 0) {
                const unsigned long long key = (static_cast<unsigned long long>(counts[i]) << 32)
                    | static_cast<unsigned int>(UINT_MAX - (first + i));
                if (key > local_best) local_best = key;
            }
        }
        unsigned long long global_best = 0;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (!global_best) break;
        const int size = static_cast<int>(global_best >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(global_best));
        const int owner = static_cast<int>((static_cast<int64_t>(seed + 1) * ranks - 1) / N);
        std::vector<int> members(size);
        if (rank == owner) {
            growSeeds<<<1, 256>>>(d_points, d_clustered, d_diameters, d_counts,
                                  d_members, N, seed, 1, threshold);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(members.data(), d_members, static_cast<size_t>(size) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({members, seed});
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < size; ++i) clustered[members[i]] = 1;
        remaining -= size;
    }
    cudaCheck(cudaFree(d_members));
    cudaCheck(cudaFree(d_counts));
    cudaCheck(cudaFree(d_diameters));
    cudaCheck(cudaFree(d_clustered));
    cudaCheck(cudaFree(d_points));
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
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) fprintf(stderr, "CUDA GPU required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count));

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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (num_points > INT_MAX / static_cast<int>(sizeof(Point))) {
        if (rank == 0) fprintf(stderr, "Too many points for MPI broadcast\n");
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
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    
    printf("Clustering time: %ld ms\n", static_cast<long>(time_sec * 1000.0));
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

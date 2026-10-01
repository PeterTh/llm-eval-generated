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
#include <climits>
#include <cfloat>
#include <cstdint>
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

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One block grows one seed's candidate cluster. Each thread owns a subset of
// potential members and retains their running diameter in global memory.
__global__ void scoreSeeds(const Point* points, const unsigned char* clustered,
                           const int* seeds, int count, int n, double threshold,
                           double* diameters, int* sizes) {
    const int block = blockIdx.x;
    if (block >= count) return;
    const int tid = threadIdx.x;
    const int seed = seeds[block];
    double* state = diameters + static_cast<size_t>(block) * n;
    __shared__ double values[256];
    __shared__ int indices[256];
    __shared__ int next;
    int previous = seed;
    int cardinality = 1;
    for (int i = tid; i < n; i += blockDim.x) state[i] = 0.0;
    __syncthreads();
    while (true) {
        double best_value = DBL_MAX;
        int best_index = INT_MAX;
        for (int i = tid; i < n; i += blockDim.x) {
            if (i == previous || clustered[i]) {
                state[i] = -1.0;
                continue;
            }
            if (state[i] < 0.0) continue;
            const double dx = points[i].x - points[previous].x;
            const double dy = points[i].y - points[previous].y;
            const double d = sqrt(dx * dx + dy * dy);
            const double current = fmax(state[i], d);
            state[i] = current;
            if (current < threshold &&
                (current < best_value || (current == best_value && i < best_index))) {
                best_value = current;
                best_index = i;
            }
        }
        values[tid] = best_value;
        indices[tid] = best_index;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset /= 2) {
            if (tid < offset &&
                (values[tid + offset] < values[tid] ||
                 (values[tid + offset] == values[tid] && indices[tid + offset] < indices[tid]))) {
                values[tid] = values[tid + offset];
                indices[tid] = indices[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) next = indices[0];
        __syncthreads();
        if (next == INT_MAX) break;
        previous = next;
        ++cardinality;
    }
    if (tid == 0) sizes[block] = cardinality;
}

// CPU scoring uses the same incremental maximum distance as the GPU kernel.
static int scoreSeedCPU(int seed, const std::vector<unsigned char>& clustered,
                        const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<double> diameters(n, 0.0);
    int previous = seed;
    int cardinality = 1;
    while (true) {
        double best_value = std::numeric_limits<double>::max();
        int best_index = -1;
        for (int i = 0; i < n; ++i) {
            if (i == previous || clustered[i]) {
                diameters[i] = -1.0;
                continue;
            }
            if (diameters[i] < 0.0) continue;
            const double dx = points[i].x - points[previous].x;
            const double dy = points[i].y - points[previous].y;
            const double d = std::sqrt(dx * dx + dy * dy);
            diameters[i] = std::max(diameters[i], d);
            if (diameters[i] < threshold && diameters[i] < best_value) {
                best_value = diameters[i];
                best_index = i;
            }
        }
        if (best_index < 0) break;
        previous = best_index;
        ++cardinality;
    }
    return cardinality;
}

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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_sizes = nullptr;
    double* d_diameters = nullptr;
    constexpr int batch_capacity = 64;
    cudaCheck(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    cudaCheck(cudaMalloc(&d_clustered, n));
    cudaCheck(cudaMalloc(&d_seeds, static_cast<size_t>(n) * sizeof(int)));
    cudaCheck(cudaMalloc(&d_sizes, static_cast<size_t>(n) * sizeof(int)));
    cudaCheck(cudaMalloc(&d_diameters, static_cast<size_t>(batch_capacity) * n * sizeof(double)));
    cudaCheck(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    while (remaining > 0) {
        std::vector<int> gpu_seeds;
        std::vector<int> cpu_seeds;
        for (int seed = rank; seed < n; seed += ranks) {
            if (clustered[seed]) continue;
            if ((seed / ranks) % 8 == 0) cpu_seeds.push_back(seed);
            else gpu_seeds.push_back(seed);
        }
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        std::vector<int> gpu_sizes(gpu_seeds.size());
        // Queue GPU batches while OpenMP scores the CPU share.
        if (!gpu_seeds.empty())
            cudaCheck(cudaMemcpy(d_seeds, gpu_seeds.data(), gpu_seeds.size() * sizeof(int), cudaMemcpyHostToDevice));
        for (size_t base = 0; base < gpu_seeds.size(); base += batch_capacity) {
            const int count = static_cast<int>(std::min<size_t>(batch_capacity, gpu_seeds.size() - base));
            scoreSeeds<<<count, 256>>>(d_points, d_clustered, d_seeds + base, count, n,
                                      threshold, d_diameters, d_sizes + base);
            cudaCheck(cudaGetLastError());
        }

        int local[2] = {-1, INT_MAX};
        #pragma omp parallel
        {
            int thread_best[2] = {-1, INT_MAX};
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < cpu_seeds.size(); ++i) {
                const int size = scoreSeedCPU(cpu_seeds[i], clustered, points, threshold);
                if (size > thread_best[0] || (size == thread_best[0] && cpu_seeds[i] < thread_best[1])) {
                    thread_best[0] = size;
                    thread_best[1] = cpu_seeds[i];
                }
            }
            #pragma omp critical
            if (thread_best[0] > local[0] || (thread_best[0] == local[0] && thread_best[1] < local[1])) {
                local[0] = thread_best[0];
                local[1] = thread_best[1];
            }
        }
        if (!gpu_seeds.empty())
            cudaCheck(cudaMemcpy(gpu_sizes.data(), d_sizes, gpu_sizes.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < gpu_seeds.size(); ++i) {
            if (gpu_sizes[i] > local[0] || (gpu_sizes[i] == local[0] && gpu_seeds[i] < local[1])) {
                local[0] = gpu_sizes[i];
                local[1] = gpu_seeds[i];
            }
        }
        int winner[2];
        MPI_Allreduce(local, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (winner[0] < 1) break;

        std::vector<int> members;
        int member_count = 0;
        if (rank == 0) {
            std::vector<bool> original_clustered(clustered.begin(), clustered.end());
            member_count = generateCandidateCluster(winner[1], original_clustered,
                                                    points, threshold, n, &members);
            clusters.push_back({members, winner[1]});
        }
        MPI_Bcast(&member_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        members.resize(member_count);
        MPI_Bcast(members.data(), member_count, MPI_INT, 0, MPI_COMM_WORLD);
        for (int member : members) clustered[member] = 1;
        remaining -= member_count;
    }
    cudaCheck(cudaFree(d_diameters));
    cudaCheck(cudaFree(d_sizes));
    cudaCheck(cudaFree(d_seeds));
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
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        if (rank == 0) std::fprintf(stderr, "QT clustering requires a CUDA device\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);
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
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);
    
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

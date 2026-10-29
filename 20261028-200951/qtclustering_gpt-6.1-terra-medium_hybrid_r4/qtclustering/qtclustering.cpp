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
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <climits>
#include <omp.h>

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

// Each GPU thread evaluates one seed.  The per-candidate maximum is updated
// incrementally as members are added; this is the same maximum used by the
// original repeated scan of all members. Candidate-major storage coalesces
// accesses by adjacent seed threads.
__global__ void evaluateSeeds(const Point* points, const unsigned char* clustered,
                              const int* seeds, int count, int n, double threshold,
                              double* maxima, int* sizes) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    int seed = seeds[lane];
    for (int i = 0; i < n; ++i) maxima[(size_t)i * count + lane] = 0.0;
    maxima[(size_t)seed * count + lane] = -1.0;
    int last = seed, size = 1;
    while (size < n) {
        int best = -1;
        double best_distance = 1.7976931348623157e+308;
        Point p = points[last];
        for (int i = 0; i < n; ++i) {
            if (clustered[i]) continue;
            double &value = maxima[(size_t)i * count + lane];
            if (value < 0.0) continue;
            double dx = points[i].x - p.x, dy = points[i].y - p.y;
            double d = sqrt(dx * dx + dy * dy);
            if (d > value) value = d;
            if (value < threshold && value < best_distance) {
                best_distance = value;
                best = i;
            }
        }
        if (best < 0) break;
        maxima[(size_t)best * count + lane] = -1.0;
        last = best;
        ++size;
    }
    sizes[lane] = size;
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks, local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(local_rank % devices));
    const int n = static_cast<int>(points.size());
    Point* device_points;
    unsigned char* device_clustered;
    int *device_seeds, *device_sizes;
    double* device_maxima;
    size_t free_memory, total_memory;
    cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // Bound temporary memory, while retaining enough seeds to occupy the GPU.
    size_t budget = std::min(free_memory / 2, size_t(512) * 1024 * 1024);
    size_t capacity = std::min(size_t(n), budget / (size_t(n) * sizeof(double) + 2 * sizeof(int)));
    if (!capacity) { fprintf(stderr, "Insufficient GPU memory\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaMalloc(&device_points, size_t(n) * sizeof(Point)));
    cudaCheck(cudaMalloc(&device_clustered, size_t(n)));
    cudaCheck(cudaMalloc(&device_seeds, capacity * sizeof(int)));
    cudaCheck(cudaMalloc(&device_sizes, capacity * sizeof(int)));
    cudaCheck(cudaMalloc(&device_maxima, size_t(n) * capacity * sizeof(double)));
    cudaCheck(cudaMemcpy(device_points, points.data(), size_t(n) * sizeof(Point), cudaMemcpyHostToDevice));
    std::vector<bool> clustered(n, false);
    std::vector<unsigned char> flags(n, 0);
    std::vector<Cluster> clusters;
    std::vector<int> seeds, sizes(capacity);
    int remaining = n;
    while (remaining) {
        seeds.clear();
        // Partition by index so MPI_MAXLOC also selects the original first seed.
        for (int i = rank; i < n; i += ranks) if (!flags[i]) seeds.push_back(i);
        cudaCheck(cudaMemcpy(device_clustered, flags.data(), size_t(n), cudaMemcpyHostToDevice));
        int local_best = 0, local_seed = INT_MAX;
        for (size_t offset = 0; offset < seeds.size(); offset += capacity) {
            int count = static_cast<int>(std::min(capacity, seeds.size() - offset));
            cudaCheck(cudaMemcpy(device_seeds, seeds.data() + offset, size_t(count) * sizeof(int), cudaMemcpyHostToDevice));
            evaluateSeeds<<<(count + 127) / 128, 128>>>(device_points, device_clustered, device_seeds, count, n, threshold, device_maxima, device_sizes);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(sizes.data(), device_sizes, size_t(count) * sizeof(int), cudaMemcpyDeviceToHost));
            // Reduction uses thread-local winners, followed by a deterministic merge.
            #pragma omp parallel if(count >= 1024)
            {
                int best = 0, seed = INT_MAX;
                #pragma omp for nowait
                for (int i = 0; i < count; ++i) {
                    int candidate = seeds[offset + i];
                    if (sizes[i] > best || (sizes[i] == best && candidate < seed)) {
                        best = sizes[i]; seed = candidate;
                    }
                }
                #pragma omp critical
                {
                    if (best > local_best || (best == local_best && seed < local_seed)) {
                        local_best = best; local_seed = seed;
                    }
                }
            }
        }
        int input[2] = {local_best, local_seed}, winner[2];
        MPI_Allreduce(input, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (winner[0] <= 0 || winner[1] == INT_MAX) break;
        Cluster cluster;
        cluster.seed_point = winner[1];
        generateCandidateCluster(cluster.seed_point, clustered, points, threshold, n, &cluster.members);
        for (int member : cluster.members) { clustered[member] = true; flags[member] = 1; }
        remaining -= static_cast<int>(cluster.members.size());
        clusters.push_back(std::move(cluster));
    }
    cudaCheck(cudaFree(device_maxima));
    cudaCheck(cudaFree(device_sizes));
    cudaCheck(cudaFree(device_seeds));
    cudaCheck(cudaFree(device_clustered));
    cudaCheck(cudaFree(device_points));
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
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalize_mpi;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0 && !freopen("/dev/null", "w", stdout)) MPI_Abort(MPI_COMM_WORLD, 1);
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

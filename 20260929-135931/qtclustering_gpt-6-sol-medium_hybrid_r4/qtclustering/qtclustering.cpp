// QT Clustering Benchmark - MPI, OpenMP, and CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
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

static void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One block builds one candidate cluster. Candidate diameters are updated
// incrementally as each new member is added, so each pair is visited once.
__global__ void candidateKernel(const Point* points, const unsigned char* clustered,
                                const int* seeds, int* sizes, double* diameters,
                                int n, int count, double threshold) {
    const int seed_id = blockIdx.x;
    if (seed_id >= count) return;
    const int seed = seeds[seed_id];
    double* state = diameters + static_cast<size_t>(seed_id) * n;
    __shared__ double best_dist[256];
    __shared__ int best_index[256];
    __shared__ int current;
    __shared__ int cardinality;
    if (threadIdx.x == 0) { current = seed; cardinality = 1; }
    __syncthreads();
    for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
        state[candidate] = candidate == seed ? -1.0 : 0.0;
    __syncthreads();

    while (true) {
        double local_dist = DBL_MAX;
        int local_index = INT_MAX;
        const Point member = points[current];
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate] || state[candidate] < 0.0 ||
                state[candidate] >= threshold) continue;
            const double dx = points[candidate].x - member.x;
            const double dy = points[candidate].y - member.y;
            const double distance = sqrt(dx * dx + dy * dy);
            const double diameter = fmax(state[candidate], distance);
            state[candidate] = diameter;
            if (diameter < threshold &&
                (diameter < local_dist || (diameter == local_dist && candidate < local_index))) {
                local_dist = diameter;
                local_index = candidate;
            }
        }
        best_dist[threadIdx.x] = local_dist;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset &&
                (best_dist[threadIdx.x + offset] < best_dist[threadIdx.x] ||
                 (best_dist[threadIdx.x + offset] == best_dist[threadIdx.x] &&
                  best_index[threadIdx.x + offset] < best_index[threadIdx.x]))) {
                best_dist[threadIdx.x] = best_dist[threadIdx.x + offset];
                best_index[threadIdx.x] = best_index[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            current = best_index[0];
            if (current != INT_MAX) {
                state[current] = -1.0;
                ++cardinality;
            }
        }
        __syncthreads();
        if (current == INT_MAX) break;
    }
    if (threadIdx.x == 0) sizes[seed_id] = cardinality;
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
int findClosestPoint(const int latest_member,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count,
                     std::vector<double>& diameters) {
    const int threads = std::min(omp_get_max_threads(), 8);
    std::vector<double> thread_dist(threads, DBL_MAX);
    std::vector<int> thread_index(threads, INT_MAX);
#pragma omp parallel num_threads(threads) if(point_count >= 256)
    {
        const int tid = omp_get_thread_num();
        double local_dist = DBL_MAX;
        int local_index = INT_MAX;
#pragma omp for schedule(static)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate] ||
                diameters[candidate] >= threshold) continue;
            // A candidate's diameter is the maximum over members seen so far.
            const double max_dist = std::max(diameters[candidate],
                                             distance(points[candidate], points[latest_member]));
            diameters[candidate] = max_dist;
            if (max_dist < threshold &&
                (max_dist < local_dist || (max_dist == local_dist && candidate < local_index))) {
                local_dist = max_dist;
                local_index = candidate;
            }
        }
        thread_dist[tid] = local_dist;
        thread_index[tid] = local_index;
    }
    double min_diameter = DBL_MAX;
    int closest_point = INT_MAX;
    for (int t = 0; t < threads; ++t) {
        if (thread_dist[t] < min_diameter ||
            (thread_dist[t] == min_diameter && thread_index[t] < closest_point)) {
            min_diameter = thread_dist[t];
            closest_point = thread_index[t];
        }
    }
    return closest_point == INT_MAX ? -1 : closest_point;
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
    std::vector<double> diameters(point_count, 0.0);
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members.back(), clustered, in_cluster, points,
                                             threshold, point_count, diameters);
        
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
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;
    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_seeds = nullptr;
    int* device_sizes = nullptr;
    double* device_diameters = nullptr;
    const int batch = std::min(N, 256);
    cudaCheck(cudaMalloc(&device_points, static_cast<size_t>(N) * sizeof(Point)));
    cudaCheck(cudaMalloc(&device_clustered, N));
    cudaCheck(cudaMalloc(&device_seeds, static_cast<size_t>(batch) * sizeof(int)));
    cudaCheck(cudaMalloc(&device_sizes, static_cast<size_t>(batch) * sizeof(int)));
    cudaCheck(cudaMalloc(&device_diameters, static_cast<size_t>(batch) * N * sizeof(double)));
    cudaCheck(cudaMemcpy(device_points, points.data(), static_cast<size_t>(N) * sizeof(Point), cudaMemcpyHostToDevice));
    std::vector<unsigned char> active(N);
    std::vector<int> seeds;
    std::vector<int> sizes(batch);
    int remaining = N;
    while (remaining > 0) {
        for (int i = 0; i < N; ++i) active[i] = clustered[i] ? 1 : 0;
        cudaCheck(cudaMemcpy(device_clustered, active.data(), N, cudaMemcpyHostToDevice));
        seeds.clear();
        for (int seed = rank; seed < N; seed += ranks)
            if (!clustered[seed]) seeds.push_back(seed);
        int local_size = -1;
        int local_seed = INT_MAX;
        for (size_t start = 0; start < seeds.size(); start += batch) {
            const int count = static_cast<int>(std::min(static_cast<size_t>(batch), seeds.size() - start));
            cudaCheck(cudaMemcpy(device_seeds, seeds.data() + start,
                                 static_cast<size_t>(count) * sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<count, 256>>>(device_points, device_clustered, device_seeds,
                                            device_sizes, device_diameters, N, count, threshold);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(sizes.data(), device_sizes,
                                 static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int j = 0; j < count; ++j) {
                const int seed = seeds[start + j];
                if (sizes[j] > local_size || (sizes[j] == local_size && seed < local_seed)) {
                    local_size = sizes[j];
                    local_seed = seed;
                }
            }
        }
        int local_pair[2] = {local_size, local_seed};
        int global_pair[2];
        cudaCheck(cudaGetLastError());
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_seed = global_pair[1];
        if (best_seed == INT_MAX) break;
        // Every rank reconstructs the same winning cluster before the next round.
        Cluster cluster;
        cluster.seed_point = best_seed;
        generateCandidateCluster(best_seed, clustered, points, threshold, N, &cluster.members);
        for (int member : cluster.members) clustered[member] = true;
        remaining -= static_cast<int>(cluster.members.size());
        clusters.push_back(std::move(cluster));
    }
    cudaFree(device_diameters);
    cudaFree(device_sizes);
    cudaFree(device_seeds);
    cudaFree(device_clustered);
    cudaFree(device_points);
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

    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count));
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    
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

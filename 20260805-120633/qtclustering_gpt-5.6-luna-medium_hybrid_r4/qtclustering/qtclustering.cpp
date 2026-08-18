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

// One CUDA thread evaluates one possible next member.  The host performs the
// deterministic reduction (including the original strict tie-breaking rule).
__global__ void candidateDistances(const Point* points, const unsigned char* clustered,
                                   const unsigned char* in_cluster, const int* members,
                                   int member_count, int point_count, double threshold,
                                   double* distances) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= point_count || clustered[candidate] || in_cluster[candidate]) {
        if (candidate < point_count) distances[candidate] = 1.7976931348623157e308;
        return;
    }
    double max_dist = 0.0;
    for (int i = 0; i < member_count; ++i) {
        const Point a = points[candidate];
        const Point b = points[members[i]];
        const double dx = a.x - b.x;
        const double dy = a.y - b.y;
        max_dist = fmax(max_dist, sqrt(dx * dx + dy * dy));
        if (max_dist >= threshold) break;
    }
    distances[candidate] = (max_dist < threshold) ? max_dist : 1.7976931348623157e308;
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

class GpuCandidateEvaluator {
public:
    GpuCandidateEvaluator(const std::vector<Point>& points, int device, bool enabled)
        : n_(static_cast<int>(points.size())), device_(device), enabled_(enabled) {
        if (!enabled_) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
        cudaCheck(cudaMalloc(&d_points_, n_ * sizeof(Point)), "cudaMalloc(points)");
        cudaCheck(cudaMalloc(&d_clustered_, n_), "cudaMalloc(clustered)");
        cudaCheck(cudaMalloc(&d_in_cluster_, n_), "cudaMalloc(in_cluster)");
        cudaCheck(cudaMalloc(&d_members_, n_ * sizeof(int)), "cudaMalloc(members)");
        cudaCheck(cudaMalloc(&d_distances_, n_ * sizeof(double)), "cudaMalloc(distances)");
        cudaCheck(cudaMemcpyAsync(d_points_, points.data(), n_ * sizeof(Point),
                                  cudaMemcpyHostToDevice, stream_), "copy points");
        cudaCheck(cudaStreamSynchronize(stream_), "upload points");
    }

    ~GpuCandidateEvaluator() {
        if (!enabled_) return;
        cudaSetDevice(device_);
        cudaFree(d_points_); cudaFree(d_clustered_); cudaFree(d_in_cluster_);
        cudaFree(d_members_); cudaFree(d_distances_); cudaStreamDestroy(stream_);
    }

    int closest(const std::vector<int>& members, const std::vector<bool>& clustered,
                const std::vector<unsigned char>& in_cluster, const std::vector<Point>&,
                double threshold) {
        if (!enabled_) return -2;
        std::vector<unsigned char> clustered_bytes(n_);
        for (int i = 0; i < n_; ++i) clustered_bytes[i] = clustered[i] ? 1 : 0;
        cudaCheck(cudaMemcpyAsync(d_clustered_, clustered_bytes.data(), n_,
                                  cudaMemcpyHostToDevice, stream_), "copy clustered");
        cudaCheck(cudaMemsetAsync(d_in_cluster_, 0, n_, stream_), "clear in_cluster");
        cudaCheck(cudaMemcpyAsync(d_members_, members.data(), members.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_), "copy members");
        cudaCheck(cudaMemcpyAsync(d_in_cluster_, in_cluster.data(), n_,
                                  cudaMemcpyHostToDevice, stream_), "copy in_cluster");
        candidateDistances<<<(n_ + 255) / 256, 256, 0, stream_>>>(
            d_points_, d_clustered_, d_in_cluster_, d_members_, static_cast<int>(members.size()),
            n_, threshold, d_distances_);
        cudaCheck(cudaGetLastError(), "candidateDistances");
        cudaCheck(cudaStreamSynchronize(stream_), "candidateDistances synchronize");
        cudaCheck(cudaMemcpy(host_distances_.data(), d_distances_, n_ * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy distances");
        int best = -1;
        double best_distance = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < n_; ++candidate) {
            if (host_distances_[candidate] < best_distance) {
                best_distance = host_distances_[candidate];
                best = candidate;
            }
        }
        return best;
    }

    bool enabled() const { return enabled_; }
private:
    int n_;
    int device_ = 0;
    bool enabled_ = false;
    cudaStream_t stream_ = nullptr;
    Point* d_points_ = nullptr;
    unsigned char* d_clustered_ = nullptr;
    unsigned char* d_in_cluster_ = nullptr;
    int* d_members_ = nullptr;
    double* d_distances_ = nullptr;
    std::vector<double> host_distances_ = std::vector<double>(n_);
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
        int group_cnt = std::max(1, static_cast<int>(frand() * (N / 30.0)));
        
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

int generateCandidateClusterAccelerated(const int seed_point,
                                        const std::vector<bool>& clustered,
                                        const std::vector<Point>& points,
                                        double threshold, GpuCandidateEvaluator& gpu,
                                        std::vector<int>* cluster_members) {
    std::vector<unsigned char> in_cluster(points.size(), 0);
    std::vector<int> members(1, seed_point);
    in_cluster[seed_point] = 1;
    while (static_cast<int>(members.size()) < static_cast<int>(points.size())) {
        const int closest = gpu.closest(members, clustered, in_cluster, points, threshold);
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
    }
    if (cluster_members) *cluster_members = std::move(members);
    return static_cast<int>(cluster_members ? cluster_members->size() : members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int mpi_rank, int mpi_size,
                                  int cuda_device, bool use_cuda) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Each rank owns a disjoint seed subset; OpenMP overlaps those CUDA
        // streams (one evaluator per thread) and the MPI reduction selects the
        // globally largest cluster, preferring the lowest seed exactly as before.
        int thread_count = 1;
        #pragma omp parallel
        #pragma omp single
        thread_count = omp_get_num_threads();
        std::vector<int> local_cardinality(thread_count, -1);
        std::vector<int> local_seed(thread_count, -1);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            GpuCandidateEvaluator gpu(points, cuda_device, use_cuda);
            int thread_best_cardinality = -1;
            int thread_best_seed = -1;
            #pragma omp for schedule(dynamic, 1)
            for (int i = mpi_rank; i < static_cast<int>(unclustered_indices.size()); i += mpi_size) {
                const int seed = unclustered_indices[i];
                std::vector<int> candidate_members;
                const int cardinality = gpu.enabled()
                    ? generateCandidateClusterAccelerated(seed, clustered, points, threshold, gpu,
                                                          &candidate_members)
                    : generateCandidateCluster(seed, clustered, points, threshold, N,
                                               &candidate_members);
                if (cardinality > thread_best_cardinality ||
                    (cardinality == thread_best_cardinality && seed < thread_best_seed)) {
                    thread_best_cardinality = cardinality;
                    thread_best_seed = seed;
                }
            }
            local_cardinality[tid] = thread_best_cardinality;
            local_seed[tid] = thread_best_seed;
        }
        for (int t = 0; t < thread_count; ++t) {
            if (local_cardinality[t] > max_cardinality ||
                (local_cardinality[t] == max_cardinality && local_seed[t] >= 0 &&
                 local_seed[t] < best_seed)) {
                max_cardinality = local_cardinality[t];
                best_seed = local_seed[t];
            }
        }
        int global_cardinality = -1;
        MPI_Allreduce(&max_cardinality, &global_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int local_winning_seed = (max_cardinality == global_cardinality) ? best_seed : N;
        MPI_Allreduce(&local_winning_seed, &best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        max_cardinality = global_cardinality;
        if (best_seed >= 0 && best_seed < N)
            generateCandidateCluster(best_seed, clustered, points, threshold, N, &best_cluster_members);
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
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
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Comm_free(&local_comm);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mpi_size, omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (mpi_rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    const bool use_cuda = device_status == cudaSuccess && device_count > 0;
    const int cuda_device = use_cuda ? local_rank % device_count : 0;
    if (mpi_rank == 0)
        printf("CUDA: %s%s\n", use_cuda ? "enabled" : "unavailable (CPU fallback)",
               use_cuda ? "" : "; MPI/OpenMP remain active");
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    MPI_Barrier(MPI_COMM_WORLD);
    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size,
                                                       cuda_device, use_cuda);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }
    
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
    
    if (mpi_rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
    }
    
    // Performance metrics
    const double time_sec = std::max(1e-9, cluster_time.count() / 1000.0);
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    if (mpi_rank == 0)
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
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
    if (validate && mpi_rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}

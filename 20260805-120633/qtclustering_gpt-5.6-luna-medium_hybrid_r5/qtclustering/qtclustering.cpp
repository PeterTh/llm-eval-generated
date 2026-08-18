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

// One CUDA thread scores one possible next member.  Squared distances are
// The kernel retains the original distance calculation and strict comparison,
// including its deterministic candidate-order tie break on the host.
__global__ void scoreCandidatesKernel(const Point* points, const int* members,
                                      int member_count, const unsigned char* clustered,
                                      int point_count, double threshold,
                                      double* scores) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= point_count || clustered[candidate]) return;
    double max_squared = 0.0;
    for (int i = 0; i < member_count; ++i) {
        const Point a = points[candidate];
        const Point b = points[members[i]];
        const double dx = a.x - b.x;
        const double dy = a.y - b.y;
        max_squared = fmax(max_squared, sqrt(dx * dx + dy * dy));
    }
    scores[candidate] = (max_squared < threshold) ? max_squared : INFINITY;
}

class CudaScorer {
public:
    CudaScorer(const std::vector<Point>& points, const std::vector<bool>& clustered,
               double threshold, int rank = 0)
        : n_(static_cast<int>(points.size())), threshold_(threshold),
          h_clustered_(n_, 0), h_scores_(n_, INFINITY) {
        int devices = 0;
        active_ = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
        if (!active_) return;
        cudaSetDevice(rank % devices);
        for (int i = 0; i < n_; ++i) h_clustered_[i] = clustered[i] ? 1 : 0;
        cudaMalloc(&d_points_, sizeof(Point) * n_);
        cudaMalloc(&d_members_, sizeof(int) * n_);
        cudaMalloc(&d_clustered_, sizeof(unsigned char) * n_);
        cudaMalloc(&d_scores_, sizeof(double) * n_);
        cudaMemcpy(d_points_, points.data(), sizeof(Point) * n_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_clustered_, h_clustered_.data(), sizeof(unsigned char) * n_, cudaMemcpyHostToDevice);
    }
    ~CudaScorer() {
        if (active_) { cudaFree(d_points_); cudaFree(d_members_); cudaFree(d_clustered_); cudaFree(d_scores_); }
    }
    bool active() const { return active_; }
    int best(const std::vector<int>& members) {
        cudaMemcpy(d_members_, members.data(), sizeof(int) * members.size(), cudaMemcpyHostToDevice);
        cudaMemset(d_scores_, 0xFF, sizeof(double) * n_);
        scoreCandidatesKernel<<<(n_ + 255) / 256, 256>>>(d_points_, d_members_,
            static_cast<int>(members.size()), d_clustered_, n_, threshold_, d_scores_);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(h_scores_.data(), d_scores_, sizeof(double) * n_, cudaMemcpyDeviceToHost) != cudaSuccess) {
            active_ = false; return -2;
        }
        int selected = -1; double best_score = std::numeric_limits<double>::max();
        for (int member : members) h_scores_[member] = INFINITY;
        for (int candidate = 0; candidate < n_; ++candidate)
            if (!h_clustered_[candidate] && h_scores_[candidate] < best_score) {
                best_score = h_scores_[candidate]; selected = candidate;
            }
        return selected;
    }
private:
    int n_; double threshold_; bool active_ = false;
    Point* d_points_ = nullptr; int* d_members_ = nullptr;
    unsigned char* d_clustered_ = nullptr; double* d_scores_ = nullptr;
    std::vector<unsigned char> h_clustered_; std::vector<double> h_scores_;
};

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

int findClosestPointCPU(const std::vector<int>& cluster_members,
                        const std::vector<bool>& clustered,
                        const std::vector<bool>& in_cluster,
                        const std::vector<Point>& points, double threshold) {
    int closest = -1; double min_diameter = std::numeric_limits<double>::max();
    #pragma omp parallel
    {
        int local_point = -1; double local_min = std::numeric_limits<double>::max();
        #pragma omp for nowait
        for (int candidate = 0; candidate < static_cast<int>(points.size()); ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            double max_dist = 0.0;
            for (int member : cluster_members) max_dist = std::max(max_dist, distance(points[candidate], points[member]));
            if (max_dist < threshold && max_dist < local_min) { local_min = max_dist; local_point = candidate; }
        }
        #pragma omp critical
        if (local_min < min_diameter || (local_min == min_diameter && local_point >= 0 &&
                                         (closest < 0 || local_point < closest))) {
            min_diameter = local_min; closest = local_point;
        }
    }
    return closest;
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

int generateCandidateClusterCUDA(int seed_point, const std::vector<bool>& clustered,
                                 const std::vector<Point>& points, double threshold,
                                 CudaScorer& scorer, std::vector<int>* output) {
    std::vector<bool> in_cluster(points.size(), false);
    std::vector<int> members(1, seed_point); in_cluster[seed_point] = true;
    while (members.size() < points.size()) {
        const int closest = scorer.best(members);
        if (closest < 0 || in_cluster[closest]) break;
        in_cluster[closest] = true; members.push_back(closest);
    }
    if (output) *output = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int ranks) {
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
        
        // Try each unclustered point as a seed
        CudaScorer scorer(points, clustered, threshold, rank);
        int local_cardinality = -1, local_seed = -1;
        if (scorer.active()) {
            for (size_t i = rank; i < unclustered_indices.size(); i += ranks) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                std::vector<int> candidate_members;
                const int cardinality = generateCandidateClusterCUDA(seed, clustered, points, threshold, scorer, &candidate_members);
                if (cardinality > local_cardinality || (cardinality == local_cardinality && seed < local_seed)) {
                    local_cardinality = cardinality; local_seed = seed;
                }
            }
        } else {
            #pragma omp parallel
            {
                int thread_cardinality = -1, thread_seed = -1;
                #pragma omp for schedule(static)
                for (int i = rank; i < static_cast<int>(unclustered_indices.size()); i += ranks) {
                    const int seed = unclustered_indices[i];
                    if (clustered[seed]) continue;
                    const int cardinality = generateCandidateCluster(seed, clustered, points, threshold, N, nullptr);
                    if (cardinality > thread_cardinality || (cardinality == thread_cardinality && seed < thread_seed)) {
                        thread_cardinality = cardinality; thread_seed = seed;
                    }
                }
                #pragma omp critical
                if (thread_cardinality > local_cardinality || (thread_cardinality == local_cardinality &&
                    thread_seed >= 0 && (local_seed < 0 || thread_seed < local_seed))) {
                    local_cardinality = thread_cardinality; local_seed = thread_seed;
                }
            }
        }

        MPI_Allreduce(&local_cardinality, &max_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        const int eligible_seed = (local_cardinality == max_cardinality) ? local_seed : N;
        MPI_Allreduce(&eligible_seed, &best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            if (rank == 0) {
                Cluster cluster;
                cluster.seed_point = best_seed;
                generateCandidateCluster(best_seed, clustered, points, threshold, N, &cluster.members);
                best_cluster_members = cluster.members;
            }
            int member_count = rank == 0 ? static_cast<int>(best_cluster_members.size()) : 0;
            MPI_Bcast(&member_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank != 0) best_cluster_members.resize(member_count);
            MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            }
            
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI ranks: %d, OpenMP threads/rank: %d, CUDA enabled)\n", ranks, omp_get_max_threads());
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);

    const double local_time = MPI_Wtime() - cluster_start;
    double cluster_time_seconds = 0.0;
    MPI_Reduce(&local_time, &cluster_time_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) { MPI_Finalize(); return 0; }
    const long cluster_time_ms = static_cast<long>(cluster_time_seconds * 1000.0);
    
    printf("Clustering time: %ld ms\n", cluster_time_ms);
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
    const double time_sec = cluster_time_seconds;
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

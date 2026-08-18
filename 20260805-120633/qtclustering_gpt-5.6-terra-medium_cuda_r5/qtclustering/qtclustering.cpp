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

// Candidate order is the original sequential tie-break: a smaller diameter
// wins, and equal diameters select the smaller point index.
struct DeviceCandidate {
    double diameter;
    int index;
};

__device__ inline DeviceCandidate betterCandidate(DeviceCandidate a,
                                                   DeviceCandidate b) {
    return (b.diameter < a.diameter ||
            (b.diameter == a.diameter && b.index < a.index)) ? b : a;
}

// One persistent block builds each seed's candidate cluster.  All remaining
// seeds run concurrently, exposing the substantial seed-level parallelism.
__global__ void generateCandidateKernel(const Point* points,
                                        const unsigned char* clustered,
                                        unsigned char* in_cluster,
                                        int point_count, double threshold,
                                        int* members, int* member_count) {
    __shared__ DeviceCandidate reduction[256];
    __shared__ DeviceCandidate best;
    __shared__ int count;
    const int tid = threadIdx.x;
    const int seed = blockIdx.x;
    unsigned char* const local_cluster = in_cluster + static_cast<size_t>(seed) * point_count;
    int* const local_members = members + static_cast<size_t>(seed) * point_count;

    if (clustered[seed]) {
        if (tid == 0) member_count[seed] = 0;
        return;
    }

    for (int i = tid; i < point_count; i += blockDim.x) local_cluster[i] = 0;
    __syncthreads();
    if (tid == 0) {
        local_cluster[seed] = 1;
        local_members[0] = seed;
        count = 1;
    }
    __syncthreads();

    while (count < point_count) {
        if (tid == 0) best = {1.0e300, point_count};
        __syncthreads();

        // Process the candidate space in tiles.  Each tile reduction and the
        // final comparison use the same index tie-break as the CPU loop.
        for (int base = 0; base < point_count; base += blockDim.x) {
            const int candidate = base + tid;
            DeviceCandidate result{1.0e300, point_count};
            if (candidate < point_count && !clustered[candidate] && !local_cluster[candidate]) {
                const Point point = points[candidate];
                double max_distance = 0.0;
                bool eligible = true;
                for (int member = 0; member < point_count; ++member) {
                    if (local_cluster[member]) {
                        const double dx = point.x - points[member].x;
                        const double dy = point.y - points[member].y;
                        const double d = sqrt(dx * dx + dy * dy);
                        if (d >= threshold) {
                            eligible = false;
                            break;
                        }
                        max_distance = fmax(max_distance, d);
                    }
                }
                if (eligible) result = {max_distance, candidate};
            }
            reduction[tid] = result;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) reduction[tid] = betterCandidate(reduction[tid], reduction[tid + stride]);
                __syncthreads();
            }
            if (tid == 0) best = betterCandidate(best, reduction[0]);
            __syncthreads();
        }
        if (best.index == point_count) break;
        if (tid == 0) {
            local_cluster[best.index] = 1;
            local_members[count++] = best.index;
        }
        __syncthreads();
    }
    if (tid == 0) member_count[seed] = count;
}

__global__ void markClusteredKernel(unsigned char* flags, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) flags[members[i]] = 1;
}

static void cudaCheck(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", action,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

class GpuQtWorkspace {
public:
    explicit GpuQtWorkspace(const std::vector<Point>& points) : count_(static_cast<int>(points.size())) {
        int devices = 0;
        cudaCheck(cudaGetDeviceCount(&devices), "CUDA device discovery");
        if (devices == 0) {
            std::fprintf(stderr, "CUDA error: no CUDA-capable device is available\n");
            std::exit(EXIT_FAILURE);
        }
        cudaCheck(cudaSetDevice(0), "CUDA device selection");
        cudaCheck(cudaMalloc(&points_, sizeof(Point) * count_), "point allocation");
        cudaCheck(cudaMalloc(&clustered_, count_), "cluster-state allocation");
        cudaCheck(cudaMalloc(&in_cluster_, static_cast<size_t>(count_) * count_), "candidate-state allocation");
        cudaCheck(cudaMalloc(&members_, sizeof(int) * static_cast<size_t>(count_) * count_), "member allocation");
        cudaCheck(cudaMalloc(&candidate_count_, sizeof(int) * count_), "candidate count allocation");
        cudaCheck(cudaMemcpy(points_, points.data(), sizeof(Point) * count_, cudaMemcpyHostToDevice),
                  "point upload");
        cudaCheck(cudaMemset(clustered_, 0, count_), "cluster-state initialization");
    }

    ~GpuQtWorkspace() {
        cudaFree(candidate_count_); cudaFree(members_); cudaFree(in_cluster_);
        cudaFree(clustered_); cudaFree(points_);
    }

    std::vector<int> generateCandidateClusters(double threshold) {
        generateCandidateKernel<<<count_, kThreads>>>(points_, clustered_, in_cluster_, count_, threshold,
                                                       members_, candidate_count_);
        cudaCheck(cudaGetLastError(), "candidate generation kernel launch");
        std::vector<int> counts(count_);
        cudaCheck(cudaMemcpy(counts.data(), candidate_count_, sizeof(int) * count_, cudaMemcpyDeviceToHost),
                  "candidate count download");
        return counts;
    }

    std::vector<int> candidateMembers(int seed, int member_count) const {
        std::vector<int> result(member_count);
        cudaCheck(cudaMemcpy(result.data(), members_ + static_cast<size_t>(seed) * count_,
                             sizeof(int) * member_count, cudaMemcpyDeviceToHost),
                  "candidate member download");
        return result;
    }

    void markClustered(const std::vector<int>& members) {
        cudaCheck(cudaMemcpy(members_, members.data(), sizeof(int) * members.size(), cudaMemcpyHostToDevice),
                  "cluster member upload");
        markClusteredKernel<<<(static_cast<int>(members.size()) + kThreads - 1) / kThreads, kThreads>>>(
            clustered_, members_, static_cast<int>(members.size()));
        cudaCheck(cudaGetLastError(), "cluster marking kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "cluster marking synchronization");
    }

private:
    static constexpr int kThreads = 256;
    int count_;
    Point* points_ = nullptr;
    unsigned char* clustered_ = nullptr;
    unsigned char* in_cluster_ = nullptr;
    int* members_ = nullptr;
    int* candidate_count_ = nullptr;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    GpuQtWorkspace gpu(points);

    // Main clustering loop.  The dependencies between selected clusters are
    // inherently greedy; candidate scoring within every dependency level is GPU parallel.
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        const std::vector<int> candidate_counts = gpu.generateCandidateClusters(threshold);
        // Select the largest candidate in original seed order.
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            const int cardinality = candidate_counts[seed];
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }
        if (best_seed >= 0) best_cluster_members = gpu.candidateMembers(best_seed, max_cardinality);
        
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
            gpu.markClustered(best_cluster_members);
            
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

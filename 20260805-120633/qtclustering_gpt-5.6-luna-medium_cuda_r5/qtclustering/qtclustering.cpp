// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
namespace {

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// One block owns one seed.  Candidate points are distributed over the block;
// the reduction uses candidate index as a deterministic tie breaker, matching
// the original ascending candidate loop and its strict '<' update.
__global__ void chooseAndAppend(const Point* points, const unsigned char* clustered,
                                int* members, int* member_counts,
                                int* chosen, const int point_count,
                                const double threshold) {
    const int seed = static_cast<int>(blockIdx.x);
    if (seed >= point_count) return;

    __shared__ double best_dist[256];
    __shared__ int best_candidate[256];
    const int lane = static_cast<int>(threadIdx.x);
    const int base = seed * point_count;
    const int member_count = member_counts[seed];

    double local_dist = DBL_MAX;
    int local_candidate = -1;
    if (!clustered[seed] && member_count < point_count) {
        for (int candidate = lane; candidate < point_count; candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            bool already_member = false;
            for (int i = 0; i < member_count; ++i) {
                if (members[base + i] == candidate) {
                    already_member = true;
                    break;
                }
            }
            if (already_member) continue;

            double max_dist = 0.0;
            for (int i = 0; i < member_count; ++i) {
                const Point a = points[candidate];
                const Point b = points[members[base + i]];
                const double dx = a.x - b.x;
                const double dy = a.y - b.y;
                max_dist = fmax(max_dist, sqrt(dx * dx + dy * dy));
            }
            if (max_dist < threshold &&
                (local_candidate < 0 || max_dist < local_dist ||
                 (max_dist == local_dist && candidate < local_candidate))) {
                local_dist = max_dist;
                local_candidate = candidate;
            }
        }
    }

    best_dist[lane] = local_dist;
    best_candidate[lane] = local_candidate;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) {
            const int other = best_candidate[lane + stride];
            if (other >= 0 &&
                (best_candidate[lane] < 0 || best_dist[lane + stride] < best_dist[lane] ||
                 (best_dist[lane + stride] == best_dist[lane] && other < best_candidate[lane]))) {
                best_dist[lane] = best_dist[lane + stride];
                best_candidate[lane] = other;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        const int candidate = best_candidate[0];
        chosen[seed] = candidate;
        if (candidate >= 0) {
            members[base + member_count] = candidate;
            member_counts[seed] = member_count + 1;
        }
    }
}

__global__ void initializeSeedClusters(const unsigned char* clustered, int* members,
                                       int* member_counts, const int point_count) {
    const int seed = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (seed < point_count && !clustered[seed]) {
        member_counts[seed] = 1;
        members[seed * point_count] = seed;
    }
}

} // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_members = nullptr;
    int* device_member_counts = nullptr;
    int* device_chosen = nullptr;
    CUDA_CHECK(cudaMalloc(&device_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&device_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&device_members, sizeof(int) * static_cast<size_t>(N) * N));
    CUDA_CHECK(cudaMalloc(&device_member_counts, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&device_chosen, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(device_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    while (true) {
        CUDA_CHECK(cudaMemcpy(device_clustered, clustered.data(), sizeof(unsigned char) * N,
                              cudaMemcpyHostToDevice));
        initializeSeedClusters<<<(N + 255) / 256, 256>>>(device_clustered, device_members,
                                                          device_member_counts, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Each launch advances every still-unclustered seed by one greedy step.
        // The host only coordinates the inherently sequential QT rounds.
        std::vector<int> chosen(N);
        while (true) {
            chooseAndAppend<<<N, 256>>>(device_points, device_clustered, device_members,
                                        device_member_counts, device_chosen, N, threshold);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(chosen.data(), device_chosen, sizeof(int) * N,
                                  cudaMemcpyDeviceToHost));
            bool advanced = false;
            for (int value : chosen) advanced |= value >= 0;
            if (!advanced) break;
        }

        std::vector<int> member_counts(N);
        CUDA_CHECK(cudaMemcpy(member_counts.data(), device_member_counts, sizeof(int) * N,
                              cudaMemcpyDeviceToHost));
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        for (int seed = 0; seed < N; ++seed) {
            if (clustered[seed]) continue;
            const int cardinality = member_counts[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            best_cluster_members.resize(max_cardinality);
            CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                  device_members + best_seed * N,
                                  sizeof(int) * max_cardinality, cudaMemcpyDeviceToHost));
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
        } else {
            break;
        }
    }

    cudaFree(device_chosen);
    cudaFree(device_member_counts);
    cudaFree(device_members);
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

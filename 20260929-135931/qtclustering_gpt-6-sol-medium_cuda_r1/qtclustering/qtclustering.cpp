// QT Clustering Benchmark - CUDA version
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
#include <stdexcept>
#include <string>
#include <utility>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Rows are cluster members; adjacent threads read adjacent candidates.
__global__ void makeDistances(const Point* points, double* distances, int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(n) * n;
    if (index >= count) return;
    const Point a = points[index / n];
    const Point b = points[index % n];
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    distances[index] = sqrt(dx * dx + dy * dy);
}

// Each block grows one seed's cluster. Candidate diameters are updated only
// for the newly added member, avoiding repeated scans of previous members.
__global__ void candidateClusters(const Point* points, const double* distances,
                                  const unsigned char* clustered, double* global_diameters,
                                  int* cardinalities, int* members, int n,
                                  double threshold, int first_seed, bool shared_state) {
    const int seed = first_seed + blockIdx.x;
    const int lane = threadIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) {
        if (lane == 0) cardinalities[seed] = 0;
        return;
    }

    extern __shared__ double shared_diameters[];
    double* diameters = shared_state ? shared_diameters
        : global_diameters + static_cast<size_t>(blockIdx.x) * n;
    __shared__ double minima[256];
    __shared__ int indices[256];
    __shared__ int last_member;
    __shared__ int size;

    for (int candidate = lane; candidate < n; candidate += blockDim.x) {
        diameters[candidate] = (clustered[candidate] || candidate == seed)
            ? DBL_MAX : 0.0;
    }
    if (lane == 0) {
        last_member = seed;
        size = 1;
        if (members) members[0] = seed;
    }
    __syncthreads();

    while (true) {
        const int new_member = last_member;
        double best_distance = DBL_MAX;
        int best_index = n;
        for (int candidate = lane; candidate < n; candidate += blockDim.x) {
            double diameter = diameters[candidate];
            if (diameter == DBL_MAX) continue;
            double dist;
            if (distances) {
                dist = distances[static_cast<size_t>(new_member) * n + candidate];
            } else {
                const double dx = points[candidate].x - points[new_member].x;
                const double dy = points[candidate].y - points[new_member].y;
                dist = sqrt(dx * dx + dy * dy);
            }
            diameter = fmax(diameter, dist);
            // Diameters can only grow, so an ineligible point stays ineligible.
            if (diameter >= threshold) {
                diameters[candidate] = DBL_MAX;
                continue;
            }
            diameters[candidate] = diameter;
            if (diameter < best_distance ||
                (diameter == best_distance && candidate < best_index)) {
                best_distance = diameter;
                best_index = candidate;
            }
        }
        minima[lane] = best_distance;
        indices[lane] = best_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (lane < offset &&
                (minima[lane + offset] < minima[lane] ||
                 (minima[lane + offset] == minima[lane] &&
                  indices[lane + offset] < indices[lane]))) {
                minima[lane] = minima[lane + offset];
                indices[lane] = indices[lane + offset];
            }
            __syncthreads();
        }
        if (lane == 0) {
            if (minima[0] < threshold) {
                last_member = indices[0];
                diameters[last_member] = DBL_MAX;
                if (members) members[size] = last_member;
                ++size;
            } else {
                last_member = -1;
            }
        }
        __syncthreads();
        if (last_member < 0) break;
    }
    if (lane == 0) cardinalities[seed] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    constexpr int block_size = 256;
    constexpr int shared_limit = 4096;
    constexpr size_t workspace_limit = 256ull * 1024 * 1024;
    const bool shared_state = n <= shared_limit;
    const bool precompute = n <= shared_limit;
    const size_t batch_size = shared_state ? static_cast<size_t>(n)
        : std::max<size_t>(1, std::min<size_t>(n, workspace_limit / (sizeof(double) * n)));

    Point* device_points = nullptr;
    double* device_distances = nullptr;
    double* device_diameters = nullptr;
    unsigned char* device_clustered = nullptr;
    int* device_cardinalities = nullptr;
    int* device_members = nullptr;
    auto cleanup = [&]() {
        cudaFree(device_members);
        cudaFree(device_cardinalities);
        cudaFree(device_clustered);
        cudaFree(device_diameters);
        cudaFree(device_distances);
        cudaFree(device_points);
    };
    try {
        cudaCheck(cudaMalloc(&device_points, sizeof(Point) * n), "allocate points");
        cudaCheck(cudaMalloc(&device_clustered, n), "allocate membership mask");
        cudaCheck(cudaMalloc(&device_cardinalities, sizeof(int) * n), "allocate cardinalities");
        cudaCheck(cudaMalloc(&device_members, sizeof(int) * n), "allocate members");
        if (precompute) {
            const size_t entries = static_cast<size_t>(n) * n;
            cudaCheck(cudaMalloc(&device_distances, sizeof(double) * entries), "allocate distances");
        } else {
            cudaCheck(cudaMalloc(&device_diameters, sizeof(double) * batch_size * n),
                      "allocate candidate state");
        }
        cudaCheck(cudaMemcpy(device_points, points.data(), sizeof(Point) * n,
                             cudaMemcpyHostToDevice), "copy points");
        if (precompute) {
            const size_t entries = static_cast<size_t>(n) * n;
            makeDistances<<<(entries + block_size - 1) / block_size, block_size>>>(
                device_points, device_distances, n);
            cudaCheck(cudaGetLastError(), "compute distances");
        }

        std::vector<unsigned char> clustered(n, 0);
        std::vector<int> cardinalities(n);
        std::vector<Cluster> clusters;
        int remaining = n;
        const size_t shared_bytes = shared_state ? sizeof(double) * n : 0;
        while (remaining > 0) {
            cudaCheck(cudaMemcpy(device_clustered, clustered.data(), n,
                                 cudaMemcpyHostToDevice), "copy membership mask");
            for (size_t first = 0; first < static_cast<size_t>(n); first += batch_size) {
                const int batch = static_cast<int>(std::min(batch_size, static_cast<size_t>(n) - first));
                candidateClusters<<<batch, block_size, shared_bytes>>>(
                    device_points, device_distances, device_clustered,
                    device_diameters, device_cardinalities, nullptr, n, threshold,
                    static_cast<int>(first), shared_state);
            }
            cudaCheck(cudaGetLastError(), "evaluate seeds");
            cudaCheck(cudaMemcpy(cardinalities.data(), device_cardinalities,
                                 sizeof(int) * n, cudaMemcpyDeviceToHost), "copy cardinalities");
            int best_seed = -1;
            int best_size = 0;
            for (int seed = 0; seed < n; ++seed) {
                if (cardinalities[seed] > best_size) {
                    best_size = cardinalities[seed];
                    best_seed = seed;
                }
            }
            if (best_seed < 0) break;
            candidateClusters<<<1, block_size, shared_bytes>>>(
                device_points, device_distances, device_clustered,
                device_diameters, device_cardinalities, device_members, n, threshold,
                best_seed, shared_state);
            cudaCheck(cudaGetLastError(), "reconstruct winning cluster");
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.resize(best_size);
            cudaCheck(cudaMemcpy(cluster.members.data(), device_members,
                                 sizeof(int) * best_size, cudaMemcpyDeviceToHost),
                      "copy winning cluster");
            for (int member : cluster.members) clustered[member] = 1;
            remaining -= best_size;
            clusters.push_back(std::move(cluster));
        }
        cleanup();
        return clusters;
    } catch (...) {
        cleanup();
        throw;
    }
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

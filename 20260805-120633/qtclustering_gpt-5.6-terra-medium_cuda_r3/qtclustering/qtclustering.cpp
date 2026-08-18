// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
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

constexpr int CUDA_THREADS = 256;

[[noreturn]] void cudaFail(cudaError_t status, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation,
            cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) cudaFail(status, operation);
}

// One block constructs the candidate cluster for one seed.  The greedy
// additions for an individual seed are necessarily ordered, but all seeds are
// independent and are therefore built concurrently.  scores[seed * N + p]
// stores p's current maximum distance to that seed's cluster.  Once a point is
// selected its score is set to infinity, which is the compact equivalent of
// the original per-seed in_cluster bitmap.
__global__ void computeDistances(const Point* points, double* distances, int pointCount) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elementCount = static_cast<size_t>(pointCount) * pointCount;
    if (element >= elementCount) return;
    const int source = element / pointCount;
    const int target = element - static_cast<size_t>(source) * pointCount;
    const double dx = points[source].x - points[target].x;
    const double dy = points[source].y - points[target].y;
    distances[element] = sqrt(dx * dx + dy * dy);
}

__global__ void buildCandidates(const double* distances, const unsigned char* clustered,
                                double* scores, int* members, int* sizes,
                                double threshold, int pointCount) {
    const int seed = blockIdx.x;
    if (seed >= pointCount || clustered[seed]) return;

    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(seed) * pointCount;
    extern __shared__ unsigned char reductionStorage[];
    double* reductionDistance = reinterpret_cast<double*>(reductionStorage);
    int* reductionIndex = reinterpret_cast<int*>(reductionDistance + blockDim.x);
    for (int point = tid; point < pointCount; point += blockDim.x) {
        double value = DBL_MAX;
        if (!clustered[point] && point != seed) {
            value = distances[base + point];
        }
        scores[base + point] = value;
    }
    if (tid == 0) {
        members[base] = seed;
        sizes[seed] = 1;
    }
    __syncthreads();

    while (true) {
    double localDistance = DBL_MAX;
    int localIndex = pointCount;
    for (int point = tid; point < pointCount; point += blockDim.x) {
        const double value = scores[base + point];
        if (value < threshold &&
            (value < localDistance || (value == localDistance && point < localIndex))) {
            localDistance = value;
            localIndex = point;
        }
    }
    reductionDistance[tid] = localDistance;
    reductionIndex[tid] = localIndex;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const double otherDistance = reductionDistance[tid + offset];
            const int otherIndex = reductionIndex[tid + offset];
            if (otherDistance < reductionDistance[tid] ||
                (otherDistance == reductionDistance[tid] && otherIndex < reductionIndex[tid])) {
                reductionDistance[tid] = otherDistance;
                reductionIndex[tid] = otherIndex;
            }
        }
        __syncthreads();
    }

    __shared__ int chosen;
    __shared__ int insertion;
    if (tid == 0) {
        chosen = reductionIndex[0] < pointCount ? reductionIndex[0] : -1;
        insertion = sizes[seed];
        if (chosen >= 0) {
            members[base + insertion] = chosen;
            sizes[seed] = insertion + 1;
            scores[base + chosen] = DBL_MAX;
        }
    }
    __syncthreads();
    if (chosen < 0) return;

    // Incorporate the new member into every remaining candidate's diameter.
    for (int point = tid; point < pointCount; point += blockDim.x) {
        const size_t offset = base + point;
        const double oldDistance = scores[offset];
        if (oldDistance != DBL_MAX) {
            const double newDistance = distances[static_cast<size_t>(chosen) * pointCount + point];
            scores[offset] = fmax(oldDistance, newDistance);
        }
    }
    __syncthreads();
    }
}

__global__ void selectLargestCandidate(const int* sizes, const unsigned char* clustered,
                                       int* bestSeed, int* bestSize, int pointCount) {
    __shared__ int blockSize[CUDA_THREADS];
    __shared__ int blockSeed[CUDA_THREADS];
    const int tid = threadIdx.x;
    int size = -1;
    int seed = pointCount;
    for (int point = tid; point < pointCount; point += blockDim.x) {
        if (!clustered[point] &&
            (sizes[point] > size || (sizes[point] == size && point < seed))) {
            size = sizes[point];
            seed = point;
        }
    }
    blockSize[tid] = size;
    blockSeed[tid] = seed;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset &&
            (blockSize[tid + offset] > blockSize[tid] ||
             (blockSize[tid + offset] == blockSize[tid] &&
              blockSeed[tid + offset] < blockSeed[tid]))) {
            blockSize[tid] = blockSize[tid + offset];
            blockSeed[tid] = blockSeed[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *bestSeed = blockSeed[0] < pointCount ? blockSeed[0] : -1;
        *bestSize = blockSize[0];
    }
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
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    double* d_distances = nullptr;
    double* d_scores = nullptr;
    int* d_members = nullptr;
    int* d_sizes = nullptr;
    int* d_best_seed = nullptr;
    int* d_best_size = nullptr;

    const size_t matrixElements = static_cast<size_t>(N) * N;
    cudaCheck(cudaMalloc(&d_points, static_cast<size_t>(N) * sizeof(Point)), "allocating points");
    cudaCheck(cudaMalloc(&d_clustered, static_cast<size_t>(N) * sizeof(unsigned char)),
              "allocating cluster flags");
    cudaCheck(cudaMalloc(&d_distances, matrixElements * sizeof(double)), "allocating distances");
    cudaCheck(cudaMalloc(&d_scores, matrixElements * sizeof(double)), "allocating candidate scores");
    cudaCheck(cudaMalloc(&d_members, matrixElements * sizeof(int)), "allocating candidate members");
    cudaCheck(cudaMalloc(&d_sizes, static_cast<size_t>(N) * sizeof(int)), "allocating candidate sizes");
    cudaCheck(cudaMalloc(&d_best_seed, sizeof(int)), "allocating best seed");
    cudaCheck(cudaMalloc(&d_best_size, sizeof(int)), "allocating best size");
    cudaCheck(cudaMemcpy(d_points, points.data(), static_cast<size_t>(N) * sizeof(Point),
                         cudaMemcpyHostToDevice), "copying points");
    const int distanceBlocks = static_cast<int>((matrixElements + CUDA_THREADS - 1) / CUDA_THREADS);
    computeDistances<<<distanceBlocks, CUDA_THREADS>>>(d_points, d_distances, N);
    cudaCheck(cudaGetLastError(), "precomputing distances");

    const size_t growSharedMemory = CUDA_THREADS * (sizeof(double) + sizeof(int));

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(N),
                             cudaMemcpyHostToDevice), "copying cluster flags");
        buildCandidates<<<N, CUDA_THREADS, growSharedMemory>>>(
            d_distances, d_clustered, d_scores, d_members, d_sizes, threshold, N);
        cudaCheck(cudaGetLastError(), "building candidate clusters");
        selectLargestCandidate<<<1, CUDA_THREADS>>>(d_sizes, d_clustered, d_best_seed,
                                                     d_best_size, N);
        cudaCheck(cudaGetLastError(), "selecting the largest candidate");

        int best_seed = -1;
        int max_cardinality = -1;
        cudaCheck(cudaMemcpy(&best_seed, d_best_seed, sizeof(int), cudaMemcpyDeviceToHost),
                  "copying selected seed");
        cudaCheck(cudaMemcpy(&max_cardinality, d_best_size, sizeof(int), cudaMemcpyDeviceToHost),
                  "copying selected size");
        std::vector<int> best_cluster_members(max_cardinality > 0 ? max_cardinality : 0);
        if (max_cardinality > 0) {
            cudaCheck(cudaMemcpy(best_cluster_members.data(),
                                 d_members + static_cast<size_t>(best_seed) * N,
                                 static_cast<size_t>(max_cardinality) * sizeof(int),
                                 cudaMemcpyDeviceToHost), "copying selected members");
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    cudaCheck(cudaFree(d_best_size), "freeing best size");
    cudaCheck(cudaFree(d_best_seed), "freeing best seed");
    cudaCheck(cudaFree(d_sizes), "freeing candidate sizes");
    cudaCheck(cudaFree(d_members), "freeing candidate members");
    cudaCheck(cudaFree(d_scores), "freeing candidate scores");
    cudaCheck(cudaFree(d_distances), "freeing distances");
    cudaCheck(cudaFree(d_clustered), "freeing cluster flags");
    cudaCheck(cudaFree(d_points), "freeing points");
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

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

// CUDA errors are fatal: this benchmark deliberately has no CPU fallback.
static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceBest {
    double diameter;
    int index;
};

__device__ inline DeviceBest better(DeviceBest a, DeviceBest b) {
    if (b.diameter < a.diameter ||
        (b.diameter == a.diameter && b.index < a.index)) return b;
    return a;
}

__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        const double dx = points[row].x - points[col].x;
        const double dy = points[row].y - points[col].y;
        distances[static_cast<size_t>(row) * n + col] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void initializeCandidates(const int* seeds, unsigned char* inCluster,
                                     int* members, int* sizes, int seedCount, int n) {
    const int row = blockIdx.x;
    if (row < seedCount && threadIdx.x == 0) {
        const int seed = seeds[row];
        inCluster[static_cast<size_t>(row) * n + seed] = 1;
        members[static_cast<size_t>(row) * n] = seed;
        sizes[row] = 1;
    }
}

// One block simulates one seed.  Every thread evaluates a striped subset of
// candidate points, then reduces with the exact sequential tie rule.
__global__ void selectNextPoints(const unsigned char* clustered,
                                 const unsigned char* inCluster,
                                 const int* members, const int* sizes,
                                 const double* distances, double threshold,
                                 int* choices, int seedCount, int n) {
    const int row = blockIdx.x;
    if (row >= seedCount) return;
    const size_t base = static_cast<size_t>(row) * n;
    const int count = sizes[row];
    DeviceBest best{DBL_MAX, n};
    for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
        if (!clustered[candidate] && !inCluster[base + candidate]) {
            double maximum = 0.0;
            for (int i = 0; i < count; ++i) {
                maximum = fmax(maximum, distances[static_cast<size_t>(candidate) * n + members[base + i]]);
            }
            if (maximum < threshold) best = better(best, DeviceBest{maximum, candidate});
        }
    }
    __shared__ DeviceBest reduced[256];
    reduced[threadIdx.x] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduced[threadIdx.x] = better(reduced[threadIdx.x], reduced[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) choices[row] = reduced[0].index == n ? -1 : reduced[0].index;
}

__global__ void appendSelectedPoints(unsigned char* inCluster, int* members,
                                     int* sizes, const int* choices, int* progress,
                                     int seedCount, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < seedCount) {
        const int choice = choices[row];
        if (choice >= 0) {
            const int slot = sizes[row];
            inCluster[static_cast<size_t>(row) * n + choice] = 1;
            members[static_cast<size_t>(row) * n + slot] = choice;
            sizes[row] = slot + 1;
            atomicExch(progress, 1);
        }
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    const size_t pointBytes = static_cast<size_t>(n) * sizeof(Point);
    const size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    const size_t stateBytes = static_cast<size_t>(n) * n;
    Point* d_points = nullptr;
    double* d_distances = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    unsigned char* d_inCluster = nullptr;
    int* d_members = nullptr;
    int* d_sizes = nullptr;
    int* d_choices = nullptr;
    int* d_progress = nullptr;
    checkCuda(cudaMalloc(&d_points, pointBytes), "allocating points");
    checkCuda(cudaMalloc(&d_distances, matrixBytes), "allocating distance matrix");
    checkCuda(cudaMalloc(&d_clustered, n), "allocating clustered flags");
    checkCuda(cudaMalloc(&d_seeds, static_cast<size_t>(n) * sizeof(int)), "allocating seeds");
    checkCuda(cudaMalloc(&d_inCluster, stateBytes), "allocating candidate state");
    checkCuda(cudaMalloc(&d_members, static_cast<size_t>(n) * n * sizeof(int)), "allocating candidate members");
    checkCuda(cudaMalloc(&d_sizes, static_cast<size_t>(n) * sizeof(int)), "allocating candidate sizes");
    checkCuda(cudaMalloc(&d_choices, static_cast<size_t>(n) * sizeof(int)), "allocating choices");
    checkCuda(cudaMalloc(&d_progress, sizeof(int)), "allocating progress flag");
    checkCuda(cudaMemcpy(d_points, points.data(), pointBytes, cudaMemcpyHostToDevice), "copying points");
    checkCuda(cudaMemset(d_clustered, 0, n), "clearing clustered flags");
    const dim3 block2d(16, 16);
    buildDistanceMatrix<<<dim3((n + 15) / 16, (n + 15) / 16), block2d>>>(d_points, d_distances, n);
    checkCuda(cudaGetLastError(), "building distance matrix");

    std::vector<int> unclustered(n);
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    std::vector<int> candidateSizes(n);
    constexpr int threads = 256;
    while (!unclustered.empty()) {
        const int seedCount = static_cast<int>(unclustered.size());
        checkCuda(cudaMemcpy(d_seeds, unclustered.data(), static_cast<size_t>(seedCount) * sizeof(int), cudaMemcpyHostToDevice), "copying seeds");
        checkCuda(cudaMemset(d_inCluster, 0, static_cast<size_t>(seedCount) * n), "clearing candidate state");
        initializeCandidates<<<seedCount, threads>>>(d_seeds, d_inCluster, d_members, d_sizes, seedCount, n);
        checkCuda(cudaGetLastError(), "initializing candidates");
        // Candidate clusters usually converge long before n additions.  The
        // tiny host transfer here avoids launching empty iterations while all
        // distance work itself remains on the GPU.
        int progress = 1;
        while (progress != 0) {
            checkCuda(cudaMemset(d_progress, 0, sizeof(int)), "clearing progress flag");
            selectNextPoints<<<seedCount, threads>>>(d_clustered, d_inCluster, d_members, d_sizes,
                                                      d_distances, threshold, d_choices, seedCount, n);
            appendSelectedPoints<<<(seedCount + threads - 1) / threads, threads>>>(
                d_inCluster, d_members, d_sizes, d_choices, d_progress, seedCount, n);
            checkCuda(cudaMemcpy(&progress, d_progress, sizeof(int), cudaMemcpyDeviceToHost), "checking candidate progress");
        }
        checkCuda(cudaMemcpy(candidateSizes.data(), d_sizes, static_cast<size_t>(seedCount) * sizeof(int), cudaMemcpyDeviceToHost), "copying candidate sizes");
        int bestRow = 0;
        for (int row = 1; row < seedCount; ++row)
            if (candidateSizes[row] > candidateSizes[bestRow]) bestRow = row;
        Cluster cluster;
        cluster.seed_point = unclustered[bestRow];
        cluster.members.resize(candidateSizes[bestRow]);
        checkCuda(cudaMemcpy(cluster.members.data(), d_members + static_cast<size_t>(bestRow) * n,
                             cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost), "copying winning cluster");
        clusters.push_back(cluster);
        markClustered<<<(candidateSizes[bestRow] + threads - 1) / threads, threads>>>(
            d_clustered, d_members + static_cast<size_t>(bestRow) * n, candidateSizes[bestRow]);
        checkCuda(cudaGetLastError(), "marking cluster members");
        std::vector<unsigned char> selected(n, 0);
        for (int member : cluster.members) selected[member] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                           [&selected](int index) { return selected[index] != 0; }), unclustered.end());
    }
    checkCuda(cudaDeviceSynchronize(), "finishing clustering");
    cudaFree(d_progress); cudaFree(d_choices); cudaFree(d_sizes); cudaFree(d_members); cudaFree(d_inCluster);
    cudaFree(d_seeds); cudaFree(d_clustered); cudaFree(d_distances); cudaFree(d_points);
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

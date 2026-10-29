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

// Each block grows one candidate cluster.  The maximum distance to its
// members is updated incrementally as each member is added.
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__global__ void candidateKernel(const Point* points, const unsigned char* clustered,
                                const int* seeds, int seed_count, int n,
                                double threshold, double* workspace, int* counts) {
    int block = blockIdx.x;
    if (block >= seed_count) return;
    int tid = threadIdx.x;
    double* maxima = workspace + static_cast<size_t>(block) * n;
    int seed = seeds[block];
    for (int i = tid; i < n; i += blockDim.x)
        maxima[i] = (clustered[i] || i == seed) ? -1.0 : 0.0;
    __shared__ double best_dist[128];
    __shared__ int best_index[128];
    __shared__ int newest, size;
    if (tid == 0) { newest = seed; size = 1; }
    __syncthreads();
    while (true) {
        int member = newest;
        double local_best = threshold;
        int local_index = n;
        for (int i = tid; i < n; i += blockDim.x) {
            double old = maxima[i];
            if (old < 0.0) continue;
            double dx = points[i].x - points[member].x;
            double dy = points[i].y - points[member].y;
            double d = sqrt(dx * dx + dy * dy);
            double value = fmax(old, d);
            // Once a point reaches the threshold it can never become eligible.
            if (value >= threshold) { maxima[i] = -1.0; continue; }
            maxima[i] = value;
            if (value < local_best || (value == local_best && i < local_index)) {
                local_best = value; local_index = i;
            }
        }
        best_dist[tid] = local_best; best_index[tid] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride && (best_dist[tid + stride] < best_dist[tid] ||
                (best_dist[tid + stride] == best_dist[tid] && best_index[tid + stride] < best_index[tid]))) {
                best_dist[tid] = best_dist[tid + stride];
                best_index[tid] = best_index[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            newest = best_index[0];
            if (newest < n) { maxima[newest] = -1.0; ++size; }
        }
        __syncthreads();
        if (newest == n) break;
    }
    if (tid == 0) counts[block] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    int rank, ranks, local_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_free(&local);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    const int n = static_cast<int>(points.size());
    Point* d_points; unsigned char* d_clustered; int *d_seeds, *d_counts; double* workspace;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * n));
    CUDA_CHECK(cudaMalloc(&d_clustered, n));
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t capacity = std::min<size_t>(256, free_mem / (sizeof(double) * static_cast<size_t>(n) * 2));
    if (!capacity) { fprintf(stderr, "Insufficient GPU memory on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaMalloc(&workspace, capacity * static_cast<size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_seeds, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, capacity * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> remaining(n), local_seeds, counts(capacity);
    for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;
    while (!remaining.empty()) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        local_seeds.clear();
        for (size_t i = rank; i < remaining.size(); i += ranks) local_seeds.push_back(remaining[i]);
        struct { int count, seed; } local_best = {-1, n}, global_best;
        for (size_t offset = 0; offset < local_seeds.size(); offset += capacity) {
            int batch = static_cast<int>(std::min(capacity, local_seeds.size() - offset));
            CUDA_CHECK(cudaMemcpy(d_seeds, local_seeds.data() + offset, batch * sizeof(int), cudaMemcpyHostToDevice));
            candidateKernel<<<batch, 128>>>(d_points, d_clustered, d_seeds, batch, n, threshold, workspace, d_counts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, batch * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batch; ++i) {
                int seed = local_seeds[offset + i];
                if (counts[i] > local_best.count || (counts[i] == local_best.count && seed < local_best.seed))
                    local_best = {counts[i], seed};
            }
        }
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Rebuild the winning candidate once, using the same incremental rule.
        std::vector<int> members;
        if (rank == 0) {
            std::vector<double> maxima(n, 0.0);
            for (int i = 0; i < n; ++i) if (clustered[i]) maxima[i] = -1.0;
            int newest = global_best.seed;
            maxima[newest] = -1.0;
            members.push_back(newest);
            while (true) {
                Point p = points[newest];
                #pragma omp parallel for if(n >= 4096) schedule(static)
                for (int i = 0; i < n; ++i) {
                    if (maxima[i] < 0.0) continue;
                    double d = distance(points[i], p);
                    maxima[i] = std::max(maxima[i], d);
                    if (maxima[i] >= threshold) maxima[i] = -1.0;
                }
                double best = threshold;
                int next = -1;
                for (int i = 0; i < n; ++i)
                    if (maxima[i] >= 0.0 && maxima[i] < best) { best = maxima[i]; next = i; }
                if (next < 0) break;
                maxima[next] = -1.0;
                members.push_back(next);
                newest = next;
            }
            if (static_cast<int>(members.size()) != global_best.count) {
                fprintf(stderr, "GPU/CPU candidate mismatch\n"); MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        int count = global_best.count;
        if (rank != 0) members.resize(count);
        MPI_Bcast(members.data(), count, MPI_INT, 0, MPI_COMM_WORLD);
        for (int member : members) clustered[member] = 1;
        if (rank == 0) clusters.push_back({std::move(members), global_best.seed});
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
            [&clustered](int i) { return clustered[i] != 0; }), remaining.end());
    }
    CUDA_CHECK(cudaFree(workspace)); CUDA_CHECK(cudaFree(d_counts)); CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered)); CUDA_CHECK(cudaFree(d_points));
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

int run(int argc, char** argv, int rank) {
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
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
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
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    if (rank != 0) return 0;
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

int main(int argc, char** argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int result = run(argc, argv, rank);
    MPI_Finalize();
    return result;
}

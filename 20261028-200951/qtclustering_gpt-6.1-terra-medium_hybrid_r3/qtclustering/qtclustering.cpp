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
#include <math_constants.h>
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

// Each block grows one seed's candidate cluster.  The maximum distance to
// each candidate is updated incrementally as members are added.
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__global__ void growClusters(const Point* points, const unsigned char* clustered,
                             const int* seeds, int count, int n, double threshold,
                             double* maxima, int* sizes, int* output) {
    int job = blockIdx.x;
    if (job >= count) return;
    int tid = threadIdx.x;
    int seed = seeds[job];
    double* distances = maxima + static_cast<size_t>(job) * n;
    __shared__ double bestDist[256];
    __shared__ int bestIndex[256];
    __shared__ int member, size;
    if (tid == 0) { member = seed; size = 1; if (output) output[0] = seed; }
    for (int i = tid; i < n; i += blockDim.x)
        distances[i] = (clustered[i] || i == seed) ? CUDART_INF : 0.0;
    __syncthreads();
    while (true) {
        double best = CUDART_INF;
        int index = n;
        Point p = points[member];
        for (int i = tid; i < n; i += blockDim.x) {
            double old = distances[i];
            if (old == CUDART_INF) continue;
            double dx = points[i].x - p.x, dy = points[i].y - p.y;
            double d = sqrt(dx * dx + dy * dy);
            double maxd = d > old ? d : old;
            distances[i] = maxd;
            if (maxd < threshold && (maxd < best || (maxd == best && i < index))) {
                best = maxd; index = i;
            }
        }
        bestDist[tid] = best; bestIndex[tid] = index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (tid < stride) {
                double d = bestDist[tid + stride];
                int i = bestIndex[tid + stride];
                if (d < bestDist[tid] || (d == bestDist[tid] && i < bestIndex[tid])) {
                    bestDist[tid] = d; bestIndex[tid] = i;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            member = bestIndex[0];
            if (member < n) {
                distances[member] = CUDART_INF;
                if (output) output[size] = member;
                ++size;
            }
        }
        __syncthreads();
        if (member == n) break;
    }
    if (tid == 0) sizes[job] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks, localRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    const int n = static_cast<int>(points.size());
    // Bound scratch space independently of the number of seeds.
    const int batch = std::max(1, std::min(n, static_cast<int>((256ULL * 1024 * 1024) / (sizeof(double) * static_cast<size_t>(n)))));
    Point* dp; unsigned char* dc; int *ds, *dz, *dm; double* scratch;
    CUDA_CHECK(cudaMalloc(&dp, sizeof(Point) * n));
    CUDA_CHECK(cudaMalloc(&dc, n));
    CUDA_CHECK(cudaMalloc(&ds, sizeof(int) * batch));
    CUDA_CHECK(cudaMalloc(&dz, sizeof(int) * batch));
    CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&scratch, sizeof(double) * static_cast<size_t>(batch) * n));
    CUDA_CHECK(cudaMemcpy(dp, points.data(), sizeof(Point) * n, cudaMemcpyHostToDevice));
    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> seeds, sizes(batch), members(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        CUDA_CHECK(cudaMemcpy(dc, clustered.data(), n, cudaMemcpyHostToDevice));
        // Parallel preparation keeps the original ascending seed order.
        std::vector<int> eligible(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) eligible[i] = (!clustered[i] && i % ranks == rank) ? i : -1;
        seeds.clear();
        for (int i : eligible) if (i >= 0) seeds.push_back(i);
        int localBest[2] = {-1, n};
        for (size_t start = 0; start < seeds.size(); start += batch) {
            int count = static_cast<int>(std::min(static_cast<size_t>(batch), seeds.size() - start));
            CUDA_CHECK(cudaMemcpy(ds, seeds.data() + start, sizeof(int) * count, cudaMemcpyHostToDevice));
            growClusters<<<count, 256>>>(dp, dc, ds, count, n, threshold, scratch, dz, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), dz, sizeof(int) * count, cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                int seed = seeds[start + i];
                if (sizes[i] > localBest[0] || (sizes[i] == localBest[0] && seed < localBest[1])) {
                    localBest[0] = sizes[i]; localBest[1] = seed;
                }
            }
        }
        int best[2];
        MPI_Allreduce(localBest, best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        int owner = best[1] % ranks;
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(ds, &best[1], sizeof(int), cudaMemcpyHostToDevice));
            growClusters<<<1, 256>>>(dp, dc, ds, 1, n, threshold, scratch, dz, dm);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(members.data(), dm, sizeof(int) * best[0], cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), best[0], MPI_INT, owner, MPI_COMM_WORLD);
        Cluster c;
        c.seed_point = best[1];
        c.members.assign(members.begin(), members.begin() + best[0]);
        for (int i : c.members) clustered[i] = 1;
        remaining -= best[0];
        clusters.push_back(std::move(c));
    }
    CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(dm)); CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(dp));
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
    struct MpiGuard { ~MpiGuard() { MPI_Finalize(); } } guard;
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
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
    
    if (rank != 0) return 0;
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

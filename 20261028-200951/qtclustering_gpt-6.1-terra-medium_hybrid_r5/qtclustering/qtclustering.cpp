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

// Reconstruct the winning cluster using the same incremental diameter update
// as the GPU, preserving the original candidate ordering for equal distances.
int generateCandidateCluster(const int seed_point,
                             const std::vector<bool>& clustered,
                             const std::vector<Point>& points,
                             const double threshold, const int point_count,
                             std::vector<int>* cluster_members) {
    std::vector<double> maxima(point_count, 0.0);
    std::vector<unsigned char> excluded(point_count);
    for (int i = 0; i < point_count; ++i) excluded[i] = clustered[i];
    std::vector<int> members(1, seed_point);
    excluded[seed_point] = 1;
    int latest = seed_point;
    while (true) {
        int best = -1;
        double best_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < point_count; ++i) {
            if (excluded[i]) continue;
            maxima[i] = std::max(maxima[i], distance(points[i], points[latest]));
            if (!(maxima[i] < threshold)) { excluded[i] = 1; continue; }
            if (maxima[i] < best_distance) { best_distance = maxima[i]; best = i; }
        }
        if (best < 0) break;
        excluded[best] = 1;
        members.push_back(best);
        latest = best;
    }
    *cluster_members = std::move(members);
    return static_cast<int>(cluster_members->size());
}

// Each block grows one seed's cluster.  A candidate's diameter only changes
// when a new member is added, so retain its maximum instead of rescanning all
// previous members on every iteration.
__global__ void candidateCounts(const Point* points, const unsigned char* clustered,
                                const int* seeds, int seed_offset, int batch,
                                int n, double threshold, double* maxima, int* counts) {
    const int row = blockIdx.x;
    if (row >= batch) return;
    const int seed = seeds[seed_offset + row];
    double* values = maxima + static_cast<size_t>(row) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int member, count;
    if (threadIdx.x == 0) { member = seed; count = 1; }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        values[i] = (clustered[i] || i == seed) ? -1.0 : 0.0;
    __syncthreads();
    while (true) {
        double local_best = 1.7976931348623157e308;
        int local_index = -1;
        const Point p = points[member];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            double v = values[i];
            if (v < 0.0) continue;
            const double dx = points[i].x - p.x;
            const double dy = points[i].y - p.y;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > v) v = d;
            // Once a point exceeds the threshold it can never become eligible.
            if (!(v < threshold)) { values[i] = -1.0; continue; }
            values[i] = v;
            if (v < local_best || (v == local_best && i < local_index)) {
                local_best = v; local_index = i;
            }
        }
        best_distance[threadIdx.x] = local_best;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                const int other = best_index[threadIdx.x + stride];
                const double d = best_distance[threadIdx.x + stride];
                if (other >= 0 && (best_index[threadIdx.x] < 0 ||
                    d < best_distance[threadIdx.x] ||
                    (d == best_distance[threadIdx.x] && other < best_index[threadIdx.x]))) {
                    best_distance[threadIdx.x] = d;
                    best_index[threadIdx.x] = other;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            member = best_index[0];
            if (member >= 0) { values[member] = -1.0; ++count; }
        }
        __syncthreads();
        if (member < 0) break;
    }
    if (threadIdx.x == 0) counts[seed_offset + row] = count;
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    std::vector<bool> clustered(n, false);
    std::vector<unsigned char> flags(n, 0);
    std::vector<int> remaining(n);
    for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;
    Point* d_points; unsigned char* d_flags; int *d_seeds, *d_counts;
    double* d_maxima;
    // Bound scratch space independently of the number of seeds.
    const int batch_capacity = static_cast<int>(std::min<size_t>(n, std::max<size_t>(1, (128ULL << 20) / (sizeof(double) * static_cast<size_t>(n)))));
    cudaCheck(cudaMalloc(&d_points, sizeof(Point) * n), "allocate points");
    cudaCheck(cudaMalloc(&d_flags, n), "allocate flags");
    cudaCheck(cudaMalloc(&d_seeds, sizeof(int) * n), "allocate seeds");
    cudaCheck(cudaMalloc(&d_counts, sizeof(int) * n), "allocate counts");
    cudaCheck(cudaMalloc(&d_maxima, sizeof(double) * static_cast<size_t>(n) * batch_capacity), "allocate scratch");
    cudaCheck(cudaMemcpy(d_points, points.data(), sizeof(Point) * n, cudaMemcpyHostToDevice), "copy points");
    while (!remaining.empty()) {
        std::vector<int> seeds;
        for (size_t i = rank; i < remaining.size(); i += ranks) seeds.push_back(remaining[i]);
        std::vector<int> counts(seeds.size());
        if (!seeds.empty()) {
            cudaCheck(cudaMemcpy(d_flags, flags.data(), n, cudaMemcpyHostToDevice), "copy flags");
            cudaCheck(cudaMemcpy(d_seeds, seeds.data(), sizeof(int) * seeds.size(), cudaMemcpyHostToDevice), "copy seeds");
            for (int offset = 0; offset < static_cast<int>(seeds.size()); offset += batch_capacity) {
                const int batch = std::min(batch_capacity, static_cast<int>(seeds.size()) - offset);
                candidateCounts<<<batch, 256>>>(d_points, d_flags, d_seeds, offset, batch, n, threshold, d_maxima, d_counts);
                cudaCheck(cudaGetLastError(), "launch candidate kernel");
            }
            cudaCheck(cudaMemcpy(counts.data(), d_counts, sizeof(int) * counts.size(), cudaMemcpyDeviceToHost), "copy counts");
        }
        // OpenMP evaluates the local winner; the critical section preserves tie order.
        int local_count = -1, local_seed = n;
        #pragma omp parallel
        {
            int thread_count = -1, thread_seed = n;
            #pragma omp for nowait
            for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
                if (counts[i] > thread_count || (counts[i] == thread_count && seeds[i] < thread_seed)) {
                    thread_count = counts[i]; thread_seed = seeds[i];
                }
            }
            #pragma omp critical
            if (thread_count > local_count || (thread_count == local_count && thread_seed < local_seed)) {
                local_count = thread_count; local_seed = thread_seed;
            }
        }
        struct { int count, seed; } local = {local_count, local_seed}, global;
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        std::vector<int> members;
        const int actual = generateCandidateCluster(global.seed, clustered, points, threshold, n, &members);
        if (actual != global.count) {
            fprintf(stderr, "GPU and CPU candidate sizes disagree for seed %d (%d vs %d)\n", global.seed, global.count, actual);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        clusters.push_back({members, global.seed});
        for (int member : members) { clustered[member] = true; flags[member] = 1; }
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                        [&flags](int i) { return flags[i] != 0; }), remaining.end());
    }
    cudaFree(d_maxima); cudaFree(d_counts); cudaFree(d_seeds); cudaFree(d_flags); cudaFree(d_points);
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
    struct MpiFinalizer { ~MpiFinalizer() { MPI_Finalize(); } } mpi_finalizer;
    int rank, local_rank, device_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    cudaCheck(cudaGetDeviceCount(&device_count), "query devices");
    if (device_count == 0) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(local_rank % device_count), "select device");
    if (rank != 0 && !freopen("/dev/null", "w", stdout)) MPI_Abort(MPI_COMM_WORLD, 1);
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
    long local_time_ms = cluster_time.count();
    long max_time_ms;
    MPI_Allreduce(&local_time_ms, &max_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(max_time_ms);
    
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

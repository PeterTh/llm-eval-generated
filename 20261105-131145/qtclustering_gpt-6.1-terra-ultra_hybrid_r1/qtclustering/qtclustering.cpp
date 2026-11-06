// QT Clustering Benchmark - MPI, OpenMP and CUDA version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
        if (N <= 30 && group_cnt == 0) group_cnt = 1;
        
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

// Each block builds one seed's candidate cluster.  The diameter from a
// candidate to the growing cluster is updated using only the newest member.
// This is exactly the maximum over all members, without repeatedly scanning
// the whole cluster. Equal distances are resolved by the original point order.
__global__ void buildCandidates(const Point* points, const unsigned char* clustered,
                                const int* seeds, int count, int n, double threshold,
                                double* distances, int* cardinalities) {
    const int row_index = blockIdx.x;
    if (row_index >= count) return;
    const int tid = threadIdx.x;
    const int seed = seeds[row_index];
    double* row = distances + static_cast<size_t>(row_index) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int newest;
    __shared__ int size;
    if (tid == 0) { newest = seed; size = 1; }
    for (int i = tid; i < n; i += blockDim.x)
        row[i] = clustered[i] ? DBL_MAX : (i == seed ? -1.0 : 0.0);
    __syncthreads();
    while (true) {
        const Point last = points[newest];
        double local_distance = DBL_MAX;
        int local_index = INT_MAX;
        for (int i = tid; i < n; i += blockDim.x) {
            double d = row[i];
            if (d < 0.0 || d == DBL_MAX || d >= threshold) continue;
            const double dx = points[i].x - last.x;
            const double dy = points[i].y - last.y;
            // GCC's native x86 build of the original expression contracts the
            // first product into an FMA when the host target supports it.
#if defined(__FMA__) && defined(__x86_64__)
            const double distance = sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
            const double distance = sqrt(dx * dx + dy * dy);
#endif
            if (d < distance) d = distance;
            row[i] = d;
            if (d < threshold && (d < local_distance ||
                                  (d == local_distance && i < local_index))) {
                local_distance = d;
                local_index = i;
            }
        }
        best_distance[tid] = local_distance;
        best_index[tid] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                const double other = best_distance[tid + stride];
                const int index = best_index[tid + stride];
                if (other < best_distance[tid] ||
                    (other == best_distance[tid] && index < best_index[tid])) {
                    best_distance[tid] = other;
                    best_index[tid] = index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            newest = best_index[0];
            if (newest != INT_MAX) { ++size; row[newest] = -static_cast<double>(size); }
        }
        __syncthreads();
        if (newest == INT_MAX) break;
    }
    if (tid == 0) cardinalities[row_index] = size;
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    checkCuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    checkCuda(cudaSetDevice(local_rank % device_count), "select CUDA device");

    const int n = static_cast<int>(points.size());
    Point* device_points;
    unsigned char* device_clustered;
    int *device_seeds, *device_counts;
    double* device_distances;
    checkCuda(cudaMalloc(&device_points, sizeof(Point) * static_cast<size_t>(n)), "allocate points");
    checkCuda(cudaMalloc(&device_clustered, static_cast<size_t>(n)), "allocate clustered flags");
    checkCuda(cudaMemcpy(device_points, points.data(), sizeof(Point) * static_cast<size_t>(n),
                         cudaMemcpyHostToDevice), "copy points");
    size_t free_memory, total_memory;
    checkCuda(cudaMemGetInfo(&free_memory, &total_memory), "query CUDA memory");
    // Bound the workspace so large inputs and multiple local ranks remain usable.
    const size_t workspace_limit = std::min(free_memory / 4, size_t(512) * 1024 * 1024);
    const size_t row_bytes = sizeof(double) * static_cast<size_t>(n);
    const int batch_capacity = static_cast<int>(std::min(static_cast<size_t>(n),
                                               std::max(size_t(1), workspace_limit / row_bytes)));
    checkCuda(cudaMalloc(&device_distances, row_bytes * batch_capacity), "allocate candidate workspace");
    checkCuda(cudaMalloc(&device_seeds, sizeof(int) * static_cast<size_t>(batch_capacity)), "allocate seeds");
    checkCuda(cudaMalloc(&device_counts, sizeof(int) * static_cast<size_t>(batch_capacity)), "allocate counts");

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> winning_order(n, 0);
    std::vector<int> unclustered(n);
    #pragma omp parallel for schedule(static) num_threads(std::min(8, omp_get_max_threads()))
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    std::vector<double> best_row(n);
    std::vector<int> local_seeds, counts(batch_capacity);
    while (!unclustered.empty()) {
        checkCuda(cudaMemcpy(device_clustered, clustered.data(), static_cast<size_t>(n),
                             cudaMemcpyHostToDevice), "copy clustered flags");
        local_seeds.clear();
        for (size_t i = static_cast<size_t>(rank); i < unclustered.size(); i += ranks)
            local_seeds.push_back(unclustered[i]);
        int local_best_size = -1, local_best_seed = INT_MAX;
        for (size_t start = 0; start < local_seeds.size(); start += batch_capacity) {
            const int batch = static_cast<int>(std::min(static_cast<size_t>(batch_capacity), local_seeds.size() - start));
            checkCuda(cudaMemcpy(device_seeds, local_seeds.data() + start, sizeof(int) * batch,
                                 cudaMemcpyHostToDevice), "copy seeds");
            buildCandidates<<<batch, 256>>>(device_points, device_clustered, device_seeds,
                                             batch, n, threshold, device_distances, device_counts);
            checkCuda(cudaGetLastError(), "launch candidate kernel");
            checkCuda(cudaMemcpy(counts.data(), device_counts, sizeof(int) * batch,
                                 cudaMemcpyDeviceToHost), "copy candidate counts");
            int batch_best = -1;
            for (int j = 0; j < batch; ++j) {
                const int seed = local_seeds[start + j];
                if (counts[j] > local_best_size ||
                    (counts[j] == local_best_size && seed < local_best_seed)) {
                    local_best_size = counts[j];
                    local_best_seed = seed;
                    batch_best = j;
                }
            }
            if (batch_best >= 0)
                checkCuda(cudaMemcpy(best_row.data(), device_distances + static_cast<size_t>(batch_best) * n,
                                     row_bytes, cudaMemcpyDeviceToHost), "copy winning candidate");
        }
        int local_pair[2] = {local_best_size, local_best_seed};
        int global_pair[2];
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // Seeds are assigned cyclically in the sorted unclustered list.
        const auto winner = std::lower_bound(unclustered.begin(), unclustered.end(), global_pair[1]);
        const int owner = static_cast<int>((winner - unclustered.begin()) % ranks);
        if (rank == owner) {
            #pragma omp parallel for if(n >= 4096) schedule(static) num_threads(std::min(8, omp_get_max_threads()))
            for (int i = 0; i < n; ++i)
                winning_order[i] = best_row[i] < 0.0 ? static_cast<int>(-best_row[i]) : 0;
        }
        MPI_Bcast(winning_order.data(), n, MPI_INT, owner, MPI_COMM_WORLD);
        Cluster cluster;
        cluster.seed_point = global_pair[1];
        cluster.members.resize(global_pair[0]);
        for (int i : unclustered) {
            if (winning_order[i]) {
                cluster.members[winning_order[i] - 1] = i;
                clustered[i] = 1;
            }
        }
        clusters.push_back(std::move(cluster));
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int i) { return clustered[i] != 0; }), unclustered.end());
    }
    checkCuda(cudaFree(device_distances), "free candidate workspace");
    checkCuda(cudaFree(device_seeds), "free seeds");
    checkCuda(cudaFree(device_counts), "free counts");
    checkCuda(cudaFree(device_clustered), "free clustered flags");
    checkCuda(cudaFree(device_points), "free points");
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                              num_points, threshold);
        MPI_Finalize();
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
    if (rank == 0) generateSyntheticData(points, num_points);
    // Use byte chunks to avoid overflowing MPI's int count for large inputs.
    const size_t point_bytes = points.size() * sizeof(Point);
    for (size_t offset = 0; offset < point_bytes;) {
        const int count = static_cast<int>(std::min(point_bytes - offset, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(reinterpret_cast<unsigned char*>(points.data()) + offset, count, MPI_BYTE, 0, MPI_COMM_WORLD);
        offset += static_cast<size_t>(count);
    }
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    cluster_time = std::chrono::milliseconds(max_cluster_time);
    
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

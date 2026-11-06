// QT Clustering Benchmark - MPI, OpenMP and CUDA implementation
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
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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
        // For very small inputs the original formula otherwise produces zero
        // indefinitely (or almost indefinitely when N == 30).
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

// A block grows one candidate cluster.  Each thread owns a subset of the
// remaining points and keeps their maximum distance to the current members.
// This incremental maximum is equivalent to rescanning every member after
// each insertion, but avoids repeating all of the old distance calculations.
__global__ void candidateClusters(const Point* points, const int* active,
                                  int nactive, const int* seeds, int nseeds,
                                  double threshold, double* distances,
                                  int* sizes, int* winning_members) {
    const int block = blockIdx.x;
    if (block >= nseeds) return;
    double* maxdist = distances + static_cast<size_t>(block) * nactive;
    const int seed = seeds[block];
    for (int i = threadIdx.x; i < nactive; i += blockDim.x)
        maxdist[i] = active[i] == seed ? -1.0 : 0.0;

    __shared__ double warp_values[32];
    __shared__ int warp_indices[32];
    __shared__ int selected;
    __shared__ int last;
    __shared__ int count;
    if (threadIdx.x == 0) {
        last = seed;
        count = 1;
        if (winning_members) winning_members[0] = seed;
    }
    __syncthreads();

    while (true) {
        const Point member = points[last];
        double best = DBL_MAX;
        int best_index = INT_MAX;
        for (int i = threadIdx.x; i < nactive; i += blockDim.x) {
            double d = maxdist[i];
            if (d < 0.0) continue;
            const Point p = points[active[i]];
            const double dx = p.x - member.x;
            const double dy = p.y - member.y;
            const double dist = sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
            // std::max in the original leaves d unchanged for a NaN distance.
            if (dist > d) d = dist;
            if (!(d < threshold && d < DBL_MAX)) {
                // A maximum distance cannot decrease as the cluster grows.
                maxdist[i] = -1.0;
            } else {
                maxdist[i] = d;
                if (d < best || (d == best && active[i] < best_index)) {
                    best = d;
                    best_index = active[i];
                }
            }
        }
        // Reduce (diameter, point index), retaining the sequential tie order.
        for (int offset = 16; offset > 0; offset /= 2) {
            double other = __shfl_down_sync(0xffffffffu, best, offset);
            int index = __shfl_down_sync(0xffffffffu, best_index, offset);
            if (other < best || (other == best && index < best_index)) {
                best = other;
                best_index = index;
            }
        }
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warp_values[warp] = best;
            warp_indices[warp] = best_index;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < blockDim.x / 32 ? warp_values[lane] : DBL_MAX;
            best_index = lane < blockDim.x / 32 ? warp_indices[lane] : INT_MAX;
            for (int offset = 16; offset > 0; offset /= 2) {
                double other = __shfl_down_sync(0xffffffffu, best, offset);
                int index = __shfl_down_sync(0xffffffffu, best_index, offset);
                if (other < best || (other == best && index < best_index)) {
                    best = other;
                    best_index = index;
                }
            }
            if (lane == 0) selected = best_index;
        }
        __syncthreads();
        if (selected == INT_MAX) break;
        for (int i = threadIdx.x; i < nactive; i += blockDim.x)
            if (active[i] == selected) maxdist[i] = -1.0;
        if (threadIdx.x == 0) {
            last = selected;
            if (winning_members) winning_members[count] = selected;
            ++count;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) sizes[block] = count;
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) checkCuda((call), #call)

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<int> active(N);
    #pragma omp parallel for num_threads(std::min(4, omp_get_max_threads()))
    for (int i = 0; i < N; ++i) active[i] = i;
    std::vector<Cluster> clusters;

    Point* d_points = nullptr;
    int *d_active = nullptr, *d_sizes = nullptr, *d_members = nullptr;
    double* d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_active, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * static_cast<size_t>(N), cudaMemcpyHostToDevice));
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Limit the scratch space to avoid exhausting a device shared by MPI ranks.
    const size_t budget = std::min<size_t>(256u * 1024u * 1024u, free_bytes / 4);
    const size_t bytes_per_seed = sizeof(double) * static_cast<size_t>(N);
    const int batch_capacity = static_cast<int>(std::min<size_t>(N,
        std::max<size_t>(1, budget / bytes_per_seed)));
    CUDA_CHECK(cudaMalloc(&d_distances, bytes_per_seed * batch_capacity));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizeof(int) * static_cast<size_t>(batch_capacity)));

    while (!active.empty()) {
        const int nactive = static_cast<int>(active.size());
        // Contiguous intervals preserve the original seed order and make the
        // winning rank independently identifiable by all ranks.
        const int begin = static_cast<int>((static_cast<long long>(nactive) * rank) / ranks);
        const int end = static_cast<int>((static_cast<long long>(nactive) * (rank + 1)) / ranks);
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), sizeof(int) * static_cast<size_t>(nactive), cudaMemcpyHostToDevice));
        std::vector<int> sizes(end - begin);
        for (int start = begin; start < end; start += batch_capacity) {
            const int batch = std::min(batch_capacity, end - start);
            candidateClusters<<<batch, 256>>>(d_points, d_active, nactive,
                d_active + start, batch, threshold, d_distances, d_sizes, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data() + (start - begin), d_sizes,
                                  sizeof(int) * static_cast<size_t>(batch), cudaMemcpyDeviceToHost));
        }

        int local_size = -1, local_seed = INT_MAX;
        #pragma omp parallel if(end - begin >= 8192) num_threads(std::min(4, omp_get_max_threads()))
        {
            int thread_size = -1, thread_seed = INT_MAX;
            #pragma omp for nowait
            for (int i = begin; i < end; ++i) {
                const int size = sizes[i - begin], seed = active[i];
                if (size > thread_size || (size == thread_size && seed < thread_seed)) {
                    thread_size = size;
                    thread_seed = seed;
                }
            }
            #pragma omp critical
            {
                if (thread_size > local_size || (thread_size == local_size && thread_seed < local_seed)) {
                    local_size = thread_size;
                    local_seed = thread_seed;
                }
            }
        }
        struct { int size, seed; } local = {local_size, local_seed}, best;
        MPI_Allreduce(&local, &best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (best.size == 1) {
            // If no seed can admit another point, all remaining clusters are
            // singletons, in the same order as the original seed scan.
            for (int seed : active) clusters.push_back({{seed}, seed});
            break;
        }
        const int position = static_cast<int>(std::lower_bound(active.begin(), active.end(), best.seed) - active.begin());
        // Invert floor(nactive * rank / ranks), including empty rank intervals.
        const int owner = static_cast<int>(((static_cast<long long>(position) + 1) * ranks - 1) / nactive);
        std::vector<int> members(best.size);
        if (rank == owner) {
            candidateClusters<<<1, 256>>>(d_points, d_active, nactive,
                d_active + position, 1, threshold, d_distances, d_sizes, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(members.data(), d_members,
                                  sizeof(int) * members.size(), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), best.size, MPI_INT, owner, MPI_COMM_WORLD);
        clusters.push_back({members, best.seed});
        std::vector<unsigned char> removed(N, 0);
        for (int member : members) removed[member] = 1;
        active.erase(std::remove_if(active.begin(), active.end(),
                         [&removed](int i) { return removed[i] != 0; }), active.end());
    }
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_points));
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct MpiFinalizer { ~MpiFinalizer() { MPI_Finalize(); } } mpi_finalizer;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        return 1;
    }
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
    
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    if (rank == 0) {
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Broadcast as doubles, avoiding assumptions about struct padding.
    std::vector<double> coordinates(static_cast<size_t>(num_points) * 2);
    if (rank == 0) {
        #pragma omp parallel for if(num_points >= 8192) num_threads(std::min(4, omp_get_max_threads()))
        for (int i = 0; i < num_points; ++i) {
            coordinates[2 * static_cast<size_t>(i)] = points[i].x;
            coordinates[2 * static_cast<size_t>(i) + 1] = points[i].y;
        }
    }
    // Chunk the broadcast so its MPI count always fits in int.
    for (size_t offset = 0; offset < coordinates.size();) {
        int count = static_cast<int>(std::min<size_t>(INT_MAX, coordinates.size() - offset));
        MPI_Bcast(coordinates.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    if (rank != 0) {
        #pragma omp parallel for if(num_points >= 8192) num_threads(std::min(4, omp_get_max_threads()))
        for (int i = 0; i < num_points; ++i) {
            points[i] = {coordinates[2 * static_cast<size_t>(i)], coordinates[2 * static_cast<size_t>(i) + 1]};
        }
    }
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

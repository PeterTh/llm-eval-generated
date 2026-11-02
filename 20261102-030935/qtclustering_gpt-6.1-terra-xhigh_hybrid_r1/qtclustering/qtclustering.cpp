// QT Clustering Benchmark - MPI, OpenMP, and CUDA implementation
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
#include <cstdint>
#include <climits>
#include <mpi.h>
#include <omp.h>
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

// Each block builds one seed's candidate cluster.  The cached maximum for
// each point is updated only for the newly added member, which is equivalent
// to rescanning all members on every iteration.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__global__ void makeDistances(const Point* points, double* distances, int n) {
    size_t total = static_cast<size_t>(n) * n;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        int i = static_cast<int>(k / n), j = static_cast<int>(k % n);
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        distances[k] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void evaluateSeeds(const Point* points, const double* distances,
                              const unsigned char* clustered, const int* seeds,
                              int n, double threshold, double* maxima,
                              int* sizes, int* members) {
    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* row = maxima + static_cast<size_t>(blockIdx.x) * n;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    __shared__ int last;
    __shared__ int count;
    if (tid == 0) {
        last = seed;
        count = 1;
        if (members) members[0] = seed;
    }
    for (int i = tid; i < n; i += blockDim.x)
        row[i] = (clustered[i] || i == seed) ? INFINITY : 0.0;
    __syncthreads();
    while (true) {
        double local_best = threshold;
        int local_index = INT_MAX;
        int current = last;
        for (int i = tid; i < n; i += blockDim.x) {
            double max_dist = row[i];
            if (i == current) { max_dist = INFINITY; row[i] = max_dist; }
            if (max_dist < threshold) {
                double dist;
                if (distances) {
                    dist = distances[static_cast<size_t>(current) * n + i];
                } else {
                    double dx = points[i].x - points[current].x;
                    double dy = points[i].y - points[current].y;
                    dist = sqrt(dx * dx + dy * dy);
                }
                if (dist > max_dist) max_dist = dist;
                row[i] = max_dist;
                if (max_dist < local_best ||
                    (max_dist == local_best && max_dist < threshold && i < local_index)) {
                    local_best = max_dist;
                    local_index = i;
                }
            }
        }
        best_distance[tid] = local_best;
        best_index[tid] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (tid < stride) {
                double other = best_distance[tid + stride];
                int other_index = best_index[tid + stride];
                if (other < best_distance[tid] ||
                    (other == best_distance[tid] && other_index < best_index[tid])) {
                    best_distance[tid] = other;
                    best_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0 && best_index[0] != INT_MAX) {
            last = best_index[0];
            if (members) members[count] = last;
            ++count;
        }
        __syncthreads();
        if (best_index[0] == INT_MAX) break;
    }
    if (tid == 0) sizes[blockIdx.x] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks, int local_rank) {
    const int n = static_cast<int>(points.size());
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local_rank % device_count));
    Point* d_points;
    unsigned char* d_clustered;
    int *d_seeds, *d_sizes, *d_members;
    double *d_distances = nullptr, *d_maxima;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    // Keep enough space for a useful number of candidate blocks, and compute
    // distances on demand when a dense matrix would consume too much memory.
    if (matrix_bytes <= free_bytes / 2 && cudaMalloc(&d_distances, matrix_bytes) == cudaSuccess) {
        int blocks = static_cast<int>(std::min<size_t>((static_cast<size_t>(n) * n + 255) / 256, 65535));
        makeDistances<<<blocks, 256>>>(d_points, d_distances, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        d_distances = nullptr;
        cudaGetLastError();
    }
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    size_t capacity = std::min<size_t>(static_cast<size_t>(n),
        std::min<size_t>(static_cast<size_t>(prop.multiProcessorCount) * 8,
                         free_bytes / 2 / (row_bytes + 2 * sizeof(int))));
    if (!capacity) {
        fprintf(stderr, "Insufficient CUDA memory on MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaMalloc(&d_maxima, capacity * row_bytes));
    CUDA_CHECK(cudaMalloc(&d_seeds, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, capacity * sizeof(int)));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> unclustered(n);
    #pragma omp parallel for num_threads(std::min(omp_get_max_threads(), std::max(1, n / 256)))
    for (int i = 0; i < n; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;
    std::vector<int> local_seeds, sizes(capacity), chosen(n);
    while (!unclustered.empty()) {
        const size_t local_count = (unclustered.size() + ranks - 1 - rank) / ranks;
        local_seeds.resize(local_count);
        #pragma omp parallel for if(local_count > 4096)
        for (size_t i = 0; i < local_count; ++i)
            local_seeds[i] = unclustered[i * ranks + rank];
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        int local_best[2] = {-1, INT_MAX};
        for (size_t offset = 0; offset < local_count; offset += capacity) {
            int count = static_cast<int>(std::min(capacity, local_count - offset));
            CUDA_CHECK(cudaMemcpy(d_seeds, local_seeds.data() + offset, count * sizeof(int), cudaMemcpyHostToDevice));
            evaluateSeeds<<<count, 256>>>(d_points, d_distances, d_clustered, d_seeds,
                                        n, threshold, d_maxima, d_sizes, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, count * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                int seed = local_seeds[offset + i];
                if (sizes[i] > local_best[0] || (sizes[i] == local_best[0] && seed < local_best[1])) {
                    local_best[0] = sizes[i];
                    local_best[1] = seed;
                }
            }
        }
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        int owner = 0;
        // The list is sorted and distributed round-robin by position.
        auto pos = std::lower_bound(unclustered.begin(), unclustered.end(), global_best[1]);
        owner = static_cast<int>((pos - unclustered.begin()) % ranks);
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(d_seeds, &global_best[1], sizeof(int), cudaMemcpyHostToDevice));
            evaluateSeeds<<<1, 256>>>(d_points, d_distances, d_clustered, d_seeds,
                                     n, threshold, d_maxima, d_sizes, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(chosen.data(), d_members, global_best[0] * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(chosen.data(), global_best[0], MPI_INT, owner, MPI_COMM_WORLD);
        Cluster cluster;
        cluster.seed_point = global_best[1];
        cluster.members.assign(chosen.begin(), chosen.begin() + global_best[0]);
        clusters.push_back(std::move(cluster));
        #pragma omp parallel for if(global_best[0] > 1024)
        for (int i = 0; i < global_best[0]; ++i) clustered[chosen[i]] = 1;
        // Filter contiguous slices independently, then join them in order so
        // the original seed ordering and tie-breaking are preserved.
        if (unclustered.size() > 4096) {
            const int filter_threads = std::min<int>(omp_get_max_threads(),
                                                       (unclustered.size() + 4095) / 4096);
            std::vector<std::vector<int>> slices(filter_threads);
            #pragma omp parallel num_threads(filter_threads)
            {
                const int tid = omp_get_thread_num();
                const int threads = omp_get_num_threads();
                size_t begin = unclustered.size() * tid / threads;
                size_t end = unclustered.size() * (tid + 1) / threads;
                auto& slice = slices[tid];
                slice.reserve(end - begin);
                for (size_t i = begin; i < end; ++i) {
                    int point = unclustered[i];
                    if (!clustered[point]) slice.push_back(point);
                }
            }
            size_t output = 0;
            for (const auto& slice : slices) {
                std::copy(slice.begin(), slice.end(), unclustered.begin() + output);
                output += slice.size();
            }
            unclustered.resize(output);
        } else {
            unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                             [&clustered](int i) { return clustered[i] != 0; }), unclustered.end());
        }
    }
    CUDA_CHECK(cudaFree(d_maxima));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_clustered));
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_points));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Diameters are independent; retain ordered reporting below.
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic) if(points.size() > 4096 && clusters.size() > 16)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                             points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
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
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks, local_rank);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    const long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) { MPI_Finalize(); return 0; }
    cluster_time = std::chrono::milliseconds(max_cluster_time_ms);
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

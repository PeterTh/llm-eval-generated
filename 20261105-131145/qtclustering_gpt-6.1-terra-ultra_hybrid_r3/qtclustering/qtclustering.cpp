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
#include <cstdint>
#include <cuda_runtime.h>
#include <mpi.h>
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

// Each block builds one candidate cluster. Its threads cooperate on the
// candidate scan and maintain the maximum distance to the current members.
// The latter changes incrementally, avoiding repeated scans of all members.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

__device__ inline double deviceDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
#if defined(__FMA__)
    return sqrt(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
#endif
}

__global__ void makeDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int a = static_cast<int>(i / n);
        const int b = static_cast<int>(i % n);
        distances[i] = deviceDistance(points[a], points[b]);
    }
}

static constexpr int THREADS = 256;
__global__ void buildCandidates(const Point* points, const double* distances,
                                const unsigned char* clustered, const int* seeds,
                                int n, double threshold, double* scratch,
                                int* sizes, int* result_members) {
    __shared__ double best_dist[THREADS];
    __shared__ int best_index[THREADS];
    __shared__ int selected;
    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    double* maxima = scratch + static_cast<size_t>(blockIdx.x) * n;
    for (int i = tid; i < n; i += blockDim.x) {
        double d = distances ? distances[static_cast<size_t>(seed) * n + i] :
                               deviceDistance(points[seed], points[i]);
        // Match std::max(0.0, d), including for a NaN distance.
        d = d > 0.0 ? d : 0.0;
        maxima[i] = (!clustered[i] && i != seed && d < threshold) ? d : -1.0;
    }
    if (tid == 0 && result_members) result_members[0] = seed;
    __syncthreads();
    int count = 1;
    while (count < n) {
        double best = threshold;
        int index = n;
        for (int i = tid; i < n; i += blockDim.x) {
            const double d = maxima[i];
            if (d >= 0.0 && (d < best || (d == best && index != n && i < index))) {
                best = d;
                index = i;
            }
        }
        best_dist[tid] = best;
        best_index[tid] = index;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int other = best_index[tid + stride];
                if (other < n && (best_index[tid] == n ||
                    best_dist[tid + stride] < best_dist[tid] ||
                    (best_dist[tid + stride] == best_dist[tid] && other < best_index[tid]))) {
                    best_dist[tid] = best_dist[tid + stride];
                    best_index[tid] = other;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected = best_index[0];
            if (selected < n && result_members) result_members[count] = selected;
        }
        __syncthreads();
        if (selected == n) break;
        for (int i = tid; i < n; i += blockDim.x) {
            if (i == selected) {
                maxima[i] = -1.0;
            } else if (maxima[i] >= 0.0) {
                const double d = distances ? distances[static_cast<size_t>(selected) * n + i] :
                                             deviceDistance(points[selected], points[i]);
                if (d > maxima[i]) maxima[i] = d < threshold ? d : -1.0;
            }
        }
        ++count;
        __syncthreads();
    }
    if (tid == 0) sizes[blockIdx.x] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int world_size) {
    const int n = static_cast<int>(points.size());
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, devices = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int *d_seeds = nullptr, *d_sizes = nullptr, *d_members = nullptr;
    double *d_distances = nullptr, *d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(n) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(n) * sizeof(Point), cudaMemcpyHostToDevice));

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t matrix_bytes = static_cast<size_t>(n) * n * sizeof(double);
    // Leave room for candidate state and for CUDA/runtime allocations. If a
    // distance matrix is too large, compute those distances in the kernel.
    if (matrix_bytes <= free_bytes / 2) {
        CUDA_CHECK(cudaMalloc(&d_distances, matrix_bytes));
        const size_t blocks = (static_cast<size_t>(n) * n + THREADS - 1) / THREADS;
        makeDistances<<<static_cast<unsigned>(std::min<size_t>(blocks, 65535)), THREADS>>>(d_points, d_distances, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t row_bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t local_seed_limit = static_cast<size_t>(n / world_size + (n % world_size != 0));
    const size_t batch_size = std::min(local_seed_limit, free_bytes / 2 / row_bytes);
    if (batch_size == 0) {
        fprintf(stderr, "Not enough GPU memory for candidate state on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    CUDA_CHECK(cudaMalloc(&d_scratch, batch_size * row_bytes));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> seeds, sizes(n);
    const int gather_threads = std::min(8, omp_get_max_threads());
    std::vector<size_t> gather_offsets(static_cast<size_t>(gather_threads) + 1);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining > 0) {
        // Each thread compacts a contiguous portion of this rank's seed
        // indices. Prefix offsets preserve the original ascending order.
        seeds.resize(local_seed_limit);
        size_t gathered_count = 0;
        #pragma omp parallel num_threads(gather_threads)
        {
            const int thread = omp_get_thread_num();
            const int threads = omp_get_num_threads();
            const size_t begin = local_seed_limit * static_cast<size_t>(thread) / threads;
            const size_t end = local_seed_limit * static_cast<size_t>(thread + 1) / threads;
            size_t count = 0;
            for (size_t k = begin; k < end; ++k) {
                const size_t i = static_cast<size_t>(rank) + k * world_size;
                if (i < static_cast<size_t>(n) && !clustered[i]) ++count;
            }
            gather_offsets[thread + 1] = count;
            #pragma omp barrier
            #pragma omp single
            {
                gather_offsets[0] = 0;
                for (int t = 0; t < threads; ++t) gather_offsets[t + 1] += gather_offsets[t];
                gathered_count = gather_offsets[threads];
            }
            size_t out = gather_offsets[thread];
            for (size_t k = begin; k < end; ++k) {
                const size_t i = static_cast<size_t>(rank) + k * world_size;
                if (i < static_cast<size_t>(n) && !clustered[i]) seeds[out++] = static_cast<int>(i);
            }
        }
        seeds.resize(gathered_count);
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), n, cudaMemcpyHostToDevice));
        int local_best[2] = {-1, n};
        for (size_t offset = 0; offset < seeds.size(); offset += batch_size) {
            const int count = static_cast<int>(std::min(batch_size, seeds.size() - offset));
            CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data() + offset, static_cast<size_t>(count) * sizeof(int), cudaMemcpyHostToDevice));
            buildCandidates<<<count, THREADS>>>(d_points, d_distances, d_clustered, d_seeds,
                                                n, threshold, d_scratch, d_sizes, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sizes.data(), d_sizes, static_cast<size_t>(count) * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                const int candidate_seed = seeds[offset + i];
                if (sizes[i] > local_best[0] || (sizes[i] == local_best[0] && candidate_seed < local_best[1])) {
                    local_best[0] = sizes[i];
                    local_best[1] = candidate_seed;
                }
            }
        }
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best[0] == 1) {
            // No two remaining points can be combined, so all later choices
            // are singleton clusters in ascending seed order.
            for (int i = 0; i < n; ++i) {
                if (!clustered[i]) clusters.push_back(Cluster{{i}, i});
            }
            break;
        }
        const int seed = global_best[1];
        const int owner = seed % world_size;
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(global_best[0]);
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(d_seeds, &seed, sizeof(int), cudaMemcpyHostToDevice));
            buildCandidates<<<1, THREADS>>>(d_points, d_distances, d_clustered, d_seeds,
                                            n, threshold, d_scratch, d_sizes, d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members,
                                  cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(cluster.members.data(), global_best[0], MPI_INT, owner, MPI_COMM_WORLD);
        #pragma omp parallel for if(global_best[0] > 256) schedule(static)
        for (int i = 0; i < global_best[0]; ++i) clustered[cluster.members[i]] = 1;
        remaining -= global_best[0];
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaFree(d_scratch));
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute independent cluster diameters in parallel, then print in order.
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for if(clusters.size() > 16) schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double maximum = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                maximum = std::max(maximum, dist);
            }
        }
        diameters[c] = maximum;
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
    struct MpiFinalizer { ~MpiFinalizer() { MPI_Finalize(); } } mpi_finalizer;
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, world_size);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    const long local_time_ms = cluster_time.count();
    long max_time_ms = 0;
    MPI_Reduce(&local_time_ms, &max_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
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

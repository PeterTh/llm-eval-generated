// Hybrid QT clustering benchmark
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
#include <climits>
#include <numeric>
#include <utility>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>
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
        
        // The original group size rounds down to zero on small inputs.
        if (N < 30) group_cnt = 1;

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

// Use the same rounding for neighborhood construction, greedy choices and validation.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(fma(dx, dx, dy * dy));
}

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t n) {
        if (n) checkCuda(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Each warp grows one seed's candidate. Short rows fit in lane registers.
// Long rows use the same striped traversal with device memory for the maxima.
template<bool shortRow>
__device__ int grow(int seed, const Point* points, const int* neighbors,
                    int degree, const unsigned char* removed, double threshold,
                    double* workspace, int* history) {
    const int lane = threadIdx.x % 32;
    int ids[4];
    double maxima[4];
    Point coordinates[4];
    if (shortRow) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int i = lane + 32 * k;
            ids[k] = i < degree ? neighbors[i] : -1;
            if (ids[k] >= 0 && !removed[ids[k]]) {
                coordinates[k] = points[ids[k]];
                maxima[k] = 0;
            } else {
                maxima[k] = CUDART_INF;
                coordinates[k] = {0, 0};
            }
        }
    } else {
        for (int i = lane; i < degree; i += 32)
            workspace[i] = removed[neighbors[i]] ? CUDART_INF : 0;
    }

    int member = seed;
    int count = 0;
    do {
        if (lane == 0) history[count] = member;
        ++count;
        const Point center = points[member];
        double best = CUDART_INF;
        int next = INT_MAX;
        for (int k = 0; k < (shortRow ? 4 : (degree - 1) / 32 + 1); ++k) {
            int i = lane + 32 * k;
            if (!shortRow && i >= degree) break;
            int id = shortRow ? ids[k] : neighbors[i];
            double score = shortRow ? maxima[k] : workspace[i];
            if (score >= threshold || score == CUDART_INF) continue;
            Point p = shortRow ? coordinates[k] : points[id];
            score = id == member ? CUDART_INF :
                    fmax(score, distance(p, center));
            if (shortRow) maxima[k] = score;
            else workspace[i] = score;
            if (score < threshold && (score < best || (score == best && id < next))) {
                best = score;
                next = id;
            }
        }
        for (int step = 16; step > 0; step /= 2) {
            double other = __shfl_down_sync(0xffffffff, best, step);
            int id = __shfl_down_sync(0xffffffff, next, step);
            if (other < best || (other == best && id < next)) {
                best = other;
                next = id;
            }
        }
        member = __shfl_sync(0xffffffff, next, 0);
    } while (member != INT_MAX);
    return count;
}

__global__ void evaluate(const Point* points, const size_t* offsets,
                         const int* neighbors, const unsigned char* removed,
                         double threshold, int rank, int ranks, int rows,
                         double* workspace, int* history, int* sizes,
                         unsigned long long* winner) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int seed = row * ranks + rank;
    if (removed[seed]) return;
    size_t start = offsets[row];
    int count = sizes[row];
    // Removing points outside a cached greedy sequence leaves every choice intact.
    bool dirty = count == 0;
    for (int i = threadIdx.x % 32; i < count; i += 32)
        dirty |= removed[history[start + i]] != 0;
    if (__any_sync(0xffffffff, dirty)) {
        int degree = offsets[row + 1] - start;
        if (degree <= 128)
            count = grow<true>(seed, points, neighbors + start, degree, removed,
                               threshold, workspace + start, history + start);
        else
            count = grow<false>(seed, points, neighbors + start, degree, removed,
                                threshold, workspace + start, history + start);
        if (threadIdx.x % 32 == 0) sizes[row] = count;
    }
    if (threadIdx.x % 32 == 0) {
        unsigned long long key = (static_cast<unsigned long long>(count) << 32)
                                  | static_cast<unsigned>(INT_MAX - seed);
        atomicMax(winner, key);
    }
}

__global__ void markRemoved(unsigned char* removed, const int* members, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) removed[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int rank, int ranks) {
    int n = points.size();
    int rows = n / ranks + (rank < n % ranks);
    // Search an x-sorted strip, then apply the exact Euclidean threshold.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return points[a].x < points[b].x; });
    std::vector<size_t> offsets(rows + 1);
    std::vector<int> neighbors;
    auto visit = [&](int row, auto emit) {
        int seed = row * ranks + rank;
        const Point p = points[seed];
        auto first = std::lower_bound(order.begin(), order.end(), p.x, [&](int id, double x) {
            return x - points[id].x > threshold;
        });
        emit(seed);
        for (auto it = first; it != order.end(); ++it) {
            int id = *it;
            if (points[id].x - p.x > threshold) break;
            if (id != seed && distance(points[id], p) < threshold) emit(id);
        }
    };
    int threads = std::min(omp_get_max_threads(), std::max(1, (rows + 31) / 32));
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(dynamic, 16)
        for (int row = 0; row < rows; ++row) {
            size_t degree = 0;
            visit(row, [&](int) { ++degree; });
            offsets[row + 1] = degree;
        }
        #pragma omp single
        {
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
            neighbors.resize(offsets.back());
        }
        #pragma omp for schedule(dynamic, 16)
        for (int row = 0; row < rows; ++row) {
            size_t pos = offsets[row];
            visit(row, [&](int id) { neighbors[pos++] = id; });
        }
    }
    DeviceBuffer<Point> coords(n);
    DeviceBuffer<size_t> starts(offsets.size());
    DeviceBuffer<int> ids(neighbors.size()), history(neighbors.size()), sizes(rows), selected(n);
    DeviceBuffer<double> workspace(neighbors.size());
    DeviceBuffer<unsigned char> removed(n);
    DeviceBuffer<unsigned long long> winner(1);
    checkCuda(cudaMemcpy(coords.ptr, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(starts.ptr, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (rows) {
        checkCuda(cudaMemcpy(ids.ptr, neighbors.data(), neighbors.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(sizes.ptr, 0, rows * sizeof(int)));
    }
    std::vector<int>().swap(neighbors);
    checkCuda(cudaMemset(removed.ptr, 0, n));
    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(rank == 0 ? n : 0);
    for (int remaining = n; remaining > 0;) {
        unsigned long long local = 0, global;
        if (rows) {
            checkCuda(cudaMemset(winner.ptr, 0, sizeof(local)));
            evaluate<<<(rows + 3) / 4, 128>>>(coords.ptr, starts.ptr, ids.ptr, removed.ptr,
                threshold, rank, ranks, rows, workspace.ptr, history.ptr, sizes.ptr, winner.ptr);
            checkCuda(cudaGetLastError());
            checkCuda(cudaMemcpy(&local, winner.ptr, sizeof(local), cudaMemcpyDeviceToHost));
        }
        MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        int count = global >> 32;
        int seed = INT_MAX - static_cast<unsigned>(global);
        if (count == 1) {
            if (rank == 0)
                for (int i = seed; i < n; ++i)
                    if (!assigned[i]) clusters.push_back({{i}, i});
            break;
        }
        std::vector<int> members(count);
        int owner = seed % ranks;
        if (rank == owner)
            checkCuda(cudaMemcpy(members.data(), history.ptr + offsets[seed / ranks],
                                 count * sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Bcast(members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(selected.ptr, members.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        markRemoved<<<(count + 127) / 128, 128>>>(removed.ptr, selected.ptr, count);
        checkCuda(cudaGetLastError());
        if (rank == 0) {
            for (int id : members) assigned[id] = 1;
            clusters.push_back({std::move(members), seed});
        }
        remaining -= count;
    }
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

static int run(int argc, char** argv, int rank, int ranks) {
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
    
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) checkCuda(cudaErrorNoDevice);
    checkCuda(cudaSetDevice(localRank % devices));

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype pointType;
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
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
    const double time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
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
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

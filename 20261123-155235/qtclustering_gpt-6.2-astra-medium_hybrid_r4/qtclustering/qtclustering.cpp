// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>
#include <climits>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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
    // Small inputs need a group-size range that can produce a nonempty group.
    const double groupRange = N <= 30 ? 2.0 : N / 30.0;
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * groupRange);
        
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceArray() { if (data) checkCuda(cudaFree(data)); }
    DeviceArray(const DeviceArray&) = delete;
};

struct Window { int begin, end; };
struct Candidate { double diameter; int id; };

// One warp grows one seed. Each lane owns a packed list of viable points.
// Updating only the distance to the newest member gives the same maximum
// as rescanning the entire cluster. Rejected points can never re-enter.
__device__ int grow(int seed, const Point* points, const int* sorted,
                    Window window, const unsigned char* removed,
                    double threshold, Candidate* candidates, int* members) {
    int lane = threadIdx.x;
    int live = 0;
    for (long long j = (long long)window.begin + lane; j < window.end; j += 32) {
        int id = sorted[j];
        if (id == seed || removed[id]) continue;
        double dist = distance(points[seed], points[id]);
        if (dist < threshold) {
            candidates[size_t(live++) * 32 + lane] = {dist, id};
        }
    }

    int size = 1;
    if (members && lane == 0) members[0] = seed;
    while (true) {
        double best = CUDART_INF;
        int closest = INT_MAX;
        for (int k = 0; k < live; ++k) {
            Candidate c = candidates[size_t(k) * 32 + lane];
            if (c.diameter < best || (c.diameter == best && c.id < closest)) {
                best = c.diameter;
                closest = c.id;
            }
        }
        for (int offset = 16; offset; offset /= 2) {
            double d = __shfl_down_sync(0xffffffff, best, offset);
            int id = __shfl_down_sync(0xffffffff, closest, offset);
            if (d < best || (d == best && id < closest)) {
                best = d;
                closest = id;
            }
        }
        closest = __shfl_sync(0xffffffff, closest, 0);
        if (closest == INT_MAX) return size;
        if (members && lane == 0) members[size] = closest;
        ++size;
        Point next = points[closest];
        for (int k = 0; k < live;) {
            Candidate& c = candidates[size_t(k) * 32 + lane];
            double d = distance(next, points[c.id]);
            if (c.id == closest || !(d < threshold)) {
                c = candidates[size_t(--live) * 32 + lane];
            } else {
                c.diameter = fmax(c.diameter, d);
                ++k;
            }
        }
    }
}

__global__ void evaluate(const Point* points, const int* sorted, const Window* windows,
                         const unsigned char* removed, int* sizes, double threshold,
                         int first, int stride, int count, Candidate* spill,
                         size_t slots, unsigned long long* best, int* members) {
    extern __shared__ Candidate shared[];
    Candidate* scratch = spill ? spill + size_t(blockIdx.x) * slots : shared;
    unsigned long long localBest = 0;
    for (long long i = blockIdx.x; i < count; i += gridDim.x) {
        int seed = first + i * stride;
        if (removed[seed]) continue;
        int size = sizes[seed];
        if (size == 0 || members) {
            size = grow(seed, points, sorted, windows[seed], removed,
                        threshold, scratch, members);
            if (threadIdx.x == 0) sizes[seed] = size;
        }
        unsigned long long key = (static_cast<unsigned long long>(size) << 32)
                                 | static_cast<unsigned>(INT_MAX - seed);
        localBest = max(localBest, key);
    }
    if (threadIdx.x == 0 && !members) atomicMax(best, localBest);
}

// Removing a point can affect only seeds within the threshold of that point.
__global__ void retire(const int* members, int count, const Point* points,
                       const int* sorted, const Window* windows, double threshold,
                       unsigned char* removed, int* sizes) {
    for (long long i = blockIdx.x; i < count; i += gridDim.x) {
        int member = members[i];
        if (threadIdx.x == 0) removed[member] = 1;
        Window w = windows[member];
        for (long long j = (long long)w.begin + threadIdx.x; j < w.end; j += blockDim.x) {
            int seed = sorted[j];
            if (distance(points[seed], points[member]) < threshold)
                atomicExch(sizes + seed, 0);
        }
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    int n = static_cast<int>(points.size());
    int localCount = n / ranks + (rank < n % ranks);
    std::vector<int> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return points[a].x < points[b].x;
    });
    std::vector<Window> windows(n);
    int maxWidth = 0;
    // Bound each seed's search by x separation before computing 2D distances.
    int threads = std::min(omp_get_max_threads(), 1 + (n - 1) / 128);
    #pragma omp parallel for num_threads(threads) reduction(max:maxWidth)
    for (int id = 0; id < n; ++id) {
        double x = points[id].x;
        auto lo = std::partition_point(sorted.begin(), sorted.end(), [&](int j) {
            return x - points[j].x > threshold;
        });
        auto hi = std::partition_point(lo, sorted.end(), [&](int j) {
            return !(points[j].x - x > threshold);
        });
        windows[id] = {int(lo - sorted.begin()), int(hi - sorted.begin())};
        if (id % ranks == rank) maxWidth = std::max(maxWidth, int(hi - lo));
    }

    DeviceArray<Point> coords(n);
    DeviceArray<Window> bounds(n);
    DeviceArray<int> order(n), sizes(n), memberIds(n);
    DeviceArray<unsigned char> removed(n);
    DeviceArray<unsigned long long> best(1);
    checkCuda(cudaMemcpy(coords.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(order.data, sorted.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(bounds.data, windows.data(), n * sizeof(Window), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(sizes.data, 0, n * sizeof(int)));
    checkCuda(cudaMemset(removed.data, 0, n));

    // Wide neighborhoods use bounded batches of global scratch space.
    size_t slots = (size_t(maxWidth) + 31) / 32 * 32;
    size_t sharedBytes = slots * sizeof(Candidate);
    int blocks = std::min(localCount, 65535);
    bool spill = sharedBytes > 32 * 1024;
    if (spill) blocks = std::min(blocks, int(std::max<size_t>(1, (64u << 20) / sharedBytes)));
    DeviceArray<Candidate> scratch(spill ? slots * blocks : 0);
    auto launch = [&](int first, int stride, int count, int grid, int* members) {
        evaluate<<<grid, 32, spill ? 0 : sharedBytes>>>(coords.data, order.data, bounds.data,
            removed.data, sizes.data, threshold, first, stride, count, scratch.data,
            slots, best.data, members);
        checkCuda(cudaGetLastError());
    };

    std::vector<Cluster> clusters;
    std::vector<unsigned char> assigned(rank == 0 ? n : 0);
    for (int remaining = n; remaining > 0;) {
        unsigned long long key = 0;
        if (localCount) {
            checkCuda(cudaMemset(best.data, 0, sizeof(key)));
            launch(rank, ranks, localCount, blocks, nullptr);
            checkCuda(cudaMemcpy(&key, best.data, sizeof(key), cudaMemcpyDeviceToHost));
        }
        MPI_Allreduce(MPI_IN_PLACE, &key, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        int size = int(key >> 32);
        int seed = INT_MAX - static_cast<int>(key & 0xffffffffu);
        if (size == 1) {
            // Removing points cannot create a pair once every seed is isolated.
            if (rank == 0) {
                for (int id = 0; id < n; ++id)
                    if (!assigned[id]) clusters.push_back({{id}, id});
            }
            break;
        }
        int owner = seed % ranks;
        Cluster winner{std::vector<int>(size), seed};
        if (rank == owner) {
            launch(seed, 1, 1, 1, memberIds.data);
            checkCuda(cudaMemcpy(winner.members.data(), memberIds.data, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(winner.members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner)
            checkCuda(cudaMemcpy(memberIds.data, winner.members.data(), size * sizeof(int),
                                 cudaMemcpyHostToDevice));
        retire<<<std::min(size, 65535), 128>>>(memberIds.data, size, coords.data, order.data,
                                               bounds.data, threshold, removed.data, sizes.data);
        checkCuda(cudaGetLastError());
        remaining -= size;
        if (rank == 0) {
            for (int id : winner.members) assigned[id] = 1;
            clusters.push_back(std::move(winner));
        }
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

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices));

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
    checkCuda(cudaDeviceSynchronize());
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_time_sec = std::chrono::duration<double>(
        cluster_end - cluster_start).count();
    double time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(time_sec));
    
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Version
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
        if (N <= 30) group_cnt = 1; // Ensure progress for small inputs.
        
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

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void selectDevice(int rank) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int local, devices;
    MPI_Comm_rank(node, &local);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(local % devices));
}

template<class T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) check(cudaFree(data)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double pointDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
#ifdef __FMA__
    return sqrt(fma(dx, dx, __dmul_rn(dy, dy)));
#else
    return sqrt(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
#endif
}

__global__ void fillDistances(const Point* points, double* table, int n) {
    size_t total = size_t(n) * n;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += size_t(gridDim.x) * blockDim.x)
        table[i] = pointDistance(points[i / n], points[i % n]);
}

struct Score {
    unsigned long long signature;
    int size;
};

__host__ __device__ unsigned long long memberBit(unsigned id) {
    return 1ULL << ((id * 2654435761u) >> 26);
}

struct Neighbor {
    double maximum;
    int id;
};

// One warp grows one seed. Each lane owns a compact list of neighbors.
// A surviving point needs just one new distance after each insertion.
template<bool Record>
__global__ void grow(const Point* points, const double* table, const unsigned char* used,
                     int n, double threshold, const int* seeds, int count,
                     Neighbor* workspace, size_t stride, Score* scores, int* members) {
    int job = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (job >= count) return;
    const int lane = threadIdx.x % 32;
    const int seed = seeds[job];
    Neighbor* list = workspace + size_t(job) * stride + lane;
    int last = seed, size = 1, live = 0;
    unsigned long long signature = memberBit(seed);
    if (Record && lane == 0) members[0] = seed;
    do {
        int best = INT_MAX, kept = 0;
        double minimum = CUDART_INF;
        const Point origin = points[last];
        const double* row = table ? table + size_t(last) * n : nullptr;
        const int limit = size == 1 ? n - lane : live * 32;
        for (int i = 0; i < limit; i += 32) {
            int id;
            double maximum;
            if (size == 1) {
                id = i + lane;
                if (used[id] || id == seed) continue;
                maximum = 0;
            } else {
                Neighbor old = list[i];
                id = old.id;
                if (id == last) continue;
                maximum = old.maximum;
            }
            double d = row ? row[id] : pointDistance(origin, points[id]);
            maximum = fmax(maximum, d);
            if (!(maximum < threshold)) continue;
            list[kept++ * 32] = {maximum, id};
            if (maximum < minimum || (maximum == minimum && id < best)) {
                minimum = maximum;
                best = id;
            }
        }
        live = kept;
        for (int offset = 16; offset; offset /= 2) {
            double other = __shfl_down_sync(0xffffffff, minimum, offset);
            int id = __shfl_down_sync(0xffffffff, best, offset);
            if (other < minimum || (other == minimum && id < best)) {
                minimum = other;
                best = id;
            }
        }
        last = __shfl_sync(0xffffffff, best, 0);
        if (last == INT_MAX) break;
        if (Record && lane == 0) members[size] = last;
        signature |= memberBit(last);
        ++size;
    } while (size < n);
    if (lane == 0) scores[job] = {signature, size};
}

__global__ void markUsed(const int* members, int count, unsigned char* used) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) used[members[i]] = 1;
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                         int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const int local = n / ranks + (rank < n % ranks);
    DeviceBuffer<Point> coords(n);
    DeviceBuffer<unsigned char> used(n);
    DeviceBuffer<int> members(n);
    check(cudaMemcpy(coords.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    check(cudaMemset(used.data, 0, n));

    size_t freeBytes, totalBytes;
    check(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t stride = (size_t(n) + 31) / 32 * 32;
    // Bound scratch space even for dense data. Cache distances only when
    // the whole matrix fits comfortably alongside the seed workspaces.
    size_t capacity = std::min(size_t(256) << 20, freeBytes / 4) / (stride * sizeof(Neighbor));
    const int batch = std::max<size_t>(1, std::min<size_t>(local, capacity));
    bool cache = local && size_t(n) <= freeBytes / 4 / sizeof(double) / n;
    DeviceBuffer<double> table(cache ? size_t(n) * n : 0);
    if (cache) {
        int blocks = std::min<size_t>(65535, (size_t(n) * n + 255) / 256);
        fillDistances<<<blocks, 256>>>(coords.data, table.data, n);
        check(cudaGetLastError());
    }
    DeviceBuffer<Neighbor> scratch(batch * stride);
    DeviceBuffer<int> seeds(batch);
    DeviceBuffer<Score> results(batch);
    std::vector<Score> scores(local, {0, 0}), fresh(batch);
    std::vector<unsigned char> removed(n);
    std::vector<int> dirty;
    std::vector<Cluster> clusters;
    Point lower{}, upper{};
    unsigned long long changed = 0;
    int remaining = n;
    // Each host thread handles several hundred seeds before adding workers.
    int workers = std::min(omp_get_max_threads(), local ? 1 + (local - 1) / 256 : 1);
    while (remaining) {
        dirty.clear();
        for (int i = 0; i < local; ++i) {
            int seed = rank + i * ranks;
            // A disjoint signature proves the previous greedy sequence is
            // still available. Hash collisions only cause extra work.
            if (removed[seed]) continue;
            if (scores[i].size) {
                if (!(scores[i].signature & changed)) continue;
                // Every member must be within threshold of its seed. Rule
                // out distant winners even when the signatures collide.
                const Point p = points[seed];
                if (lower.x - p.x >= threshold || p.x - upper.x >= threshold ||
                    lower.y - p.y >= threshold || p.y - upper.y >= threshold) continue;
            }
            dirty.push_back(seed);
        }
        for (size_t start = 0; start < dirty.size(); start += batch) {
            int count = std::min<size_t>(batch, dirty.size() - start);
            check(cudaMemcpy(seeds.data, dirty.data() + start, count * sizeof(int), cudaMemcpyHostToDevice));
            grow<false><<<(count + 3) / 4, 128>>>(coords.data, table.data, used.data, n,
                threshold, seeds.data, count, scratch.data, stride, results.data, nullptr);
            check(cudaGetLastError());
            check(cudaMemcpy(fresh.data(), results.data, count * sizeof(Score), cudaMemcpyDeviceToHost));
            #pragma omp parallel for num_threads(workers) if(count >= 512)
            for (int j = 0; j < count; ++j)
                scores[dirty[start + j] / ranks] = fresh[j];
        }
        // Pack descending cardinality and ascending seed in one reduction.
        unsigned long long winner = 0;
        #pragma omp parallel for reduction(max:winner) num_threads(workers) if(workers > 1)
        for (int i = 0; i < local; ++i) {
            int seed = rank + i * ranks;
            if (!removed[seed]) {
                auto key = (static_cast<unsigned long long>(scores[i].size) << 32)
                    | static_cast<unsigned>(INT_MAX - seed);
                winner = std::max(winner, key);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &winner, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        int size = winner >> 32;
        int seed = INT_MAX - static_cast<int>(winner & 0xffffffffULL);
        if (size == 1) {
            if (rank == 0)
                for (int id = 0; id < n; ++id)
                    if (!removed[id]) clusters.push_back({{id}, id});
            break;
        }
        Cluster selected{std::vector<int>(size), seed};
        int owner = seed % ranks;
        if (rank == owner) {
            check(cudaMemcpy(seeds.data, &seed, sizeof(int), cudaMemcpyHostToDevice));
            grow<true><<<1, 32>>>(coords.data, table.data, used.data, n, threshold, seeds.data,
                1, scratch.data, stride, results.data, members.data);
            check(cudaGetLastError());
            check(cudaMemcpy(selected.members.data(), members.data, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(selected.members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner)
            check(cudaMemcpy(members.data, selected.members.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        markUsed<<<(size + 255) / 256, 256>>>(members.data, size, used.data);
        check(cudaGetLastError());
        changed = 0;
        lower = upper = points[seed];
        for (int id : selected.members) {
            removed[id] = 1;
            changed |= memberBit(id);
            lower.x = std::min(lower.x, points[id].x);
            lower.y = std::min(lower.y, points[id].y);
            upper.x = std::max(upper.x, points[id].x);
            upper.y = std::max(upper.y, points[id].y);
        }
        if (rank == 0) clusters.push_back(std::move(selected));
        remaining -= size;
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

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    struct Finalize { ~Finalize() { MPI_Finalize(); } } finalize;
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
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    selectDevice(rank);
    
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

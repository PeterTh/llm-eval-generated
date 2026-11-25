// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <numeric>
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
    // Ensure that tiny inputs can draw a nonempty group.
    const double groupScale = N <= 30 ? 2.0 : N / 30.0;
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * groupScale);
        
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

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
struct DeviceBuffer {
    T* data;
    explicit DeviceBuffer(size_t count) {
        checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct SeedWindow {
    int seed, begin, end;
};

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

constexpr int warpsPerBlock = 4;
constexpr int sharedCapacity = 128;

// One warp owns one seed. Compact its eligible neighbors, then update each
// neighbor's maximum distance using only the most recently added member.
__global__ void growCandidates(const SeedWindow* jobs, int count, const Point* points,
                               const int* byX, const unsigned char* removed, double threshold,
                               int pitch, int* neighborSpace, double* distanceSpace,
                               int* members, int* sizes) {
    __shared__ int localIds[warpsPerBlock][sharedCapacity];
    __shared__ double localMax[warpsPerBlock][sharedCapacity];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * warpsPerBlock + warp;
    if (row >= count) return;
    constexpr unsigned mask = 0xffffffffu;
    const SeedWindow job = jobs[row];
    const size_t offset = static_cast<size_t>(row) * pitch;
    const Point seed = points[job.seed];
    int total = 0;
    for (int base = job.begin; base < job.end; base += 32) {
        int id = -1;
        double d = 0;
        if (base + lane < job.end) {
            id = byX[base + lane];
            if (id == job.seed || removed[id]) id = -1;
            if (id >= 0) d = gpuDistance(points[id], seed);
        }
        unsigned eligible = __ballot_sync(mask, id >= 0 && d < threshold);
        if (eligible & (1u << lane)) {
            int slot = total + __popc(eligible & ((1u << lane) - 1));
            if (slot < sharedCapacity) {
                localIds[warp][slot] = id;
                localMax[warp][slot] = d;
            } else {
                neighborSpace[offset + slot] = id;
                distanceSpace[offset + slot] = d;
            }
        }
        total += __popc(eligible);
    }
    __syncwarp();
    if (!lane) members[offset] = job.seed;
    int size = 1, last = job.seed;
    while (size <= total) {
        const Point p = points[last];
        double best = threshold;
        int bestId = INT_MAX;
        for (int i = lane; i < total; i += 32) {
            bool local = i < sharedCapacity;
            int id = local ? localIds[warp][i] : neighborSpace[offset + i];
            double& maximum = local ? localMax[warp][i] : distanceSpace[offset + i];
            double d = maximum;
            if (!(d < threshold)) continue;
            if (size > 1) {
                d = id == last ? threshold : fmax(d, gpuDistance(points[id], p));
                maximum = d;
            }
            if (d < threshold && (d < best || (d == best && id < bestId))) {
                best = d;
                bestId = id;
            }
        }
        for (int delta = 16; delta; delta /= 2) {
            double other = __shfl_down_sync(mask, best, delta);
            int id = __shfl_down_sync(mask, bestId, delta);
            if (other < best || (other == best && id < bestId)) {
                best = other;
                bestId = id;
            }
        }
        last = __shfl_sync(mask, bestId, 0);
        if (last == INT_MAX) break;
        if (!lane) members[offset + size] = last;
        ++size;
    }
    if (!lane) sizes[row] = size;
}

// A single maximum reduction selects the largest cluster, then the lowest seed.
using Winner = unsigned long long;
static Winner priority(int size, int seed) {
    return (static_cast<Winner>(size) << 32) | static_cast<unsigned>(INT_MAX - seed);
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = points.size();
    std::vector<int> byX(n);
    std::iota(byX.begin(), byX.end(), 0);
    std::sort(byX.begin(), byX.end(), [&](int a, int b) { return points[a].x < points[b].x; });

    const int owned = n / ranks + (rank < n % ranks);
    std::vector<SeedWindow> windows(owned);
    int pitch = 1;
    const int threads = std::min(omp_get_max_threads(), std::max(1, owned / 64 + (owned % 64 != 0)));
    #pragma omp parallel for num_threads(threads) reduction(max:pitch)
    for (int i = 0; i < owned; ++i) {
        const int seed = rank + i * ranks;
        const double x = points[seed].x;
        int begin = 0, end = n;
        if (std::isfinite(threshold)) {
            auto lo = std::lower_bound(byX.begin(), byX.end(), x,
                [&](int id, double center) { return center - points[id].x > threshold; });
            auto hi = std::upper_bound(lo, byX.end(), x,
                [&](double center, int id) { return points[id].x - center > threshold; });
            begin = lo - byX.begin();
            end = hi - byX.begin();
        }
        windows[i] = {seed, begin, end};
        pitch = std::max(pitch, end - begin);
    }

    DeviceBuffer<Point> dPoints(n);
    DeviceBuffer<int> dOrder(n);
    DeviceBuffer<unsigned char> dRemoved(n);
    checkCuda(cudaMemcpy(dPoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dOrder.data, byX.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Batch independently of N to bound scratch even for very wide thresholds.
    const size_t budget = std::min<size_t>(freeBytes / 4, 128u << 20);
    const int batch = std::max<size_t>(1, std::min<size_t>(owned, budget / (16ull * pitch + 32)));
    const size_t slots = static_cast<size_t>(batch) * pitch;
    DeviceBuffer<int> dMembers(slots), dIds(pitch > sharedCapacity ? slots : 1), dSizes(batch);
    DeviceBuffer<double> dMax(pitch > sharedCapacity ? slots : 1);
    DeviceBuffer<SeedWindow> dJobs(batch);
    std::vector<int> sizes(batch), rows;
    std::vector<SeedWindow> jobs;
    jobs.reserve(owned);
    std::vector<std::vector<int>> cached(owned);
    std::vector<unsigned char> removed(n, 0);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        jobs.clear();
        for (int i = 0; i < owned; ++i) {
            if (!removed[windows[i].seed] && cached[i].empty()) jobs.push_back(windows[i]);
        }
        if (!jobs.empty())
            checkCuda(cudaMemcpy(dRemoved.data, removed.data(), n, cudaMemcpyHostToDevice));
        for (size_t start = 0; start < jobs.size(); start += batch) {
            int count = std::min<size_t>(batch, jobs.size() - start);
            checkCuda(cudaMemcpy(dJobs.data, jobs.data() + start, count * sizeof(SeedWindow),
                                 cudaMemcpyHostToDevice));
            growCandidates<<<(count + warpsPerBlock - 1) / warpsPerBlock, 32 * warpsPerBlock>>>(
                dJobs.data, count, dPoints.data, dOrder.data, dRemoved.data, threshold, pitch,
                dIds.data, dMax.data, dMembers.data, dSizes.data);
            checkCuda(cudaGetLastError());
            checkCuda(cudaMemcpy(sizes.data(), dSizes.data, count * sizeof(int), cudaMemcpyDeviceToHost));
            int width = *std::max_element(sizes.begin(), sizes.begin() + count);
            rows.resize(static_cast<size_t>(count) * width);
            checkCuda(cudaMemcpy2D(rows.data(), width * sizeof(int), dMembers.data,
                                   pitch * sizeof(int), width * sizeof(int), count, cudaMemcpyDeviceToHost));
            #pragma omp parallel for num_threads(threads) if(count >= 64)
            for (int j = 0; j < count; ++j) {
                auto first = rows.begin() + static_cast<size_t>(j) * width;
                cached[jobs[start + j].seed / ranks].assign(first, first + sizes[j]);
            }
        }
        Winner local = 0, winner;
        #pragma omp parallel for num_threads(threads) reduction(max:local)
        for (int i = 0; i < owned; ++i)
            if (!cached[i].empty()) local = std::max(local, priority(cached[i].size(), windows[i].seed));
        MPI_Allreduce(&local, &winner, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int size = winner >> 32;
        if (size == 1) {
            // No remaining pair is eligible. Sequential ties follow input order.
            if (!rank) {
                for (int i = 0; i < n; ++i) {
                    if (!removed[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }
        const int seed = INT_MAX - static_cast<unsigned>(winner);
        std::vector<int> members;
        if (rank == seed % ranks) members.swap(cached[seed / ranks]);
        else members.resize(size);
        MPI_Bcast(members.data(), size, MPI_INT, seed % ranks, MPI_COMM_WORLD);
        for (int id : members) removed[id] = 1;
        remaining -= size;
        // Removing points outside a candidate preserves every greedy choice.
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int i = 0; i < owned; ++i) {
            auto& candidate = cached[i];
            if (std::any_of(candidate.begin(), candidate.end(), [&](int id) { return removed[id]; }))
                candidate.clear();
        }
        if (!rank) clusters.push_back({std::move(members), seed});
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
    struct Finalize { ~Finalize() { MPI_Finalize(); } } finalize;
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (!rank) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, devices;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(nodeRank % devices));
    if (!rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (!rank) generateSyntheticData(points, num_points);
    MPI_Datatype pointType;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
    double global_time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &global_time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank) return 0;
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(global_time_sec));
    
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
    const double time_sec = global_time_sec;
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

// QT Clustering Benchmark - MPI / OpenMP / CUDA
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
    // Below 30 points the original group-size expression always rounds to zero.
    const double group_scale = N < 30 ? 2.0 : N / 30.0;
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * group_scale);
        
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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define GPU(call) checkCuda((call), #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        GPU(cudaMalloc(&data, std::max<size_t>(1, count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    void upload(const T* src, size_t count) {
        GPU(cudaMemcpy(data, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* dst, size_t count) const {
        GPU(cudaMemcpy(dst, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

struct SeedRange {
    int seed, begin, end;
};

__device__ double separation(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Each warp owns a disjoint scratch row. Compact the x-window into actual
// neighbors, then maintain each neighbor's maximum distance incrementally.
__global__ void grow(const SeedRange* jobs, int count, const int* sorted,
                     const Point* points, const unsigned char* used, double threshold,
                     size_t pitch, int* neighbors, double* maxima, int* sizes, int* members) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= count) return;
    const int lane = threadIdx.x % 32;
    const unsigned mask = 0xffffffffu;
    const SeedRange job = jobs[warp];
    neighbors += warp * pitch;
    maxima += warp * pitch;
    int remaining = 0;
    const Point seed = points[job.seed];
    for (long long start = job.begin; start < job.end; start += 32) {
        int id = -1;
        double d = CUDART_INF;
        if (start + lane < job.end) {
            id = sorted[start + lane];
            if (id != job.seed && !used[id]) d = separation(points[id], seed);
        }
        const unsigned eligible = __ballot_sync(mask, d < threshold);
        if (d < threshold) {
            int slot = remaining + __popc(eligible & ((1u << lane) - 1));
            neighbors[slot] = id;
            maxima[slot] = d;
        }
        remaining += __popc(eligible);
    }
    __syncwarp();
    int size = 1, last = job.seed;
    if (members && !lane) members[0] = last;
    while (remaining) {
        const Point newest = points[last];
        double best = CUDART_INF;
        int best_id = INT_MAX;
        for (int i = lane; i < remaining; i += 32) {
            int id = neighbors[i];
            double d = maxima[i];
            if (size > 1 && d < threshold) {
                d = id == last ? CUDART_INF : fmax(d, separation(points[id], newest));
                maxima[i] = d;
            }
            if (d < threshold && (d < best || (d == best && id < best_id))) {
                best = d;
                best_id = id;
            }
        }
        for (int offset = 16; offset; offset /= 2) {
            double other = __shfl_down_sync(mask, best, offset);
            int id = __shfl_down_sync(mask, best_id, offset);
            if (other < best || (other == best && id < best_id)) {
                best = other;
                best_id = id;
            }
        }
        last = __shfl_sync(mask, best_id, 0);
        if (last == INT_MAX) break;
        if (members && !lane) members[size] = last;
        ++size;
    }
    if (!lane) sizes[warp] = size;
}

__global__ void removeMembers(const int* members, int count, unsigned char* used) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) used[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = points.size();
    const int local = n / ranks + (rank < n % ranks);
    const int threads = std::min(omp_get_max_threads(), std::max(1, (local - 1) / 256 + 1));
    std::vector<int> sorted(n);
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return points[a].x < points[b].x; });

    std::vector<SeedRange> ranges(local);
    int width = 1;
    #pragma omp parallel for num_threads(threads) reduction(max:width)
    for (int i = 0; i < local; ++i) {
        int seed = rank + i * ranks;
        double x = points[seed].x;
        auto begin = std::partition_point(sorted.begin(), sorted.end(), [&](int id) {
            return x - points[id].x >= threshold;
        });
        auto end = std::partition_point(begin, sorted.end(), [&](int id) {
            return points[id].x - x < threshold;
        });
        ranges[i] = {seed, int(begin - sorted.begin()), int(end - sorted.begin())};
        width = std::max(width, int(end - begin));
    }

    DeviceBuffer<Point> coords(n);
    DeviceBuffer<int> order(n), member_ids(n);
    DeviceBuffer<unsigned char> used(n);
    coords.upload(points.data(), n);
    order.upload(sorted.data(), n);
    GPU(cudaMemset(used.data, 0, n));
    // Bound temporary storage even when every point is a neighbor.
    size_t free_bytes, total_bytes;
    GPU(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t budget = std::min<size_t>(free_bytes / 4, 128ULL << 20);
    int batch = std::min<size_t>(std::max(local, 1), std::max<size_t>(1, budget /
                                 (size_t(width) * (sizeof(double) + sizeof(int)))));
    DeviceBuffer<int> neighbors(size_t(batch) * width), sizes(batch);
    DeviceBuffer<double> maxima(size_t(batch) * width);
    DeviceBuffer<SeedRange> device_jobs(batch);
    auto evaluate = [&](const SeedRange* jobs, int count, int* result, int* members) {
        device_jobs.upload(jobs, count);
        grow<<<(count + 3) / 4, 128>>>(device_jobs.data, count, order.data, coords.data,
            used.data, threshold, width, neighbors.data, maxima.data, sizes.data, members);
        GPU(cudaGetLastError());
        sizes.download(result, count);
    };

    std::vector<Cluster> clusters;
    std::vector<unsigned char> taken(n);
    std::vector<int> scores(local), results(batch);
    std::vector<SeedRange> jobs;
    jobs.reserve(local);
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    int left = n;
    while (left) {
        jobs.clear();
        for (int i = 0; i < local; ++i) {
            int seed = ranges[i].seed;
            if (taken[seed]) continue;
            Point p = points[seed];
            // A removed point outside the seed's threshold could never have
            // participated in its greedy sequence. Its cached size stays exact.
            if (left == n || (p.x - xmax < threshold && xmin - p.x < threshold &&
                              p.y - ymax < threshold && ymin - p.y < threshold))
                jobs.push_back(ranges[i]);
        }
        for (size_t offset = 0; offset < jobs.size();) {
            int count = std::min<size_t>(batch, jobs.size() - offset);
            evaluate(jobs.data() + offset, count, results.data(), nullptr);
            for (int j = 0; j < count; ++j)
                scores[jobs[offset + j].seed / ranks] = results[j];
            offset += count;
        }

        // Unsigned lexicographic key: larger clusters first, then smaller seeds.
        unsigned long long key = 0;
        #pragma omp parallel for num_threads(threads) reduction(max:key)
        for (int i = 0; i < local; ++i) {
            int seed = ranges[i].seed;
            if (!taken[seed]) key = std::max(key, (static_cast<unsigned long long>(scores[i]) << 32)
                                                 | unsigned(INT_MAX - seed));
        }
        MPI_Allreduce(MPI_IN_PLACE, &key, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int size = key >> 32;
        if (size == 1) {
            if (!rank) {
                for (int i = 0; i < n; ++i) {
                    if (!taken[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }
        int seed = INT_MAX - int(key & 0xffffffffu);
        int owner = seed % ranks;
        std::vector<int> members(size);
        if (rank == owner) {
            int actual;
            evaluate(&ranges[seed / ranks], 1, &actual, member_ids.data);
            if (actual != size) MPI_Abort(MPI_COMM_WORLD, 1);
            member_ids.download(members.data(), size);
        }
        MPI_Bcast(members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) member_ids.upload(members.data(), size);
        removeMembers<<<(size - 1) / 256 + 1, 256>>>(member_ids.data, size, used.data);
        GPU(cudaGetLastError());
        xmin = xmax = points[seed].x;
        ymin = ymax = points[seed].y;
        for (int id : members) {
            taken[id] = 1;
            xmin = std::min(xmin, points[id].x);
            xmax = std::max(xmax, points[id].x);
            ymin = std::min(ymin, points[id].y);
            ymax = std::max(ymax, points[id].y);
        }
        if (!rank) clusters.push_back({std::move(members), seed});
        left -= size;
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
    struct Finalize {
        ~Finalize() { MPI_Finalize(); }
    } finalize;
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
    int local_rank, devices;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
    GPU(cudaGetDeviceCount(&devices));
    if (!devices) MPI_Abort(MPI_COMM_WORLD, 1);
    GPU(cudaSetDevice(local_rank % devices));
    GPU(cudaFree(nullptr));

    std::vector<Point> points(num_points);
    if (!rank) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    if (rank) return 0;
    
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

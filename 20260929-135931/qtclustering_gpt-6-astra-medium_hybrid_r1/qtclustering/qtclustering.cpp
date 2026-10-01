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
#include <limits>
#include <vector>
#include <cstdint>
#include <climits>
#include <cfloat>
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

// Fail collectively instead of leaving other ranks waiting in a collective.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(n, size_t(1)) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK = 256;

// One block owns a seed. Maintain each neighbor's maximum distance incrementally,
// reducing growth from repeated all-member scans to one distance per neighbor.
// A cached candidate is still exact if none of its members has been removed:
// removing unselected alternatives cannot change any of its greedy choices.
__global__ void growCandidates(const Point* points, const size_t* offsets,
                               const int* neighbors, double* maxima, int* members,
                               int* counts, const unsigned char* removed,
                               int localCount, int rank, int ranks, double threshold) {
    __shared__ double distances[BLOCK];
    __shared__ int indices[BLOCK];
    __shared__ int dirty;
    const int lane = threadIdx.x;
    for (int slot = blockIdx.x; slot < localCount; slot += gridDim.x) {
        const int seed = int(static_cast<long long>(slot) * ranks + rank);
        if (removed[seed]) {
            if (lane == 0) counts[slot] = 0;
            continue;
        }
        const size_t begin = offsets[slot], end = offsets[slot + 1];
        int* output = members + begin + slot;
        if (lane == 0) dirty = counts[slot] == 0;
        __syncthreads();
        for (int j = lane; j < counts[slot]; j += BLOCK)
            if (removed[output[j]]) atomicExch(&dirty, 1);
        __syncthreads();
        const bool rebuild = dirty != 0;
        __syncthreads();
        if (!rebuild) continue;
        for (size_t j = begin + lane; j < end; j += BLOCK)
            maxima[j] = removed[neighbors[j]] ? -1.0 : 0.0;
        if (lane == 0) output[0] = seed;
        __syncthreads();
        int last = seed, count = 1;
        while (true) {
            double best = DBL_MAX;
            int bestIndex = INT_MAX;
            for (size_t j = begin + lane; j < end; j += BLOCK) {
                if (maxima[j] < 0.0) continue;
                const int candidate = neighbors[j];
                if (candidate == last) { maxima[j] = -1.0; continue; }
                const double dx = points[candidate].x - points[last].x;
                const double dy = points[candidate].y - points[last].y;
                const double d = fmax(maxima[j], sqrt(dx * dx + dy * dy));
                // Once outside the threshold, a candidate can never return.
                maxima[j] = d < threshold ? d : -1.0;
                if (d < threshold && (d < best || (d == best && candidate < bestIndex))) {
                    best = d;
                    bestIndex = candidate;
                }
            }
            distances[lane] = best;
            indices[lane] = bestIndex;
            __syncthreads();
            for (int stride = BLOCK / 2; stride; stride /= 2) {
                if (lane < stride) {
                    const double other = distances[lane + stride];
                    const int index = indices[lane + stride];
                    if (other < distances[lane] ||
                        (other == distances[lane] && index < indices[lane])) {
                        distances[lane] = other;
                        indices[lane] = index;
                    }
                }
                __syncthreads();
            }
            last = indices[0];
            if (last == INT_MAX) break;
            if (lane == 0) output[count] = last;
            ++count;
            __syncthreads();
        }
        if (lane == 0) counts[slot] = count;
        __syncthreads();
    }
}

// Packed ordering: largest cardinality, then smallest global seed index.
__global__ void bestCandidate(const int* counts, int n, int rank, int ranks,
                              unsigned long long* result) {
    __shared__ unsigned long long keys[BLOCK];
    unsigned long long best = 0;
    for (int i = threadIdx.x; i < n; i += BLOCK) {
        if (counts[i]) {
            const unsigned int seed = static_cast<unsigned int>(static_cast<long long>(i) * ranks + rank);
            const unsigned long long key = (static_cast<unsigned long long>(counts[i]) << 32) |
                                           (UINT_MAX - seed);
            best = best > key ? best : key;
        }
    }
    keys[threadIdx.x] = best;
    __syncthreads();
    for (int stride = BLOCK / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            keys[threadIdx.x] = max(keys[threadIdx.x], keys[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = keys[0];
}

__global__ void removeMembers(unsigned char* removed, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        removed[members[i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int n = static_cast<int>(points.size());
    const int localCount = n / ranks + (rank < n % ranks);
    // Store only local seeds' threshold neighbors, not a replicated dense matrix.
    // Two passes keep host peak memory bounded by the final CSR representation.
    std::vector<size_t> offsets(localCount + 1, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < localCount; ++s) {
        const int seed = int(static_cast<long long>(s) * ranks + rank);
        size_t count = 0;
        for (int j = 0; j < n; ++j)
            if (j != seed && distance(points[seed], points[j]) < threshold) ++count;
        offsets[s + 1] = count;
    }
    for (int s = 0; s < localCount; ++s) offsets[s + 1] += offsets[s];
    std::vector<int> neighbors(offsets.back());
    #pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < localCount; ++s) {
        const int seed = int(static_cast<long long>(s) * ranks + rank);
        size_t pos = offsets[s];
        for (int j = 0; j < n; ++j)
            if (j != seed && distance(points[seed], points[j]) < threshold) neighbors[pos++] = j;
    }
    DeviceBuffer<Point> dp(n);
    DeviceBuffer<size_t> doff(offsets.size());
    DeviceBuffer<int> dn(neighbors.size()), dm(neighbors.size() + localCount), dc(localCount), chosen(n);
    DeviceBuffer<double> maxima(neighbors.size());
    DeviceBuffer<unsigned char> removed(n);
    DeviceBuffer<unsigned long long> winner(1);
    cudaCheck(cudaMemcpy(dp.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(doff.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    if (!neighbors.empty())
        cudaCheck(cudaMemcpy(dn.data, neighbors.data(), neighbors.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int>().swap(neighbors);
    cudaCheck(cudaMemset(dc.data, 0, localCount * sizeof(int)));
    cudaCheck(cudaMemset(removed.data, 0, n));
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        if (localCount)
            growCandidates<<<std::min(localCount, 65535), BLOCK>>>(dp.data, doff.data, dn.data,
                maxima.data, dm.data, dc.data, removed.data, localCount, rank, ranks, threshold);
        bestCandidate<<<1, BLOCK>>>(dc.data, localCount, rank, ranks, winner.data);
        cudaCheck(cudaGetLastError());
        unsigned long long localBest, globalBest;
        cudaCheck(cudaMemcpy(&localBest, winner.data, sizeof(localBest), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int count = static_cast<int>(globalBest >> 32);
        // If the largest candidate is a singleton, all remaining winners are
        // singletons in ascending seed order. Avoid one collective per point.
        if (count == 1) {
            if (rank == 0) {
                std::vector<unsigned char> done(n);
                cudaCheck(cudaMemcpy(done.data(), removed.data, n, cudaMemcpyDeviceToHost));
                for (int seed = 0; seed < n; ++seed)
                    if (!done[seed]) clusters.push_back({{seed}, seed});
            }
            break;
        }
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(globalBest));
        const int owner = seed % ranks;
        std::vector<int> members(count);
        if (rank == owner) {
            const int slot = seed / ranks;
            cudaCheck(cudaMemcpy(members.data(), dm.data + offsets[slot] + slot,
                                 count * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(chosen.data, members.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        removeMembers<<<std::min(65535, (count - 1) / BLOCK + 1), BLOCK>>>(removed.data, chosen.data, count);
        cudaCheck(cudaGetLastError());
        if (rank == 0) clusters.push_back({std::move(members), seed});
        remaining -= count;
    }
    cudaCheck(cudaDeviceSynchronize());
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

// MPI calls stay on the main thread; OpenMP workers only prepare local data.
struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    ~MpiSession() { MPI_Finalize(); }
};

int main(int argc, char** argv) {
    MpiSession mpi(argc, argv);
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
    
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, devices;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA accelerator is required on every MPI node.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    cudaCheck(cudaFree(nullptr));

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    // Broadcast in bounded chunks to avoid MPI's int count limit.
    for (size_t offset = 0; offset < points.size(); offset += 1048576) {
        const int count = static_cast<int>(std::min(size_t(1048576), points.size() - offset));
        MPI_Bcast(reinterpret_cast<double*>(points.data() + offset), 2 * count,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    const auto cluster_time = std::chrono::milliseconds(max_cluster_time);
    
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

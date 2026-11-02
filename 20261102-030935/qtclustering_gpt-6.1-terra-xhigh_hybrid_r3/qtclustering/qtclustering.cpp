// QT Clustering Benchmark - MPI, OpenMP and CUDA Version
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
#include <climits>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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
        // Otherwise small valid inputs can generate only empty groups forever.
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


// Each block grows one seed's cluster. The cached maximum is updated only
// with the most recently added member; it is exactly the maximum that the
// sequential implementation would recompute for that candidate.
static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int BLOCK_SIZE = 128;

__global__ void makeDistances(const Point* points, double* distances, int n) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const Point a = points[k / n], b = points[k % n];
        const double dx = a.x - b.x, dy = a.y - b.y;
        distances[k] = sqrt(dx * dx + dy * dy);
    }
}

__global__ void growClusters(const Point* points, const double* distances,
                             const unsigned char* clustered, const int* seeds,
                             int n, double threshold, double* maxima,
                             int* counts, int* members) {
    __shared__ double values[BLOCK_SIZE / 32];
    __shared__ int indices[BLOCK_SIZE / 32];
    __shared__ int last, count, nextIndex;
    const int tid = threadIdx.x;
    double* cache = maxima + static_cast<size_t>(blockIdx.x) * n;
    if (tid == 0) {
        last = seeds[blockIdx.x];
        count = 1;
        if (members) members[0] = last;
    }
    for (int i = tid; i < n; i += BLOCK_SIZE) cache[i] = 0.0;
    __syncthreads();

    while (true) {
        double best = CUDART_INF;
        int bestIndex = INT_MAX;
        const Point p = points[last];
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            if (clustered[i] || i == last) {
                cache[i] = CUDART_INF;
                continue;
            }
            double d = cache[i];
            if (d < threshold) {
                double next;
                if (distances) {
                    next = distances[static_cast<size_t>(last) * n + i];
                } else {
                    const double dx = points[i].x - p.x;
                    const double dy = points[i].y - p.y;
                    next = sqrt(dx * dx + dy * dy);
                }
                d = fmax(d, next);
                cache[i] = d;
                if (d < threshold && (d < best || (d == best && i < bestIndex))) {
                    best = d;
                    bestIndex = i;
                }
            }
        }
        // Reduce first within each warp and then across the four warp winners.
        const int lane = tid % 32;
        for (int offset = 16; offset > 0; offset /= 2) {
            const double other = __shfl_down_sync(0xffffffff, best, offset);
            const int otherIndex = __shfl_down_sync(0xffffffff, bestIndex, offset);
            if (other < best || (other == best && otherIndex < bestIndex)) {
                best = other;
                bestIndex = otherIndex;
            }
        }
        if (lane == 0) {
            values[tid / 32] = best;
            indices[tid / 32] = bestIndex;
        }
        __syncthreads();
        if (tid < 32) {
            best = lane < BLOCK_SIZE / 32 ? values[lane] : CUDART_INF;
            bestIndex = lane < BLOCK_SIZE / 32 ? indices[lane] : INT_MAX;
            for (int offset = 16; offset > 0; offset /= 2) {
                const double other = __shfl_down_sync(0xffffffff, best, offset);
                const int otherIndex = __shfl_down_sync(0xffffffff, bestIndex, offset);
                if (other < best || (other == best && otherIndex < bestIndex)) {
                    best = other;
                    bestIndex = otherIndex;
                }
            }
            if (lane == 0) {
                nextIndex = bestIndex;
                if (bestIndex != INT_MAX) {
                    last = bestIndex;
                    if (members) members[count] = last;
                    ++count;
                }
            }
        }
        __syncthreads();
        if (nextIndex == INT_MAX) break;
    }
    if (tid == 0) counts[blockIdx.x] = count;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    Point* dPoints = nullptr;
    unsigned char* dClustered = nullptr;
    int *dSeeds = nullptr, *dCounts = nullptr, *dMembers = nullptr;
    double *dMaxima = nullptr, *dDistances = nullptr;
    CUDA_CHECK(cudaMalloc(&dPoints, sizeof(Point) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&dClustered, static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&dSeeds, sizeof(int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&dCounts, sizeof(int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&dMembers, sizeof(int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMemcpy(dPoints, points.data(), sizeof(Point) * static_cast<size_t>(n), cudaMemcpyHostToDevice));

    size_t freeBytes, totalBytes;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    // A distance table amortizes sqrt over all seeds and clustering rounds.
    // For large inputs use the same calculation on demand instead.
    if (matrixBytes <= freeBytes / 3 && matrixBytes <= size_t(1024) * 1024 * 1024) {
        CUDA_CHECK(cudaMalloc(&dDistances, matrixBytes));
        const size_t blocks = (static_cast<size_t>(n) * n + 255) / 256;
        makeDistances<<<static_cast<int>(std::min<size_t>(blocks, 65535)), 256>>>(dPoints, dDistances, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t perSeed = sizeof(double) * static_cast<size_t>(n);
    const size_t desiredSeeds = (static_cast<size_t>(n) + ranks - 1) / ranks;
    const size_t budget = std::min(freeBytes / 2, size_t(512) * 1024 * 1024);
    const int batchCapacity = static_cast<int>(std::min(desiredSeeds, budget / perSeed));
    if (batchCapacity < 1) {
        fprintf(stderr, "Not enough GPU memory for a candidate cluster on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaMalloc(&dMaxima, perSeed * batchCapacity));

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> remaining(n), seeds, counts(batchCapacity);
    for (int i = 0; i < n; ++i) remaining[i] = i;
    std::vector<Cluster> clusters;
    while (!remaining.empty()) {
        const int localCount = rank < static_cast<int>(remaining.size())
            ? (static_cast<int>(remaining.size()) - 1 - rank) / ranks + 1 : 0;
        seeds.resize(localCount);
        #pragma omp parallel for if(localCount > 4096) schedule(static)
        for (int i = 0; i < localCount; ++i) seeds[i] = remaining[rank + i * ranks];
        CUDA_CHECK(cudaMemcpy(dClustered, clustered.data(), n, cudaMemcpyHostToDevice));
        int localBestSize = 0, localBestSeed = INT_MAX;
        for (int start = 0; start < localCount; start += batchCapacity) {
            const int batch = std::min(batchCapacity, localCount - start);
            CUDA_CHECK(cudaMemcpy(dSeeds, seeds.data() + start, batch * sizeof(int), cudaMemcpyHostToDevice));
            growClusters<<<batch, BLOCK_SIZE>>>(dPoints, dDistances, dClustered, dSeeds, n,
                                               threshold, dMaxima, dCounts, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(counts.data(), dCounts, batch * sizeof(int), cudaMemcpyDeviceToHost));
            // Select the first seed in input order on ties, as in the original.
            #pragma omp parallel if(batch >= 256) num_threads(std::min(omp_get_max_threads(), (batch + 255) / 256))
            {
                int bestSize = 0, bestSeed = INT_MAX;
                #pragma omp for nowait schedule(static)
                for (int i = 0; i < batch; ++i) {
                    int seed = seeds[start + i], size = counts[i];
                    if (size > bestSize || (size == bestSize && seed < bestSeed)) {
                        bestSize = size; bestSeed = seed;
                    }
                }
                #pragma omp critical
                {
                    if (bestSize > localBestSize || (bestSize == localBestSize && bestSeed < localBestSeed)) {
                        localBestSize = bestSize; localBestSeed = bestSeed;
                    }
                }
            }
        }
        int localPair[2] = {localBestSize, localBestSeed}, globalPair[2];
        MPI_Allreduce(localPair, globalPair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int winner = static_cast<int>(std::lower_bound(remaining.begin(), remaining.end(), globalPair[1]) - remaining.begin()) % ranks;
        Cluster cluster;
        cluster.seed_point = globalPair[1];
        cluster.members.resize(globalPair[0]);
        if (rank == winner) {
            CUDA_CHECK(cudaMemcpy(dSeeds, &cluster.seed_point, sizeof(int), cudaMemcpyHostToDevice));
            growClusters<<<1, BLOCK_SIZE>>>(dPoints, dDistances, dClustered, dSeeds, n,
                                           threshold, dMaxima, dCounts, dMembers);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(cluster.members.data(), dMembers, globalPair[0] * sizeof(int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(cluster.members.data(), globalPair[0], MPI_INT, winner, MPI_COMM_WORLD);
        for (int member : cluster.members) clustered[member] = 1;
        clusters.push_back(std::move(cluster));
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                         [&clustered](int i) { return clustered[i] != 0; }), remaining.end());
    }
    CUDA_CHECK(cudaFree(dMaxima));
    if (dDistances) CUDA_CHECK(cudaFree(dDistances));
    CUDA_CHECK(cudaFree(dMembers));
    CUDA_CHECK(cudaFree(dCounts));
    CUDA_CHECK(cudaFree(dSeeds));
    CUDA_CHECK(cudaFree(dClustered));
    CUDA_CHECK(cudaFree(dPoints));
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide required thread support\n");
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two doubles");
    MPI_Datatype pointType;
    MPI_Type_contiguous(2, MPI_DOUBLE, &pointType);
    MPI_Type_commit(&pointType);
    MPI_Bcast(points.data(), num_points, pointType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&pointType);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
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

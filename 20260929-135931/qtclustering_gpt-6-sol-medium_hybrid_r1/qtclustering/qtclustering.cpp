// QT Clustering Benchmark - MPI + OpenMP + CUDA
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
#include <cstdint>
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <cuda_runtime.h>
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

// A block owns one seed.  Its threads own disjoint candidate points and keep
// their current maximum distance in the row of scratch storage.  Each new
// member therefore requires only one new distance per candidate.
__global__ void growClusters(const Point* points, const unsigned char* clustered,
                             const int* seeds, int seedCount, int n, double threshold,
                             double* maxima, int* sizes, int* members) {
    const int block = blockIdx.x;
    if (block >= seedCount) return;
    const int tid = threadIdx.x;
    const int seed = seeds[block];
    double* row = maxima + static_cast<size_t>(block) * n;
    __shared__ double bestDistance[256];
    __shared__ int bestIndex[256];
    __shared__ int chosen;
    __shared__ int size;
    if (tid == 0) {
        chosen = seed;
        size = 1;
        if (members) members[0] = seed;
    }
    __syncthreads();

    while (true) {
        const Point last = points[chosen];
        double localDistance = DBL_MAX;
        int localIndex = INT_MAX;
        for (int candidate = tid; candidate < n; candidate += blockDim.x) {
            double d = (size == 1) ? 0.0 : row[candidate];
            if (clustered[candidate] || candidate == chosen || d >= threshold) {
                row[candidate] = DBL_MAX;
                continue;
            }
            const double dx = points[candidate].x - last.x;
            const double dy = points[candidate].y - last.y;
            const double dist = sqrt(dx * dx + dy * dy);
            if (dist > d) d = dist;
            row[candidate] = d;
            if (d < threshold && (d < localDistance ||
                                  (d == localDistance && candidate < localIndex))) {
                localDistance = d;
                localIndex = candidate;
            }
        }
        bestDistance[tid] = localDistance;
        bestIndex[tid] = localIndex;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset &&
                (bestDistance[tid + offset] < bestDistance[tid] ||
                 (bestDistance[tid + offset] == bestDistance[tid] &&
                  bestIndex[tid + offset] < bestIndex[tid]))) {
                bestDistance[tid] = bestDistance[tid + offset];
                bestIndex[tid] = bestIndex[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            chosen = bestIndex[0];
            if (chosen != INT_MAX) {
                if (members) members[size] = chosen;
                ++size;
            } else {
                sizes[block] = size;
            }
        }
        __syncthreads();
        if (chosen == INT_MAX) break;
    }
}

static void checkCuda(cudaError_t status, MPI_Comm comm, const char* action) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, action, cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const int n = static_cast<int>(points.size());
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), comm, "detecting CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(comm, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    checkCuda(cudaSetDevice(localRank % deviceCount), comm, "selecting CUDA device");

    Point* dPoints = nullptr;
    unsigned char* dClustered = nullptr;
    int *dSeeds = nullptr, *dSizes = nullptr, *dMembers = nullptr;
    double* dMaxima = nullptr;
    checkCuda(cudaMalloc(&dPoints, sizeof(Point) * n), comm, "allocating points");
    checkCuda(cudaMalloc(&dClustered, n), comm, "allocating cluster flags");
    checkCuda(cudaMalloc(&dMembers, sizeof(int) * n), comm, "allocating members");
    checkCuda(cudaMemcpy(dPoints, points.data(), sizeof(Point) * n,
                         cudaMemcpyHostToDevice), comm, "copying points");

    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), comm, "querying GPU memory");
    // Bound the per-seed state to one quarter of currently free GPU memory.
    const size_t batchCapacity = std::min(static_cast<size_t>((n + ranks - 1) / ranks),
        std::min(static_cast<size_t>(65535), freeBytes / (4 * (sizeof(double) * n + sizeof(int) * 2))));
    if (batchCapacity == 0) {
        fprintf(stderr, "Rank %d: insufficient GPU memory for one seed\n", rank);
        MPI_Abort(comm, 1);
    }
    checkCuda(cudaMalloc(&dSeeds, sizeof(int) * batchCapacity), comm, "allocating seeds");
    checkCuda(cudaMalloc(&dSizes, sizeof(int) * batchCapacity), comm, "allocating sizes");
    checkCuda(cudaMalloc(&dMaxima, sizeof(double) * batchCapacity * n), comm,
              "allocating candidate state");

    std::vector<unsigned char> clustered(n, 0);
    std::vector<int> localSeeds, sizes(batchCapacity), winnerMembers(n);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        localSeeds.clear();
        for (int seed = rank; seed < n; seed += ranks)
            if (!clustered[seed]) localSeeds.push_back(seed);
        checkCuda(cudaMemcpy(dClustered, clustered.data(), n, cudaMemcpyHostToDevice),
                  comm, "copying cluster flags");

        unsigned long long localBest = 0;
        for (size_t start = 0; start < localSeeds.size(); start += batchCapacity) {
            const int count = static_cast<int>(std::min(batchCapacity, localSeeds.size() - start));
            checkCuda(cudaMemcpy(dSeeds, localSeeds.data() + start, sizeof(int) * count,
                                 cudaMemcpyHostToDevice), comm, "copying seeds");
            growClusters<<<count, 256>>>(dPoints, dClustered, dSeeds, count, n,
                                          threshold, dMaxima, dSizes, nullptr);
            checkCuda(cudaGetLastError(), comm, "launching candidate clusters");
            checkCuda(cudaMemcpy(sizes.data(), dSizes, sizeof(int) * count,
                                 cudaMemcpyDeviceToHost), comm, "copying cluster sizes");
            // Packed key maximizes cardinality, then minimizes the seed index.
#pragma omp parallel for reduction(max:localBest)
            for (int i = 0; i < count; ++i) {
                const unsigned long long key =
                    (static_cast<unsigned long long>(sizes[i]) << 32) |
                    static_cast<unsigned int>(UINT_MAX - localSeeds[start + i]);
                if (key > localBest) localBest = key;
            }
        }
        unsigned long long globalBest = 0;
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        const int bestSeed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(globalBest));
        const int bestSize = static_cast<int>(globalBest >> 32);
        if (bestSize <= 0) break;

        const int owner = bestSeed % ranks;
        if (rank == owner) {
            checkCuda(cudaMemcpy(dSeeds, &bestSeed, sizeof(int), cudaMemcpyHostToDevice),
                      comm, "copying winning seed");
            growClusters<<<1, 256>>>(dPoints, dClustered, dSeeds, 1, n,
                                      threshold, dMaxima, dSizes, dMembers);
            checkCuda(cudaGetLastError(), comm, "launching winning cluster");
            checkCuda(cudaMemcpy(winnerMembers.data(), dMembers, sizeof(int) * bestSize,
                                 cudaMemcpyDeviceToHost), comm, "copying winning members");
        }
        MPI_Bcast(winnerMembers.data(), bestSize, MPI_INT, owner, comm);
        for (int i = 0; i < bestSize; ++i) clustered[winnerMembers[i]] = 1;
        remaining -= bestSize;
        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = bestSeed;
            cluster.members.assign(winnerMembers.begin(), winnerMembers.begin() + bestSize);
            clusters.push_back(std::move(cluster));
        }
    }

    cudaFree(dMaxima);
    cudaFree(dMembers);
    cudaFree(dSizes);
    cudaFree(dSeeds);
    cudaFree(dClustered);
    cudaFree(dPoints);
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
    MPI_Init(&argc, &argv);
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
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
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)), MPI_BYTE,
              0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD);
    
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

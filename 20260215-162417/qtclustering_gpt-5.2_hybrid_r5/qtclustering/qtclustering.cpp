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

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void distance_matrix_kernel(const Point* __restrict__ points,
                                      double* __restrict__ dist,
                                      int N) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    dist[(size_t)i * (size_t)N + (size_t)j] = sqrt(dx * dx + dy * dy);
}

static std::vector<double> computeDistanceMatrixCUDA(const std::vector<Point>& points, int rank) {
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int device = rank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    const int N = static_cast<int>(points.size());
    std::vector<double> hostDist((size_t)N * (size_t)N);

    Point* d_points = nullptr;
    double* d_dist = nullptr;
    cudaCheck(cudaMalloc((void**)&d_points, (size_t)N * sizeof(Point)), "cudaMalloc points");
    cudaCheck(cudaMalloc((void**)&d_dist, (size_t)N * (size_t)N * sizeof(double)), "cudaMalloc dist");

    cudaCheck(cudaMemcpy(d_points, points.data(), (size_t)N * sizeof(Point), cudaMemcpyHostToDevice), "cudaMemcpy points");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    distance_matrix_kernel<<<grid, block>>>(d_points, d_dist, N);
    cudaCheck(cudaGetLastError(), "distance_matrix_kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "distance_matrix_kernel sync");

    cudaCheck(cudaMemcpy(hostDist.data(), d_dist, (size_t)N * (size_t)N * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy dist");

    cudaCheck(cudaFree(d_points), "cudaFree points");
    cudaCheck(cudaFree(d_dist), "cudaFree dist");
    return hostDist;
}

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

struct ThreadScratch {
    std::vector<int> in_cluster_mark;
    int mark_id = 1;
    std::vector<int> members;
};

static inline void scratchInit(ThreadScratch& s, int N) {
    if ((int)s.in_cluster_mark.size() != N) s.in_cluster_mark.assign(N, 0);
    s.members.clear();
}

static inline void scratchNextSeed(ThreadScratch& s) {
    s.members.clear();
    ++s.mark_id;
    if (s.mark_id == std::numeric_limits<int>::max()) {
        std::fill(s.in_cluster_mark.begin(), s.in_cluster_mark.end(), 0);
        s.mark_id = 1;
    }
}

static inline bool isInCluster(const ThreadScratch& s, int idx) {
    return s.in_cluster_mark[idx] == s.mark_id;
}

static inline void markInCluster(ThreadScratch& s, int idx) {
    s.in_cluster_mark[idx] = s.mark_id;
}

static inline double distFromMatrix(const double* dist, int N, int a, int b) {
    return dist[(size_t)a * (size_t)N + (size_t)b];
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
    return clusters;
}

static int findClosestPointHybrid(const std::vector<int>& cluster_members,
                                 const std::vector<unsigned char>& clustered,
                                 ThreadScratch& scratch,
                                 const double* dist,
                                 const double threshold,
                                 const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || isInCluster(scratch, candidate)) continue;

        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double d = distFromMatrix(dist, point_count, candidate, member);
            max_dist = std::max(max_dist, d);
        }

        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

static int generateCandidateClusterHybrid(const int seed_point,
                                         const std::vector<unsigned char>& clustered,
                                         const double* dist,
                                         const double threshold,
                                         const int point_count,
                                         ThreadScratch& scratch,
                                         std::vector<int>* cluster_members = nullptr) {
    scratchNextSeed(scratch);

    markInCluster(scratch, seed_point);
    scratch.members.push_back(seed_point);

    while ((int)scratch.members.size() < point_count) {
        const int closest = findClosestPointHybrid(scratch.members, clustered, scratch, dist, threshold, point_count);
        if (closest < 0) break;
        markInCluster(scratch, closest);
        scratch.members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = scratch.members;
    }

    return (int)scratch.members.size();
}

static std::vector<Cluster> qtClusteringHybridMPI(const std::vector<Point>& points,
                                                 const double threshold,
                                                 MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int N = (int)points.size();
    const std::vector<double> dist = computeDistanceMatrixCUDA(points, rank);

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        struct Best { int card; int order; int seed; };
        Best local{-1, std::numeric_limits<int>::max(), -1};

        const size_t total = unclustered_indices.size();
        const size_t start = (total * (size_t)rank) / (size_t)size;
        const size_t end   = (total * (size_t)(rank + 1)) / (size_t)size;

        #pragma omp parallel
        {
            ThreadScratch scratch;
            scratchInit(scratch, N);
            scratch.members.reserve(N);
            Best thread{-1, std::numeric_limits<int>::max(), -1};

            #pragma omp for schedule(static)
            for (size_t oi = start; oi < end; ++oi) {
                const int seed = unclustered_indices[oi];
                if (clustered[seed]) continue;

                const int card = generateCandidateClusterHybrid(seed, clustered, dist.data(), threshold, N, scratch, nullptr);
                const int order = (int)oi;
                if (card > thread.card || (card == thread.card && order < thread.order)) {
                    thread = {card, order, seed};
                }
            }

            #pragma omp critical
            {
                if (thread.card > local.card || (thread.card == local.card && thread.order < local.order)) {
                    local = thread;
                }
            }
        }

        int send[3] = {local.card, local.order, local.seed};
        std::vector<int> recv(3 * (size_t)size);
        MPI_Allgather(send, 3, MPI_INT, recv.data(), 3, MPI_INT, comm);

        int best_card = -1;
        int best_order = std::numeric_limits<int>::max();
        int best_seed = -1;
        for (int r = 0; r < size; ++r) {
            const int card = recv[3 * r + 0];
            const int order = recv[3 * r + 1];
            const int seed = recv[3 * r + 2];
            if (card > best_card || (card == best_card && order < best_order)) {
                best_card = card;
                best_order = order;
                best_seed = seed;
            }
        }

        // All ranks compute the same best_{seed,card}; broadcast for safety.
        MPI_Bcast(&best_seed, 1, MPI_INT, 0, comm);
        MPI_Bcast(&best_card, 1, MPI_INT, 0, comm);

        if (best_seed < 0 || best_card <= 0) break;

        std::vector<int> best_members;
        int member_count = 0;
        if (rank == 0) {
            ThreadScratch scratch;
            scratchInit(scratch, N);
            scratch.members.reserve(N);
            generateCandidateClusterHybrid(best_seed, clustered, dist.data(), threshold, N, scratch, &best_members);
            member_count = (int)best_members.size();
        }

        MPI_Bcast(&member_count, 1, MPI_INT, 0, comm);
        if (rank != 0) best_members.resize((size_t)member_count);
        MPI_Bcast(best_members.data(), member_count, MPI_INT, 0, comm);

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(std::move(cluster));
        }

        for (int idx : best_members) clustered[idx] = 1;

        // Compact unclustered list (deterministic order).
        size_t write = 0;
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int idx = unclustered_indices[i];
            if (!clustered[idx]) unclustered_indices[write++] = idx;
        }
        unclustered_indices.resize(write);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int badArgs = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                badArgs = 1;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        }

        if (badArgs) {
            printUsage(argv[0]);
        }

        if (!showHelp && !badArgs && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            badArgs = 1;
        }
    }

    int iparams[4] = {num_points, validate, printResults, badArgs ? 1 : 0};
    MPI_Bcast(iparams, 4, MPI_INT, 0, MPI_COMM_WORLD);
    num_points = iparams[0];
    validate = iparams[1];
    printResults = iparams[2];
    badArgs = iparams[3];
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (showHelp || badArgs) {
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI=%d ranks, OpenMP enabled, CUDA distance matrix)\n", size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points((size_t)num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), (int)((size_t)num_points * sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClusteringHybridMPI(points, threshold, MPI_COMM_WORLD);

    const double t1 = MPI_Wtime();
    const double local_time_sec = t1 - t0;
    double global_time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &global_time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;

    if (rank == 0) {
        const double ms = global_time_sec * 1000.0;
        printf("Clustering time: %ld ms\n", (long)ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = (int)clusters[i].members.size();
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 : (double)total_clustered / (double)clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * (double)total_clustered / (double)num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = global_time_sec;
        const double clusters_per_sec = time_sec > 0.0 ? (clusters.size() / time_sec) : 0.0;
        const double points_per_sec = time_sec > 0.0 ? ((double)num_points / time_sec) : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve((size_t)num_points);
            std::vector<int> membership((size_t)num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[clusters[c].members[i]] = (int)c;
                }
            }
            for (int m : membership) membershipData.push_back((double)m);
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            rc = ok ? 0 : 1;
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}

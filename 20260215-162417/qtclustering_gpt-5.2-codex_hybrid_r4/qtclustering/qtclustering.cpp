// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be two doubles");

struct CudaContext {
    Point* d_points = nullptr;
    int* d_members = nullptr;
    double* d_max_dists = nullptr;
    int point_count = 0;
    std::vector<double> h_max_dists;
};

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void maxDistanceKernel(const Point* points,
                                  int point_count,
                                  const int* members,
                                  int member_count,
                                  double* max_dists) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;
    const Point p = points[idx];
    double max_dist = 0.0;
    for (int i = 0; i < member_count; ++i) {
        const Point m = points[members[i]];
        const double dx = p.x - m.x;
        const double dy = p.y - m.y;
        const double dist = sqrt(dx * dx + dy * dy);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    max_dists[idx] = max_dist;
}

static void initCudaContext(CudaContext& ctx, const std::vector<Point>& points, int mpi_rank) {
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA devices detected.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = mpi_rank % device_count;
    cudaCheck(cudaSetDevice(device_id), "cudaSetDevice");

    ctx.point_count = static_cast<int>(points.size());
    ctx.h_max_dists.resize(ctx.point_count);

    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ctx.d_points),
                         ctx.point_count * sizeof(Point)),
              "cudaMalloc d_points");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ctx.d_members),
                         ctx.point_count * sizeof(int)),
              "cudaMalloc d_members");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ctx.d_max_dists),
                         ctx.point_count * sizeof(double)),
              "cudaMalloc d_max_dists");
    cudaCheck(cudaMemcpy(ctx.d_points, points.data(),
                         ctx.point_count * sizeof(Point),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy points");
}

static void freeCudaContext(CudaContext& ctx) {
    if (ctx.d_points) cudaCheck(cudaFree(ctx.d_points), "cudaFree d_points");
    if (ctx.d_members) cudaCheck(cudaFree(ctx.d_members), "cudaFree d_members");
    if (ctx.d_max_dists) cudaCheck(cudaFree(ctx.d_max_dists), "cudaFree d_max_dists");
    ctx.d_points = nullptr;
    ctx.d_members = nullptr;
    ctx.d_max_dists = nullptr;
    ctx.point_count = 0;
    ctx.h_max_dists.clear();
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const double threshold,
                     const int point_count,
                     CudaContext& ctx) {
    if (cluster_members.empty()) {
        return -1;
    }

    const int member_count = static_cast<int>(cluster_members.size());
    cudaCheck(cudaMemcpy(ctx.d_members, cluster_members.data(),
                         member_count * sizeof(int),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy members");

    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    maxDistanceKernel<<<blocks, threads>>>(ctx.d_points, point_count,
                                           ctx.d_members, member_count,
                                           ctx.d_max_dists);
    cudaCheck(cudaGetLastError(), "maxDistanceKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "maxDistanceKernel sync");
    cudaCheck(cudaMemcpy(ctx.h_max_dists.data(), ctx.d_max_dists,
                         point_count * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy max_dists");

    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int local_best = -1;
        double local_min = min_diameter;
        #pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            const double max_dist = ctx.h_max_dists[candidate];
            if (max_dist < threshold &&
                (max_dist < local_min ||
                 (max_dist == local_min && candidate < local_best))) {
                local_min = max_dist;
                local_best = candidate;
            }
        }
        #pragma omp critical
        {
            if (local_best >= 0 &&
                (closest_point < 0 ||
                 local_min < min_diameter ||
                 (local_min == min_diameter && local_best < closest_point))) {
                min_diameter = local_min;
                closest_point = local_best;
            }
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const double threshold,
                              const int point_count,
                              CudaContext& ctx,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster,
                                             threshold, point_count, ctx);
        
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
                                  const double threshold,
                                  const int mpi_rank,
                                  const int mpi_size,
                                  CudaContext& ctx) {
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
        int local_best_cardinality = -1;
        int local_best_seed = -1;
        int local_best_pos = std::numeric_limits<int>::max();

        // Try each unclustered point as a seed (distributed by MPI rank)
        for (size_t pos = static_cast<size_t>(mpi_rank);
             pos < unclustered_indices.size();
             pos += static_cast<size_t>(mpi_size)) {
            const int seed = unclustered_indices[pos];
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(seed, clustered,
                                                             threshold, N, ctx, nullptr);

            const int pos_int = static_cast<int>(pos);
            if (cardinality > local_best_cardinality ||
                (cardinality == local_best_cardinality && pos_int < local_best_pos)) {
                local_best_cardinality = cardinality;
                local_best_seed = seed;
                local_best_pos = pos_int;
            }
        }

        std::array<int, 4> local_info = {
            local_best_cardinality,
            local_best_pos,
            local_best_seed,
            mpi_rank
        };
        std::vector<int> gathered(4 * mpi_size, 0);
        MPI_Allgather(local_info.data(), 4, MPI_INT,
                      gathered.data(), 4, MPI_INT, MPI_COMM_WORLD);

        int best_cardinality = -1;
        int best_seed = -1;
        int best_pos = std::numeric_limits<int>::max();
        int best_rank = 0;
        for (int r = 0; r < mpi_size; ++r) {
            const int card = gathered[4 * r];
            const int pos = gathered[4 * r + 1];
            const int seed = gathered[4 * r + 2];
            const int rank = gathered[4 * r + 3];
            if (card > best_cardinality ||
                (card == best_cardinality && pos < best_pos)) {
                best_cardinality = card;
                best_seed = seed;
                best_pos = pos;
                best_rank = rank;
            }
        }

        if (best_seed < 0 || best_cardinality <= 0) {
            break;
        }

        std::vector<int> best_cluster_members;
        if (mpi_rank == best_rank) {
            generateCandidateCluster(best_seed, clustered,
                                     threshold, N, ctx, &best_cluster_members);
        }

        int member_count = static_cast<int>(best_cluster_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, best_rank, MPI_COMM_WORLD);
        if (mpi_rank != best_rank) {
            best_cluster_members.resize(member_count);
        }
        if (member_count > 0) {
            MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT,
                      best_rank, MPI_COMM_WORLD);
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(cluster);

        for (size_t i = 0; i < best_cluster_members.size(); ++i) {
            clustered[best_cluster_members[i]] = true;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
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
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const bool is_root = (mpi_rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool parse_error = false;
    
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
            show_help = true;
        } else {
            parse_error = true;
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (show_help) {
        if (is_root) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (parse_error) {
        if (is_root) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (is_root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (is_root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (is_root) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(reinterpret_cast<double*>(points.data()),
              num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CudaContext cuda_ctx;
    initCudaContext(cuda_ctx, points, mpi_rank);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                       mpi_rank, mpi_size,
                                                       cuda_ctx);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    freeCudaContext(cuda_ctx);
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (is_root) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }
    
    // Calculate statistics and performance metrics
    if (is_root) {
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
    }
    
    // Print results for external validation
    if (printResults && is_root) {
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
    int exit_code = 0;
    if (validate) {
        if (is_root) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}

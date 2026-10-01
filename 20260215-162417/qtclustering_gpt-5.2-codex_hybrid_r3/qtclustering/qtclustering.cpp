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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;

static int g_mpi_rank = 0;
static int g_mpi_size = 1;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static void checkCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error on rank %d (%s): %s\n",
                g_mpi_rank, context, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void maxDistancesKernel(const Point* points,
                                   const int* members,
                                   int member_count,
                                   int point_count,
                                   double* out_max) {
    const int candidate = blockIdx.x;
    if (candidate >= point_count) {
        return;
    }
    double local_max = 0.0;
    for (int i = threadIdx.x; i < member_count; i += blockDim.x) {
        const int member = members[i];
        const double dx = points[candidate].x - points[member].x;
        const double dy = points[candidate].y - points[member].y;
        const double dist = sqrt(dx * dx + dy * dy);
        if (dist > local_max) {
            local_max = dist;
        }
    }
    extern __shared__ double sdata[];
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double other = sdata[threadIdx.x + stride];
            if (other > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_max[candidate] = sdata[0];
    }
}

struct CudaDistanceHelper {
    Point* d_points = nullptr;
    int* d_members = nullptr;
    double* d_max_dists = nullptr;
    int point_count = 0;
    std::vector<double> h_max_dists;

    void initialize(const std::vector<Point>& points) {
        point_count = static_cast<int>(points.size());
        if (point_count <= 0) {
            return;
        }
        h_max_dists.resize(point_count);
        checkCuda(cudaMalloc(&d_points, point_count * sizeof(Point)), "cudaMalloc d_points");
        checkCuda(cudaMemcpy(d_points, points.data(),
                             point_count * sizeof(Point),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy points");
        checkCuda(cudaMalloc(&d_members, point_count * sizeof(int)), "cudaMalloc d_members");
        checkCuda(cudaMalloc(&d_max_dists, point_count * sizeof(double)), "cudaMalloc d_max_dists");
    }

    const std::vector<double>& computeMaxDistances(const std::vector<int>& members,
                                                    int member_count) {
        if (point_count <= 0) {
            return h_max_dists;
        }
        if (member_count <= 0) {
            std::fill(h_max_dists.begin(), h_max_dists.end(), 0.0);
            return h_max_dists;
        }
        checkCuda(cudaMemcpy(d_members, members.data(),
                             member_count * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy members");
        const dim3 block(CUDA_THREADS);
        const dim3 grid(point_count);
        const size_t shared_bytes = CUDA_THREADS * sizeof(double);
        maxDistancesKernel<<<grid, block, shared_bytes>>>(d_points,
                                                          d_members,
                                                          member_count,
                                                          point_count,
                                                          d_max_dists);
        checkCuda(cudaGetLastError(), "maxDistancesKernel launch");
        checkCuda(cudaMemcpy(h_max_dists.data(), d_max_dists,
                             point_count * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy max distances");
        return h_max_dists;
    }

    void cleanup() {
        if (d_points) {
            cudaFree(d_points);
            d_points = nullptr;
        }
        if (d_members) {
            cudaFree(d_members);
            d_members = nullptr;
        }
        if (d_max_dists) {
            cudaFree(d_max_dists);
            d_max_dists = nullptr;
        }
        point_count = 0;
        h_max_dists.clear();
    }
};

static CudaDistanceHelper g_cuda;

static void initializeCudaForRank() {
    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", g_mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = g_mpi_rank % device_count;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
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
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    static_cast<void>(points);
    if (cluster_members.empty()) {
        return -1;
    }
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    const auto& max_dists = g_cuda.computeMaxDistances(
        cluster_members, static_cast<int>(cluster_members.size()));

    #pragma omp parallel
    {
        int local_best = -1;
        double local_min = std::numeric_limits<double>::max();

        #pragma omp for nowait schedule(static)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) {
                continue;
            }
            const double max_dist = max_dists[candidate];
            if (max_dist < threshold && max_dist < local_min) {
                local_min = max_dist;
                local_best = candidate;
            }
        }

        #pragma omp critical
        {
            if (local_best >= 0) {
                if (local_min < min_diameter ||
                    (local_min == min_diameter &&
                     (closest_point < 0 || local_best < closest_point))) {
                    min_diameter = local_min;
                    closest_point = local_best;
                }
            }
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
                                  const double threshold,
                                  MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
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
        int best_pos = std::numeric_limits<int>::max();
        
        // Try each unclustered point as a seed
        const int total_seeds = static_cast<int>(unclustered_indices.size());
        for (int i = rank; i < total_seeds; i += size) {
            const int seed = unclustered_indices[static_cast<size_t>(i)];
            if (clustered[seed]) continue;
            
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N, 
                                                       nullptr);
            
            if (cardinality > max_cardinality ||
                (cardinality == max_cardinality && i < best_pos)) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_pos = i;
            }
        }

        int local_data[3] = {max_cardinality, best_pos, best_seed};
        std::vector<int> all_data(3 * size, -1);
        MPI_Allgather(local_data, 3, MPI_INT,
                      all_data.data(), 3, MPI_INT, comm);

        if (rank == 0) {
            max_cardinality = -1;
            best_seed = -1;
            best_pos = std::numeric_limits<int>::max();
            for (int r = 0; r < size; ++r) {
                const int card = all_data[3 * r];
                const int pos = all_data[3 * r + 1];
                const int seed = all_data[3 * r + 2];
                if (card > max_cardinality ||
                    (card == max_cardinality && pos < best_pos)) {
                    max_cardinality = card;
                    best_pos = pos;
                    best_seed = seed;
                }
            }
        }

        MPI_Bcast(&best_seed, 1, MPI_INT, 0, comm);
        MPI_Bcast(&max_cardinality, 1, MPI_INT, 0, comm);
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            std::vector<int> best_cluster_members;
            if (rank == 0) {
                generateCandidateCluster(best_seed, clustered, points,
                                         threshold, N,
                                         &best_cluster_members);
            }
            int best_size = rank == 0 ? static_cast<int>(best_cluster_members.size()) : 0;
            MPI_Bcast(&best_size, 1, MPI_INT, 0, comm);
            best_cluster_members.resize(static_cast<size_t>(best_size));
            MPI_Bcast(best_cluster_members.data(), best_size, MPI_INT, 0, comm);

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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int parse_ok = 1;
    int exit_code = 0;
    
    // Parse command line arguments
    if (g_mpi_rank == 0) {
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
                printUsage(argv[0]);
                parse_ok = 0;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_ok = 0;
                exit_code = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validate_flag = validate ? 1 : 0;
    int results_flag = printResults ? 1 : 0;
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&results_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_flag != 0;
    printResults = results_flag != 0;
    
    int params_ok = (num_points > 0 && threshold > 0.0) ? 1 : 0;
    if (g_mpi_rank == 0 && !params_ok) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
    }
    MPI_Bcast(&params_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!params_ok) {
        MPI_Finalize();
        return 1;
    }
    
    if (g_mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", g_mpi_size, omp_get_max_threads());
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (g_mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    initializeCudaForRank();
    g_cuda.initialize(points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_ms = cluster_time.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (g_mpi_rank == 0) {
        cluster_time = std::chrono::milliseconds(max_ms);
    }
    
    if (g_mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }
    
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
    
    if (g_mpi_rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
    }
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    if (g_mpi_rank == 0) {
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && g_mpi_rank == 0) {
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
    int valid_flag = 1;
    if (validate && g_mpi_rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);
        valid_flag = valid ? 1 : 0;
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    g_cuda.cleanup();
    MPI_Finalize();
    if (validate) {
        return valid_flag ? 0 : 1;
    }
    return 0;
}

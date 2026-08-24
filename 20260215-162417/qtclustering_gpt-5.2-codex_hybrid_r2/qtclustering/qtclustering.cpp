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
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CHECK_MPI(call)                                                      \
    do {                                                                     \
        const int mpi_status = (call);                                       \
        if (mpi_status != MPI_SUCCESS) {                                     \
            fprintf(stderr, "MPI error %d at %s:%d\n",                        \
                    mpi_status, __FILE__, __LINE__);                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        const cudaError_t cuda_status = (call);                              \
        if (cuda_status != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(cuda_status));    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

inline size_t distIndex(const int i, const int j, const int n) {
    return static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j);
}

__global__ void computeDistanceMatrixKernel(const Point* points, const int n, double* dist_matrix) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) {
        return;
    }
    const double dx = points[i].x - points[j].x;
    const double dy = points[i].y - points[j].y;
    dist_matrix[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
        sqrt(dx * dx + dy * dy);
}

void setCudaDeviceForRank(const int mpi_rank) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "CUDA device query failed or no devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = mpi_rank % device_count;
    CHECK_CUDA(cudaSetDevice(device_id));
}

void computeDistanceMatrixCUDA(const std::vector<Point>& points,
                               std::vector<double>& dist_matrix) {
    const int n = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    dist_matrix.assign(matrix_size, 0.0);

    if (matrix_size == 0) {
        return;
    }

    Point* d_points = nullptr;
    double* d_matrix = nullptr;
    const size_t points_bytes = static_cast<size_t>(n) * sizeof(Point);
    const size_t matrix_bytes = matrix_size * sizeof(double);

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_points), points_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_matrix), matrix_bytes));
    CHECK_CUDA(cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    computeDistanceMatrixKernel<<<grid, block>>>(d_points, n, d_matrix);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(dist_matrix.data(), d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_matrix));
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<uint8_t>& clustered,
                     const std::vector<uint8_t>& in_cluster,
                     const double* dist_matrix,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int local_best = -1;
        double local_best_dist = std::numeric_limits<double>::max();

        #pragma omp for schedule(static)
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) {
                continue;
            }

            double max_dist = 0.0;
            for (size_t i = 0; i < cluster_members.size(); ++i) {
                const int member = cluster_members[i];
                const double dist =
                    dist_matrix[distIndex(candidate, member, point_count)];
                if (dist > max_dist) {
                    max_dist = dist;
                    if (max_dist >= threshold) {
                        break;
                    }
                }
            }

            if (max_dist < threshold) {
                if (max_dist < local_best_dist ||
                    (max_dist == local_best_dist &&
                     (local_best < 0 || candidate < local_best))) {
                    local_best_dist = max_dist;
                    local_best = candidate;
                }
            }
        }

        #pragma omp critical
        {
            if (local_best >= 0 &&
                (local_best_dist < min_diameter ||
                 (local_best_dist == min_diameter &&
                  (closest_point < 0 || local_best < closest_point)))) {
                min_diameter = local_best_dist;
                closest_point = local_best;
            }
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<uint8_t>& clustered,
                              const double* dist_matrix,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<uint8_t> in_cluster(point_count, 0);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, dist_matrix,
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = 1;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI + OpenMP, distances precomputed on GPU)
std::vector<Cluster> qtClustering(const double* dist_matrix,
                                  const double threshold,
                                  const int point_count,
                                  const int mpi_rank,
                                  const int mpi_size) {
    std::vector<uint8_t> clustered(point_count, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < point_count; ++i) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        int local_best_seed = -1;
        int local_best_cardinality = -1;
        std::vector<int> local_best_members;

        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (static_cast<int>(i % mpi_size) != mpi_rank) {
                continue;
            }

            const int seed = unclustered_indices[i];
            if (clustered[seed]) {
                continue;
            }

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, dist_matrix,
                                                             threshold, point_count,
                                                             &candidate_members);
            if (cardinality > local_best_cardinality ||
                (cardinality == local_best_cardinality &&
                 (local_best_seed < 0 || seed < local_best_seed))) {
                local_best_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        int local_info[2] = {local_best_cardinality, local_best_seed};
        std::vector<int> all_info(static_cast<size_t>(mpi_size) * 2);
        CHECK_MPI(MPI_Allgather(local_info, 2, MPI_INT,
                                all_info.data(), 2, MPI_INT,
                                MPI_COMM_WORLD));

        int best_seed = -1;
        int best_cardinality = -1;
        int best_owner = -1;

        if (mpi_rank == 0) {
            for (int r = 0; r < mpi_size; ++r) {
                const int card = all_info[static_cast<size_t>(r) * 2];
                const int seed = all_info[static_cast<size_t>(r) * 2 + 1];
                if (seed < 0 || card < 0) {
                    continue;
                }
                if (card > best_cardinality ||
                    (card == best_cardinality &&
                     (best_seed < 0 || seed < best_seed))) {
                    best_cardinality = card;
                    best_seed = seed;
                    best_owner = r;
                }
            }
        }

        CHECK_MPI(MPI_Bcast(&best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Bcast(&best_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Bcast(&best_owner, 1, MPI_INT, 0, MPI_COMM_WORLD));

        if (best_seed < 0 || best_cardinality <= 0) {
            break;
        }

        std::vector<int> best_cluster_members;
        if (mpi_rank == best_owner) {
            if (local_best_seed == best_seed && !local_best_members.empty()) {
                best_cluster_members = std::move(local_best_members);
            } else {
                generateCandidateCluster(best_seed, clustered, dist_matrix, threshold,
                                         point_count, &best_cluster_members);
            }
        }

        int member_count = static_cast<int>(best_cluster_members.size());
        CHECK_MPI(MPI_Bcast(&member_count, 1, MPI_INT, best_owner, MPI_COMM_WORLD));
        if (mpi_rank != best_owner) {
            best_cluster_members.resize(member_count);
        }
        if (member_count > 0) {
            CHECK_MPI(MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT,
                                best_owner, MPI_COMM_WORLD));
        }

        for (int member : best_cluster_members) {
            clustered[member] = 1;
        }

        if (mpi_rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const double* dist_matrix,
                      const int point_count,
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
                const int member_i = cluster.members[i];
                const int member_j = cluster.members[j];
                const double dist = dist_matrix[distIndex(member_i, member_j, point_count)];
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
    std::vector<int> membership(static_cast<size_t>(point_count), -1);
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
    
    printf("Total points: %d, Clustered: %d, Unclustered: %d\n",
           point_count, clustered_count, point_count - clustered_count);
    
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
    const int mpi_init_status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (mpi_init_status != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization failed.\n");
        return 1;
    }

    int mpi_rank = 0;
    int mpi_size = 1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    if (provided < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            fprintf(stderr, "MPI does not provide required threading level.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_dynamic(0);
    omp_set_nested(0);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;
    int should_exit = 0;
    int exit_code = 0;

    if (mpi_rank == 0) {
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
                printUsage(argv[0]);
                should_exit = 1;
                exit_code = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                should_exit = 1;
                exit_code = 1;
            }
        }

        if (!should_exit && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            should_exit = 1;
            exit_code = 1;
        }
    }

    CHECK_MPI(MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (should_exit) {
        MPI_Finalize();
        return exit_code;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI/OpenMP/CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    std::vector<Point> points(static_cast<size_t>(num_points));
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
    }

    const size_t point_bytes = static_cast<size_t>(num_points) * sizeof(Point);
    if (point_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (mpi_rank == 0) {
            fprintf(stderr, "Point buffer too large for MPI broadcast.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_MPI(MPI_Bcast(points.data(), static_cast<int>(point_bytes),
                        MPI_BYTE, 0, MPI_COMM_WORLD));

    setCudaDeviceForRank(mpi_rank);

    std::vector<double> dist_matrix;
    computeDistanceMatrixCUDA(points, dist_matrix);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto cluster_start = std::chrono::high_resolution_clock::now();
    std::vector<Cluster> clusters = qtClustering(dist_matrix.data(), threshold,
                                                 num_points, mpi_rank, mpi_size);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start).count());
    long long cluster_time_ms = 0;
    CHECK_MPI(MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1,
                         MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD));

    if (mpi_rank == 0) {
        printf("Clustering time: %lld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

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

        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
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

        if (validate) {
            const bool valid = validateClusters(clusters, dist_matrix.data(),
                                                num_points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    CHECK_MPI(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}

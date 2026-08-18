// QT Clustering Benchmark - hybrid MPI/OpenMP/CUDA version
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
#include <cfloat>
#include <climits>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// One block independently constructs one candidate cluster.  Candidate
// distances are distributed across the block; the lexicographic reduction
// preserves the sequential algorithm's lowest-index tie breaking.
__global__ void candidateClustersKernel(const Point* points,
                                        const std::uint8_t* clustered,
                                        const int* seeds, int seed_count,
                                        int point_count, double threshold,
                                        int* cardinalities, int* all_members) {
    const int row = blockIdx.x;
    if (row >= seed_count) return;
    int* members = all_members + static_cast<std::size_t>(row) * point_count;
    __shared__ double best_dist[256];
    __shared__ int best_point[256];
    __shared__ int member_count;
    __shared__ int finished;

    if (threadIdx.x == 0) {
        members[0] = seeds[row];
        member_count = 1;
        finished = 0;
    }
    __syncthreads();

    while (!finished && member_count < point_count) {
        double thread_dist = DBL_MAX;
        int thread_point = INT_MAX;
        for (int candidate = threadIdx.x; candidate < point_count;
             candidate += blockDim.x) {
            if (clustered[candidate]) continue;
            bool already_member = false;
            double max_dist = 0.0;
            for (int j = 0; j < member_count; ++j) {
                const int member = members[j];
                if (member == candidate) {
                    already_member = true;
                    break;
                }
                const double dx = points[candidate].x - points[member].x;
                const double dy = points[candidate].y - points[member].y;
                max_dist = fmax(max_dist, sqrt(dx * dx + dy * dy));
            }
            if (!already_member && max_dist < threshold &&
                (max_dist < thread_dist ||
                 (max_dist == thread_dist && candidate < thread_point))) {
                thread_dist = max_dist;
                thread_point = candidate;
            }
        }
        best_dist[threadIdx.x] = thread_dist;
        best_point[threadIdx.x] = thread_point;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset; offset >>= 1) {
            if (threadIdx.x < offset) {
                const double other_dist = best_dist[threadIdx.x + offset];
                const int other_point = best_point[threadIdx.x + offset];
                if (other_dist < best_dist[threadIdx.x] ||
                    (other_dist == best_dist[threadIdx.x] &&
                     other_point < best_point[threadIdx.x])) {
                    best_dist[threadIdx.x] = other_dist;
                    best_point[threadIdx.x] = other_point;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (best_point[0] == INT_MAX) {
                finished = 1;
            } else {
                members[member_count++] = best_point[0];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[row] = member_count;
}

// Main QT clustering algorithm. Every rank holds the small global state while
// seed candidates are split between ranks and evaluated on rank-local GPUs.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank = 0, rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "A CUDA device is required on every MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(local_rank % device_count), "cudaSetDevice");

    Point* d_points = nullptr;
    std::uint8_t* d_clustered = nullptr;
    cudaCheck(cudaMalloc(&d_points, static_cast<std::size_t>(N) * sizeof(Point)), "points allocation");
    cudaCheck(cudaMalloc(&d_clustered, static_cast<std::size_t>(N)), "state allocation");
    cudaCheck(cudaMemcpy(d_points, points.data(), static_cast<std::size_t>(N) * sizeof(Point),
                         cudaMemcpyHostToDevice), "points upload");

    std::vector<std::uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    unclustered_indices.resize(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    while (!unclustered_indices.empty()) {
        const int local_count = (static_cast<int>(unclustered_indices.size()) +
                                 rank_count - 1 - rank) / rank_count;
        std::vector<int> local_seeds(local_count);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_count; ++i)
            local_seeds[i] = unclustered_indices[rank + i * rank_count];

        cudaCheck(cudaMemcpy(d_clustered, clustered.data(), static_cast<std::size_t>(N),
                             cudaMemcpyHostToDevice), "state upload");

        int local_best_cardinality = -1;
        int local_best_seed = INT_MAX;
        std::vector<int> local_best_members;
        if (local_count > 0) {
            std::size_t free_bytes = 0, total_bytes = 0;
            cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
            const std::size_t bytes_per_seed = static_cast<std::size_t>(N) * sizeof(int);
            const std::size_t memory_capacity = (free_bytes * 3 / 5) /
                                                std::max<std::size_t>(bytes_per_seed, 1);
            const int batch_capacity = std::max(1, static_cast<int>(std::min<std::size_t>(
                static_cast<std::size_t>(local_count), memory_capacity)));
            int *d_seeds = nullptr, *d_cards = nullptr, *d_members = nullptr;
            cudaCheck(cudaMalloc(&d_seeds, static_cast<std::size_t>(batch_capacity) * sizeof(int)), "seed allocation");
            cudaCheck(cudaMalloc(&d_cards, static_cast<std::size_t>(batch_capacity) * sizeof(int)), "cardinality allocation");
            cudaCheck(cudaMalloc(&d_members, static_cast<std::size_t>(batch_capacity) * bytes_per_seed), "member allocation");
            std::vector<int> cards(batch_capacity);

            for (int begin = 0; begin < local_count; begin += batch_capacity) {
                const int count = std::min(batch_capacity, local_count - begin);
                cudaCheck(cudaMemcpy(d_seeds, local_seeds.data() + begin,
                                     static_cast<std::size_t>(count) * sizeof(int),
                                     cudaMemcpyHostToDevice), "seed upload");
                candidateClustersKernel<<<count, 256>>>(d_points, d_clustered, d_seeds,
                    count, N, threshold, d_cards, d_members);
                cudaCheck(cudaGetLastError(), "candidate kernel launch");
                cudaCheck(cudaMemcpy(cards.data(), d_cards,
                                     static_cast<std::size_t>(count) * sizeof(int),
                                     cudaMemcpyDeviceToHost), "cardinality download");
                for (int i = 0; i < count; ++i) {
                    const int seed = local_seeds[begin + i];
                    if (cards[i] > local_best_cardinality ||
                        (cards[i] == local_best_cardinality && seed < local_best_seed)) {
                        local_best_cardinality = cards[i];
                        local_best_seed = seed;
                        local_best_members.resize(cards[i]);
                        cudaCheck(cudaMemcpy(local_best_members.data(),
                            d_members + static_cast<std::size_t>(i) * N,
                            static_cast<std::size_t>(cards[i]) * sizeof(int),
                            cudaMemcpyDeviceToHost), "winning members download");
                    }
                }
            }
            cudaFree(d_members);
            cudaFree(d_cards);
            cudaFree(d_seeds);
        }

        int local_pair[2] = {local_best_cardinality, local_best_seed};
        int global_pair[2] = {-1, INT_MAX};
        MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_cardinality = global_pair[0];
        const int best_seed = global_pair[1];
        if (best_cardinality <= 0) break;

        int owner = (local_best_seed == best_seed) ? rank : -1;
        MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        std::vector<int> best_members(best_cardinality);
        if (rank == owner) best_members = std::move(local_best_members);
        MPI_Bcast(best_members.data(), best_cardinality, MPI_INT, owner, MPI_COMM_WORLD);

        clusters.push_back({best_members, best_seed});
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < best_cardinality; ++i)
            clustered[best_members[i]] = 1;
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    cudaFree(d_clustered);
    cudaFree(d_points);
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
    int rank = 0;
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
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(time_sec * 1000.0));
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
    
    if (rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double safe_time = std::max(time_sec, std::numeric_limits<double>::min());
        const double clusters_per_sec = clusters.size() / safe_time;
        const double points_per_sec = num_points / safe_time;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
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
    int return_code = 0;
    if (validate && rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return_code = valid ? 0 : 1;
    }
    MPI_Bcast(&return_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return return_code;
}

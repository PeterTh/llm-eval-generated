// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// MPI Parallelization Strategy:
// - Replicate all points on every rank
// - At each clustering iteration, distribute seed evaluation across ranks
// - Each rank independently generates candidate clusters for its assigned seeds
// - Global reduction to find the best (largest) cluster with deterministic tie-breaking
// - Synchronize clustered status across all ranks
// - Repeat until convergence

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

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

// Generate synthetic 2D point data in clusters (only on rank 0, then broadcast)
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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
                     const std::vector<char>& clustered,
                     const std::vector<char>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            if (dist >= threshold) {
                // Early exit: candidate violates diameter constraint
                max_dist = threshold;
                break;
            }
            max_dist = std::max(max_dist, dist);
        }

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
                              const std::vector<char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members_out) {
    std::vector<char> in_cluster(point_count, 0);
    std::vector<int> members;
    members.reserve(point_count);

    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             threshold, point_count);
        if (closest < 0) break;

        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    if (cluster_members_out) {
        *cluster_members_out = std::move(members);
        return static_cast<int>(cluster_members_out->size());
    }

    return static_cast<int>(members.size());
}

// Find the best cluster among assigned seeds on this rank
// Returns {cardinality, seed_point, members}
static Cluster findLocalBestCluster(const std::vector<int>& local_seeds,
                                     const std::vector<char>& clustered,
                                     const std::vector<Point>& points,
                                     const double threshold,
                                     const int point_count) {
    Cluster best;
    best.seed_point = -1;
    int max_cardinality = -1;

    for (const int seed : local_seeds) {
        if (clustered[seed]) continue;

        std::vector<int> candidate_members;
        const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                         threshold, point_count,
                                                         &candidate_members);

        if (cardinality > max_cardinality) {
            max_cardinality = cardinality;
            best.seed_point = seed;
            best.members = std::move(candidate_members);
        }
    }

    return best;
}

// Main QT clustering algorithm - MPI parallelized
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                      const double threshold,
                                      int mpi_rank, int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    std::vector<Cluster> clusters;
    // Reserve to avoid reallocation
    clusters.reserve(N / 2);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Distribute seeds among MPI ranks
        const int num_seeds = static_cast<int>(unclustered_indices.size());
        const int base_count = num_seeds / mpi_size;
        const int remainder = num_seeds % mpi_size;
        const int my_count = base_count + (mpi_rank < remainder ? 1 : 0);
        const int my_start = mpi_rank * base_count + std::min(mpi_rank, remainder);

        std::vector<int> local_seeds;
        local_seeds.reserve(my_count);
        for (int i = 0; i < my_count; ++i) {
            local_seeds.push_back(unclustered_indices[my_start + i]);
        }

        // Each rank finds its local best cluster
        Cluster local_best = findLocalBestCluster(local_seeds, clustered, points,
                                                  threshold, N);

        // Two-phase global reduction:
        // Phase 1: Find the global maximum cardinality
        int local_cardinality = local_best.members.empty() ? -1 :
                                static_cast<int>(local_best.members.size());
        int global_max_cardinality;
        MPI_Allreduce(&local_cardinality, &global_max_cardinality, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_cardinality <= 0) {
            // No more clusters can be formed
            break;
        }

        // Phase 2: Among ranks with max cardinality, find the minimum seed point
        // (deterministic tie-breaking matching the sequential version)
        int local_seed = local_cardinality == global_max_cardinality ?
                         local_best.seed_point : N; // Use N as "infinity" for non-winners
        int global_best_seed;
        MPI_Allreduce(&local_seed, &global_best_seed, 1,
                      MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Only the rank that found the best cluster has the members.
        // Gather member count from the winning rank(s)
        int local_member_count = (local_best.seed_point == global_best_seed) ?
                                  static_cast<int>(local_best.members.size()) : 0;
        int global_member_count;
        MPI_Allreduce(&local_member_count, &global_member_count, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Broadcast the best cluster members from the winning rank.
        // Use mpi_size as sentinel for non-winning ranks (guaranteed > any valid rank).
        int root = mpi_size;
        if (local_best.seed_point == global_best_seed) {
            root = mpi_rank;
        }
        int root_found;
        MPI_Allreduce(&root, &root_found, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Broadcast members from the winning rank
        Cluster global_best;
        global_best.seed_point = global_best_seed;
        global_best.members.resize(global_member_count);

        if (mpi_rank == root_found) {
            global_best.members = local_best.members;
        }

        MPI_Bcast(global_best.members.data(), global_member_count, MPI_INT,
                  root_found, MPI_COMM_WORLD);

        // Add the cluster and update clustered status
        clusters.push_back(std::move(global_best));

        for (int i = 0; i < global_member_count; ++i) {
            clustered[clusters.back().members[i]] = 1;
        }

        // Remove clustered points from unclustered list
        auto new_end = std::remove_if(unclustered_indices.begin(),
                                       unclustered_indices.end(),
                                       [&clustered](int idx) { return clustered[idx]; });
        unclustered_indices.erase(new_end, unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

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

    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);

    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (on all ranks, same args)
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // Generate and broadcast synthetic data
    std::vector<Point> points(num_points);
    if (mpi_rank == 0) {
        generateSyntheticData(points, num_points);
        if (mpi_size > 1) {
            printf("QT Clustering Benchmark (MPI: %d ranks)\n", mpi_size);
        } else {
            printf("QT Clustering Benchmark\n");
        }
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Broadcast points to all ranks
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Synchronize stdout before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold,
                                                           mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Calculate statistics
    int total_clustered = 0;
    int max_cluster_size = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }

    const double avg_cluster_size = clusters.empty() ? 0.0 :
        static_cast<double>(total_clustered) / clusters.size();

    // Print results from rank 0 only
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double time_sec = cluster_time.count() / 1000.0;
        if (time_sec > 0.0) {
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters.size() / time_sec, num_points / time_sec);
        }
    }

    // Print results for external validation (rank 0)
    if (printResults && mpi_rank == 0) {
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

    // Validation (rank 0)
    if (validate && mpi_rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}

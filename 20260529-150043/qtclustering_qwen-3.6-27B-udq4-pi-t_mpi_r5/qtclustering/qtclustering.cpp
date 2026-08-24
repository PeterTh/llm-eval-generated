// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (MPI distributed memory):
// - All ranks hold a full copy of the point data and clustered[] flags
// - Each clustering iteration distributes candidate seed evaluation across ranks
// - Ranks communicate local best candidates; rank 0 selects the global winner
// - Winner broadcasts its cluster members to all ranks
// - Tie-breaking exactly matches sequential semantics (lowest seed index wins)

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
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
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             threshold, point_count);

        if (closest < 0) break;

        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// MPI-parallel QT clustering algorithm
// All ranks hold full copies of points and clustered flags.
// Each iteration distributes seed evaluation across ranks.
// Produces identical results to the sequential version.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     const int rank,
                                     const int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Communication buffers
    // Each rank sends: {cardinality, seed} to rank 0
    struct CandidateInfo {
        int cardinality;
        int seed;
    };
    CandidateInfo my_candidate{0, -1};
    std::vector<CandidateInfo> all_candidates(num_ranks);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // --- Partition unclustered seeds across ranks (block distribution) ---
        const int total_unclustered = static_cast<int>(unclustered_indices.size());
        const int local_start = (rank * total_unclustered) / num_ranks;
        const int local_end = ((rank + 1) * total_unclustered) / num_ranks;

        // --- Each rank evaluates its local seeds ---
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        for (int idx = local_start; idx < local_end; ++idx) {
            const int seed = unclustered_indices[idx];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                            threshold, N,
                                                            &candidate_members);

            // Match sequential semantics: strict > for cardinality,
            // so first seed with max cardinality wins (lowest index since
            // unclustered_indices is sorted).
            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // --- Gather local bests to rank 0 ---
        my_candidate = {local_max_cardinality, local_best_seed};
        MPI_Gather(&my_candidate, 2, MPI_INT,
                   all_candidates.data(), 2, MPI_INT,
                   0, MPI_COMM_WORLD);

        // --- Rank 0 selects the global best candidate ---
        // Match sequential semantics: highest cardinality, lowest seed on tie
        int global_max_cardinality = -1;
        int global_best_seed = -1;
        int global_best_rank = -1;

        if (rank == 0) {
            for (int r = 0; r < num_ranks; ++r) {
                const int card = all_candidates[r].cardinality;
                const int seed = all_candidates[r].seed;

                if (card > global_max_cardinality) {
                    global_max_cardinality = card;
                    global_best_seed = seed;
                    global_best_rank = r;
                } else if (card == global_max_cardinality && card > 0) {
                    // Tie-break: prefer lower seed index
                    if (seed < global_best_seed) {
                        global_best_seed = seed;
                        global_best_rank = r;
                    }
                }
            }
        }

        // --- Broadcast selection to all ranks ---
        MPI_Bcast(&global_max_cardinality, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // If no valid cluster found, terminate
        if (global_best_seed < 0 || global_max_cardinality <= 0) {
            break;
        }

        // --- Winner broadcasts its cluster members to all ranks ---
        // First broadcast the member count
        int member_count = 0;
        if (rank == global_best_rank) {
            member_count = static_cast<int>(local_best_members.size());
        }
        MPI_Bcast(&member_count, 1, MPI_INT, global_best_rank, MPI_COMM_WORLD);

        // Then broadcast the members
        std::vector<int> received_members;
        if (member_count > 0) {
            received_members.resize(member_count);
            if (rank == global_best_rank) {
                received_members = local_best_members;
            }
            MPI_Bcast(received_members.data(), member_count, MPI_INT,
                      global_best_rank, MPI_COMM_WORLD);
        }

        // --- All ranks update their state ---
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = std::move(received_members);
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (const int member : clusters.back().members) {
            clustered[member] = true;
        }

        // Remove clustered points from unclustered list
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
    // Initialize MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(&argc, &argv);
    }

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same args)
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
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI Parallel)\n");
        printf("MPI ranks: %d\n", num_ranks);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data on rank 0, broadcast to all ranks
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }

    // Broadcast points to all ranks
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering (parallel)
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time_ms = cluster_time.count();
    long global_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &global_cluster_time_ms, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

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

    // Print results (rank 0 only)
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", global_cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = global_cluster_time_ms / 1000.0;
        if (time_sec > 0) {
            const double clusters_per_sec = clusters.size() / time_sec;
            const double points_per_sec = num_points / time_sec;
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters_per_sec, points_per_sec);
        }

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
    }

    MPI_Finalize();
    return 0;
}

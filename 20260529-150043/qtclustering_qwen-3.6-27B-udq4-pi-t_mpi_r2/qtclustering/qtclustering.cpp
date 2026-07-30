// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   - In each outer iteration, unclustered seed candidates are distributed
//     across MPI ranks via block distribution.
//   - Each rank independently generates candidate clusters for its assigned
//     seeds and tracks its local best (largest cardinality).
//   - Rank 0 gathers all local best cardinalities, determines the global
//     winner, broadcasts the winner rank, and receives the winning cluster's
//     member list from the winning rank.
//   - The clustered bitmask is broadcast to all ranks each iteration.

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

// Parallel QT clustering algorithm using MPI
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                      const double threshold,
                                      int rank, int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Buffers reused across iterations to avoid reallocation
    std::vector<char> clustered_mask(N);
    std::vector<int> remote_unclustered(N);
    std::vector<int> all_cardinalities(num_ranks);
    std::vector<int> all_member_sizes(num_ranks);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // ================================================================
        // Phase 1: Broadcast current state from rank 0 to all ranks
        // ================================================================
        int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Build compact clustered mask
        for (int i = 0; i < N; ++i) {
            clustered_mask[i] = static_cast<char>(clustered[i]);
        }

        // Broadcast: count, mask, indices
        MPI_Bcast(&num_unclustered, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(clustered_mask.data(), N, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(unclustered_indices.data(), num_unclustered, MPI_INT, 0, MPI_COMM_WORLD);

        // Update local clustered array on all ranks
        for (int i = 0; i < N; ++i) {
            clustered[i] = clustered_mask[i] != 0;
        }

        // Copy unclustered indices into local buffer
        remote_unclustered.assign(unclustered_indices.begin(),
                                  unclustered_indices.begin() + num_unclustered);

        // ================================================================
        // Phase 2: Block-distribute seeds and evaluate locally
        // ================================================================
        int base = num_unclustered / num_ranks;
        int extra = num_unclustered % num_ranks;
        int start_idx = base * rank + std::min(rank, extra);
        int end_idx = base * (rank + 1) + std::min(rank + 1, extra);

        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        for (int si = start_idx; si < end_idx; ++si) {
            const int seed = remote_unclustered[si];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                             threshold, N,
                                                             &candidate_members);

            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // ================================================================
        // Phase 3: Gather cardinalities and member sizes to rank 0
        // ================================================================
        MPI_Gather(&local_max_cardinality, 1, MPI_INT,
                   all_cardinalities.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        int my_member_size = local_best_members.empty() ? 0
            : static_cast<int>(local_best_members.size());
        MPI_Gather(&my_member_size, 1, MPI_INT,
                   all_member_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // ================================================================
        // Phase 4: Rank 0 determines winner, then broadcasts it
        // ================================================================
        int winner_rank = -1;
        if (rank == 0) {
            int gbr = 0;
            int gbc = all_cardinalities[0];
            for (int r = 1; r < num_ranks; ++r) {
                if (all_cardinalities[r] > gbc) {
                    gbc = all_cardinalities[r];
                    gbr = r;
                }
            }
            winner_rank = (gbc > 0) ? gbr : -1;
        }
        MPI_Bcast(&winner_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // ================================================================
        // Phase 5: Winner sends its best cluster data to rank 0
        // ================================================================
        if (winner_rank >= 0) {
            if (winner_rank == rank && rank != 0) {
                // We are the winning rank — send data to rank 0
                MPI_Send(&local_best_seed, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
                MPI_Send(local_best_members.data(),
                         static_cast<int>(local_best_members.size()),
                         MPI_INT, 0, 101, MPI_COMM_WORLD);
            } else if (rank == 0 && winner_rank != 0) {
                // We are rank 0 — receive data from winner
                int remote_seed;
                int remote_size = all_member_sizes[winner_rank];

                MPI_Recv(&remote_seed, 1, MPI_INT, winner_rank, 100,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::vector<int> remote_members(remote_size);
                MPI_Recv(remote_members.data(), remote_size, MPI_INT,
                         winner_rank, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                Cluster cluster;
                cluster.seed_point = remote_seed;
                cluster.members = std::move(remote_members);
                clusters.push_back(cluster);
            } else if (rank == 0 && winner_rank == 0) {
                // Rank 0 is the winner — use local data directly
                Cluster cluster;
                cluster.seed_point = local_best_seed;
                cluster.members = std::move(local_best_members);
                clusters.push_back(cluster);
            }
            // Non-root, non-winner ranks do nothing here
        }

        // ================================================================
        // Phase 6: Rank 0 updates state and broadcasts continue flag
        // ================================================================
        int should_continue = 0;
        if (rank == 0) {
            if (winner_rank >= 0) {
                // Mark all members of the new cluster as clustered
                const auto& best_cluster = clusters.back();
                for (size_t i = 0; i < best_cluster.members.size(); ++i) {
                    clustered[best_cluster.members[i]] = true;
                }

                // Remove clustered points from unclustered list
                unclustered_indices.erase(
                    std::remove_if(unclustered_indices.begin(),
                                   unclustered_indices.end(),
                                   [&clustered](int idx) {
                                       return clustered[idx];
                                   }),
                    unclustered_indices.end()
                );
            }
            // Continue only if we found a valid cluster AND there are still unclustered points
            should_continue = (winner_rank >= 0 && !unclustered_indices.empty()) ? 1 : 0;
        }
        MPI_Bcast(&should_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!should_continue) break;
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
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_ranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0, then broadcast
    if (rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI: %d ranks)\n", num_ranks);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data on all ranks (same seed = same data)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Only rank 0 has the clusters
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
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

        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        // Print results for external validation
        if (printResults) {
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

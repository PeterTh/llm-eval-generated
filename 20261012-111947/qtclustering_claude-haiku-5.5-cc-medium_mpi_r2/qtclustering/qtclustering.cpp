// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: every rank holds the full point set and the replicated
// clustering state. Each iteration, the candidate seeds are split across ranks,
// each rank grows its candidate clusters, and a single MPI_Allreduce selects the
// global best (largest cardinality, smallest seed index on ties, matching the
// sequential first-wins order). All ranks then apply the same update, so no
// further communication is needed to keep the state in sync.

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

// (cardinality, seed) pair for the MPI_MAXLOC reduction. Layout matches MPI_2INT.
struct CandidateBest {
    int cardinality;
    int seed;
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

// Generate a candidate cluster starting from a seed point. Repeatedly adds the
// unclustered point whose maximum distance to the current members is smallest,
// as long as that maximum stays below threshold.
// Members are written to `members` in insertion order (seed first).
// Returns the cardinality (size) of the cluster.
//
// The maximum distance of each candidate to the cluster is maintained incrementally
// rather than recomputed from all members each step. A max over the same values is
// order-independent, so results are identical. Candidates that reach the threshold
// are dropped, since their maximum distance can only grow.
int generateCandidateCluster(const int seed_point,
                             const std::vector<bool>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int point_count,
                             std::vector<int>& cand,
                             std::vector<double>& cand_maxd,
                             std::vector<int>& members) {
    members.clear();
    members.push_back(seed_point);

    // Active candidates in ascending index order, with max distance to the cluster
    cand.clear();
    cand_maxd.clear();
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed_point) continue;
        const double d = distance(points[c], points[seed_point]);
        if (d < threshold) {
            cand.push_back(c);
            cand_maxd.push_back(d);
        }
    }

    while (!cand.empty()) {
        // Closest candidate by max distance (first index wins ties)
        size_t best = 0;
        for (size_t k = 1; k < cand.size(); ++k) {
            if (cand_maxd[k] < cand_maxd[best]) best = k;
        }

        const int added = cand[best];
        members.push_back(added);

        // Update remaining candidates with the new member, dropping those that can no
        // longer stay below threshold. Compacts in place, preserving order.
        size_t w = 0;
        for (size_t k = 0; k < cand.size(); ++k) {
            if (k == best) continue;
            const double d = std::max(cand_maxd[k], distance(points[cand[k]], points[added]));
            if (d < threshold) {
                cand[w] = cand[k];
                cand_maxd[w] = d;
                ++w;
            }
        }
        cand.resize(w);
        cand_maxd.resize(w);
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm. Collective: all ranks must call it.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Reusable workspaces
    std::vector<int> cand;
    std::vector<double> cand_maxd;
    std::vector<int> members;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Try this rank's share of the unclustered points as seeds (strided).
        // Seeds are visited in ascending order, so strict '>' keeps the smallest seed on ties.
        CandidateBest local{-1, -1};
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size();
             i += static_cast<size_t>(num_ranks)) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                             threshold, N,
                                                             cand, cand_maxd, members);

            if (cardinality > local.cardinality) {
                local.cardinality = cardinality;
                local.seed = seed;
            }
        }

        // Global best: maximum cardinality, minimum seed index on ties
        CandidateBest best{-1, -1};
        MPI_Allreduce(&local, &best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // If we found a cluster, add it
        if (best.seed >= 0 && best.cardinality > 0) {
            // Each rank recomputes the winning cluster. This is deterministic and costs
            // one seed's growth, which avoids communicating the member list.
            generateCandidateCluster(best.seed, clustered, points, threshold, N,
                                     cand, cand_maxd, members);

            Cluster cluster;
            cluster.seed_point = best.seed;
            cluster.members = members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < members.size(); ++i) {
                clustered[members[i]] = true;
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

// Validation: check that clusters satisfy the QT clustering properties.
// Diameters are computed in parallel across ranks. Output is written by rank 0.
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int rank,
                     const int num_ranks) {
    bool valid = true;

    if (rank == 0) printf("Validating clusters:\n");

    // Max distance between any two members of each cluster, computed by its owning rank
    std::vector<double> diameters(clusters.size(), 0.0);
    for (size_t c = static_cast<size_t>(rank); c < clusters.size();
         c += static_cast<size_t>(num_ranks)) {
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
        diameters[c] = max_diameter;
    }
    MPI_Allreduce(MPI_IN_PLACE, diameters.data(), static_cast<int>(diameters.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

        if (rank == 0 && c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            if (rank == 0) {
                printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                       c, max_diameter, threshold);
            }
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                if (rank == 0) {
                    printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                           member, membership[member], c);
                }
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

    if (rank == 0) {
        printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
               points.size(), clustered_count, points.size() - clustered_count);
    }

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

int runBenchmark(int argc, char** argv, const int rank, const int num_ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, num_ranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    if (rank == 0) {
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

    if (rank == 0) {
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

    // Validation (collective: every rank takes the same branch)
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, num_ranks);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const int status = runBenchmark(argc, argv, rank, num_ranks);

    MPI_Finalize();
    return status;
}

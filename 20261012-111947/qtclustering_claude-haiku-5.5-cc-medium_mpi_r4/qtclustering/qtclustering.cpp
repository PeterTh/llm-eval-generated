// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in every round, each unclustered point is a candidate seed
// and grows a candidate cluster. The seeds are dealt out cyclically across MPI
// ranks, each rank finds its best local seed, and an MPI_MAXLOC reduction picks
// the global winner (largest cluster, smallest index on ties, matching the
// sequential order). The owning rank broadcasts the winning cluster so every
// rank applies the same update to the clustered set.

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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

// Print only from the root rank so the output matches a single-process run
template <typename... Args>
void rootPrintf(int rank, const char* fmt, Args... args) {
    if (rank == 0) {
        printf(fmt, args...);
    }
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

// A candidate point that may still join the cluster being grown, together with
// its maximum distance to the cluster members added so far
struct Candidate {
    int index;
    double max_dist;
};

// Reusable buffer so growing a cluster does not allocate per seed
using CandidateBuffer = std::vector<Candidate>;

// Generate a candidate cluster starting from a seed point.
// `unclustered` lists the not-yet-clustered points in ascending order.
// Returns the cardinality (size) of the cluster; members are written to `members`.
//
// Each step adds the candidate with the smallest maximum distance to the members
// that is still below the threshold (ties go to the lowest index). The maximum
// distance only grows as members are added, so a candidate that reaches the
// threshold is dropped for good. This gives the same result as recomputing every
// candidate's maximum distance from scratch each step.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             CandidateBuffer& active,
                             std::vector<int>& members) {
    members.clear();
    members.push_back(seed_point);

    active.clear();
    for (int candidate : unclustered) {
        if (candidate != seed_point) {
            active.push_back({candidate, 0.0});
        }
    }

    // `last` is the member added in the previous step; its distances are folded in here
    int last = seed_point;
    while (true) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        size_t kept = 0;

        for (size_t i = 0; i < active.size(); ++i) {
            Candidate cand = active[i];
            if (cand.index == last) continue; // already a member

            const double dist = distance(points[cand.index], points[last]);
            cand.max_dist = std::max(cand.max_dist, dist);

            if (cand.max_dist >= threshold) continue; // can never join again

            active[kept++] = cand;
            if (cand.max_dist < min_diameter) {
                min_diameter = cand.max_dist;
                closest = cand.index;
            }
        }
        active.resize(kept);

        if (closest < 0) break; // No more points can be added

        members.push_back(closest);
        last = closest;
    }

    return static_cast<int>(members.size());
}

// Local result of one round: the best cardinality found and its seed.
// Laid out as the (int, int) pair MPI_2INT expects for MPI_MAXLOC.
struct SeedScore {
    int cardinality;
    int seed;
};

// Main QT clustering algorithm, run collectively by all ranks.
// Every rank returns the same cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    CandidateBuffer active;
    std::vector<int> members;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Each rank scores a cyclic share of the unclustered seeds.
        // Seeds are ascending, so the strict '>' keeps the lowest index on ties.
        SeedScore local{-1, N};
        for (size_t i = rank; i < unclustered_indices.size(); i += num_ranks) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(seed, unclustered_indices, points,
                                                             threshold, active, members);
            if (cardinality > local.cardinality) {
                local.cardinality = cardinality;
                local.seed = seed;
            }
        }

        // MPI_MAXLOC returns the largest cardinality and, on ties, the lowest seed
        MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        const int best_seed = local.seed;
        const int max_cardinality = local.cardinality;

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            // The rank that owns the winning seed rebuilds its cluster and broadcasts it,
            // so every rank applies identical updates.
            const size_t pos = std::lower_bound(unclustered_indices.begin(),
                                                unclustered_indices.end(), best_seed)
                               - unclustered_indices.begin();
            const int owner = static_cast<int>(pos % num_ranks);

            if (rank == owner) {
                generateCandidateCluster(best_seed, unclustered_indices, points,
                                         threshold, active, members);
            }

            int member_count = static_cast<int>(members.size());
            MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
            members.resize(member_count);
            MPI_Bcast(members.data(), member_count, MPI_INT, owner, MPI_COMM_WORLD);

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (int member : members) {
                clustered[member] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
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
            rootPrintf(rank, "Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        rootPrintf(rank, "Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }

    if (rank == 0) printf("QT Clustering Benchmark\n");
    rootPrintf(rank, "Number of points: %d\n", num_points);
    rootPrintf(rank, "Distance threshold: %.2f\n", threshold);
    rootPrintf(rank, "Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data on the root and share it so all ranks agree exactly
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long cluster_time_ms = cluster_time.count();
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(cluster_time_ms);

    rootPrintf(rank, "Clustering time: %ld ms\n", cluster_time.count());
    rootPrintf(rank, "Clusters found: %zu\n", clusters.size());

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

    rootPrintf(rank, "Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
    rootPrintf(rank, "Average cluster size: %.2f\n", avg_cluster_size);
    rootPrintf(rank, "Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    rootPrintf(rank, "Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

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

    // Validation runs on the root only; its verdict is shared so all ranks exit alike
    if (validate) {
        int valid = 0;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

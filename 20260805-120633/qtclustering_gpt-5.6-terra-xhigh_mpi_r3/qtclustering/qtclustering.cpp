// QT Clustering Benchmark - MPI distributed-memory implementation
//
// Candidate clusters for different seed points are independent while the
// current set of clustered points is fixed.  Each MPI rank therefore evaluates
// a cyclic subset of the seeds, and all ranks agree on the sequential winner
// before moving on to the next cluster.

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
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// A bounded distance cache removes repeated square-root calculations for the
// normal benchmark sizes without imposing an unbounded per-rank memory cost.
// Its rows are built collectively and then replicated because every seed can
// examine every point while constructing a candidate cluster.
class DistanceLookup {
public:
    DistanceLookup(const std::vector<Point>& points, const int rank, const int ranks)
        : points_(points), point_count_(static_cast<int>(points.size())) {
        constexpr std::size_t max_cache_bytes = 64U * 1024U * 1024U;
        const std::size_t count = static_cast<std::size_t>(point_count_);
        cached_ = count <= max_cache_bytes / sizeof(double) / count;

        if (!cached_) {
            return;
        }

        cache_.resize(count * count);
        const int first_row = static_cast<int>(static_cast<long long>(point_count_) * rank / ranks);
        const int last_row = static_cast<int>(static_cast<long long>(point_count_) * (rank + 1) / ranks);

        for (int candidate = first_row; candidate < last_row; ++candidate) {
            double* const row = cache_.data() + static_cast<std::size_t>(candidate) * point_count_;
            for (int member = 0; member < point_count_; ++member) {
                // Keep directed entries: this uses precisely the same operand
                // order as the original distance(points[candidate], points[member]).
                row[member] = distance(points_[candidate], points_[member]);
            }
        }

        std::vector<int> receive_counts(ranks);
        std::vector<int> displacements(ranks);
        for (int process = 0; process < ranks; ++process) {
            const int begin = static_cast<int>(static_cast<long long>(point_count_) * process / ranks);
            const int end = static_cast<int>(static_cast<long long>(point_count_) * (process + 1) / ranks);
            receive_counts[process] = (end - begin) * point_count_;
            displacements[process] = begin * point_count_;
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cache_.data(),
                       receive_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    [[nodiscard]] inline double operator()(const int candidate, const int member) const {
        if (cached_) {
            return cache_[static_cast<std::size_t>(candidate) * point_count_ + member];
        }
        return distance(points_[candidate], points_[member]);
    }

private:
    const std::vector<Point>& points_;
    int point_count_;
    bool cached_ = false;
    std::vector<double> cache_;
};

// Reused working storage avoids an allocation for every seed candidate.
struct CandidateWorkspace {
    explicit CandidateWorkspace(const int point_count)
        : in_cluster(point_count, 0), maximum_distances(point_count, 0.0) {
        members.reserve(point_count);
    }

    std::vector<unsigned char> in_cluster;
    std::vector<double> maximum_distances;
    std::vector<int> members;
};

// Generate one candidate cluster.  maximum_distances[candidate] is updated
// only for the member just added; it is exactly the maximum over all current
// members that the sequential implementation recomputed each iteration.
void generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const DistanceLookup& distances,
                              const double threshold,
                              CandidateWorkspace& workspace) {
    const int point_count = static_cast<int>(clustered.size());
    std::fill(workspace.in_cluster.begin(), workspace.in_cluster.end(), 0);
    std::fill(workspace.maximum_distances.begin(), workspace.maximum_distances.end(), 0.0);
    workspace.members.clear();

    workspace.in_cluster[seed_point] = 1;
    workspace.members.push_back(seed_point);

    while (static_cast<int>(workspace.members.size()) < point_count) {
        const int newest_member = workspace.members.back();

        // The original implementation recomputed these maxima from all
        // members.  Incremental maxima have identical comparison semantics
        // for the finite distances generated by this benchmark.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || workspace.in_cluster[candidate]) {
                continue;
            }
            const double candidate_distance = distances(candidate, newest_member);
            if (candidate_distance > workspace.maximum_distances[candidate]) {
                workspace.maximum_distances[candidate] = candidate_distance;
            }
        }

        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || workspace.in_cluster[candidate]) {
                continue;
            }

            const double max_distance = workspace.maximum_distances[candidate];
            if (max_distance < threshold && max_distance < min_diameter) {
                min_diameter = max_distance;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) {
            break;
        }
        workspace.in_cluster[closest_point] = 1;
        workspace.members.push_back(closest_point);
    }
}

// Main QT clustering algorithm.  The current state is replicated on every
// rank; only the independent seed evaluations are partitioned.  MPI_MAXLOC
// reproduces the sequential choice in one collective: largest cardinality,
// then the earliest seed in the ordered unclustered list (smallest index).
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int ranks) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    std::vector<Cluster> clusters;
    std::vector<int> local_best_members;
    CandidateWorkspace workspace(point_count);
    DistanceLookup distances(points, rank, ranks);

    for (int point = 0; point < point_count; ++point) {
        unclustered_indices[point] = point;
    }

    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = std::numeric_limits<int>::max();
        local_best_members.clear();

        // Cyclic assignment balances the variable candidate-cluster costs
        // better than a single contiguous block as points are removed.
        for (std::size_t position = static_cast<std::size_t>(rank);
             position < unclustered_indices.size(); position += static_cast<std::size_t>(ranks)) {
            const int seed = unclustered_indices[position];
            generateCandidateCluster(seed, clustered, distances, threshold, workspace);
            const int cardinality = static_cast<int>(workspace.members.size());

            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = workspace.members;
            }
        }

        int local_choice[2] = {local_max_cardinality, local_best_seed};
        int global_choice[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int max_cardinality = global_choice[0];
        const int best_seed = global_choice[1];

        if (best_seed < 0 || max_cardinality <= 0) {
            break;
        }

        int owner_rank = -1;
        for (std::size_t position = 0; position < unclustered_indices.size(); ++position) {
            if (unclustered_indices[position] == best_seed) {
                owner_rank = static_cast<int>(position % static_cast<std::size_t>(ranks));
                break;
            }
        }

        // The winning rank retained this candidate; all ranks receive it so
        // the next selection round observes the same clustered-point mask.
        if (rank != owner_rank) {
            local_best_members.resize(static_cast<std::size_t>(max_cardinality));
        }
        MPI_Bcast(local_best_members.data(), max_cardinality, MPI_INT, owner_rank, MPI_COMM_WORLD);

        if (rank == 0) {
            clusters.push_back({local_best_members, best_seed});
        }

        for (const int member : local_best_members) {
            clustered[member] = 1;
        }
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
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
        if (membership[i] >= 0) {
            clustered_count++;
        }
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

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;

    // Parse command line arguments.  All ranks receive the same argv, but
    // only rank zero emits user-facing output.
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate once and replicate the input so all ranks evaluate identical
    // candidate clusters using the original deterministic data set.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering.  The reported time is the slowest rank's elapsed
    // time, which is the elapsed time of the distributed operation as a whole.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_seconds = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto cluster_time = std::chrono::milliseconds(
            static_cast<long long>(cluster_seconds * 1000.0));
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

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

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = std::max(cluster_seconds, std::numeric_limits<double>::min());
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

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
            for (const int membership_id : membership) {
                membershipData.push_back(static_cast<double>(membership_id));
            }
            print_results(membershipData, "ClusterMembership");
        }

        // Validation
        if (validate && !validateClusters(clusters, points, threshold)) {
            exit_code = 1;
        }
        if (validate) {
            printf("Validation: %s\n", exit_code == 0 ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

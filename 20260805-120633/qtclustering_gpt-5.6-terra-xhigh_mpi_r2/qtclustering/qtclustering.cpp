// QT Clustering Benchmark
//
// QT (Quality Threshold) clustering greedily selects the largest candidate
// cluster.  Candidate clusters for distinct seeds are independent within one
// greedy iteration, so MPI ranks evaluate disjoint seed sets in parallel.

#include <algorithm>
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

        // For benchmark-sized inputs this retains the original sampling
        // sequence (zero-sized groups are simply skipped).  Very small valid
        // inputs otherwise can never make progress because N / 30.0 <= 1.
        if (group_cnt == 0 && N <= 30) {
            group_cnt = 1;
        }

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

// Scratch storage used to construct successive candidate clusters on one MPI
// rank.  Updating each candidate's running diameter when a member is added is
// equivalent to recomputing all member distances, but removes the redundant
// work from the sequential implementation.
class CandidateClusterGenerator {
public:
    CandidateClusterGenerator(const std::vector<Point>& points, const int point_count)
        : points_(points),
          point_count_(point_count),
          member_generation_(point_count, 0),
          max_distance_(point_count),
          generation_(0) {
        members_.reserve(point_count);
    }

    int generate(const int seed_point,
                 const std::vector<unsigned char>& clustered,
                 const double threshold) {
        // A generation stamp avoids clearing an O(N) membership array for
        // every seed.  Reset safely in the impractical event of wraparound.
        if (generation_ == std::numeric_limits<int>::max()) {
            std::fill(member_generation_.begin(), member_generation_.end(), 0);
            generation_ = 0;
        }
        const int this_generation = ++generation_;

        members_.clear();
        members_.push_back(seed_point);
        member_generation_[seed_point] = this_generation;

        int newest_member = seed_point;
        bool first_member = true;

        while (static_cast<int>(members_.size()) < point_count_) {
            int closest_point = -1;
            double min_diameter = std::numeric_limits<double>::max();

            // For every possible candidate, update its maximum distance to
            // the current cluster with the just-added member.  The update
            // order matches the original member-order reduction exactly.
            for (int candidate = 0; candidate < point_count_; ++candidate) {
                if (clustered[candidate] ||
                    member_generation_[candidate] == this_generation) {
                    continue;
                }

                const double newest_distance = distance(points_[candidate], points_[newest_member]);
                const double candidate_diameter = first_member
                    ? newest_distance
                    : std::max(max_distance_[candidate], newest_distance);
                max_distance_[candidate] = candidate_diameter;

                // Strict comparisons intentionally preserve the original
                // lower-index tie breaking for otherwise equal candidates.
                if (candidate_diameter < threshold && candidate_diameter < min_diameter) {
                    min_diameter = candidate_diameter;
                    closest_point = candidate;
                }
            }

            if (closest_point < 0) {
                break;
            }

            member_generation_[closest_point] = this_generation;
            members_.push_back(closest_point);
            newest_member = closest_point;
            first_member = false;
        }

        return static_cast<int>(members_.size());
    }

    const std::vector<int>& members() const {
        return members_;
    }

private:
    const std::vector<Point>& points_;
    int point_count_;
    std::vector<int> member_generation_;
    std::vector<double> max_distance_;
    std::vector<int> members_;
    int generation_;
};

// Main MPI QT clustering algorithm.  The full input and the small clustered
// bitmap are replicated, while every greedy iteration distributes candidate
// seed construction round-robin over ranks.  All ranks select the same global
// winner (maximum cardinality, then minimum seed index) and receive its
// members, preserving the sequential algorithm's deterministic semantics.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     const int rank,
                                     const int world_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<Cluster> clusters;
    if (rank == 0) {
        clusters.reserve(point_count);
    }

    CandidateClusterGenerator generator(points, point_count);
    // This encoding orders candidates by descending cardinality and then by
    // ascending seed.  It lets MPI_MAX select the global winner with a
    // logarithmic-time all-reduce instead of gathering one pair per rank.
    const long long candidate_base = static_cast<long long>(point_count) + 1;
    std::vector<int> best_cluster_members;
    int remaining_points = point_count;

    while (remaining_points > 0) {
        int local_max_cardinality = -1;
        int local_best_seed = std::numeric_limits<int>::max();

        // A cyclic partition balances neighbouring input points, which tend
        // to have similar cluster sizes, across all MPI ranks.
        for (int seed = rank; seed < point_count; seed += world_size) {
            if (clustered[seed]) {
                continue;
            }

            const int cardinality = generator.generate(seed, clustered, threshold);
            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
            }
        }

        long long local_candidate = -1;
        if (local_max_cardinality > 0) {
            local_candidate = static_cast<long long>(local_max_cardinality) * candidate_base +
                              (point_count - local_best_seed);
        }
        long long global_candidate = -1;
        MPI_Allreduce(&local_candidate, &global_candidate, 1, MPI_LONG_LONG_INT,
                      MPI_MAX, MPI_COMM_WORLD);

        if (global_candidate < 0) {
            break;
        }

        const int global_max_cardinality =
            static_cast<int>(global_candidate / candidate_base);
        const int global_best_seed = point_count -
            static_cast<int>(global_candidate % candidate_base);
        const int winning_rank = global_best_seed % world_size;

        if (rank == winning_rank) {
            // Candidate memberships are only materialized for the global
            // winner.  Its capacity is retained for the next iteration.
            const int regenerated_cardinality =
                generator.generate(global_best_seed, clustered, threshold);
            best_cluster_members = generator.members();

            // The same state and deterministic calculation must reproduce
            // the cardinality used in the collective selection above.
            if (regenerated_cardinality != global_max_cardinality) {
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        } else {
            best_cluster_members.resize(global_max_cardinality);
        }

        MPI_Bcast(best_cluster_members.data(), global_max_cardinality, MPI_INT,
                  winning_rank, MPI_COMM_WORLD);

        if (rank == 0) {
            clusters.push_back({best_cluster_members, global_best_seed});
        }

        for (const int member : best_cluster_members) {
            clustered[member] = 1;
        }
        remaining_points -= global_max_cardinality;
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
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int print_results_requested = 0;
    int should_run = 1;
    int exit_status = 0;

    // Parsing and diagnostics are performed once, then configuration is
    // broadcast to guarantee identical control flow on every MPI rank.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_results_requested = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                should_run = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                should_run = 0;
                exit_status = 1;
            }
        }

        if (should_run && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            should_run = 0;
            exit_status = 1;
        }
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_results_requested, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&should_run, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!should_run) {
        MPI_Finalize();
        return exit_status;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate once and distribute the identical point array to all ranks.
    // Each rank needs random read access to every point for its seed subset.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    static_assert(sizeof(Point) == 2 * sizeof(double),
                  "Point must be two contiguous doubles for MPI broadcast");
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClusteringMPI(points, threshold, rank, world_size);
    const double local_cluster_time = MPI_Wtime() - cluster_start;

    // Report the critical-path (slowest rank) wall time rather than a local
    // timing that could hide load imbalance.
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time * 1000.0));
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double clusters_per_sec = cluster_time > 0.0 ? clusters.size() / cluster_time : 0.0;
        const double points_per_sec = cluster_time > 0.0 ? num_points / cluster_time : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        // Print results for external validation
        if (print_results_requested) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int membership_id : membership) {
                membership_data.push_back(static_cast<double>(membership_id));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate) {
            exit_status = validateClusters(clusters, points, threshold) ? 0 : 1;
            printf("Validation: %s\n", exit_status == 0 ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_status;
}

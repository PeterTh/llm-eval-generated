// QT Clustering Benchmark - distributed-memory MPI version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <mpi.h>

#include <algorithm>
#include <climits>
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
        // The original expression cannot produce a point when N <= 30.
        if (N <= 30) group_cnt = 1;
        
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

inline double distanceSquared(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Scratch storage is retained across seeds, avoiding allocation in the hot path.
// A point that reaches the threshold can be dropped permanently from a candidate:
// its maximum distance can only increase as members are added.
class CandidateWorkspace {
public:
    explicit CandidateWorkspace(const int point_count)
        : max_distance_squared_(point_count) {
        eligible_.reserve(point_count);
        members.reserve(point_count);
    }

    int generate(const int seed_point,
                 const std::vector<int>& unclustered,
                 const std::vector<Point>& points,
                 const double threshold_squared) {
        eligible_.clear();
        members.clear();
        members.push_back(seed_point);

        int closest_point = -1;
        double min_diameter_squared = std::numeric_limits<double>::max();

        // Build the seed's feasible set in point-index order.  Strict comparisons
        // retain the sequential implementation's lower-index tie breaking.
        for (const int candidate : unclustered) {
            if (candidate == seed_point) continue;

            const double diameter_squared =
                distanceSquared(points[candidate], points[seed_point]);
            if (diameter_squared < threshold_squared) {
                max_distance_squared_[candidate] = diameter_squared;
                eligible_.push_back(candidate);
                if (diameter_squared < min_diameter_squared) {
                    min_diameter_squared = diameter_squared;
                    closest_point = candidate;
                }
            }
        }

        while (closest_point >= 0) {
            members.push_back(closest_point);

            int next_closest = -1;
            double next_min_diameter_squared =
                std::numeric_limits<double>::max();
            size_t output = 0;

            // Incrementally update each point's diameter and compact away both
            // the selected point and points that can never again be feasible.
            for (const int candidate : eligible_) {
                if (candidate == closest_point) continue;

                const double new_distance_squared =
                    distanceSquared(points[candidate], points[closest_point]);
                const double diameter_squared =
                    std::max(max_distance_squared_[candidate], new_distance_squared);

                if (diameter_squared < threshold_squared) {
                    max_distance_squared_[candidate] = diameter_squared;
                    eligible_[output++] = candidate;
                    if (diameter_squared < next_min_diameter_squared) {
                        next_min_diameter_squared = diameter_squared;
                        next_closest = candidate;
                    }
                }
            }

            eligible_.resize(output);
            closest_point = next_closest;
        }

        return static_cast<int>(members.size());
    }

    std::vector<int> members;

private:
    std::vector<double> max_distance_squared_;
    std::vector<int> eligible_;
};

// The seed evaluations dominate QT clustering and are independent within an
// outer iteration.  They are assigned cyclically by position to MPI ranks.
// MPI_MAXLOC reproduces the sequential rule: largest cluster, then lowest seed.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count,
                                  MPI_Comm communicator) {
    const int point_count = static_cast<int>(points.size());
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    std::vector<Cluster> clusters;
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    if (rank == 0) clusters.reserve(point_count);

    CandidateWorkspace workspace(point_count);
    std::vector<int> local_best_members;
    std::vector<int> winning_members;
    local_best_members.reserve(point_count);
    winning_members.reserve(point_count);
    
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = INT_MAX;

        // Cyclic placement balances geometrically related, consecutively
        // generated points better than contiguous blocks.
        for (size_t position = static_cast<size_t>(rank);
             position < unclustered_indices.size();
             position += static_cast<size_t>(process_count)) {
            const int seed = unclustered_indices[position];
            const int cardinality = workspace.generate(
                seed, unclustered_indices, points, threshold_squared);

            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = workspace.members;
            }
        }

        int local_choice[2] = {local_max_cardinality, local_best_seed};
        int global_choice[2] = {-1, INT_MAX};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC,
                      communicator);

        const int max_cardinality = global_choice[0];
        const int best_seed = global_choice[1];
        if (best_seed == INT_MAX || max_cardinality <= 0) break;

        // If every remaining candidate is a singleton, removing points cannot
        // make a larger candidate possible.  Emit the sequentially identical
        // singleton suffix without O(n) additional collective iterations.
        if (max_cardinality == 1) {
            if (rank == 0) {
                for (const int seed : unclustered_indices) {
                    clusters.push_back(Cluster{{seed}, seed});
                }
            }
            break;
        }

        const auto seed_position = std::lower_bound(
            unclustered_indices.begin(), unclustered_indices.end(), best_seed);
        const int owner = static_cast<int>(
            std::distance(unclustered_indices.begin(), seed_position) % process_count);

        if (rank == owner) {
            winning_members = local_best_members;
        } else {
            winning_members.resize(max_cardinality);
        }
        MPI_Bcast(winning_members.data(), max_cardinality, MPI_INT, owner,
                  communicator);

        if (rank == 0) {
            clusters.push_back(Cluster{winning_members, best_seed});
        }

        for (const int member : winning_members) clustered[member] = 1;
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) {
                               return clustered[index] != 0;
                           }),
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Error: MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    int validate_flag = 0;
    int print_results_flag = 0;
    int parse_status = 0;

    // Rank zero owns command-line diagnostics and distributes one canonical
    // configuration, avoiding duplicate output under mpirun.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                print_results_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parse_status = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 1;
                break;
            }
        }

        if (parse_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            parse_status = 1;
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    int integer_options[3] = {num_points, validate_flag, print_results_flag};
    MPI_Bcast(integer_options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    num_points = integer_options[0];
    validate_flag = integer_options[1];
    print_results_flag = integer_options[2];

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate_flag ? "enabled" : "disabled");
    }
    
    // Deterministic local generation avoids distributing the replicated point
    // array and produces byte-identical input on every rank.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, process_count, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;

    double cluster_time_seconds = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        const long cluster_time_milliseconds =
            static_cast<long>(cluster_time_seconds * 1000.0);
        printf("Clustering time: %ld ms\n", cluster_time_milliseconds);
        printf("Clusters found: %zu\n", clusters.size());
    
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
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double clusters_per_sec = clusters.size() / cluster_time_seconds;
        const double points_per_sec = num_points / cluster_time_seconds;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (print_results_flag) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int membership_index : membership) {
                membership_data.push_back(static_cast<double>(membership_index));
            }
            print_results(membership_data, "ClusterMembership");
        }

        if (validate_flag) {
            const bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

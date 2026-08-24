// QT Clustering Benchmark - MPI distributed-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cstdint>
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

        // For N <= 30 the original expression can never (or, for N == 30,
        // practically never) produce a point, so guarantee forward progress.
        if (N <= 30) {
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
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct CandidateWorkspace {
    std::vector<std::uint8_t> state;
    std::vector<double> max_distance;
    std::vector<int> members;

    explicit CandidateWorkspace(const int point_count)
        : state(point_count), max_distance(point_count) {
        members.reserve(point_count);
    }
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<std::uint8_t>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              CandidateWorkspace& workspace) {
    const int point_count = static_cast<int>(points.size());
    std::fill(workspace.max_distance.begin(), workspace.max_distance.end(), 0.0);
    workspace.members.clear();

    // State 0 is eligible, 1 is already in this candidate, and 2 can be
    // skipped permanently. A diameter can only grow as members are added.
    for (int i = 0; i < point_count; ++i) {
        workspace.state[i] = clustered[i] ? 2 : 0;
    }
    workspace.state[seed_point] = 1;
    workspace.members.push_back(seed_point);

    // Incrementally update each point's diameter with only the newest member.
    // This is equivalent to rescanning all members, but removes a factor of
    // the candidate cluster size from the dominant computation.
    while (true) {
        const int newest_member = workspace.members.back();
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (workspace.state[candidate] != 0) {
                continue;
            }

            const double dist = distance(points[candidate], points[newest_member]);
            const double diameter = std::max(workspace.max_distance[candidate], dist);
            workspace.max_distance[candidate] = diameter;

            if (diameter >= threshold) {
                workspace.state[candidate] = 2;
            } else if (diameter < min_diameter) {
                min_diameter = diameter;
                closest = candidate;
            }
        }

        if (closest < 0) {
            break;
        }
        workspace.state[closest] = 1;
        workspace.members.push_back(closest);
    }

    return static_cast<int>(workspace.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count) {
    const int N = static_cast<int>(points.size());
    std::vector<std::uint8_t> clustered(N, 0);
    std::vector<Cluster> clusters;
    CandidateWorkspace workspace(N);
    std::vector<int> local_best_members;
    std::vector<int> winning_members;
    int remaining = N;

    // Main clustering loop
    while (remaining > 0) {
        int local_choice[2] = {-1, INT_MAX};

        // Cyclic placement balances seed counts without scheduling messages.
        for (int seed = rank; seed < N; seed += process_count) {
            if (clustered[seed]) {
                continue;
            }

            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, workspace);
            if (cardinality > local_choice[0] ||
                (cardinality == local_choice[0] && seed < local_choice[1])) {
                local_choice[0] = cardinality;
                local_choice[1] = seed;
                local_best_members = workspace.members;
            }
        }

        // MPI_MAXLOC selects maximum cardinality and, on ties, the lowest
        // seed index, matching the original increasing-order sequential scan.
        int global_choice[2];
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        const int best_seed = global_choice[1];
        const int member_count = global_choice[0];
        const int owner = best_seed % process_count;
        if (rank == owner) {
            winning_members = local_best_members;
        } else {
            winning_members.resize(member_count);
        }
        MPI_Bcast(winning_members.data(), member_count, MPI_INT, owner,
                  MPI_COMM_WORLD);

        for (const int member : winning_members) {
            clustered[member] = 1;
        }
        remaining -= member_count;

        // Only rank zero retains the result; all ranks retain just the compact
        // clustered bitmap needed for the next distributed iteration.
        if (rank == 0) {
            clusters.push_back(Cluster{winning_members, best_seed});
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int print_results_requested = 0;
    int parse_status = 0; // 0: run, 1: help, 2: invalid arguments

    // Rank zero parses once, then broadcasts a single consistent configuration.
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
                parse_status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 2;
                break;
            }
        }

        if (parse_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            parse_status = 2;
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 1 ? 0 : 1;
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_results_requested, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI processes: %d\n", process_count);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, process_count);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_cluster_seconds = MPI_Wtime() - cluster_start;
    double cluster_seconds = 0.0;
    MPI_Reduce(&local_cluster_seconds, &cluster_seconds, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        const long long cluster_milliseconds =
            static_cast<long long>(cluster_seconds * 1000.0);
        printf("Clustering time: %lld ms\n", cluster_milliseconds);
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
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        const double clusters_per_sec = clusters.size() / cluster_seconds;
        const double points_per_sec = num_points / cluster_seconds;
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
            for (const int membership_index : membership) {
                membership_data.push_back(static_cast<double>(membership_index));
            }
            print_results(membership_data, "ClusterMembership");
        }

        // Validation is intentionally performed once because rank zero owns
        // the materialized cluster list and all ranks have identical points.
        if (validate && !validateClusters(clusters, points, threshold)) {
            printf("Validation: FAILED\n");
            exit_code = 1;
        } else if (validate) {
            printf("Validation: PASSED\n");
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

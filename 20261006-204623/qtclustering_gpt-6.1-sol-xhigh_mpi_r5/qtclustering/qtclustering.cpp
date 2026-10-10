// QT Clustering Benchmark - Distributed MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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
        // Otherwise small inputs repeatedly generate empty groups forever.
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
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct Neighbor {
    int point;
    double diameter;
};

struct CandidateCluster {
    int seed;
    bool initialized = false;
    std::vector<Neighbor> neighbors;
    std::vector<int> members;
};

// Maintain each candidate's maximum distance incrementally. A rejected point
// cannot become eligible as a cluster grows. Keep actual Euclidean distances:
// comparing squared distances would change ties after sqrt rounding.
void generateCandidateCluster(CandidateCluster& cluster,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              std::vector<Neighbor>& workspace) {
    workspace.clear();
    cluster.members.clear();
    cluster.members.push_back(cluster.seed);
    double min_diameter = std::numeric_limits<double>::max();
    int closest = -1;
    size_t closest_position = 0;

    for (const Neighbor& neighbor : cluster.neighbors) {
        if (clustered[neighbor.point]) continue;
        if (neighbor.diameter < min_diameter ||
            (neighbor.diameter == min_diameter && neighbor.point < closest)) {
            min_diameter = neighbor.diameter;
            closest = neighbor.point;
            closest_position = workspace.size();
        }
        workspace.push_back(neighbor);
    }

    while (closest >= 0) {
        const int added = closest;
        cluster.members.push_back(added);
        workspace[closest_position] = workspace.back();
        workspace.pop_back();

        closest = -1;
        min_diameter = std::numeric_limits<double>::max();
        size_t kept = 0;
        for (size_t i = 0; i < workspace.size(); ++i) {
            Neighbor candidate = workspace[i];
            candidate.diameter = std::max(candidate.diameter,
                distance(points[candidate.point], points[added]));
            if (!(candidate.diameter < threshold)) continue;

            workspace[kept] = candidate;
            // Compaction can reorder candidates; explicitly preserve the
            // sequential algorithm's lowest-point-index tie break.
            if (candidate.diameter < min_diameter ||
                (candidate.diameter == min_diameter && candidate.point < closest)) {
                min_diameter = candidate.diameter;
                closest = candidate.point;
                closest_position = kept;
            }
            ++kept;
        }
        workspace.resize(kept);
    }
}

// Collective QT clustering; only rank zero retains the final cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank, rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;
    std::vector<CandidateCluster> candidates;
    if (rank < N) candidates.reserve((N - 1 - rank) / rank_count + 1);
    // Cyclic ownership spreads nearby, potentially expensive seeds over ranks.
    // Only owned seeds retain neighborhoods and cached cluster memberships.
    for (size_t seed = static_cast<size_t>(rank); seed < points.size();
         seed += static_cast<size_t>(rank_count)) {
        candidates.push_back({static_cast<int>(seed), false, {}, {}});
    }
    std::vector<Neighbor> workspace;
    std::vector<int> selected_members;
    int remaining = N;

    while (remaining > 0) {
        int max_cardinality = 0;
        int best_seed = std::numeric_limits<int>::max();
        CandidateCluster* best_candidate = nullptr;
        for (CandidateCluster& candidate : candidates) {
            if (clustered[candidate.seed]) continue;
            if (!candidate.initialized) {
                // Any member must be within threshold of the seed. Build this
                // sparse neighborhood once, without a replicated N-by-N table.
                for (int point = 0; point < N; ++point) {
                    if (point == candidate.seed || clustered[point]) continue;
                    const double dist = distance(points[point], points[candidate.seed]);
                    if (dist < threshold) candidate.neighbors.push_back({point, dist});
                }
                candidate.initialized = true;
            }
            // Seeds are visited in increasing order, so equal cardinalities
            // cannot improve the local winner. This is a safe upper bound.
            if (candidate.neighbors.size() + 1 <= static_cast<size_t>(max_cardinality))
                continue;

            // Removing points outside a cached cluster leaves every greedy
            // choice unchanged, including the stopping condition.
            bool reusable = !candidate.members.empty();
            for (int member : candidate.members) {
                if (clustered[member]) {
                    reusable = false;
                    break;
                }
            }
            if (!reusable) {
                int upper_bound = 1;
                for (const Neighbor& neighbor : candidate.neighbors)
                    upper_bound += !clustered[neighbor.point];
                if (upper_bound <= max_cardinality) continue;
                generateCandidateCluster(candidate, clustered, points, threshold, workspace);
            }
            const int cardinality = static_cast<int>(candidate.members.size());
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = candidate.seed;
                best_candidate = &candidate;
            }
            if (max_cardinality == remaining) break;
        }

        // MINLOC on (-size, seed) chooses the largest cluster and then the
        // lowest seed, exactly as the sequential traversal does.
        const int local_best[2] = {-max_cardinality, best_seed};
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        const int cardinality = -global_best[0];
        const int seed = global_best[1];

        // If every candidate is a singleton, all remaining rounds have a known
        // outcome. Emit them in seed order without one collective per point.
        if (cardinality == 1) {
            if (rank == 0) {
                for (int point = 0; point < N; ++point)
                    if (!clustered[point]) clusters.push_back({{point}, point});
            }
            break;
        }

        const int owner = seed % rank_count;
        selected_members.resize(cardinality);
        if (rank == owner) {
            std::copy(best_candidate->members.begin(), best_candidate->members.end(),
                      selected_members.begin());
        }
        MPI_Bcast(selected_members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        for (int member : selected_members) clustered[member] = 1;
        if (rank == 0) clusters.push_back({selected_members, seed});
        remaining -= cardinality;
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

int runBenchmark(int argc, char** argv) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0; // 0: run, 1: help, 2: invalid arguments
    
    // Rank zero parses and prints; all ranks receive the same configuration.
    for (int i = 1; rank == 0 && i < argc; ++i) {
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
            parse_status = 1;
            break;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            parse_status = 2;
            break;
        }
    }
    
    if (rank == 0 && parse_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        parse_status = 2;
    }

    int configuration[4] = {num_points, validate, printResults, parse_status};
    MPI_Bcast(configuration, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    num_points = configuration[0];
    validate = configuration[1];
    printResults = configuration[2];
    if (configuration[3] != 0) return configuration[3] == 1 ? 0 : 1;

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate once to preserve the original random sequence. A point datatype
    // avoids overflowing the MPI count by multiplying num_points by two.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_status = 0;
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(time_sec * 1000.0));
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
                exit_status = 1;
            }
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exit_status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int exit_status;
    try {
        exit_status = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exit_status;
}

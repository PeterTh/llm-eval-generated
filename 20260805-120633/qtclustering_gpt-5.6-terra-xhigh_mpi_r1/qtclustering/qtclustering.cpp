// QT Clustering Benchmark - MPI distributed-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
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

// Find the closest point that maintains the diameter constraint.  The distance
// matrix is indexed as [candidate][member], matching the operation order of
// the original distance(points[candidate], points[member]) call.
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<unsigned char>& clustered,
                     const std::vector<int>& in_cluster_mark,
                     const int current_mark,
                     const std::vector<double>& distances,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster_mark[candidate] == current_mark) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        const double* const candidate_distances =
            distances.data() + static_cast<size_t>(candidate) * point_count;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = candidate_distances[member];
            max_dist = std::max(max_dist, dist);

            // Both comparisons below are strict in the reference algorithm;
            // once either bound is reached, this candidate cannot win.
            if (max_dist >= threshold || max_dist >= min_diameter) {
                break;
            }
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
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
                              const std::vector<unsigned char>& clustered,
                              std::vector<int>& in_cluster_mark,
                              const int current_mark,
                              const std::vector<double>& distances,
                              const double threshold,
                              const int point_count,
                              std::vector<int>& members) {
    members.clear();
    
    // Add seed point
    in_cluster_mark[seed_point] = current_mark;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster_mark,
                                             current_mark, distances, threshold,
                                             point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster_mark[closest] = current_mark;
        members.push_back(closest);
    }
    
    return static_cast<int>(members.size());
}

// Build a replicated distance matrix.  The point data is small compared with
// the repeated O(N^3+) candidate work, and caching distances eliminates the
// dominant redundant square-root calculations.  Rows are generated on their
// owning ranks and then shared once with MPI_Allgatherv.
std::vector<double> buildDistanceMatrix(const std::vector<Point>& points,
                                        const int rank,
                                        const int world_size) {
    const int point_count = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(point_count) * point_count;
    std::vector<double> distances(matrix_size);

    const int first_row = static_cast<int>(
        static_cast<long long>(point_count) * rank / world_size);
    const int last_row = static_cast<int>(
        static_cast<long long>(point_count) * (rank + 1) / world_size);

    for (int row = first_row; row < last_row; ++row) {
        double* const row_distances =
            distances.data() + static_cast<size_t>(row) * point_count;
        for (int column = 0; column < point_count; ++column) {
            row_distances[column] = distance(points[row], points[column]);
        }
    }

    std::vector<int> receive_counts(world_size);
    std::vector<int> displacements(world_size);
    for (int process = 0; process < world_size; ++process) {
        const int process_first_row = static_cast<int>(
            static_cast<long long>(point_count) * process / world_size);
        const int process_last_row = static_cast<int>(
            static_cast<long long>(point_count) * (process + 1) / world_size);
        receive_counts[process] = (process_last_row - process_first_row) * point_count;
        displacements[process] = process_first_row * point_count;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, distances.data(),
                   receive_counts.data(), displacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    return distances;
}

// Main QT clustering algorithm.  All ranks retain the current clustered state;
// candidate seeds are cyclically partitioned for balance.  A global reduction
// selects the largest candidate, resolving ties by the original lowest-seed
// rule, and the owner broadcasts that cluster for the next iteration.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     const int rank,
                                     const int world_size) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;
    const std::vector<double> distances = buildDistanceMatrix(points, rank, world_size);
    std::vector<int> in_cluster_mark(N, 0);
    std::vector<int> candidate_members;
    std::vector<int> best_cluster_members;
    candidate_members.reserve(N);
    best_cluster_members.reserve(N);
    int next_mark = 1;
    int remaining = N;
    
    // Main clustering loop
    while (remaining > 0) {
        int max_cardinality = -1;
        int best_seed = -1;
        
        // Cyclic ownership spreads consecutive synthetic-data groups across
        // ranks more evenly than contiguous blocks.
        for (int seed = rank; seed < N; seed += world_size) {
            if (clustered[seed]) continue;
            
            if (next_mark == std::numeric_limits<int>::max()) {
                std::fill(in_cluster_mark.begin(), in_cluster_mark.end(), 0);
                next_mark = 1;
            }
            const int current_mark = next_mark++;
            const int cardinality = generateCandidateCluster(
                seed, clustered, in_cluster_mark, current_mark, distances,
                threshold, N, candidate_members);
            
            if (cardinality > max_cardinality ||
                (cardinality == max_cardinality && seed < best_seed)) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }

        int global_cardinality = 0;
        MPI_Allreduce(&max_cardinality, &global_cardinality, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD);
        const int local_seed_for_global_best =
            max_cardinality == global_cardinality ? best_seed : N;
        int global_best_seed = N;
        MPI_Allreduce(&local_seed_for_global_best, &global_best_seed, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD);
        
        // If we found a cluster, add it
        if (global_best_seed < N && global_cardinality > 0) {
            const int owner = global_best_seed % world_size;
            if (rank != owner) {
                best_cluster_members.resize(global_cardinality);
            }
            MPI_Bcast(best_cluster_members.data(), global_cardinality, MPI_INT,
                      owner, MPI_COMM_WORLD);

            if (rank == 0) {
                Cluster cluster;
                cluster.seed_point = global_best_seed;
                cluster.members = best_cluster_members;
                clusters.push_back(std::move(cluster));
            }
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            remaining -= global_cardinality;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;
    
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
    
    if (num_points <= 0 || !(threshold > 0.0)) {
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
    
    // Rank zero preserves the reference data-generation order; every rank
    // receives the identical point set needed for its distributed seed work.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClusteringMPI(points, threshold, rank, world_size);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time * 1000.0));
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
    
        // Performance metrics use the slowest rank's elapsed time.
        const double clusters_per_sec =
            cluster_time > 0.0 ? clusters.size() / cluster_time : 0.0;
        const double points_per_sec =
            cluster_time > 0.0 ? num_points / cluster_time : 0.0;
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
    
        // Validation is output-only and is therefore performed once.
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

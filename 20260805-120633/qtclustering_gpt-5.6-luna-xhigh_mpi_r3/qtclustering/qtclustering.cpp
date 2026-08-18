// QT Clustering Benchmark - MPI Version
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
#include <utility>
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
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<double>& max_distances,
                              std::vector<int>& in_cluster_generation,
                              const int generation,
                              std::vector<int>& members) {
    members.clear();
    
    // Add seed point
    in_cluster_generation[seed_point] = generation;
    members.push_back(seed_point);

    // The first candidate scan has one cluster member.  Keeping the current
    // maximum distance for every candidate makes each later member addition a
    // single distance update instead of rescanning the whole cluster.
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster_generation[candidate] == generation) {
            continue;
        }
        max_distances[candidate] = distance(points[candidate], points[seed_point]);
    }
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // Try each unclustered point as a candidate.  Scanning in ascending
        // index order retains the original deterministic tie behavior.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster_generation[candidate] == generation) {
                continue;
            }

            const double max_dist = max_distances[candidate];
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest = candidate;
            }
        }
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster_generation[closest] = generation;
        members.push_back(closest);

        // Update the maximum distance for all candidates that can still be
        // added.  The update order is the same as the original member scan.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster_generation[candidate] == generation) {
                continue;
            }

            const double dist = distance(points[candidate], points[closest]);
            max_distances[candidate] = std::max(max_distances[candidate], dist);
        }
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int world_size,
                                  const MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;
    if (rank == 0) {
        clusters.reserve(N);
    }

    // These buffers are reused for every seed.  In particular, avoiding an
    // O(N) vector initialization for every seed is important for large data
    // sets and does not change the candidate selection semantics.
    std::vector<double> max_distances(N, 0.0);
    std::vector<int> in_cluster_generation(N, 0);
    std::vector<int> candidate_members;
    std::vector<int> local_best_members;
    candidate_members.reserve(N);
    local_best_members.reserve(N);
    int generation = 0;
    
    // The outer loop is sequential by definition of QT clustering: the
    // selected cluster changes the set of available seeds.  The expensive
    // independent seed evaluations inside each iteration are distributed in
    // a cyclic layout for balanced work as points are removed.
    while (true) {
        int local_selection[2] = {-1, std::numeric_limits<int>::max()};

        for (int seed = rank; seed < N; ) {
            if (!clustered[seed]) {
                ++generation;
                if (generation == std::numeric_limits<int>::max()) {
                    std::fill(in_cluster_generation.begin(), in_cluster_generation.end(), 0);
                    generation = 1;
                }

                const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                                   threshold, N,
                                                                   max_distances,
                                                                   in_cluster_generation,
                                                                   generation,
                                                                   candidate_members);

                if (cardinality > local_selection[0]) {
                    local_selection[0] = cardinality;
                    local_selection[1] = seed;
                    local_best_members = candidate_members;
                }
            }

            if (seed > N - world_size) break;
            seed += world_size;
        }

        // MPI_MAXLOC maximizes the cardinality first and selects the lowest
        // index on an equal-cardinality tie.  Using the seed as the index
        // therefore retains the original deterministic tie behavior.
        int global_selection[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(local_selection, global_selection, 1, MPI_2INT, MPI_MAXLOC,
                      communicator);

        if (global_selection[0] <= 0) {
            break;
        }

        const int best_seed = global_selection[1];
        const int winner = best_seed % world_size;
        int selected_size = global_selection[0];

        if (rank != winner) {
            local_best_members.clear();
        }
        MPI_Bcast(&selected_size, 1, MPI_INT, winner, communicator);
        if (rank != winner) {
            local_best_members.resize(selected_size);
        }
        MPI_Bcast(local_best_members.data(), selected_size, MPI_INT, winner,
                  communicator);

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = local_best_members;
            clusters.push_back(std::move(cluster));
        }

        for (const int member : local_best_members) {
            clustered[member] = 1;
        }
    }
    
    return clusters;
}

// Broadcast point data in chunks so the benchmark remains usable when the
// number of points exceeds MPI's traditional int-count limit.
void broadcastPoints(std::vector<Point>& points, const int root,
                     const MPI_Comm communicator) {
    auto* data = reinterpret_cast<unsigned char*>(points.data());
    size_t remaining = points.size() * sizeof(Point);
    const size_t max_chunk = static_cast<size_t>(std::numeric_limits<int>::max());

    while (remaining > 0) {
        const int chunk = static_cast<int>(std::min(remaining, max_chunk));
        MPI_Bcast(data, chunk, MPI_BYTE, root, communicator);
        data += chunk;
        remaining -= static_cast<size_t>(chunk);
    }
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
    bool show_help = false;
    bool parse_error = false;
    const char* unknown_option = nullptr;
    
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
            show_help = true;
            break;
        } else {
            parse_error = true;
            unknown_option = argv[i];
        }
    }

    if (rank == 0 && show_help) {
        printUsage(argv[0]);
    }
    if (show_help || parse_error) {
        if (rank == 0 && parse_error) {
            printf("Unknown option: %s\n", unknown_option);
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_error ? 1 : 0;
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
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    broadcastPoints(points, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank,
                                                       world_size, MPI_COMM_WORLD);
    const double local_cluster_time_seconds = MPI_Wtime() - cluster_start;
    double cluster_time_seconds = 0.0;
    MPI_Reduce(&local_cluster_time_seconds, &cluster_time_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long cluster_time_ms = static_cast<long>(cluster_time_seconds * 1000.0);
        printf("Clustering time: %ld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());
    }
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    if (rank == 0) {
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
        const double clusters_per_sec = clusters.size() / cluster_time_seconds;
        const double points_per_sec = num_points / cluster_time_seconds;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
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
    int exit_code = 0;
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exit_code = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return exit_code;
}

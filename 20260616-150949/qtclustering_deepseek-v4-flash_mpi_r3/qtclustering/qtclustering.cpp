// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   Within each clustering iteration, candidate cluster evaluation for
//   each unclustered seed is distributed across MPI ranks (embarrassingly
//   parallel). The rank with the best cluster broadcasts its result so
//   all ranks update their local clustered state synchronously.

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
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
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
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel)
// Candidate seed evaluation is distributed across MPI ranks.
// Within each iteration every rank evaluates a disjoint subset of seeds,
// then the best result (largest cardinality) is gathered to rank 0,
// which selects the global best (with smallest seed index for ties,
// matching sequential semantics). The winning rank broadcasts its
// cluster members so all ranks update their local clustered state.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     int mpi_rank, int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;
    
    if (mpi_rank == 0) {
        clusters.reserve(static_cast<size_t>(N));
    }
    
    while (true) {
        // -------------------------------------------------------
        // Phase 1: Each rank evaluates its cyclic subset of seeds
        // -------------------------------------------------------
        int local_best_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;
        
        for (int i = mpi_rank; i < N; i += mpi_size) {
            if (clustered[i]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(
                i, clustered, points, threshold, N, &candidate_members);
            
            if (cardinality > local_best_cardinality) {
                local_best_cardinality = cardinality;
                local_best_seed = i;
                local_best_members = std::move(candidate_members);
            }
        }
        
        // -------------------------------------------------------
        // Phase 2: Gather local bests to rank 0 and pick global best
        // -------------------------------------------------------
        int sendbuf[2] = {local_best_cardinality, local_best_seed};
        std::vector<int> recvbuf;
        
        if (mpi_rank == 0) {
            recvbuf.resize(2 * mpi_size);
        }
        
        MPI_Gather(sendbuf, 2, MPI_INT,
                   recvbuf.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        
        int global_best_cardinality = -1;
        int global_best_seed = -1;
        int winning_rank = -1;
        
        if (mpi_rank == 0) {
            // First pass: find maximum cardinality
            for (int r = 0; r < mpi_size; ++r) {
                const int card = recvbuf[2 * r];
                if (card > global_best_cardinality) {
                    global_best_cardinality = card;
                }
            }
            
            if (global_best_cardinality > 0) {
                // Second pass: pick smallest seed index with that
                // cardinality (matches sequential tie-breaking where
                // seeds are evaluated in ascending index order)
                global_best_seed = N + 1;
                for (int r = 0; r < mpi_size; ++r) {
                    const int card = recvbuf[2 * r];
                    const int seed = recvbuf[2 * r + 1];
                    if (card == global_best_cardinality && seed < global_best_seed) {
                        global_best_seed = seed;
                        winning_rank = r;
                    }
                }
            }
        }
        
        // -------------------------------------------------------
        // Phase 3: Broadcast global best info to all ranks
        // -------------------------------------------------------
        int results[3] = {global_best_cardinality, global_best_seed, winning_rank};
        MPI_Bcast(results, 3, MPI_INT, 0, MPI_COMM_WORLD);
        global_best_cardinality = results[0];
        global_best_seed       = results[1];
        winning_rank           = results[2];
        
        if (global_best_cardinality <= 0) break;
        
        // -------------------------------------------------------
        // Phase 4: Winning rank broadcasts its cluster members
        // -------------------------------------------------------
        MPI_Bcast(&global_best_cardinality, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);
        
        std::vector<int> winner_members(static_cast<size_t>(global_best_cardinality));
        if (mpi_rank == winning_rank) {
            winner_members = local_best_members;
        }
        MPI_Bcast(winner_members.data(), global_best_cardinality, MPI_INT,
                  winning_rank, MPI_COMM_WORLD);
        
        // -------------------------------------------------------
        // Phase 5: All ranks update their local clustered state
        // -------------------------------------------------------
        for (int i = 0; i < global_best_cardinality; ++i) {
            clustered[winner_members[i]] = true;
        }
        
        // -------------------------------------------------------
        // Phase 6: Rank 0 stores the cluster
        // -------------------------------------------------------
        if (mpi_rank == 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members = winner_members;
            clusters.push_back(std::move(cluster));
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same args)
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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI processes: %d\n", mpi_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (all ranks with same deterministic seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering (parallel)
    double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold,
                                                          mpi_rank, mpi_size);
    
    double cluster_end = MPI_Wtime();
    long cluster_time_ms = static_cast<long>((cluster_end - cluster_start) * 1000.0);
    
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
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
        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
        
        // Print results for external validation (rank 0 only)
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
        
        // Validation (rank 0 only)
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}

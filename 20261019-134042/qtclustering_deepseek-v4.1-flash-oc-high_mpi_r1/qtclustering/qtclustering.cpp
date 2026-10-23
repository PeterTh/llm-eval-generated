// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
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

// Squared Euclidean distance. Using squared distances avoids sqrt in the hot
// path while preserving the exact same ordering and threshold comparisons.
inline double distanceSquared(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster.
//
// The maximum distance from every candidate to the current cluster is kept
// incrementally, which avoids re-scanning all cluster members for every
// candidate. Selection order (ascending candidate index, strict improvement)
// matches the original implementation exactly.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold_sq,
                             const int point_count,
                             std::vector<int>& members,
                             std::vector<int>& in_cluster_stamp,
                             const int stamp,
                             std::vector<double>& max_dist) {
    members.clear();
    members.reserve(static_cast<size_t>(point_count));

    // Add seed point
    in_cluster_stamp[seed_point] = stamp;
    members.push_back(seed_point);

    const Point& seed = points[seed_point];
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || candidate == seed_point) continue;
        max_dist[candidate] = distanceSquared(points[candidate], seed);
    }

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // Try each unclustered point as a candidate, in ascending order.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster_stamp[candidate] == stamp) continue;

            const double max_d = max_dist[candidate];
            if (max_d < threshold_sq && max_d < min_diameter) {
                min_diameter = max_d;
                closest = candidate;
            }
        }

        if (closest < 0) break; // No more points can be added

        in_cluster_stamp[closest] = stamp;
        members.push_back(closest);

        // Account for the newly added member in every remaining candidate's
        // maximum distance to the cluster.
        const Point& added = points[closest];
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster_stamp[candidate] == stamp) continue;
            const double d = distanceSquared(points[candidate], added);
            if (d > max_dist[candidate]) max_dist[candidate] = d;
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm, parallelized with MPI.
//
// Each iteration evaluates every remaining seed, which is the dominant cost.
// Seeds are distributed cyclically across ranks; each rank keeps its locally
// best candidate. A single all-reduce selects the global best using a packed
// key that reproduces the sequential tie-break rule (largest cardinality,
// then smallest seed index). The winning cluster is then broadcast so that
// every rank updates its clustered set identically.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;
    const long long key_base = static_cast<long long>(N) + 1;

    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    std::vector<int> candidate_members;
    std::vector<int> best_members;
    std::vector<int> in_cluster_stamp(N, 0);
    std::vector<double> max_dist(N, 0.0);
    candidate_members.reserve(static_cast<size_t>(N));
    best_members.reserve(static_cast<size_t>(N));

    int stamp = 0;

    // Main clustering loop
    while (true) {
        // Find the first unclustered point. All ranks keep identical state,
        // so this decision is consistent without communication.
        int first_unclustered = -1;
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) {
                first_unclustered = i;
                break;
            }
        }
        if (first_unclustered < 0) break;

        long long local_key = -1;

        // Try the seeds owned by this rank.
        for (int seed = rank; seed < N; seed += nprocs) {
            if (clustered[seed]) continue;

            if (++stamp == INT_MAX) {
                std::fill(in_cluster_stamp.begin(), in_cluster_stamp.end(), 0);
                stamp = 1;
            }

            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold_sq, N,
                candidate_members, in_cluster_stamp, stamp, max_dist);

            // Pack (cardinality, -seed) so that MPI_MAX selects the largest
            // cardinality and, on ties, the smallest seed index.
            const long long key =
                static_cast<long long>(cardinality) * key_base + (N - 1 - seed);

            if (key > local_key) {
                local_key = key;
                best_members = candidate_members;
            }
        }

        long long global_key = -1;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_LONG_LONG, MPI_MAX,
                      MPI_COMM_WORLD);

        const int best_seed = N - 1 - static_cast<int>(global_key % key_base);
        const int best_owner = best_seed % nprocs;

        int count = (rank == best_owner)
                        ? static_cast<int>(best_members.size())
                        : 0;
        MPI_Bcast(&count, 1, MPI_INT, best_owner, MPI_COMM_WORLD);
        if (rank != best_owner) {
            best_members.resize(static_cast<size_t>(count));
        }
        MPI_Bcast(best_members.data(), count, MPI_INT, best_owner,
                  MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(std::move(cluster));

        // Mark all members as clustered
        for (int member : best_members) {
            clustered[member] = 1;
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI ranks: %d\n", nprocs);
    }
    
    // Generate synthetic data on rank 0 and distribute it so that all ranks
    // operate on exactly the same points as the sequential version.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * static_cast<int>(sizeof(Point)),
              MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, nprocs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    int exit_code = 0;

    if (rank == 0) {
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
        const double time_sec = cluster_time.count() / 1000.0;
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
                exit_code = 1;
            }
        }
    }
    
    MPI_Finalize();
    return exit_code;
}

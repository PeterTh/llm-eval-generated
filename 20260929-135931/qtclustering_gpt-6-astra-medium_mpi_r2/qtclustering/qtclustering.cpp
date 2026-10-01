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
        // The original generator cannot make progress for N <= 30.
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

// A rank stores only the neighborhoods of the seeds it owns. Entries stay
// in point-index order, preserving the sequential closest-point tie break.
struct Neighbor {
    int point;
    double diameter;
};

// Maintain each eligible point's maximum distance to the growing cluster.
// Rejected points can never become eligible again as the diameter only grows.
int generateCandidateCluster(int seed, const std::vector<Neighbor>& neighbors,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points, double threshold,
                             int best_size, std::vector<Neighbor>& candidates,
                             std::vector<int>& members) {
    candidates.clear();
    for (const auto& neighbor : neighbors) {
        if (!clustered[neighbor.point]) candidates.push_back(neighbor);
    }
    members.clear();
    members.push_back(seed);
    while (!candidates.empty()) {
        // Seeds are visited in ascending order on each rank, so equal-sized
        // candidates cannot displace that rank's existing winner.
        if (members.size() + candidates.size() <= static_cast<size_t>(best_size))
            return 0;
        size_t closest = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].diameter < candidates[closest].diameter)
                closest = i;
        }
        const int next = candidates[closest].point;
        members.push_back(next);
        size_t kept = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == closest) continue;
            Neighbor candidate = candidates[i];
            candidate.diameter = std::max(candidate.diameter,
                                           distance(points[candidate.point], points[next]));
            if (candidate.diameter < threshold) candidates[kept++] = candidate;
        }
        candidates.resize(kept);
    }
    return static_cast<int>(members.size());
}

// All ranks participate; only rank zero retains the output clusters.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Cyclic ownership spreads spatially correlated input groups across ranks.
    // Precomputation and neighborhood storage are distributed, not replicated.
    std::vector<int> seeds;
    std::vector<std::vector<Neighbor>> neighborhoods;
    for (size_t seed = rank; seed < points.size(); seed += ranks) {
        seeds.push_back(static_cast<int>(seed));
        neighborhoods.emplace_back();
        auto& neighbors = neighborhoods.back();
        for (int candidate = 0; candidate < N; ++candidate) {
            if (candidate == static_cast<int>(seed)) continue;
            const double d = distance(points[candidate], points[seed]);
            if (d < threshold) neighbors.push_back({candidate, d});
        }
    }

    std::vector<Neighbor> candidates;
    std::vector<int> members, best_members;
    int remaining = N;
    while (remaining > 0) {
        // MPI_MAXLOC selects maximum cardinality and then the smallest seed,
        // exactly matching the sequential scan, regardless of rank count.
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        for (size_t i = 0; i < seeds.size(); ++i) {
            const int seed = seeds[i];
            if (clustered[seed]) continue;
            if (neighborhoods[i].size() + 1 <= static_cast<size_t>(local_best[0]))
                continue;
            const int size = generateCandidateCluster(seed, neighborhoods[i], clustered,
                points, threshold, local_best[0], candidates, members);
            if (size > local_best[0]) {
                local_best[0] = size;
                local_best[1] = seed;
                best_members.swap(members);
                if (size == remaining) break; // No later seed can beat this.
            }
        }
        int winner[2];
        MPI_Allreduce(local_best, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // If no seed can grow, every remaining cluster is a singleton.
        // Finish in index order without one collective per remaining point.
        if (winner[0] == 1) {
            if (rank == 0) {
                for (int seed = 0; seed < N; ++seed) {
                    if (!clustered[seed]) clusters.push_back({{seed}, seed});
                }
            }
            break;
        }
        const int owner = winner[1] % ranks;
        best_members.resize(winner[0]);
        MPI_Bcast(best_members.data(), winner[0], MPI_INT, owner, MPI_COMM_WORLD);
        for (int member : best_members) clustered[member] = 1;
        remaining -= winner[0];
        if (rank == 0) clusters.push_back({best_members, winner[1]});
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate once and replicate the O(N) coordinates. The O(N^2) distance
    // work and seed neighborhoods are distributed by qtClustering.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Keep output and validation on rank zero, then share its exit status.
    int result = 0;
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
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}

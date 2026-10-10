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

// Each rank owns cyclically assigned seeds and only their neighborhoods.
// Points and the removal mask are replicated; no full distance matrix is stored.
struct Neighbor {
    int point;
    double max_distance;
};

struct SeedWork {
    int seed;
    std::vector<Neighbor> neighbors;
    std::vector<int> members;
};

// Maintain each candidate's maximum distance incrementally. A point that fails
// the strict threshold can never become eligible again as this cluster grows.
void generateCandidateCluster(SeedWork& work,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              std::vector<Neighbor>& candidates) {
    candidates.clear();
    for (const Neighbor& neighbor : work.neighbors) {
        if (!clustered[neighbor.point]) candidates.push_back(neighbor);
    }
    work.members.clear();
    work.members.push_back(work.seed);

    while (!candidates.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            // Stable compaction preserves ascending point indices, so strict
            // comparison reproduces the sequential closest-point tie break.
            if (candidates[i].max_distance < candidates[best].max_distance) {
                best = i;
            }
        }
        const int member = candidates[best].point;
        work.members.push_back(member);
        size_t remaining = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == best) continue;
            Neighbor candidate = candidates[i];
            candidate.max_distance = std::max(candidate.max_distance,
                distance(points[candidate.point], points[member]));
            if (candidate.max_distance < threshold) {
                candidates[remaining++] = candidate;
            }
        }
        candidates.resize(remaining);
    }
}

// All ranks participate in every round. Only rank zero retains the final
// clusters for reporting; the elected cluster's members are broadcast to all.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<SeedWork> seeds;
    seeds.reserve(N / process_count + (rank < N % process_count));
    for (long long seed = rank; seed < N; seed += process_count) {
        SeedWork work;
        work.seed = static_cast<int>(seed);
        for (int candidate = 0; candidate < N; ++candidate) {
            if (candidate == seed) continue;
            const double dist = distance(points[candidate], points[seed]);
            // Keep the original sqrt and strict comparison, including at
            // threshold boundaries, rather than compare squared distances.
            if (dist < threshold) work.neighbors.push_back({candidate, dist});
        }
        seeds.push_back(std::move(work));
    }

    std::vector<Cluster> clusters;
    std::vector<Neighbor> candidates;
    std::vector<int> selected_members;
    int remaining = N;
    while (remaining > 0) {
        // MPI_MAXLOC chooses largest cardinality, then smallest seed index.
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        for (SeedWork& work : seeds) {
            if (clustered[work.seed]) continue;
            // Removing points outside a cached cluster cannot change any of
            // its greedy selections. Rebuild only if a member was removed.
            const bool stale = work.members.empty() ||
                std::any_of(work.members.begin(), work.members.end(),
                            [&clustered](int member) { return clustered[member]; });
            if (stale) {
                generateCandidateCluster(work, clustered, points, threshold, candidates);
            }
            const int cardinality = static_cast<int>(work.members.size());
            if (cardinality > local_best[0]) {
                local_best[0] = cardinality;
                local_best[1] = work.seed;
            }
        }

        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        // If every candidate is a singleton, later removals cannot create a
        // larger cluster. Emit the remaining seeds in the original order and
        // avoid one pair of collectives per point in the sparse tail.
        if (global_best[0] == 1) {
            if (rank == 0) {
                for (int seed = 0; seed < N; ++seed) {
                    if (!clustered[seed]) clusters.push_back({{seed}, seed});
                }
            }
            break;
        }
        const int best_seed = global_best[1];
        const int owner = best_seed % process_count;
        selected_members.resize(global_best[0]);
        if (rank == owner) {
            selected_members = seeds[best_seed / process_count].members;
        }
        MPI_Bcast(selected_members.data(), global_best[0], MPI_INT, owner, MPI_COMM_WORLD);
        for (int member : selected_members) clustered[member] = 1;
        remaining -= global_best[0];
        if (rank == 0) clusters.push_back({selected_members, best_seed});
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
    // Finalize MPI on all normal exits, including help and invalid arguments.
    struct MpiLifetime {
        ~MpiLifetime() { MPI_Finalize(); }
    } mpi_lifetime;
    int rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                              num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate once, then distribute exactly the same coordinates to all ranks.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be two packed doubles");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, process_count);
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    printf("Clustering time: %lld ms\n", static_cast<long long>(time_sec * 1000.0));
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

// QT Clustering Benchmark - Distributed MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
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

struct CandidatePoint {
    int point;
    double max_distance;
};

// Each rank owns only its seeds' threshold neighborhoods and cached clusters.
struct SeedCandidate {
    int seed;
    std::vector<CandidatePoint> neighbors;
    std::vector<int> members;
    bool valid = false;
};

inline bool canImprove(const int cardinality, const int seed,
                       const int best_cardinality, const int best_seed) {
    return cardinality > best_cardinality ||
           (cardinality == best_cardinality && seed < best_seed);
}

// Maintain each candidate's maximum distance incrementally. A rejected point
// can never become eligible again as the cluster grows. Stable compaction keeps
// point indices ascending, preserving the sequential algorithm's distance ties.
// A zero result means an upper bound proved this seed cannot win; partial
// clusters must never be cached as completed candidate clusters.
int generateCandidateCluster(const SeedCandidate& seed,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int best_cardinality, const int best_seed,
                             std::vector<CandidatePoint>& active,
                             std::vector<int>& members) {
    active.clear();
    members.clear();
    members.push_back(seed.seed);

    size_t closest = 0;
    double min_distance = std::numeric_limits<double>::max();
    for (const CandidatePoint& neighbor : seed.neighbors) {
        if (clustered[neighbor.point]) continue;
        if (neighbor.max_distance < min_distance) {
            min_distance = neighbor.max_distance;
            closest = active.size();
        }
        active.push_back(neighbor);
    }

    while (!active.empty()) {
        const int upper_bound = static_cast<int>(members.size() + active.size());
        if (!canImprove(upper_bound, seed.seed, best_cardinality, best_seed)) {
            return 0;
        }

        const int member = active[closest].point;
        members.push_back(member);
        size_t kept = 0;
        size_t next_closest = 0;
        min_distance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < active.size(); ++i) {
            if (i == closest) continue;
            CandidatePoint candidate = active[i];
            candidate.max_distance = std::max(
                candidate.max_distance, distance(points[candidate.point], points[member]));
            if (candidate.max_distance < threshold) {
                if (candidate.max_distance < min_distance) {
                    min_distance = candidate.max_distance;
                    next_closest = kept;
                }
                active[kept++] = candidate;
            }
        }
        active.resize(kept);
        closest = next_closest;
    }
    return static_cast<int>(members.size());
}

// All ranks participate; only rank zero retains the final ordered clusters.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Cyclic ownership spreads nearby, similarly expensive seeds across ranks.
    const int local_count = N / rank_count + (rank < N % rank_count ? 1 : 0);
    std::vector<SeedCandidate> seeds(local_count);
    size_t max_neighbors = 0;
    for (size_t i = 0; i < seeds.size(); ++i) {
        SeedCandidate& seed = seeds[i];
        seed.seed = static_cast<int>(i * rank_count + rank);
        for (int candidate = 0; candidate < N; ++candidate) {
            if (candidate == seed.seed) continue;
            const double dist = distance(points[candidate], points[seed.seed]);
            if (dist < threshold) seed.neighbors.push_back({candidate, dist});
        }
        max_neighbors = std::max(max_neighbors, seed.neighbors.size());
    }

    // Reuse scratch storage across all seeds and rounds.
    std::vector<CandidatePoint> active;
    std::vector<int> members;
    active.reserve(max_neighbors);
    members.reserve(max_neighbors + 1);
    int remaining = N;
    while (remaining > 0) {
        int best_cardinality = 0;
        int best_seed = std::numeric_limits<int>::max();

        // Removing points outside a cached cluster leaves its entire greedy
        // selection sequence unchanged. Start with these exact cached results
        // to obtain a strong pruning bound before rebuilding affected seeds.
        for (SeedCandidate& seed : seeds) {
            if (clustered[seed.seed] || !seed.valid) continue;
            for (int member : seed.members) {
                if (clustered[member]) {
                    seed.valid = false;
                    break;
                }
            }
            const int cardinality = static_cast<int>(seed.members.size());
            if (seed.valid && canImprove(cardinality, seed.seed,
                                         best_cardinality, best_seed)) {
                best_cardinality = cardinality;
                best_seed = seed.seed;
            }
        }

        for (SeedCandidate& seed : seeds) {
            if (clustered[seed.seed] || seed.valid) continue;
            const int upper_bound = static_cast<int>(seed.neighbors.size() + 1);
            if (!canImprove(upper_bound, seed.seed, best_cardinality, best_seed)) continue;
            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, best_cardinality, best_seed,
                active, members);
            if (cardinality == 0) continue;
            seed.members = members;
            seed.valid = true;
            if (canImprove(cardinality, seed.seed, best_cardinality, best_seed)) {
                best_cardinality = cardinality;
                best_seed = seed.seed;
            }
        }

        // MINLOC on (-size, seed) selects the largest cluster, then the lowest
        // original point index, exactly as the ascending serial seed scan does.
        int local_best[2] = {-best_cardinality, best_seed};
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        const int cardinality = -global_best[0];
        const int seed = global_best[1];

        // If the global maximum is one, no remaining pair is compatible.
        // Finish the singleton tail in serial seed order without N collectives.
        if (cardinality == 1) {
            if (rank == 0) {
                for (int point = 0; point < N; ++point) {
                    if (!clustered[point]) clusters.push_back({{point}, point});
                }
            }
            break;
        }

        const int owner = seed % rank_count;
        members.resize(cardinality);
        if (rank == owner) members = seeds[seed / rank_count].members;
        MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        for (int member : members) clustered[member] = 1;
        remaining -= cardinality;
        if (rank == 0) clusters.push_back({members, seed});
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

int runBenchmark(int argc, char** argv, const int rank) {
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
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
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
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (rank != 0) return 0;
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_code;
    try {
        exit_code = runBenchmark(argc, argv, rank);
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exit_code;
}

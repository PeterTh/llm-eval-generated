// QT Clustering Benchmark - Distributed-memory MPI Version
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

struct Neighbor {
    int point;
    double diameter;
};

// Only the owning rank stores a seed's neighborhood and cached candidate.
struct SeedState {
    std::vector<Neighbor> neighbors;
    std::vector<int> members;
};

// Maintain each remaining point's maximum distance incrementally. Points that
// exceed the threshold can never become eligible again as the cluster grows.
void generateCandidateCluster(const int seed,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              SeedState& state,
                              std::vector<Neighbor>& candidates) {
    state.members.clear();
    state.members.push_back(seed);
    candidates.clear();
    for (const auto& neighbor : state.neighbors) {
        if (!clustered[neighbor.point]) candidates.push_back(neighbor);
    }

    while (!candidates.empty()) {
        size_t best = candidates.size();
        double min_diameter = std::numeric_limits<double>::max();
        // Stable point order and strict comparison retain the original ties.
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].diameter < min_diameter) {
                best = i;
                min_diameter = candidates[i].diameter;
            }
        }
        if (best == candidates.size()) break;
        const int closest = candidates[best].point;
        state.members.push_back(closest);

        size_t count = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == best) continue;
            Neighbor candidate = candidates[i];
            candidate.diameter = std::max(candidate.diameter,
                                         distance(points[candidate.point], points[closest]));
            if (candidate.diameter < threshold) candidates[count++] = candidate;
        }
        candidates.resize(count);
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    std::vector<unsigned char> clustered(N, false);
    std::vector<Cluster> clusters;
    const int local_count = N / ranks + (rank < N % ranks ? 1 : 0);
    std::vector<SeedState> seeds(local_count);
    std::vector<Neighbor> candidates;

    // Cyclic ownership spreads spatially grouped seeds across ranks. Neighborhood
    // storage is distributed, rather than replicating an N-by-N distance matrix.
    for (int i = 0; i < local_count; ++i) {
        const int seed = rank + i * ranks;
        for (int point = 0; point < N; ++point) {
            if (point == seed) continue;
            const double dist = distance(points[point], points[seed]);
            if (dist < threshold) seeds[i].neighbors.push_back({point, dist});
        }
    }

    int remaining = N;
    while (remaining > 0) {
        struct { int cardinality; int seed; } local_best = {-1, N}, global_best;
        for (int i = 0; i < local_count; ++i) {
            const int seed = rank + i * ranks;
            if (clustered[seed]) continue;
            auto& state = seeds[i];
            // Removing a point outside the cached result cannot change any
            // greedy choice. Recompute only when a member has been removed.
            if (state.members.empty() ||
                std::any_of(state.members.begin(), state.members.end(),
                            [&clustered](int point) { return clustered[point]; })) {
                generateCandidateCluster(seed, clustered, points, threshold, state, candidates);
            }
            const int cardinality = static_cast<int>(state.members.size());
            if (cardinality > local_best.cardinality) local_best = {cardinality, seed};
        }

        // MAXLOC selects the largest cluster, then the smallest seed, exactly
        // matching the sequential seed traversal, independently of rank count.
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best.cardinality == 1) {
            // All remaining clusters are singletons; their order is known on
            // every rank, so avoid a collective pair for each isolated point.
            if (rank == 0) {
                for (int point = 0; point < N; ++point) {
                    if (!clustered[point]) clusters.push_back({{point}, point});
                }
            }
            break;
        }
        const int owner = global_best.seed % ranks;
        std::vector<int> members(global_best.cardinality);
        if (rank == owner) members = seeds[global_best.seed / ranks].members;
        MPI_Bcast(members.data(), global_best.cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        for (const int point : members) {
            clustered[point] = true;
            if (point % ranks == rank) {
                // Release state for seeds that can no longer be considered.
                SeedState empty;
                std::swap(seeds[point / ranks], empty);
            }
        }
        remaining -= global_best.cardinality;
        if (rank == 0) clusters.push_back({std::move(members), global_best.seed});
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

// Finalize MPI on every normal return, including help and invalid arguments.
struct MPIEnvironment {
    MPIEnvironment(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
    ~MPIEnvironment() { MPI_Finalize(); }
};

int main(int argc, char** argv) {
    MPIEnvironment mpi(argc, argv);
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
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two packed doubles");
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
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    // Only rank zero owns the output clusters and emits benchmark results.
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

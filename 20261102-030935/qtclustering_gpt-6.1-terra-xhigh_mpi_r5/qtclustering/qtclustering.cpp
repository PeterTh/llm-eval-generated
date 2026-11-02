// QT Clustering Benchmark - MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};
static_assert(sizeof(Point) == 2 * sizeof(double) &&
              offsetof(Point, y) == sizeof(double));

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

// Build a candidate while maintaining each remaining point's maximum distance
// to the members already selected. Candidates stay in point-index order, so
// equal distances are resolved exactly as in the original scan.
struct CandidatePoint {
    int index;
    double max_distance;
};

void generateCandidateCluster(int seed, const std::vector<int>& unclustered,
                              const std::vector<Point>& points, double threshold,
                              std::vector<int>& members,
                              std::vector<CandidatePoint>& candidates) {
    members.clear();
    candidates.clear();
    members.push_back(seed);
    for (int index : unclustered) {
        if (index == seed) continue;
        const double d = std::max(0.0, distance(points[index], points[seed]));
        if (d < threshold && d < std::numeric_limits<double>::max())
            candidates.push_back({index, d});
    }

    while (!candidates.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].max_distance < candidates[best].max_distance)
                best = i;
        }
        const int selected = candidates[best].index;
        members.push_back(selected);

        size_t kept = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == best) continue;
            CandidatePoint candidate = candidates[i];
            candidate.max_distance = std::max(candidate.max_distance,
                distance(points[candidate.index], points[selected]));
            if (candidate.max_distance < threshold &&
                candidate.max_distance < std::numeric_limits<double>::max())
                candidates[kept++] = candidate;
        }
        candidates.resize(kept);
    }
}

// Each seed belongs to one rank. A cached candidate remains valid as long as
// none of its members has been removed: removing points that it never chose
// cannot change any of its choices. Cap the cache to bound per-rank memory.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, int rank, int ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;
    std::vector<Cluster> clusters;
    std::vector<std::vector<int>> cache(N);
    constexpr size_t cache_limit = 64 * 1024 * 1024;
    size_t cache_bytes = 0;
    std::vector<int> members, local_best;
    std::vector<CandidatePoint> candidates;
    members.reserve(N);
    candidates.reserve(N);

    while (!unclustered_indices.empty()) {
        int local_result[2] = {0, std::numeric_limits<int>::max()};
        local_best.clear();
        for (int seed : unclustered_indices) {
            if (seed % ranks != rank) continue;
            auto& saved = cache[seed];
            if (!saved.empty() && std::any_of(saved.begin(), saved.end(),
                    [&clustered](int index) { return clustered[index] != 0; })) {
                cache_bytes -= saved.capacity() * sizeof(int);
                std::vector<int>().swap(saved);
            }
            const std::vector<int>* candidate = &saved;
            if (saved.empty()) {
                generateCandidateCluster(seed, unclustered_indices, points,
                                         threshold, members, candidates);
                const size_t bytes = members.size() * sizeof(int);
                if (bytes <= cache_limit - cache_bytes) {
                    saved = members;
                    const size_t allocated = saved.capacity() * sizeof(int);
                    if (allocated <= cache_limit - cache_bytes) {
                        cache_bytes += allocated;
                        candidate = &saved;
                    } else {
                        std::vector<int>().swap(saved);
                        candidate = &members;
                    }
                } else {
                    candidate = &members;
                }
            }
            const int cardinality = static_cast<int>(candidate->size());
            if (cardinality > local_result[0]) {
                local_result[0] = cardinality;
                local_result[1] = seed;
                local_best = *candidate;
            }
        }

        int global_result[2];
        MPI_Allreduce(local_result, global_result, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        // If every candidate is a singleton, removing points cannot make a
        // larger cluster possible. Finish in the original seed order.
        if (global_result[0] == 1) {
            if (rank == 0) {
                for (int seed : unclustered_indices)
                    clusters.push_back({{seed}, seed});
            }
            break;
        }
        const int best_seed = global_result[1];
        const int winner = best_seed % ranks;
        if (rank != winner) local_best.resize(global_result[0]);
        MPI_Bcast(local_best.data(), global_result[0], MPI_INT, winner,
                  MPI_COMM_WORLD);
        for (int index : local_best) {
            clustered[index] = 1;
            auto& saved = cache[index];
            cache_bytes -= saved.capacity() * sizeof(int);
            std::vector<int>().swap(saved);
        }
        if (rank == 0) clusters.push_back({local_best, best_seed});
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long elapsed_ms = cluster_time.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(elapsed_ms);
    
    if (rank != 0) {
        int status;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return status;
    }
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
    int status = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

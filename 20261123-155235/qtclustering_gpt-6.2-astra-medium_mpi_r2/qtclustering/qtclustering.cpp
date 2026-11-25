// QT Clustering Benchmark - MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
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

struct Candidate {
    int point;
    double max_distance;
};

// Only distances to the newest member need updating. Break distance ties by
// point index to reproduce the original scan order.
static void growCluster(int seed, const std::vector<Point>& points, double threshold,
                        const std::vector<int>& available, int best_size,
                        std::vector<Candidate>& candidates, std::vector<int>& members) {
    candidates.clear();
    // A candidate must lie within one threshold of the seed along x.
    const double x = points[seed].x;
    auto first = std::partition_point(available.begin(), available.end(), [&](int p) {
        return x - points[p].x > threshold;
    });
    for (auto it = first; it != available.end(); ++it) {
        const int point = *it;
        if (points[point].x - x > threshold) break;
        if (point == seed) continue;
        const double dist = distance(points[point], points[seed]);
        if (dist < threshold) candidates.push_back({point, dist});
    }

    members.assign(1, seed);
    while (!candidates.empty()) {
        // Earlier local seeds win ties. An incomplete cluster is never cached.
        if (members.size() + candidates.size() <= static_cast<size_t>(best_size)) {
            members.clear();
            return;
        }
        auto closest = std::min_element(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.max_distance < b.max_distance ||
                       (a.max_distance == b.max_distance && a.point < b.point);
            });
        const int next = closest->point;
        members.push_back(next);
        size_t kept = 0;
        for (const Candidate& candidate : candidates) {
            if (candidate.point == next) continue;
            const double dist = distance(points[candidate.point], points[next]);
            if (dist < threshold) {
                candidates[kept++] = {candidate.point, std::max(candidate.max_distance, dist)};
            }
        }
        candidates.resize(kept);
    }
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, int rank, int ranks) {
    std::vector<int> available(points.size());
    std::iota(available.begin(), available.end(), 0);
    std::sort(available.begin(), available.end(), [&](int a, int b) {
        return points[a].x < points[b].x;
    });
    std::vector<char> removed(points.size(), false);
    // Interleave nearby seeds across ranks, including seeds from the same
    // synthetic group. Each candidate's membership lives only on its owner.
    std::vector<std::vector<int>> cached((points.size() + ranks - 1 - rank) / ranks);
    std::vector<Candidate> candidates;
    std::vector<Cluster> clusters;
    while (!available.empty()) {
        int local[2] = {0, 0};
        for (size_t i = 0; i < cached.size(); ++i) {
            const int seed = static_cast<int>(i * ranks + rank);
            if (removed[seed]) continue;
            auto& members = cached[i];
            // Removing an unchosen point cannot change any greedy choice.
            if (members.empty() || std::any_of(members.begin(), members.end(),
                                               [&](int p) { return removed[p]; })) {
                growCluster(seed, points, threshold, available, local[0], candidates, members);
            }
            if (static_cast<int>(members.size()) > local[0]) {
                local[0] = static_cast<int>(members.size());
                local[1] = seed;
                if (members.size() == available.size()) break;
            }
        }

        int winner[2];
        MPI_Allreduce(local, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int size = winner[0];
        const int seed = winner[1];
        if (size == 1) {
            // No remaining pair is within the threshold.
            if (rank == 0) {
                std::sort(available.begin(), available.end());
                for (int point : available) clusters.push_back({{point}, point});
            }
            break;
        }
        const int owner = seed % ranks;
        std::vector<int> members;
        if (rank == owner) members.swap(cached[seed / ranks]);
        else members.resize(size);
        MPI_Bcast(members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        for (int point : members) removed[point] = true;
        available.erase(std::remove_if(available.begin(), available.end(),
                                       [&](int p) { return removed[p]; }), available.end());
        if (rank == 0) clusters.push_back({std::move(members), seed});
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

static int run(int argc, char** argv, int rank, int ranks) {
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
    
    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const auto clusters = qtClustering(points, threshold, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double time_sec;
    MPI_Reduce(&elapsed, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    printf("Clustering time: %.0f ms\n", time_sec * 1000.0);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

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

        // The original formula yields zero for every draw when N <= 30,
        // which would never make progress.  Keep the original stream and
        // behavior for normal benchmark sizes while making all valid positive
        // input sizes terminate.
        if (N <= 30 && group_cnt == 0) {
            group_cnt = 1;
        }
        
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

// Workspace reused for every seed evaluated by a rank.  For a fixed seed, the
// largest distance from every candidate to the current cluster is maintained
// incrementally.  The original implementation recomputed this same maximum
// from every member on every iteration.
struct CandidateWorkspace {
    std::vector<double> max_distance;
    std::vector<unsigned char> in_cluster;
    std::vector<int> members;

    explicit CandidateWorkspace(const int point_count)
        : max_distance(point_count, 0.0),
          in_cluster(point_count, 0) {
        members.reserve(point_count);
    }

    void reset() {
        std::fill(in_cluster.begin(), in_cluster.end(), 0);
        members.clear();
    }
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster.  Candidate indices are
// examined in increasing order, retaining the original deterministic tie
// breaking behavior.
int generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              CandidateWorkspace& workspace,
                              std::vector<int>* cluster_members = nullptr) {
    workspace.reset();

    workspace.in_cluster[seed_point] = 1;
    workspace.members.push_back(seed_point);

    // With only the seed in the cluster, this is exactly the first distance
    // calculation performed by the sequential implementation.
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (!clustered[candidate] && candidate != seed_point) {
            workspace.max_distance[candidate] = distance(points[candidate],
                                                         points[seed_point]);
        }
    }

    while (static_cast<int>(workspace.members.size()) < point_count) {
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || workspace.in_cluster[candidate]) {
                continue;
            }

            const double max_dist = workspace.max_distance[candidate];
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) {
            break;
        }

        workspace.in_cluster[closest_point] = 1;
        workspace.members.push_back(closest_point);

        // The maximum can only increase as members are added.  Candidates
        // already at or above the threshold can therefore be skipped.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || workspace.in_cluster[candidate] ||
                workspace.max_distance[candidate] >= threshold) {
                continue;
            }

            workspace.max_distance[candidate] = std::max(
                workspace.max_distance[candidate],
                distance(points[candidate], points[closest_point]));
        }
    }

    if (cluster_members != nullptr) {
        *cluster_members = workspace.members;
    }

    return static_cast<int>(workspace.members.size());
}

// Main QT clustering algorithm.  Every rank retains the point set and the
// current clustered bitmap, while the independent seed trials in each QT
// iteration are distributed across ranks.  The globally best seed is then
// selected with MPI_MAXLOC: maximize cardinality and, on a tie, minimize the
// seed index exactly as the original sequential loop does.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int world_size,
                                  MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    CandidateWorkspace workspace(N);

    // Main clustering loop
    while (true) {
        int local_best[2] = {-1, N};
        std::vector<int> best_cluster_members;

        // Cyclic assignment keeps the variable-cost seed trials balanced as
        // cluster sizes change between iterations.
        for (int seed = rank; seed < N; seed += world_size) {
            if (clustered[seed]) {
                continue;
            }

            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, N, workspace);

            if (cardinality > local_best[0]) {
                local_best[0] = cardinality;
                local_best[1] = seed;
                best_cluster_members = workspace.members;
            }
        }

        int global_best[2] = {-1, N};
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC,
                      communicator);

        if (global_best[0] <= 0 || global_best[1] >= N) {
            break;
        }

        // Identify the rank that already computed the winning candidate so
        // that its member order can be sent without regenerating the cluster.
        const int local_owner =
            (local_best[0] == global_best[0] &&
             local_best[1] == global_best[1]) ? rank : world_size;
        int owner = world_size;
        MPI_Allreduce(&local_owner, &owner, 1, MPI_INT, MPI_MIN,
                      communicator);

        int selected_size = global_best[0];
        if (rank == owner) {
            selected_size = static_cast<int>(best_cluster_members.size());
        }
        MPI_Bcast(&selected_size, 1, MPI_INT, owner, communicator);

        std::vector<int> selected_members(selected_size);
        if (rank == owner) {
            selected_members = best_cluster_members;
        }
        MPI_Bcast(selected_members.data(), selected_size, MPI_INT, owner,
                  communicator);

        Cluster cluster;
        cluster.seed_point = global_best[1];
        cluster.members = std::move(selected_members);
        clusters.push_back(std::move(cluster));

        for (const int member : clusters.back().members) {
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
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    
    // Parse command line arguments on every rank so all ranks take the same
    // control path before entering collectives.
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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError || num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or incomplete command-line option\n");
            } else if (!showHelp && (num_points <= 0 || threshold <= 0.0)) {
                printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                       num_points, threshold);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (parseError || (!showHelp && (num_points <= 0 || threshold <= 0.0)))
                   ? 1
                   : 0;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate data once and broadcast it.  This avoids repeating generation
    // work and ensures every rank uses precisely the same point coordinates.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }

    static_assert(sizeof(Point) == 2 * sizeof(double),
                  "Point must be two contiguous doubles for MPI broadcast");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // All ranks participate in the timed distributed computation.  The
    // maximum elapsed time is reported because it is the wall-clock time of
    // the slowest rank.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, world_size, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time_seconds = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long cluster_time_ms = static_cast<long>(cluster_time_seconds * 1000.0);
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
        const double time_sec = std::max(cluster_time_seconds,
                                         std::numeric_limits<double>::min());
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
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}

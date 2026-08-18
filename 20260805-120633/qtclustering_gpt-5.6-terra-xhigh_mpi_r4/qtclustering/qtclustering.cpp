// QT Clustering Benchmark - MPI distributed-memory implementation
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

// Storage reused for every seed evaluated by a rank.  Keeping the running
// maximum distance for each candidate makes adding a member O(number of
// remaining points), rather than recomputing all prior member distances.
// The update order is the same as the original implementation, so its strict
// comparison and lowest-index tie-breaking semantics are preserved.
struct CandidateWorkspace {
    explicit CandidateWorkspace(const int point_count)
        : in_cluster(point_count), max_distance(point_count) {
        members.reserve(point_count);
    }

    std::vector<unsigned char> in_cluster;
    std::vector<double> max_distance;
    std::vector<int> members;
};

// Generate one candidate cluster.  active_indices is sorted in point-index
// order and is exactly the set that has not yet been assigned to a cluster.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& active_indices,
                             const std::vector<Point>& points,
                             const double threshold,
                             CandidateWorkspace& workspace,
                             std::vector<int>* cluster_members = nullptr) {
    // Only clear flags that the preceding candidate set.  A full point-count
    // reset here becomes costly in late rounds when few points remain.
    for (const int member : workspace.members) {
        workspace.in_cluster[member] = 0;
    }
    workspace.members.clear();
    workspace.in_cluster[seed_point] = 1;
    workspace.members.push_back(seed_point);

    // This is the first pass over the one-member cluster in the sequential
    // code.  Subsequent passes incrementally append exactly one distance.
    for (const int candidate : active_indices) {
        if (candidate != seed_point) {
            workspace.max_distance[candidate] = distance(points[candidate], points[seed_point]);
        }
    }

    while (workspace.members.size() < active_indices.size()) {
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // active_indices retains the original increasing candidate order.
        for (const int candidate : active_indices) {
            if (workspace.in_cluster[candidate]) {
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

        // Preserve the member insertion order used by the sequential maximum
        // reduction: seed, then every previously selected closest point.
        for (const int candidate : active_indices) {
            if (!workspace.in_cluster[candidate]) {
                const double candidate_distance = distance(points[candidate], points[closest_point]);
                if (candidate_distance > workspace.max_distance[candidate]) {
                    workspace.max_distance[candidate] = candidate_distance;
                }
            }
        }
    }

    if (cluster_members != nullptr) {
        *cluster_members = workspace.members;
    }
    return static_cast<int>(workspace.members.size());
}

// Distributed QT clustering.  Every rank owns a cyclic subset of seed
// evaluations, while all ranks retain the small shared state needed to begin
// the next greedy round.  MPI_MAXLOC selects candidates deterministically:
// largest cardinality first, then the lowest seed, matching the serial loop.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     const int rank,
                                     const int world_size) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> active_indices(point_count);
    std::vector<Cluster> clusters;
    CandidateWorkspace workspace(point_count);

    for (int point = 0; point < point_count; ++point) {
        active_indices[point] = point;
    }
    if (rank == 0) {
        clusters.reserve(point_count);
    }

    while (!active_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = std::numeric_limits<int>::max();

        // Cyclic assignment balances seed workloads without changing the
        // globally selected seed or the serial candidate order.
        for (size_t position = static_cast<size_t>(rank);
             position < active_indices.size();
             position += static_cast<size_t>(world_size)) {
            const int seed = active_indices[position];
            const int cardinality = generateCandidateCluster(
                seed, active_indices, points, threshold, workspace);
            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
            }
        }

        // MPI_MAXLOC returns the minimum location for equal values.  Using
        // the seed as that location exactly implements the serial tie-break.
        int local_choice[2] = {local_max_cardinality, local_best_seed};
        int global_choice[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        const int max_cardinality = global_choice[0];
        const int best_seed = global_choice[1];
        if (max_cardinality < 1) {
            break;
        }

        // If every remaining candidate is a singleton, no later serial round
        // can create a multi-point cluster.  Finish the equivalent suffix
        // locally and avoid one collective per remaining point.
        if (max_cardinality == 1) {
            if (rank == 0) {
                for (const int seed : active_indices) {
                    clusters.push_back({{seed}, seed});
                }
            }
            break;
        }

        const auto seed_position = std::lower_bound(active_indices.begin(), active_indices.end(),
                                                    best_seed);
        const int winner_rank = static_cast<int>(
            (seed_position - active_indices.begin()) % world_size);

        std::vector<int> best_cluster_members;
        if (rank == winner_rank) {
            generateCandidateCluster(best_seed, active_indices, points, threshold,
                                     workspace, &best_cluster_members);
        }

        int member_count = static_cast<int>(best_cluster_members.size());
        MPI_Bcast(&member_count, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);
        if (rank != winner_rank) {
            best_cluster_members.resize(member_count);
        }
        MPI_Bcast(best_cluster_members.data(), member_count, MPI_INT, winner_rank,
                  MPI_COMM_WORLD);

        if (rank == 0) {
            clusters.push_back({best_cluster_members, best_seed});
        }
        for (const int member : best_cluster_members) {
            clustered[member] = 1;
        }
        active_indices.erase(
            std::remove_if(active_indices.begin(), active_indices.end(),
                           [&clustered](const int point) { return clustered[point] != 0; }),
            active_indices.end());
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", world_size);
    }
    
    // Generate once, then replicate the input so every rank can evaluate its
    // assigned seeds without communicating inside a candidate construction.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    
    // The reported interval is the slowest rank's elapsed clustering time.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, rank, world_size);
    
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double max_cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    int exit_code = 0;
    if (rank == 0) {
        const long long cluster_time_ms = static_cast<long long>(max_cluster_time * 1000.0);
    
        printf("Clustering time: %lld ms\n", cluster_time_ms);
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
    
        // Performance metrics use the full-resolution distributed elapsed time.
        const double clusters_per_sec = max_cluster_time > 0.0
            ? clusters.size() / max_cluster_time : 0.0;
        const double points_per_sec = max_cluster_time > 0.0
            ? num_points / max_cluster_time : 0.0;
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
    
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

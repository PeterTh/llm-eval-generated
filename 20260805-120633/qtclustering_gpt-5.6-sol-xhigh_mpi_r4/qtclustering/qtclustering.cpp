// QT Clustering Benchmark - Distributed MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <mpi.h>

#include <algorithm>
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
static_assert(sizeof(Point) == 2 * sizeof(double),
              "Point must be two contiguous doubles for MPI transfer");

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

// Precompute the matrix once per MPI process. Candidate growth always accesses
// a row sequentially, which is substantially faster than repeatedly evaluating
// the same square roots in the innermost loop.
std::vector<double> buildDistanceMatrix(const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    const size_t matrix_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> distances(matrix_size);

    for (int i = 0; i < N; ++i) {
        double* const row = distances.data() + static_cast<size_t>(i) * N;
        const Point& p = points[i];
        for (int j = 0; j < N; ++j) {
            const double dx = p.x - points[j].x;
            const double dy = p.y - points[j].y;
            row[j] = std::sqrt(dx * dx + dy * dy);
        }
    }

    return distances;
}

// Generate one candidate using running maximum distances. After a point is
// appended, only its distance to every remaining point is new; retaining the
// prior maxima is exactly equivalent to rescanning all cluster members.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& active_points,
                             const std::vector<double>& distances,
                             const double threshold,
                             const int point_count,
                             std::vector<double>& candidate_diameters,
                             std::vector<int>& members) {
    const double infinity = std::numeric_limits<double>::max();
    const double* const seed_row = distances.data() +
                                   static_cast<size_t>(seed_point) * point_count;

    members.clear();
    members.push_back(seed_point);

    for (const int candidate : active_points) {
        candidate_diameters[candidate] = seed_row[candidate];
    }
    candidate_diameters[seed_point] = infinity;

    while (members.size() < active_points.size()) {
        int closest_point = -1;
        double min_diameter = infinity;

        // active_points remains sorted, preserving the sequential version's
        // lowest-point-index tie break for equal diameters.
        for (const int candidate : active_points) {
            const double diameter = candidate_diameters[candidate];
            if (diameter < threshold && diameter < min_diameter) {
                min_diameter = diameter;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) {
            break;
        }

        members.push_back(closest_point);
        candidate_diameters[closest_point] = infinity;

        const double* const new_member_row = distances.data() +
                                             static_cast<size_t>(closest_point) * point_count;
        for (const int candidate : active_points) {
            candidate_diameters[candidate] =
                std::max(candidate_diameters[candidate], new_member_row[candidate]);
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm. Each rank builds candidates for a cyclic
// subset of the current seeds. MPI_MAXLOC selects maximum cardinality and, on
// ties, the smallest seed exactly as the sequential traversal does.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count,
                                  MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    const std::vector<double> distances = buildDistanceMatrix(points);
    std::vector<int> active_points(N);
    for (int i = 0; i < N; ++i) {
        active_points[i] = i;
    }

    std::vector<unsigned char> active(N, 1);
    std::vector<double> candidate_diameters(N);
    std::vector<int> candidate_members;
    std::vector<int> local_best_members;
    std::vector<int> winning_members;
    candidate_members.reserve(N);
    local_best_members.reserve(N);
    winning_members.reserve(N);

    std::vector<Cluster> clusters;
    if (rank == 0) {
        clusters.reserve(N);
    }

    while (!active_points.empty()) {
        int local_best_cardinality = -1;
        int local_best_seed = std::numeric_limits<int>::max();

        // Cyclic assignment balances both the seed count and spatially varying
        // candidate sizes without moving point data between processes.
        for (size_t position = static_cast<size_t>(rank);
             position < active_points.size();
             position += static_cast<size_t>(process_count)) {
            const int seed = active_points[position];
            const int cardinality = generateCandidateCluster(
                seed, active_points, distances, threshold, N,
                candidate_diameters, candidate_members);

            if (cardinality > local_best_cardinality) {
                local_best_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = candidate_members;
            }
        }

        const int local_result[2] = {local_best_cardinality, local_best_seed};
        int global_result[2] = {-1, std::numeric_limits<int>::max()};
        MPI_Allreduce(local_result, global_result, 1, MPI_2INT,
                      MPI_MAXLOC, communicator);

        const int max_cardinality = global_result[0];
        const int best_seed = global_result[1];

        // If every remaining candidate is a singleton, removing points cannot
        // make any later candidate larger. Emit them in the same order in one
        // step and avoid a long tail of collectives.
        if (max_cardinality == 1) {
            if (rank == 0) {
                for (const int seed : active_points) {
                    clusters.push_back(Cluster{{seed}, seed});
                }
            }
            break;
        }

        const auto best_position = std::lower_bound(active_points.begin(),
                                                     active_points.end(), best_seed);
        const int owner = static_cast<int>(best_position - active_points.begin()) %
                          process_count;

        winning_members.resize(static_cast<size_t>(max_cardinality));
        if (rank == owner) {
            std::copy(local_best_members.begin(), local_best_members.end(),
                      winning_members.begin());
        }
        MPI_Bcast(winning_members.data(), max_cardinality, MPI_INT,
                  owner, communicator);

        if (rank == 0) {
            clusters.push_back(Cluster{winning_members, best_seed});
        }

        for (const int member : winning_members) {
            active[member] = 0;
        }
        active_points.erase(
            std::remove_if(active_points.begin(), active_points.end(),
                           [&active](const int point) { return active[point] == 0; }),
            active_points.end());
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
    int process_count = 1;
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
    }
    
    // Generate the input once, then replicate the read-only point set. This
    // guarantees identical inputs even on a heterogeneous cluster.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Time from a synchronized start and report the slowest rank, which is the
    // distributed operation's actual wall time.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, process_count, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %lld ms\n",
               static_cast<long long>(cluster_time * 1000.0));
        printf("Clusters found: %zu\n", clusters.size());
    }
    
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

    if (rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double clusters_per_sec = clusters.size() / cluster_time;
        const double points_per_sec = num_points / cluster_time;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
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
    int exit_code = 0;
    if (rank == 0 && validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

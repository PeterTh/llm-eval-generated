// QT Clustering Benchmark - MPI Parallel Version
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

struct CandidateWorkspace {
    std::vector<double> max_dist;
    std::vector<double> max_dist_sq;
    std::vector<unsigned char> in_cluster;
    std::vector<int> members;

    explicit CandidateWorkspace(int n)
        : max_dist(static_cast<size_t>(n)),
          max_dist_sq(static_cast<size_t>(n)),
          in_cluster(static_cast<size_t>(n)),
          members() {
        members.reserve(static_cast<size_t>(n));
    }
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

inline double distanceSquared(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& candidate_indices,
                             const std::vector<Point>& points,
                             const double threshold,
                             CandidateWorkspace& ws) {
    const int point_count = static_cast<int>(points.size());
    const Point* pts = points.data();
    double* max_dist = ws.max_dist.data();
    double* max_dist_sq = ws.max_dist_sq.data();
    unsigned char* in_cluster = ws.in_cluster.data();

    std::memset(in_cluster, 0, static_cast<size_t>(point_count));
    ws.members.clear();

    in_cluster[seed_point] = 1;
    ws.members.push_back(seed_point);

    const Point seed = pts[seed_point];
    for (const int c : candidate_indices) {
        if (in_cluster[c]) continue;
        const double dist_sq = distanceSquared(pts[c], seed);
        max_dist_sq[c] = dist_sq;
        max_dist[c] = std::sqrt(dist_sq);
    }

    const int candidate_count = static_cast<int>(candidate_indices.size());
    while (static_cast<int>(ws.members.size()) < candidate_count) {
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (const int candidate : candidate_indices) {
            if (in_cluster[candidate]) continue;

            const double dist = max_dist[candidate];
            if (dist < threshold && dist < min_diameter) {
                min_diameter = dist;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) break;

        in_cluster[closest_point] = 1;
        ws.members.push_back(closest_point);

        const Point added = pts[closest_point];
        for (const int candidate : candidate_indices) {
            if (in_cluster[candidate]) continue;
            const double dist_sq = distanceSquared(pts[candidate], added);
            if (dist_sq > max_dist_sq[candidate]) {
                max_dist_sq[candidate] = dist_sq;
                max_dist[candidate] = std::sqrt(dist_sq);
            }
        }
    }

    return static_cast<int>(ws.members.size());
}

// Main QT clustering algorithm, parallelized over candidate seeds
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    CandidateWorkspace ws(N);

    unclustered_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    const int int_max = std::numeric_limits<int>::max();

    while (!unclustered_indices.empty()) {
        int local_best_cardinality = -1;
        int local_best_position = int_max;
        const int unclustered_count = static_cast<int>(unclustered_indices.size());

        for (int j = rank; j < unclustered_count; j += num_ranks) {
            const int seed = unclustered_indices[j];
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold, ws);

            if (cardinality > local_best_cardinality) {
                local_best_cardinality = cardinality;
                local_best_position = j;
            }
        }

        unsigned long long local_key = 0;
        if (local_best_cardinality > 0) {
            local_key =
                (static_cast<unsigned long long>(local_best_cardinality) << 32) |
                static_cast<unsigned int>(int_max - local_best_position);
        }

        unsigned long long global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const int global_best_cardinality =
            static_cast<int>(global_key >> 32);
        const int global_best_position =
            int_max - static_cast<int>(static_cast<unsigned int>(global_key));
        const int member_count = global_best_cardinality;

        if (member_count > 0) {
            const int owner = global_best_position % num_ranks;
            const int best_seed = unclustered_indices[global_best_position];

            if (rank == owner) {
                generateCandidateCluster(
                    best_seed, unclustered_indices, points, threshold, ws);
            } else {
                ws.members.resize(static_cast<size_t>(member_count));
            }

            MPI_Bcast(ws.members.data(), member_count, MPI_INT,
                      owner, MPI_COMM_WORLD);
        }

        if (member_count > 0) {
            const int best_seed = unclustered_indices[global_best_position];

            if (rank == 0) {
                Cluster cluster;
                cluster.seed_point = best_seed;
                cluster.members = ws.members;
                clusters.push_back(std::move(cluster));
            }

            for (int i = 0; i < member_count; ++i) {
                clustered[ws.members[i]] = 1;
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
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
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

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
    }

    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);

    std::vector<Point> points(static_cast<size_t>(num_points));
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&point_type);

    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, num_ranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    int validation_ok = 1;

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
            validation_ok = valid ? 1 : 0;
        }
    }

    if (validate) {
        MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", validation_ok ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return validation_ok ? 0 : 1;
}

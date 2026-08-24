// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// MPI parallelization strategy:
// - All ranks hold identical copies of the point data (broadcast at init)
// - Seed evaluations are distributed across ranks via cyclic partitioning
// - Each iteration: ranks evaluate their assigned seeds in parallel,
//   then MPI_Allreduce finds the global best cluster
// - MPI_Bcast distributes the winning cluster members to all ranks
// - Optimizations: squared distances, incremental max-dist updates,
//   preallocated buffers, vector<char> instead of vector<bool>

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

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
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

// Generate a candidate cluster starting from a seed point.
// Uses squared distances and incremental max-dist updates for performance.
// Returns the cardinality (size) of the cluster.
// If out_members is non-null, fills it with the cluster members.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold_sq,
                             const int point_count,
                             std::vector<int>& out_members) {
    out_members.clear();
    out_members.reserve(64);

    // in_cluster: tracks points in this candidate cluster
    std::vector<char> in_cluster(point_count, 0);

    // max_dist_sq[candidate] = max squared distance from candidate to all current members
    // -1 means candidate is ineligible (already clustered, in cluster, or exceeded threshold)
    std::vector<double> max_dist_sq(point_count, 0.0);

    // Initialize: mark clustered points as ineligible
    for (int i = 0; i < point_count; ++i) {
        if (clustered[i]) max_dist_sq[i] = -1.0;
    }

    // Add seed point
    in_cluster[seed_point] = 1;
    out_members.push_back(seed_point);
    max_dist_sq[seed_point] = -1.0; // exclude from future consideration

    const double seed_x = points[seed_point].x;
    const double seed_y = points[seed_point].y;

    // Update max distances with seed point
    for (int c = 0; c < point_count; ++c) {
        if (max_dist_sq[c] < 0.0) continue;
        double dx = points[c].x - seed_x;
        double dy = points[c].y - seed_y;
        double d2 = dx * dx + dy * dy;
        if (d2 > max_dist_sq[c]) max_dist_sq[c] = d2;
        if (max_dist_sq[c] >= threshold_sq) max_dist_sq[c] = -1.0; // ineligible
    }

    // Iteratively add closest points
    while (static_cast<int>(out_members.size()) < point_count) {
        // Find the candidate with minimum max_dist_sq (greedy addition)
        int best_candidate = -1;
        double best_d2 = std::numeric_limits<double>::max();

        for (int c = 0; c < point_count; ++c) {
            if (max_dist_sq[c] >= 0.0 && max_dist_sq[c] < best_d2) {
                best_d2 = max_dist_sq[c];
                best_candidate = c;
            }
        }

        if (best_candidate < 0) break; // No more points can be added

        in_cluster[best_candidate] = 1;
        out_members.push_back(best_candidate);
        max_dist_sq[best_candidate] = -1.0;

        // Update max distances with the newly added point
        const double new_x = points[best_candidate].x;
        const double new_y = points[best_candidate].y;
        for (int c = 0; c < point_count; ++c) {
            if (max_dist_sq[c] < 0.0) continue;
            double dx = points[c].x - new_x;
            double dy = points[c].y - new_y;
            double d2 = dx * dx + dy * dy;
            if (d2 > max_dist_sq[c]) max_dist_sq[c] = d2;
            if (max_dist_sq[c] >= threshold_sq) max_dist_sq[c] = -1.0;
        }
    }

    return static_cast<int>(out_members.size());
}

// Main QT clustering algorithm with MPI parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int rank, int size) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;

    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Preallocated buffers
    std::vector<int> local_members;
    local_members.reserve(N);
    std::vector<int> recv_buf(N + 2); // [cardinality, seed, member0, member1, ...]

    int iteration = 0;

    // Main clustering loop
    while (true) {
        // Build list of unclustered points
        std::vector<int> unclustered;
        unclustered.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) unclustered.push_back(i);
        }

        if (unclustered.empty()) break;

        const int num_seeds = static_cast<int>(unclustered.size());

        // Each rank evaluates its assigned seeds (cyclic distribution)
        int local_best_card = 0;
        int local_best_seed = -1;
        local_members.clear();

        for (int idx = rank; idx < num_seeds; idx += size) {
            const int seed = unclustered[idx];
            std::vector<int> candidate;
            candidate.reserve(64);
            const int card = generateCandidateCluster(seed, clustered, points,
                                                      threshold_sq, N, candidate);
            if (card > local_best_card ||
                (card == local_best_card && seed < local_best_seed)) {
                local_best_card = card;
                local_best_seed = seed;
                local_members = candidate;
            }
        }

        // MPI_Allreduce to find global best cluster
        // Pack: [cardinality, seed] then use MPI_MAX with a custom approach
        // We need to find the max cardinality, and among ties the min seed.
        // Use two Allreduces: one for max cardinality, then conditional for seed.
        int global_max_card = 0;
        MPI_Allreduce(&local_best_card, &global_max_card, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_card <= 0) break;

        // Among ranks that achieved global_max_card, find the one with smallest seed
        int candidate_seed = (local_best_card == global_max_card) ? local_best_seed : N;
        int global_best_seed = N;
        MPI_Allreduce(&candidate_seed, &global_best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // The rank with global_best_seed broadcasts its cluster members
        int winner_rank = -1;
        // Find which rank has this seed
        // The rank that evaluated seed global_best_seed is: its index in unclustered mapped to rank
        // Find the index of global_best_seed in unclustered
        int seed_idx = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (unclustered[i] == global_best_seed) {
                seed_idx = i;
                break;
            }
        }
        winner_rank = seed_idx % size;

        // Broadcast members from winner rank
        if (rank == winner_rank) {
            // Pack: cardinality, seed, then members
            recv_buf[0] = local_best_card;
            recv_buf[1] = local_best_seed;
            for (int i = 0; i < local_best_card; ++i) {
                recv_buf[2 + i] = local_members[i];
            }
        }
        MPI_Bcast(recv_buf.data(), 2 + global_max_card, MPI_INT, winner_rank, MPI_COMM_WORLD);

        // All ranks now have the same winning cluster - apply it identically
        const int best_card = recv_buf[0];
        const int best_seed = recv_buf[1];

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(best_card);
        for (int i = 0; i < best_card; ++i) {
            cluster.members[i] = recv_buf[2 + i];
            clustered[recv_buf[2 + i]] = 1;
        }
        clusters.push_back(std::move(cluster));

        iteration++;
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0, then broadcast)
    if (rank == 0) {
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
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        
        if (num_points <= 0 || threshold <= 0.0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validate_int = validate ? 1 : 0;
    int printResults_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_int != 0);
    printResults = (printResults_int != 0);

    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI, %d ranks)\n", size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (all ranks generate identically)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double cluster_end = MPI_Wtime();
    long cluster_time_ms = static_cast<long>((cluster_end - cluster_start) * 1000.0);
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());
        
        int total_clustered = 0;
        int max_cluster_size = 0;
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }
        
        const double avg_cluster_size = clusters.empty() ? 0.0 : 
            static_cast<double>(total_clustered) / clusters.size();
        
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = (time_sec > 0.0) ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = (time_sec > 0.0) ? num_points / time_sec : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
        
        if (printResults) {
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
        
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}

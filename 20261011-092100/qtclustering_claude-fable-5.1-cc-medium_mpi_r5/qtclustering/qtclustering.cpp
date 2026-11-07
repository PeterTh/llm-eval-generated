// QT Clustering Benchmark - MPI Distributed-Memory Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * Every rank holds a replica of the (deterministically generated) point
//     set and of the clustering state (which points are already clustered).
//   * In every outer iteration the candidate-cluster generation for the
//     remaining unclustered seeds is distributed across ranks in a strided
//     (round-robin) fashion for load balance.
//   * The best candidate (max cardinality, lowest seed index on ties - exactly
//     the sequential selection rule) is determined with an MPI_Allreduce
//     (MPI_MAXLOC) and its membership is broadcast from the owning rank.
//   * All ranks then update their replicated state identically, so results
//     are bit-identical to the sequential algorithm.

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

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster and fills `members`.
//
// This is an incremental formulation of the original algorithm: for every
// candidate we maintain the maximum distance to the current cluster members
// (max is exact and order independent, so the values are bit-identical to the
// original recomputation). Candidates whose max distance reaches the threshold
// can never be added again (the max only grows) and are dropped.
// Selection rule is identical to the original: among candidates with
// max_dist < threshold take the minimum max_dist, lowest index on ties.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered_indices,
                             const std::vector<Point>& points,
                             const double threshold,
                             std::vector<int>& members,
                             std::vector<int>& cand,
                             std::vector<double>& cand_maxd) {
    members.clear();
    members.push_back(seed_point);

    // Initial candidate set: all unclustered points except the seed, in
    // ascending index order (unclustered_indices is kept sorted).
    cand.clear();
    cand_maxd.clear();
    const Point ps = points[seed_point];
    for (size_t i = 0; i < unclustered_indices.size(); ++i) {
        const int c = unclustered_indices[i];
        if (c == seed_point) continue;
        const double d = distance(points[c], ps);
        if (d < threshold) {
            cand.push_back(c);
            cand_maxd.push_back(d);
        }
    }

    while (!cand.empty()) {
        // Find closest candidate (min of max distance), lowest index on ties.
        size_t best = 0;
        double min_diameter = cand_maxd[0];
        const size_t nc = cand.size();
        for (size_t i = 1; i < nc; ++i) {
            if (cand_maxd[i] < min_diameter) {
                min_diameter = cand_maxd[i];
                best = i;
            }
        }

        const int closest = cand[best];
        members.push_back(closest);

        // Update max distances with the new member and compact the list.
        const Point pn = points[closest];
        size_t w = 0;
        for (size_t i = 0; i < nc; ++i) {
            if (i == best) continue;
            const double d = distance(points[cand[i]], pn);
            const double md = std::max(cand_maxd[i], d);
            if (md < threshold) {
                cand[w] = cand[i];
                cand_maxd[w] = md;
                ++w;
            }
        }
        cand.resize(w);
        cand_maxd.resize(w);
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel, state replicated on all ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int mpi_rank,
                                  const int mpi_size) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Scratch buffers (reused across seeds to avoid allocations)
    std::vector<int> candidate_members;
    std::vector<int> best_cluster_members;
    std::vector<int> cand;
    std::vector<double> cand_maxd;
    candidate_members.reserve(N);
    best_cluster_members.reserve(N);
    cand.reserve(N);
    cand_maxd.reserve(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        best_cluster_members.clear();

        // Try each unclustered point as a seed; seeds are distributed across
        // ranks round-robin for load balance.
        const size_t n_uncl = unclustered_indices.size();
        for (size_t i = static_cast<size_t>(mpi_rank); i < n_uncl;
             i += static_cast<size_t>(mpi_size)) {
            const int seed = unclustered_indices[i];

            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold,
                candidate_members, cand, cand_maxd);

            // Seeds are visited in ascending index order, so '>' keeps the
            // lowest seed index among equal cardinalities (as the original).
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members.swap(candidate_members);
            }
        }

        // Global selection: maximum cardinality, lowest seed index on ties.
        // MPI_MAXLOC picks the max value and, among equal values, the minimum
        // "location", which here is the seed index itself.
        struct { int value; int loc; } local, global;
        local.value = max_cardinality;
        local.loc = (best_seed >= 0) ? best_seed : std::numeric_limits<int>::max();
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        max_cardinality = global.value;
        best_seed = (global.value > 0) ? global.loc : -1;

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            // Owner is the rank that processed best_seed (position in the
            // sorted unclustered list modulo number of ranks).
            const size_t pos = static_cast<size_t>(
                std::lower_bound(unclustered_indices.begin(),
                                 unclustered_indices.end(), best_seed) -
                unclustered_indices.begin());
            const int owner = static_cast<int>(pos % static_cast<size_t>(mpi_size));

            best_cluster_members.resize(max_cardinality);
            MPI_Bcast(best_cluster_members.data(), max_cardinality, MPI_INT,
                      owner, MPI_COMM_WORLD);

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(std::move(cluster));

            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
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
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const bool is_root = (mpi_rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (is_root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (is_root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpi_size);
    }
    
    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const std::chrono::milliseconds cluster_time(max_cluster_time);
    
    int exit_code = 0;
    if (is_root) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
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
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        fflush(stdout);
    }
    
    // All ranks return the same exit code
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

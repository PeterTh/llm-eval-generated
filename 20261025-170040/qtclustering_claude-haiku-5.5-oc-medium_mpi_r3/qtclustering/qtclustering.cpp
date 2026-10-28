// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in every round, the candidate seeds are distributed
// cyclically across MPI ranks. Each rank grows candidate clusters for its
// seeds, the globally best seed (largest cardinality, ties broken by lowest
// seed index) is selected with MPI_MAXLOC, and the owning rank broadcasts the
// winning cluster members to all ranks.

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
// (deterministic: every rank generates an identical data set)
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

// Grow a candidate cluster from a seed point.
//
// Semantically identical to repeatedly selecting, among unclustered points,
// the candidate whose maximum distance to the current members is smallest
// (ties -> lowest index) while that maximum stays below the threshold.
// Optimizations (exact, no change in results):
//  - the per-candidate max distance to the members is maintained incrementally
//    (max is order-independent, so the values are bit-identical to recomputing);
//  - a candidate whose max distance reaches the threshold can never become
//    admissible again, so it is dropped from the active list permanently;
//  - the active list stays in ascending index order, so a strict '<' scan
//    yields the lowest-index minimum, as in the original tie-breaking.
// Returns the cardinality; the members (insertion order) are written to `members`.
int growCluster(const int seed,
                const std::vector<char>& clustered,
                const std::vector<Point>& points,
                const double threshold,
                const int point_count,
                std::vector<double>& maxd,
                std::vector<int>& active,
                std::vector<int>& members) {
    members.clear();
    members.push_back(seed);

    active.clear();
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed) continue;
        const double m = std::max(0.0, distance(points[c], points[seed]));
        if (m < threshold) {
            maxd[c] = m;
            active.push_back(c);
        }
    }

    while (!active.empty()) {
        // Admissible candidate with the smallest max distance (lowest index on ties)
        size_t bi = 0;
        for (size_t i = 1; i < active.size(); ++i) {
            if (maxd[active[i]] < maxd[active[bi]]) bi = i;
        }
        const int best = active[bi];
        members.push_back(best);

        // Update remaining candidates with the newly added member and prune
        size_t w = 0;
        for (size_t i = 0; i < active.size(); ++i) {
            if (i == bi) continue;
            const int c = active[i];
            const double m = std::max(maxd[c], distance(points[c], points[best]));
            if (m < threshold) {
                maxd[c] = m;
                active[w++] = c;
            }
        }
        active.resize(w);
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (collective: all ranks must call it)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs,
                                  MPI_Comm comm) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    // Per-rank workspace
    std::vector<double> maxd(N, 0.0);
    std::vector<int> active;
    active.reserve(N);
    std::vector<int> cand;
    std::vector<int> best_members;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const size_t U = unclustered_indices.size();

        // Local search: seeds at positions rank, rank+nprocs, ... (cyclic distribution)
        int local_card = -1;
        int local_seed = -1;
        for (size_t p = static_cast<size_t>(rank); p < U; p += static_cast<size_t>(nprocs)) {
            const int seed = unclustered_indices[p];
            const int card = growCluster(seed, clustered, points, threshold, N,
                                         maxd, active, cand);
            // Strict '>' keeps the lowest seed index among equal cardinalities
            if (card > local_card) {
                local_card = card;
                local_seed = seed;
                best_members.swap(cand);
            }
        }

        // Global winner: maximum cardinality, ties -> lowest seed index
        struct { int val; int idx; } loc;
        loc.val = local_card;
        loc.idx = (local_seed >= 0) ? local_seed : N;
        MPI_Allreduce(MPI_IN_PLACE, &loc, 1, MPI_2INT, MPI_MAXLOC, comm);

        const int max_cardinality = loc.val;
        const int best_seed = loc.idx;
        if (max_cardinality <= 0 || best_seed < 0 || best_seed >= N) {
            break; // No more clusters can be formed
        }

        // The owner of the winning seed holds its members; broadcast them
        const size_t pos = static_cast<size_t>(
            std::lower_bound(unclustered_indices.begin(), unclustered_indices.end(), best_seed)
            - unclustered_indices.begin());
        const int root = static_cast<int>(pos % static_cast<size_t>(nprocs));

        int len = (rank == root) ? static_cast<int>(best_members.size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, root, comm);
        best_members.resize(len);
        MPI_Bcast(best_members.data(), len, MPI_INT, root, comm);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(std::move(cluster));

        // Mark all members as clustered
        for (int m : best_members) {
            clustered[m] = 1;
        }

        // Remove clustered points from unclustered list (order preserved)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );
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
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    const bool root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;
    
    // Parse command line arguments (identical on all ranks)
    bool done = false;
    for (int i = 1; i < argc && !done; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            done = true;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exit_code = 1;
            done = true;
        }
    }
    
    if (!done) {
        if (num_points <= 0 || threshold <= 0.0) {
            if (root) {
                printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                       num_points, threshold);
            }
            exit_code = 1;
        }
    }

    if (done || exit_code != 0) {
        MPI_Finalize();
        return exit_code;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs, comm);
    
    MPI_Barrier(comm);
    const double t1 = MPI_Wtime();
    const double local_elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const long cluster_ms = static_cast<long>(max_elapsed * 1000.0);
    
    if (root) {
        printf("Clustering time: %ld ms\n", cluster_ms);
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
    
    if (root) {
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        // Performance metrics
        const double time_sec = cluster_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && root) {
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
    
    // Validation (performed on root; result shared with all ranks)
    if (validate) {
        int valid = 0;
        if (root) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        
        if (root) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        exit_code = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return exit_code;
}

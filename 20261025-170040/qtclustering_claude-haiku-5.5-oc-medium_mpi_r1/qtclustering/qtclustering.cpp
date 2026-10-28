// QT Clustering Benchmark - MPI distributed-memory version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   - Every rank holds the full point set (generated deterministically, so
//     all ranks produce identical data).
//   - In each round of the outer loop, the unclustered seeds are dealt out
//     cyclically across ranks. Each rank grows candidate clusters for its own
//     seeds and keeps its local best.
//   - One MPI_Allgather selects the global best seed using the same rule as
//     the serial algorithm: largest cardinality, ties broken by lowest index.
//     The owning rank broadcasts the winning members so every rank keeps an
//     identical clustering state.
//   - Validation diameters are computed in parallel and reduced.
//
// Semantics are identical to the serial algorithm: the same points are chosen
// in the same order, with the same tie-breaking.

#include <algorithm>
#include <cmath>
#include <cstddef>
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

// Squared Euclidean distance between two points
inline double squaredDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(squaredDistance(p1, p2));
}

// Per-rank scratch space, reused across candidate-cluster growths
struct GrowthScratch {
    std::vector<int> active;    // candidates that may still join, ascending index
    std::vector<double> maxd2;  // per point: max squared distance to current members
};

// Grow a candidate cluster from `seed`. Repeatedly adds the unclustered point
// whose maximum distance to the current members is smallest (ties -> lowest
// index), as long as that maximum distance is below `threshold`.
//
// Returns the cardinality of the cluster and fills `members` in insertion
// order. If the cluster provably cannot exceed `floor_card`, growth stops early
// and 0 is returned (the result is then irrelevant to the caller).
//
// `unclustered` must list the unclustered point indices in ascending order.
int growCandidateCluster(const int seed,
                         const std::vector<int>& unclustered,
                         const std::vector<Point>& points,
                         const double threshold,
                         const int floor_card,
                         GrowthScratch& s,
                         std::vector<int>& members) {
    members.clear();
    members.push_back(seed);

    // Every unclustered point except the seed is a candidate. The maximum
    // distance to the (single-element) member set starts at zero.
    s.active.clear();
    for (const int c : unclustered) {
        if (c == seed) continue;
        s.active.push_back(c);
        s.maxd2[c] = 0.0;
    }

    int newest = seed;
    while (true) {
        // Fold the newest member into each candidate's max distance. Because
        // the max distance only grows as members are added, a candidate that
        // reaches the threshold can never become valid again and is dropped.
        // sqrt is monotone, so max(sqrt(d2)) == sqrt(max(d2)) exactly.
        const Point& np = points[newest];
        size_t w = 0;
        size_t best_pos = 0;
        double best_dist = std::numeric_limits<double>::max();
        for (size_t r = 0; r < s.active.size(); ++r) {
            const int c = s.active[r];
            const double m2 = std::max(s.maxd2[c], squaredDistance(points[c], np));
            s.maxd2[c] = m2;
            const double dist = std::sqrt(m2);
            if (dist < threshold) {
                s.active[w] = c;
                if (dist < best_dist) {  // strict: lowest index wins ties
                    best_dist = dist;
                    best_pos = w;
                }
                ++w;
            }
        }
        s.active.resize(w);

        if (s.active.empty()) break;  // No more points can be added

        // Each remaining step adds one candidate, so the final cardinality is
        // at most members + active. Stop if that cannot beat the floor.
        if (static_cast<int>(members.size() + s.active.size()) <= floor_card) {
            return 0;
        }

        const int chosen = s.active[best_pos];
        s.active.erase(s.active.begin() + static_cast<std::ptrdiff_t>(best_pos));
        members.push_back(chosen);
        newest = chosen;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (collective: every rank must call it)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }
    std::vector<Cluster> clusters;

    GrowthScratch scratch;
    scratch.maxd2.assign(N, 0.0);
    scratch.active.reserve(N);

    std::vector<int> members;
    std::vector<int> best_members;   // local best cluster for this rank
    std::vector<int> global_best;    // winning cluster, received via broadcast
    std::vector<int> gathered(2 * static_cast<size_t>(nprocs));

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Each rank tries its share of the seeds (cyclic distribution)
        int local_card = 0;
        int local_seed = -1;
        for (size_t j = static_cast<size_t>(rank); j < unclustered_indices.size();
             j += static_cast<size_t>(nprocs)) {
            const int seed = unclustered_indices[j];
            const int card = growCandidateCluster(seed, unclustered_indices, points,
                                                  threshold, local_card, scratch, members);
            if (card > local_card) {
                local_card = card;
                local_seed = seed;
                best_members = members;
            }
        }

        // Global selection: max cardinality, then lowest seed index
        const int mine[2] = {local_card, local_seed};
        MPI_Allgather(mine, 2, MPI_INT, gathered.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int max_cardinality = 0;
        int best_seed = -1;
        int owner = -1;
        for (int r = 0; r < nprocs; ++r) {
            const int card = gathered[2 * r];
            const int seed = gathered[2 * r + 1];
            if (card <= 0) continue;
            if (card > max_cardinality ||
                (card == max_cardinality && seed < best_seed)) {
                max_cardinality = card;
                best_seed = seed;
                owner = r;
            }
        }

        // No more clusters can be formed (identical decision on all ranks)
        if (best_seed < 0 || max_cardinality <= 0) {
            break;
        }

        // Share the winning members with all ranks
        int msize = (rank == owner) ? static_cast<int>(best_members.size()) : 0;
        MPI_Bcast(&msize, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank == owner) {
            global_best = best_members;
        } else {
            global_best.resize(msize);
        }
        MPI_Bcast(global_best.data(), msize, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = global_best;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (const int m : cluster.members) {
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

// Diameter of one cluster: max distance between any two of its members
double clusterDiameter(const Cluster& cluster, const std::vector<Point>& points) {
    double max_diameter = 0.0;
    for (size_t i = 0; i < cluster.members.size(); ++i) {
        for (size_t j = i + 1; j < cluster.members.size(); ++j) {
            const double dist = distance(points[cluster.members[i]],
                                       points[cluster.members[j]]);
            max_diameter = std::max(max_diameter, dist);
        }
    }
    return max_diameter;
}

// Validation: check that clusters satisfy the QT clustering properties.
// `diameters[c]` holds the precomputed diameter of cluster c. Output is only
// produced when `print` is true; the result is the same on every rank.
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<double>& diameters,
                      const std::vector<Point>& points,
                      const double threshold,
                      const bool print) {
    bool valid = true;
    
    if (print) printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
        if (print && c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            if (print) {
                printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                       c, max_diameter, threshold);
            }
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                if (print) {
                    printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                           member, membership[member], c);
                }
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
    
    if (print) {
        printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
               points.size(), clustered_count, points.size() - clustered_count);
    }
    
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
    int exit_code = -1;
    for (int i = 1; i < argc && exit_code < 0; ++i) {
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
            exit_code = 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exit_code = 1;
        }
    }
    
    if (exit_code < 0 && (num_points <= 0 || threshold <= 0.0)) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        exit_code = 1;
    }

    if (exit_code >= 0) {
        MPI_Finalize();
        return exit_code;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
    }
    
    // Generate synthetic data (deterministic, so every rank gets the same points)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);
    
    double elapsed = MPI_Wtime() - cluster_start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long cluster_ms = static_cast<long>(elapsed * 1000.0);
    
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
    
    // Validation (diameters computed in parallel, then reduced)
    if (validate) {
        const size_t C = clusters.size();
        std::vector<double> diameters(C, 0.0);
        for (size_t c = static_cast<size_t>(rank); c < C;
             c += static_cast<size_t>(nprocs)) {
            diameters[c] = clusterDiameter(clusters[c], points);
        }
        if (C > 0) {
            MPI_Allreduce(MPI_IN_PLACE, diameters.data(), static_cast<int>(C),
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }

        const bool valid = validateClusters(clusters, diameters, points, threshold, root);
        
        if (root) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}

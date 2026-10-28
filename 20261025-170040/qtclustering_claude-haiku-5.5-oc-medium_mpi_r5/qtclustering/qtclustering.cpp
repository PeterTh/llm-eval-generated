// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization (MPI, distributed memory):
//   * Every rank deterministically generates the identical point set, so no
//     point data has to be communicated.
//   * In each round of the main clustering loop the unclustered seed points
//     are distributed cyclically over the ranks; each rank grows candidate
//     clusters for its seeds and keeps its best one.
//   * An allgather of (cardinality, seed) pairs selects the global winner:
//     largest cardinality, ties broken by lowest seed index (identical to the
//     sequential rule). The owning rank broadcasts the winning member list so
//     that every rank keeps identical clustering state.
//   * Cluster diameters for validation are computed in parallel; rank 0
//     performs all console output.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// (deterministic: every rank produces the same data)
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

// Reusable scratch space for candidate cluster growth
struct GrowthWorkspace {
    std::vector<int> active;   // eligible candidates, ascending point index
    std::vector<double> maxd;  // per point: max distance to current members
};

// Greedily grow a candidate cluster starting from `seed`.
//
// Semantics are identical to the sequential algorithm: at every step the
// unclustered point with the smallest maximum distance to the current members
// (ties -> lowest index), provided that distance is below `threshold`, is added.
// Two exact optimizations are used:
//   * The maximum distance to the members is maintained incrementally. A max
//     over the same set of values is order independent, so the results are
//     bit-identical to recomputing them from scratch.
//   * A candidate whose maximum distance reaches the threshold can never be
//     added later (the maximum only grows), so it is dropped permanently.
//
// Returns false if the cluster provably cannot exceed `min_card` members, in
// which case the contents of `members` are unspecified. Returns true with the
// full member list (in insertion order) otherwise; the size is then > min_card.
bool growCandidateCluster(const int seed,
                          const std::vector<uint8_t>& clustered,
                          const std::vector<Point>& points,
                          const double threshold,
                          const int N,
                          const int min_card,
                          GrowthWorkspace& ws,
                          std::vector<int>& members) {
    members.clear();
    members.push_back(seed);

    ws.active.clear();
    for (int c = 0; c < N; ++c) {
        if (!clustered[c] && c != seed) {
            ws.active.push_back(c);
            ws.maxd[c] = 0.0;
        }
    }

    int last = seed;
    while (true) {
        // Update every eligible candidate with the most recently added member,
        // find the best one and compact away candidates that are now ineligible.
        int best = -1;
        double best_d = 0.0;
        size_t kept = 0;
        for (size_t k = 0; k < ws.active.size(); ++k) {
            const int c = ws.active[k];
            const double d = std::max(ws.maxd[c], distance(points[c], points[last]));
            ws.maxd[c] = d;
            if (d >= threshold) continue;  // can never be added to this cluster
            ws.active[kept++] = c;
            if (best < 0 || d < best_d) {  // ascending order => first minimum wins ties
                best = c;
                best_d = d;
            }
        }
        ws.active.resize(kept);

        // Upper bound on the final cardinality: current members + eligible candidates.
        if (static_cast<int>(members.size() + kept) <= min_card) return false;
        if (best < 0) return true;  // no point can be added any more

        members.push_back(best);
        ws.active.erase(std::lower_bound(ws.active.begin(), ws.active.end(), best));
        last = best;
    }
}

// Main QT clustering algorithm (distributed over MPI ranks).
// All ranks return identical cluster lists.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    GrowthWorkspace ws;
    ws.maxd.assign(N, 0.0);
    ws.active.reserve(N);
    std::vector<int> members;
    std::vector<int> local_best_members;
    std::vector<int> gathered(2 * static_cast<size_t>(nprocs));

    // Main clustering loop
    while (!unclustered.empty()) {
        // Phase 1: search this rank's share of seeds (cyclic distribution over
        // the unclustered list). Seeds are visited in ascending index order.
        int local_card = -1;
        int local_seed = -1;
        local_best_members.clear();
        for (size_t j = static_cast<size_t>(rank); j < unclustered.size();
             j += static_cast<size_t>(nprocs)) {
            const int seed = unclustered[j];
            if (!growCandidateCluster(seed, clustered, points, threshold, N,
                                      local_card, ws, members)) {
                continue;  // cannot beat this rank's current best
            }
            local_card = static_cast<int>(members.size());
            local_seed = seed;
            local_best_members.swap(members);
        }

        // Phase 2: global winner = largest cardinality, ties -> lowest seed index.
        const int send[2] = {local_card, local_seed};
        MPI_Allgather(send, 2, MPI_INT, gathered.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int best_card = -1;
        int best_seed = -1;
        int owner = -1;
        for (int r = 0; r < nprocs; ++r) {
            const int card = gathered[2 * r];
            const int seed = gathered[2 * r + 1];
            if (card < 0) continue;  // rank had no seeds this round
            if (card > best_card || (card == best_card && seed < best_seed)) {
                best_card = card;
                best_seed = seed;
                owner = r;
            }
        }

        // No cluster can be formed (identical decision on every rank)
        if (owner < 0 || best_card <= 0) break;

        // Phase 3: owner broadcasts the winning members (insertion order preserved)
        int msize = (rank == owner) ? best_card : 0;
        MPI_Bcast(&msize, 1, MPI_INT, owner, MPI_COMM_WORLD);

        std::vector<int> best_members;
        if (rank == owner) {
            best_members.swap(local_best_members);
        } else {
            best_members.resize(static_cast<size_t>(msize));
        }
        MPI_Bcast(best_members.data(), msize, MPI_INT, owner, MPI_COMM_WORLD);

        // Phase 4: update the (replicated) clustering state
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        for (int m : cluster.members) {
            clustered[m] = 1;
        }
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
        clusters.push_back(std::move(cluster));
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
// Diameters are computed in parallel; all ranks return the same result,
// only rank 0 prints.
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold,
                      const int rank,
                      const int nprocs) {
    bool valid = true;
    const bool root = (rank == 0);

    if (root) printf("Validating clusters:\n");

    // Each rank computes the diameters of its share of clusters; the other
    // entries stay 0, so an element-wise MAX gives every rank all diameters.
    std::vector<double> diameters(clusters.size(), 0.0);
    for (size_t c = static_cast<size_t>(rank); c < clusters.size();
         c += static_cast<size_t>(nprocs)) {
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
        diameters[c] = max_diameter;
    }
    MPI_Allreduce(MPI_IN_PLACE, diameters.data(), static_cast<int>(diameters.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

        if (root && c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            if (root) {
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
                if (root) {
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

    if (root) {
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

static int runBenchmark(int argc, char** argv, const int rank, const int nprocs) {
    const bool root = (rank == 0);
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_end = MPI_Wtime();
    const long cluster_ms = static_cast<long>((cluster_end - cluster_start) * 1000.0);

    if (root) {
        printf("Clustering time: %ld ms\n", cluster_ms);
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
        const double time_sec = cluster_ms / 1000.0;
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
    }

    // Validation (collective: all ranks take part)
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, nprocs);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int rc = runBenchmark(argc, argv, rank, nprocs);

    MPI_Finalize();
    return rc;
}

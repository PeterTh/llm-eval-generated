// QT Clustering Benchmark - MPI Distributed Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (MPI, distributed memory):
//   The dominant cost of the algorithm is the "candidate cluster" generation
//   that is performed independently for every still-unclustered point acting
//   as a seed.  Those seed evaluations are embarrassingly parallel, so the
//   seeds of each outer iteration are distributed over the MPI ranks.
//   Every rank keeps a full replica of the (small) point set and of the
//   clustered/unclustered bookkeeping, so no point data has to be exchanged.
//   Per outer iteration the ranks exchange
//     * the per-seed cardinalities (one MPI_Allreduce(MPI_MAX) over the
//       unclustered positions), which yields the winning seed with exactly
//       the same tie-breaking rule as the sequential code (largest
//       cardinality, smallest point index), and
//     * the member list of the winning cluster (one MPI_Bcast from the rank
//       that owns the winning seed).
//   The cardinalities of the previous iteration are also used as a cost model
//   to statically load balance the seeds with an LPT (longest processing time
//   first) schedule that every rank computes redundantly and identically, so
//   the balancing itself needs no extra communication.
//
// Serial optimizations (semantically equivalent to the original code):
//   * The maximum distance of every candidate to the current cluster is
//     maintained incrementally, which reduces the cost of generating one
//     candidate cluster from O(N * k^2) to O(N * k) for a cluster of size k.
//   * All distance comparisons are performed on squared distances (monotone
//     transformation), which removes the square root from the hot loop.
//   * The candidate data is kept in a compact structure-of-arrays layout so
//     the hot loop vectorizes; already-taken candidates are masked with
//     +infinity instead of being removed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
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

// Per-rank scratch space for the candidate cluster generation.  Kept alive
// across seeds/iterations so that no allocation happens in the hot path.
struct Workspace {
    // Structure-of-arrays copy of the currently unclustered points
    std::vector<double> cx, cy;
    // Maximum squared distance of each candidate to the current cluster
    std::vector<double> cmax;
    // Global point index of each candidate
    std::vector<int> cid;
    // Members of the cluster that is currently being built
    std::vector<int> members;
};

static constexpr int kMinLanes = 8;  // unroll factor for the vectorized kernel

// Update the incremental maxima with the newly added cluster member
// (px, py) and return the smallest maximum found.  Candidates that are
// already part of the cluster carry +infinity and are therefore never
// selected again.
static inline double updateAndMin(const double* __restrict cx,
                                  const double* __restrict cy,
                                  double* __restrict cmax,
                                  const int n,
                                  const double px, const double py) {
    // Independent accumulator lanes so that the min-reduction vectorizes
    // without requiring floating point reassociation.
    double lane[kMinLanes];
    for (int k = 0; k < kMinLanes; ++k) lane[k] = std::numeric_limits<double>::infinity();

    int i = 0;
    for (; i + kMinLanes <= n; i += kMinLanes) {
        for (int k = 0; k < kMinLanes; ++k) {
            const double dx = cx[i + k] - px;
            const double dy = cy[i + k] - py;
            const double d2 = dx * dx + dy * dy;
            const double prev = cmax[i + k];
            const double m = d2 > prev ? d2 : prev;
            cmax[i + k] = m;
            lane[k] = m < lane[k] ? m : lane[k];
        }
    }

    double minv = std::numeric_limits<double>::infinity();
    for (int k = 0; k < kMinLanes; ++k) minv = lane[k] < minv ? lane[k] : minv;

    for (; i < n; ++i) {
        const double dx = cx[i] - px;
        const double dy = cy[i] - py;
        const double d2 = dx * dx + dy * dy;
        const double prev = cmax[i];
        const double m = d2 > prev ? d2 : prev;
        cmax[i] = m;
        minv = m < minv ? m : minv;
    }

    return minv;
}

// Generate a candidate cluster starting from a seed point.
// 'ws' must hold the compacted list of all unclustered points (including the
// seed); 'seed_pos' is the position of the seed within that list.
// Returns the cardinality (size) of the cluster; the members are left in
// ws.members (in insertion order, matching the sequential algorithm).
static int generateCandidateCluster(const int seed_pos,
                                    const double threshold_sq,
                                    Workspace& ws) {
    const int n = static_cast<int>(ws.cid.size());
    const double inf = std::numeric_limits<double>::infinity();

    std::fill(ws.cmax.begin(), ws.cmax.end(), 0.0);
    ws.cmax[seed_pos] = inf;  // mask the seed itself

    ws.members.clear();
    const int seed_point = ws.cid[seed_pos];
    ws.members.push_back(seed_point);

    double px = ws.cx[seed_pos];
    double py = ws.cy[seed_pos];

    // Iteratively add the point that keeps the cluster diameter smallest
    while (true) {
        const double minv = updateAndMin(ws.cx.data(), ws.cy.data(), ws.cmax.data(),
                                         n, px, py);
        if (!(minv < threshold_sq)) break;  // no more points can be added

        // First candidate attaining the minimum (matches the sequential
        // code, which scans candidates in increasing point index order and
        // only replaces the best on a strictly smaller diameter).
        int best = 0;
        while (ws.cmax[best] != minv) ++best;

        ws.members.push_back(ws.cid[best]);
        px = ws.cx[best];
        py = ws.cy[best];
        ws.cmax[best] = inf;
    }

    return static_cast<int>(ws.members.size());
}

// Distribute the seed positions [0, n_pos) over 'nranks' ranks using an LPT
// schedule based on the given per-position cost estimate.  Deterministic, so
// every rank obtains the identical assignment without communication.
static void assignSeeds(const std::vector<int>& cost, const int nranks,
                        std::vector<int>& owner, std::vector<int>& order) {
    const int n_pos = static_cast<int>(cost.size());
    owner.resize(n_pos);

    order.resize(n_pos);
    for (int i = 0; i < n_pos; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&cost](int a, int b) {
        if (cost[a] != cost[b]) return cost[a] > cost[b];
        return a < b;
    });

    // Min-heap of (accumulated load, rank)
    typedef std::pair<long long, int> Load;
    std::priority_queue<Load, std::vector<Load>, std::greater<Load>> heap;
    for (int r = 0; r < nranks; ++r) heap.emplace(0LL, r);

    for (int k = 0; k < n_pos; ++k) {
        const Load top = heap.top();
        heap.pop();
        const int pos = order[k];
        owner[pos] = top.second;
        heap.emplace(top.first + cost[pos], top.second);
    }
}

// Main QT clustering algorithm (MPI parallel over the seed evaluations)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;

    // Replicated SoA copy of all points for the vectorized kernel
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;

    // Cost model for the load balancing: cardinality this point produced in
    // the previous outer iteration (1 = "unknown", giving a cyclic schedule).
    std::vector<int> prev_card(N, 1);

    std::vector<Cluster> clusters;
    Workspace ws;
    std::vector<int> cards, cost, owner, order, best_members, my_seeds;

    // Main clustering loop
    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());

        // Compact the unclustered points into the workspace (shared by all
        // seeds of this iteration)
        ws.cx.resize(U);
        ws.cy.resize(U);
        ws.cmax.resize(U);
        ws.cid.resize(U);
        for (int p = 0; p < U; ++p) {
            const int idx = unclustered[p];
            ws.cid[p] = idx;
            ws.cx[p] = px[idx];
            ws.cy[p] = py[idx];
        }

        // Static load balanced assignment of the seeds to the ranks
        cost.resize(U);
        for (int p = 0; p < U; ++p) cost[p] = prev_card[unclustered[p]];
        assignSeeds(cost, nranks, owner, order);

        my_seeds.clear();
        for (int p = 0; p < U; ++p) {
            if (owner[p] == rank) my_seeds.push_back(p);
        }

        // Evaluate the local share of the seeds
        cards.assign(U, 0);
        int local_best_card = 0;
        int local_best_pos = -1;
        for (const int p : my_seeds) {
            const int card = generateCandidateCluster(p, threshold_sq, ws);
            cards[p] = card;
            // Best local candidate: largest cardinality, smallest index
            if (card > local_best_card) {
                local_best_card = card;
                local_best_pos = p;
                best_members = ws.members;
            }
        }

        // Combine the cardinalities of all ranks.  Positions not owned by a
        // rank contribute 0, owned ones contribute a cardinality >= 1.
        MPI_Allreduce(MPI_IN_PLACE, cards.data(), U, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Winning seed: maximum cardinality, smallest point index (the
        // unclustered list is sorted ascending, so the first position wins).
        int best_pos = -1;
        int max_cardinality = 0;
        for (int p = 0; p < U; ++p) {
            if (cards[p] > max_cardinality) {
                max_cardinality = cards[p];
                best_pos = p;
            }
        }

        if (best_pos < 0) break;  // No more clusters can be formed

        // Fetch the member list of the winning cluster from its owner
        best_members.resize(max_cardinality);
        if (nranks > 1) {
            const int root = owner[best_pos];
            if (rank == root && local_best_pos != best_pos) {
                // Should not happen: the local winner is the global winner
                // whenever this rank owns it.
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Bcast(best_members.data(), max_cardinality, MPI_INT, root, MPI_COMM_WORLD);
        }

        Cluster cluster;
        cluster.seed_point = unclustered[best_pos];
        cluster.members = best_members;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (const int m : best_members) clustered[m] = 1;

        // Remember the cardinalities as cost estimate for the next iteration
        for (int p = 0; p < U; ++p) prev_card[unclustered[p]] = cards[p] > 0 ? cards[p] : 1;

        // Remove clustered points from unclustered list
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end()
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
        printf("MPI ranks: %d\n", nranks);
    }

    // Generate synthetic data (identical on every rank: generated on rank 0
    // and broadcast so that all ranks see bit-identical coordinates)
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_cluster_time_ms = cluster_time.count();
    long cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

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
    const double time_sec = cluster_time_ms / 1000.0;
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
    int exit_code = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exit_code = 1;
        }
    }

    MPI_Finalize();
    return exit_code;
}

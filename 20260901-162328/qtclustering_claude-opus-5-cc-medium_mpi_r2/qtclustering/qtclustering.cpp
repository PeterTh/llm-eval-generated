// QT Clustering Benchmark - MPI Distributed-Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: the expensive part of the algorithm is the evaluation of
// every remaining unclustered point as a candidate seed, which is independent
// across seeds. Those seed evaluations are distributed cyclically over all MPI
// ranks; a single MPI_Allreduce(MPI_MAXLOC) then selects the globally best
// candidate cluster (largest cardinality, lowest point index on ties, exactly
// as the sequential code does) and its owner broadcasts the member list so
// that every rank keeps an identical view of the clustering state.

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

static int g_rank = 0;
static int g_nranks = 1;

// Only rank 0 produces program output
#define RPRINTF(...)                       \
    do {                                   \
        if (g_rank == 0) printf(__VA_ARGS__); \
    } while (0)

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

// Compact, structure-of-arrays view of the points that are still unclustered.
// It is rebuilt (identically) on every rank once per outer iteration.
struct PointSet {
    std::vector<double> x, y;
    std::vector<int> index;  // original index of each entry
};

// Uniform bucket grid over a PointSet. The cell size is never smaller than the
// clustering threshold, so every point within the threshold of a given point is
// contained in the 3x3 block of cells around that point's own cell.
struct Grid {
    int nx = 1, ny = 1;
    double inv_cell = 1.0;
    std::vector<int> start;  // nx*ny+1 offsets into items
    std::vector<int> items;  // entry ids, ascending within each cell

    inline int cellX(const double px) const {
        return std::min(nx - 1, static_cast<int>(px * inv_cell));
    }
    inline int cellY(const double py) const {
        return std::min(ny - 1, static_cast<int>(py * inv_cell));
    }
};

// Reusable buffers for the candidate list of a single seed
struct Scratch {
    std::vector<int> id;         // entry ids of the candidates
    std::vector<double> x, y;    // candidate coordinates
    std::vector<double> maxd;    // max distance from candidate to cluster so far
};

void buildGrid(const PointSet& ps, const double threshold, Grid& g) {
    const int n = static_cast<int>(ps.index.size());

    // Keep the number of cells proportional to the number of points, while
    // still covering the threshold radius with a 3x3 block of cells.
    const double min_cell = std::sqrt(MAX_WIDTH * MAX_HEIGHT / std::max(1, n));
    const double cell = std::max(threshold, min_cell);

    g.inv_cell = 1.0 / cell;
    g.nx = static_cast<int>(MAX_WIDTH * g.inv_cell) + 1;
    g.ny = static_cast<int>(MAX_HEIGHT * g.inv_cell) + 1;

    const int ncells = g.nx * g.ny;
    g.start.assign(ncells + 1, 0);
    g.items.resize(n);

    // Counting sort of the entries into their cells
    for (int i = 0; i < n; ++i) {
        ++g.start[g.cellY(ps.y[i]) * g.nx + g.cellX(ps.x[i]) + 1];
    }
    for (int c = 0; c < ncells; ++c) {
        g.start[c + 1] += g.start[c];
    }
    std::vector<int> cursor(g.start.begin(), g.start.end() - 1);
    for (int i = 0; i < n; ++i) {
        g.items[cursor[g.cellY(ps.y[i]) * g.nx + g.cellX(ps.x[i])]++] = i;
    }
}

// Grow a candidate cluster from the seed entry `seed` of `ps`, using the same
// greedy rule as the sequential reference: repeatedly add the unclustered point
// whose resulting maximum distance to all current members is minimal and below
// the threshold (ties broken towards the lowest point index).
//
// Two exact optimizations are applied:
//   * only points closer than the threshold to the seed can ever be added,
//     since the seed itself is a member; the grid supplies them directly.
//   * the maximum distance of every candidate to the current members is
//     maintained incrementally instead of being recomputed from scratch.
//
// `members` receives the original point indices in insertion order.
// Returns the cardinality of the candidate cluster.
int generateCandidateCluster(const int seed,
                             const PointSet& ps,
                             const Grid& g,
                             const double threshold,
                             std::vector<int>& members,
                             Scratch& scr) {
    const double sx = ps.x[seed];
    const double sy = ps.y[seed];

    members.clear();
    members.push_back(ps.index[seed]);

    // Collect the candidates: unclustered points within the threshold of the seed
    scr.id.clear();
    const int cx0 = g.cellX(sx);
    const int cy0 = g.cellY(sy);
    const int cx_lo = std::max(0, cx0 - 1), cx_hi = std::min(g.nx - 1, cx0 + 1);
    const int cy_lo = std::max(0, cy0 - 1), cy_hi = std::min(g.ny - 1, cy0 + 1);

    for (int cy = cy_lo; cy <= cy_hi; ++cy) {
        for (int cx = cx_lo; cx <= cx_hi; ++cx) {
            const int c = cy * g.nx + cx;
            const int end = g.start[c + 1];
            for (int k = g.start[c]; k < end; ++k) {
                const int t = g.items[k];
                if (t == seed) continue;
                const double dx = ps.x[t] - sx;
                const double dy = ps.y[t] - sy;
                if (std::sqrt(dx * dx + dy * dy) < threshold) {
                    scr.id.push_back(t);
                }
            }
        }
    }

    // Ascending entry ids reproduce the candidate scan order of the reference
    std::sort(scr.id.begin(), scr.id.end());

    const int nc = static_cast<int>(scr.id.size());
    scr.x.resize(nc);
    scr.y.resize(nc);
    scr.maxd.resize(nc);
    for (int i = 0; i < nc; ++i) {
        const int t = scr.id[i];
        const double dx = ps.x[t] - sx;
        const double dy = ps.y[t] - sy;
        scr.x[i] = ps.x[t];
        scr.y[i] = ps.y[t];
        scr.maxd[i] = std::sqrt(dx * dx + dy * dy);
    }

    const double* __restrict cand_x = scr.x.data();
    const double* __restrict cand_y = scr.y.data();
    double* __restrict maxd = scr.maxd.data();
    const double kInf = std::numeric_limits<double>::infinity();

    // Iteratively add closest points
    for (;;) {
        // Find closest point to current cluster that maintains diameter < threshold
        double best = threshold;
        int best_i = -1;
        for (int i = 0; i < nc; ++i) {
            if (maxd[i] < best) {
                best = maxd[i];
                best_i = i;
            }
        }

        if (best_i < 0) break;  // No more points can be added

        members.push_back(ps.index[scr.id[best_i]]);

        // Members are excluded from further consideration via an infinite bound
        const double px = cand_x[best_i];
        const double py = cand_y[best_i];
        maxd[best_i] = kInf;
        for (int i = 0; i < nc; ++i) {
            const double dx = cand_x[i] - px;
            const double dy = cand_y[i] - py;
            const double d = std::sqrt(dx * dx + dy * dy);
            maxd[i] = std::max(maxd[i], d);
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (distributed over MPI ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    PointSet ps;
    Grid grid;
    Scratch scr;
    std::vector<int> best_members, candidate_members;
    int remaining = N;

    // Main clustering loop
    while (remaining > 0) {
        // Rebuild the compact set of unclustered points and its lookup grid.
        // This is replicated work, but only O(N) per cluster found.
        ps.x.clear();
        ps.y.clear();
        ps.index.clear();
        ps.x.reserve(remaining);
        ps.y.reserve(remaining);
        ps.index.reserve(remaining);
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) {
                ps.x.push_back(points[i].x);
                ps.y.push_back(points[i].y);
                ps.index.push_back(i);
            }
        }
        const int U = static_cast<int>(ps.index.size());
        buildGrid(ps, threshold, grid);

        // Try each unclustered point as a seed; seeds are spread cyclically
        // across the ranks, which balances the varying per-seed cost well
        // because consecutive indices are spatially correlated.
        int local_best[2] = {-1, -1};  // {cardinality, seed entry id}
        best_members.clear();
        for (int seed = g_rank; seed < U; seed += g_nranks) {
            const int cardinality = generateCandidateCluster(seed, ps, grid, threshold,
                                                             candidate_members, scr);
            if (cardinality > local_best[0]) {
                local_best[0] = cardinality;
                local_best[1] = seed;
                best_members.swap(candidate_members);
            }
        }

        // Largest cardinality wins; MPI_MAXLOC breaks ties towards the smallest
        // entry id, i.e. the lowest point index, matching the sequential order.
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // If we found a cluster, add it
        if (global_best[0] <= 0) break;  // No more clusters can be formed

        const int cardinality = global_best[0];
        const int seed = global_best[1];
        const int owner = seed % g_nranks;

        // The owner's local best is exactly the global best: it retained the
        // lowest-id seed of maximal cardinality among its own seeds.
        if (g_rank != owner) {
            best_members.assign(cardinality, 0);
        }
        MPI_Bcast(best_members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = ps.index[seed];
        cluster.members = best_members;
        clusters.push_back(std::move(cluster));

        // Mark all members as clustered
        for (int member : clusters.back().members) {
            clustered[member] = 1;
        }
        remaining -= cardinality;
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
    RPRINTF("Usage: %s [options]\n", progName);
    RPRINTF("Options:\n");
    RPRINTF("  -n <num>     Number of points (default: 1000)\n");
    RPRINTF("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    RPRINTF("  -v           Enable validation\n");
    RPRINTF("  -r           Print results for external validation\n");
    RPRINTF("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            RPRINTF("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        RPRINTF("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
               num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    RPRINTF("QT Clustering Benchmark\n");
    RPRINTF("Number of points: %d\n", num_points);
    RPRINTF("Distance threshold: %.2f\n", threshold);
    RPRINTF("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data. The generator is cheap and deterministic, so
    // every rank replicates it instead of broadcasting the point set.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    RPRINTF("Clustering time: %ld ms\n", cluster_time.count());
    RPRINTF("Clusters found: %zu\n", clusters.size());

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

    RPRINTF("Points clustered: %d / %d (%.1f%%)\n",
           total_clustered, num_points,
           100.0 * total_clustered / num_points);
    RPRINTF("Average cluster size: %.2f\n", avg_cluster_size);
    RPRINTF("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    RPRINTF("Performance: %.1f clusters/s, %.1f points/s\n",
           clusters_per_sec, points_per_sec);

    // Print results for external validation
    if (printResults && g_rank == 0) {
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
        // The clustering state is replicated, so rank 0 validates and shares
        // the verdict to keep the exit code consistent across ranks.
        int valid = 1;
        if (g_rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (valid) {
            RPRINTF("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            RPRINTF("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}

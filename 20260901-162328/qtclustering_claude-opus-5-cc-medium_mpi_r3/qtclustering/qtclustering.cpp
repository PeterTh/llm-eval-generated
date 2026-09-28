// QT Clustering Benchmark - MPI Distributed Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * Every rank redundantly generates the (deterministic) input data, so no
//     input distribution is required.
//   * Every rank owns a fixed round-robin share of the points as seeds and
//     grows the candidate clusters of its own seeds independently. Since the
//     points are generated in spatially coherent runs of consecutive indices,
//     the round-robin split spreads the (very unevenly sized) work evenly.
//   * Per round of the outer loop only two collectives are needed: an
//     MPI_Allreduce(MPI_MAXLOC) that selects the globally best seed with the
//     same tie-breaking rule as the sequential code (largest cardinality,
//     smallest point index on ties), plus a small MPI_Allreduce that publishes
//     a global pruning floor. The owner of the winning seed broadcasts its
//     members, so all ranks keep an identical view of the clustering state and
//     no final gather is required.
//   * Per-seed state is kept purely rank local, and the cache invalidation
//     sweep runs against a second grid holding only the rank's own points, so
//     no step of a round is replicated across ranks.
//
// Serial optimizations that preserve the exact results of the original code:
//   * Incremental candidate diameters: instead of recomputing the maximum
//     distance from a candidate to every cluster member, each candidate keeps
//     a running maximum which is updated with the distance to the newly added
//     member only (max over a set is order independent, hence bit-identical).
//   * Candidates whose running maximum reached the threshold are dropped
//     permanently (the maximum can only grow).
//   * A uniform grid restricts the initial candidate set of a seed to the
//     points that are closer than the threshold to that seed - all other
//     points can never join the cluster since the seed is a cluster member.
//   * Cardinalities are cached and only recomputed for the seeds whose
//     neighborhood actually changed, and seeds that provably cannot beat the
//     best cardinality of the round are never grown at all (see qtClustering).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
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

// Uniform bucket grid over the still unclustered points. It is used to find,
// for a given seed, all points that are closer than the threshold - only those
// can ever become members of the seed's candidate cluster.
struct NeighborGrid {
    double cell = 1.0;
    int nx = 1, ny = 1;
    int reach = 1;                 // number of cells to search in each direction
    std::vector<int> cell_start;   // size nx*ny+1
    std::vector<int> items;        // point indices, ascending within each cell
    std::vector<double> item_x;    // the corresponding coordinates, so that the
    std::vector<double> item_y;    // neighborhood scans are purely sequential
    int live_at_build = 0;         // number of points present at last rebuild

    void configure(const double threshold) {
        // Keep the number of cells bounded for very small thresholds; the
        // search radius compensates for cells larger than the threshold.
        cell = std::max(threshold, MAX_WIDTH / 512.0);
        nx = static_cast<int>(MAX_WIDTH / cell) + 1;
        ny = static_cast<int>(MAX_HEIGHT / cell) + 1;
        reach = static_cast<int>(std::ceil(threshold / cell));
        if (reach < 1) reach = 1;
        cell_start.assign(static_cast<size_t>(nx) * ny + 1, 0);
    }

    inline int cellX(const Point& p) const {
        int c = static_cast<int>(p.x / cell);
        return std::min(std::max(c, 0), nx - 1);
    }
    inline int cellY(const Point& p) const {
        int c = static_cast<int>(p.y / cell);
        return std::min(std::max(c, 0), ny - 1);
    }

    // Counting sort of the given point indices into the grid cells.
    void build(const std::vector<Point>& points, const std::vector<int>& live) {
        const size_t ncells = static_cast<size_t>(nx) * ny;
        std::fill(cell_start.begin(), cell_start.end(), 0);
        for (const int idx : live) {
            const size_t c = static_cast<size_t>(cellY(points[idx])) * nx + cellX(points[idx]);
            ++cell_start[c + 1];
        }
        for (size_t c = 0; c < ncells; ++c) {
            cell_start[c + 1] += cell_start[c];
        }
        items.resize(live.size());
        item_x.resize(live.size());
        item_y.resize(live.size());
        std::vector<int> cursor(cell_start.begin(), cell_start.end() - 1);
        for (const int idx : live) {
            const size_t c = static_cast<size_t>(cellY(points[idx])) * nx + cellX(points[idx]);
            const int slot = cursor[c]++;
            items[slot] = idx;
            item_x[slot] = points[idx].x;
            item_y[slot] = points[idx].y;
        }
        live_at_build = static_cast<int>(live.size());
    }
};

// Scratch buffers for growing a candidate cluster (reused across seeds).
struct ClusterWorkspace {
    std::vector<int> idx;      // candidate point indices
    std::vector<double> px;    // candidate coordinates (kept next to the index
    std::vector<double> py;    // arrays so the update loop can vectorize)
    std::vector<double> maxd;  // running max distance to the current members
    std::vector<int> members;

    void reserve(const int n) {
        idx.reserve(n);
        px.reserve(n);
        py.reserve(n);
        maxd.reserve(n);
        members.reserve(n);
    }
};

// Upper bound on the cardinality of the candidate cluster grown from a seed:
// only points closer than the threshold to the seed can ever be members, since
// the seed itself belongs to the cluster.
static int candidateCardinalityBound(const int seed_point,
                                     const std::vector<bool>& clustered,
                                     const std::vector<Point>& points,
                                     const double threshold,
                                     const NeighborGrid& grid) {
    const Point seed = points[seed_point];
    const int cx = grid.cellX(seed);
    const int cy = grid.cellY(seed);
    const int x0 = std::max(cx - grid.reach, 0);
    const int x1 = std::min(cx + grid.reach, grid.nx - 1);
    const int y0 = std::max(cy - grid.reach, 0);
    const int y1 = std::min(cy + grid.reach, grid.ny - 1);

    int count = 1; // the seed
    for (int gy = y0; gy <= y1; ++gy) {
        const size_t row = static_cast<size_t>(gy) * grid.nx;
        const int begin = grid.cell_start[row + x0];
        const int end = grid.cell_start[row + x1 + 1];
        for (int k = begin; k < end; ++k) {
            const int c = grid.items[k];
            if (c == seed_point || clustered[c]) continue;
            const double dx = grid.item_x[k] - seed.x;
            const double dy = grid.item_y[k] - seed.y;
            if (std::sqrt(dx * dx + dy * dy) < threshold) ++count;
        }
    }
    return count;
}

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster; the members are left in
// work.members.
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<bool>& clustered,
                                    const std::vector<Point>& points,
                                    const double threshold,
                                    const NeighborGrid& grid,
                                    ClusterWorkspace& work) {
    std::vector<int>& idx = work.idx;
    std::vector<double>& px = work.px;
    std::vector<double>& py = work.py;
    std::vector<double>& maxd = work.maxd;
    std::vector<int>& members = work.members;

    idx.clear();
    px.clear();
    py.clear();
    maxd.clear();
    members.clear();
    members.push_back(seed_point);

    const Point seed = points[seed_point];

    // Collect all unclustered points closer than the threshold to the seed.
    const int cx = grid.cellX(seed);
    const int cy = grid.cellY(seed);
    const int x0 = std::max(cx - grid.reach, 0);
    const int x1 = std::min(cx + grid.reach, grid.nx - 1);
    const int y0 = std::max(cy - grid.reach, 0);
    const int y1 = std::min(cy + grid.reach, grid.ny - 1);

    int best = -1;
    int best_idx = 0;
    double best_d = std::numeric_limits<double>::max();

    for (int gy = y0; gy <= y1; ++gy) {
        const size_t row = static_cast<size_t>(gy) * grid.nx;
        const int begin = grid.cell_start[row + x0];
        const int end = grid.cell_start[row + x1 + 1];
        for (int k = begin; k < end; ++k) {
            const int c = grid.items[k];
            if (c == seed_point || clustered[c]) continue;
            const double cx_ = grid.item_x[k];
            const double cy_ = grid.item_y[k];
            const double ddx = cx_ - seed.x;
            const double ddy = cy_ - seed.y;
            const double d = std::sqrt(ddx * ddx + ddy * ddy);
            if (d >= threshold) continue;
            if (d < best_d || (d == best_d && c < best_idx)) {
                best = static_cast<int>(idx.size());
                best_idx = c;
                best_d = d;
            }
            idx.push_back(c);
            px.push_back(cx_);
            py.push_back(cy_);
            maxd.push_back(d);
        }
    }

    // Iteratively add the candidate with the smallest cluster diameter.
    while (best >= 0) {
        const int added = idx[best];
        members.push_back(added);

        const double ax = px[best];
        const double ay = py[best];
        const int n = static_cast<int>(idx.size());

        // Update the running maxima (branch free, vectorizable).
        for (int i = 0; i < n; ++i) {
            const double dx = px[i] - ax;
            const double dy = py[i] - ay;
            const double d = std::sqrt(dx * dx + dy * dy);
            maxd[i] = maxd[i] > d ? maxd[i] : d;
        }

        // Compact out the candidates that can no longer join and pick the next
        // best one. The newly added member is dropped as well.
        int out = 0;
        int nbest = -1;
        int nbest_idx = 0;
        double nbest_d = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            if (i == best || maxd[i] >= threshold) continue;
            const double d = maxd[i];
            const int c = idx[i];
            if (d < nbest_d || (d == nbest_d && c < nbest_idx)) {
                nbest = out;
                nbest_idx = c;
                nbest_d = d;
            }
            idx[out] = c;
            px[out] = px[i];
            py[out] = py[i];
            maxd[out] = d;
            ++out;
        }
        idx.resize(out);
        px.resize(out);
        py.resize(out);
        maxd.resize(out);
        best = nbest;
        best_idx = nbest_idx;
        best_d = nbest_d;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (distributed over MPI_COMM_WORLD)
//
// Two properties are exploited to avoid redundant work:
//   * The candidate cluster grown from a seed depends only on the unclustered
//     points that are closer than the threshold to that seed. Hence, after a
//     cluster has been removed, only seeds that had one of the removed points
//     in their neighborhood can change - all other cardinalities stay valid and
//     are reused from the previous round.
//   * The number of unclustered points within the threshold of a seed is an
//     upper bound for its cardinality and is maintained exactly and cheaply.
//     Stale seeds are visited in order of decreasing bound, so once a bound
//     drops below the best cardinality found so far, no later seed can win and
//     the scan stops.
//
// Every rank owns a fixed round-robin share of the points as seeds, which keeps
// the per-seed caches purely local: only the winning cluster is exchanged.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    NeighborGrid grid;
    grid.configure(threshold);
    grid.build(points, unclustered_indices);

    // A second grid holding only the seeds owned by this rank. It makes the
    // per-round cache invalidation sweep touch 1/nranks of the points, so that
    // step scales with the rank count instead of being replicated work.
    std::vector<int> owned_unclustered;
    for (int i = rank; i < N; i += nranks) owned_unclustered.push_back(i);
    NeighborGrid owned_grid;
    owned_grid.configure(threshold);
    owned_grid.build(points, owned_unclustered);

    ClusterWorkspace work;
    work.reserve(N);

    // Per-seed state, maintained only for the seeds owned by this rank.
    std::vector<int> nbrs(N, 0);              // unclustered points within threshold
    std::vector<int> card(N, INT_MAX);        // cardinality (exact, or upper bound)
    std::vector<char> exact(N, 0);            // is card[] the current cardinality?
    std::vector<int> dirty;                   // owned seeds needing a recomputation
    std::vector<int> best_cluster_members;
    std::vector<int> bcast_members(N);

    dirty.reserve(N / nranks + 1);

    // Initial neighbor counts for the owned seeds.
    for (int i = rank; i < N; i += nranks) {
        nbrs[i] = candidateCardinalityBound(i, clustered, points, threshold, grid);
    }

    struct { int card; int seed; } local, global;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int remaining = static_cast<int>(unclustered_indices.size());

        // Rebuild the grid once a significant fraction of its points has been
        // clustered, so that the neighborhood scans stay cheap.
        const bool rebuild = remaining * 4 <= grid.live_at_build * 3;
        if (rebuild) {
            grid.build(points, unclustered_indices);
        }

        // Best cached (still valid) cardinality among the owned seeds; the
        // ascending scan implements the "smallest index wins ties" rule.
        local.card = -1;
        local.seed = N;
        bool have_members = false;
        dirty.clear();
        owned_unclustered.clear();
        for (int i = rank; i < N; i += nranks) {
            if (clustered[i]) continue;
            owned_unclustered.push_back(i);
            if (exact[i]) {
                if (card[i] > local.card) {
                    local.card = card[i];
                    local.seed = i;
                }
            } else {
                dirty.push_back(i);
            }
        }

        if (rebuild) {
            owned_grid.build(points, owned_unclustered);
        }

        // Visit the stale seeds by decreasing upper bound.
        std::stable_sort(dirty.begin(), dirty.end(),
                         [&nbrs](int a, int b) { return nbrs[a] > nbrs[b]; });

        // A cardinality that some rank already achieved with a cached value is
        // a free global pruning floor for this round.
        int floor_card = local.card;
        MPI_Allreduce(MPI_IN_PLACE, &floor_card, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (const int seed : dirty) {
            const int bound = nbrs[seed];
            // Bounds are non-increasing along the scan: once a seed cannot beat
            // the best cardinality known so far, neither can any later one.
            if (bound < std::max(floor_card, local.card)) break;
            // On an equal bound only a smaller point index could still win.
            if (bound == local.card && seed > local.seed) continue;

            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                             threshold, grid, work);
            card[seed] = cardinality;
            exact[seed] = 1;
            if (cardinality > local.card ||
                (cardinality == local.card && seed < local.seed)) {
                local.card = cardinality;
                local.seed = seed;
                best_cluster_members = work.members;
                have_members = true;
            }
        }

        // MPI_MAXLOC keeps the largest cardinality and, on ties, the smallest
        // point index - exactly the sequential selection rule.
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global.seed >= N || global.card <= 0) {
            // No more clusters can be formed
            break;
        }

        // The winning seed is owned by exactly one rank, which broadcasts its
        // members. If the winner came from the cache its members are not around
        // any more and are regenerated (a single cluster, negligible cost).
        const int owner = global.seed % nranks;
        if (rank == owner) {
            if (!have_members) {
                generateCandidateCluster(global.seed, clustered, points,
                                         threshold, grid, work);
                best_cluster_members = work.members;
            }
            std::copy(best_cluster_members.begin(), best_cluster_members.end(),
                      bcast_members.begin());
        }
        MPI_Bcast(bcast_members.data(), global.card, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global.seed;
        cluster.members.assign(bcast_members.begin(), bcast_members.begin() + global.card);

        // Mark all members as clustered
        for (const int m : cluster.members) {
            clustered[m] = true;
        }

        // Invalidate the cached cardinality of every owned seed that had one of
        // the removed points in its neighborhood, and update its neighbor count.
        for (const int m : cluster.members) {
            const Point pm = points[m];
            const int cx = owned_grid.cellX(pm);
            const int cy = owned_grid.cellY(pm);
            const int x0 = std::max(cx - owned_grid.reach, 0);
            const int x1 = std::min(cx + owned_grid.reach, owned_grid.nx - 1);
            const int y0 = std::max(cy - owned_grid.reach, 0);
            const int y1 = std::min(cy + owned_grid.reach, owned_grid.ny - 1);
            for (int gy = y0; gy <= y1; ++gy) {
                const size_t row = static_cast<size_t>(gy) * owned_grid.nx;
                const int begin = owned_grid.cell_start[row + x0];
                const int end = owned_grid.cell_start[row + x1 + 1];
                for (int k = begin; k < end; ++k) {
                    const int c = owned_grid.items[k];
                    if (clustered[c]) continue;
                    const double dx = owned_grid.item_x[k] - pm.x;
                    const double dy = owned_grid.item_y[k] - pm.y;
                    if (std::sqrt(dx * dx + dy * dy) < threshold) {
                        // The neighbor count stays exact and remains a valid
                        // bound; the cardinality itself does not (a greedy
                        // cluster can grow when a point is taken away).
                        --nbrs[c];
                        card[c] = INT_MAX;
                        exact[c] = 0;
                    }
                }
            }
        }

        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
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

    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

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

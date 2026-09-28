// QT Clustering Benchmark - MPI Distributed Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   The expensive part of each outer iteration is the evaluation of every
//   still-unclustered point as a candidate seed; those evaluations are
//   completely independent of each other.  The seeds of one iteration are
//   distributed round-robin over the MPI ranks, the globally best candidate
//   cluster is selected with a single MPI_Allreduce(MPI_MAXLOC) and its
//   member list is broadcast so that every rank keeps an identical view of
//   the clustering state.
//
// Serial optimizations (semantics preserving):
//   * the maximum distance of a candidate to the current cluster is updated
//     incrementally instead of being recomputed from scratch,
//   * candidates that can never be added again (max distance >= threshold is
//     monotonically non-decreasing) are compacted away,
//   * a uniform grid with cell size == threshold restricts the initial
//     candidate set to the neighbourhood of the seed, since every cluster
//     member must be within the threshold of the seed point,
//   * squared distances are used throughout (monotone transformation of the
//     Euclidean distance, so all comparisons are equivalent).

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

// Uniform bucket grid over the still unclustered points.  The cell size equals
// the clustering threshold, hence all points that may end up in a cluster with
// a given seed live in the 3x3 cell block around the seed's cell.
struct Grid {
    int nx = 0, ny = 0;
    double inv_cell = 0.0;
    std::vector<int> start;    // CSR offsets, size nx*ny+1
    std::vector<int> idx;      // point ids, ascending within each cell
    std::vector<double> x, y;  // coordinates in grid order

    inline int cellX(const double px) const {
        int c = static_cast<int>(px * inv_cell);
        if (c < 0) c = 0;
        if (c >= nx) c = nx - 1;
        return c;
    }
    inline int cellY(const double py) const {
        int c = static_cast<int>(py * inv_cell);
        if (c < 0) c = 0;
        if (c >= ny) c = ny - 1;
        return c;
    }
};

// (Re-)build the grid from the given (ascending) list of unclustered points
static void buildGrid(Grid& g, const std::vector<Point>& points,
                      const std::vector<int>& active, const double threshold) {
    const double cell = threshold;
    g.inv_cell = 1.0 / cell;
    g.nx = static_cast<int>(MAX_WIDTH / cell) + 1;
    g.ny = static_cast<int>(MAX_HEIGHT / cell) + 1;

    const int ncells = g.nx * g.ny;
    g.start.assign(ncells + 1, 0);
    const int n = static_cast<int>(active.size());
    g.idx.resize(n);
    g.x.resize(n);
    g.y.resize(n);

    // Counting sort of the active points into their cells
    std::vector<int> cellOf(n);
    for (int i = 0; i < n; ++i) {
        const Point& p = points[active[i]];
        const int c = g.cellY(p.y) * g.nx + g.cellX(p.x);
        cellOf[i] = c;
        ++g.start[c + 1];
    }
    for (int c = 0; c < ncells; ++c) {
        g.start[c + 1] += g.start[c];
    }
    std::vector<int> cursor(g.start.begin(), g.start.end() - 1);
    for (int i = 0; i < n; ++i) {
        const int pos = cursor[cellOf[i]]++;
        const int id = active[i];
        g.idx[pos] = id;
        g.x[pos] = points[id].x;
        g.y[pos] = points[id].y;
    }
}

// Scratch buffers for the candidate evaluation (reused across seeds)
struct Workspace {
    std::vector<int> idx;
    std::vector<double> x, y, d2;
    std::vector<int> order;
    std::vector<int> members;
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster; the members are stored in
// ws.members (ordered exactly like in the sequential reference version).
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<Point>& points,
                                    const Grid& grid,
                                    const double threshold,
                                    Workspace& ws) {
    const double thr2 = threshold * threshold;
    const Point sp = points[seed_point];

    ws.members.clear();
    ws.members.push_back(seed_point);

    // Collect all candidates within the threshold of the seed
    ws.order.clear();
    const int cx = grid.cellX(sp.x);
    const int cy = grid.cellY(sp.y);
    const int x0 = std::max(cx - 1, 0), x1 = std::min(cx + 1, grid.nx - 1);
    const int y0 = std::max(cy - 1, 0), y1 = std::min(cy + 1, grid.ny - 1);
    for (int gy = y0; gy <= y1; ++gy) {
        const int row = gy * grid.nx;
        for (int gx = x0; gx <= x1; ++gx) {
            const int b = grid.start[row + gx], e = grid.start[row + gx + 1];
            for (int p = b; p < e; ++p) {
                if (grid.idx[p] == seed_point) continue;
                const double dx = grid.x[p] - sp.x;
                const double dy = grid.y[p] - sp.y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < thr2) ws.order.push_back(p);
            }
        }
    }

    // The reference implementation scans candidates in ascending point id and
    // keeps the first of several equally good ones; reproduce that ordering.
    std::sort(ws.order.begin(), ws.order.end(),
              [&grid](int a, int b) { return grid.idx[a] < grid.idx[b]; });

    int n = static_cast<int>(ws.order.size());
    ws.idx.resize(n);
    ws.x.resize(n);
    ws.y.resize(n);
    ws.d2.resize(n);
    int best = -1;
    double best_d2 = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        const int p = ws.order[i];
        ws.idx[i] = grid.idx[p];
        ws.x[i] = grid.x[p];
        ws.y[i] = grid.y[p];
        const double dx = grid.x[p] - sp.x;
        const double dy = grid.y[p] - sp.y;
        const double d2 = dx * dx + dy * dy;
        ws.d2[i] = d2;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }

    int* __restrict cidx = ws.idx.data();
    double* __restrict cx_ = ws.x.data();
    double* __restrict cy_ = ws.y.data();
    double* __restrict cd2 = ws.d2.data();

    // Iteratively add the candidate with the smallest cluster diameter
    while (best >= 0) {
        const int newMember = cidx[best];
        const double nx = cx_[best], ny = cy_[best];
        ws.members.push_back(newMember);

        // Update the per-candidate maximum distance to the cluster, drop the
        // candidates that exceeded the threshold and pick the next best one.
        int w = 0;
        int nextBest = -1;
        double nextBestD2 = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            if (i == best) continue;
            const double dx = cx_[i] - nx;
            const double dy = cy_[i] - ny;
            const double d2 = dx * dx + dy * dy;
            const double m = d2 > cd2[i] ? d2 : cd2[i];
            if (m < thr2) {
                cidx[w] = cidx[i];
                cx_[w] = cx_[i];
                cy_[w] = cy_[i];
                cd2[w] = m;
                if (m < nextBestD2) {
                    nextBestD2 = m;
                    nextBest = w;
                }
                ++w;
            }
        }
        n = w;
        best = nextBest;
    }

    return static_cast<int>(ws.members.size());
}

// Main QT clustering algorithm (distributed over MPI ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    Grid grid;
    Workspace ws;
    std::vector<int> best_members;
    best_members.reserve(N);
    std::vector<int> recv_members(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        buildGrid(grid, points, unclustered_indices, threshold);

        int max_cardinality = -1;
        int best_seed = -1;
        best_members.clear();

        // Try each unclustered point as a seed; seeds are spread round-robin
        // over the ranks, which balances the very different per-seed costs.
        const int nseeds = static_cast<int>(unclustered_indices.size());
        for (int i = rank; i < nseeds; i += nranks) {
            const int seed = unclustered_indices[i];
            const int cardinality =
                generateCandidateCluster(seed, points, grid, threshold, ws);

            // Strictly better only: the reference scans seeds in ascending
            // order and keeps the first of several equally large clusters
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_members = ws.members;
            }
        }

        // Global reduction: maximize cardinality, break ties by smallest seed.
        // The key packs both criteria so that a single MAXLOC also yields the
        // rank owning the winning cluster.
        struct { long key; int rank; } in, out;
        in.key = (max_cardinality >= 0)
                     ? static_cast<long>(max_cardinality) * (N + 1L) + (N - best_seed)
                     : -1L;
        in.rank = rank;
        MPI_Allreduce(&in, &out, 1, MPI_LONG_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (out.key < 0) break; // No more clusters can be formed

        const int winner = out.rank;
        const int global_card = static_cast<int>(out.key / (N + 1L));
        const int global_seed = N - static_cast<int>(out.key % (N + 1L));
        if (global_card <= 0) break;

        int count = global_card;
        const int* src;
        if (rank == winner) {
            src = best_members.data();
            MPI_Bcast(const_cast<int*>(src), count, MPI_INT, winner, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(recv_members.data(), count, MPI_INT, winner, MPI_COMM_WORLD);
            src = recv_members.data();
        }

        Cluster cluster;
        cluster.seed_point = global_seed;
        cluster.members.assign(src, src + count);
        for (int i = 0; i < count; ++i) {
            clustered[src[i]] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list
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

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time = cluster_time.count();
    long global_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &global_cluster_time, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

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
        printf("Clustering time: %ld ms\n", global_cluster_time);
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = global_cluster_time / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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
    int status = 0;
    if (validate) {
        if (rank == 0) {
            const bool valid = validateClusters(clusters, points, threshold);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return status;
}

// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: every rank holds the point set and a replicated copy of the
// clustering state. In each round the candidate clusters of the unclustered
// seeds are split round-robin across ranks, the best seed (largest cardinality,
// lowest index on ties) is selected with MPI_MAXLOC, and every rank applies the
// same update. The result is identical to the sequential algorithm for any
// number of ranks.

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
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
static_assert(sizeof(Point) == 2 * sizeof(double), "Point must be two packed doubles");

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// (value, location) pair for MPI_MAXLOC: largest value wins, lowest location on ties
struct ValueLoc {
    int value;
    int loc;
};

// Unclustered candidate of the cluster being grown, with its maximum distance
// to the current cluster members
struct Candidate {
    int point;
    double max_dist;
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

// Uniform grid over the bounding box of the points. Cells are at least as wide
// as the threshold, so every point closer than the threshold to a seed lies in
// the seed's 3x3 cell neighbourhood.
struct SpatialGrid {
    double min_x = 0.0, min_y = 0.0;
    double cell_size = 1.0;
    int nx = 1, ny = 1;
    std::vector<int> cell_start;   // CSR offsets into cell_points, size nx*ny+1
    std::vector<int> cell_points;  // point indices grouped by cell

    int cellX(double x) const {
        return std::min(nx - 1, static_cast<int>((x - min_x) / cell_size));
    }
    int cellY(double y) const {
        return std::min(ny - 1, static_cast<int>((y - min_y) / cell_size));
    }
};

SpatialGrid buildSpatialGrid(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    double min_x = points[0].x, max_x = points[0].x;
    double min_y = points[0].y, max_y = points[0].y;
    for (const Point& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    const double width = max_x - min_x;
    const double height = max_y - min_y;

    // Cell size is at least the threshold, and large enough that the grid has
    // O(N) cells. The small factor keeps the neighbour test safe from rounding
    // at cell boundaries.
    SpatialGrid grid;
    grid.min_x = min_x;
    grid.min_y = min_y;
    grid.cell_size = std::max({threshold, std::sqrt(width * height / N),
                               std::max(width, height) / N}) * (1.0 + 1e-6);
    grid.nx = static_cast<int>(width / grid.cell_size) + 1;
    grid.ny = static_cast<int>(height / grid.cell_size) + 1;

    // Counting sort of the points into cells (CSR layout)
    const int num_cells = grid.nx * grid.ny;
    std::vector<int> cell_of(N);
    grid.cell_start.assign(num_cells + 1, 0);
    for (int i = 0; i < N; ++i) {
        cell_of[i] = grid.cellY(points[i].y) * grid.nx + grid.cellX(points[i].x);
        ++grid.cell_start[cell_of[i] + 1];
    }
    for (int c = 0; c < num_cells; ++c) {
        grid.cell_start[c + 1] += grid.cell_start[c];
    }
    std::vector<int> fill(grid.cell_start.begin(), grid.cell_start.end() - 1);
    grid.cell_points.resize(N);
    for (int i = 0; i < N; ++i) {
        grid.cell_points[fill[cell_of[i]]++] = i;
    }
    return grid;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const SpatialGrid& grid,
                              const double threshold,
                              std::vector<Candidate>& candidates,
                              std::vector<int>* cluster_members = nullptr) {
    // A point at distance >= threshold from the seed can never be added, because
    // the diameter is at least that distance. Only nearby unclustered points are
    // therefore candidates, and a candidate whose max distance to the members
    // reaches the threshold can be dropped for good, since it never decreases.
    candidates.clear();
    const Point& seed = points[seed_point];
    const int sx = grid.cellX(seed.x);
    const int sy = grid.cellY(seed.y);
    for (int cy = std::max(sy - 1, 0); cy <= std::min(sy + 1, grid.ny - 1); ++cy) {
        for (int cx = std::max(sx - 1, 0); cx <= std::min(sx + 1, grid.nx - 1); ++cx) {
            const int cell = cy * grid.nx + cx;
            for (int k = grid.cell_start[cell]; k < grid.cell_start[cell + 1]; ++k) {
                const int p = grid.cell_points[k];
                if (p == seed_point || clustered[p]) continue;
                const double dist = distance(points[p], seed);
                if (dist < threshold) {
                    candidates.push_back({p, dist});
                }
            }
        }
    }

    int cardinality = 1;
    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    // Iteratively add the candidate with the smallest max distance to the cluster
    // (lowest point index on ties)
    while (!candidates.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < candidates.size(); ++i) {
            const Candidate& c = candidates[i];
            const Candidate& b = candidates[best];
            if (c.max_dist < b.max_dist || (c.max_dist == b.max_dist && c.point < b.point)) {
                best = i;
            }
        }

        const int closest = candidates[best].point;
        candidates[best] = candidates.back();
        candidates.pop_back();
        ++cardinality;
        if (cluster_members) {
            cluster_members->push_back(closest);
        }

        size_t i = 0;
        while (i < candidates.size()) {
            Candidate& c = candidates[i];
            c.max_dist = std::max(c.max_dist, distance(points[c.point], points[closest]));
            if (c.max_dist >= threshold) {
                candidates[i] = candidates.back();
                candidates.pop_back();
            } else {
                ++i;
            }
        }
    }

    return cardinality;
}

// Main QT clustering algorithm. Every rank returns the same clusters.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int num_ranks) {
    const int N = static_cast<int>(points.size());
    const SpatialGrid grid = buildSpatialGrid(points, threshold);
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    std::vector<Candidate> candidates;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Try this rank's share of the unclustered points as seeds. The list is
        // kept in ascending order, so the first maximum found is the lowest index.
        ValueLoc local{-1, std::numeric_limits<int>::max()};
        for (size_t i = rank; i < unclustered_indices.size(); i += num_ranks) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(seed, clustered, points, grid,
                                                             threshold, candidates);
            if (cardinality > local.value) {
                local = {cardinality, seed};
            }
        }

        // Global best: largest cardinality, lowest seed index on ties
        ValueLoc global{-1, -1};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_seed = global.loc;
        const int max_cardinality = global.value;

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            generateCandidateCluster(best_seed, clustered, points, grid, threshold,
                                     candidates, &cluster.members);
            clusters.push_back(std::move(cluster));

            // Mark all members as clustered
            for (int member : clusters.back().members) {
                clustered[member] = true;
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

// Validation: check that clusters satisfy the QT clustering properties.
// Cluster diameters are computed in parallel and shared with all ranks; only
// rank 0 prints.
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int rank,
                     const int num_ranks) {
    const bool root = rank == 0;
    bool valid = true;

    if (root) printf("Validating clusters:\n");

    // Each rank computes the diameters of its share of clusters. The other
    // entries stay 0, so a max-reduction gives every rank the full set.
    std::vector<double> diameters(clusters.size(), 0.0);
    for (size_t c = rank; c < clusters.size(); c += num_ranks) {
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

    // Check each cluster
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

// Benchmark driver. All ranks parse the same arguments and return the same
// status; only rank 0 prints.
static int runBenchmark(int argc, char** argv, const int rank, const int num_ranks) {
    const bool root = rank == 0;
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

    // Generate synthetic data on rank 0 and share it with all ranks
    std::vector<Point> points(num_points);
    if (root) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1,
               MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (root) cluster_time = std::chrono::milliseconds(max_cluster_time_ms);

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
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
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

    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, num_ranks);

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
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const int status = runBenchmark(argc, argv, rank, num_ranks);

    MPI_Finalize();
    return status;
}

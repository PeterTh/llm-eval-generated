// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: every rank holds the full point set and the replicated
// cluster state. Seed point s is evaluated by rank s % size. Each rank keeps a
// priority queue of the candidate-cluster sizes of its own seeds, and one
// MPI_Allreduce(MPI_MAXLOC) per round picks the globally largest candidate
// cluster (lowest seed index on ties), exactly as the sequential algorithm does.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const double MAX_GRID_CELLS_PER_AXIS = 1024.0;

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

// Uniform grid over the points. Cells are at least as wide as the threshold,
// so every point within the threshold of p lies in the 3x3 block of cells
// around p. Callers still apply the exact distance test.
class PointGrid {
  public:
    PointGrid(const std::vector<Point>& points, const double threshold) {
        min_x_ = points[0].x;
        min_y_ = points[0].y;
        double max_x = min_x_;
        double max_y = min_y_;
        for (const Point& p : points) {
            min_x_ = std::min(min_x_, p.x);
            min_y_ = std::min(min_y_, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }

        // Slightly widened cells absorb rounding in the cell index; the cap on
        // cells per axis keeps the grid small for very small thresholds.
        const double extent = std::max(max_x - min_x_, max_y - min_y_);
        cell_ = std::max(threshold * 1.0001, extent / MAX_GRID_CELLS_PER_AXIS);
        nx_ = static_cast<int>((max_x - min_x_) / cell_) + 1;
        ny_ = static_cast<int>((max_y - min_y_) / cell_) + 1;

        // Counting sort of point indices by cell (CSR layout)
        start_.assign(static_cast<size_t>(nx_) * ny_ + 1, 0);
        for (const Point& p : points) {
            ++start_[cellOf(p) + 1];
        }
        std::partial_sum(start_.begin(), start_.end(), start_.begin());

        std::vector<int> fill(start_.begin(), start_.end() - 1);
        items_.resize(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            items_[fill[cellOf(points[i])]++] = static_cast<int>(i);
        }
    }

    // Calls visit(index) for every point in the 3x3 cell block around p
    template <typename Visit>
    void forEachNear(const Point& p, Visit&& visit) const {
        const int cx = cellX(p.x);
        const int cy = cellY(p.y);
        for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, ny_ - 1); ++y) {
            for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, nx_ - 1); ++x) {
                const int cell = y * nx_ + x;
                for (int k = start_[cell]; k < start_[cell + 1]; ++k) {
                    visit(items_[k]);
                }
            }
        }
    }

  private:
    int cellX(const double x) const {
        return std::min(nx_ - 1, static_cast<int>((x - min_x_) / cell_));
    }
    int cellY(const double y) const {
        return std::min(ny_ - 1, static_cast<int>((y - min_y_) / cell_));
    }
    int cellOf(const Point& p) const {
        return cellY(p.y) * nx_ + cellX(p.x);
    }

    double min_x_ = 0.0, min_y_ = 0.0, cell_ = 1.0;
    int nx_ = 1, ny_ = 1;
    std::vector<int> start_;
    std::vector<int> items_;
};

// Grows the candidate cluster of a seed point. Only unclustered points within
// the threshold of the seed can ever join, so the search is restricted to them.
// Each candidate's maximum distance to the cluster is maintained incrementally,
// and candidates that reach the threshold are dropped for good.
class CandidateGrower {
  public:
    CandidateGrower(const std::vector<Point>& points, const PointGrid& grid,
                    const double threshold, const std::vector<char>& clustered)
        : points_(points), grid_(grid), threshold_(threshold), clustered_(clustered) {}

    // Writes the members (seed first, then in insertion order) and returns their count
    int grow(const int seed, std::vector<int>& members) {
        members.assign(1, seed);
        active_.clear();

        grid_.forEachNear(points_[seed], [&](int candidate) {
            if (candidate == seed || clustered_[candidate]) return;
            const double dist = distance(points_[candidate], points_[seed]);
            if (dist < threshold_) {
                active_.push_back({candidate, dist});
            }
        });
        // Ascending index order makes ties resolve to the lowest index
        std::sort(active_.begin(), active_.end(),
                  [](const Candidate& a, const Candidate& b) { return a.index < b.index; });

        while (!active_.empty()) {
            size_t best = 0;
            for (size_t i = 1; i < active_.size(); ++i) {
                if (active_[i].max_dist < active_[best].max_dist) best = i;
            }
            const int chosen = active_[best].index;
            members.push_back(chosen);
            active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(best));

            size_t kept = 0;
            for (size_t i = 0; i < active_.size(); ++i) {
                Candidate cand = active_[i];
                cand.max_dist = std::max(cand.max_dist,
                                         distance(points_[cand.index], points_[chosen]));
                if (cand.max_dist < threshold_) {
                    active_[kept++] = cand;
                }
            }
            active_.resize(kept);
        }

        return static_cast<int>(members.size());
    }

  private:
    struct Candidate {
        int index;
        double max_dist;
    };

    const std::vector<Point>& points_;
    const PointGrid& grid_;
    const double threshold_;
    const std::vector<char>& clustered_;
    std::vector<Candidate> active_;
};

// Candidate-cluster size of a seed, tagged with the seed version it was computed at
struct SeedEntry {
    int cardinality;
    int seed;
    int version;
};

// Max-heap order: larger cluster first, lower seed index on ties
struct SeedOrder {
    bool operator()(const SeedEntry& a, const SeedEntry& b) const {
        if (a.cardinality != b.cardinality) return a.cardinality < b.cardinality;
        return a.seed > b.seed;
    }
};

// Main QT clustering algorithm, distributed over MPI ranks.
// Every rank returns the same clusters.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int size) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> version(N, 0);
    std::vector<int> stamp(N, -1);
    std::vector<Cluster> clusters;

    const PointGrid grid(points, threshold);
    CandidateGrower grower(points, grid, threshold, clustered);

    std::priority_queue<SeedEntry, std::vector<SeedEntry>, SeedOrder> heap;
    std::vector<int> scratch;

    // Recompute the candidate cluster of an owned seed and queue its size
    auto refresh = [&](const int seed) {
        ++version[seed];
        heap.push({grower.grow(seed, scratch), seed, version[seed]});
    };

    for (int s = rank; s < N; s += size) {
        refresh(s);
    }

    int remaining = N;
    std::vector<int> affected;
    for (int round = 0; remaining > 0; ++round) {
        // Best owned seed: discard stale heap entries until the top is current
        int local[2] = {-1, N};  // {cardinality, seed}
        while (!heap.empty()) {
            const SeedEntry top = heap.top();
            if (!clustered[top.seed] && top.version == version[top.seed]) {
                local[0] = top.cardinality;
                local[1] = top.seed;
                break;
            }
            heap.pop();
        }

        int global[2];
        MPI_Allreduce(local, global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int max_cardinality = global[0];
        const int best_seed = global[1];

        // No cluster can be formed (matches the sequential early exit)
        if (best_seed < 0 || max_cardinality <= 0) break;

        // Every rank regrows the winning cluster; the result is identical everywhere
        clusters.emplace_back();
        Cluster& cluster = clusters.back();
        cluster.seed_point = best_seed;
        grower.grow(best_seed, cluster.members);

        for (int p : cluster.members) {
            clustered[p] = 1;
        }
        remaining -= static_cast<int>(cluster.members.size());

        // Owned seeds whose candidate pool contained a newly clustered point
        // must be recomputed; all other cached sizes remain valid
        affected.clear();
        for (int p : cluster.members) {
            grid.forEachNear(points[p], [&](int q) {
                if (q % size != rank || clustered[q] || stamp[q] == round) return;
                if (distance(points[q], points[p]) < threshold) {
                    stamp[q] = round;
                    affected.push_back(q);
                }
            });
        }
        for (int q : affected) {
            refresh(q);
        }
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties.
// Diameters are computed in parallel and all ranks reach the same verdict;
// only rank 0 prints.
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int rank, const int size) {
    const bool root = rank == 0;
    bool valid = true;

    if (root) printf("Validating clusters:\n");

    // Check diameter (max distance between any two points) of owned clusters
    std::vector<double> diameters(clusters.size(), 0.0);
    for (size_t c = static_cast<size_t>(rank); c < clusters.size();
         c += static_cast<size_t>(size)) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
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
            if (root) printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
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
                if (root) printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    const bool root = rank == 0;
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

    // Generate synthetic data (deterministic, so every rank builds the same set)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_end = MPI_Wtime();
    const long cluster_ms = static_cast<long>((cluster_end - cluster_start) * 1000.0);

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
        printf("Clustering time: %ld ms\n", cluster_ms);
        printf("Clusters found: %zu\n", clusters.size());
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
        if (root) print_results(membershipData, "ClusterMembership");
    }

    // Validation (collective: every rank takes part)
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, size);

        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}

// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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

// Uniform grid over the domain used to find all points within the threshold
// of a given point. Cells are slightly larger than the threshold so that any
// pair of points at distance < threshold lies in the same or adjacent cells,
// even in the presence of floating point rounding.
struct SpatialGrid {
    double cell_size;
    int nx, ny;
    std::vector<int> cell_start;  // CSR offsets, size nx*ny+1
    std::vector<int> cell_points; // point indices, ascending within each cell

    SpatialGrid(const std::vector<Point>& points, const double threshold) {
        const int MAX_CELLS_PER_DIM = 1024;
        const double max_dim = std::max(MAX_WIDTH, MAX_HEIGHT);
        cell_size = std::max(threshold * 1.0001, max_dim / MAX_CELLS_PER_DIM);
        nx = std::max(1, std::min(MAX_CELLS_PER_DIM, static_cast<int>(MAX_WIDTH / cell_size) + 1));
        ny = std::max(1, std::min(MAX_CELLS_PER_DIM, static_cast<int>(MAX_HEIGHT / cell_size) + 1));

        const int N = static_cast<int>(points.size());
        std::vector<int> cell_of(N);
        cell_start.assign(static_cast<size_t>(nx) * ny + 1, 0);
        for (int i = 0; i < N; ++i) {
            cell_of[i] = cellIndex(points[i]);
            cell_start[cell_of[i] + 1]++;
        }
        for (size_t c = 0; c + 1 < cell_start.size(); ++c) {
            cell_start[c + 1] += cell_start[c];
        }
        cell_points.resize(N);
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int i = 0; i < N; ++i) {
            cell_points[fill[cell_of[i]]++] = i;
        }
    }

    inline int cellX(double x) const {
        return std::clamp(static_cast<int>(std::floor(x / cell_size)), 0, nx - 1);
    }
    inline int cellY(double y) const {
        return std::clamp(static_cast<int>(std::floor(y / cell_size)), 0, ny - 1);
    }
    inline int cellIndex(const Point& p) const {
        return cellY(p.y) * nx + cellX(p.x);
    }

    // Call f(j) for every point j in the 3x3 cell neighbourhood of p
    template <typename F>
    inline void forEachNear(const Point& p, F&& f) const {
        const int cx = cellX(p.x), cy = cellY(p.y);
        for (int yy = std::max(0, cy - 1); yy <= std::min(ny - 1, cy + 1); ++yy) {
            for (int xx = std::max(0, cx - 1); xx <= std::min(nx - 1, cx + 1); ++xx) {
                const int c = yy * nx + xx;
                for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                    f(cell_points[k]);
                }
            }
        }
    }
};

// Reusable scratch space for candidate cluster generation
struct Workspace {
    std::vector<int> cand;      // active candidate indices (ascending order)
    std::vector<double> maxd;   // current max distance from candidate to cluster
    std::vector<int> members;
};

// Generate a candidate cluster starting from a seed point.
// Equivalent to the greedy procedure: repeatedly add the unclustered point
// with the smallest maximum distance to the current members (ties broken by
// lowest index) as long as that distance is below the threshold.
// Only points within the threshold of the seed can ever be added, and the
// max distance per candidate is maintained incrementally (max is exact, so
// results are bit-identical to recomputing it from scratch).
// Returns the cardinality (size) of the cluster; members are left in ws.members.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const SpatialGrid& grid,
                             const double threshold,
                             Workspace& ws) {
    auto& cand = ws.cand;
    auto& maxd = ws.maxd;
    auto& members = ws.members;
    cand.clear();
    members.clear();
    members.push_back(seed_point);

    const Point sp = points[seed_point];
    grid.forEachNear(sp, [&](int j) {
        if (j != seed_point && !clustered[j] && distance(points[j], sp) < threshold) {
            cand.push_back(j);
        }
    });
    if (cand.empty()) return 1;
    std::sort(cand.begin(), cand.end());
    const int m = static_cast<int>(cand.size());
    maxd.resize(m);
    for (int k = 0; k < m; ++k) {
        maxd[k] = distance(points[cand[k]], sp);
    }

    int count = m;
    while (count > 0) {
        // Select best candidate (strictly smaller wins -> lowest index on ties)
        int best = 0;
        double bestd = maxd[0];
        for (int k = 1; k < count; ++k) {
            if (maxd[k] < bestd) {
                bestd = maxd[k];
                best = k;
            }
        }
        const int added = cand[best];
        members.push_back(added);
        const Point ap = points[added];

        // Update remaining candidates with the new member, dropping those
        // that can no longer stay below the threshold (order preserved).
        int w = 0;
        for (int k = 0; k < count; ++k) {
            if (k == best) continue;
            const int c = cand[k];
            const double d = std::max(maxd[k], distance(points[c], ap));
            if (d < threshold) {
                cand[w] = c;
                maxd[w] = d;
                ++w;
            }
        }
        count = w;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel over candidate seeds)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;
    const SpatialGrid grid(points, threshold);
    Workspace ws;

    // Seeds owned by this rank (cyclic distribution: seed s lives on rank
    // s % nprocs at local slot s / nprocs).
    // A seed's candidate cluster depends only on the unclustered points within
    // the threshold of it, so a computed cardinality stays valid until one of
    // those points becomes clustered. Additionally, 1 + (number of unclustered
    // neighbours) is a cheap upper bound on the cardinality. Seeds are kept in
    // a lazy max-heap ordered by (value desc, seed asc); exact cardinalities
    // are only computed for seeds that could still be the best.
    const int n_owned = (N > rank) ? (N - rank + nprocs - 1) / nprocs : 0;
    std::vector<int> version(n_owned, 0);
    std::vector<char> dirty(n_owned, 0);
    std::vector<int> dirty_list;

    struct Entry { int val; int seed; int ver; bool exact; };
    auto worse = [](const Entry& a, const Entry& b) {
        return a.val < b.val || (a.val == b.val && a.seed > b.seed);
    };
    std::vector<Entry> heap;

    auto boundOf = [&](int s) {
        const Point sp = points[s];
        int cnt = 1;
        grid.forEachNear(sp, [&](int j) {
            if (j != s && !clustered[j] && distance(points[j], sp) < threshold) ++cnt;
        });
        return cnt;
    };

    heap.reserve(static_cast<size_t>(n_owned) * 2);
    for (int k = 0; k < n_owned; ++k) {
        const int s = rank + k * nprocs;
        heap.push_back({boundOf(s), s, 0, false});
    }
    std::make_heap(heap.begin(), heap.end(), worse);

    int remaining = N;
    while (remaining > 0) {
        // Refresh bounds of seeds whose neighbourhood changed
        for (const int k : dirty_list) {
            dirty[k] = 0;
            const int s = rank + k * nprocs;
            if (clustered[s]) continue;
            heap.push_back({boundOf(s), s, ++version[k], false});
            std::push_heap(heap.begin(), heap.end(), worse);
        }
        dirty_list.clear();

        // Find the local best (max cardinality, ties -> lowest seed index)
        struct { int card; int seed; } local{-1, N}, global;
        while (!heap.empty()) {
            const Entry top = heap.front();
            const int k = top.seed / nprocs;
            if (clustered[top.seed] || top.ver != version[k]) {
                std::pop_heap(heap.begin(), heap.end(), worse);
                heap.pop_back();
                continue;
            }
            if (top.exact) {
                local.card = top.val;
                local.seed = top.seed;
                break;
            }
            std::pop_heap(heap.begin(), heap.end(), worse);
            heap.pop_back();
            const int c = generateCandidateCluster(top.seed, clustered, points, grid,
                                                   threshold, ws);
            heap.push_back({c, top.seed, top.ver, true});
            std::push_heap(heap.begin(), heap.end(), worse);
        }

        // MAXLOC: max cardinality, ties resolved to the smallest seed index
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global.card <= 0 || global.seed >= N) break;

        // Every rank regenerates the winning cluster (deterministic, cheap)
        generateCandidateCluster(global.seed, clustered, points, grid, threshold, ws);
        Cluster cluster;
        cluster.seed_point = global.seed;
        cluster.members = ws.members;

        for (const int p : cluster.members) clustered[p] = 1;
        remaining -= static_cast<int>(cluster.members.size());

        // Invalidate owned seeds whose threshold-neighbourhood changed
        for (const int p : cluster.members) {
            const Point pp = points[p];
            grid.forEachNear(pp, [&](int s) {
                if (s % nprocs == rank && !clustered[s]) {
                    const int k = s / nprocs;
                    if (!dirty[k] && distance(points[s], pp) < threshold) {
                        dirty[k] = 1;
                        dirty_list.push_back(k);
                    }
                }
            });
        }

        // Compact the heap occasionally to drop stale entries
        if (heap.size() > static_cast<size_t>(2 * n_owned + 64)) {
            heap.erase(std::remove_if(heap.begin(), heap.end(), [&](const Entry& e) {
                           return clustered[e.seed] || e.ver != version[e.seed / nprocs];
                       }), heap.end());
            std::make_heap(heap.begin(), heap.end(), worse);
        }

        clusters.push_back(std::move(cluster));
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

int runBenchmark(int argc, char** argv) {
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 produces output; all ranks compute identical results
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) {
            fclose(stdout);
        }
    }

    const int ret = runBenchmark(argc, argv);

    fflush(stdout);
    MPI_Finalize();
    return ret;
}

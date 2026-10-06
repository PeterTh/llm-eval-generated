// QT Clustering Benchmark - OpenMP Parallel Version
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

#include <omp.h>

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
inline double distanceSq(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(distanceSq(p1, p2));
}

// Smallest double L such that sqrt(L) >= threshold. Since sqrt is monotone
// and correctly rounded, sqrt(d2) < threshold  <=>  d2 < L, and
// max_i sqrt(d2_i) == sqrt(max_i d2_i). This lets the clustering work on
// squared distances while making exactly the same decisions.
double squaredLimit(const double threshold) {
    double c = threshold * threshold;
    while (c > 0.0 && std::sqrt(c) >= threshold) c = std::nextafter(c, 0.0);
    while (std::sqrt(c) < threshold) c = std::nextafter(c, std::numeric_limits<double>::infinity());
    return c;
}

// Uniform spatial grid over the (bounded) point domain. Cell size is at least
// the threshold, so all points within `threshold` of a point lie in the
// 3x3 block of cells around it.
struct Grid {
    double cell;
    int nx, ny;
    std::vector<std::vector<int>> cells;

    Grid(const std::vector<Point>& points, const double threshold) {
        const int max_cells_per_dim = 1024;
        cell = std::max(threshold * (1.0 + 1e-9), std::max(MAX_WIDTH, MAX_HEIGHT) / max_cells_per_dim);
        nx = std::max(1, static_cast<int>(std::ceil(MAX_WIDTH / cell)));
        ny = std::max(1, static_cast<int>(std::ceil(MAX_HEIGHT / cell)));
        cells.resize(static_cast<size_t>(nx) * ny);
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            cells[cellOf(points[i])].push_back(i);
        }
    }
    int cx(double x) const { return std::clamp(static_cast<int>(x / cell), 0, nx - 1); }
    int cy(double y) const { return std::clamp(static_cast<int>(y / cell), 0, ny - 1); }
    size_t cellOf(const Point& p) const { return static_cast<size_t>(cy(p.y)) * nx + cx(p.x); }
};

// Per-thread scratch buffers for candidate cluster construction (SoA layout)
struct Workspace {
    std::vector<std::pair<int, double>> gather;
    std::vector<double> cx, cy, md2;
    std::vector<int> idx;
};

// Generate the candidate cluster starting from a seed point.
// Equivalent to repeatedly choosing the unclustered point that minimizes the
// maximum distance to all current members (ties -> lowest index), while that
// maximum stays below the threshold. Only unclustered points within the
// threshold of the seed can ever qualify. The per-candidate maximum (squared)
// distance is maintained incrementally; candidates that reach the limit can
// never become eligible again (the maximum only grows).
// Returns the cardinality of the cluster.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const Grid& grid,
                             const double limit_sq,
                             Workspace& ws,
                             std::vector<int>* cluster_members = nullptr) {
    const Point& sp = points[seed_point];
    const int gx = grid.cx(sp.x), gy = grid.cy(sp.y);
    ws.gather.clear();
    for (int y = std::max(0, gy - 1); y <= std::min(grid.ny - 1, gy + 1); ++y) {
        for (int x = std::max(0, gx - 1); x <= std::min(grid.nx - 1, gx + 1); ++x) {
            for (const int p : grid.cells[static_cast<size_t>(y) * grid.nx + x]) {
                if (p == seed_point || clustered[p]) continue;
                const double d2 = distanceSq(points[p], sp);
                if (d2 < limit_sq) ws.gather.emplace_back(p, d2);
            }
        }
    }
    // Candidates are kept in ascending index order for tie-breaking
    std::sort(ws.gather.begin(), ws.gather.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int n = static_cast<int>(ws.gather.size());
    ws.cx.resize(n); ws.cy.resize(n); ws.md2.resize(n); ws.idx.resize(n);
    double* __restrict cxs = ws.cx.data();
    double* __restrict cys = ws.cy.data();
    double* __restrict mds = ws.md2.data();
    int* __restrict ids = ws.idx.data();
    double mn = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const int p = ws.gather[i].first;
        ids[i] = p;
        cxs[i] = points[p].x;
        cys[i] = points[p].y;
        mds[i] = ws.gather[i].second;
        mn = std::min(mn, mds[i]);
    }

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }
    int count = 1;
    int dead = 0;
    const double inf = std::numeric_limits<double>::infinity();

    while (mn < limit_sq) {
        // All candidates whose distance (sqrt) equals sqrt(mn) are tied;
        // they form the interval [mn, upper]. Pick the lowest index.
        const double s = std::sqrt(mn);
        double upper = mn;
        for (;;) {
            const double nx = std::nextafter(upper, inf);
            if (std::sqrt(nx) != s) break;
            upper = nx;
        }
        int best = 0;
        while (mds[best] > upper) ++best;

        if (cluster_members) cluster_members->push_back(ids[best]);
        ++count;
        const double px = cxs[best], py = cys[best];
        mds[best] = inf;

        // Compact occasionally to drop candidates that can no longer qualify
        if (dead > n / 4) {
            int w = 0;
            for (int i = 0; i < n; ++i) {
                if (mds[i] < limit_sq) {
                    cxs[w] = cxs[i]; cys[w] = cys[i]; mds[w] = mds[i]; ids[w] = ids[i];
                    ++w;
                }
            }
            n = w;
        }

        // Update maximum distances and find the new minimum
        mn = inf;
        dead = 0;
        #pragma omp simd reduction(min:mn) reduction(+:dead)
        for (int i = 0; i < n; ++i) {
            const double dx = cxs[i] - px;
            const double dy = cys[i] - py;
            const double m = std::max(mds[i], dx * dx + dy * dy);
            mds[i] = m;
            mn = std::min(mn, m);
            dead += (m >= limit_sq);
        }
    }
    return count;
}

// Upper bound on the cardinality of a seed's candidate cluster:
// the seed plus all unclustered points within the threshold.
int cardinalityBound(const int seed_point,
                     const std::vector<Point>& points,
                     const Grid& grid,
                     const double limit_sq) {
    const Point& sp = points[seed_point];
    const int gx = grid.cx(sp.x), gy = grid.cy(sp.y);
    int count = 1;
    for (int y = std::max(0, gy - 1); y <= std::min(grid.ny - 1, gy + 1); ++y) {
        for (int x = std::max(0, gx - 1); x <= std::min(grid.nx - 1, gx + 1); ++x) {
            for (const int p : grid.cells[static_cast<size_t>(y) * grid.nx + x]) {
                if (p != seed_point && distanceSq(points[p], sp) < limit_sq) ++count;
            }
        }
    }
    return count;
}

// Main QT clustering algorithm
//
// The candidate cluster of a seed depends only on the clustered state of the
// points within `threshold` of the seed. Each unclustered seed keeps a value
// that is either its exact cardinality or an upper bound on it; seeds are kept
// in a max-heap ordered by (value desc, index asc). The best seed is the first
// heap entry that is exact. Bounds at the top of the heap are evaluated
// exactly in parallel batches; their member lists are cached.
// After a cluster is formed, a cached exact cluster stays valid unless one of
// its own members was clustered: removing a candidate that the greedy growth
// never selected cannot change any of its choices. Bounds are invalidated if a
// newly clustered point lies within `threshold` of the seed.
// This selects exactly the seed the exhaustive search would select.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    enum : char { BOUND = 0, EXACT = 1, PENDING = 2 };
    std::vector<char> clustered(N, 0);
    std::vector<char> dirty(N, 0);
    std::vector<char> state(N, BOUND);
    std::vector<int> value(N, 0);
    std::vector<std::vector<int>> seed_members(N);
    std::vector<Cluster> clusters;
    Grid grid(points, threshold);
    const double limit_sq = squaredLimit(threshold);

    struct Entry {
        int value, idx;
        char state;
        bool operator<(const Entry& o) const {
            return value < o.value || (value == o.value && idx > o.idx);
        }
    };
    std::vector<Entry> heap;
    heap.reserve(2 * static_cast<size_t>(N));
    auto push = [&heap](const Entry& e) {
        heap.push_back(e);
        std::push_heap(heap.begin(), heap.end());
    };
    auto pop = [&heap]() {
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
    };

    const int nthreads = omp_get_max_threads();
    // Speculative evaluation is rarely wasted, so large batches are used
    // for good load balance.
    const int batch_size = std::max(256, 8 * nthreads);
    std::vector<Workspace> workspaces(nthreads);
    std::vector<int> candidates;
    std::vector<int> batch;
    std::vector<int> best_members;
    int remaining = N;

    // Initial bounds for all seeds
    #pragma omp parallel for schedule(dynamic, 64)
    for (int s = 0; s < N; ++s) {
        value[s] = cardinalityBound(s, points, grid, limit_sq);
    }
    for (int s = 0; s < N; ++s) heap.push_back({value[s], s, BOUND});
    std::make_heap(heap.begin(), heap.end());

    while (remaining > 0) {
        // Find the best seed
        int best_seed = -1;
        while (!heap.empty()) {
            batch.clear();
            while (!heap.empty() && static_cast<int>(batch.size()) < batch_size) {
                const Entry top = heap.front();
                const int s = top.idx;
                if (clustered[s] || state[s] != top.state || value[s] != top.value) {
                    pop(); // stale entry
                    continue;
                }
                if (top.state == EXACT) {
                    if (batch.empty()) best_seed = s;
                    break;
                }
                pop();
                state[s] = PENDING;
                batch.push_back(s);
            }
            if (batch.empty()) break;

            const int nb = static_cast<int>(batch.size());
            #pragma omp parallel for schedule(dynamic, 1)
            for (int k = 0; k < nb; ++k) {
                const int s = batch[k];
                value[s] = generateCandidateCluster(s, clustered, points, grid, limit_sq,
                                                    workspaces[omp_get_thread_num()],
                                                    &seed_members[s]);
            }
            for (const int s : batch) {
                state[s] = EXACT;
                push({value[s], s, EXACT});
            }
        }
        if (best_seed < 0) break;
        pop();

        best_members = std::move(seed_members[best_seed]);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(std::move(cluster));

        for (const int m : best_members) {
            clustered[m] = 1;
            std::vector<int>().swap(seed_members[m]);
        }
        remaining -= static_cast<int>(best_members.size());
        if (remaining == 0) break;

        // Remove clustered points from affected grid cells
        for (const int m : best_members) {
            auto& c = grid.cells[grid.cellOf(points[m])];
            c.erase(std::remove_if(c.begin(), c.end(),
                                   [&clustered](int idx) { return clustered[idx] != 0; }),
                    c.end());
        }

        // Collect unclustered seeds that may be within threshold of a new member
        // (all members lie within threshold of the best seed).
        candidates.clear();
        const Point& bp = points[best_seed];
        const int r = static_cast<int>(std::ceil(2.0 * threshold / grid.cell)) + 1;
        const int gx = grid.cx(bp.x), gy = grid.cy(bp.y);
        for (int y = std::max(0, gy - r); y <= std::min(grid.ny - 1, gy + r); ++y) {
            for (int x = std::max(0, gx - r); x <= std::min(grid.nx - 1, gx + r); ++x) {
                for (const int p : grid.cells[static_cast<size_t>(y) * grid.nx + x]) {
                    candidates.push_back(p);
                }
            }
        }

        // Invalidate affected seeds and compute their new bounds
        const int nc = static_cast<int>(candidates.size());
        const int nm = static_cast<int>(best_members.size());
        #pragma omp parallel for schedule(dynamic, 16) if (nc * static_cast<long>(nm) > 20000)
        for (int k = 0; k < nc; ++k) {
            const int s = candidates[k];
            bool affected = false;
            if (state[s] == EXACT) {
                for (const int m : seed_members[s]) {
                    if (clustered[m]) { affected = true; break; }
                }
            } else {
                const Point& sp = points[s];
                for (int j = 0; j < nm; ++j) {
                    if (distanceSq(sp, points[best_members[j]]) < limit_sq) {
                        affected = true;
                        break;
                    }
                }
            }
            if (affected) {
                dirty[s] = 1;
                value[s] = cardinalityBound(s, points, grid, limit_sq);
                state[s] = BOUND;
                std::vector<int>().swap(seed_members[s]);
            }
        }
        for (const int s : candidates) {
            if (dirty[s]) {
                dirty[s] = 0;
                push({value[s], s, BOUND});
            }
        }
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
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
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

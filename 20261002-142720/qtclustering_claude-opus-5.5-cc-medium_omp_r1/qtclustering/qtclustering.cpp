// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <atomic>
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
inline double sqDistance(const double x1, const double y1, const double x2, const double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(sqDistance(p1.x, p1.y, p2.x, p2.y));
}

// Smallest squared distance s with sqrt(s) >= threshold, so that
// sqrt(s) < threshold  <=>  s < squaredLimit(threshold)  (sqrt is monotonic)
double squaredLimit(const double threshold) {
    double s = threshold * threshold;
    while (s > 0.0 && std::sqrt(s) >= threshold) s = std::nextafter(s, 0.0);
    while (std::sqrt(s) < threshold) s = std::nextafter(s, std::numeric_limits<double>::infinity());
    return s;
}

// Uniform grid over the point domain used to enumerate the neighbours of a
// point (all points within the distance threshold lie in the 3x3 cell block).
struct Grid {
    double cell;
    int nx, ny;
    std::vector<int> cell_start;   // size nx*ny+1
    std::vector<int> cell_points;  // point indices, ascending within each cell
    std::vector<int> cell_of;      // cell id of each point

    inline int cellX(double x) const {
        int c = static_cast<int>(x / cell);
        return std::clamp(c, 0, nx - 1);
    }
    inline int cellY(double y) const {
        int c = static_cast<int>(y / cell);
        return std::clamp(c, 0, ny - 1);
    }

    Grid(const std::vector<Point>& points, const double threshold) {
        const int MAX_CELLS_PER_DIM = 1024;
        const double min_cell = std::max(MAX_WIDTH, MAX_HEIGHT) / MAX_CELLS_PER_DIM;
        cell = (threshold >= min_cell) ? threshold : min_cell;  // also handles NaN
        nx = std::max(1, std::min(MAX_CELLS_PER_DIM, static_cast<int>(std::ceil(MAX_WIDTH / cell))));
        ny = std::max(1, std::min(MAX_CELLS_PER_DIM, static_cast<int>(std::ceil(MAX_HEIGHT / cell))));
        const int N = static_cast<int>(points.size());
        cell_start.assign(static_cast<size_t>(nx) * ny + 1, 0);
        cell_of.resize(N);
        std::vector<int>& cid = cell_of;
        for (int i = 0; i < N; ++i) {
            cid[i] = cellY(points[i].y) * nx + cellX(points[i].x);
            cell_start[cid[i] + 1]++;
        }
        for (size_t c = 0; c < static_cast<size_t>(nx) * ny; ++c) cell_start[c + 1] += cell_start[c];
        cell_points.resize(N);
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int i = 0; i < N; ++i) cell_points[fill[cid[i]]++] = i;
    }

    // Call f(c) for every cell id c in the 3x3 cell block around point i
    template <typename F>
    inline void forBlockCells(const int i, F&& f) const {
        const int cx = cell_of[i] % nx, cy = cell_of[i] / nx;
        const int x0 = std::max(0, cx - 1), x1 = std::min(nx - 1, cx + 1);
        const int y0 = std::max(0, cy - 1), y1 = std::min(ny - 1, cy + 1);
        for (int yy = y0; yy <= y1; ++yy)
            for (int xx = x0; xx <= x1; ++xx) f(yy * nx + xx);
    }

    // Call f(j) for every point j in the 3x3 cell block around point i
    template <typename F>
    inline void forNeighborhood(const int i, F&& f) const {
        const int cx = cell_of[i] % nx, cy = cell_of[i] / nx;
        const int x0 = std::max(0, cx - 1), x1 = std::min(nx - 1, cx + 1);
        const int y0 = std::max(0, cy - 1), y1 = std::min(ny - 1, cy + 1);
        for (int yy = y0; yy <= y1; ++yy) {
            const int rs = cell_start[yy * nx + x0];
            const int re = cell_start[yy * nx + x1 + 1];
            for (int k = rs; k < re; ++k) f(cell_points[k]);
        }
    }
};

// Per-thread scratch buffers for candidate cluster generation
struct Workspace {
    std::vector<int> idx;
    std::vector<double> px, py, maxd;
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
//
// Equivalent to the greedy procedure: repeatedly add the unclustered point
// with the smallest maximum distance to all current members (ties broken by
// lowest index) as long as that distance is below the threshold. The
// per-candidate maximum distance is maintained incrementally, and candidates
// whose maximum distance reached the threshold are discarded permanently
// (the maximum can only grow).
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const Grid& grid,
                             const double threshold,
                             Workspace& ws,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<int>& idx = ws.idx;
    idx.clear();
    const Point& sp = points[seed_point];
    grid.forNeighborhood(seed_point, [&](int j) {
        if (j == seed_point || clustered[j]) return;
        if (distance(points[j], sp) < threshold) idx.push_back(j);
    });
    std::sort(idx.begin(), idx.end());

    int n = static_cast<int>(idx.size());
    ws.px.resize(n);
    ws.py.resize(n);
    ws.maxd.resize(n);
    double* __restrict px = ws.px.data();
    double* __restrict py = ws.py.data();
    double* __restrict md = ws.maxd.data();
    int* __restrict ix = idx.data();

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    // Squared distances are tracked (max commutes with the monotonic sqrt).
    // A candidate is valid while s < sq_limit; since the maximum only grows,
    // invalid entries stay invalid and are removed by lazy compaction.
    const double sq_limit = squaredLimit(threshold);
    const double inf = std::numeric_limits<double>::infinity();

    double mn = inf;
    int live = 0;
    for (int k = 0; k < n; ++k) {
        const Point& c = points[ix[k]];
        px[k] = c.x;
        py[k] = c.y;
        const double v = sqDistance(c.x, c.y, sp.x, sp.y);
        md[k] = v;
        mn = (v < mn) ? v : mn;
        live += (v < sq_limit);
    }

    int size = 1;
    while (mn < sq_limit) {
        // Lazily drop invalid candidates (order preserved)
        if (2 * live < n) {
            int w = 0;
            for (int k = 0; k < n; ++k) {
                const double v = md[k];
                ix[w] = ix[k];
                px[w] = px[k];
                py[w] = py[k];
                md[w] = v;
                w += (v < sq_limit);
            }
            n = w;
        }

        // Select the lowest-index candidate whose distance sqrt(s) equals
        // the minimum distance sqrt(mn), i.e. s in [mn, s_hi]
        const double r = std::sqrt(mn);
        double s_hi = mn;
        for (;;) {
            const double next = std::nextafter(s_hi, inf);
            if (std::sqrt(next) != r) break;
            s_hi = next;
        }
        int best = 0;
        while (md[best] > s_hi) ++best;

        // Add the selected candidate to the cluster
        const double nx = px[best], ny = py[best];
        if (cluster_members) cluster_members->push_back(ix[best]);
        ++size;
        md[best] = inf;

        // Update maxima with the distance to the new member and find the
        // minimum / number of valid candidates for the next selection
        mn = inf;
        live = 0;
        #pragma omp simd reduction(min : mn) reduction(+ : live)
        for (int k = 0; k < n; ++k) {
            const double m = md[k];
            const double d = sqDistance(px[k], py[k], nx, ny);
            const double v = (m < d) ? d : m;
            md[k] = v;
            mn = (v < mn) ? v : mn;
            live += (v < sq_limit);
        }
    }
    return size;
}

// Main QT clustering algorithm
//
// Candidate cluster cardinalities are cached per seed: a seed's candidate
// cluster depends only on the clustered status of points within the threshold
// distance of the seed, so it only needs recomputation when such a point gets
// clustered. Dirty seeds are evaluated in parallel. Member lists of candidate
// clusters are cached as well (within a memory budget) so that the selected
// cluster does not have to be regenerated.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    const Grid grid(points, threshold);

    std::vector<int> cardinality(N, 0);
    std::vector<char> dirty(N, 1);
    std::vector<std::pair<long long, int>> dirty_list;  // (estimated cost, seed)
    dirty_list.reserve(N);

    // Unclustered point count per grid cell (for work estimation)
    std::vector<int> cell_count(grid.cell_start.size() - 1);
    for (size_t c = 0; c < cell_count.size(); ++c)
        cell_count[c] = grid.cell_start[c + 1] - grid.cell_start[c];

    // Cached member lists, bounded by a total budget of stored indices
    const long long MEMBER_CACHE_BUDGET = 1LL << 26;
    std::vector<std::vector<int>> cached_members(N);
    std::vector<char> has_members(N, 0);
    std::atomic<long long> cached_total{0};

    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    const int nthreads = omp_get_max_threads();
    std::vector<Workspace> workspaces(nthreads);
    std::vector<std::vector<int>> thread_members(nthreads);

    // Approximate cost units handled per thread before adding another thread
    const long long WORK_PER_THREAD = 1LL << 16;
    int team = -1;

    std::vector<int> cell_stamp(cell_count.size(), -1);
    std::vector<int> touched_cells;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        dirty_list.clear();
        long long total_work = 0;
        for (int s : unclustered_indices) {
            if (!dirty[s]) continue;
            dirty[s] = 0;
            long long nb = 0;
            grid.forBlockCells(s, [&](int c) { nb += cell_count[c]; });
            const long long w = nb * nb + 16;
            total_work += w;
            dirty_list.emplace_back(w, s);
        }
        // Most expensive seeds first for better dynamic load balance
        std::sort(dirty_list.begin(), dirty_list.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first || (a.first == b.first && a.second < b.second);
                  });
        const int ndirty = static_cast<int>(dirty_list.size());
        const int nunc = static_cast<int>(unclustered_indices.size());

        if (ndirty > 0) {
            // Fixed team size (chosen from the initial workload) avoids costly
            // team resizing; iterations with little work run sequentially.
            if (team < 0) {
                team = static_cast<int>(std::clamp<long long>(
                    total_work / WORK_PER_THREAD, 1, std::min(nthreads, ndirty)));
            }
            #pragma omp parallel num_threads(team) if (total_work >= WORK_PER_THREAD && ndirty > 1)
            {
                const int tid = omp_get_thread_num();
                Workspace& ws = workspaces[tid];
                std::vector<int>& members = thread_members[tid];
                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < ndirty; ++i) {
                    const int seed = dirty_list[i].second;
                    const int card = generateCandidateCluster(seed, clustered, points, grid,
                                                              threshold, ws, &members);
                    cardinality[seed] = card;
                    // Cache the member list if within the (soft) memory budget
                    std::vector<int>& cm = cached_members[seed];
                    const long long old = has_members[seed] ? static_cast<long long>(cm.size()) : 0;
                    if (cached_total.load(std::memory_order_relaxed) - old + card <= MEMBER_CACHE_BUDGET) {
                        cm.assign(members.begin(), members.end());
                        has_members[seed] = 1;
                        cached_total.fetch_add(card - old, std::memory_order_relaxed);
                    } else if (has_members[seed]) {
                        std::vector<int>().swap(cm);
                        has_members[seed] = 0;
                        cached_total.fetch_sub(old, std::memory_order_relaxed);
                    }
                }
            }
        }

        // Argmax over seeds: largest cardinality, lowest seed index on ties
        int max_cardinality = -1;
        int best_seed = -1;
        #pragma omp parallel num_threads(std::max(team, 1)) if (nunc >= 65536)
        {
            int lc = -1, ls = -1;
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < nunc; ++i) {
                const int s = unclustered_indices[i];
                if (cardinality[s] > lc) { lc = cardinality[s]; ls = s; }
            }
            #pragma omp critical
            {
                if (lc > max_cardinality || (lc == max_cardinality && ls >= 0 && ls < best_seed)) {
                    max_cardinality = lc;
                    best_seed = ls;
                }
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            if (has_members[best_seed]) {
                cluster.members = cached_members[best_seed];
            } else {
                generateCandidateCluster(best_seed, clustered, points, grid, threshold,
                                         workspaces[0], &cluster.members);
            }

            // Mark all members as clustered and invalidate affected seeds
            for (const int m : cluster.members) {
                clustered[m] = 1;
                cell_count[grid.cell_of[m]]--;
                if (has_members[m]) {
                    cached_total -= static_cast<long long>(cached_members[m].size());
                    std::vector<int>().swap(cached_members[m]);
                    has_members[m] = 0;
                }
            }
            // (union of the 3x3 cell blocks around all members)
            const int stamp = static_cast<int>(clusters.size());
            touched_cells.clear();
            for (const int m : cluster.members) {
                grid.forBlockCells(m, [&](int c) {
                    if (cell_stamp[c] != stamp) { cell_stamp[c] = stamp; touched_cells.push_back(c); }
                });
            }
            for (const int c : touched_cells) {
                for (int k = grid.cell_start[c]; k < grid.cell_start[c + 1]; ++k) {
                    const int j = grid.cell_points[k];
                    if (!clustered[j]) dirty[j] = 1;
                }
            }
            clusters.push_back(std::move(cluster));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
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

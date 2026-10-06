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
#include <set>
#include <utility>
#include <vector>

#include <mpi.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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

// Squared Euclidean distance. The FMA form matches the contraction the
// compiler applies to the expression dx * dx + dy * dy on FMA targets.
inline double sqDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __FMA__
    return std::fma(dx, dx, dy * dy);
#else
    return dx * dx + dy * dy;
#endif
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(sqDistance(p1, p2));
}

// Smallest x with sqrt(x) >= threshold, so that for any squared distance x:
// sqrt(x) < threshold  <=>  x < sqCutoff(threshold)  (exact).
double sqCutoff(const double threshold) {
    double x = threshold * threshold;
    while (x > 0 && std::sqrt(x) >= threshold) x = std::nextafter(x, 0.0);
    while (std::sqrt(x) < threshold) x = std::nextafter(x, std::numeric_limits<double>::infinity());
    return x;
}

// Uniform grid for neighbor (distance < threshold) queries over the points
// with index i where i % stride == offset. Clustered points can be
// removed from their cells.
struct Grid {
    double cell;
    double t2_cut;  // squared distance cutoff equivalent to "< threshold"
    int nx, ny;
    std::vector<int> cell_start, cell_end;  // CSR layout, live range per cell
    std::vector<int> cell_items;
    std::vector<Point> cell_pts;

    int cx(double x) const { return std::clamp(static_cast<int>(x / cell), 0, nx - 1); }
    int cy(double y) const { return std::clamp(static_cast<int>(y / cell), 0, ny - 1); }
    int cellOf(const Point& p) const { return cy(p.y) * nx + cx(p.x); }

    Grid(const std::vector<Point>& points, double threshold, int stride = 1, int offset = 0) {
        const int MAX_CELLS = 2048;
        t2_cut = sqCutoff(threshold);
        cell = std::max(threshold * 1.0001, std::max(MAX_WIDTH, MAX_HEIGHT) / MAX_CELLS);
        nx = std::max(1, static_cast<int>(MAX_WIDTH / cell) + 1);
        ny = std::max(1, static_cast<int>(MAX_HEIGHT / cell) + 1);
        const int N = static_cast<int>(points.size());
        const size_t ncells = static_cast<size_t>(nx) * ny;
        cell_start.assign(ncells + 1, 0);
        for (int i = offset; i < N; i += stride) cell_start[cellOf(points[i]) + 1]++;
        for (size_t c = 0; c < ncells; ++c) cell_start[c + 1] += cell_start[c];
        cell_items.resize(cell_start[ncells]);
        cell_pts.resize(cell_start[ncells]);
        cell_end.assign(cell_start.begin(), cell_start.end() - 1);
        for (int i = offset; i < N; i += stride) {  // ascending index per cell
            const int k = cell_end[cellOf(points[i])]++;
            cell_items[k] = i;
            cell_pts[k] = points[i];
        }
    }

    // Calls f(j) for every live point j with distance(p, points[j]) < threshold
    // (includes p's own index if it is a live point of the grid).
    template <typename F>
    void forNeighbors(const Point& p, F&& f) const {
        const int x0 = std::max(cx(p.x) - 1, 0), x1 = std::min(cx(p.x) + 1, nx - 1);
        const int y0 = std::max(cy(p.y) - 1, 0), y1 = std::min(cy(p.y) + 1, ny - 1);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const int c = y * nx + x;
                for (int k = cell_start[c]; k < cell_end[c]; ++k) {
                    if (sqDistance(cell_pts[k], p) < t2_cut) f(cell_items[k]);
                }
            }
        }
    }

    // Remove clustered points from cell c (order preserving)
    void compactCell(const int c, const std::vector<char>& clustered) {
        int w = cell_start[c];
        for (int k = cell_start[c]; k < cell_end[c]; ++k) {
            if (!clustered[cell_items[k]]) {
                cell_items[w] = cell_items[k];
                cell_pts[w] = cell_pts[k];
                ++w;
            }
        }
        cell_end[c] = w;
    }
};

// Generate a candidate cluster starting from a seed point.
// Equivalent to iteratively adding the unclustered point with the smallest
// maximum distance to all current members (ties -> smallest index), as long
// as that maximum distance is below the threshold. Only points within the
// threshold of the seed can ever qualify, and max distances are maintained
// incrementally (max is order independent, so values are bit-identical).
// For every candidate i: md[i] = max(md[i], sqDistance(candidate i, p)).
// Returns the minimum of the updated values; live counts values below cut.
static inline double updateMaxDistances(double* __restrict md,
                                        const double* __restrict xs,
                                        const double* __restrict ys,
                                        const size_t n, const Point p,
                                        const double cut, long& live) {
    double mn = std::numeric_limits<double>::infinity();
    long cnt = 0;
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256d vpx = _mm256_set1_pd(p.x), vpy = _mm256_set1_pd(p.y);
    const __m256d vcut = _mm256_set1_pd(cut);
    __m256d vmn = _mm256_set1_pd(mn);
    for (; i + 4 <= n; i += 4) {
        const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(xs + i), vpx);
        const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ys + i), vpy);
        const __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
        const __m256d d = _mm256_max_pd(_mm256_loadu_pd(md + i), d2);
        _mm256_storeu_pd(md + i, d);
        vmn = _mm256_min_pd(vmn, d);
        cnt += __builtin_popcount(_mm256_movemask_pd(_mm256_cmp_pd(d, vcut, _CMP_LT_OQ)));
    }
    alignas(32) double t[4];
    _mm256_store_pd(t, vmn);
    mn = std::min(std::min(t[0], t[1]), std::min(t[2], t[3]));
#endif
    for (; i < n; ++i) {
        const double d = std::max(md[i], sqDistance(Point{xs[i], ys[i]}, p));
        md[i] = d;
        mn = std::min(mn, d);
        cnt += (d < cut) ? 1 : 0;
    }
    live = cnt;
    return mn;
}

// Index of the first entry with md[i] <= lim (one must exist)
static inline size_t findFirstAtMost(const double* md, const size_t n, const double lim) {
    size_t i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256d vl = _mm256_set1_pd(lim);
    for (; i + 4 <= n; i += 4) {
        const int m = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(md + i), vl, _CMP_LE_OQ));
        if (m) return i + static_cast<size_t>(__builtin_ctz(m));
    }
#endif
    while (!(md[i] <= lim)) ++i;
    return i;
}

struct CandidateBuilder {
    std::vector<int> cand;
    std::vector<double> cxs, cys, md2;
    double t2_cut;  // sqrt(x) < threshold  <=>  x < t2_cut

    explicit CandidateBuilder(const double threshold) : t2_cut(sqCutoff(threshold)) {}

    // The grid must only contain unclustered points.
    int build(const int seed_point,
              const std::vector<Point>& points,
              const Grid& grid,
              std::vector<int>* cluster_members = nullptr) {
        constexpr double INF = std::numeric_limits<double>::infinity();
        cand.clear();
        grid.forNeighbors(points[seed_point], [&](int j) {
            if (j != seed_point) cand.push_back(j);
        });
        std::sort(cand.begin(), cand.end());
        size_t n = cand.size();
        cxs.resize(n);
        cys.resize(n);
        md2.resize(n);
        int* __restrict ci = cand.data();
        double* __restrict xs = cxs.data();
        double* __restrict ys = cys.data();
        double* __restrict md = md2.data();
        const Point ps = points[seed_point];
        double mn = INF;
        for (size_t i = 0; i < n; ++i) {
            const Point pc = points[ci[i]];
            xs[i] = pc.x;
            ys[i] = pc.y;
            md[i] = sqDistance(pc, ps);
            mn = std::min(mn, md[i]);
        }
        if (cluster_members) {
            cluster_members->clear();
            cluster_members->push_back(seed_point);
        }
        int count = 1;
        size_t alive = n;
        // Squared max distances are tracked; sqrt is monotone so the max is
        // preserved exactly. Selection reproduces the original's comparison of
        // sqrt values (first index among equal sqrt minima).
        while (mn < t2_cut) {
            const double s = std::sqrt(mn);
            double hi = mn;
            for (;;) {
                const double nx = std::nextafter(hi, INF);
                if (std::sqrt(nx) != s) break;
                hi = nx;
            }
            const size_t b = findFirstAtMost(md, n, hi);

            ++count;
            if (cluster_members) cluster_members->push_back(ci[b]);
            const Point pb{xs[b], ys[b]};
            md[b] = INF;
            --alive;

            // Update squared max distances, find min and count live entries
            long live = 0;
            mn = updateMaxDistances(md, xs, ys, n, pb, t2_cut, live);
            alive = static_cast<size_t>(live);
            // Periodically compact to drop disqualified candidates
            if (alive * 4 < n * 3) {
                size_t w = 0;
                for (size_t i = 0; i < n; ++i) {
                    ci[w] = ci[i]; xs[w] = xs[i]; ys[w] = ys[i]; md[w] = md[i];
                    w += (md[i] < t2_cut);
                }
                n = w;
            }
        }
        return count;
    }
};

// Main QT clustering algorithm (MPI-parallel over seed points).
// Seeds are distributed cyclically across ranks. Each rank caches the
// candidate cluster of its seeds. When a cluster is committed, a cached
// candidate stays exact unless one of its own members got clustered:
// removing points that were never selected does not change any greedy
// choice. Stale seeds are lazily recomputed in order of an upper bound
// (number of unclustered points within the threshold) so that seeds that
// cannot beat the current best are never rebuilt.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;
    Grid grid(points, threshold);                        // all unclustered points
    Grid owned_grid(points, threshold, nprocs, rank);    // unclustered owned seeds
    std::vector<int> cell_stamp(grid.cell_end.size(), -1);
    CandidateBuilder builder(threshold);

    // Per seed state (only entries of owned seeds are used)
    std::vector<int> card(N, 0);    // cached cardinality (valid if clean)
    std::vector<int> ub(N, 0);      // unclustered neighbors incl. self
    std::vector<int> old_ub(N, 0);
    std::vector<char> dirty(N, 0);
    std::vector<char> has_members(N, 0);
    std::vector<std::vector<int>> cached(N);
    std::vector<int> touch_stamp(N, -1), clustered_round(N, -1);
    auto owned = [&](int i) { return i % nprocs == rank; };

    // Memory budget (ints per rank) for cached candidate member lists;
    // seeds without a cached list fall back to invalidation on any change
    // in their neighborhood.
    const long MEMBER_BUDGET = 64L * 1024 * 1024;
    long stored = 0;

    // Ordered by (value desc, index asc)
    using Key = std::pair<int, int>;  // (-value, index)
    std::set<Key> clean_set, dirty_set;

    for (int i = rank; i < N; i += nprocs) {
        int cnt = 0;
        grid.forNeighbors(points[i], [&](int) { ++cnt; });
        ub[i] = cnt;
        dirty[i] = 1;
        dirty_set.insert({-cnt, i});
    }

    auto dropCache = [&](int s) {
        if (has_members[s]) {
            stored -= static_cast<long>(cached[s].capacity());
            std::vector<int>().swap(cached[s]);
            has_members[s] = 0;
        }
    };

    int remaining = N;
    int round = 0;
    std::vector<int> members, touched;
    while (remaining > 0) {
        int best_card = -1, best_seed = std::numeric_limits<int>::max();
        if (!clean_set.empty()) {
            best_card = -clean_set.begin()->first;
            best_seed = clean_set.begin()->second;
        }
        // Lazily evaluate dirty seeds that could still beat the local best
        while (!dirty_set.empty()) {
            const Key k = *dirty_set.begin();
            const int bound = -k.first, s = k.second;
            if (bound < best_card || (bound == best_card && s > best_seed)) break;
            dirty_set.erase(dirty_set.begin());
            int c;
            if (stored < MEMBER_BUDGET) {
                c = builder.build(s, points, grid, &cached[s]);
                cached[s].shrink_to_fit();
                stored += static_cast<long>(cached[s].capacity());
                has_members[s] = 1;
            } else {
                c = builder.build(s, points, grid);
            }
            card[s] = c;
            dirty[s] = 0;
            clean_set.insert({-c, s});
            if (c > best_card || (c == best_card && s < best_seed)) {
                best_card = c;
                best_seed = s;
            }
        }

        // Global best: max cardinality, ties -> smallest seed index
        struct { int val; int idx; } in{best_card, best_seed}, out;
        MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (out.val <= 0) break;
        const int seed = out.idx;

        // The owner of the winning seed broadcasts its cluster members
        const int owner = seed % nprocs;
        int msize = 0;
        if (rank == owner) {
            if (has_members[seed]) members = cached[seed];
            else builder.build(seed, points, grid, &members);
            msize = static_cast<int>(members.size());
        }
        MPI_Bcast(&msize, 1, MPI_INT, owner, MPI_COMM_WORLD);
        members.resize(msize);
        MPI_Bcast(members.data(), msize, MPI_INT, owner, MPI_COMM_WORLD);

        for (int m : members) {
            clustered[m] = 1;
            clustered_round[m] = round;
            if (owned(m)) {
                if (dirty[m]) dirty_set.erase({-ub[m], m});
                else clean_set.erase({-card[m], m});
                dropCache(m);
            }
        }
        remaining -= msize;

        // Remove clustered points from the grids
        for (int m : members) {
            const int c = grid.cellOf(points[m]);
            if (cell_stamp[c] != round) {
                cell_stamp[c] = round;
                grid.compactCell(c, clustered);
                owned_grid.compactCell(c, clustered);
            }
        }

        // Update neighbor counts of affected (unclustered) owned seeds
        touched.clear();
        for (int m : members) {
            owned_grid.forNeighbors(points[m], [&](int j) {
                if (touch_stamp[j] != round) {
                    touch_stamp[j] = round;
                    old_ub[j] = ub[j];
                    touched.push_back(j);
                }
                --ub[j];
            });
        }
        // Re-key / invalidate affected seeds
        for (int j : touched) {
            if (dirty[j]) {
                dirty_set.erase({-old_ub[j], j});
                dirty_set.insert({-ub[j], j});
                continue;
            }
            bool stale = true;
            if (has_members[j]) {
                stale = false;
                for (int q : cached[j]) {
                    if (clustered_round[q] == round) { stale = true; break; }
                }
            }
            if (stale) {
                clean_set.erase({-card[j], j});
                dropCache(j);
                dirty[j] = 1;
                dirty_set.insert({-ub[j], j});
            }
        }

        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members = members;
        clusters.push_back(std::move(cluster));
        ++round;
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

static int runBenchmark(int argc, char** argv, const int rank) {
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
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    const long local_cluster_time_ms = cluster_time.count();
    long max_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &max_cluster_time_ms, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) cluster_time = std::chrono::milliseconds(max_cluster_time_ms);
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
    if (validate && rank == 0) {
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
    // Only rank 0 produces output
    if (rank != 0 && !freopen("/dev/null", "w", stdout)) {
        fclose(stdout);
    }
    const int ret = runBenchmark(argc, argv, rank);
    fflush(stdout);
    MPI_Finalize();
    return ret;
}

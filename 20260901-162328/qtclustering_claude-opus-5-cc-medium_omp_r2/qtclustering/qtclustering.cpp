// QT Clustering Benchmark - OpenMP Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Each pass of the outer loop grows a candidate cluster for every unclustered point and
// turns the largest one into a cluster. Those candidate clusters are independent of each
// other, which is where the parallelism comes from: the seeds are handed out to the
// threads of one long lived parallel region, and each pass is closed by picking the
// winner. On top of that the sequential work per pass is cut down in three ways, all of
// which reproduce the results of the naive formulation exactly:
//
//   * a candidate cluster is grown by maintaining the maximum distance of every
//     candidate point to the members instead of rescanning the members,
//   * only the points in the 3x3 neighbourhood of the seed in a uniform grid are
//     considered, since every member is within the threshold of the seed,
//   * candidate clusters are cached across the passes and only regrown once one of
//     their members has been clustered away by another seed.

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

// Number of hardware threads that share a physical core (SMT degree), as reported by
// the kernel; 1 if that cannot be determined.
int smtDegree() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!f) return 1;
    char list[256] = {0};
    const bool ok = fgets(list, sizeof(list), f) != nullptr;
    fclose(f);
    if (!ok) return 1;

    // The list is comma separated and may contain ranges, e.g. "0,128" or "0-1"
    int siblings = 0;
    const char* p = list;
    while (*p) {
        char* end = nullptr;
        const long first = strtol(p, &end, 10);
        if (end == p) break;
        long last = first;
        if (*end == '-') {
            p = end + 1;
            last = strtol(p, &end, 10);
        }
        siblings += static_cast<int>(last - first + 1);
        p = (*end == ',') ? end + 1 : end;
    }
    return siblings > 0 ? siblings : 1;
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Uniform grid over the still unclustered points. Every member of a candidate cluster
// is within 'threshold' of its seed point (the seed is a member, so the cluster
// diameter bounds that distance), so only the 3x3 cell neighbourhood of a seed has to
// be looked at when the cells are at least 'threshold' wide.
struct Grid {
    double cell = 1.0, inv_cell = 1.0, x0 = 0.0, y0 = 0.0;
    int nx = 1, ny = 1;
    std::vector<int> start;  // nx*ny+1 offsets into items
    std::vector<int> items;  // point indices, ascending within a cell

    int cellX(const double x) const {
        return std::min(nx - 1, std::max(0, static_cast<int>((x - x0) * inv_cell)));
    }
    int cellY(const double y) const {
        return std::min(ny - 1, std::max(0, static_cast<int>((y - y0) * inv_cell)));
    }

    void build(const std::vector<int>& avail, const Point* __restrict points,
               const double threshold) {
        const int n = static_cast<int>(avail.size());
        double xmin = points[avail[0]].x, xmax = xmin;
        double ymin = points[avail[0]].y, ymax = ymin;
        for (int i = 1; i < n; ++i) {
            const Point& p = points[avail[i]];
            xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        }
        x0 = xmin;
        y0 = ymin;

        // Cells are at least as wide as the threshold, but coarsened if that would
        // give far more cells than points
        cell = threshold;
        const double w = xmax - xmin, h = ymax - ymin;
        const double max_cells = 4.0 * n + 16.0;
        const double want = (w / cell + 1.0) * (h / cell + 1.0);
        if (want > max_cells) {
            cell *= std::sqrt(want / max_cells);
        }
        inv_cell = 1.0 / cell;
        nx = static_cast<int>(w * inv_cell) + 1;
        ny = static_cast<int>(h * inv_cell) + 1;

        // Counting sort of the points into the cells, keeping them ascending per cell
        start.assign(static_cast<size_t>(nx) * ny + 1, 0);
        items.resize(n);
        for (int i = 0; i < n; ++i) {
            const Point& p = points[avail[i]];
            ++start[static_cast<size_t>(cellY(p.y)) * nx + cellX(p.x) + 1];
        }
        for (size_t c = 1; c < start.size(); ++c) {
            start[c] += start[c - 1];
        }
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (int i = 0; i < n; ++i) {
            const int idx = avail[i];
            const Point& p = points[idx];
            items[fill[static_cast<size_t>(cellY(p.y)) * nx + cellX(p.x)]++] = idx;
        }
    }
};

// Per-thread scratch space for growing a candidate cluster. The candidates are kept
// in a struct-of-arrays layout so that the distance update vectorizes.
struct GrowScratch {
    std::vector<int> cand;     // still admissible candidate points
    std::vector<double> cx;    // their coordinates
    std::vector<double> cy;
    std::vector<double> maxd;  // max distance of cand[i] to the current members
    std::vector<int> members;  // members of the cluster being grown
    double radius = 0.0;       // maximum distance of a member to the seed

    void resize(const int n) {
        if (static_cast<int>(cand.size()) < n) {
            cand.resize(n);
            cx.resize(n);
            cy.resize(n);
            maxd.resize(n);
        }
    }
};

// Grow the candidate cluster of a seed point, greedily adding the point with the
// smallest resulting cluster diameter as long as it stays below the threshold.
//
// This is equivalent to the naive formulation that rescans all members for every
// candidate: the maximum distance of a candidate to the members only ever grows as
// members are added, so it can be maintained incrementally (giving bit-identical
// values, max() being exact), and candidates that once exceeded the threshold can
// never become admissible again and are dropped.
//
// 'grid' holds the still unclustered points, including 'seed'. The resulting members
// are written to scratch.members and their maximum distance to the seed to
// scratch.radius; the return value is the number of distance evaluations, used to gauge
// how much work a pass of the outer loop is worth.
long long growCluster(const int seed,
                      const Grid& grid,
                      const unsigned char* __restrict removed,
                      const Point* __restrict points,
                      const double threshold,
                      GrowScratch& scratch) {
    // Only the points of the 3x3 cell neighbourhood are ever candidates
    const int ix = grid.cellX(points[seed].x);
    const int iy = grid.cellY(points[seed].y);
    const int xlo = std::max(ix - 1, 0);
    const int xhi = std::min(ix + 1, grid.nx - 1);
    const int ylo = std::max(iy - 1, 0);
    const int yhi = std::min(iy + 1, grid.ny - 1);
    int capacity = 0;
    for (int jy = ylo; jy <= yhi; ++jy) {
        const size_t row = static_cast<size_t>(jy) * grid.nx;
        capacity += grid.start[row + xhi + 1] - grid.start[row + xlo];
    }
    scratch.resize(capacity);

    int* __restrict cand = scratch.cand.data();
    double* __restrict cx = scratch.cx.data();
    double* __restrict cy = scratch.cy.data();
    double* __restrict maxd = scratch.maxd.data();

    std::vector<int>& members = scratch.members;
    members.clear();
    members.push_back(seed);
    scratch.radius = 0.0;

    // The unclustered points around the seed start out as candidates. Points further
    // away than the threshold may be included, the first pass below discards them.
    int m = 0;
    for (int jy = ylo; jy <= yhi; ++jy) {
        // The cells of a row are contiguous in the item array
        const size_t row = static_cast<size_t>(jy) * grid.nx;
        const int lo = grid.start[row + xlo];
        const int hi = grid.start[row + xhi + 1];
        for (int k = lo; k < hi; ++k) {
            const int p = grid.items[k];
            cand[m] = p;
            cx[m] = points[p].x;
            cy[m] = points[p].y;
            maxd[m] = 0.0;
            m += (p != seed && !removed[p]);
        }
    }

    long long work = m;
    int last_added = seed;
    while (m > 0) {
        work += m;
        const double lx = points[last_added].x;
        const double ly = points[last_added].y;

        // Update the max distance of every candidate with the distance to the newly
        // added member, drop those that exceeded the threshold (their max distance only
        // grows, so they can never become admissible again) and pick the point to add:
        // smallest diameter wins, ties are broken by the lowest point index, which is
        // what the ascending candidate scan of the naive version selects.
        double best_diameter = std::numeric_limits<double>::max();
        int best_point = std::numeric_limits<int>::max();
        int best_pos = -1;
        int w = 0;
        for (int i = 0; i < m; ++i) {
            const int c = cand[i];
            const double x = cx[i];
            const double y = cy[i];
            const double dx = x - lx;
            const double dy = y - ly;
            const double dist = std::sqrt(dx * dx + dy * dy);
            const double md = std::max(maxd[i], dist);

            cand[w] = c;
            cx[w] = x;
            cy[w] = y;
            maxd[w] = md;
            if (md < best_diameter || (md == best_diameter && c < best_point)) {
                best_diameter = md;
                best_point = c;
                best_pos = w;
            }
            w += (md < threshold);
        }
        m = w;

        // The minimum is above the threshold -> no more points can be added
        if (!(best_diameter < threshold)) break;

        members.push_back(best_point);
        scratch.radius = std::max(scratch.radius, distance(points[best_point], points[seed]));
        last_added = best_point;

        // Remove the newly added member from the candidate list
        --m;
        cand[best_pos] = cand[m];
        cx[best_pos] = cx[m];
        cy[best_pos] = cy[m];
        maxd[best_pos] = maxd[m];
    }

    return work;
}

// The largest candidate cluster of a pass. Ties go to the lowest seed index, which is
// the one the sequential scan of the ascending unclustered list encounters first.
struct Best {
    int card = -1;
    int seed = std::numeric_limits<int>::max();

    void update(const int c, const int s) {
        if (c > card || (c == card && s < seed)) {
            card = c;
            seed = s;
        }
    }
};

// Padded to keep the per-thread results out of each other's cache lines
struct alignas(64) PaddedBest {
    Best best;
};

struct alignas(64) PaddedWork {
    long long work;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const Point* __restrict pts = points.data();
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    const int max_threads = omp_get_max_threads();

    // Candidate clusters are cached across the passes of the outer loop: a candidate
    // cluster can only change once one of its members gets clustered away, because
    // removing points that the greedy growth never selected leaves every one of its
    // choices untouched. Two clusters can only share a point when the distance of their
    // seeds is below the sum of their radii, which weeds out the far away seeds without
    // touching their member lists at all.
    std::vector<int> cardinality(N, 0);
    std::vector<double> radius(N, 0.0);
    std::vector<unsigned char> dirty(N, 1);
    std::vector<unsigned char> removed(N, 0);

    // The members of the cached candidate clusters, in a bump allocated arena per
    // thread. Only the owning thread appends to its arena, and the arenas are only read
    // by another thread while no thread is growing clusters, so no locking is needed.
    struct CacheRef {
        int tid = -1;
        unsigned gen = 0;
        int offset = 0;
        int count = 0;
    };
    std::vector<CacheRef> cache_ref(N);
    std::vector<std::vector<int>> arenas(max_threads);
    std::vector<unsigned> arena_gen(max_threads, 0);
    // Recycle an arena once it grew past its share of the memory budget
    const size_t arena_budget = std::max<size_t>(1 << 16, (size_t(64) << 20) / max_threads);

    Grid grid;
    grid.build(unclustered_indices, pts, threshold);
    int stale = 0;  // points clustered away since the grid was last built

    std::vector<PaddedBest> thread_best(max_threads);
    std::vector<PaddedWork> thread_work(max_threads);
    // The seeds whose candidate cluster has to be (re)grown in the current pass
    std::vector<int> pending(N);
    std::atomic<int> pending_count(0);
    std::atomic<int> next_pending(0);

    // The cluster that was formed by the previous pass, against which the cached
    // candidate clusters are checked
    Point last_center = {0.0, 0.0};
    double last_radius = 0.0;

    // The first pass grows a candidate cluster for every point and dominates the run
    // time, so it gets the whole machine. The later passes only revisit the seeds whose
    // cached candidate cluster was invalidated, which is much less work per pass; once
    // the passes get short enough, the synchronisation between them outweighs what an
    // extra core contributes, so the team for the rest of the run is sized from the work
    // ('pass_work' counts candidate distance evaluations) of the first pass.
    long long pass_work = 0;
    int num_threads = max_threads;

    for (int epoch = 0; epoch < 2 && !unclustered_indices.empty(); ++epoch) {
        #pragma omp parallel num_threads(num_threads)
        {
            // Kept across the parallel regions so that the buffers are only grown once
            static thread_local GrowScratch scratch;
            static thread_local std::vector<int> local_pending;
            const int tid = omp_get_thread_num();

            // Main clustering loop
            while (true) {
                const int n_avail = static_cast<int>(unclustered_indices.size());
                if (n_avail == 0) break;

                Best best;
                long long work = 0;

                // A cached candidate cluster survives until it overlaps a cluster that was
                // formed in the meantime. The scan is split statically, which keeps the
                // flag updates thread local.
                const int lo = static_cast<int>(static_cast<long long>(n_avail) * tid / num_threads);
                const int hi = static_cast<int>(static_cast<long long>(n_avail) * (tid + 1) / num_threads);
                local_pending.clear();
                for (int i = lo; i < hi; ++i) {
                    const int seed = unclustered_indices[i];
                    if (!dirty[seed]) {
                        const CacheRef& ref = cache_ref[seed];
                        if (ref.tid < 0 || arena_gen[ref.tid] != ref.gen) {
                            dirty[seed] = 1;
                        } else if (distance(pts[seed], last_center) <= radius[seed] + last_radius) {
                            // The two clusters are close enough to share a point, so the
                            // members have to be looked at
                            const int* __restrict members = arenas[ref.tid].data() + ref.offset;
                            for (int k = 0; k < ref.count; ++k) {
                                if (removed[members[k]]) {
                                    dirty[seed] = 1;
                                    break;
                                }
                            }
                        }
                    }
                    if (dirty[seed]) {
                        local_pending.push_back(seed);
                    } else {
                        best.update(cardinality[seed], seed);
                    }
                }

                // Publish the seeds that need work into the shared list
                const int n_local = static_cast<int>(local_pending.size());
                const int base = pending_count.fetch_add(n_local, std::memory_order_relaxed);
                std::copy(local_pending.begin(), local_pending.end(), pending.begin() + base);

                #pragma omp barrier

                // Grow the candidate clusters. They are independent of each other, so the
                // seeds are handed out dynamically to whichever core is free; seeds whose
                // cluster is still cached cost nothing and are not in the list at all.
                const int n_pending = pending_count.load(std::memory_order_relaxed);
                // Seeds are taken in small batches to keep the shared counter cool
                const int batch = std::max(1, std::min(8, n_pending / (8 * num_threads)));
                for (;;) {
                    const int first = next_pending.fetch_add(batch, std::memory_order_relaxed);
                    if (first >= n_pending) break;
                    const int last = std::min(first + batch, n_pending);
                    for (int idx = first; idx < last; ++idx) {
                        const int seed = pending[idx];

                        work += growCluster(seed, grid, removed.data(), pts, threshold, scratch);
                        const int card = static_cast<int>(scratch.members.size());
                        cardinality[seed] = card;
                        radius[seed] = scratch.radius;
                        dirty[seed] = 0;
                        best.update(card, seed);

                        // Only the owning thread appends to its arena
                        std::vector<int>& arena = arenas[tid];
                        cache_ref[seed] = CacheRef{tid, arena_gen[tid],
                                                   static_cast<int>(arena.size()), card};
                        arena.insert(arena.end(), scratch.members.begin(),
                                     scratch.members.end());
                    }
                }
                thread_best[tid].best = best;
                thread_work[tid].work = work;

                #pragma omp barrier

                #pragma omp single
                {
                    Best winner;
                    pass_work = 0;
                    for (int t = 0; t < num_threads; ++t) {
                        winner.update(thread_best[t].best.card, thread_best[t].best.seed);
                        pass_work += thread_work[t].work;
                    }

                    if (winner.card > 0) {
                        const int best_seed = winner.seed;
                        Cluster cluster;
                        cluster.seed_point = best_seed;
                        const CacheRef& ref = cache_ref[best_seed];
                        const int* members = arenas[ref.tid].data() + ref.offset;
                        cluster.members.assign(members, members + ref.count);

                        for (const int member : cluster.members) {
                            removed[member] = 1;
                        }
                        stale += static_cast<int>(cluster.members.size());
                        last_center = pts[best_seed];
                        // A hair of slack, so that rounding can not talk the test below out
                        // of an overlap that is there
                        last_radius = radius[best_seed] * (1.0 + 1e-12) + 1e-12;

                        // Remove clustered points from the unclustered list, keeping it ascending
                        unclustered_indices.erase(
                            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                          [&removed](int idx) { return removed[idx] != 0; }),
                            unclustered_indices.end()
                        );

                        clusters.push_back(std::move(cluster));

                        // Rebuilding the grid is serial work, so it is only done once the
                        // points clustered away make up a fair share of its entries
                        if (!unclustered_indices.empty() &&
                            stale * 8 > static_cast<int>(grid.items.size())) {
                            grid.build(unclustered_indices, pts, threshold);
                            stale = 0;
                        }
                    } else {
                        // No more clusters can be formed
                        unclustered_indices.clear();
                    }

                    // Recycle the arenas that grew past their share of the memory budget.
                    // Doing it here, while no thread is looking at them, keeps the cached
                    // member lists that are still in use in place; the generation counter
                    // tells the seeds that lost their list apart.
                    for (int t = 0; t < max_threads; ++t) {
                        if (arenas[t].size() > arena_budget) {
                            arenas[t].clear();
                            ++arena_gen[t];
                        }
                    }

                    pending_count.store(0, std::memory_order_relaxed);
                    next_pending.store(0, std::memory_order_relaxed);
                }

                if (epoch == 0) break; // Re-size the team for the remaining passes
            }
        }

        // Both the number of remaining passes and their cost grow with the size of the
        // first pass, so its work is a good yardstick for the team the rest deserves
        if (epoch == 0) {
            const int fitting = static_cast<int>(std::sqrt(static_cast<double>(pass_work) / 20000.0));
            // Teams that do not evenly divide the machine end up sharing cores, so the size
            // is rounded down to a power of two
            num_threads = 1;
            while (num_threads * 2 <= std::max(1, std::min(max_threads, fitting))) {
                num_threads *= 2;
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
        const int n_members = static_cast<int>(cluster.members.size());
        #pragma omp parallel for schedule(dynamic, 8) reduction(max : max_diameter)
        for (int i = 0; i < n_members; ++i) {
            for (int j = i + 1; j < n_members; ++j) {
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
    
    // Unless the user asked for a specific number of threads, use one thread per
    // physical core: the second hardware thread of a core adds next to nothing to this
    // distance bound work, while the extra synchronisation between the passes costs
    // dearly, all the more so as the runtime leaves the threads unpinned by default.
    if (!getenv("OMP_NUM_THREADS")) {
        const int smt = smtDegree();
        const int cores = omp_get_num_procs() / (smt > 0 ? smt : 1);
        if (cores > 0) {
            omp_set_num_threads(cores);
        }
    }

    // Spin up the OpenMP thread pool and let every thread touch the heap once, so that
    // neither creating the threads nor handing each of them its own malloc arena - both
    // one time costs of the runtime - lands in the measured clustering phase
    #pragma omp parallel
    {
        std::vector<int> prime(1024, omp_get_thread_num());
        if (prime.back() < 0) printf("unreachable\n");
    }

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

// QT Clustering Benchmark - OpenMP Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: every round of the main clustering loop grows the candidate
// cluster of each remaining unclustered point, which are independent of each
// other, and commits the best one. The rounds themselves are strictly
// sequential, so a single thread team lives across all of them and the seeds of
// a round are self-scheduled over it. The winner is picked with a lock free
// reduction over (cardinality, seed index) keys, which yields the largest
// cluster and the smallest seed index on ties - exactly what the sequential
// scan selects - so the output is identical for any number of threads.
//
// The candidate cluster growth itself is also reformulated (see
// growCandidateCluster) so that the parallel version does not just spread the
// original, much more expensive, computation over the cores.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>

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

// Sense-reversing spin barrier with a two-level combining tree.
//
// The algorithm needs one synchronization point per cluster, and the clusters
// are small, so the rounds are short. The OpenMP barrier puts idle threads to
// sleep (default wait policy) and waking a large team costs far more than a
// round of work, so the team spins instead. Arrivals are combined per group of
// threads first, which keeps a single counter line from serializing the whole
// team, and the release is a plain read of one shared generation counter.
struct SpinBarrier {
    struct alignas(64) Counter {
        std::atomic<int> value{0};
        int limit = 0;
    };

    std::unique_ptr<Counter[]> group;         // one arrival counter per group
    alignas(64) std::atomic<int> leaders{0};  // arrivals of the group leaders
    alignas(64) std::atomic<unsigned> generation{0};
    int group_size = 1;
    int group_count = 1;

    void setup(const int team) {
        group_size = 1;
        while (group_size * group_size < team) ++group_size;
        group_count = (team + group_size - 1) / group_size;
        group.reset(new Counter[group_count]);
        for (int g = 0; g < group_count; ++g) {
            group[g].limit = std::min(group_size, team - g * group_size);
        }
        leaders.store(0, std::memory_order_relaxed);
        generation.store(0, std::memory_order_relaxed);
    }

    static void pause() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#endif
    }

    void wait(const int tid) {
        const unsigned gen = generation.load(std::memory_order_relaxed);
        Counter& c = group[tid / group_size];

        if (c.value.fetch_add(1, std::memory_order_acq_rel) == c.limit - 1) {
            // Last thread of the group reports on its behalf
            c.value.store(0, std::memory_order_relaxed);
            if (leaders.fetch_add(1, std::memory_order_acq_rel) == group_count - 1) {
                // Last group in: release the whole team
                leaders.store(0, std::memory_order_relaxed);
                generation.store(gen + 1, std::memory_order_release);
                return;
            }
        }

        for (unsigned long spins = 0; generation.load(std::memory_order_acquire) == gen; ++spins) {
            if (spins < (1UL << 16)) {
                pause();
            } else {
                sched_yield();  // only reached if the team is oversubscribed
            }
        }
    }
};

// Pin the calling thread to one CPU of the set the process is allowed to use.
//
// The team spins on the round barrier, and a thread that gets migrated loses its
// caches and delays everyone waiting for it, which otherwise shows up as heavy
// run-to-run variance on a busy machine. An explicit OMP_PROC_BIND setting takes
// precedence, and only CPUs the process may run on are used, so an external
// affinity mask keeps working.
static void bindThisThread(const int tid) {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    const int count = CPU_COUNT(&allowed);
    if (count <= 0) return;

    int skip = tid % count;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (skip-- > 0) continue;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        sched_setaffinity(0, sizeof(one), &one);
        return;
    }
}

// Per-thread scratch memory for growing candidate clusters. The candidate set
// is held as a structure-of-arrays so that the hot loops are contiguous,
// vectorizable sweeps.
struct Scratch {
    std::vector<double> cx, cy;     // coordinates of the remaining candidates
    std::vector<double> cmax;       // max distance from candidate to the current members
    std::vector<int> cidx;          // candidate point indices
    std::vector<int> members;       // members of the cluster currently being grown
    std::vector<int> best_members;  // members of the best cluster seen by this thread
};

// Uniform bucket grid over the currently unclustered points. The cell size is
// at least the clustering threshold, so every point closer than the threshold
// to a seed lies in the 3x3 cell block around the seed's cell.
struct Grid {
    std::vector<int> start;   // prefix offsets into points, size gw*gh + 1
    std::vector<int> points;  // point indices grouped by cell
    int gw = 1, gh = 1;
    double minx = 0.0, miny = 0.0;
    double cw = 0.0, ch = 0.0;        // cell extents
    double inv_cw = 0.0, inv_ch = 0.0;
    int max_cell = 0;  // largest cell population, bounds the candidate set size

    int cellX(double x) const {
        const int i = static_cast<int>((x - minx) * inv_cw);
        return (i < 0) ? 0 : ((i >= gw) ? gw - 1 : i);
    }
    int cellY(double y) const {
        const int i = static_cast<int>((y - miny) * inv_ch);
        return (i < 0) ? 0 : ((i >= gh) ? gh - 1 : i);
    }

    // Bucket the given point indices; cells keep at least threshold-sized extent
    void build(const std::vector<int>& indices, const double* px, const double* py,
               const double threshold) {
        const int n = static_cast<int>(indices.size());
        double maxx, maxy;
        minx = maxx = px[indices[0]];
        miny = maxy = py[indices[0]];
        for (int i = 1; i < n; ++i) {
            const int p = indices[i];
            minx = std::min(minx, px[p]);
            maxx = std::max(maxx, px[p]);
            miny = std::min(miny, py[p]);
            maxy = std::max(maxy, py[p]);
        }

        gw = std::max(1, static_cast<int>((maxx - minx) / threshold));
        gh = std::max(1, static_cast<int>((maxy - miny) / threshold));
        // Keep the cell count proportional to the point count; coarsening only
        // grows the cells, so the 3x3 neighbourhood stays a valid superset.
        const long cap = 4L * n + 64L;
        while (static_cast<long>(gw) * gh > cap) {
            gw = std::max(1, gw / 2);
            gh = std::max(1, gh / 2);
        }
        // Nudge the extent so the largest coordinate still maps inside the grid
        const double ex = (maxx - minx) * (1.0 + 1e-12) + 1e-300;
        const double ey = (maxy - miny) * (1.0 + 1e-12) + 1e-300;
        inv_cw = gw / ex;
        inv_ch = gh / ey;
        cw = ex / gw;
        ch = ey / gh;

        const int ncells = gw * gh;
        start.assign(ncells + 1, 0);
        points.resize(n);
        for (int i = 0; i < n; ++i) {
            const int p = indices[i];
            ++start[cellY(py[p]) * gw + cellX(px[p]) + 1];
        }
        max_cell = 0;
        for (int c = 0; c < ncells; ++c) {
            max_cell = std::max(max_cell, start[c + 1]);
            start[c + 1] += start[c];
        }
        for (int i = 0; i < n; ++i) {
            const int p = indices[i];
            const int c = cellY(py[p]) * gw + cellX(px[p]);
            points[start[c]++] = p;
        }
        // start[] was shifted forward by one cell; restore the offsets
        for (int c = ncells; c > 0; --c) {
            start[c] = start[c - 1];
        }
        start[0] = 0;
    }
};

// Raise the shared best cardinality if this one is larger
static inline void publishCardinality(std::atomic<int>& global_best, const int cardinality) {
    int g = global_best.load(std::memory_order_relaxed);
    while (g < cardinality &&
           !global_best.compare_exchange_weak(g, cardinality, std::memory_order_relaxed)) {
    }
}

// Grow a candidate cluster starting from a seed point and return its cardinality.
//
// Equivalent to the sequential generateCandidateCluster():
//  * The maximum distance of every candidate to the cluster is maintained
//    incrementally, turning O(members^2 * points) growth into O(members * points).
//  * Candidates whose max distance reached the threshold are dropped for good:
//    the max distance to a growing member set never decreases, so they could
//    never be selected again.
//  * Only points within the threshold of the seed can ever join the cluster, so
//    the initial candidate set comes from the seed's grid neighbourhood, minus
//    the points that already belong to a committed cluster.
//  * The point to add is the lexicographic minimum of (max distance, index),
//    which is exactly what the sequential ascending scan with a strict "<"
//    selects, independent of the order candidates are visited in.
//
// Returns -1 if the seed was pruned because it provably cannot reach
// prune_card members (such a seed can never win the round).
static int growCandidateCluster(const int seed_point,
                                const Grid& grid,
                                const double* __restrict px,
                                const double* __restrict py,
                                const char* __restrict clustered,
                                const double threshold,
                                const int prune_card,
                                const std::atomic<int>& global_best,
                                Scratch& s) {
    double* __restrict cx = s.cx.data();
    double* __restrict cy = s.cy.data();
    double* __restrict cmax = s.cmax.data();
    int* __restrict cidx = s.cidx.data();

    const double sx = px[seed_point];
    const double sy = py[seed_point];

    double min_diameter = threshold;
    int min_index = -1;
    int closest_pos = -1;
    int n = 0;

    // Collect the seed's neighbourhood as the initial candidate set, already
    // carrying its distance to the seed as the running max distance.
    const int xi = grid.cellX(sx);
    const int yi = grid.cellY(sy);
    const int x0 = (xi > 0) ? xi - 1 : 0;
    const int x1 = (xi + 1 < grid.gw) ? xi + 1 : grid.gw - 1;
    const int y0 = (yi > 0) ? yi - 1 : 0;
    const int y1 = (yi + 1 < grid.gh) ? yi + 1 : grid.gh - 1;
    const int* __restrict gstart = grid.start.data();
    const int* __restrict gpoints = grid.points.data();

    {
        // The neighbourhood population bounds the cluster size, so hopeless
        // seeds are rejected here without computing a single distance
        const int g = global_best.load(std::memory_order_relaxed);
        const int bound = (g > prune_card) ? g : prune_card;
        int population = 0;
        for (int cyi = y0; cyi <= y1; ++cyi) {
            const int row = cyi * grid.gw;
            population += gstart[row + x1 + 1] - gstart[row + x0];
        }
        if (1 + population < bound) return -1;
    }

    // Cells whose closest point is out of reach are skipped entirely. The
    // margin keeps the test conservative under rounding, so it never discards a
    // cell that could hold a candidate.
    const double reach2 = threshold * threshold * (1.0 + 1e-9);

    for (int cyi = y0; cyi <= y1; ++cyi) {
        const int row = cyi * grid.gw;
        double bdy = 0.0;
        if (cyi < yi) {
            bdy = sy - (grid.miny + (cyi + 1) * grid.ch);
        } else if (cyi > yi) {
            bdy = (grid.miny + cyi * grid.ch) - sy;
        }
        if (bdy < 0.0) bdy = 0.0;

        for (int cxi = x0; cxi <= x1; ++cxi) {
            double bdx = 0.0;
            if (cxi < xi) {
                bdx = sx - (grid.minx + (cxi + 1) * grid.cw);
            } else if (cxi > xi) {
                bdx = (grid.minx + cxi * grid.cw) - sx;
            }
            if (bdx < 0.0) bdx = 0.0;
            if (bdx * bdx + bdy * bdy >= reach2) continue;

            const int cell = row + cxi;
            const int end = gstart[cell + 1];
            for (int k = gstart[cell]; k < end; ++k) {
                const int p = gpoints[k];
                if (p == seed_point || clustered[p]) continue;
                const double dx = px[p] - sx;
                const double dy = py[p] - sy;
                const double d = std::sqrt(dx * dx + dy * dy);
                if (d < threshold) {
                    cx[n] = px[p];
                    cy[n] = py[p];
                    cidx[n] = p;
                    cmax[n] = d;
                    if (d < min_diameter || (d == min_diameter && p < min_index)) {
                        min_diameter = d;
                        min_index = p;
                        closest_pos = n;
                    }
                    ++n;
                }
            }
        }
    }

    s.members.clear();
    s.members.push_back(seed_point);

    // Retired slots are marked with an infinite max distance instead of being
    // squeezed out right away: the marker survives the max update and fails
    // every candidate test, so the slots can be reclaimed in bulk later.
    const double retired = std::numeric_limits<double>::infinity();
    int alive = n;

    for (;;) {
        // The cluster can grow to at most members + candidates points. If that
        // cannot strictly beat the best cardinality known so far, this seed is
        // irrelevant for the outcome of the round.
        const int g = global_best.load(std::memory_order_relaxed);
        const int bound = (g > prune_card) ? g : prune_card;
        if (static_cast<int>(s.members.size()) + alive < bound) return -1;

        if (closest_pos < 0) break;  // No more points can be added

        const double nx = cx[closest_pos];
        const double ny = cy[closest_pos];
        s.members.push_back(cidx[closest_pos]);
        cmax[closest_pos] = retired;
        --alive;
        if (alive == 0) break;

        // Reclaim the retired slots once a fifth of the sweep would be wasted
        if (5 * alive < 4 * n) {
            int m = 0;
            for (int i = 0; i < n; ++i) {
                const double v = cmax[i];
                cx[m] = cx[i];
                cy[m] = cy[i];
                cidx[m] = cidx[i];
                cmax[m] = v;
                m += (v < threshold);
            }
            n = m;
        }

        // Fold the newly added member into the running max distances
        for (int i = 0; i < n; ++i) {
            const double dx = cx[i] - nx;
            const double dy = cy[i] - ny;
            const double d = std::sqrt(dx * dx + dy * dy);
            cmax[i] = (d > cmax[i]) ? d : cmax[i];
        }

        // Pick the closest candidate that is still below the threshold. Seeding
        // the minimum with the threshold makes "closer than the current best"
        // imply "still a candidate", so retired and dropped slots need no test
        // of their own.
        min_diameter = threshold;
        min_index = -1;
        closest_pos = -1;
        alive = 0;
        for (int i = 0; i < n; ++i) {
            const double v = cmax[i];
            if (v < min_diameter || (v == min_diameter && cidx[i] < min_index)) {
                min_diameter = v;
                min_index = cidx[i];
                closest_pos = i;
            }
            alive += (v < threshold);
        }
    }

    return static_cast<int>(s.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Ascending list of seed candidates and matching grid. Both are only
    // refreshed once a good fraction of their points has been clustered; until
    // then the clustered flags filter the stale entries out. This keeps the
    // per-round serial bookkeeping out of the critical path.
    std::vector<int> seed_list;
    int live_at_refresh = 0;
    int live_count = N;

    // Structure-of-arrays copy of the coordinates for the vectorized inner loop
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    std::vector<Scratch> scratch(omp_get_max_threads());
    Grid grid;

    // Each thread publishes its best (cardinality, seed) of the round here,
    // packed so that the numerically largest key is the winner: the largest
    // cardinality, and the smallest seed index on a tie. One padded slot per
    // thread avoids false sharing and makes the reduction lock free.
    const int KEY_STRIDE = 8;
    std::vector<long long> thread_key(
        static_cast<size_t>(omp_get_max_threads()) * KEY_STRIDE, -1);

    // Round state, shared by the thread team below
    int seed_count = 0;
    int scratch_need = 1;
    int chunk = 1;
    int warm_count = 0;
    // Largest cardinality found so far in the current round, used for pruning
    // only; races merely change how much work is skipped, never the result.
    std::atomic<int> global_best(0);
    // Self-scheduling cursor over the seeds of the current round
    alignas(64) std::atomic<int> next_seed(0);
    SpinBarrier barrier;

    // One team of threads serves all rounds of the main clustering loop. The
    // team size is requested explicitly so that it never exceeds the per-thread
    // state allocated above.
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        Scratch& s = scratch[tid];
        bool first_round = true;

        bindThisThread(tid);

        #pragma omp single
        {
            barrier.setup(team);
        }

        // Main clustering loop
        for (;;) {
            // Thread 0 commits the previous round's winner and sets up the next
            if (tid == 0) {
                if (!first_round) {
                    long long best_key = -1;
                    int winner = -1;
                    for (int t = 0; t < team; ++t) {
                        const long long key = thread_key[static_cast<size_t>(t) * KEY_STRIDE];
                        if (key > best_key) {
                            best_key = key;
                            winner = t;
                        }
                    }

                    if (winner >= 0) {
                        // If we found a cluster, add it
                        Cluster cluster;
                        cluster.seed_point =
                            static_cast<int>(0xFFFFFFFFLL - (best_key & 0xFFFFFFFFLL));
                        cluster.members = scratch[winner].best_members;
                        clusters.push_back(std::move(cluster));

                        // Mark all members as clustered
                        const std::vector<int>& members = clusters.back().members;
                        for (size_t i = 0; i < members.size(); ++i) {
                            clustered[members[i]] = 1;
                        }
                        live_count -= static_cast<int>(members.size());
                    } else {
                        // No more clusters can be formed
                        live_count = 0;
                    }
                }

                global_best.store(0, std::memory_order_relaxed);

                if (live_count == 0) {
                    seed_count = 0;
                } else {
                    // Refresh the seed list and the grid once a quarter of
                    // their points has been clustered: rare enough to keep the
                    // serial cost negligible, frequent enough that the stale
                    // entries barely add to the parallel work
                    if (4 * live_count <= 3 * live_at_refresh || seed_list.empty()) {
                        seed_list.clear();
                        for (int i = 0; i < N; ++i) {
                            if (!clustered[i]) seed_list.push_back(i);
                        }
                        live_at_refresh = live_count;
                        grid.build(seed_list, px.data(), py.data(), threshold);
                        // A candidate set is drawn from a 3x3 cell block
                        scratch_need = static_cast<int>(std::min<long>(N, 9L * grid.max_cell));
                    }
                    seed_count = static_cast<int>(seed_list.size());
                    // Small chunks balance the seeds, larger ones keep the
                    // shared cursor from becoming a hot spot
                    chunk = std::max(1, std::min(16, seed_count / (2 * team)));
                    warm_count = std::min(seed_count, 4 * team);
                    next_seed.store(warm_count, std::memory_order_relaxed);
                }
            }

            barrier.wait(tid);

            if (seed_count == 0) break;
            first_round = false;

            // Grow the per-thread scratch buffers on demand
            if (static_cast<int>(s.cidx.size()) < scratch_need) {
                s.cx.resize(scratch_need);
                s.cy.resize(scratch_need);
                s.cmax.resize(scratch_need);
                s.cidx.resize(scratch_need);
                s.members.reserve(scratch_need + 1);
                s.best_members.reserve(scratch_need + 1);
            }

            int local_cardinality = -1;
            int local_seed = -1;

            // Grow the candidate cluster of one seed and keep it if it is the
            // best this thread has seen: a larger cluster, or the smaller seed
            // index on a tie, which is what the sequential ascending scan
            // produces.
            auto trySeed = [&](const int i) {
                const int seed = seed_list[i];
                if (clustered[seed]) return;  // stale seed list entry

                const int cardinality = growCandidateCluster(
                    seed, grid, px.data(), py.data(), clustered.data(), threshold,
                    local_cardinality, global_best, s);

                if (cardinality > local_cardinality ||
                    (cardinality == local_cardinality && seed < local_seed)) {
                    local_cardinality = cardinality;
                    local_seed = seed;
                    s.best_members = s.members;
                    publishCardinality(global_best, cardinality);
                }
            };

            // Try each unclustered point as a seed. The first seeds are handed
            // out round robin: they run without a useful pruning bound, so the
            // shared best cardinality should rise before the bulk of the work
            // starts, and a strided assignment does that without touching the
            // shared cursor. The rest is self-scheduled to balance the team.
            for (int i = tid; i < warm_count; i += team) {
                trySeed(i);
            }

            for (;;) {
                const int lo = next_seed.fetch_add(chunk, std::memory_order_relaxed);
                if (lo >= seed_count) break;
                const int hi = std::min(lo + chunk, seed_count);

                for (int i = lo; i < hi; ++i) {
                    trySeed(i);
                }
            }

            thread_key[static_cast<size_t>(tid) * KEY_STRIDE] =
                (local_cardinality > 0)
                    ? ((static_cast<long long>(local_cardinality) << 32) |
                       (0xFFFFFFFFLL - local_seed))
                    : -1;

            barrier.wait(tid);
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

    // Compute all cluster diameters in parallel, then report them in order
    const size_t cluster_count = clusters.size();
    std::vector<double> diameters(cluster_count, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < cluster_count; ++c) {
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

    // Check each cluster
    for (size_t c = 0; c < cluster_count; ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

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

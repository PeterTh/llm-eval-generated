// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization (shared memory, OpenMP): every round, each unclustered
// point is tried as the seed of a candidate cluster and the largest
// candidate (lowest seed index on ties) is accepted. Candidate clusters of
// different seeds are independent, so each round evaluates them in parallel
// on a persistent OpenMP team (each thread owning a fixed subset of seeds),
// with a deterministic reduction that produces exactly the clusters of the
// sequential algorithm. Work is kept small by
//   * caching candidate clusters across rounds (a seed only needs to be
//     re-evaluated if the accepted cluster took one of its members),
//   * skipping seeds whose neighbor count cannot beat the best known
//     cardinality of the round,
//   * a uniform grid to gather a seed's candidates,
//   * incremental, vectorized maintenance of each candidate's distance to
//     the growing cluster, using exact squared-distance comparisons.
// Results (cluster membership, order of clusters, printed statistics and
// result hash) are identical to the sequential version.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef __GLIBC__
#include <malloc.h>
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Exact squared-distance thresholds.
//
// The reference algorithm compares Euclidean distances d = sqrt(d2) with
// the threshold and with each other. Correctly rounded sqrt is monotonic
// (non-decreasing) in its argument, so
//   sqrt(d2) < threshold   <=>   d2 <= sq_limit
// where sq_limit is the largest double whose square root is still below
// the threshold. Working with squared distances therefore reproduces every
// decision of the reference exactly, while avoiding the sqrt in the hot
// loops. sq_limit is found once by bisection over the ordered bit patterns
// of non-negative doubles.
static double largestSquaredBelow(const double threshold) {
    auto bits = [](double v) { uint64_t b; std::memcpy(&b, &v, sizeof b); return b; };
    auto from = [](uint64_t b) { double v; std::memcpy(&v, &b, sizeof v); return v; };
    uint64_t lo = bits(0.0);  // sqrt(0) = 0 < threshold (threshold > 0)
    uint64_t hi = bits(std::numeric_limits<double>::max());
    if (std::sqrt(from(hi)) < threshold) return from(hi);
    // invariant: sqrt(lo) < threshold <= sqrt(hi)
    while (hi - lo > 1) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (std::sqrt(from(mid)) < threshold) lo = mid; else hi = mid;
    }
    return from(lo);
}

// Uniform grid with cell size >= threshold. Points are stored grouped by
// cell (counting sort), so every point within the threshold of a seed lies
// in the 3x3 block of cells around the seed's cell and those points are
// contiguous ranges of the grouped arrays.
struct PointGrid {
    int nx = 1, ny = 1;
    double inv_cell = 0.0;
    std::vector<int> cell_start;     // size nx*ny + 1
    std::vector<double> gx, gy;      // coordinates in grouped order
    std::vector<int> gid;            // original point index in grouped order
    std::vector<int> pos_of;         // original index -> grouped position
    std::vector<unsigned char> gclustered;  // clustered flag in grouped order

    void build(const std::vector<double>& px, const std::vector<double>& py,
               const double threshold) {
        const int N = static_cast<int>(px.size());
        double minx = 0.0, maxx = 0.0, miny = 0.0, maxy = 0.0;
        if (N > 0) {
            minx = maxx = px[0];
            miny = maxy = py[0];
            for (int i = 1; i < N; ++i) {
                minx = std::min(minx, px[i]); maxx = std::max(maxx, px[i]);
                miny = std::min(miny, py[i]); maxy = std::max(maxy, py[i]);
            }
        }
        // Keep the grid at a sane size even for tiny thresholds
        const double extent = std::max(maxx - minx, maxy - miny);
        double cell = threshold;
        const double max_cells_per_dim = 1024.0;
        if (cell <= 0.0 || extent / cell > max_cells_per_dim) cell = extent / max_cells_per_dim;
        if (!(cell > 0.0)) cell = 1.0;
        inv_cell = 1.0 / cell;
        nx = std::max(1, static_cast<int>((maxx - minx) * inv_cell) + 1);
        ny = std::max(1, static_cast<int>((maxy - miny) * inv_cell) + 1);
        origin_x = minx;
        origin_y = miny;

        const int ncells = nx * ny;
        std::vector<int> cell_of(N);
        cell_start.assign(ncells + 1, 0);
        for (int i = 0; i < N; ++i) {
            const int c = cellIndex(px[i], py[i]);
            cell_of[i] = c;
            ++cell_start[c + 1];
        }
        for (int c = 0; c < ncells; ++c) cell_start[c + 1] += cell_start[c];
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        gx.resize(N); gy.resize(N); gid.resize(N); pos_of.resize(N);
        gclustered.assign(N, 0);
        for (int i = 0; i < N; ++i) {           // ascending index within a cell
            const int p = fill[cell_of[i]]++;
            gx[p] = px[i]; gy[p] = py[i]; gid[p] = i; pos_of[i] = p;
        }
    }

    int cellX(const double x) const {
        return std::min(nx - 1, std::max(0, static_cast<int>((x - origin_x) * inv_cell)));
    }
    int cellY(const double y) const {
        return std::min(ny - 1, std::max(0, static_cast<int>((y - origin_y) * inv_cell)));
    }
    int cellIndex(const double x, const double y) const { return cellY(y) * nx + cellX(x); }

  private:
    double origin_x = 0.0, origin_y = 0.0;
};

// Per-thread scratch space for candidate cluster generation.
// Candidate data is kept in contiguous arrays so the per-step update is a
// pure streaming, vectorizable loop.
struct CandidateWorkspace {
    std::vector<int> cand;       // candidate point indices
    std::vector<double> cx, cy;  // candidate coordinates
    std::vector<double> maxd2;   // squared max distance to members, INF = dead
    std::vector<double> blockmin; // minimum of maxd2 per block of candidates
    std::vector<int> members;    // members of the candidate cluster being built
};

// Next representable double above a finite, non-negative value
static inline double nextUp(const double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    ++b;
    double r;
    std::memcpy(&r, &b, sizeof r);
    return r;
}

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster; members are left in ws.members.
//
// Semantics are identical to the naive algorithm (repeatedly add the
// unclustered point whose maximum distance to all current members is
// smallest and below the threshold, lowest index wins ties), but:
//   * only points within the threshold of the seed can ever join, and they
//     are gathered from the 3x3 grid cells around the seed;
//   * the maximum distance of every candidate to the cluster is maintained
//     incrementally, so each growth step costs O(#candidates) instead of
//     O(#candidates * #members);
//   * squared distances are used throughout (see largestSquaredBelow);
//     max and min commute with sqrt, and the tie rule is honoured by
//     selecting, among all candidates whose squared value has the same
//     square root as the minimum, the one with the lowest point index;
//   * candidates that can never be added (distance to seed or to a member
//     >= threshold) are marked dead (INF) and periodically compacted away.
static int generateCandidateCluster(const int seed_point,
                                    const PointGrid& grid,
                                    const double sq_limit,
                                    CandidateWorkspace& ws) {
    constexpr double INF = std::numeric_limits<double>::infinity();

    std::vector<int>& cand = ws.cand;
    std::vector<double>& cxv = ws.cx;
    std::vector<double>& cyv = ws.cy;
    std::vector<double>& maxdv = ws.maxd2;
    std::vector<int>& members = ws.members;
    cand.clear();
    cxv.clear();
    cyv.clear();
    maxdv.clear();
    members.clear();

    const int seed_pos = grid.pos_of[seed_point];
    const double sx = grid.gx[seed_pos];
    const double sy = grid.gy[seed_pos];

    // Initial candidate set: unclustered points (other than the seed) whose
    // distance to the seed is below the threshold.
    {
        const double* __restrict gx = grid.gx.data();
        const double* __restrict gy = grid.gy.data();
        const int* __restrict gid = grid.gid.data();
        const unsigned char* __restrict gcl = grid.gclustered.data();
        const int cx0 = grid.cellX(sx);
        const int cy0 = grid.cellY(sy);
        for (int cy = std::max(0, cy0 - 1); cy <= std::min(grid.ny - 1, cy0 + 1); ++cy) {
            for (int cx = std::max(0, cx0 - 1); cx <= std::min(grid.nx - 1, cx0 + 1); ++cx) {
                const int c = cy * grid.nx + cx;
                const int beg = grid.cell_start[c];
                const int e = grid.cell_start[c + 1];
                for (int k = beg; k < e; ++k) {
                    const double dx = gx[k] - sx;
                    const double dy = gy[k] - sy;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 <= sq_limit && !gcl[k] && k != seed_pos) {
                        cand.push_back(gid[k]);
                        cxv.push_back(gx[k]);
                        cyv.push_back(gy[k]);
                        maxdv.push_back(d2);
                    }
                }
            }
        }
    }

    members.push_back(seed_point);

    // Candidates are processed in blocks; the minimum of each block is kept so
    // that locating the argmin only has to scan the blocks that can tie.
    constexpr int BLOCK = 64;
    int n = static_cast<int>(cand.size());
    std::vector<double>& bmin = ws.blockmin;
    bmin.assign((n + BLOCK - 1) / BLOCK, INF);
    double best2 = INF;   // smallest live squared max-distance
    {
        const double* __restrict md = maxdv.data();
        for (int b = 0; b * BLOCK < n; ++b) {
            const int e = std::min(n, (b + 1) * BLOCK);
            double bm = INF;
            #pragma omp simd reduction(min : bm)
            for (int i = b * BLOCK; i < e; ++i) bm = std::min(bm, md[i]);
            bmin[b] = bm;
            best2 = std::min(best2, bm);
        }
    }

    int steps_since_compact = 0;
    while (best2 != INF) {
        int* __restrict ci = cand.data();
        double* __restrict cxp = cxv.data();
        double* __restrict cyp = cyv.data();
        double* __restrict md = maxdv.data();
        double* __restrict bm = bmin.data();
        const int nblocks = (n + BLOCK - 1) / BLOCK;

        // All candidates whose squared value maps to the same square root as
        // the minimum tie in the reference; the lowest point index among them
        // wins. Such values form a tiny contiguous range [best2, hi].
        const double s = std::sqrt(best2);
        double hi = best2;
        for (;;) {
            const double next = nextUp(hi);
            if (std::sqrt(next) == s) hi = next; else break;
        }
        int best_id = std::numeric_limits<int>::max();
        int best_blk = 0;
        for (int blk = 0; blk < nblocks; ++blk) {
            if (bm[blk] > hi) continue;
            const int beg = blk * BLOCK;
            const int e = std::min(n, beg + BLOCK);
            int bid = std::numeric_limits<int>::max();
            #pragma omp simd reduction(min : bid)
            for (int i = beg; i < e; ++i) {
                const int id = (md[i] <= hi) ? ci[i] : std::numeric_limits<int>::max();
                bid = std::min(bid, id);
            }
            if (bid < best_id) {
                best_id = bid;
                best_blk = blk;
            }
        }
        int best = best_blk * BLOCK;
        while (ci[best] != best_id) ++best;

        const double x = cxp[best];
        const double y = cyp[best];
        members.push_back(best_id);
        md[best] = INF;

        // Streaming update: new squared max distance for every candidate,
        // killing those that can no longer be added, plus per-block minima.
        double new_best = INF;
        for (int blk = 0; blk < nblocks; ++blk) {
            const int beg = blk * BLOCK;
            const int e = std::min(n, beg + BLOCK);
            double blockbest = INF;
            #pragma omp simd reduction(min : blockbest)
            for (int i = beg; i < e; ++i) {
                const double dx = cxp[i] - x;
                const double dy = cyp[i] - y;
                const double d2 = dx * dx + dy * dy;
                double m = (d2 > md[i]) ? d2 : md[i];
                m = (m <= sq_limit) ? m : INF;
                md[i] = m;
                blockbest = (m < blockbest) ? m : blockbest;
            }
            bm[blk] = blockbest;
            new_best = std::min(new_best, blockbest);
        }
        best2 = new_best;

        // Periodically compact away dead entries
        if (++steps_since_compact >= 8) {
            steps_since_compact = 0;
            int alive = 0;
            #pragma omp simd reduction(+ : alive)
            for (int i = 0; i < n; ++i) alive += (md[i] != INF) ? 1 : 0;
            if (alive < (n * 3) / 4) {
                int w = 0;
                for (int i = 0; i < n; ++i) {
                    if (md[i] != INF) {
                        ci[w] = ci[i];
                        cxp[w] = cxp[i];
                        cyp[w] = cyp[i];
                        md[w] = md[i];
                        ++w;
                    }
                }
                n = w;
                cand.resize(n);
                cxv.resize(n);
                cyv.resize(n);
                maxdv.resize(n);
                // Recompute block minima for the compacted layout
                const int nb = (n + BLOCK - 1) / BLOCK;
                bmin.resize(nb);
                for (int blk = 0; blk < nb; ++blk) {
                    const int beg = blk * BLOCK;
                    const int e = std::min(n, beg + BLOCK);
                    double blockbest = INF;
                    #pragma omp simd reduction(min : blockbest)
                    for (int i = beg; i < e; ++i) blockbest = std::min(blockbest, md[i]);
                    bmin[blk] = blockbest;
                }
            }
        }
    }

    return static_cast<int>(members.size());
}

// Thread placement.
//
// The team never exceeds the number of physical cores (Linux sysfs
// topology): the growth kernel saturates the vector units, so SMT siblings
// add nothing but double the cost of the per-round synchronizations. If the
// user configured placement (OMP_PLACES or OMP_PROC_BIND) the runtime's
// placement is kept; otherwise each thread is pinned to its own core, which
// avoids migrations between the rounds. OMP_NUM_THREADS is honoured up to
// the physical core count.
static std::vector<int> g_pin_cpus;   // CPU to pin thread i to (may be empty)

#ifdef __linux__
// Parse a sysfs CPU list such as "0-3,8,10-11" into a set
static bool readCpuList(const char* path, cpu_set_t* out) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[4096];
    const bool ok = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!ok) return false;
    CPU_ZERO(out);
    const char* p = buf;
    while (*p) {
        char* endp;
        const long a = std::strtol(p, &endp, 10);
        if (endp == p) break;
        long b = a;
        p = endp;
        if (*p == '-') { b = std::strtol(p + 1, &endp, 10); p = endp; }
        for (long s = a; s <= b; ++s) {
            if (s >= 0 && s < CPU_SETSIZE) CPU_SET(s, out);
        }
        if (*p == ',') ++p; else break;
    }
    return true;
}

// Lowest CPU of every physical core within `candidates`, in CPU order
static std::vector<int> physicalCores(const cpu_set_t& candidates) {
    std::vector<int> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &candidates)) continue;
        char path[128];
        std::snprintf(path, sizeof path,
                      "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        cpu_set_t siblings;
        if (!readCpuList(path, &siblings)) return {};
        int lowest = cpu;
        for (int s = 0; s < cpu; ++s) {
            if (CPU_ISSET(s, &siblings) && CPU_ISSET(s, &candidates)) { lowest = s; break; }
        }
        if (lowest == cpu) cores.push_back(cpu);
    }
    return cores;
}
#endif

static int computeTeamSize() {
    const int max_threads = omp_get_max_threads();
#ifdef __linux__
    const bool user_placement = std::getenv("OMP_PLACES") != nullptr ||
                                std::getenv("OMP_PROC_BIND") != nullptr;
    cpu_set_t candidates;
    bool have = false;
    if (user_placement) {
        // The runtime may already have bound this thread; use all online CPUs
        have = readCpuList("/sys/devices/system/cpu/online", &candidates);
    } else {
        CPU_ZERO(&candidates);
        have = sched_getaffinity(0, sizeof candidates, &candidates) == 0;
    }
    if (have) {
        const std::vector<int> cores = physicalCores(candidates);
        if (!cores.empty()) {
            const int team = std::min(max_threads, static_cast<int>(cores.size()));
            if (!user_placement) g_pin_cpus.assign(cores.begin(), cores.begin() + team);
            return team;
        }
    }
#endif
    return max_threads;
}

// Team size used by all parallel regions (computed once, before any pinning)
static int defaultTeamSize() {
    static const int team = computeTeamSize();
    return team;
}

// Pin the calling OpenMP thread to its core (no-op unless chosen above)
static void pinCurrentThread() {
#ifdef __linux__
    const int tid = omp_get_thread_num();
    if (tid < static_cast<int>(g_pin_cpus.size())) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(g_pin_cpus[tid], &set);
        sched_setaffinity(0, sizeof set, &set);
    }
#endif
}

// Start the OpenMP worker pool with the team size used by qtClustering and
// let every worker touch its heap once, so that one-time runtime costs
// (thread creation, per-thread malloc arenas) are paid before timing starts.
static void warmUpThreads() {
#ifdef __GLIBC__
    // Worker threads allocate their candidate-cluster buffers from
    // per-thread arenas. By default glibc grows and trims those arenas in
    // small steps, each step being an mprotect that serializes on the
    // process-wide memory lock and triggers a TLB shootdown on every core.
    // Grow in large steps and never trim instead.
    mallopt(M_TOP_PAD, 64 << 20);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
#endif
    const int team = defaultTeamSize();
    volatile int sink = 0;
    #pragma omp parallel num_threads(team) proc_bind(close)
    {
        pinCurrentThread();
        std::vector<int> touch(1024, omp_get_thread_num());
        if (touch[0] == 0) sink = omp_get_num_threads();
    }
    (void)sink;
}

// Cached result of the last evaluation of a seed.
//
// The greedy growth from a seed only depends on which points are still
// unclustered. If none of the points clustered since the last evaluation
// were ever picked by that growth, every greedy choice is unchanged (the
// chosen point is still present and the removed points were never the
// argmin), so the cached cluster is still exactly what this round would
// compute. Each round therefore re-evaluates only the seeds whose cached
// cluster intersects the cluster accepted in the previous round.
struct SeedCache {
    std::vector<int> members;  // the full candidate cluster
    int cardinality = 0;
    bool valid = false;
};

// A seed queued for evaluation together with its potential (an upper bound
// on the size of its cluster)
struct EvalItem {
    int seed;
    int potential;
};

// Per-thread work cursor on its own cache line (only touched in heavy rounds)
struct alignas(64) WorkCursor {
    std::atomic<int> next{0};
};

// Per-thread partial result (largest cardinality, lowest seed on ties)
struct ThreadBest {
    int cardinality = -1;
    int seed = -1;
    void offer(const int c, const int s) {
        if (c > cardinality || (c == cardinality && s < seed)) {
            cardinality = c;
            seed = s;
        }
    }
};

// Main QT clustering algorithm
//
// Every round picks, among all unclustered seeds, the candidate cluster with
// the largest cardinality (lowest seed index on ties) and removes it. The
// candidate clusters of different seeds are independent, so each round
// evaluates them in parallel. Rounds are inherently sequential; one
// persistent OpenMP team runs all of them with three barriers per round and
// no other shared synchronization: every thread owns the seeds whose index
// is congruent to its id modulo the team size, keeps their cached results
// and evaluates them itself, so the per-round cost stays small even on
// large teams.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Structure-of-arrays copy of the coordinates
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // Exact squared-distance equivalent of "distance < threshold"
    const double sq_limit = largestSquaredBelow(threshold);

    PointGrid grid;
    grid.build(px, py, threshold);

    const int team = defaultTeamSize();
    std::vector<CandidateWorkspace> workspaces(team);
    // Compact per-thread results: written once per round by their owner,
    // read by everyone after a barrier.
    std::vector<ThreadBest> best_cached(team);    // best valid cached result
    std::vector<ThreadBest> best_evaluated(team); // best result evaluated this round
    // Per-thread evaluation queues (sorted, largest potential first). In
    // heavy rounds threads that run out of own work steal from these.
    std::vector<std::vector<EvalItem>> queues(team);
    std::vector<int> queue_size(team, 0);
    std::vector<WorkCursor> cursors(team);
    std::vector<SeedCache> cache(N);
    std::vector<int> neighbor_count(N);   // unclustered points within threshold
    std::vector<int> removed;             // members of the cluster accepted last round

    // Shared per-round state (written by the master between barriers)
    bool first_round = true;
    bool done = (N == 0);
    int clustered_count = 0;
    // Lower bound on the best cardinality of this round. It only ever holds
    // a cardinality some seed actually achieves in this round, so a seed that
    // provably cannot reach it can be skipped without changing the result
    // (ties go to the smaller seed index, and such a seed is strictly
    // smaller, so it could never have won a tie either).
    std::atomic<int> bound{0};

    #pragma omp parallel num_threads(team) proc_bind(close)
    {
        pinCurrentThread();
        const int tid = omp_get_thread_num();
        CandidateWorkspace& ws = workspaces[tid];

        // Seeds owned by this thread that are still unclustered
        std::vector<int> my_seeds;
        for (int s = tid; s < N; s += team) my_seeds.push_back(s);
        // Owned seeds that need (re)evaluation this round
        std::vector<EvalItem>& queue = queues[tid];
        queue.reserve(my_seeds.size());

        // Evaluate one queued seed (unless it cannot beat the current bound)
        ThreadBest tb_eval;
        auto evaluate = [&](const EvalItem item, const int cached_bound) {
            const int cur_bound = std::max(cached_bound, bound.load(std::memory_order_relaxed));
            if (item.potential < cur_bound) return;

            const int cardinality = generateCandidateCluster(item.seed, grid, sq_limit, ws);

            SeedCache& sc = cache[item.seed];
            sc.members.assign(ws.members.begin(), ws.members.end());
            sc.cardinality = cardinality;
            sc.valid = true;
            tb_eval.offer(cardinality, item.seed);

            // Publish the achieved cardinality as a new lower bound
            int prev = bound.load(std::memory_order_relaxed);
            while (prev < cardinality &&
                   !bound.compare_exchange_weak(prev, cardinality,
                                                std::memory_order_relaxed)) {
            }
        };

        while (!done) {
            const unsigned char* clustered_ptr = clustered.data();
            const double* px_ptr = px.data();
            const double* py_ptr = py.data();

            // Phase 1: drop seeds that got clustered, update neighbor counts
            // (upper bound on cluster size), validate caches, record the best
            // valid cached result and collect the seeds that need evaluation.
            ThreadBest tb_cached;
            queue.clear();
            if (first_round) {
                for (const int s : my_seeds) {
                    const double sx = px_ptr[s];
                    const double sy = py_ptr[s];
                    int cnt = 0;
                    const int cx0 = grid.cellX(sx);
                    const int cy0 = grid.cellY(sy);
                    for (int cy = std::max(0, cy0 - 1); cy <= std::min(grid.ny - 1, cy0 + 1); ++cy) {
                        for (int cx = std::max(0, cx0 - 1); cx <= std::min(grid.nx - 1, cx0 + 1); ++cx) {
                            const int c = cy * grid.nx + cx;
                            const int beg = grid.cell_start[c];
                            const int e = grid.cell_start[c + 1];
                            const double* __restrict gx = grid.gx.data();
                            const double* __restrict gy = grid.gy.data();
                            #pragma omp simd reduction(+ : cnt)
                            for (int k = beg; k < e; ++k) {
                                const double dx = gx[k] - sx;
                                const double dy = gy[k] - sy;
                                cnt += (dx * dx + dy * dy <= sq_limit) ? 1 : 0;
                            }
                        }
                    }
                    neighbor_count[s] = cnt;  // includes the seed itself
                    queue.push_back({s, cnt});
                }
            } else {
                const int removed_n = static_cast<int>(removed.size());
                const int* removed_ptr = removed.data();
                size_t w = 0;
                for (size_t i = 0; i < my_seeds.size(); ++i) {
                    const int s = my_seeds[i];
                    if (clustered_ptr[s]) continue;
                    my_seeds[w++] = s;

                    const double sx = px_ptr[s];
                    const double sy = py_ptr[s];
                    int cnt = 0;
                    #pragma omp simd reduction(+ : cnt)
                    for (int j = 0; j < removed_n; ++j) {
                        const int c = removed_ptr[j];
                        const double dx = px_ptr[c] - sx;
                        const double dy = py_ptr[c] - sy;
                        cnt += (dx * dx + dy * dy <= sq_limit) ? 1 : 0;
                    }
                    neighbor_count[s] -= cnt;

                    SeedCache& sc = cache[s];
                    if (sc.valid) {
                        for (int m : sc.members) {
                            if (clustered_ptr[m]) { sc.valid = false; break; }
                        }
                    }
                    if (sc.valid) {
                        tb_cached.offer(sc.cardinality, s);
                    } else {
                        queue.push_back({s, neighbor_count[s]});
                    }
                }
                my_seeds.resize(w);
            }
            best_cached[tid] = tb_cached;

            // Largest potential first, so that the shared bound becomes tight
            // quickly and prunes the rest.
            std::sort(queue.begin(), queue.end(), [](const EvalItem& a, const EvalItem& b) {
                if (a.potential != b.potential) return a.potential > b.potential;
                return a.seed < b.seed;
            });
            queue_size[tid] = static_cast<int>(queue.size());
            cursors[tid].next.store(0, std::memory_order_relaxed);

            #pragma omp barrier

            // Phase 2: evaluate the queued seeds. The best cached cardinality
            // of any thread is a valid bound from the start. Light rounds
            // (the vast majority) are processed without any shared traffic;
            // in heavy rounds threads steal from each other's queues.
            int cached_bound = 0;
            int total_queued = 0;
            for (int t = 0; t < team; ++t) {
                cached_bound = std::max(cached_bound, best_cached[t].cardinality);
                total_queued += queue_size[t];
            }
            tb_eval = ThreadBest();
            const bool heavy = total_queued > 8 * team;
            if (!heavy) {
                for (const EvalItem& item : queue) evaluate(item, cached_bound);
            } else {
                for (int k = 0; k < team; ++k) {
                    const int v = (tid + k) % team;
                    const int size = queue_size[v];
                    if (cursors[v].next.load(std::memory_order_relaxed) >= size) continue;
                    const std::vector<EvalItem>& q = queues[v];
                    for (;;) {
                        const int i = cursors[v].next.fetch_add(1, std::memory_order_relaxed);
                        if (i >= size) break;
                        evaluate(q[i], cached_bound);
                    }
                }
            }
            best_evaluated[tid] = tb_eval;

            #pragma omp barrier

            // Phase 3 (master): pick the winner, accept the cluster and
            // prepare the next round.
            #pragma omp master
            {
                ThreadBest best;
                for (int t = 0; t < team; ++t) {
                    best.offer(best_cached[t].cardinality, best_cached[t].seed);
                    best.offer(best_evaluated[t].cardinality, best_evaluated[t].seed);
                }
                if (best.seed >= 0 && best.cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = best.seed;
                    cluster.members = cache[best.seed].members;

                    // Mark all members as clustered
                    for (size_t i = 0; i < cluster.members.size(); ++i) {
                        clustered[cluster.members[i]] = 1;
                        grid.gclustered[grid.pos_of[cluster.members[i]]] = 1;
                    }
                    removed = cluster.members;
                    clusters.push_back(std::move(cluster));
                    clustered_count += best.cardinality;
                    first_round = false;
                    done = (clustered_count >= N);
                } else {
                    // No more clusters can be formed
                    done = true;
                }
                bound.store(0, std::memory_order_relaxed);
            }

            #pragma omp barrier
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

    // Create the OpenMP worker pool up front (one-time runtime cost) so that
    // the timed section measures clustering only.
    warmUpThreads();
    
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

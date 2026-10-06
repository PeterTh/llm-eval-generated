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

#if defined(__AVX__)
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Uniform grid over the domain. Cell size is >= threshold, so every point
// within distance < threshold of a point lies in the 3x3 cell neighborhood.
struct Grid {
    int ncx = 1, ncy = 1;
    double csx = MAX_WIDTH, csy = MAX_HEIGHT;
    std::vector<int> cell_start;  // CSR offsets, size ncx*ncy+1
    std::vector<int> cell_end;    // end of the live (unclustered) part of each cell
    std::vector<int> cell_pts;    // point indices grouped by cell
    std::vector<double> sx, sy;   // coordinates, same layout as cell_pts
    std::vector<int> cell_of;     // cell index of each point
    std::vector<char> touched;

    void build(const std::vector<Point>& points, const double threshold) {
        const double t = threshold * (1.0 + 1e-9);
        ncx = static_cast<int>(std::min(2048.0, std::max(1.0, std::floor(MAX_WIDTH / t))));
        ncy = static_cast<int>(std::min(2048.0, std::max(1.0, std::floor(MAX_HEIGHT / t))));
        csx = MAX_WIDTH / ncx;
        csy = MAX_HEIGHT / ncy;
        const int N = static_cast<int>(points.size());
        cell_of.resize(N);
        cell_start.assign(static_cast<size_t>(ncx) * ncy + 1, 0);
        for (int i = 0; i < N; ++i) {
            int cx = static_cast<int>(points[i].x / csx);
            int cy = static_cast<int>(points[i].y / csy);
            cx = std::min(std::max(cx, 0), ncx - 1);
            cy = std::min(std::max(cy, 0), ncy - 1);
            cell_of[i] = cy * ncx + cx;
            cell_start[cell_of[i] + 1]++;
        }
        for (size_t c = 0; c + 1 < cell_start.size(); ++c) cell_start[c + 1] += cell_start[c];
        cell_pts.resize(N);
        sx.resize(N);
        sy.resize(N);
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int i = 0; i < N; ++i) {
            const int k = fill[cell_of[i]]++;
            cell_pts[k] = i;
            sx[k] = points[i].x;
            sy[k] = points[i].y;
        }
        cell_end.assign(cell_start.begin() + 1, cell_start.end());
        touched.assign(cell_end.size(), 0);
    }

    // Remove newly clustered points from their cells
    void removeClustered(const std::vector<int>& members, const std::vector<char>& clustered) {
        for (const int m : members) {
            const int c = cell_of[m];
            if (touched[c]) continue;
            touched[c] = 1;
            int w = cell_start[c];
            for (int k = cell_start[c]; k < cell_end[c]; ++k) {
                if (!clustered[cell_pts[k]]) {
                    cell_pts[w] = cell_pts[k]; sx[w] = sx[k]; sy[w] = sy[k];
                    ++w;
                }
            }
            cell_end[c] = w;
        }
        for (const int m : members) touched[cell_of[m]] = 0;
    }

    // Call f(nc) for every cell nc in the 3x3 neighborhood of cell c
    template <typename F>
    inline void forNeighborCells(const int c, F&& f) const {
        const int cx = c % ncx, cy = c / ncx;
        const int y0 = std::max(cy - 1, 0), y1 = std::min(cy + 1, ncy - 1);
        const int x0 = std::max(cx - 1, 0), x1 = std::min(cx + 1, ncx - 1);
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) f(y * ncx + x);
    }

    // Call f(j, x, y) for every unclustered point j in the 3x3 cell
    // neighborhood of point i
    template <typename F>
    inline void forNeighborhood(const int i, F&& f) const {
        forNeighborCells(cell_of[i], [&](const int c) {
            const int e = cell_end[c];
            for (int k = cell_start[c]; k < e; ++k) f(cell_pts[k], sx[k], sy[k]);
        });
    }
};

// Scratch buffers for candidate cluster generation
struct Workspace {
    std::vector<double> x, y, md;
    std::vector<int> id;
};

// Generate a candidate cluster starting from a seed point, considering the
// unclustered points held in the grid.
// Returns the cardinality (size) of the cluster.
//
// Equivalent to repeatedly picking the unclustered point with the smallest
// maximum distance to all current members (ties -> lowest index), subject to
// that maximum being < threshold. The per-candidate maximum distance is kept
// incrementally; once it reaches the threshold the candidate is dropped for
// good, since the maximum can only grow.
int generateCandidateCluster(const int seed_point,
                             const std::vector<Point>& points,
                             const Grid& grid,
                             const double threshold,
                             Workspace& ws,
                             std::vector<int>* cluster_members = nullptr,
                             double* bbox = nullptr) {
    ws.x.clear(); ws.y.clear(); ws.md.clear(); ws.id.clear();
    const Point sp = points[seed_point];
    grid.forNeighborhood(seed_point, [&](int j, double x, double y) {
        if (j == seed_point) return;
        const double d = distance(Point{x, y}, sp);
        if (d < threshold) {
            ws.x.push_back(x);
            ws.y.push_back(y);
            ws.md.push_back(d);
            ws.id.push_back(j);
        }
    });

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    double* __restrict cx = ws.x.data();
    double* __restrict cy = ws.y.data();
    double* __restrict cmd = ws.md.data();
    int* __restrict cid = ws.id.data();
    int n = static_cast<int>(ws.id.size());
    int size = 1;
    double bx0 = sp.x, bx1 = sp.x, by0 = sp.y, by1 = sp.y;

    // Candidates whose max distance reached the threshold (or that were
    // added to the cluster, marked with +inf) are dead; they are kept in the
    // arrays until enough accumulate, so the hot loops stay branch-free.
    constexpr int LANES = 4;
    constexpr double INF = std::numeric_limits<double>::infinity();
    int alive = n;

    // Lowest-index candidate at max-distance bd (scalar; used initially and
    // whenever several candidates tie at the minimum)
    auto lowestAt = [&](const double bd) {
        int best = -1, bi = std::numeric_limits<int>::max();
        for (int k = 0; k < n; ++k)
            if (cmd[k] == bd && cid[k] < bi) { bi = cid[k]; best = k; }
        return best;
    };

    double bd = INF;
    for (int k = 0; k < n; ++k) bd = cmd[k] < bd ? cmd[k] : bd;
    int best = (bd < threshold) ? lowestAt(bd) : -1;

    while (best >= 0) {
        // Add chosen candidate to the cluster
        const Point np = {cx[best], cy[best]};
        if (cluster_members) cluster_members->push_back(cid[best]);
        bx0 = std::min(bx0, np.x); bx1 = std::max(bx1, np.x);
        by0 = std::min(by0, np.y); by1 = std::max(by1, np.y);
        ++size;
        cmd[best] = INF;
        --alive;

        // Compact once at least half of the entries are dead
        if (alive * 2 < n) {
            int w = 0;
            for (int k = 0; k < n; ++k) {
                if (cmd[k] < threshold) {
                    cx[w] = cx[k]; cy[w] = cy[k]; cmd[w] = cmd[k]; cid[w] = cid[k];
                    ++w;
                }
            }
            n = w;
        }

        // Update max distances with the new member, tracking the minimum
        // (per lane: minimum, its position, and how many entries equal it)
        double lane[LANES], pos[LANES], eq[LANES], cnt[LANES];
        int k = 0;
#if defined(__AVX__)
        static_assert(LANES == 4, "AVX path uses 4 lanes");
        {
            const __m256d vnx = _mm256_set1_pd(np.x), vny = _mm256_set1_pd(np.y);
            const __m256d vthr = _mm256_set1_pd(threshold), one = _mm256_set1_pd(1.0);
            const __m256d four = _mm256_set1_pd(4.0);
            __m256d vmin = _mm256_set1_pd(INF), vpos = _mm256_set1_pd(-1.0);
            __m256d veq = _mm256_setzero_pd(), vcnt = _mm256_setzero_pd();
            __m256d vk = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
            for (; k + 4 <= n; k += 4) {
                const __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(cx + k), vnx);
                const __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(cy + k), vny);
#if defined(__FMA__)
                const __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
#else
                const __m256d d2 = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
#endif
                const __m256d d = _mm256_sqrt_pd(d2);
                const __m256d m = _mm256_max_pd(d, _mm256_loadu_pd(cmd + k));
                _mm256_storeu_pd(cmd + k, m);
                const __m256d lt = _mm256_cmp_pd(m, vmin, _CMP_LT_OQ);
                const __m256d ise = _mm256_cmp_pd(m, vmin, _CMP_EQ_OQ);
                veq = _mm256_blendv_pd(_mm256_add_pd(veq, _mm256_and_pd(ise, one)), one, lt);
                vpos = _mm256_blendv_pd(vpos, vk, lt);
                vmin = _mm256_blendv_pd(vmin, m, lt);
                vcnt = _mm256_add_pd(vcnt, _mm256_and_pd(_mm256_cmp_pd(m, vthr, _CMP_LT_OQ), one));
                vk = _mm256_add_pd(vk, four);
            }
            _mm256_storeu_pd(lane, vmin);
            _mm256_storeu_pd(pos, vpos);
            _mm256_storeu_pd(eq, veq);
            _mm256_storeu_pd(cnt, vcnt);
        }
#else
        for (int j = 0; j < LANES; ++j) { lane[j] = INF; pos[j] = -1; eq[j] = 0; cnt[j] = 0; }
#endif
        for (; k < n; ++k) {
            const Point cp = {cx[k], cy[k]};
            const double d = distance(cp, np);
            const double m = std::max(cmd[k], d);
            cmd[k] = m;
            const bool lt = m < lane[0];
            eq[0] = lt ? 1 : eq[0] + (m == lane[0]);
            pos[0] = lt ? k : pos[0];
            lane[0] = lt ? m : lane[0];
            cnt[0] += (m < threshold);
        }

        bd = INF;
        alive = 0;
        for (int j = 0; j < LANES; ++j) {
            bd = lane[j] < bd ? lane[j] : bd;
            alive += static_cast<int>(cnt[j]);
        }
        best = -1;
        if (bd < threshold) {
            int ties = 0;
            for (int j = 0; j < LANES; ++j)
                if (lane[j] == bd) { ties += static_cast<int>(eq[j]); best = static_cast<int>(pos[j]); }
            if (ties > 1) best = lowestAt(bd);
        }
    }
    if (bbox) { bbox[0] = bx0; bbox[1] = bx1; bbox[2] = by0; bbox[3] = by1; }
    return size;
}

// Main QT clustering algorithm (MPI-parallel)
//
// Every rank holds the full point set and identical bookkeeping. Each seed has
// a cached value that is either its exact candidate-cluster cardinality or an
// upper bound (1 + number of unclustered points within threshold). Seeds are
// processed in (value desc, index asc) order; the first exact entry is the
// winner, which matches the sequential "first maximum" selection. Inexact
// seeds at the top are evaluated in batches that the ranks share dynamically
// through an atomic counter (MPI one-sided). A seed's exact result can only
// change if one of its members gets clustered, so after a cluster is formed
// only seeds within threshold of a newly clustered point whose member bounding
// box contains that point are reset to their upper bound.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;
    Workspace ws;

    Grid grid;
    grid.build(points, threshold);

    // Neighbor counts (unclustered points within threshold, excluding self)
    std::vector<int> nbcount(N, 0);
    {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            const long long b = static_cast<long long>(N) * r / nprocs;
            const long long e = static_cast<long long>(N) * (r + 1) / nprocs;
            displs[r] = static_cast<int>(b);
            counts[r] = static_cast<int>(e - b);
        }
        const int b = displs[rank], e = b + counts[rank];
        for (int i = b; i < e; ++i) {
            const Point pi = points[i];
            int c = 0;
            grid.forNeighborhood(i, [&](int j, double x, double y) {
                if (j != i && distance(Point{x, y}, pi) < threshold) ++c;
            });
            nbcount[i] = c;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, nbcount.data(),
                       counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
    }

    // Shared work counter on rank 0 for dynamic load balancing
    long* counter = nullptr;
    MPI_Win win;
    MPI_Win_allocate(rank == 0 ? sizeof(long) : 0, sizeof(long), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &counter, &win);
    if (rank == 0) *counter = 0;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);
    long ticket_base = 0;

    std::vector<int> value(N), version(N, 0), evaluator(N, -1);
    int gver = 0; // global version counter: every (re)push gets a unique stamp
    std::vector<char> exact(N, 0);
    std::vector<double> bbox(static_cast<size_t>(N) * 4); // member bounding box of exact seeds

    // Member lists of seeds this rank evaluated (to avoid recomputing winners)
    std::vector<std::vector<int>> cache(N);
    size_t cache_used = 0;
    const size_t cache_budget = static_cast<size_t>(16) << 20; // ints per rank
    auto dropCache = [&](int s) {
        if (!cache[s].empty()) {
            cache_used -= cache[s].size();
            std::vector<int>().swap(cache[s]);
        }
    };

    struct Entry { int val; int idx; int ver; };
    auto cmp = [](const Entry& a, const Entry& b) {
        // max-heap on (val desc, idx asc)
        return a.val < b.val || (a.val == b.val && a.idx > b.idx);
    };
    std::vector<Entry> heap;
    heap.reserve(static_cast<size_t>(N) * 2);
    for (int i = 0; i < N; ++i) {
        value[i] = 1 + nbcount[i];
        heap.push_back({value[i], i, 0});
    }
    std::make_heap(heap.begin(), heap.end(), cmp);

    auto popValid = [&](Entry& out) -> bool {
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            const Entry e = heap.back();
            heap.pop_back();
            if (!clustered[e.idx] && e.ver == version[e.idx]) { out = e; return true; }
        }
        return false;
    };
    auto pushEntry = [&](const Entry& e) {
        heap.push_back(e);
        std::push_heap(heap.begin(), heap.end(), cmp);
    };

    std::vector<int> batch, members, tmp_members;
    std::vector<int> region_cells, contrib;
    std::vector<int> region_off(grid.cell_end.size(), -1);
    std::vector<Entry> skipped;
    std::vector<double> results;
    int remaining = N;

    while (remaining > 0) {
        int winner = -1;
        int batch_mult = 1;
        while (winner < 0) {
            const size_t M = static_cast<size_t>(nprocs) * batch_mult;
            batch.clear();
            skipped.clear();
            Entry e;
            if (!popValid(e)) break;
            if (exact[e.idx]) { winner = e.idx; break; }
            batch.push_back(e.idx);
            // Extend with the following inexact seeds. Once an exact seed is
            // reached, keep filling only up to one seed per rank (speculative
            // work that would otherwise leave ranks idle); exact ones are put
            // back.
            const size_t S = static_cast<size_t>(nprocs);
            while (batch.size() < (skipped.empty() ? M : S) &&
                   skipped.size() < 4 * S && popValid(e)) {
                if (exact[e.idx]) skipped.push_back(e);
                else batch.push_back(e.idx);
            }
            for (const Entry& x : skipped) pushEntry(x);

            // Per seed: {cardinality, bbox x0, x1, y0, y1, evaluating rank + 1};
            // exactly one rank contributes non-zero values, so SUM is exact.
            const long B = static_cast<long>(batch.size());
            results.assign(batch.size() * 6, 0.0);
            auto evalItem = [&](long k) {
                const int s = batch[k];
                std::vector<int>* mem = nullptr;
                if (cache_used < cache_budget) mem = &tmp_members;
                const int c = generateCandidateCluster(s, points, grid, threshold,
                                                       ws, mem, &results[6 * k + 1]);
                results[6 * k] = c;
                results[6 * k + 5] = rank + 1;
                if (mem) {
                    cache[s].assign(tmp_members.begin(), tmp_members.end());
                    cache_used += cache[s].size();
                }
            };
            if (nprocs > 1) {
                const long one = 1;
                for (;;) {
                    long ticket;
                    MPI_Fetch_and_op(&one, &ticket, MPI_LONG, 0, 0, MPI_SUM, win);
                    MPI_Win_flush(0, win);
                    const long k = ticket - ticket_base;
                    if (k >= B) break;
                    evalItem(k);
                }
                // Each rank performs exactly one unsuccessful fetch per batch
                ticket_base += B + nprocs;
                MPI_Allreduce(MPI_IN_PLACE, results.data(), static_cast<int>(results.size()),
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            } else {
                for (long k = 0; k < B; ++k) evalItem(k);
            }
            for (long k = 0; k < B; ++k) {
                const int s = batch[k];
                value[s] = static_cast<int>(results[6 * k]);
                for (int b = 0; b < 4; ++b) bbox[4 * static_cast<size_t>(s) + b] = results[6 * k + 1 + b];
                evaluator[s] = static_cast<int>(results[6 * k + 5]) - 1;
                exact[s] = 1;
                version[s] = ++gver;
                pushEntry({value[s], s, version[s]});
            }
            batch_mult = std::min(batch_mult * 2, 16);
        }
        if (winner < 0) break; // No more clusters can be formed

        // Obtain the winning cluster's members from the rank that evaluated it
        const int owner = evaluator[winner];
        int msize = 0;
        if (rank == owner) {
            if (!cache[winner].empty()) {
                members = cache[winner];
            } else {
                generateCandidateCluster(winner, points, grid, threshold, ws,
                                         &members);
            }
            msize = static_cast<int>(members.size());
        }
        if (nprocs > 1) {
            MPI_Bcast(&msize, 1, MPI_INT, owner, MPI_COMM_WORLD);
            members.resize(msize);
            MPI_Bcast(members.data(), msize, MPI_INT, owner, MPI_COMM_WORLD);
        }

        Cluster cluster;
        cluster.seed_point = winner;
        cluster.members = members;
        clusters.push_back(std::move(cluster));

        for (const int m : members) {
            clustered[m] = 1;
            dropCache(m);
        }
        grid.removeClustered(members, clustered);
        remaining -= static_cast<int>(members.size());

        // Invalidate seeds whose candidate cluster may have changed.
        // Affected seeds are the live points in cells adjacent to a member's
        // cell (the "region"). Ranks split the members; per region point we
        // accumulate the number of new members within threshold and whether
        // the seed's exact result became stale (an exact result only changes
        // if q was one of its members; members lie inside the bounding box).
        region_cells.clear();
        int R = 0;
        for (const int q : members) {
            grid.forNeighborCells(grid.cell_of[q], [&](const int c) {
                if (region_off[c] >= 0) return;
                region_off[c] = R;
                R += grid.cell_end[c] - grid.cell_start[c];
                region_cells.push_back(c);
            });
        }
        contrib.assign(2 * static_cast<size_t>(R), 0);
        int* __restrict dec = contrib.data();
        int* __restrict stale = contrib.data() + R;
        for (size_t i = rank; i < members.size(); i += nprocs) {
            const Point pq = points[members[i]];
            grid.forNeighborCells(grid.cell_of[members[i]], [&](const int c) {
                const int b = grid.cell_start[c], e = grid.cell_end[c];
                const int off = region_off[c] - b;
                for (int k = b; k < e; ++k) {
                    if (distance(Point{grid.sx[k], grid.sy[k]}, pq) < threshold) {
                        ++dec[off + k];
                        const int s = grid.cell_pts[k];
                        const double* bb = &bbox[4 * static_cast<size_t>(s)];
                        if (!exact[s] || (pq.x >= bb[0] && pq.x <= bb[1] &&
                                          pq.y >= bb[2] && pq.y <= bb[3]))
                            stale[off + k] = 1;
                    }
                }
            });
        }
        if (nprocs > 1 && R > 0) {
            MPI_Allreduce(MPI_IN_PLACE, contrib.data(), 2 * R, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
        }
        for (const int c : region_cells) {
            const int b = grid.cell_start[c], e = grid.cell_end[c];
            const int off = region_off[c] - b;
            region_off[c] = -1;
            for (int k = b; k < e; ++k) {
                if (dec[off + k] == 0) continue;
                const int s = grid.cell_pts[k];
                nbcount[s] -= dec[off + k];
                if (stale[off + k]) {
                    exact[s] = 0;
                    version[s] = ++gver;
                    value[s] = 1 + nbcount[s];
                    dropCache(s);
                    pushEntry({value[s], s, version[s]});
                }
            }
        }
        // Periodically drop stale heap entries
        if (heap.size() > static_cast<size_t>(4) * N + 1024) {
            std::vector<Entry> fresh;
            fresh.reserve(static_cast<size_t>(remaining) * 2);
            for (const Entry& e : heap)
                if (!clustered[e.idx] && e.ver == version[e.idx]) fresh.push_back(e);
            heap.swap(fresh);
            std::make_heap(heap.begin(), heap.end(), cmp);
        }
    }

    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
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
    const bool root = (rank == 0);

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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (root) {
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (!root) {
        MPI_Finalize();
        return 0;
    }

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
    
    int rc = 0;
    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }
    
    fflush(stdout);
    MPI_Finalize();
    return rc;
}

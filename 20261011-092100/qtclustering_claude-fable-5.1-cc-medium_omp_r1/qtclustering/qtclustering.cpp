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
#include <memory>
#include <thread>
#include <vector>

#include <omp.h>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
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

// Per-thread scratch space for candidate-cluster generation. All arrays have
// capacity N and live in slices of one shared allocation, so that pages are
// first touched by the thread that uses them.
struct alignas(64) Workspace {
    int* idx = nullptr;          // candidate point indices (kept in ascending order)
    double* x = nullptr;         // candidate x coordinates
    double* y = nullptr;         // candidate y coordinates
    double* md = nullptr;        // max distance from candidate to current cluster members
    int* members = nullptr;      // members of the cluster being grown
    int n_members = 0;
    int initial_active = 0;      // candidates within threshold of the last seed
    // Best candidate cluster found by this thread in the current round
    int best_card = -1;
    int best_seed = -1;
    int* best_members = nullptr;
    int n_best = 0;
};

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

// Two-level sense-reversing spin barrier. The OpenMP runtime's barrier may put
// waiting threads to sleep; re-waking a large thread team several times per
// cluster costs far more than the work itself, so the per-round
// synchronisation uses a spinning barrier instead (with a yield fallback if a
// thread is descheduled). Arrivals are first counted per group of threads and
// then per team, so no single cache line sees more than ~16-way contention.
struct SpinBarrier {
    static constexpr int GROUP = 16;
    struct alignas(64) Counter { std::atomic<int> value{0}; };
    std::vector<Counter> group_count;
    alignas(64) std::atomic<int> team_count{0};
    alignas(64) std::atomic<unsigned> generation{0};
    int n;
    explicit SpinBarrier(int n_) : group_count((n_ + GROUP - 1) / GROUP), n(n_) {}
    void wait(const int tid) {
        const unsigned gen = generation.load(std::memory_order_acquire);
        const int g = tid / GROUP;
        const int group_size = std::min(GROUP, n - g * GROUP);
        const int groups = (n + GROUP - 1) / GROUP;
        if (group_count[g].value.fetch_add(1, std::memory_order_acq_rel) == group_size - 1) {
            group_count[g].value.store(0, std::memory_order_relaxed);
            if (team_count.fetch_add(1, std::memory_order_acq_rel) == groups - 1) {
                team_count.store(0, std::memory_order_relaxed);
                generation.store(gen + 1, std::memory_order_release);
                return;
            }
        }
        // Spin briefly, then keep yielding so that an oversubscribed machine
        // still makes progress.
        unsigned spins = 0;
        while (generation.load(std::memory_order_acquire) == gen) {
            if (spins < 50000u) { cpuRelax(); ++spins; }
            else { std::this_thread::yield(); }
        }
    }
};

// Pin the calling OpenMP thread to the tid-th CPU of the process's affinity
// mask. Without binding, the OS tends to place several spinning threads on the
// same CPU and leave others idle, which throttles the thread on the critical
// path. Respects an explicit OMP_PROC_BIND setting and never pins when the
// team is larger than the number of available CPUs.
void pinThreadToCpu(const int tid, const int num_threads) {
#if defined(__linux__)
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    if (CPU_COUNT(&allowed) < num_threads) return;
    int seen = -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        if (++seen == tid) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
            return;
        }
    }
#else
    (void)tid; (void)num_threads;
#endif
}

// Update every candidate's max distance with respect to a newly added member at
// (px, py), and return the minimum of the updated values together with the
// number of candidates still below the threshold. Candidates already at or
// above the threshold (including the "dead" marker) stay there, since max()
// never decreases. The arithmetic is exactly the same expression as distance()
// so the compiler applies the same FMA contraction and results are
// bit-identical to the scalar function.
static inline void updateCandidates(const double* __restrict cx,
                                    const double* __restrict cy,
                                    double* __restrict md,
                                    const int n,
                                    const double px, const double py,
                                    const double threshold,
                                    double& out_min, int& out_live) {
    double m = std::numeric_limits<double>::max();
    int live = 0;
    int k = 0;
#if defined(__AVX2__)
    const __m256d vpx = _mm256_set1_pd(px);
    const __m256d vpy = _mm256_set1_pd(py);
    const __m256d vthr = _mm256_set1_pd(threshold);
    __m256d vmin = _mm256_set1_pd(m);
    for (; k + 4 <= n; k += 4) {
        const __m256d dx = _mm256_loadu_pd(cx + k) - vpx;
        const __m256d dy = _mm256_loadu_pd(cy + k) - vpy;
        const __m256d d = _mm256_sqrt_pd(dx * dx + dy * dy);
        const __m256d v = _mm256_max_pd(_mm256_loadu_pd(md + k), d);
        _mm256_storeu_pd(md + k, v);
        vmin = _mm256_min_pd(vmin, v);
        live += __builtin_popcount(_mm256_movemask_pd(_mm256_cmp_pd(v, vthr, _CMP_LT_OQ)));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, vmin);
    for (int l = 0; l < 4; ++l) m = std::min(m, lanes[l]);
#endif
    for (; k < n; ++k) {
        const double dx = cx[k] - px;
        const double dy = cy[k] - py;
        const double d = std::sqrt(dx * dx + dy * dy);
        const double v = std::max(md[k], d);
        md[k] = v;
        m = std::min(m, v);
        live += (v < threshold) ? 1 : 0;
    }
    out_min = m;
    out_live = live;
}

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster, or -1 if the search was
// aborted early because the final cardinality would provably be strictly
// less than `abort_below`.
//
// Semantics match the original greedy procedure exactly: at each step the
// unclustered point with the smallest maximum distance to all current members
// (ties broken by smallest point index) is added, as long as that distance is
// below the threshold. The per-candidate maximum distance is maintained
// incrementally; since max() is order independent the values are bit-identical
// to recomputing them from scratch, so the chosen points are identical.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered_indices,
                             const std::vector<Point>& points,
                             const double threshold,
                             Workspace& ws,
                             const int abort_below) {
    constexpr double DEAD = std::numeric_limits<double>::max();

    int* __restrict idx = ws.idx;
    double* __restrict cx = ws.x;
    double* __restrict cy = ws.y;
    double* __restrict md = ws.md;
    int* __restrict members = ws.members;
    int n_members = 0;
    members[n_members++] = seed_point;

    // Initial candidate set: unclustered points (other than the seed) within the
    // threshold of the seed. Only these can ever join the cluster, because the
    // seed is always a member and max distances only grow. unclustered_indices
    // is in ascending order, so candidates are too.
    const Point sp = points[seed_point];
    int n = 0;  // stored candidates (live + dead)
    for (const int i : unclustered_indices) {
        if (i == seed_point) continue;
        const double d = distance(points[i], sp);
        if (d < threshold) {
            idx[n] = i;
            cx[n] = points[i].x;
            cy[n] = points[i].y;
            md[n] = d;
            ++n;
        }
    }

    int live = n;  // candidates still addable
    ws.initial_active = n;

    // First candidate: minimal max-distance, first one wins ties (ascending index)
    int best = -1;
    double best_md = DEAD;
    for (int k = 0; k < n; ++k) {
        if (md[k] < best_md) { best_md = md[k]; best = k; }
    }

    while (live > 0) {
        // Upper bound on final cardinality: current members + addable candidates
        if (n_members + live < abort_below) { ws.n_members = n_members; return -1; }

        // Add the chosen candidate to the cluster and retire it
        const double px = cx[best];
        const double py = cy[best];
        members[n_members++] = idx[best];
        md[best] = DEAD;
        --live;
        if (live == 0) break;

        // Update max distances w.r.t. the new member
        double m;
        updateCandidates(cx, cy, md, n, px, py, threshold, m, live);
        if (live == 0) break;

        // Compact once dead candidates dominate; stable, so ascending index
        // order (and therefore tie-breaking) is preserved.
        if (n - live > live) {
            int w = 0;
            for (int k = 0; k < n; ++k) {
                if (md[k] < threshold) {
                    idx[w] = idx[k]; cx[w] = cx[k]; cy[w] = cy[k]; md[w] = md[k];
                    ++w;
                }
            }
            n = w;
        }

        // Next candidate: first (lowest index) candidate attaining the minimum.
        // m < threshold here because live > 0.
        best = 0;
        while (md[best] != m) ++best;
    }

    ws.n_members = n_members;
    return n_members;
}

// Main QT clustering algorithm (OpenMP parallel over candidate seeds)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Per-seed upper bound on cluster cardinality: number of unclustered points
    // within the threshold (including the seed itself). Bounds only shrink as
    // points get clustered, so a stale value is still a valid upper bound; a
    // seed's bound is refreshed to the exact value whenever it is evaluated.
    std::vector<int> bound(N, 0);

    std::vector<int> order;        // seeds of this round, by decreasing bound
    std::vector<int> order_bound;  // bound snapshot matching `order`
    std::vector<int> bucket(N + 2); // counting-sort scratch (bounds are in [0, N])
    order.reserve(N);
    order_bound.reserve(N);

    // Per-thread scratch space, allocated once (untouched, so pages are faulted
    // in by the owning thread) and reused for every cluster.
    omp_set_dynamic(0);
    const int num_threads = omp_get_max_threads();
    const size_t stride = (static_cast<size_t>(N) + 15) & ~static_cast<size_t>(15); // 64B-aligned slices
    std::unique_ptr<double[]> dpool(new double[stride * 3 * num_threads]);
    std::unique_ptr<int[]> ipool(new int[stride * 3 * num_threads]);
    std::vector<Workspace> workspaces(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        Workspace& ws = workspaces[t];
        ws.x = dpool.get() + stride * (3 * t);
        ws.y = dpool.get() + stride * (3 * t + 1);
        ws.md = dpool.get() + stride * (3 * t + 2);
        ws.idx = ipool.get() + stride * (3 * t);
        ws.members = ipool.get() + stride * (3 * t + 1);
        ws.best_members = ipool.get() + stride * (3 * t + 2);
    }

    // Shared state for one round (one cluster). Atomics live on their own cache
    // lines so that seed claiming does not invalidate the pruning bound.
    alignas(64) std::atomic<int> global_best{0}; // best cardinality so far (pruning bound)
    alignas(64) std::atomic<int> next_seed{0};   // next position in `order` to claim
    alignas(64) int M = 0;                        // number of seeds in this round
    bool done = false;
    std::vector<int> result_card(num_threads, -1); // per-thread best, compact for
    std::vector<int> result_seed(num_threads, -1); // the serial reduction
    SpinBarrier barrier(num_threads);

    // One parallel region for the whole algorithm. Each round: thread 0 commits
    // the previous round's best cluster and prepares the seed order, then all
    // threads search seeds in parallel. Two barriers per round.
    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        #pragma omp single
        barrier.n = team_size; // actual team size (never larger than requested)
        pinThreadToCpu(tid, team_size);
        Workspace& ws = workspaces[tid];
        bool first_round = true;

        // Initial bounds: all points are unclustered
        #pragma omp for schedule(static) nowait
        for (int s = 0; s < N; ++s) {
            const Point sp = points[s];
            int cnt = 0;
            for (int j = 0; j < N; ++j) {
                cnt += (distance(points[j], sp) < threshold) ? 1 : 0;
            }
            bound[s] = cnt;
        }
        barrier.wait(tid);

        for (;;) {
            // --- Commit previous round + set up this round (thread 0 only) ---
            if (tid == 0) {
                if (!first_round) {
                    // Deterministic combine: largest cardinality, then smallest
                    // seed index (equivalent to the original first-wins scan in
                    // ascending index order).
                    int max_cardinality = -1;
                    int best_seed = -1;
                    int best_tid = -1;
                    for (int t = 0; t < team_size; ++t) {
                        if (result_card[t] > max_cardinality ||
                            (result_card[t] == max_cardinality && result_seed[t] < best_seed)) {
                            max_cardinality = result_card[t];
                            best_seed = result_seed[t];
                            best_tid = t;
                        }
                    }

                    if (best_seed >= 0 && max_cardinality > 0) {
                        const Workspace& w = workspaces[best_tid];

                        Cluster cluster;
                        cluster.seed_point = best_seed;
                        cluster.members.assign(w.best_members, w.best_members + w.n_best);
                        clusters.push_back(cluster);

                        // Mark all members as clustered
                        for (int i = 0; i < w.n_best; ++i) {
                            clustered[w.best_members[i]] = 1;
                        }

                        // Remove clustered points from unclustered list
                        unclustered_indices.erase(
                            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                          [&clustered](int idx) { return clustered[idx]; }),
                            unclustered_indices.end()
                        );
                    } else {
                        // No more clusters can be formed
                        done = true;
                    }
                }
                first_round = false;

                if (!done) {
                    if (unclustered_indices.empty()) {
                        done = true;
                    } else {
                        // Seeds are processed in decreasing bound order so a good
                        // lower bound on the best cardinality is found early,
                        // which lets most seeds be pruned.
                        M = static_cast<int>(unclustered_indices.size());
                        order.resize(M);
                        order_bound.resize(M);
                        // Stable counting sort by decreasing bound (O(M + N))
                        std::fill(bucket.begin(), bucket.end(), 0);
                        for (const int seed : unclustered_indices) bucket[N - bound[seed]]++;
                        int pos = 0;
                        for (int b = 0; b <= N; ++b) { const int c = bucket[b]; bucket[b] = pos; pos += c; }
                        for (const int seed : unclustered_indices) {
                            const int s = bucket[N - bound[seed]]++;
                            order[s] = seed;
                            order_bound[s] = bound[seed];
                        }
                        global_best.store(0, std::memory_order_relaxed);
                        next_seed.store(0, std::memory_order_relaxed);
                    }
                }
            }
            barrier.wait(tid);
            if (done) break;

            // --- Parallel search over seeds ---
            // Each thread starts with seed position tid (no contention), then
            // claims further positions dynamically.
            ws.best_card = -1;
            ws.best_seed = -1;
            for (int s = tid; ; s = team_size + next_seed.fetch_add(1, std::memory_order_relaxed)) {
                if (s >= M) break;

                const int gb = global_best.load(std::memory_order_relaxed);

                // A seed whose bound is strictly below the best known cardinality
                // can neither beat it nor tie with it. Since `order` is sorted by
                // decreasing bound and global_best never decreases, every later
                // seed would be pruned as well, so this thread is done. Ties are
                // still evaluated so the smallest-index tie-break is exact.
                if (order_bound[s] < gb) break;

                const int seed = order[s];
                const int cardinality = generateCandidateCluster(
                    seed, unclustered_indices, points, threshold, ws, gb);
                // Refresh to the exact bound (write only on change to avoid
                // needless cache-line invalidations)
                if (bound[seed] != ws.initial_active + 1) bound[seed] = ws.initial_active + 1;
                if (cardinality < 0) continue;       // aborted: strictly worse than gb

                if (cardinality > ws.best_card ||
                    (cardinality == ws.best_card && seed < ws.best_seed)) {
                    ws.best_card = cardinality;
                    ws.best_seed = seed;
                    ws.n_best = ws.n_members;
                    std::copy(ws.members, ws.members + ws.n_members, ws.best_members);
                }

                // Publish an improved pruning bound (atomic max)
                int cur = global_best.load(std::memory_order_relaxed);
                while (cardinality > cur &&
                       !global_best.compare_exchange_weak(cur, cardinality,
                                                          std::memory_order_relaxed)) {
                }
            }
            result_card[tid] = ws.best_card;
            result_seed[tid] = ws.best_seed;
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

    // Start the OpenMP thread pool up front (and settle thread placement) so
    // that runtime initialisation is not attributed to the clustering itself.
    omp_set_dynamic(0);
    #pragma omp parallel
    {
        pinThreadToCpu(omp_get_thread_num(), omp_get_num_threads());
        #pragma omp master
        printf("OpenMP threads: %d\n", omp_get_num_threads());
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

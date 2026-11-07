// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization and optimization strategy (results are identical to the
// sequential program, including all tie-breaking):
//  * Every unclustered point is tried as a seed for a candidate cluster. The
//    trials are independent, so they are distributed over the OpenMP threads
//    with dynamic scheduling inside one persistent parallel region; rounds of
//    the algorithm are separated by barriers rather than fork/join.
//  * Growing one candidate cluster keeps, per remaining candidate, the running
//    maximum of its (squared) distances to the current members, so adding a
//    member costs O(#candidates) instead of O(#candidates * #members). The
//    sweep is kept in structure-of-array form so it vectorizes.
//  * A seed's candidate cluster only depends on the set of unclustered points,
//    and only changes when one of its members gets clustered. Candidate
//    clusters are therefore cached across rounds and only the affected seeds
//    are recomputed.
//  * The best seed (largest cardinality, ties broken by the smallest seed
//    index, exactly like the sequential scan) is selected deterministically.
//  * The team is configured for low barrier latency: by default one pinned
//    thread per physical core, with the allocator warmed up outside the timed
//    region (see configureOpenMPThreads).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>
#if defined(__linux__)
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

// Per-thread scratch storage for candidate cluster generation, reused across
// seeds to avoid repeated allocation. Candidates are kept in structure-of-array
// form (index, coordinates, running maximum squared distance) so the update
// loop is a contiguous, vectorizable sweep.
struct CandidateWorkspace {
    std::vector<int> cand;        // Indices of still-eligible candidate points
    std::vector<double> cx, cy;   // Coordinates of the candidates
    std::vector<double> max_d2;   // Running max squared distance to the members
    std::vector<int> members;     // Members of the candidate cluster being built
};

// Squared Euclidean distance. Written exactly as in distance() so that the
// compiler generates the identical floating-point operations (including any
// fused multiply-add contraction) before the square root.
inline double distance2(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Generate a candidate cluster starting from a seed point.
// `unclustered` lists all currently unclustered point indices in ascending order.
// Returns the cardinality (size) of the cluster; members are left in ws.members.
//
// Semantics are identical to the sequential version: at each step, the eligible
// candidate (not clustered, not already a member) with the smallest maximum
// distance to all current members is added, provided that distance is below the
// threshold. Ties are broken by the lowest point index.
//
// Implementation notes:
//  * Each candidate carries the running maximum of its squared distances to
//    the members added so far. Since sqrt is monotonic and correctly rounded,
//    sqrt(max d2) == max sqrt(d2) exactly, so this equals the maximum distance
//    the sequential code recomputes from scratch.
//  * Comparisons against the threshold and between candidates are done on
//    squared values whenever that is provably identical to comparing the
//    rounded square roots; the rare borderline cases (values within a few ulps
//    of the threshold squared, or of the minimum) are resolved by computing the
//    actual square roots exactly as the sequential code does.
//  * Candidates whose running maximum reaches the threshold can never be added
//    (the maximum only grows); they are marked dead (+inf) and physically
//    compacted out of the arrays only now and then.
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<int>& unclustered,
                                    const std::vector<Point>& points,
                                    const double threshold,
                                    CandidateWorkspace& ws) {
    const double INF = std::numeric_limits<double>::infinity();
    // Relative margin covering the rounding of x*x, sqrt and the comparison:
    // if d2 < thr2_lo then sqrt(d2) < threshold for sure; if d2 > thr2_hi then
    // sqrt(d2) >= threshold for sure; in between we check with sqrt.
    const double EPS_MARGIN = 1.0 - 0x1p-48;
    const double thr2 = threshold * threshold;
    const double thr2_lo = thr2 * EPS_MARGIN;
    const double thr2_hi = thr2 / EPS_MARGIN;

    std::vector<int>& cand = ws.cand;
    std::vector<double>& cx = ws.cx;
    std::vector<double>& cy = ws.cy;
    std::vector<double>& max_d2 = ws.max_d2;
    std::vector<int>& members = ws.members;

    const int n_uncl = static_cast<int>(unclustered.size());
    cand.resize(n_uncl);
    cx.resize(n_uncl);
    cy.resize(n_uncl);
    max_d2.resize(n_uncl);
    members.clear();
    members.push_back(seed_point);

    // Initialize the candidate set with every unclustered point except the
    // seed that lies within the threshold of the seed, in ascending order.
    const Point ps = points[seed_point];
    int n_cand = 0;
    for (int i = 0; i < n_uncl; ++i) {
        const int c = unclustered[i];
        if (c == seed_point) continue;
        const Point pc = points[c];
        const double d2 = distance2(pc, ps);
        if (d2 < thr2_lo || (d2 <= thr2_hi && std::sqrt(d2) < threshold)) {
            cand[n_cand] = c;
            cx[n_cand] = pc.x;
            cy[n_cand] = pc.y;
            max_d2[n_cand] = d2;
            ++n_cand;
        }
    }

    int* __restrict__ cand_p = cand.data();
    double* __restrict__ cx_p = cx.data();
    double* __restrict__ cy_p = cy.data();
    double* __restrict__ md_p = max_d2.data();

    while (n_cand > 0) {
        // Find the live candidate with the smallest running maximum distance.
        // First the minimum squared value (vectorized), then the first entry
        // whose rounded square root equals that of the minimum: candidates are
        // in ascending index order (compaction preserves it), so this
        // reproduces the first-minimum tie-breaking of the sequential scan.
        // Dead candidates hold +inf and are never selected.
        double min_d2 = INF;
        #pragma omp simd reduction(min : min_d2)
        for (int i = 0; i < n_cand; ++i) {
            min_d2 = std::min(min_d2, md_p[i]);
        }
        if (min_d2 == INF) break; // Nothing left below the threshold

        const double min_d = std::sqrt(min_d2);
        const double near = min_d2 / EPS_MARGIN; // Candidates possibly tying
        int best = 0;
        for (;; ++best) {
            const double v = md_p[best];
            if (v <= near && std::sqrt(v) == min_d) break;
        }

        const int closest = cand_p[best];
        members.push_back(closest);
        md_p[best] = INF; // Now a member: no longer a candidate

        // Update the running maxima against the new member; candidates whose
        // maximum certainly reaches the threshold become dead. This sweep is
        // branch-free and vectorizes.
        const Point pc = points[closest];
        int n_dead = 0;
        int n_border = 0; // Entries at or above thr2_lo (includes the dead)
        for (int i = 0; i < n_cand; ++i) {
            const double d2 = distance2(Point{cx_p[i], cy_p[i]}, pc);
            const double old = md_p[i];
            const double m = old < d2 ? d2 : old;
            const bool dead = m > thr2_hi;
            md_p[i] = dead ? INF : m;
            n_dead += dead ? 1 : 0;
            n_border += m >= thr2_lo ? 1 : 0;
        }
        // Rare: entries within rounding distance of the threshold are decided
        // exactly with the square root, as the sequential code does.
        if (n_border > n_dead) {
            for (int i = 0; i < n_cand; ++i) {
                const double m = md_p[i];
                if (m >= thr2_lo && m != INF && !(std::sqrt(m) < threshold)) {
                    md_p[i] = INF;
                    ++n_dead;
                }
            }
        }

        // Compact once a good fraction of the entries are dead.
        if (n_dead > (n_cand >> 2)) {
            int w = 0;
            for (int i = 0; i < n_cand; ++i) {
                if (md_p[i] != INF) {
                    cand_p[w] = cand_p[i];
                    cx_p[w] = cx_p[i];
                    cy_p[w] = cy_p[i];
                    md_p[w] = md_p[i];
                    ++w;
                }
            }
            n_cand = w;
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
//
// In addition to running seed trials in parallel, candidate clusters are
// cached across rounds. The candidate cluster grown from a seed depends only on
// the set of unclustered points; if none of the points removed in a round were
// members of a seed's candidate cluster, that cluster (and its cardinality) is
// unchanged in the next round: every point chosen during its growth is still
// available, the running distances are identical, and the removed points were
// never the (first) minimum at any step. Only seeds whose cached candidate
// cluster intersects the newly formed cluster are recomputed, giving exactly
// the sequential result with far less work.
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

    const int num_threads = omp_get_max_threads();
    std::vector<CandidateWorkspace> workspaces(num_threads);
    for (auto& ws : workspaces) {
        ws.cand.reserve(N);
        ws.cx.reserve(N);
        ws.cy.reserve(N);
        ws.max_d2.reserve(N);
        ws.members.reserve(N);
    }

    // Cached candidate cluster per seed: cardinality (-1 = must recompute) and
    // member list. Member lists are kept only while the total cached size stays
    // within a memory budget; a seed without a stored member list is treated as
    // stale after the round in which it was computed.
    std::vector<int> seed_card(N, -1);
    std::vector<std::vector<int>> seed_members(N);
    std::vector<char> seed_has_members(N, 0);
    std::vector<int> seed_prev_card(N, 0); // Last known cardinality of stale seeds
    const long long cache_budget = 1LL << 28; // ints (1 GiB)
    long long cached_total = 0;

    std::vector<int> todo(N);
    int todo_count = 0; // Shared append counter for the next round's work list
    int n_uncl = N;
    bool done = false;
    double bb_xmin = 0.0, bb_xmax = 0.0, bb_ymin = 0.0, bb_ymax = 0.0;

    // One persistent parallel region runs the whole clustering loop; rounds
    // are separated by barriers instead of repeated fork/join.
    #pragma omp parallel num_threads(num_threads)
    {
        CandidateWorkspace& ws = workspaces[omp_get_thread_num()];
        std::vector<int> local_stale;
        local_stale.reserve(N / num_threads + 16);

        // Build the initial work list: all points, ordered by a cheap local
        // density estimate (points in the surrounding 3x3 grid cells of side
        // `threshold`) so that the most expensive candidate clusters are
        // started first. The evaluation order does not affect the result.
        #pragma omp single
        {
            // Cell side: the threshold, but never finer than 1/64 of the domain
            // (this is only a load-balancing heuristic).
            const double cell = std::max(threshold, std::min(MAX_WIDTH, MAX_HEIGHT) / 64.0);
            const int gx = static_cast<int>(MAX_WIDTH / cell) + 2;
            const int gy = static_cast<int>(MAX_HEIGHT / cell) + 2;
            std::vector<int> counts(static_cast<size_t>(gx) * gy, 0);
            auto cellOf = [&](const Point& p, int& ix, int& iy) {
                ix = std::min(gx - 1, std::max(0, static_cast<int>(p.x / cell)));
                iy = std::min(gy - 1, std::max(0, static_cast<int>(p.y / cell)));
            };
            for (int i = 0; i < N; ++i) {
                int ix, iy;
                cellOf(points[i], ix, iy);
                counts[static_cast<size_t>(iy) * gx + ix]++;
            }
            std::vector<int> density(N, 0);
            for (int i = 0; i < N; ++i) {
                int ix, iy;
                cellOf(points[i], ix, iy);
                int sum = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = iy + dy;
                    if (yy < 0 || yy >= gy) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = ix + dx;
                        if (xx < 0 || xx >= gx) continue;
                        sum += counts[static_cast<size_t>(yy) * gx + xx];
                    }
                }
                density[i] = sum;
            }
            for (int i = 0; i < N; ++i) todo[i] = i;
            std::stable_sort(todo.begin(), todo.end(), [&density](int a, int b) {
                return density[a] > density[b];
            });
        }
        // (implicit barrier)
        int n_todo = N; // Private copy of the current work-list length

        // Main clustering loop (all threads iterate in lockstep)
        while (true) {
            // Compute candidate clusters for stale seeds, in parallel.
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < n_todo; ++i) {
                const int seed = todo[i];
                const int cardinality = generateCandidateCluster(
                    seed, unclustered_indices, points, threshold, ws);
                seed_card[seed] = cardinality;

                long long prev_total;
                #pragma omp atomic capture
                { prev_total = cached_total; cached_total += cardinality; }
                if (prev_total + cardinality <= cache_budget) {
                    seed_members[seed].assign(ws.members.begin(), ws.members.end());
                    seed_has_members[seed] = 1;
                } else {
                    #pragma omp atomic
                    cached_total -= cardinality;
                    seed_has_members[seed] = 0;
                }
            }
            // (implicit barrier)

            // Select the best seed: largest cardinality, first (smallest
            // index) on ties, exactly as the sequential scan over
            // unclustered_indices; record the cluster and remove its members
            // from the unclustered list.
            #pragma omp single
            {
                int max_cardinality = -1;
                int best_seed = -1;
                for (int i = 0; i < n_uncl; ++i) {
                    const int seed = unclustered_indices[i];
                    const int card = seed_card[seed];
                    if (card > max_cardinality) {
                        max_cardinality = card;
                        best_seed = seed;
                    }
                }

                if (best_seed >= 0 && max_cardinality > 0) {
                    if (!seed_has_members[best_seed]) {
                        // Member list was not cached (memory budget); recompute it.
                        generateCandidateCluster(best_seed, unclustered_indices,
                                                 points, threshold, ws);
                        seed_members[best_seed] = ws.members;
                        seed_has_members[best_seed] = 1;
                    }
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = seed_members[best_seed];
                    clusters.push_back(std::move(cluster));
                    // Iterate over the stored copy: the seed is a member of
                    // its own cluster and its cache is released below.
                    const std::vector<int>& best_cluster_members = clusters.back().members;

                    // Mark all members as clustered, release their caches and
                    // record the bounding box of the new cluster for the
                    // invalidation filter.
                    bb_xmin = bb_ymin = std::numeric_limits<double>::max();
                    bb_xmax = bb_ymax = std::numeric_limits<double>::lowest();
                    for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                        const int m = best_cluster_members[i];
                        clustered[m] = 1;
                        bb_xmin = std::min(bb_xmin, points[m].x);
                        bb_xmax = std::max(bb_xmax, points[m].x);
                        bb_ymin = std::min(bb_ymin, points[m].y);
                        bb_ymax = std::max(bb_ymax, points[m].y);
                        if (seed_has_members[m]) {
                            cached_total -= static_cast<long long>(seed_members[m].size());
                            seed_members[m].clear();
                            seed_has_members[m] = 0;
                        }
                        seed_card[m] = -1;
                    }

                    // Remove clustered points from the unclustered list
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                      [&clustered](int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end()
                    );
                    n_uncl = static_cast<int>(unclustered_indices.size());
                    todo_count = 0;
                    if (n_uncl == 0) done = true;
                } else {
                    // No more clusters can be formed
                    done = true;
                }
            }
            // (implicit barrier)
            if (done) break;

            // Invalidate cached candidate clusters that lost a member and
            // collect the stale seeds as the next round's work list.
            // Every member of a seed's candidate cluster lies within the
            // threshold of the seed, so a seed whose distance to the new
            // cluster's bounding box is at least the threshold cannot have lost
            // a member: its member list need not be scanned.
            local_stale.clear();
            long long released = 0;
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < n_uncl; ++i) {
                const int seed = unclustered_indices[i];
                std::vector<int>& mem = seed_members[seed];
                bool stale = !seed_has_members[seed];
                if (!stale) {
                    const double sx = points[seed].x;
                    const double sy = points[seed].y;
                    const double gx = std::max(0.0, std::max(bb_xmin - sx, sx - bb_xmax));
                    const double gy = std::max(0.0, std::max(bb_ymin - sy, sy - bb_ymax));
                    if (gx * gx + gy * gy >= threshold * threshold) continue;
                    for (size_t k = 0; k < mem.size(); ++k) {
                        if (clustered[mem[k]]) { stale = true; break; }
                    }
                }
                if (stale) {
                    if (seed_has_members[seed]) {
                        released += static_cast<long long>(mem.size());
                        mem.clear(); // Keep capacity: avoids cross-thread frees
                        seed_has_members[seed] = 0;
                    }
                    seed_prev_card[seed] = seed_card[seed];
                    seed_card[seed] = -1;
                    local_stale.push_back(seed);
                }
            }
            // Start the most expensive (previously largest) candidate
            // clusters first for better load balance.
            std::sort(local_stale.begin(), local_stale.end(),
                      [&seed_prev_card](int a, int b) {
                          return seed_prev_card[a] > seed_prev_card[b];
                      });
            if (!local_stale.empty()) {
                int base;
                #pragma omp atomic capture
                { base = todo_count; todo_count += static_cast<int>(local_stale.size()); }
                std::copy(local_stale.begin(), local_stale.end(), todo.begin() + base);
            }
            #pragma omp atomic
            cached_total -= released;
            #pragma omp barrier
            n_todo = todo_count;
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
        const int n = static_cast<int>(cluster.members.size());
        #pragma omp parallel for schedule(dynamic, 16) reduction(max : max_diameter)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
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

// Parse a sysfs CPU list such as "0,128" or "0-1,64-65" and mark its entries.
static void markCpuList(const char* list, std::vector<char>& seen) {
    const char* p = list;
    while (*p) {
        char* end;
        long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        p = end;
        if (*p == '-') {
            b = strtol(p + 1, &end, 10);
            p = end;
        }
        for (long c = a; c <= b && c < static_cast<long>(seen.size()); ++c) {
            if (c >= 0) seen[c] = 1;
        }
        if (*p == ',') ++p; else break;
    }
}

// Configure the OpenMP team used for clustering.
//
// The algorithm proceeds in many short synchronized rounds, so it is sensitive
// to barrier latency and to anything that stalls a single thread:
//  * By default (no OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES set) one
//    thread per physical core is used, each pinned to its core; this gives the
//    best and most stable round times on SMT machines. Explicit user settings
//    are honored as-is.
//  * With a hundred-plus threads running, every mmap/munmap/mprotect issued by
//    the process must interrupt all of their CPUs (TLB shootdown), costing
//    milliseconds per call on large machines. glibc's allocator is therefore
//    told to serve large blocks from the heaps instead of mmap, never to give
//    heap memory back while running, and to grow heaps in large steps; and
//    every thread creates and pre-grows its per-thread heap here, outside the
//    timed clustering phase.
static void configureOpenMPThreads() {
    omp_set_dynamic(0);

#if defined(__linux__)
    if (!getenv("OMP_NUM_THREADS") && !getenv("OMP_PROC_BIND") && !getenv("OMP_PLACES")) {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        std::vector<int> cpus; // One CPU per physical core
        if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
            std::vector<char> seen(CPU_SETSIZE, 0);
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (!CPU_ISSET(c, &allowed) || seen[c]) continue;
                cpus.push_back(c);
                seen[c] = 1;
                char path[128];
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
                FILE* f = fopen(path, "r");
                if (!f) continue;
                char buf[4096];
                if (fgets(buf, sizeof(buf), f)) markCpuList(buf, seen);
                fclose(f);
            }
        }
        if (!cpus.empty()) {
            const int nthreads = static_cast<int>(cpus.size());
            omp_set_num_threads(nthreads);
            // Pin each worker of the (persistent) team to its own physical core.
            #pragma omp parallel num_threads(nthreads)
            {
                const int tid = omp_get_thread_num();
                if (tid < nthreads) {
                    cpu_set_t one;
                    CPU_ZERO(&one);
                    CPU_SET(cpus[tid], &one);
                    sched_setaffinity(0, sizeof(one), &one);
                }
            }
        }
    }
#endif

#ifdef __GLIBC__
    mallopt(M_MMAP_THRESHOLD, 32 << 20); // Largest value glibc accepts
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
    mallopt(M_TOP_PAD, 64 << 20);
#endif

    // Let every thread of the team create and pre-grow its malloc heap.
    #pragma omp parallel
    {
        void* warm = malloc(static_cast<size_t>(4) << 20);
        // Make the allocation observable: the compiler is allowed to elide an
        // unused malloc/free pair, which would defeat the warm-up.
        __asm__ __volatile__("" : : "r"(warm) : "memory");
        free(warm);
    }
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
    
    configureOpenMPThreads();
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
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

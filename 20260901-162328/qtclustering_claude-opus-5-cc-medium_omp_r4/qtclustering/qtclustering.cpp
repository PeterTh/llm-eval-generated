// QT Clustering Benchmark - OpenMP Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: within one round of the main loop every unclustered point
// is tried as a seed, and those candidate-cluster expansions are completely
// independent of each other, so they are distributed over the OpenMP team out
// of a single long-lived parallel region. Only the per-round bookkeeping (pick
// the largest cluster, retire its points) is serial.
//
// The expansion itself is also reorganised, without changing what it computes:
//   * each candidate keeps a running maximum distance to the members instead of
//     rescanning them, and is dropped once that maximum reaches the threshold;
//   * a uniform grid restricts the candidates to the seed's threshold ball;
//   * a seed's cardinality is cached across rounds and only recomputed once one
//     of its members has been claimed by another cluster.
// All of these are exact: the results are bit-for-bit identical to the
// sequential version, including its tie-breaking.

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
#include <pthread.h>
#include <sched.h>
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

// Pin the calling OpenMP thread to one of the CPUs the process is allowed to
// run on. The algorithm is round based and therefore barrier heavy, and letting
// the scheduler migrate threads across the sockets between rounds costs far
// more than the expansions themselves. This is only done when the runtime was
// left unbound, so an explicit OMP_PROC_BIND / OMP_PLACES setting always wins.
static void bindThreadToCpu() {
#if defined(__linux__)
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const int ncpus = CPU_COUNT(&allowed);
    if (ncpus <= 0) return;

    // Linux enumerates one logical CPU per physical core first and only then
    // the SMT siblings, so taking the CPUs in order fills whole cores on the
    // first socket before spilling over -- the locality-friendly order here.
    const int wanted = omp_get_thread_num() % ncpus;

    cpu_set_t one;
    CPU_ZERO(&one);
    for (int cpu = 0, seen = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (seen++ == wanted) { CPU_SET(cpu, &one); break; }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
#endif
}

// Uniform bucket grid over the still-unclustered points.
//
// A candidate can only ever join the cluster seeded at s if its distance to s
// stays below the threshold (the running maximum over the members never
// decreases, and s is always a member). The grid cell size is therefore chosen
// >= threshold so that the 3x3 cell neighbourhood of the seed is a superset of
// that ball. Buckets are filled in ascending point-index order, which the
// candidate expansion relies on for tie-breaking.
struct Grid {
    double min_x = 0.0, min_y = 0.0, inv_cell = 1.0;
    int nx = 1, ny = 1;
    std::vector<int> start;  // nx*ny + 1 bucket base offsets into items
    std::vector<int> end;    // nx*ny current bucket ends (buckets shrink in place)
    std::vector<int> items;  // point indices, ascending within a bucket

    void configure(const std::vector<double>& px, const std::vector<double>& py,
                   const double threshold) {
        const int n = static_cast<int>(px.size());
        double max_x = px.empty() ? 0.0 : px[0];
        double max_y = py.empty() ? 0.0 : py[0];
        min_x = max_x;
        min_y = max_y;
        for (int i = 1; i < n; ++i) {
            min_x = std::min(min_x, px[i]);
            max_x = std::max(max_x, px[i]);
            min_y = std::min(min_y, py[i]);
            max_y = std::max(max_y, py[i]);
        }

        // Keep the bucket count proportional to the point count; growing the
        // cell size beyond the threshold stays correct, it only widens the ball.
        double cell = threshold;
        const long long budget = 4LL * n + 64;
        while (static_cast<long long>((max_x - min_x) / cell + 1.0) *
                   static_cast<long long>((max_y - min_y) / cell + 1.0) > budget) {
            cell *= 2.0;
        }
        inv_cell = 1.0 / cell;
        nx = static_cast<int>((max_x - min_x) * inv_cell) + 1;
        ny = static_cast<int>((max_y - min_y) * inv_cell) + 1;

        start.assign(static_cast<size_t>(nx) * ny + 1, 0);
        end.resize(static_cast<size_t>(nx) * ny);
        items.resize(n);
    }

    inline int cellX(const double x) const {
        const int c = static_cast<int>((x - min_x) * inv_cell);
        return std::min(std::max(c, 0), nx - 1);
    }
    inline int cellY(const double y) const {
        const int c = static_cast<int>((y - min_y) * inv_cell);
        return std::min(std::max(c, 0), ny - 1);
    }

    inline size_t cellOf(const int idx, const std::vector<double>& px,
                         const std::vector<double>& py) const {
        return static_cast<size_t>(cellY(py[idx])) * nx + cellX(px[idx]);
    }

    // Bucket the given point indices, which must be in ascending order.
    void build(const std::vector<int>& indices,
               const std::vector<double>& px, const std::vector<double>& py) {
        const size_t ncells = static_cast<size_t>(nx) * ny;
        std::fill(start.begin(), start.end(), 0);
        for (const int idx : indices) ++start[cellOf(idx, px, py) + 1];
        for (size_t c = 0; c < ncells; ++c) {
            start[c + 1] += start[c];
            end[c] = start[c];
        }
        for (const int idx : indices) items[end[cellOf(idx, px, py)]++] = idx;
    }

    // Delete a single point, keeping its bucket ascending and contiguous. Only
    // the winning cluster leaves the point set each round, so this is far
    // cheaper than rebuilding the whole grid.
    void erase(const int idx, const std::vector<double>& px, const std::vector<double>& py) {
        const size_t c = cellOf(idx, px, py);
        const int e = end[c];
        for (int k = start[c]; k < e; ++k) {
            if (items[k] == idx) {
                for (int m = k + 1; m < e; ++m) items[m - 1] = items[m];
                end[c] = e - 1;
                return;
            }
        }
    }
};

// Per-thread scratch space for one candidate-cluster expansion.
struct SeedScratch {
    std::vector<int> cidx;                  // global index of each live candidate
    std::vector<double> cx, cy;             // its coordinates (packed, contiguous)
    std::vector<double> md;                 // max distance to the current members
    std::vector<int> members;               // resulting cluster

    void reserve(const int n) {
        cidx.reserve(n); cx.reserve(n); cy.reserve(n); md.reserve(n);
        members.reserve(n);
    }
};

// Grow the candidate cluster seeded at seed_point, leaving its members in
// scratch.members. Returns the cardinality.
//
// This is the original O(members^2 * N) expansion rewritten to keep, for every
// live candidate, the running maximum distance to the members added so far.
// The maximum of a set is order independent and involves no accumulation, so
// the incremental values are bit-for-bit identical to the recomputed ones.
// Candidates whose running maximum reaches the threshold can never come back
// (the maximum is monotonically non-decreasing) and are compacted away, and
// candidates outside the seed's threshold ball are never enumerated at all.
static int growCluster(const int seed_point, const Grid& grid,
                       const std::vector<double>& px, const std::vector<double>& py,
                       const double threshold, SeedScratch& s) {
    s.members.clear();
    s.members.push_back(seed_point);

    const double sx = px[seed_point];
    const double sy = py[seed_point];

    // Collect the seed's threshold ball from the 3x3 cell neighbourhood.
    s.cidx.clear();
    const int ci = grid.cellX(sx);
    const int cj = grid.cellY(sy);
    const int j0 = std::max(cj - 1, 0), j1 = std::min(cj + 1, grid.ny - 1);
    const int i0 = std::max(ci - 1, 0), i1 = std::min(ci + 1, grid.nx - 1);
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const size_t cell = static_cast<size_t>(j) * grid.nx + i;
            const int cell_end = grid.end[cell];
            for (int k = grid.start[cell]; k < cell_end; ++k) {
                const int p = grid.items[k];
                if (p == seed_point) continue;
                const double dx = px[p] - sx;
                const double dy = py[p] - sy;
                if (std::sqrt(dx * dx + dy * dy) < threshold) s.cidx.push_back(p);
            }
        }
    }
    if (s.cidx.empty()) return 1;

    // The expansion scans candidates in ascending global index order and picks
    // the strictly smallest running maximum, exactly as the original did.
    std::sort(s.cidx.begin(), s.cidx.end());

    const int cand_count = static_cast<int>(s.cidx.size());
    s.cx.resize(cand_count);
    s.cy.resize(cand_count);
    s.md.resize(cand_count);
    int best = 0;
    double best_val = std::numeric_limits<double>::max();
    for (int i = 0; i < cand_count; ++i) {
        const int p = s.cidx[i];
        const double dx = px[p] - sx;
        const double dy = py[p] - sy;
        const double d = std::sqrt(dx * dx + dy * dy);
        s.cx[i] = px[p];
        s.cy[i] = py[p];
        s.md[i] = d;
        if (d < best_val) { best_val = d; best = i; }
    }

    int n = cand_count;
    int* __restrict cidx = s.cidx.data();
    double* __restrict cx = s.cx.data();
    double* __restrict cy = s.cy.data();
    double* __restrict md = s.md.data();

    while (n > 0) {
        s.members.push_back(cidx[best]);
        const double mx = cx[best];
        const double my = cy[best];

        // Update every remaining candidate against the freshly added member,
        // drop the ones that exceeded the threshold, and pick the next best.
        int w = 0;
        int next_best = -1;
        double next_val = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            if (i == best) continue;
            const double dx = cx[i] - mx;
            const double dy = cy[i] - my;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double m = std::max(md[i], d);
            if (m < threshold) {
                cidx[w] = cidx[i];
                cx[w] = cx[i];
                cy[w] = cy[i];
                md[w] = m;
                if (m < next_val) { next_val = m; next_best = w; }
                ++w;
            }
        }
        n = w;
        best = next_best;
    }

    return static_cast<int>(s.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Structure-of-arrays copy of the coordinates: the expansion kernel is
    // memory bound and benefits from unit-stride, vectorizable accesses.
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    Grid grid;
    grid.configure(px, py, threshold);
    grid.build(unclustered_indices, px, py);

    // Cardinality cache. Removing the winning cluster cannot change the
    // candidate cluster of a seed whose own members were left untouched: the
    // greedy expansion picks a minimum over the available points, and dropping
    // points it never picked leaves every one of its choices intact. So a seed
    // only has to be re-expanded once one of its members has been taken away,
    // which -- since every member is within the threshold of the seed -- can
    // only happen to seeds within the threshold of a removed point.
    std::vector<int> cached_card(N, -1);
    std::vector<char> cache_valid(N, 0);

    // Members of the cluster removed by the previous round, plus their bounding
    // box grown by the threshold: seeds outside it keep their cache for free.
    std::vector<int> removed;
    double rem_x0 = 1.0, rem_x1 = -1.0, rem_y0 = 1.0, rem_y1 = -1.0;

    std::vector<int> remaining;
    remaining.reserve(N);

    // One persistent parallel region for the whole run: the rounds are
    // separated by barriers instead of repeated team creation. The round
    // structure means two barriers per cluster found, so for small inputs the
    // team is trimmed to keep a worthwhile amount of work per thread; the
    // user's thread count always remains the upper bound.
    const int team_size = std::min(omp_get_max_threads(), std::max(1, N / 64));

    #pragma omp parallel num_threads(team_size)
    {
        bindThreadToCpu();

        SeedScratch scratch;
        scratch.reserve(N);

        // Main clustering loop
        while (true) {
            const int num_seeds = static_cast<int>(unclustered_indices.size());

            // Invalidate the seeds affected by the previous round's removal and
            // re-expand everything whose cached cardinality went stale. The
            // expansions only read shared state, so they are fully independent;
            // cluster sizes vary widely, hence the dynamic schedule.
            #pragma omp for schedule(nonmonotonic:dynamic, 4)
            for (int i = 0; i < num_seeds; ++i) {
                const int seed = unclustered_indices[i];
                const double sx = px[seed], sy = py[seed];
                if (cache_valid[seed] &&
                    sx >= rem_x0 && sx <= rem_x1 && sy >= rem_y0 && sy <= rem_y1) {
                    for (const int r : removed) {
                        const double dx = px[r] - sx, dy = py[r] - sy;
                        if (std::sqrt(dx * dx + dy * dy) < threshold) {
                            cache_valid[seed] = 0;
                            break;
                        }
                    }
                }
                if (cache_valid[seed]) continue;
                cached_card[seed] = growCluster(seed, grid, px, py, threshold, scratch);
                cache_valid[seed] = 1;
            }

            bool done;
            #pragma omp single copyprivate(done)
            {
                // Pick the largest candidate cluster; the sequential version
                // scans seeds in ascending index order and only replaces the
                // incumbent on a strictly larger cardinality, so ties go to the
                // lowest seed index.
                int max_cardinality = -1;
                int best_seed = -1;
                for (const int seed : unclustered_indices) {
                    if (cached_card[seed] > max_cardinality) {
                        max_cardinality = cached_card[seed];
                        best_seed = seed;
                    }
                }

                // If we found a cluster, add it
                if (best_seed >= 0 && max_cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    // Re-expand the winner to recover its member list (cheap:
                    // one expansion out of the whole round).
                    growCluster(best_seed, grid, px, py, threshold, scratch);
                    cluster.members = scratch.members;

                    // Drop the newly clustered points from the unclustered list
                    std::sort(cluster.members.begin(), cluster.members.end());
                    remaining.clear();
                    std::set_difference(unclustered_indices.begin(), unclustered_indices.end(),
                                        cluster.members.begin(), cluster.members.end(),
                                        std::back_inserter(remaining));
                    unclustered_indices.swap(remaining);

                    done = unclustered_indices.empty();
                    if (!done) {
                        // Take the winner out of the grid and publish its points
                        // (and their threshold-grown bounding box) so the next
                        // round can invalidate the affected caches in parallel.
                        rem_x0 = rem_y0 = std::numeric_limits<double>::max();
                        rem_x1 = rem_y1 = -std::numeric_limits<double>::max();
                        for (const int r : cluster.members) {
                            grid.erase(r, px, py);
                            rem_x0 = std::min(rem_x0, px[r]);
                            rem_x1 = std::max(rem_x1, px[r]);
                            rem_y0 = std::min(rem_y0, py[r]);
                            rem_y1 = std::max(rem_y1, py[r]);
                        }
                        rem_x0 -= threshold; rem_x1 += threshold;
                        rem_y0 -= threshold; rem_y1 += threshold;
                        removed = cluster.members;
                    }

                    clusters.push_back(std::move(cluster));
                } else {
                    // No more clusters can be formed
                    done = true;
                }
            }
            if (done) break;
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

    // Check diameter (max distance between any two points) of every cluster.
    // The clusters are independent, so this all-pairs pass runs in parallel and
    // the (ordered) reporting happens afterwards.
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

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
    for (size_t c = 0; c < clusters.size(); ++c) {
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

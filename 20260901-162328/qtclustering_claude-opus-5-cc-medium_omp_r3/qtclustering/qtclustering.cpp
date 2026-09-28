// QT Clustering Benchmark - OpenMP Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: the search over all candidate seeds in a clustering round is
// embarrassingly parallel, so it is distributed over OpenMP threads with a
// dynamic schedule (per-seed cost varies strongly). A single parallel region
// spans the whole run, with the rounds separated by barriers. Ties are resolved
// deterministically (lowest seed index wins), matching the sequential order, so
// the output is bit-identical to the reference for any thread count.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>
#include <unistd.h>

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

// Per-thread scratch space for candidate cluster generation.
// The candidate set is kept as a structure of arrays so the hot update loop
// streams contiguous memory instead of gathering through point indices.
struct Scratch {
    std::vector<int> idx;      // original point index of each live candidate
    std::vector<double> cx;    // candidate coordinates
    std::vector<double> cy;
    std::vector<double> md;    // squared max distance from candidate to cluster members

    // Allocated lazily by the owning thread, so only threads that actually run
    // work pay for it and the pages are first-touched where they are used.
    void ensure(int n) {
        if (static_cast<int>(idx.size()) >= n) return;
        idx.resize(n);
        cx.resize(n);
        cy.resize(n);
        md.resize(n);
    }
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster; members are written to `out`.
//
// This is the same greedy as the reference implementation, but instead of
// recomputing the distance from every candidate to every member on each step it
// keeps a running maximum per candidate (max is exact in floating point, so the
// selected points are identical). Candidates whose running maximum reaches the
// threshold can never be admitted later, so they are dropped from the working
// set, which shrinks the per-step scan dramatically. Comparisons are done on
// squared distances, which is order-equivalent to comparing distances.
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<char>& clustered,
                                    const Point* points,
                                    const double threshold_sq,
                                    const int point_count,
                                    Scratch& s,
                                    std::vector<int>& out) {
    const double sx = points[seed_point].x;
    const double sy = points[seed_point].y;

    int* __restrict__ idx = s.idx.data();
    double* __restrict__ cx = s.cx.data();
    double* __restrict__ cy = s.cy.data();
    double* __restrict__ md = s.md.data();

    // Seed the working set with every unclustered point within the threshold of
    // the seed, tracking the closest one (lowest index wins ties, as in the
    // sequential ascending scan).
    int live = 0;
    int best_pos = -1;
    double best_d = std::numeric_limits<double>::max();
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed_point) continue;
        const double dx = points[c].x - sx;
        const double dy = points[c].y - sy;
        const double d = dx * dx + dy * dy;
        if (d < threshold_sq) {
            idx[live] = c;
            cx[live] = points[c].x;
            cy[live] = points[c].y;
            md[live] = d;
            if (d < best_d) {
                best_d = d;
                best_pos = live;
            }
            ++live;
        }
    }

    out.clear();
    out.push_back(seed_point);

    // Iteratively absorb the candidate with the smallest cluster diameter.
    while (best_pos >= 0) {
        const int member = idx[best_pos];
        const double mx = cx[best_pos];
        const double my = cy[best_pos];
        out.push_back(member);

        // Single fused pass: update running maxima, compact out candidates that
        // can no longer be admitted, and pick the next best candidate.
        // Compaction is stable, so the array stays sorted by point index and
        // ties keep resolving to the lowest index.
        int w = 0;
        int next_pos = -1;
        double next_d = std::numeric_limits<double>::max();
        for (int i = 0; i < live; ++i) {
            if (i == best_pos) continue;
            const double dx = cx[i] - mx;
            const double dy = cy[i] - my;
            const double d = dx * dx + dy * dy;
            const double nd = md[i] > d ? md[i] : d;
            if (nd < threshold_sq) {
                idx[w] = idx[i];
                cx[w] = cx[i];
                cy[w] = cy[i];
                md[w] = nd;
                if (nd < next_d) {
                    next_d = nd;
                    next_pos = w;
                }
                ++w;
            }
        }
        live = w;
        best_pos = next_pos;
    }

    return static_cast<int>(out.size());
}

// Number of physical cores visible on this machine, or 0 if the topology
// cannot be read. The candidate search is a tight floating-point kernel that
// saturates a core on its own, so SMT siblings add no throughput; worse,
// occupying every logical CPU leaves nothing for the OS and makes the OpenMP
// spin barriers between rounds collapse (measured >10x slowdown).
static int physicalCoreCount() {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) return 0;

    // Hardware threads sharing one core, e.g. "0,128" or "0-1".
    std::ifstream f("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    std::string list;
    if (!f || !std::getline(f, list)) return 0;

    int siblings = 0;
    size_t pos = 0;
    while (pos < list.size()) {
        const size_t comma = list.find(',', pos);
        const std::string range = list.substr(pos, comma - pos);
        const size_t dash = range.find('-');
        if (dash == std::string::npos) {
            siblings += 1;
        } else {
            siblings += atoi(range.c_str() + dash + 1) - atoi(range.c_str()) + 1;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    if (siblings < 1) return 0;
    return static_cast<int>(online / siblings);
}

// Resolved once at start-up so that reading the topology never lands inside the
// timed region.
static const int g_physical_cores = physicalCoreCount();

// Team size for the clustering run: never more threads than there is work to
// keep them busy, and never more than one thread per physical core.
static int clusteringThreads(const int N) {
    int threads = omp_get_max_threads();
    if (g_physical_cores > 0 && threads > g_physical_cores) threads = g_physical_cores;

    // Every round sweeps all points once per seed, so N*N is a lower bound on
    // the work; below a few hundred microseconds per thread the barriers cost
    // more than the work they synchronise.
    const long long affordable = (static_cast<long long>(N) * N) >> 18;
    if (threads > affordable) threads = static_cast<int>(affordable);
    return threads < 1 ? 1 : threads;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;
    const Point* pts = points.data();

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    // Candidate cluster of each seed, cached across rounds. A seed's candidate
    // cluster is a deterministic function of the seed and the set of available
    // points; removing points that the greedy never selected cannot change any
    // of its choices. So a cached result only becomes stale when one of its own
    // members gets clustered.
    std::vector<std::vector<int>> cached_members(N);
    std::vector<char> stale(N, 1);

    std::vector<Scratch> scratch(omp_get_max_threads());
    std::vector<char> newly_clustered(N, 0);
    std::vector<int> todo;
    todo.reserve(N);

    // One parallel region covers the whole clustering run: the team is created
    // once and the rounds are separated by barriers, instead of forking a fresh
    // team (and possibly resizing the runtime's thread pool) hundreds of times.
    int num_seeds = 0;
    const int* seeds = nullptr;
    int num_todo = 0;
    int max_cardinality = -1;
    int best_seed = -1;
    double wx = 0.0, wy = 0.0;
    bool done = false;
    const double reach_sq = 4.0 * threshold_sq;
    std::vector<int> winners;

    #pragma omp parallel num_threads(clusteringThreads(N))
    {
        Scratch& s = scratch[omp_get_thread_num()];

        // Main clustering loop
        while (true) {
            #pragma omp single
            {
                num_seeds = static_cast<int>(unclustered_indices.size());
                seeds = unclustered_indices.data();

                // Seeds whose cached candidate cluster has to be (re)built.
                // Keeping them in their own compact list means the expensive
                // dynamically scheduled loop below iterates over real work only,
                // instead of hammering the shared loop counter for thousands of
                // cache hits every round.
                todo.clear();
                for (int i = 0; i < num_seeds; ++i) {
                    if (stale[seeds[i]]) todo.push_back(seeds[i]);
                }
                num_todo = static_cast<int>(todo.size());
            }

            // Build the missing candidate clusters. Per-seed cost varies by
            // orders of magnitude, hence the fine-grained dynamic schedule.
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < num_todo; ++i) {
                const int seed = todo[i];
                s.ensure(N); // lazy: only threads that get work allocate
                generateCandidateCluster(seed, clustered, pts, threshold_sq,
                                         N, s, cached_members[seed]);
                stale[seed] = 0;
            }

            #pragma omp single
            {
                // Pick the largest candidate cluster; lowest seed index wins
                // ties, matching the sequential ascending scan. This is a cheap
                // scan over already-computed sizes.
                max_cardinality = -1;
                best_seed = -1;
                for (int i = 0; i < num_seeds; ++i) {
                    const int seed = seeds[i];
                    const int cardinality = static_cast<int>(cached_members[seed].size());
                    if (cardinality > max_cardinality) {
                        max_cardinality = cardinality;
                        best_seed = seed;
                    }
                }

                if (best_seed < 0 || max_cardinality <= 0) {
                    done = true; // No more clusters can be formed
                } else {
                    winners = cached_members[best_seed];

                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = winners;
                    clusters.push_back(std::move(cluster));

                    for (size_t i = 0; i < winners.size(); ++i) {
                        clustered[winners[i]] = 1;
                        newly_clustered[winners[i]] = 1;
                    }
                    wx = pts[best_seed].x;
                    wy = pts[best_seed].y;
                }
            }

            if (done) break;

            // Invalidate every cached candidate cluster that contains a point
            // which just became unavailable. Every member of a candidate cluster
            // lies within `threshold` of its seed, so two clusters whose seeds
            // are at least 2*threshold apart cannot share a point and can be
            // skipped outright.
            #pragma omp for schedule(static)
            for (int i = 0; i < num_seeds; ++i) {
                const int seed = seeds[i];
                if (clustered[seed] || stale[seed]) continue;
                const double dx = pts[seed].x - wx;
                const double dy = pts[seed].y - wy;
                if (dx * dx + dy * dy >= reach_sq) continue;
                const std::vector<int>& m = cached_members[seed];
                for (size_t k = 0; k < m.size(); ++k) {
                    if (newly_clustered[m[k]]) {
                        stale[seed] = 1;
                        break;
                    }
                }
            }

            #pragma omp single
            {
                // Remove clustered points from the unclustered list, releasing
                // their caches
                for (size_t i = 0; i < winners.size(); ++i) {
                    newly_clustered[winners[i]] = 0;
                    std::vector<int>().swap(cached_members[winners[i]]);
                }

                unclustered_indices.erase(
                    std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                  [&clustered](int idx) { return clustered[idx] != 0; }),
                    unclustered_indices.end()
                );

                if (unclustered_indices.empty()) done = true;
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

    // Compute cluster diameters in parallel, then report them in order
    const int num_clusters = static_cast<int>(clusters.size());
    std::vector<double> diameters(num_clusters, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < num_clusters; ++c) {
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

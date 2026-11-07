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
#include <thread>
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Per-thread scratch buffers, allocated once by the main thread before the
// parallel region (no heap allocation happens inside the clustering loop).
struct Scratch {
    std::vector<int> candidates;     // unclustered points still eligible (ascending index order)
    std::vector<double> max_dist;    // max distance from each point to the current cluster members
    std::vector<int> members;        // members of the cluster currently being grown
    std::vector<char> clustered;     // private copy of the bookkeeping (identical in all threads)
    std::vector<int> unclustered;    // private copy, ascending order
    std::vector<int> order;          // seed processing order for the current iteration
    std::vector<int> best_members[2];// local best cluster of the iteration (by parity)
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster; members are left in scratch.members.
// Returns -1 if the cluster is abandoned because it provably cannot reach
// min_required points (it could then never be selected as the best cluster);
// upper_bound receives the last known upper bound on (or the exact) size.
//
// Semantics are identical to the original formulation (repeatedly add the
// unclustered point whose maximum distance to the current members is smallest
// and below the threshold, breaking ties by lowest point index).  Instead of
// recomputing all member distances for every candidate at every step, the
// per-candidate maximum distance is maintained incrementally (max is exact
// regardless of evaluation order), and candidates whose maximum distance has
// already reached the threshold are dropped since it can only grow.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered_indices,
                             const Point* __restrict points,
                             const double threshold,
                             const int min_required,
                             Scratch& s,
                             int& upper_bound) {
    std::vector<int>& cand = s.candidates;
    std::vector<double>& md = s.max_dist;
    std::vector<int>& members = s.members;

    members.clear();
    members.push_back(seed_point);

    // Initial candidate set: all unclustered points except the seed, with their
    // distance to the seed; prune those already at/over the threshold.
    cand.clear();
    const Point ps = points[seed_point];
    for (size_t i = 0; i < unclustered_indices.size(); ++i) {
        const int c = unclustered_indices[i];
        if (c == seed_point) continue;
        const double d = distance(points[c], ps);
        if (d < threshold) {
            md[c] = d;
            cand.push_back(c);
        }
    }

    while (!cand.empty()) {
        // Upper bound on the final size: everything still eligible gets added.
        upper_bound = static_cast<int>(members.size() + cand.size());
        if (upper_bound < min_required) {
            return -1;
        }

        // Find the candidate with the smallest max distance (first index wins ties)
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (size_t i = 0; i < cand.size(); ++i) {
            const int c = cand[i];
            const double d = md[c];
            if (d < min_diameter) {
                min_diameter = d;
                closest = c;
            }
        }
        if (closest < 0) break;

        members.push_back(closest);

        // Update max distances w.r.t. the newly added member and compact the
        // candidate list, removing the chosen point and those now >= threshold.
        const Point pn = points[closest];
        size_t w = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            const int c = cand[i];
            if (c == closest) continue;
            const double d = distance(points[c], pn);
            const double m = (d > md[c]) ? d : md[c];
            if (m < threshold) {
                md[c] = m;
                cand[w++] = c;
            }
        }
        cand.resize(w);
    }

    const int card = static_cast<int>(members.size());
    upper_bound = card;
    return (card < min_required) ? -1 : card;
}

// Per-thread result slot, padded to its own cache line to avoid false sharing
struct alignas(64) ThreadResult {
    int cardinality = -1;
    int seed = -1;
    const std::vector<int>* members = nullptr;   // owning thread's buffer
};

struct alignas(64) PaddedAtomic {
    std::atomic<int> value{0};
};

// Sense-reversing spin barrier for the threads of one parallel region.  The
// clustering loop needs one barrier per cluster found; with many threads
// (e.g. 256 across two sockets) the runtime barrier may put waiting threads to
// sleep and the wake-up latency then dominates the short per-iteration work.
// Spinning keeps that latency in the microsecond range independent of runtime
// tuning.  The barrier needs no initialization beyond zeroing, so no
// synchronized setup inside the region is required.
class SpinBarrier {
  public:
    void wait(int& local_sense, const int nthreads) {
        local_sense ^= 1;
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
            arrived_.store(0, std::memory_order_relaxed);
            sense_.store(local_sense, std::memory_order_release);
        } else {
            int spins = 0;
            while (sense_.load(std::memory_order_acquire) != local_sense) {
                if (++spins > 4096) {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
        }
    }

  private:
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<int> sense_{0};
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    const Point* pts = points.data();

    // Each outer iteration selects the largest candidate cluster (ties broken
    // by lowest seed index, matching the sequential first-wins order), removes
    // its members, and repeats.  The candidate clusters of one iteration are
    // evaluated in parallel across seeds.
    //
    // Only one barrier per iteration is needed: every thread publishes its
    // local best in a per-thread slot; after the barrier all threads
    // redundantly determine the winner and update a private copy of the
    // bookkeeping (cheap, O(N)).  No critical sections or single regions.
    //
    // All cross-iteration shared state (result slots, member buffers, pruning
    // bound, size estimates) is double-buffered by iteration parity: a buffer
    // used in iteration k is next written in iteration k+2, which no thread
    // can reach before every thread has passed barrier k+1, i.e. finished
    // reading it.
    //
    // Seeds are processed in order of decreasing estimated cluster size (the
    // size, or the pruning bound, obtained in the previous iteration; the
    // neighbour count within the threshold for the first iteration).  This
    // makes the best cardinality known early so most seeds are pruned, and
    // starts the expensive seeds first for better load balance.  Ordering does
    // not affect the result: the winner is chosen by (size, lowest index).
    const int max_threads = omp_get_max_threads();
    std::vector<ThreadResult> results[2] = {
        std::vector<ThreadResult>(max_threads), std::vector<ThreadResult>(max_threads)};
    PaddedAtomic global_best[2];   // best cardinality seen this iteration (pruning only)
    std::vector<int> estimate[2] = {std::vector<int>(N, 0), std::vector<int>(N, 0)};
    SpinBarrier barrier;

    // All per-thread buffers are owned here, outside the parallel region: they
    // outlive it (a winner's member list is still read by slower threads in
    // the final iteration) and no thread ever allocates inside the loop.
    std::vector<Scratch> scratch(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        Scratch& s = scratch[t];
        s.candidates.reserve(N);
        s.max_dist.resize(N);
        s.members.reserve(N);
        s.clustered.assign(N, 0);
        s.unclustered.resize(N);
        for (int i = 0; i < N; ++i) s.unclustered[i] = i;
        s.order.reserve(N);
        s.best_members[0].reserve(N);
        s.best_members[1].reserve(N);
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        int sense = 0;

        Scratch& s = scratch[tid];
        std::vector<char>& clustered = s.clustered;
        std::vector<int>& unclustered_indices = s.unclustered;
        std::vector<int>& order = s.order;

        // Initial estimate: 1 + number of other points within the threshold
        {
            int* est0 = estimate[0].data();
            #pragma omp for schedule(dynamic, 16) nowait
            for (int i = 0; i < N; ++i) {
                const Point pi = pts[i];
                int cnt = 0;
                for (int j = 0; j < N; ++j) {
                    cnt += (distance(pts[j], pi) < threshold) ? 1 : 0;
                }
                est0[i] = cnt;   // includes i itself
            }
            barrier.wait(sense, nthreads);
        }

        for (int iter = 0; !unclustered_indices.empty(); ++iter) {
            const int parity = iter & 1;
            const int num_seeds = static_cast<int>(unclustered_indices.size());
            std::vector<int>& my_members = s.best_members[parity];
            std::atomic<int>& bound = global_best[parity].value;
            const int* est = estimate[parity].data();
            int* est_next = estimate[parity ^ 1].data();

            // Processing order: decreasing estimate, then increasing index
            order.assign(unclustered_indices.begin(), unclustered_indices.end());
            std::sort(order.begin(), order.end(), [est](int a, int b) {
                return (est[a] != est[b]) ? (est[a] > est[b]) : (a < b);
            });

            int local_max = -1;
            int local_seed = -1;

            // Try each unclustered point as a seed, in parallel.
            #pragma omp for schedule(dynamic, 1) nowait
            for (int i = 0; i < num_seeds; ++i) {
                const int seed = order[i];

                // A seed can only displace the best if it reaches at least the
                // best cardinality (a tie may win on the lower index).  Reading a
                // stale/lower value only weakens the prune, never the result.
                int required = bound.load(std::memory_order_relaxed);

                int upper = 0;
                const int cardinality = generateCandidateCluster(seed, unclustered_indices,
                                                                 pts, threshold, required, s,
                                                                 upper);
                est_next[seed] = (cardinality < 0) ? upper : cardinality;
                if (cardinality < 0) continue;

                if (cardinality > local_max ||
                    (cardinality == local_max && seed < local_seed)) {
                    local_max = cardinality;
                    local_seed = seed;
                    my_members = s.members;
                    if (cardinality > required) {
                        // Monotonic max update (compare-and-swap loop).
                        int cur = required;
                        while (cardinality > cur &&
                               !bound.compare_exchange_weak(cur, cardinality,
                                                            std::memory_order_relaxed)) {
                        }
                    }
                }
            }

            ThreadResult& mine = results[parity][tid];
            mine.cardinality = local_max;
            mine.seed = local_seed;
            mine.members = &my_members;

            barrier.wait(sense, nthreads);

            // Every thread determines the winner of this iteration
            int max_cardinality = -1;
            int best_seed = -1;
            const std::vector<int>* best_cluster_members = nullptr;
            const ThreadResult* res = results[parity].data();
            for (int t = 0; t < nthreads; ++t) {
                const ThreadResult& r = res[t];
                if (r.cardinality > max_cardinality ||
                    (r.cardinality == max_cardinality && r.seed >= 0 && r.seed < best_seed)) {
                    max_cardinality = r.cardinality;
                    best_seed = r.seed;
                    best_cluster_members = r.members;
                }
            }

            if (!(best_seed >= 0 && max_cardinality > 0)) {
                break; // No more clusters can be formed (same decision in all threads)
            }

            if (tid == 0) {
                // Record the cluster
                Cluster cluster;
                cluster.seed_point = best_seed;
                cluster.members = *best_cluster_members;
                clusters.push_back(std::move(cluster));

                // Reset this parity's pruning bound for iteration iter+2.  All
                // reads of it (iteration iter) completed before the barrier
                // above, and iteration iter+2 cannot start anywhere before this
                // thread reaches barrier iter+1.
                bound.store(0, std::memory_order_relaxed);
            }

            // Update private bookkeeping: mark members as clustered and remove
            // them from the (ascending) unclustered list.
            for (size_t i = 0; i < best_cluster_members->size(); ++i) {
                clustered[(*best_cluster_members)[i]] = 1;
            }
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
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
        const int msize = static_cast<int>(cluster.members.size());
        #pragma omp parallel for schedule(dynamic, 16) reduction(max : max_diameter)
        for (int i = 0; i < msize; ++i) {
            for (int j = i + 1; j < msize; ++j) {
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
    
    // Spin up the OpenMP thread pool once so that one-time runtime thread
    // creation is not attributed to the clustering itself.
    #pragma omp parallel
    {
        volatile int warm = omp_get_thread_num();
        (void)warm;
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

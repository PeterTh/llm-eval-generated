// QT Clustering Benchmark - Simplified Sequential Version
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
#include <omp.h>
#include <dirent.h>
#include <vector>

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

// Per-thread scratch buffers for candidate cluster generation.
// The candidate set is kept in a structure-of-arrays layout (ids + coordinates +
// running maximum distance to the current cluster members) so that the hot loops
// are contiguous and vectorizable.
struct ClusterScratch {
    std::vector<int> cand_id;
    std::vector<double> cand_x;
    std::vector<double> cand_y;
    std::vector<double> cand_maxd;  // max distance from candidate to current members
    std::vector<int> members;

    void reserve(const int n) {
        cand_id.reserve(n);
        cand_x.reserve(n);
        cand_y.reserve(n);
        cand_maxd.reserve(n);
        members.reserve(n);
    }
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster.
//
// This is the incremental formulation of the original O(N * k^2) search: instead
// of recomputing the distance from every candidate to every cluster member in
// each step, the running maximum distance per candidate is updated when a member
// is added. Since the maximum of a set of values is order independent, the values
// compared against the threshold are bit-for-bit identical to the original code.
// Candidates whose running maximum reaches the threshold can never become
// eligible again (the maximum only grows), so they are dropped permanently. The
// candidate array stays sorted by point index, which preserves the original
// "first strictly smaller diameter wins" tie-breaking.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int point_count,
                             ClusterScratch& scratch,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<int>& cand_id = scratch.cand_id;
    std::vector<double>& cand_x = scratch.cand_x;
    std::vector<double>& cand_y = scratch.cand_y;
    std::vector<double>& cand_maxd = scratch.cand_maxd;
    std::vector<int>& members = scratch.members;

    cand_id.clear();
    cand_x.clear();
    cand_y.clear();
    cand_maxd.clear();
    members.clear();

    members.push_back(seed_point);

    // Only points closer than the threshold to the seed can ever join the
    // cluster, since the seed is always a member.
    const double seed_x = points[seed_point].x;
    const double seed_y = points[seed_point].y;
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed_point) continue;
        const double x = points[c].x;
        const double y = points[c].y;
        const double dx = x - seed_x;
        const double dy = y - seed_y;
        const double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < threshold) {
            cand_id.push_back(c);
            cand_x.push_back(x);
            cand_y.push_back(y);
            cand_maxd.push_back(dist);
        }
    }

    // Iteratively add the closest point that keeps the diameter below threshold
    while (!cand_id.empty()) {
        const int n = static_cast<int>(cand_id.size());

        int best_pos = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            const double d = cand_maxd[i];
            if (d < min_diameter) {
                min_diameter = d;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;  // no eligible candidate left

        const int chosen = cand_id[best_pos];
        const double px = cand_x[best_pos];
        const double py = cand_y[best_pos];
        members.push_back(chosen);

        // Update running maxima against the new member, dropping the chosen
        // point and everything that can no longer stay below the threshold.
        int w = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = cand_x[i] - px;
            const double dy = cand_y[i] - py;
            const double dist = std::sqrt(dx * dx + dy * dy);
            const double nd = std::max(cand_maxd[i], dist);
            if (nd >= threshold || i == best_pos) continue;
            cand_id[w] = cand_id[i];
            cand_x[w] = cand_x[i];
            cand_y[w] = cand_y[i];
            cand_maxd[w] = nd;
            ++w;
        }
        cand_id.resize(w);
        cand_x.resize(w);
        cand_y.resize(w);
        cand_maxd.resize(w);
    }

    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Number of physical cores, derived from the CPU topology (0 if unavailable).
// This workload is compute bound and synchronizes at a barrier once per cluster
// round; running a second thread on each SMT sibling adds no throughput but
// makes every barrier vastly more expensive, so the thread count is capped at
// one thread per core.
static int detectPhysicalCores() {
    DIR* dir = opendir("/sys/devices/system/cpu");
    if (!dir) return 0;

    int cores = 0;
    while (const struct dirent* entry = readdir(dir)) {
        int cpu = -1;
        if (sscanf(entry->d_name, "cpu%d", &cpu) != 1 || cpu < 0) continue;

        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE* file = fopen(path, "r");
        if (!file) continue;

        // The first sibling in the list represents the core; count it once.
        int first = -1;
        if (fscanf(file, "%d", &first) == 1 && first == cpu) ++cores;
        fclose(file);
    }

    closedir(dir);
    return cores;
}

// Number of threads to use: the OpenMP default (honouring OMP_NUM_THREADS),
// never more than one per physical core.
static int chooseThreadCount() {
    int threads = omp_get_max_threads();
    const int cores = detectPhysicalCores();
    if (cores > 0 && cores < threads) threads = cores;
    return threads;
}

// Per-thread reduction slot, padded to a cache line to avoid false sharing.
struct alignas(64) BestCandidate {
    int cardinality;
    int seed;
    char pad[64 - 2 * sizeof(int)];
};

// Main QT clustering algorithm
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

    const int num_threads = chooseThreadCount();
    std::vector<BestCandidate> best(num_threads);
    std::vector<std::vector<int>> best_members(num_threads);

    int num_seeds = N;
    bool finished = false;

    // One persistent parallel region spans the whole clustering loop: the region
    // is entered once instead of once per round, and the per-thread scratch
    // buffers are allocated only once.
    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        ClusterScratch scratch;
        scratch.reserve(N);
        std::vector<int>& my_members = best_members[tid];
        my_members.reserve(N);

        while (true) {
            if (finished) break;

            int local_cardinality = -1;
            int local_seed = -1;
            my_members.clear();

            // Try each unclustered point as a seed. The candidate clusters are
            // fully independent of each other, which is the main source of
            // parallelism. Reduction rule: highest cardinality wins, ties go to
            // the lowest point index, which matches the sequential scan over the
            // ascending seed list.
            #pragma omp for schedule(dynamic, 8) nowait
            for (int i = 0; i < num_seeds; ++i) {
                const int seed = unclustered_indices[i];

                const int cardinality = generateCandidateCluster(
                    seed, clustered, points, threshold, N, scratch);

                if (cardinality > local_cardinality ||
                    (cardinality == local_cardinality && seed < local_seed)) {
                    local_cardinality = cardinality;
                    local_seed = seed;
                    my_members.assign(scratch.members.begin(), scratch.members.end());
                }
            }

            best[tid].cardinality = local_cardinality;
            best[tid].seed = local_seed;

            #pragma omp barrier

            #pragma omp single
            {
                // Sequential reduction over the per-thread winners.
                int max_cardinality = -1;
                int best_seed = -1;
                int best_thread = -1;
                for (int t = 0; t < num_threads; ++t) {
                    if (best[t].seed < 0) continue;
                    if (best[t].cardinality > max_cardinality ||
                        (best[t].cardinality == max_cardinality && best[t].seed < best_seed)) {
                        max_cardinality = best[t].cardinality;
                        best_seed = best[t].seed;
                        best_thread = t;
                    }
                }

                if (best_seed < 0 || max_cardinality <= 0) {
                    // No more clusters can be formed
                    finished = true;
                } else if (max_cardinality == 1) {
                    // Every remaining point forms a singleton cluster: no point
                    // can be paired with another one, and removing points never
                    // enables new pairings. The sequential algorithm would emit
                    // them one by one in ascending index order, so do that
                    // directly.
                    for (int i = 0; i < num_seeds; ++i) {
                        Cluster cluster;
                        cluster.seed_point = unclustered_indices[i];
                        cluster.members.push_back(unclustered_indices[i]);
                        clusters.push_back(cluster);
                        clustered[unclustered_indices[i]] = 1;
                    }
                    unclustered_indices.clear();
                    finished = true;
                } else {
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = best_members[best_thread];
                    clusters.push_back(cluster);

                    // Mark all members as clustered
                    for (size_t i = 0; i < cluster.members.size(); ++i) {
                        clustered[cluster.members[i]] = 1;
                    }

                    // Remove clustered points from unclustered list
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                      [&clustered](int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end()
                    );

                    num_seeds = static_cast<int>(unclustered_indices.size());
                    if (num_seeds == 0) finished = true;
                }
            } // implicit barrier publishes finished / num_seeds / unclustered_indices
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
    
    // Compute all cluster diameters in parallel, then report them in order so
    // that the output stays identical to the sequential version.
    const int num_clusters = static_cast<int>(clusters.size());
    std::vector<double> diameters(num_clusters, 0.0);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(chooseThreadCount())
    for (int c = 0; c < num_clusters; ++c) {
        const auto& cluster = clusters[c];
        const int members = static_cast<int>(cluster.members.size());
        double max_diameter = 0.0;

        // Check diameter (max distance between any two points)
        for (int i = 0; i < members; ++i) {
            const Point& pi = points[cluster.members[i]];
            for (int j = i + 1; j < members; ++j) {
                const double dist = distance(pi, points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    for (int c = 0; c < num_clusters; ++c) {
        const double max_diameter = diameters[c];

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, clusters[c].members.size(), clusters[c].seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n", 
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

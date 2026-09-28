// QT Clustering Benchmark - OpenMP Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: candidate clusters for all unclustered seed points are
// generated independently in parallel; the reduction picks the largest
// cluster with the same tie-breaking order as the sequential version.

#include <algorithm>
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Per-thread scratch buffers reused across seeds to avoid repeated allocation
struct SeedScratch {
    std::vector<double> max_dist; // max distance from candidate to cluster members
    std::vector<char> excluded;   // candidate can no longer join this cluster
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
//
// Instead of recomputing the max distance from each candidate to every
// cluster member on each iteration, the per-candidate max distance is
// updated incrementally with the distance to the most recently added
// member. Since the max distance only grows, candidates at or beyond the
// threshold are excluded permanently. This yields the exact same cluster
// as the original quadratic scan.
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              SeedScratch& scratch,
                              std::vector<int>& members) {
    std::vector<double>& max_dist = scratch.max_dist;
    std::vector<char>& excluded = scratch.excluded;
    max_dist.assign(point_count, 0.0);
    excluded.assign(point_count, 0);

    for (int i = 0; i < point_count; ++i) {
        if (clustered[i]) excluded[i] = 1;
    }
    excluded[seed_point] = 1;

    members.clear();
    members.push_back(seed_point);

    int last_added = seed_point;

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        const Point last = points[last_added];
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (excluded[candidate]) continue;

            // Update max distance with the distance to the newest member
            const double dist = distance(points[candidate], last);
            double md = max_dist[candidate];
            if (dist > md) md = dist;
            max_dist[candidate] = md;

            if (md >= threshold) {
                excluded[candidate] = 1; // max distance only grows
                continue;
            }
            if (md < min_diameter) {
                min_diameter = md;
                closest = candidate;
            }
        }

        if (closest < 0) break; // No more points can be added

        excluded[closest] = 1;
        members.push_back(closest);
        last_added = closest;
    }

    return static_cast<int>(members.size());
}

// Number of SMT siblings per physical core (Linux), or 1 if unknown
static int smtSiblingsPerCore() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!f) return 1;
    char buf[256];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return 1;
    // The list is comma-separated with ranges, e.g. "0,128" or "0-1"
    int siblings = 0;
    for (char* p = buf; *p && *p != '\n';) {
        char* end;
        const long a = strtol(p, &end, 10);
        if (end == p) break;
        p = end;
        if (*p == '-') {
            const long b = strtol(p + 1, &p, 10);
            siblings += static_cast<int>(b - a + 1);
        } else {
            siblings += 1;
        }
        if (*p == ',') ++p;
    }
    return siblings > 0 ? siblings : 1;
}

// Default team size: one thread per physical core. The clustering kernel is
// compute-bound, so SMT siblings only contend with each other. An explicit
// OMP_NUM_THREADS setting is respected as-is.
static int defaultTeamSize() {
    const int max_threads = omp_get_max_threads();
    if (getenv("OMP_NUM_THREADS") != nullptr) return max_threads;
    const int siblings = smtSiblingsPerCore();
    if (siblings > 1 && max_threads == omp_get_num_procs()) {
        return std::max(1, max_threads / siblings);
    }
    return max_threads;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    const int team_size = defaultTeamSize();

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const size_t num_seeds = unclustered_indices.size();
        int max_cardinality = -1;
        size_t best_pos = num_seeds; // position in unclustered_indices
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        // Try each unclustered point as a seed, in parallel. Candidate
        // clusters only read the shared `clustered` state, so all seeds are
        // independent. Ties in cardinality are broken by the smallest seed
        // position, matching the sequential iteration order.
        const int num_threads = static_cast<int>(
            std::min<size_t>(team_size, num_seeds));
        #pragma omp parallel num_threads(num_threads)
        {
            SeedScratch scratch;
            std::vector<int> candidate_members;
            int local_max = -1;
            size_t local_pos = num_seeds;
            int local_seed = -1;
            std::vector<int> local_members;

            #pragma omp for schedule(dynamic) nowait
            for (size_t i = 0; i < num_seeds; ++i) {
                const int seed = unclustered_indices[i];

                const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                           threshold, N,
                                                           scratch, candidate_members);

                if (cardinality > local_max ||
                    (cardinality == local_max && i < local_pos)) {
                    local_max = cardinality;
                    local_pos = i;
                    local_seed = seed;
                    local_members.swap(candidate_members);
                }
            }

            #pragma omp critical
            {
                if (local_max > max_cardinality ||
                    (local_max == max_cardinality && local_pos < best_pos)) {
                    max_cardinality = local_max;
                    best_pos = local_pos;
                    best_seed = local_seed;
                    best_cluster_members.swap(local_members);
                }
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
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

// QT Clustering Benchmark - OpenMP Shared-Memory Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (results are identical to the sequential algorithm):
//  - Seed level: candidate clusters for all unclustered seeds are generated
//    concurrently. Each thread keeps its own best (cardinality, seed) pair;
//    ties are resolved by the lowest seed index, matching the sequential
//    "first seed with maximum cardinality wins" rule.
//  - Scan level: when fewer seeds remain than threads, the seeds are processed
//    one at a time and the closest-candidate scan inside each greedy step is
//    parallelized instead. The two levels never nest.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// Best candidate found by a scan: smallest diameter, ties broken by lowest index.
struct Closest {
    double diameter;
    int index;
};

// Returns true if (diameter, index) should replace the current best candidate.
// Equivalent to the sequential rule "strictly smaller diameter wins; on equal
// diameters the earlier (lower-index) candidate wins", independent of scan order.
inline bool isBetterCandidate(const double diameter, const int index, const Closest& best) {
    return best.index < 0 || diameter < best.diameter ||
           (diameter == best.diameter && index < best.index);
}

// Maximum distance from point p to any of the cluster members
inline double maxDistanceTo(const Point& p, const std::vector<Point>& member_pts) {
    double max_dist = 0.0;
    for (size_t i = 0; i < member_pts.size(); ++i) {
        max_dist = std::max(max_dist, distance(p, member_pts[i]));
    }
    return max_dist;
}

// Minimum amount of work (candidates * members) before a single closest-point
// scan is split across threads; below this the fork/join overhead dominates.
constexpr double kParallelScanMinWork = 65536.0;

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists. When nthreads > 1 the candidate scan is
// parallelized (must not be called from inside another parallel region).
int findClosestPoint(const std::vector<Point>& member_pts,
                     const std::vector<char>& clustered,
                     const std::vector<char>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count,
                     const int nthreads) {
    Closest best{std::numeric_limits<double>::max(), -1};

    if (nthreads > 1 &&
        static_cast<double>(point_count) * static_cast<double>(member_pts.size()) >= kParallelScanMinWork) {
        std::vector<Closest> partial(nthreads, Closest{std::numeric_limits<double>::max(), -1});

        #pragma omp parallel num_threads(nthreads)
        {
            Closest& local = partial[omp_get_thread_num()];

            #pragma omp for schedule(dynamic, 64)
            for (int candidate = 0; candidate < point_count; ++candidate) {
                if (clustered[candidate] || in_cluster[candidate]) continue;
                const double max_dist = maxDistanceTo(points[candidate], member_pts);
                if (max_dist < threshold && isBetterCandidate(max_dist, candidate, local)) {
                    local.diameter = max_dist;
                    local.index = candidate;
                }
            }
        }

        for (int t = 0; t < nthreads; ++t) {
            const Closest& p = partial[t];
            if (p.index >= 0 && isBetterCandidate(p.diameter, p.index, best)) {
                best = p;
            }
        }
    } else {
        // Try each unclustered point as a candidate
        for (int candidate = 0; candidate < point_count; ++candidate) {
            // Skip if already clustered or already in this cluster
            if (clustered[candidate] || in_cluster[candidate]) continue;

            // If adding this point keeps diameter below threshold and is better than current best
            const double max_dist = maxDistanceTo(points[candidate], member_pts);
            if (max_dist < threshold && isBetterCandidate(max_dist, candidate, best)) {
                best.diameter = max_dist;
                best.index = candidate;
            }
        }
    }

    return best.index;
}

// Per-thread scratch space reused across candidate-cluster generations
struct Workspace {
    std::vector<char> in_cluster;   // all zero between uses
    std::vector<int> members;       // members of the last generated cluster
    std::vector<Point> member_pts;  // coordinates of members (same order)

    explicit Workspace(int point_count) : in_cluster(point_count, 0) {
        members.reserve(point_count);
        member_pts.reserve(point_count);
    }
};

// Generate a candidate cluster starting from a seed point.
// The members are left in ws.members. Returns the cardinality (size) of the cluster.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int point_count,
                             const int nthreads,
                             Workspace& ws) {
    ws.members.clear();
    ws.member_pts.clear();

    // Add seed point
    ws.in_cluster[seed_point] = 1;
    ws.members.push_back(seed_point);
    ws.member_pts.push_back(points[seed_point]);

    // Iteratively add closest points
    while (static_cast<int>(ws.members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(ws.member_pts, clustered, ws.in_cluster, points,
                                             threshold, point_count, nthreads);

        if (closest < 0) break; // No more points can be added

        ws.in_cluster[closest] = 1;
        ws.members.push_back(closest);
        ws.member_pts.push_back(points[closest]);
    }

    // Reset the membership flags so the workspace can be reused
    for (size_t i = 0; i < ws.members.size(); ++i) {
        ws.in_cluster[ws.members[i]] = 0;
    }

    return static_cast<int>(ws.members.size());
}

// Best seed found so far: maximum cardinality, ties broken by lowest seed index
struct BestSeed {
    int cardinality = -1;
    int seed = INT_MAX;
    std::vector<int> members;
};

inline void offerSeed(BestSeed& best, const int cardinality, const int seed,
                      const std::vector<int>& members) {
    if (cardinality > best.cardinality ||
        (cardinality == best.cardinality && seed < best.seed)) {
        best.cardinality = cardinality;
        best.seed = seed;
        best.members.assign(members.begin(), members.end());
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const int nthreads = std::max(1, omp_get_max_threads());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // One scratch workspace per thread
    std::vector<Workspace> workspaces;
    workspaces.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        workspaces.emplace_back(N);
    }

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());
        BestSeed overall;

        if (num_seeds >= nthreads) {
            // Seed-level parallelism: each thread evaluates a share of the seeds
            std::vector<BestSeed> per_thread(nthreads);

            #pragma omp parallel num_threads(nthreads)
            {
                const int tid = omp_get_thread_num();
                Workspace& ws = workspaces[tid];
                BestSeed& local = per_thread[tid];

                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < num_seeds; ++i) {
                    const int seed = unclustered_indices[i];
                    if (clustered[seed]) continue;

                    const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                                     threshold, N, 1, ws);
                    offerSeed(local, cardinality, seed, ws.members);
                }
            }

            for (int t = 0; t < nthreads; ++t) {
                const BestSeed& b = per_thread[t];
                if (b.seed != INT_MAX) {
                    offerSeed(overall, b.cardinality, b.seed, b.members);
                }
            }
        } else {
            // Few seeds remain: evaluate them in order, parallelizing each scan
            Workspace& ws = workspaces[0];
            for (int i = 0; i < num_seeds; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                                 threshold, N, nthreads, ws);
                offerSeed(overall, cardinality, seed, ws.members);
            }
        }

        // If we found a cluster, add it
        if (overall.seed != INT_MAX && overall.cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = overall.seed;
            cluster.members = std::move(overall.members);
            clusters.push_back(std::move(cluster));

            // Mark all members as clustered
            const std::vector<int>& members = clusters.back().members;
            for (size_t i = 0; i < members.size(); ++i) {
                clustered[members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
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

    // Compute the diameter of every cluster in parallel (max is order-independent)
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
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
        diameters[c] = max_diameter;
    }

    // Check each cluster (serial, to keep output ordering deterministic)
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

// QT Clustering Benchmark
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
#include <new>
#include <omp.h>
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

// Each candidate keeps its maximum distance to the members already added.
// Candidates remain in point-index order, preserving the sequential tie break.
struct Candidate {
    int point;
    double max_distance;
};

struct SeedResult {
    int seed = -1;
    std::vector<int> members;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;
    std::vector<Cluster> clusters;
    // Avoid creating more workers than the seed workload can use effectively.
    const int thread_count = std::min(omp_get_max_threads(), std::max(1, N / 32));

    // Cache distances when the matrix is reasonably sized. Larger inputs use
    // the same calculation on demand instead of requiring quadratic storage.
    constexpr size_t cache_limit = 256u * 1024u * 1024u;
    std::vector<double> distances;
    if (N > 0 && static_cast<size_t>(N) <= cache_limit / sizeof(double) / N) {
        try {
            distances.resize(static_cast<size_t>(N) * N);
        } catch (const std::bad_alloc&) {
            // The on-demand path has identical distance and comparison semantics.
        }
    }
    if (!distances.empty()) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < N; ++i) {
            double* row = distances.data() + static_cast<size_t>(i) * N;
            for (int j = 0; j < N; ++j) row[j] = distance(points[i], points[j]);
        }
    }

    std::vector<SeedResult> results(thread_count);
    bool done = false;
    #pragma omp parallel num_threads(thread_count) shared(done, unclustered_indices, clusters, results)
    {
        const int thread = omp_get_thread_num();
        std::vector<Candidate> candidates;
        std::vector<int> members;
        candidates.reserve(N);
        members.reserve(N);
        while (true) {
            SeedResult& local = results[thread];
            local.seed = -1;
            local.members.clear();

            #pragma omp for schedule(dynamic, 1)
            for (size_t s = 0; s < unclustered_indices.size(); ++s) {
                const int seed = unclustered_indices[s];
                candidates.clear();
                members.clear();
                members.push_back(seed);
                for (int point : unclustered_indices) {
                    if (point != seed) candidates.push_back({point, 0.0});
                }

                int last = seed;
                while (!candidates.empty()) {
                    const double* row = distances.empty() ? nullptr :
                        distances.data() + static_cast<size_t>(last) * N;
                    double min_diameter = std::numeric_limits<double>::max();
                    int closest = -1;
                    size_t kept = 0;
                    for (const Candidate& candidate : candidates) {
                        if (candidate.point == last) continue;
                        const double dist = row ? row[candidate.point] :
                            distance(points[candidate.point], points[last]);
                        const double maximum = std::max(candidate.max_distance, dist);
                        // A rejected point cannot become eligible later.
                        if (!(maximum < threshold)) continue;
                        candidates[kept++] = {candidate.point, maximum};
                        if (maximum < min_diameter) {
                            min_diameter = maximum;
                            closest = candidate.point;
                        }
                    }
                    candidates.resize(kept);
                    if (closest < 0) break;
                    members.push_back(closest);
                    last = closest;
                }
                if (members.size() > local.members.size() ||
                    (members.size() == local.members.size() &&
                     (local.seed < 0 || seed < local.seed))) {
                    local.seed = seed;
                    local.members = members;
                }
            }

            #pragma omp single
            {
                int best = -1;
                for (int t = 0; t < omp_get_num_threads(); ++t) {
                    if (results[t].seed >= 0 &&
                        (best < 0 || results[t].members.size() > results[best].members.size() ||
                         (results[t].members.size() == results[best].members.size() &&
                          results[t].seed < results[best].seed))) best = t;
                }
                if (best < 0) {
                    done = true;
                } else {
                    clusters.push_back({std::move(results[best].members), results[best].seed});
                    std::vector<unsigned char> clustered(N, 0);
                    for (int point : clusters.back().members) clustered[point] = 1;
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](int point) { return clustered[point] != 0; }),
                        unclustered_indices.end());
                    done = unclustered_indices.empty();
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

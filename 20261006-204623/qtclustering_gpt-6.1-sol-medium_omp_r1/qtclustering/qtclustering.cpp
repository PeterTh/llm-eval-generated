// QT Clustering Benchmark - OpenMP Version
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

// Only points within the threshold of the seed can join its cluster.
// Entries remain in point-index order to preserve the original tie breaking.
struct Candidate {
    int point;
    double max_distance;
};

struct alignas(64) WorkerState {
    std::vector<Candidate> candidates;
    std::vector<int> members;
    std::vector<int> best_members;
    int best_seed = -1;
};

static void generateCandidateCluster(const int seed,
                                     const std::vector<unsigned char>& clustered,
                                     const std::vector<Point>& points,
                                     const std::vector<Candidate>& neighbors,
                                     const double threshold,
                                     WorkerState& worker) {
    auto& candidates = worker.candidates;
    auto& members = worker.members;
    candidates.clear();
    members.clear();
    members.push_back(seed);
    for (const auto& neighbor : neighbors) {
        if (!clustered[neighbor.point]) candidates.push_back(neighbor);
    }

    // Even accepting every eligible point cannot improve the local winner.
    // This bound also respects ties, so it never skips a smaller winning seed.
    const size_t upper_bound = candidates.size() + 1;
    if (worker.best_seed >= 0 &&
        (upper_bound < worker.best_members.size() ||
         (upper_bound == worker.best_members.size() && seed > worker.best_seed))) {
        return;
    }

    while (!candidates.empty()) {
        size_t closest = 0;
        double min_distance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].max_distance < min_distance) {
                min_distance = candidates[i].max_distance;
                closest = i;
            }
        }
        const int next = candidates[closest].point;
        members.push_back(next);

        // Each maximum already accounts for all previous members. Only the
        // distance to the new member is needed. Rejected points can never
        // become eligible again because these maxima only increase.
        size_t remaining = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (i == closest) continue;
            Candidate candidate = candidates[i];
            candidate.max_distance = std::max(
                candidate.max_distance, distance(points[candidate.point], points[next]));
            if (candidate.max_distance < threshold) {
                candidates[remaining++] = candidate;
            }
        }
        candidates.resize(remaining);
    }
}

// The cluster-extraction order is sequential, but all seeds in an extraction
// round are independent and share an immutable membership snapshot.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    std::vector<std::vector<Candidate>> neighbors(N);
    std::vector<WorkerState> workers(omp_get_max_threads());

    // Keep one team alive for preprocessing and every extraction round.
    #pragma omp parallel
    {
        const int thread = omp_get_thread_num();
        auto& worker = workers[thread];
        #pragma omp for schedule(static)
        for (int seed = 0; seed < N; ++seed) {
            unclustered_indices[seed] = seed;
            auto& row = neighbors[seed];
            for (int candidate = 0; candidate < N; ++candidate) {
                if (candidate == seed) continue;
                const double dist = distance(points[candidate], points[seed]);
                if (dist < threshold) row.push_back({candidate, dist});
            }
        }

        while (!unclustered_indices.empty()) {
            worker.best_seed = -1;
            worker.best_members.clear();
            const int count = static_cast<int>(unclustered_indices.size());
            const int chunk = std::max(1, std::min(32, count / (8 * omp_get_num_threads())));
            #pragma omp for schedule(dynamic, chunk)
            for (int i = 0; i < count; ++i) {
                const int seed = unclustered_indices[i];
                generateCandidateCluster(seed, clustered, points, neighbors[seed],
                                         threshold, worker);
                if (worker.best_seed < 0 ||
                    worker.members.size() > worker.best_members.size() ||
                    (worker.members.size() == worker.best_members.size() &&
                     seed < worker.best_seed)) {
                    worker.best_seed = seed;
                    worker.best_members.swap(worker.members);
                }
            }

            // The worksharing barrier publishes each thread's winner. Resolve
            // ties by seed index, matching the sequential ascending seed scan.
            #pragma omp single
            {
                int winner = -1;
                for (int t = 0; t < omp_get_num_threads(); ++t) {
                    const auto& candidate = workers[t];
                    if (candidate.best_seed < 0) continue;
                    if (winner < 0 ||
                        candidate.best_members.size() > workers[winner].best_members.size() ||
                        (candidate.best_members.size() == workers[winner].best_members.size() &&
                         candidate.best_seed < workers[winner].best_seed)) {
                        winner = t;
                    }
                }
                auto& best = workers[winner];
                clusters.push_back({best.best_members, best.best_seed});
                for (const int member : best.best_members) clustered[member] = 1;
                unclustered_indices.erase(
                    std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                   [&clustered](int idx) { return clustered[idx] != 0; }),
                    unclustered_indices.end());
            }
            // The single-region barrier completes membership updates before
            // any thread starts evaluating seeds in the next round.
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

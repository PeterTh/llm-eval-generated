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
#include <numeric>
#include <omp.h>
#include <set>
#include <utility>
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
inline double squaredDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(squaredDistance(p1, p2));
}

struct Neighbor {
    int index;
    double squared_distance;
};

struct CandidateWorkspace {
    std::vector<int> indices;
    std::vector<double> x, y, maxima;
};

// Each diameter is the maximum distance to all members chosen so far.
// Keeping the coordinates contiguous lets each worker update a whole row with SIMD.
static void growCandidate(int seed, const std::vector<Point>& points,
                          const std::vector<Neighbor>& neighbors,
                          const std::vector<char>& assigned, double threshold,
                          CandidateWorkspace& work, std::vector<int>& members) {
    members.clear();
    members.push_back(seed);
    work.indices.clear();
    work.x.clear();
    work.y.clear();
    work.maxima.clear();
    const double infinity = std::numeric_limits<double>::infinity();
    double minimum = infinity;
    for (const Neighbor& neighbor : neighbors) {
        const int p = neighbor.index;
        if (assigned[p]) continue;
        work.indices.push_back(p);
        work.x.push_back(points[p].x);
        work.y.push_back(points[p].y);
        work.maxima.push_back(neighbor.squared_distance);
        minimum = std::min(minimum, neighbor.squared_distance);
    }

    const int count = static_cast<int>(work.indices.size());
    for (int added = 0; added < count && std::sqrt(minimum) < threshold; ++added) {
        // Distinct squared distances may round to the same Euclidean distance.
        // Check that narrow range in index order to preserve the original tie break.
        const double closest = std::sqrt(minimum);
        const double next = std::nextafter(closest, infinity);
        const double upper = next * next;
        const auto match = std::find_if(work.maxima.begin(), work.maxima.end(),
            [&](double value) { return value <= upper && std::sqrt(value) == closest; });
        const int slot = static_cast<int>(match - work.maxima.begin());
        members.push_back(work.indices[slot]);
        const Point member{work.x[slot], work.y[slot]};
        work.maxima[slot] = infinity;
        minimum = infinity;
        #pragma omp simd reduction(min:minimum)
        for (int i = 0; i < count; ++i) {
            const Point candidate{work.x[i], work.y[i]};
            const double diameter = std::max(work.maxima[i], squaredDistance(candidate, member));
            work.maxima[i] = diameter;
            minimum = std::min(minimum, diameter);
        }
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<int> by_x(n);
    std::iota(by_x.begin(), by_x.end(), 0);
    std::sort(by_x.begin(), by_x.end(), [&](int a, int b) {
        return points[a].x < points[b].x;
    });

    std::vector<std::vector<Neighbor>> neighbors(n);
    std::vector<int> bound(n);
    const int max_threads = omp_get_max_threads();
    const int graph_threads = std::min(max_threads, std::max(1, n / 32));
    #pragma omp parallel for schedule(static) num_threads(graph_threads)
    for (int i = 0; i < n; ++i) {
        const int seed = by_x[i];
        auto& row = neighbors[seed];
        for (int direction : {-1, 1}) {
            for (int j = i + direction; j >= 0 && j < n; j += direction) {
                const int other = by_x[j];
                if (std::abs(points[seed].x - points[other].x) > threshold) break;
                const double d = squaredDistance(points[other], points[seed]);
                if (std::sqrt(d) < threshold) row.push_back({other, d});
            }
        }
        std::sort(row.begin(), row.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.index < b.index;
        });
        bound[seed] = 1 + static_cast<int>(row.size());
    }

    std::vector<std::vector<int>> candidates(n);
    std::vector<char> assigned(n, false);
    // A negative size sorts larger clusters first, with seed index breaking ties.
    // Unexamined seeds carry upper bounds, examined seeds carry exact sizes.
    std::set<std::pair<int, int>> ranking;
    for (int seed = 0; seed < n; ++seed) ranking.emplace(-bound[seed], seed);
    std::vector<std::vector<int>> dependents(n);
    std::vector<Cluster> clusters;
    std::vector<int> pending;
    while (!ranking.empty()) {
        auto [negative_size, seed] = *ranking.begin();
        if (assigned[seed]) {
            ranking.erase(ranking.begin());
            continue;
        }
        auto& members = candidates[seed];
        if (!members.empty() || negative_size == -1) {
            ranking.erase(ranking.begin());
            if (members.empty()) members.push_back(seed);
            clusters.push_back({std::move(members), seed});
            for (int p : clusters.back().members) {
                assigned[p] = true;
                for (const Neighbor& neighbor : neighbors[p]) --bound[neighbor.index];
            }
            for (int p : clusters.back().members) {
                for (int affected : dependents[p]) {
                    auto& cached = candidates[affected];
                    if (assigned[affected] || cached.empty()) continue;
                    // A previous evaluation may also have registered this seed.
                    // Invalidating it again simply restores a safe upper bound.
                    ranking.erase({-static_cast<int>(cached.size()), affected});
                    cached.clear();
                    ranking.emplace(-bound[affected], affected);
                }
                std::vector<int>().swap(dependents[p]);
            }
            continue;
        }

        pending.clear();
        const size_t batch = std::max(32, graph_threads * 16);
        while (!ranking.empty() && pending.size() < batch) {
            auto [key, next] = *ranking.begin();
            if (assigned[next] || !candidates[next].empty() || key == -1) break;
            ranking.erase(ranking.begin());
            // Neighborhood bounds can only decrease between evaluations.
            if (-key != bound[next]) ranking.emplace(-bound[next], next);
            else pending.push_back(next);
        }
        if (pending.empty()) continue;
        const int count = static_cast<int>(pending.size());
        const long long work_estimate = 1LL * count * bound[pending.front()];
        const int workers = static_cast<int>(std::min<long long>(
            std::min(max_threads, count), 1 + work_estimate / 1024));
        #pragma omp parallel num_threads(workers)
        {
            CandidateWorkspace work;
            #pragma omp for schedule(dynamic, 2)
            for (int i = 0; i < count; ++i) {
                const int next = pending[i];
                growCandidate(next, points, neighbors[next], assigned, threshold,
                              work, candidates[next]);
            }
        }
        for (int next : pending) {
            const auto& cached = candidates[next];
            ranking.emplace(-static_cast<int>(cached.size()), next);
            for (int p : cached) dependents[p].push_back(next);
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

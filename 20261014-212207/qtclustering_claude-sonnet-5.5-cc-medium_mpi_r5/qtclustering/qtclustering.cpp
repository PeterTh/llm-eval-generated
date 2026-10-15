// QT Clustering Benchmark - MPI Version
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

#include <mpi.h>

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

// Grow a candidate cluster from a seed. Only unclustered points closer than the
// threshold to the seed can ever join, so we track, for each of those, the maximum
// distance to the current members (incrementally) and repeatedly add the one with
// the smallest such distance (lowest index on ties), as in the sequential algorithm.
static void growCluster(const int seed,
                        const std::vector<int>& unclustered,
                        const std::vector<Point>& points,
                        const double threshold,
                        std::vector<int>& members,
                        std::vector<int>& cand,
                        std::vector<double>& maxd) {
    members.clear();
    cand.clear();
    maxd.clear();
    members.push_back(seed);
    const Point ps = points[seed];
    int best = -1;
    double bestd = std::numeric_limits<double>::max();
    for (const int u : unclustered) {
        if (u == seed) continue;
        const double d = distance(points[u], ps);
        if (d < threshold) {
            if (d < bestd) { bestd = d; best = static_cast<int>(cand.size()); }
            cand.push_back(u);
            maxd.push_back(d);
        }
    }
    while (best >= 0) {
        const int chosen = cand[best];
        members.push_back(chosen);
        const Point pc = points[chosen];
        size_t w = 0;
        best = -1;
        bestd = std::numeric_limits<double>::max();
        const size_t n = cand.size();
        for (size_t k = 0; k < n; ++k) {
            const int u = cand[k];
            if (u == chosen) continue;
            const double d = std::max(maxd[k], distance(points[u], pc));
            if (d < threshold) {
                if (d < bestd) { bestd = d; best = static_cast<int>(w); }
                cand[w] = u;
                maxd[w] = d;
                ++w;
            }
        }
        cand.resize(w);
        maxd.resize(w);
    }
}

// Main QT clustering algorithm (MPI). Points are replicated on every rank; seeds are
// distributed cyclically. A seed's candidate cluster only changes when one of its
// members gets clustered, so results are cached and recomputed only when invalidated.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    const int N = static_cast<int>(points.size());

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;

    // Owned seeds: cached cluster members (empty => needs recomputation); dead if clustered
    std::vector<int> owned;
    for (int i = rank; i < N; i += nprocs) owned.push_back(i);
    std::vector<std::vector<int>> cache(owned.size());
    std::vector<char> valid(owned.size(), 0);
    std::vector<char> alive(owned.size(), 1);

    std::vector<int> cand;
    std::vector<double> maxd;
    std::vector<int> winner;

    while (!unclustered.empty()) {
        long long localKey = -1;
        for (size_t o = 0; o < owned.size(); ++o) {
            if (!alive[o]) continue;
            if (!valid[o]) {
                growCluster(owned[o], unclustered, points, threshold, cache[o], cand, maxd);
                valid[o] = 1;
            }
            // maximise cardinality, then smallest seed index
            const long long key = static_cast<long long>(cache[o].size()) * (N + 1) +
                                  (N - owned[o]);
            if (key > localKey) localKey = key;
        }
        long long key;
        MPI_Allreduce(&localKey, &key, 1, MPI_LONG_LONG, MPI_MAX, comm);
        if (key < 0) break;

        const int card = static_cast<int>(key / (N + 1));
        const int seed = N - static_cast<int>(key % (N + 1));
        const int owner = seed % nprocs;
        winner.resize(card);
        if (rank == owner) winner = cache[seed / nprocs];
        MPI_Bcast(winner.data(), card, MPI_INT, owner, comm);

        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members = winner;
        for (const int m : winner) clustered[m] = 1;
        clusters.push_back(std::move(cluster));

        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                                         [&clustered](int idx) { return clustered[idx]; }),
                          unclustered.end());

        // Invalidate affected cached clusters
        for (size_t o = 0; o < owned.size(); ++o) {
            if (!alive[o]) continue;
            if (clustered[owned[o]]) {
                alive[o] = 0;
                std::vector<int>().swap(cache[o]);
                continue;
            }
            for (const int m : cache[o]) {
                if (clustered[m]) {
                    valid[o] = 0;
                    break;
                }
            }
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int rc = [&]() -> int {
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_ms = 0;
    MPI_Reduce(&local_cluster_ms, &max_cluster_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto cluster_time = std::chrono::milliseconds(max_cluster_ms);
    
    if (rank != 0) return 0;

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
    }();
    MPI_Finalize();
    return rc;
}

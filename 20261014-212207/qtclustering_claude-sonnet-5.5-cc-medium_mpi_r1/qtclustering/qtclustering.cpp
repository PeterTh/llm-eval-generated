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

// Scratch buffers reused across seeds
struct Scratch {
    std::vector<int> cand;       // candidate positions (into compact arrays), index order
    std::vector<double> cmax;    // max distance from candidate to current members
    std::vector<int> members;    // positions of members (compact index)
};

// Generate a candidate cluster from the seed at compact position `seed`.
// xs/ys hold the coordinates of all currently unclustered points (in index order).
// Equivalent to the original greedy: repeatedly add the candidate with the smallest
// max-distance to the cluster (first index wins ties) while that is < threshold.
// Max distances are maintained incrementally and over-threshold candidates are
// dropped permanently (their max distance can only grow).
// Returns cardinality; fills sc.members.
static int generateCandidateCluster(const int seed, const double* xs, const double* ys,
                                    const int M, const double threshold, Scratch& sc) {
    sc.members.clear();
    sc.cand.clear();
    sc.cmax.clear();
    sc.members.push_back(seed);

    const double sx = xs[seed], sy = ys[seed];
    for (int c = 0; c < M; ++c) {
        if (c == seed) continue;
        const double dx = xs[c] - sx, dy = ys[c] - sy;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            sc.cand.push_back(c);
            sc.cmax.push_back(d);
        }
    }

    while (!sc.cand.empty()) {
        // Pick the closest candidate (first on ties)
        const int n = static_cast<int>(sc.cand.size());
        int best = 0;
        double bd = sc.cmax[0];
        for (int k = 1; k < n; ++k) {
            if (sc.cmax[k] < bd) { bd = sc.cmax[k]; best = k; }
        }
        const int chosen = sc.cand[best];
        sc.members.push_back(chosen);
        const double cx = xs[chosen], cy = ys[chosen];

        // Update the remaining candidates and compact, preserving order
        int w = 0;
        for (int k = 0; k < n; ++k) {
            if (k == best) continue;
            const int c = sc.cand[k];
            const double dx = xs[c] - cx, dy = ys[c] - cy;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double m = std::max(sc.cmax[k], d);
            if (m < threshold) {
                sc.cand[w] = c;
                sc.cmax[w] = m;
                ++w;
            }
        }
        sc.cand.resize(w);
        sc.cmax.resize(w);
    }
    return static_cast<int>(sc.members.size());
}

// Main QT clustering algorithm (MPI: seeds distributed cyclically over ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered(N);
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) {
        unclustered[i] = i;
        xs[i] = points[i].x;
        ys[i] = points[i].y;
    }
    std::vector<Cluster> clusters;
    Scratch sc;
    std::vector<int> best_local, winner;
    std::vector<char> remove(N);

    while (!unclustered.empty()) {
        const int M = static_cast<int>(unclustered.size());
        long long local_key = -1;
        best_local.clear();
        int local_card = -1;

        for (int pos = rank; pos < M; pos += nprocs) {
            const int card = generateCandidateCluster(pos, xs.data(), ys.data(), M, threshold, sc);
            if (card > local_card) {
                local_card = card;
                local_key = static_cast<long long>(card) * (static_cast<long long>(N) + 1) + (N - pos);
                best_local = sc.members;
            }
        }

        // Max cardinality, ties -> smallest position (matches sequential strict '>')
        long long gkey = -1;
        MPI_Allreduce(&local_key, &gkey, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int card = static_cast<int>(gkey / (static_cast<long long>(N) + 1));
        const int pos = N - static_cast<int>(gkey % (static_cast<long long>(N) + 1));
        const int owner = pos % nprocs;

        winner.resize(card);
        if (rank == owner) winner = best_local;
        MPI_Bcast(winner.data(), card, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = unclustered[pos];
        cluster.members.resize(card);
        for (int i = 0; i < card; ++i) cluster.members[i] = unclustered[winner[i]];
        clusters.push_back(std::move(cluster));

        // Remove clustered points, keeping index order
        for (int i = 0; i < card; ++i) remove[winner[i]] = 1;
        int w = 0;
        for (int i = 0; i < M; ++i) {
            if (remove[i]) { remove[i] = 0; continue; }
            unclustered[w] = unclustered[i];
            xs[w] = xs[i];
            ys[w] = ys[i];
            ++w;
        }
        unclustered.resize(w);
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

static int realMain(int argc, char** argv, const int rank, const int nprocs) {
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (rank == 0) printf("QT Clustering Benchmark\n");
    if (rank == 0) printf("Number of points: %d\n", num_points);
    if (rank == 0) printf("Distance threshold: %.2f\n", threshold);
    if (rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_ms = cluster_time.count();
    long max_cluster_ms = 0;
    MPI_Reduce(&local_cluster_ms, &max_cluster_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) cluster_time = std::chrono::milliseconds(max_cluster_ms);
    
    if (rank == 0) printf("Clustering time: %ld ms\n", cluster_time.count());
    if (rank == 0) printf("Clusters found: %zu\n", clusters.size());
    
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
    
    if (rank == 0) printf("Points clustered: %d / %d (%.1f%%)\n", 
           total_clustered, num_points, 
           100.0 * total_clustered / num_points);
    if (rank == 0) printf("Average cluster size: %.2f\n", avg_cluster_size);
    if (rank == 0) printf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    if (rank == 0) printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults && rank == 0) {
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
    if (validate && rank == 0) {
        const bool valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int rc = realMain(argc, argv, rank, nprocs);
    MPI_Finalize();
    return rc;
}

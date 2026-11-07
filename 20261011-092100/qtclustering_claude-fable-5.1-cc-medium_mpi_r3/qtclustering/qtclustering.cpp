// QT Clustering Benchmark - MPI Distributed-Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * Every rank holds the full (replicated) point set and clustered state.
//   * In each outer iteration the candidate seeds (all unclustered points)
//     are distributed cyclically over the ranks. Each rank grows a candidate
//     cluster for each of its seeds and keeps its local best.
//   * The global best is selected with an MPI_MAXLOC reduction on
//     (cardinality, seed index): the largest cluster wins, ties go to the
//     smallest seed index, which is exactly the sequential tie-break rule.
//   * The owner of the winning seed broadcasts the member list and every
//     rank updates its clustered state identically.
//
// The per-seed cluster growth keeps, for every candidate point, the maximum
// distance to the current cluster members and updates it incrementally when
// a member is added. The maximum of exact double values is order-independent,
// so the resulting clusters are identical to the sequential algorithm.

#include <mpi.h>

#include <algorithm>
#include <chrono>
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

// Scratch buffers reused across candidate cluster generations (avoids
// re-allocating O(N) memory for every seed).
struct GrowScratch {
    std::vector<int> cand;       // viable candidate indices (compact list)
    std::vector<double> cand_x;  // x of viable candidates
    std::vector<double> cand_y;  // y of viable candidates
    std::vector<double> maxd;    // max distance from candidate to cluster members
    std::vector<int> members;
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster. Growth is abandoned early
// (returning 0) if the cluster provably cannot exceed `beat` members; callers
// only ever accept strictly larger clusters, so this does not change results.
//
// Semantics are identical to the sequential reference: at each step the
// unclustered point whose maximum distance to all current members is smallest
// (and < threshold) is added; ties resolve to the smallest point index.
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const double* __restrict px,
                             const double* __restrict py,
                             const double threshold,
                             const int point_count,
                             const int beat,
                             GrowScratch& s) {
    // Collect viable candidates: unclustered points that are within the
    // threshold of the seed (anything further can never join this cluster).
    s.cand.clear();
    s.cand_x.clear();
    s.cand_y.clear();
    s.maxd.clear();
    const double sx = px[seed_point];
    const double sy = py[seed_point];
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed_point) continue;
        const double dx = px[c] - sx;
        const double dy = py[c] - sy;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            s.cand.push_back(c);
            s.cand_x.push_back(px[c]);
            s.cand_y.push_back(py[c]);
            s.maxd.push_back(d);
        }
    }
    
    s.members.clear();
    s.members.push_back(seed_point);
    
    int n = static_cast<int>(s.cand.size());
    // Upper bound on final size: 1 + viable candidates
    if (1 + n <= beat) return 0;
    
    int* __restrict cand = s.cand.data();
    double* __restrict cx = s.cand_x.data();
    double* __restrict cy = s.cand_y.data();
    double* __restrict maxd = s.maxd.data();
    
    while (n > 0) {
        // Select the candidate with the smallest max-distance (ties: smallest index)
        int best = -1;
        int best_idx = std::numeric_limits<int>::max();
        double best_d = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            const double d = maxd[i];
            const int idx = cand[i];
            if (d < best_d || (d == best_d && idx < best_idx)) {
                best_d = d;
                best_idx = idx;
                best = i;
            }
        }
        
        const double bx = cx[best];
        const double by = cy[best];
        s.members.push_back(best_idx);
        
        // Remove the chosen candidate (swap with last)
        --n;
        cand[best] = cand[n];
        cx[best] = cx[n];
        cy[best] = cy[n];
        maxd[best] = maxd[n];
        
        // Update max distances with the new member and compact the list by
        // dropping candidates that can no longer satisfy the threshold.
        int w = 0;
        for (int i = 0; i < n; ++i) {
            const double dx = cx[i] - bx;
            const double dy = cy[i] - by;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double m = (d > maxd[i]) ? d : maxd[i];
            if (m < threshold) {
                cand[w] = cand[i];
                cx[w] = cx[i];
                cy[w] = cy[i];
                maxd[w] = m;
                ++w;
            }
        }
        n = w;
        
        if (static_cast<int>(s.members.size()) + n <= beat) return 0;
    }
    
    return static_cast<int>(s.members.size());
}

// Main QT clustering algorithm (MPI parallel, results identical on all ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }
    
    // Initialize unclustered indices (kept sorted ascending at all times)
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    GrowScratch scratch;
    scratch.cand.reserve(N);
    scratch.cand_x.reserve(N);
    scratch.cand_y.reserve(N);
    scratch.maxd.reserve(N);
    scratch.members.reserve(N);
    std::vector<int> best_cluster_members;
    best_cluster_members.reserve(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        best_cluster_members.clear();
        
        // Try the seeds assigned to this rank (cyclic distribution)
        const int M = static_cast<int>(unclustered_indices.size());
        for (int i = rank; i < M; i += nranks) {
            const int seed = unclustered_indices[i];
            
            const int cardinality = generateCandidateCluster(seed, clustered, px.data(), py.data(),
                                                             threshold, N, max_cardinality, scratch);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = scratch.members;
            }
        }
        
        // Global selection: largest cardinality, ties -> smallest seed index
        // (MPI_MAXLOC breaks ties by the smallest "location").
        int local[2] = {max_cardinality, (best_seed >= 0) ? best_seed : std::numeric_limits<int>::max()};
        int global[2];
        MPI_Allreduce(local, global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int g_card = global[0];
        const int g_seed = global[1];
        
        // Owner rank: position of g_seed in the (sorted) unclustered list mod nranks
        const int pos = static_cast<int>(
            std::lower_bound(unclustered_indices.begin(), unclustered_indices.end(), g_seed) -
            unclustered_indices.begin());
        const int owner = pos % nranks;
        
        best_cluster_members.resize(g_card);
        MPI_Bcast(best_cluster_members.data(), g_card, MPI_INT, owner, MPI_COMM_WORLD);
        
        Cluster cluster;
        cluster.seed_point = g_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(std::move(cluster));
        
        // Mark all members as clustered
        for (size_t i = 0; i < best_cluster_members.size(); ++i) {
            clustered[best_cluster_members[i]] = 1;
        }
        
        // Remove clustered points from unclustered list (order preserved)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool is_root = (rank == 0);
    
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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (is_root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (is_root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
    }
    
    // Generate synthetic data (deterministic; replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_cluster_ms = cluster_time.count();
    long max_cluster_ms = 0;
    MPI_Reduce(&local_cluster_ms, &max_cluster_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(max_cluster_ms);
    
    int ret = 0;
    if (is_root) {
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
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}

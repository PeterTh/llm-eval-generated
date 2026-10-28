// QT Clustering Benchmark - MPI Distributed Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in every round of the main loop, the candidate seeds are
// distributed cyclically across MPI ranks. Each rank grows candidate clusters
// from its local seeds and the globally best candidate (largest cardinality,
// lowest seed index on ties, matching the sequential first-wins rule) is
// selected with an allgather. The owning rank broadcasts the winning members.
// Every rank then applies the identical state update, so the clustering result
// is identical to the sequential algorithm for any number of processes.

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Scratch buffers reused across candidate growths to avoid reallocation
struct GrowScratch {
    std::vector<int> active;     // unclustered candidates still admissible, ascending index
    std::vector<double> maxd;    // running max distance from each active candidate to the cluster
};

// Greedily grow a candidate cluster from `seed`.
//
// Semantically identical to repeatedly calling "find the unclustered, not yet
// in-cluster point whose max distance to the cluster is smallest and < threshold
// (lowest index on ties)" and adding it. The max distance of each candidate is
// tracked incrementally instead of being recomputed against all members. Because
// max is exact and order-independent, the values are bit-identical to the
// recomputed ones. A candidate whose max distance reaches the threshold can never
// be admitted again, so it is dropped permanently.
void growCluster(const int seed,
                 const std::vector<char>& clustered,
                 const std::vector<Point>& points,
                 const double threshold,
                 GrowScratch& scratch,
                 std::vector<int>& members) {
    const int N = static_cast<int>(points.size());
    std::vector<int>& active = scratch.active;
    std::vector<double>& maxd = scratch.maxd;

    members.clear();
    members.push_back(seed);

    active.clear();
    maxd.clear();
    const Point& seed_pt = points[seed];
    for (int c = 0; c < N; ++c) {
        if (c == seed || clustered[c]) continue;
        const double d = distance(points[c], seed_pt);
        if (d < threshold) {
            active.push_back(c);
            maxd.push_back(d);
        }
    }
    if (active.empty()) return;

    // Position of the admissible candidate with the smallest max distance
    // (first occurrence on ties, i.e. lowest index since active is ascending)
    size_t best = 0;
    for (size_t k = 1; k < active.size(); ++k) {
        if (maxd[k] < maxd[best]) best = k;
    }

    while (true) {
        const int chosen = active[best];
        members.push_back(chosen);
        const Point& chosen_pt = points[chosen];

        // Update max distances against the new member and compact survivors
        // (order preserved), tracking the next best candidate in the same pass.
        size_t w = 0;
        size_t next_best = static_cast<size_t>(-1);
        double next_val = 0.0;
        const size_t n = active.size();
        for (size_t k = 0; k < n; ++k) {
            if (k == best) continue;
            const double m = std::max(maxd[k], distance(points[active[k]], chosen_pt));
            if (m < threshold) {
                active[w] = active[k];
                maxd[w] = m;
                if (next_best == static_cast<size_t>(-1) || m < next_val) {
                    next_best = w;
                    next_val = m;
                }
                ++w;
            }
        }
        active.resize(w);
        maxd.resize(w);

        if (next_best == static_cast<size_t>(-1)) break;
        best = next_best;
    }
}

// Upper bound on the cardinality of any cluster grown from `seed`: every member
// other than the seed must be an unclustered point within the threshold of the seed
// (the candidate's max distance includes the seed itself).
int cardinalityBound(const int seed,
                     const std::vector<int>& unclustered,
                     const std::vector<Point>& points,
                     const double threshold) {
    int bound = 1;
    const Point& seed_pt = points[seed];
    for (const int u : unclustered) {
        if (u != seed && distance(points[u], seed_pt) < threshold) {
            ++bound;
        }
    }
    return bound;
}

// Main QT clustering algorithm (MPI-parallel over candidate seeds)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered(N);  // always ascending
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    GrowScratch scratch;
    std::vector<int> candidate_members;
    std::vector<std::pair<int, int>> order;  // (bound, seed)

    while (!unclustered.empty()) {
        // Local seeds: positions p in the unclustered list with p % nprocs == rank
        order.clear();
        const size_t U = unclustered.size();
        for (size_t p = static_cast<size_t>(rank); p < U; p += static_cast<size_t>(nprocs)) {
            const int seed = unclustered[p];
            order.emplace_back(cardinalityBound(seed, unclustered, points, threshold), seed);
        }
        // Evaluate most promising seeds first; ties by ascending seed index
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });

        // Local best under (cardinality desc, seed asc)
        int best_card = 0;
        int best_seed = INT_MAX;
        std::vector<int> best_members;
        for (const auto& [bound, seed] : order) {
            // Remaining seeds cannot beat the local best (sorted by bound, then seed)
            if (bound < best_card || (bound == best_card && seed > best_seed)) break;

            growCluster(seed, clustered, points, threshold, scratch, candidate_members);
            const int card = static_cast<int>(candidate_members.size());
            if (card > best_card || (card == best_card && seed < best_seed)) {
                best_card = card;
                best_seed = seed;
                best_members = candidate_members;
            }
        }

        // Global best across ranks with the same ordering
        const int local[2] = {best_card, best_seed};
        std::vector<int> all(2 * static_cast<size_t>(nprocs));
        MPI_Allgather(local, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int win_card = 0;
        int win_seed = INT_MAX;
        int win_rank = -1;
        for (int r = 0; r < nprocs; ++r) {
            const int c = all[2 * r];
            const int s = all[2 * r + 1];
            if (c > 0 && (c > win_card || (c == win_card && s < win_seed))) {
                win_card = c;
                win_seed = s;
                win_rank = r;
            }
        }

        // No cluster can be formed (cannot happen while unclustered points exist)
        if (win_rank < 0 || win_card <= 0) break;

        // Share the winning cluster's members with all ranks
        std::vector<int> members;
        if (rank == win_rank) {
            members = best_members;
        } else {
            members.resize(win_card);
        }
        MPI_Bcast(members.data(), win_card, MPI_INT, win_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = win_seed;
        cluster.members = std::move(members);
        for (const int m : cluster.members) {
            clustered[m] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list (order preserved)
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
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
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);
    
    const double t_local = MPI_Wtime() - t_start;
    double t_max = 0.0;
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long cluster_ms = static_cast<long>(t_max * 1000.0);
    
    if (root) {
        printf("Clustering time: %ld ms\n", cluster_ms);
        printf("Clusters found: %zu\n", clusters.size());
    }
    
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
    
    if (root) {
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        // Performance metrics
        const double time_sec = cluster_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && root) {
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
    
    // Validation (performed on root; status broadcast so all ranks agree on exit code)
    int status = 0;
    if (validate) {
        int valid = 1;
        if (root) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        status = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return status;
}

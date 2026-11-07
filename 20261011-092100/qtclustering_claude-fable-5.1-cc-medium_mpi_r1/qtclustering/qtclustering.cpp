// QT Clustering Benchmark - MPI Distributed-Memory Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (MPI, SPMD):
//   * Every rank holds the full (deterministically generated) point set and
//     the full "clustered" state, so no point data has to be exchanged.
//   * In each round the unclustered seeds are dealt round-robin across ranks.
//     Each rank grows a candidate cluster for each of its seeds and keeps the
//     best one (largest cardinality, lowest seed index on ties - exactly the
//     ordering of the sequential code).
//   * The global winner is found with a single MPI_Allreduce(MPI_MAXLOC) on
//     (cardinality, position-in-seed-list); MAXLOC's tie rule (smallest index)
//     reproduces the sequential "first seed with maximal cardinality".
//   * The owning rank broadcasts the member list; all ranks update the shared
//     state identically and continue with the next round.
//
// Candidate-cluster growth is implemented incrementally: for every remaining
// candidate the maximum distance to the current members is maintained and
// updated when a member is added. Because max() is exact in floating point,
// this yields bit-identical results to recomputing the maximum from scratch,
// while reducing the per-seed cost from O(n * k^2) to O(n * k).

#include <mpi.h>

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

inline double distance(const double x1, const double y1, const double x2, const double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// Scratch buffers reused across candidate-cluster generations (avoid reallocations)
struct ClusterScratch {
    std::vector<int> cand;      // remaining candidates (ascending index order)
    std::vector<double> cmax;   // max distance from candidate to current members
};

// Generate a candidate cluster starting from a seed point.
//
// `pool` is the ascending list of currently unclustered point indices.
// Returns the cardinality (size) of the cluster. If, while growing, it becomes
// certain that the final cardinality cannot exceed `prune_limit`, the growth is
// abandoned and a value <= prune_limit is returned (such a cluster can never be
// selected because selection requires a strictly larger cardinality).
//
// Semantics are identical to the sequential algorithm: at each step the
// unclustered, not-yet-member point with the smallest max-distance to all
// members (strictly below threshold) is added; ties go to the lowest index.
static int generateCandidateCluster(const int seed_point,
                                    const double* __restrict px,
                                    const double* __restrict py,
                                    const std::vector<int>& pool,
                                    const double threshold,
                                    const int prune_limit,
                                    std::vector<int>& members,
                                    ClusterScratch& scratch) {
    std::vector<int>& cand = scratch.cand;
    std::vector<double>& cmax = scratch.cmax;

    members.clear();
    members.push_back(seed_point);

    // Initialise candidates: distance to the seed (the only member so far).
    // Candidates at or beyond the threshold can never be added (the max distance
    // only grows), so they are dropped immediately.
    cand.clear();
    cmax.clear();
    const double sx = px[seed_point];
    const double sy = py[seed_point];
    int best_idx = -1;
    double best_val = std::numeric_limits<double>::max();
    for (size_t i = 0; i < pool.size(); ++i) {
        const int c = pool[i];
        if (c == seed_point) continue;
        const double d = distance(px[c], py[c], sx, sy);
        if (d < threshold) {
            if (d < best_val) {
                best_val = d;
                best_idx = static_cast<int>(cand.size());
            }
            cand.push_back(c);
            cmax.push_back(d);
        }
    }

    // Iteratively add the closest point
    while (!cand.empty()) {
        // Upper bound on final size: everything still eligible gets added
        if (static_cast<int>(members.size() + cand.size()) <= prune_limit) {
            return prune_limit;
        }

        const int m = cand[best_idx];
        members.push_back(m);

        // Update max distances with the new member, compact the candidate list
        // (dropping the new member and anything now at/over the threshold), and
        // locate the next closest candidate in the same pass.
        const double mx = px[m];
        const double my = py[m];
        const size_t n = cand.size();
        size_t w = 0;
        int nbest = -1;
        double nval = std::numeric_limits<double>::max();
        for (size_t i = 0; i < n; ++i) {
            const int c = cand[i];
            if (c == m) continue;
            const double d = distance(px[c], py[c], mx, my);
            const double nm = (d > cmax[i]) ? d : cmax[i];
            if (nm < threshold) {
                if (nm < nval) {
                    nval = nm;
                    nbest = static_cast<int>(w);
                }
                cand[w] = c;
                cmax[w] = nm;
                ++w;
            }
        }
        cand.resize(w);
        cmax.resize(w);
        best_idx = nbest;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel, all ranks return the full result)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Structure-of-arrays copy of the coordinates for the hot loops
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    ClusterScratch scratch;
    scratch.cand.reserve(N);
    scratch.cmax.reserve(N);
    std::vector<int> candidate_members;
    std::vector<int> best_cluster_members;
    candidate_members.reserve(N);
    best_cluster_members.reserve(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int pool_size = static_cast<int>(unclustered_indices.size());

        int max_cardinality = -1;
        int best_pos = INT_MAX;   // position of best seed within unclustered_indices
        best_cluster_members.clear();

        // Try each of this rank's seeds (round-robin distribution of the seed list)
        for (int pos = rank; pos < pool_size; pos += nprocs) {
            const int seed = unclustered_indices[pos];

            const int cardinality = generateCandidateCluster(
                seed, px.data(), py.data(), unclustered_indices, threshold,
                max_cardinality, candidate_members, scratch);

            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_pos = pos;
                std::swap(best_cluster_members, candidate_members);
            }
        }

        // Global winner: largest cardinality, lowest seed position on ties
        struct { int value; int index; } local_best{max_cardinality, best_pos}, global_best{};
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global_best.index == INT_MAX || global_best.value <= 0) {
            // No more clusters can be formed
            break;
        }

        const int owner = global_best.index % nprocs;
        const int best_seed = unclustered_indices[global_best.index];

        // Distribute the winning member list from the owning rank
        best_cluster_members.resize(global_best.value);
        MPI_Bcast(best_cluster_members.data(), global_best.value, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(std::move(cluster));

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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool is_root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks receive identical argv)
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
        printf("MPI processes: %d\n", nprocs);
    }
    
    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    int exit_code = 0;

    if (is_root) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
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
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

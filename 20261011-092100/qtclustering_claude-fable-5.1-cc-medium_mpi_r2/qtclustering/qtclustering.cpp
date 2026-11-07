// QT Clustering Benchmark - MPI Distributed-Memory Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (MPI):
//   Every rank holds the full (small) point set and the shared "clustered"
//   state.  In each round of the outer loop the candidate-cluster generation
//   for the remaining seeds is distributed cyclically over the ranks.  Each
//   rank determines its locally best candidate cluster; a single
//   MPI_Allreduce(MPI_MAXLOC) selects the global winner (largest cardinality,
//   lowest seed index on ties - identical to the sequential tie-breaking),
//   and the owning rank broadcasts the member list.  All ranks then update
//   their replicated state identically, so no further synchronisation is
//   needed.
//
// Candidate-cluster growth itself is implemented incrementally: for each
// still-eligible candidate point the maximum distance to the current cluster
// members is maintained and updated with only the newest member.  Candidates
// whose maximum distance reaches the threshold can never be added and are
// dropped.  The computed distances/maxima are bit-identical to the original
// algorithm, so the resulting clusters are identical.  A seed whose cluster
// can provably not beat the best cardinality found so far on this rank is
// abandoned early (exact pruning, since the sequential code only replaces
// the best cluster on strictly greater cardinality).

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

// Scratch space reused across candidate-cluster generations (avoids
// re-allocation in the hot loop).
struct GrowWorkspace {
    std::vector<double> ax, ay;   // coordinates of still-eligible candidates
    std::vector<double> amax;     // max distance of candidate to cluster members
    std::vector<int>    aidx;     // original point index of candidate
    std::vector<int>    members;  // members of the cluster being grown

    void reserve(const size_t n) {
        ax.resize(n); ay.resize(n); amax.resize(n); aidx.resize(n);
        members.reserve(n);
    }
};

// Generate a candidate cluster starting from a seed point.
//
// ux/uy/uidx describe the currently unclustered points (structure-of-arrays).
// Returns the cardinality (size) of the cluster, or -1 if growth was
// abandoned because the cluster can provably not exceed `prune_bound`
// members (in which case it would not have replaced the current best).
// The members are left in ws.members (in insertion order).
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<double>& ux,
                                    const std::vector<double>& uy,
                                    const std::vector<int>& uidx,
                                    const double threshold,
                                    const int prune_bound,
                                    GrowWorkspace& ws) {
    const int U = static_cast<int>(uidx.size());
    double* __restrict ax   = ws.ax.data();
    double* __restrict ay   = ws.ay.data();
    double* __restrict amax = ws.amax.data();
    int*    __restrict aidx = ws.aidx.data();
    std::vector<int>& members = ws.members;
    members.clear();

    // Initial candidate set: all unclustered points except the seed.
    int active = 0;
    for (int i = 0; i < U; ++i) {
        if (uidx[i] == seed_point) continue;
        ax[active] = ux[i];
        ay[active] = uy[i];
        amax[active] = 0.0;
        aidx[active] = uidx[i];
        ++active;
    }

    // Add seed point
    members.push_back(seed_point);
    double nx = ux[0], ny = uy[0];
    for (int i = 0; i < U; ++i) {
        if (uidx[i] == seed_point) { nx = ux[i]; ny = uy[i]; break; }
    }

    // Iteratively add closest points
    while (active > 0) {
        // Update the max distance of every candidate with the newest member.
        // (max over members is order independent, so this equals the
        //  sequential recomputation bit for bit.)
        for (int i = 0; i < active; ++i) {
            const double dx = ax[i] - nx;
            const double dy = ay[i] - ny;
            const double d = std::sqrt(dx * dx + dy * dy);
            amax[i] = std::max(amax[i], d);
        }

        // Drop candidates that can never be added any more, and find the
        // closest remaining one (smallest max distance, lowest index on ties).
        int kept = 0;
        int best = -1;
        double best_d = std::numeric_limits<double>::max();
        int best_idx = std::numeric_limits<int>::max();
        for (int i = 0; i < active; ++i) {
            const double d = amax[i];
            if (!(d < threshold)) continue;
            if (kept != i) {
                ax[kept] = ax[i]; ay[kept] = ay[i]; amax[kept] = d; aidx[kept] = aidx[i];
            }
            const int idx = aidx[kept];
            if (d < best_d || (d == best_d && idx < best_idx)) {
                best_d = d; best_idx = idx; best = kept;
            }
            ++kept;
        }
        active = kept;
        if (best < 0) break; // No more points can be added

        // Exact pruning: final cardinality <= |members| + active.
        if (static_cast<int>(members.size()) + active <= prune_bound) {
            return -1;
        }

        // Add the closest point and remove it from the candidate set.
        members.push_back(best_idx);
        nx = ax[best]; ny = ay[best];
        --active;
        ax[best] = ax[active]; ay[best] = ay[active];
        amax[best] = amax[active]; aidx[best] = aidx[active];
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel, replicated state on all ranks)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<double> ux(N), uy(N);
    GrowWorkspace ws;
    ws.reserve(N);
    std::vector<int> best_cluster_members;
    best_cluster_members.reserve(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Structure-of-arrays copy of the unclustered points
        for (int i = 0; i < U; ++i) {
            ux[i] = points[unclustered_indices[i]].x;
            uy[i] = points[unclustered_indices[i]].y;
        }
        ux.resize(U); uy.resize(U);

        int max_cardinality = -1;
        int best_seed = -1;
        best_cluster_members.clear();
        
        // Try each of this rank's seeds (cyclic distribution for load balance;
        // increasing seed order preserves the sequential tie-breaking).
        for (int i = rank; i < U; i += nranks) {
            const int seed = unclustered_indices[i];
            
            const int cardinality = generateCandidateCluster(seed, ux, uy,
                                                             unclustered_indices,
                                                             threshold,
                                                             max_cardinality, ws);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = ws.members;
            }
        }

        // Global winner: largest cardinality, lowest seed index on ties.
        int local[2] = {max_cardinality, best_seed};
        int global[2];
        MPI_Allreduce(local, global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        max_cardinality = global[0];
        best_seed = global[1];
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            // The owner rank is determined by the seed's position in the
            // (identical on all ranks) unclustered list.
            const int pos = static_cast<int>(
                std::lower_bound(unclustered_indices.begin(),
                                 unclustered_indices.end(), best_seed)
                - unclustered_indices.begin());
            const int owner = pos % nranks;
            best_cluster_members.resize(max_cardinality);
            MPI_Bcast(best_cluster_members.data(), max_cardinality, MPI_INT,
                      owner, MPI_COMM_WORLD);

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
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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
        printf("MPI ranks: %d\n", nranks);
    }
    
    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    int ret = 0;
    if (root) {
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
        fflush(stdout);
    }

    MPI_Finalize();
    return ret;
}

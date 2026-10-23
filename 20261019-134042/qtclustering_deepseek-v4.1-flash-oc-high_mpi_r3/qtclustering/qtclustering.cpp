// QT Clustering Benchmark - MPI Distributed Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The seed-candidate evaluations within each clustering round are independent,
// so they are distributed cyclically across MPI ranks. Each rank keeps a
// replicated copy of the (small) global state (clustered mask and unclustered
// index list). Every round the ranks agree on the best candidate with a
// deterministic reduction that preserves the sequential tie-breaking rule
// (largest cardinality, then earliest unclustered seed), and the winning
// cluster membership is broadcast so that all ranks advance identically.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
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

// Reusable scratch storage for candidate-cluster generation. The stamp array
// lets each candidate evaluation start from a clean "in cluster" state without
// clearing an O(N) array for every seed.
struct CandidateBuffers {
    std::vector<double> max_dist;
    std::vector<int> in_stamp;
    int stamp;

    explicit CandidateBuffers(int n) : max_dist(n), in_stamp(n, 0), stamp(0) {}
};

// Generate a candidate cluster starting from a seed point.
//
// This is mathematically identical to repeatedly calling findClosestPoint():
// max_dist[c] always holds the maximum distance from candidate c to every
// member of the cluster built so far, so adding a member only requires
// updating it with that one new distance. Squared distances are used for the
// comparisons because the square root is monotonic (the compared order and
// threshold predicate are unchanged).
// Returns the cardinality (size) of the cluster.
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<int>& unclustered,
                                    const std::vector<Point>& points,
                                    const double threshold_sq,
                                    std::vector<int>& members,
                                    CandidateBuffers& buf) {
    const int U = static_cast<int>(unclustered.size());

    ++buf.stamp;
    if (buf.stamp == 0) {
        std::fill(buf.in_stamp.begin(), buf.in_stamp.end(), 0);
        buf.stamp = 1;
    }
    const int stamp = buf.stamp;

    // Add seed point
    members.clear();
    members.push_back(seed_point);
    buf.in_stamp[seed_point] = stamp;

    // Initialize the maximum distance to the current cluster ({seed} only)
    const Point seed = points[seed_point];
    for (int k = 0; k < U; ++k) {
        const int c = unclustered[k];
        if (c == seed_point) continue;
        const double dx = points[c].x - seed.x;
        const double dy = points[c].y - seed.y;
        buf.max_dist[c] = dx * dx + dy * dy;
    }

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < U) {
        int best = -1;
        double best_dist = std::numeric_limits<double>::max();

        for (int k = 0; k < U; ++k) {
            const int c = unclustered[k];
            if (buf.in_stamp[c] == stamp) continue;

            const double d = buf.max_dist[c];
            if (d < threshold_sq && d < best_dist) {
                best_dist = d;
                best = c;
            }
        }

        if (best < 0) break; // No more points can be added

        buf.in_stamp[best] = stamp;
        members.push_back(best);

        // Incrementally update the maximum distance of every remaining candidate
        const Point added = points[best];
        for (int k = 0; k < U; ++k) {
            const int c = unclustered[k];
            if (buf.in_stamp[c] == stamp) continue;
            const double dx = points[c].x - added.x;
            const double dy = points[c].y - added.y;
            const double d = dx * dx + dy * dy;
            if (d > buf.max_dist[c]) buf.max_dist[c] = d;
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI distributed-memory version)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    CandidateBuffers buf(N);
    std::vector<int> scratch;
    std::vector<int> best_members;
    scratch.reserve(N);
    best_members.reserve(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        int local_best_cardinality = 0;
        int local_best_pos = -1;

        // Try every unclustered point as a seed, distributed cyclically.
        for (int pos = rank; pos < U; pos += nprocs) {
            const int seed = unclustered_indices[pos];
            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold_sq, scratch, buf);

            if (cardinality > local_best_cardinality ||
                (cardinality == local_best_cardinality &&
                 (local_best_pos < 0 || pos < local_best_pos))) {
                local_best_cardinality = cardinality;
                local_best_pos = pos;
                best_members.swap(scratch);
            }
        }

        // Reduce to the global best. The key packs cardinality (high bits) and
        // the negated position in the unclustered list (low bits), so a plain
        // MAX reduction reproduces the sequential rule exactly: highest
        // cardinality, with ties broken in favour of the earliest seed.
        unsigned long long local_key = 0;
        if (local_best_pos >= 0) {
            local_key =
                (static_cast<unsigned long long>(local_best_cardinality + 1) << 32) |
                static_cast<unsigned>(0xFFFFFFFFu - static_cast<unsigned>(local_best_pos));
        }
        unsigned long long global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const int global_pos = static_cast<int>(
            0xFFFFFFFFu - static_cast<unsigned>(global_key & 0xFFFFFFFFull));
        const int best_seed = unclustered_indices[global_pos];
        const int owner = global_pos % nprocs;

        // The winning rank broadcasts the membership so every rank can update
        // the replicated state identically.
        int count = (rank == owner) ? static_cast<int>(best_members.size()) : 0;
        MPI_Bcast(&count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank != owner) {
            best_members.resize(count);
        }
        MPI_Bcast(best_members.data(), count, MPI_INT, owner, MPI_COMM_WORLD);

        // Mark all members as clustered
        for (int i = 0; i < count; ++i) {
            clustered[best_members[i]] = 1;
        }

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(std::move(cluster));
        }

        // Remove clustered points from unclustered list, preserving order
        int write = 0;
        for (int k = 0; k < U; ++k) {
            const int idx = unclustered_indices[k];
            if (!clustered[idx]) {
                unclustered_indices[write++] = idx;
            }
        }
        unclustered_indices.resize(write);
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data. Generation is deterministic (fixed seed), so
    // every rank independently produces the exact same point set.
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    int rc = 0;
    if (rank == 0) {
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
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}

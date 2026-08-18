// QT Clustering Benchmark - distributed-memory MPI version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};
static_assert(sizeof(Point) == 2 * sizeof(double),
              "Point must be two contiguous doubles for MPI transfer");

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
        // Avoid a zero-sized group forever when N is smaller than 30.
        if (group_cnt == 0 && N < 30) group_cnt = 1;
        
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

// Workspace is reused for every seed.  generation_marks avoids clearing an
// N-element in-cluster array for each candidate cluster.
struct CandidateWorkspace {
    explicit CandidateWorkspace(const int n)
        : generation_marks(n, 0), diameters(n), generation(0) {
        members.reserve(n);
    }

    std::vector<int> generation_marks;
    std::vector<double> diameters;
    std::vector<int> members;
    int generation;
};

// Generate exactly the same greedy candidate as the original implementation.
// A candidate's current diameter is updated only for the newly-added member,
// instead of being recomputed against every member on every iteration.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             CandidateWorkspace& work) {
    ++work.generation;
    if (work.generation == INT_MAX) {
        std::fill(work.generation_marks.begin(), work.generation_marks.end(), 0);
        work.generation = 1;
    }
    const int generation = work.generation;
    work.members.clear();
    work.members.push_back(seed_point);
    work.generation_marks[seed_point] = generation;

    // This is the first member-to-candidate distance in the original order.
    for (const int candidate : unclustered) {
        if (candidate != seed_point) {
            work.diameters[candidate] =
                std::max(0.0, distance(points[candidate], points[seed_point]));
        }
    }

    while (work.members.size() < unclustered.size()) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // unclustered is sorted, preserving the original lowest-index tie break.
        for (const int candidate : unclustered) {
            if (work.generation_marks[candidate] == generation) continue;
            const double candidate_diameter = work.diameters[candidate];
            if (candidate_diameter < threshold && candidate_diameter < min_diameter) {
                min_diameter = candidate_diameter;
                closest = candidate;
            }
        }
        if (closest < 0) break;

        work.generation_marks[closest] = generation;
        work.members.push_back(closest);

        // Add the newest member's contribution to each remaining diameter.
        for (const int candidate : unclustered) {
            if (work.generation_marks[candidate] == generation) continue;
            const double dist = distance(points[candidate], points[closest]);
            work.diameters[candidate] = std::max(work.diameters[candidate], dist);
        }
    }

    return static_cast<int>(work.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int rank_count) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    CandidateWorkspace work(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_cardinality = -1;
        int local_seed = INT_MAX;
        std::vector<int> local_best_members;
        
        // Cyclic assignment keeps differently-sized candidate clusters balanced.
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size();
             i += static_cast<size_t>(rank_count)) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold, work);

            if (cardinality > local_cardinality ||
                (cardinality == local_cardinality && seed < local_seed)) {
                local_cardinality = cardinality;
                local_seed = seed;
                local_best_members = work.members;
            }
        }

        // MPI_MAXLOC gives maximum cardinality and, on a tie, the lowest seed,
        // which is precisely the sequential algorithm's selection rule.
        int local_choice[2] = {local_cardinality, local_seed};
        int global_choice[2] = {-1, INT_MAX};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        const int best_cardinality = global_choice[0];
        const int best_seed = global_choice[1];
        if (best_seed == INT_MAX || best_cardinality <= 0) break;

        const auto seed_position = std::lower_bound(unclustered_indices.begin(),
                                                    unclustered_indices.end(), best_seed);
        const int owner = static_cast<int>(seed_position - unclustered_indices.begin()) %
                          rank_count;
        std::vector<int> best_cluster_members;
        if (rank == owner) best_cluster_members.swap(local_best_members);
        best_cluster_members.resize(static_cast<size_t>(best_cardinality));
        MPI_Bcast(best_cluster_members.data(), best_cardinality, MPI_INT, owner,
                  MPI_COMM_WORLD);

        if (rank == 0) clusters.push_back({best_cluster_members, best_seed});
        for (const int member : best_cluster_members) clustered[member] = 1;
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) { return clustered[index] != 0; }),
            unclustered_indices.end());
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
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

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
        printf("MPI ranks: %d\n", rank_count);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, rank_count);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    printf("Clustering time: %.3f ms\n", cluster_time * 1000.0);
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
    const double time_sec = cluster_time;
    const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
    const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}

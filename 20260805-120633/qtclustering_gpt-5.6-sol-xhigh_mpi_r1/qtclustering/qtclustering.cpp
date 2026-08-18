// QT Clustering Benchmark - MPI distributed-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>

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
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Per-rank reusable scratch space.  The generation array avoids clearing an
// N-element in-cluster bitmap for every candidate seed.
struct CandidateWorkspace {
    explicit CandidateWorkspace(const int point_count)
        : generation_by_point(point_count, 0), max_diameter(point_count) {
        members.reserve(point_count);
    }

    void beginCandidate() {
        ++generation;
        if (generation == 0) {
            std::fill(generation_by_point.begin(), generation_by_point.end(), 0);
            generation = 1;
        }
        members.clear();
    }

    std::vector<std::uint32_t> generation_by_point;
    std::vector<double> max_diameter;
    std::vector<int> members;
    std::uint32_t generation = 0;
};

// Reusing pairwise distances removes most square roots from the hot loop for
// normal benchmark sizes.  Bound the replicated cache so large distributed
// runs do not suffer unbounded per-rank memory growth.
std::vector<double> buildDistanceCache(const std::vector<Point>& points) {
    constexpr std::size_t MAX_CACHE_BYTES = 128ULL * 1024ULL * 1024ULL;
    const std::size_t count = points.size();
    if (count == 0 || count > MAX_CACHE_BYTES / sizeof(double) / count) {
        return {};
    }

    const std::size_t entries = count * count;
    std::vector<double> distances;
    try {
        distances.resize(entries);
    } catch (const std::bad_alloc&) {
        return {};
    }

    for (std::size_t member = 0; member < count; ++member) {
        double* const row = distances.data() + member * count;
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            row[candidate] = distance(points[candidate], points[member]);
        }
    }
    return distances;
}

// Generate exactly the same greedy candidate as the sequential algorithm.
// A candidate's current diameter is updated only for the newly added member;
// recomputing distances to all older members would produce the same maximum.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered_indices,
                             const std::vector<Point>& points,
                             const std::vector<double>& distance_cache,
                             const double threshold,
                             CandidateWorkspace& workspace) {
    const std::size_t point_count = points.size();
    workspace.beginCandidate();
    workspace.generation_by_point[seed_point] = workspace.generation;
    workspace.members.push_back(seed_point);

    int newest_member = seed_point;
    bool first_member = true;
    while (true) {
        const double* const cached_row = distance_cache.empty()
            ? nullptr
            : distance_cache.data() + static_cast<std::size_t>(newest_member) * point_count;
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // unclustered_indices is kept in ascending point order.  Strictly
        // improving comparisons therefore retain the original tie breaking.
        for (const int candidate : unclustered_indices) {
            if (workspace.generation_by_point[candidate] == workspace.generation) {
                continue;
            }

            const double dist = cached_row
                ? cached_row[candidate]
                : distance(points[candidate], points[newest_member]);
            double& candidate_diameter = workspace.max_diameter[candidate];
            if (first_member) {
                candidate_diameter = dist;
            } else {
                candidate_diameter = std::max(candidate_diameter, dist);
            }

            if (candidate_diameter < threshold && candidate_diameter < min_diameter) {
                min_diameter = candidate_diameter;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) {
            break;
        }
        workspace.generation_by_point[closest_point] = workspace.generation;
        workspace.members.push_back(closest_point);
        newest_member = closest_point;
        first_member = false;
    }

    return static_cast<int>(workspace.members.size());
}

// Candidate seeds are distributed cyclically across MPI ranks.  Every rank
// keeps the active-point state in lockstep; only rank zero retains the final
// cluster list needed for output and validation.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count,
                                  MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> active(N, 1);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    CandidateWorkspace workspace(N);
    const std::vector<double> distance_cache = buildDistanceCache(points);

    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = INT_MAX;
        std::vector<int> local_best_members;

        // Cyclic scheduling balances candidate counts and spreads spatially
        // adjacent (and often similarly expensive) seeds among ranks.
        for (std::size_t i = static_cast<std::size_t>(rank);
             i < unclustered_indices.size();
             i += static_cast<std::size_t>(process_count)) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, distance_cache, threshold, workspace);

            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = workspace.members;
            }
        }

        // MPI_MAXLOC gives maximum cardinality, then the lowest seed on ties,
        // exactly matching the sequential scan order.
        const int local_choice[2] = {local_max_cardinality, local_best_seed};
        int global_choice[2] = {-1, INT_MAX};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC, communicator);
        const int max_cardinality = global_choice[0];
        const int best_seed = global_choice[1];
        if (best_seed == INT_MAX || max_cardinality <= 0) {
            break;
        }

        const auto best_position = std::lower_bound(
            unclustered_indices.begin(), unclustered_indices.end(), best_seed);
        const int owner = static_cast<int>(
            std::distance(unclustered_indices.begin(), best_position)
            % static_cast<std::ptrdiff_t>(process_count));

        std::vector<int> best_cluster_members(static_cast<std::size_t>(max_cardinality));
        if (rank == owner) {
            best_cluster_members = local_best_members;
        }
        MPI_Bcast(best_cluster_members.data(), max_cardinality, MPI_INT, owner, communicator);

        if (rank == 0) {
            clusters.push_back(Cluster{best_cluster_members, best_seed});
        }
        for (const int member : best_cluster_members) {
            active[member] = 0;
        }
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&active](const int index) { return active[index] == 0; }),
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Error: MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI processes: %d\n", process_count);
    }

    // Rank zero produces the canonical data set, then distributes it once.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, process_count, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int exit_status = 0;
    if (rank != 0) {
        MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exit_status;
    }

    const long long cluster_milliseconds = static_cast<long long>(cluster_time * 1000.0);
    printf("Clustering time: %lld ms\n", cluster_milliseconds);
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
    const double measured_time = std::max(cluster_time, std::numeric_limits<double>::min());
    const double clusters_per_sec = clusters.size() / measured_time;
    const double points_per_sec = num_points / measured_time;
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
    
    // Validation is intentionally done once: all ranks constructed the same
    // clusters, while rank zero retained their serialized representation.
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exit_status = 1;
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_status;
}

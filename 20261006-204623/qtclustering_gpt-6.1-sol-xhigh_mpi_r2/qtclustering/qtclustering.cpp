// QT Clustering Benchmark - MPI distributed seed evaluation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
        // For N <= 30 the original expression always produced zero.
        if (N <= 30) group_cnt = 1;
        
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

// Cache only the candidates owned by this rank. An unchanged candidate remains
// exact when points outside its membership are removed: none of the greedy
// choices that produced it can change.
struct SeedCandidate {
    int seed;
    int upper_bound;
    std::vector<int> members;
    bool complete = false;
};

struct DiameterCandidate {
    int point;
    double diameter;
};

static bool canBeat(int cardinality, int seed, int best_cardinality, int best_seed) {
    return cardinality > best_cardinality ||
           (cardinality == best_cardinality && seed < best_seed);
}

// Grow a candidate using incremental maximum distances. The frontier contains
// only points within the threshold of every member so far. Each new member
// requires one distance per surviving point, rather than rescanning all members.
static bool generateCandidateCluster(SeedCandidate& candidate,
                                     const std::vector<int>& remaining,
                                     const std::vector<Point>& points,
                                     double threshold,
                                     std::vector<DiameterCandidate>& frontier,
                                     int best_cardinality, int best_seed) {
    candidate.complete = false;
    candidate.members.clear();
    candidate.members.push_back(candidate.seed);
    frontier.clear();

    int closest = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (int point : remaining) {
        if (point == candidate.seed) continue;
        const double dist = distance(points[point], points[candidate.seed]);
        if (dist < threshold) {
            frontier.push_back({point, dist});
            if (dist < min_diameter ||
                (dist == min_diameter && point < closest)) {
                min_diameter = dist;
                closest = point;
            }
        }
    }

    // Removing points cannot increase the number of eligible seed neighbors.
    candidate.upper_bound = static_cast<int>(frontier.size()) + 1;
    if (!canBeat(candidate.upper_bound, candidate.seed, best_cardinality, best_seed)) {
        return false;
    }

    while (closest >= 0) {
        const int member = closest;
        candidate.members.push_back(member);
        closest = -1;
        min_diameter = std::numeric_limits<double>::max();
        size_t retained = 0;

        for (DiameterCandidate point : frontier) {
            if (point.point == member) continue;
            const double dist = distance(points[point.point], points[member]);
            if (!(dist < threshold)) continue;

            point.diameter = std::max(point.diameter, dist);
            frontier[retained++] = point;
            // Explicit point-index ties reproduce the ascending serial scan.
            if (point.diameter < min_diameter ||
                (point.diameter == min_diameter && point.point < closest)) {
                min_diameter = point.diameter;
                closest = point.point;
            }
        }
        frontier.resize(retained);

        // Even adding every survivor cannot make this candidate win.
        if (!frontier.empty() &&
            !canBeat(static_cast<int>(candidate.members.size() + frontier.size()),
                     candidate.seed, best_cardinality, best_seed)) {
            return false;
        }
    }

    candidate.complete = true;
    return true;
}

// All ranks share the coordinates and removal state, but own disjoint seeds
// and candidate caches. Only the winning seed/size and its members communicate
// each round. The complete result is retained on rank zero.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, rank_count;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &rank_count);

    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> remaining(N);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<Cluster> clusters;

    // Cyclic ownership spreads nearby (and often similarly expensive) seeds
    // across ranks, including when there are fewer active seeds than ranks.
    std::vector<SeedCandidate> seeds;
    for (size_t seed = static_cast<size_t>(rank); seed < points.size();
         seed += static_cast<size_t>(rank_count)) {
        seeds.push_back({static_cast<int>(seed), N, {}, false});
    }

    std::vector<DiameterCandidate> frontier;
    frontier.reserve(N);
    std::vector<int> winning_members;
    while (!remaining.empty()) {
        seeds.erase(std::remove_if(seeds.begin(), seeds.end(),
                                   [&clustered](const SeedCandidate& seed) {
                                       return clustered[seed.seed] != 0;
                                   }), seeds.end());

        const int remaining_count = static_cast<int>(remaining.size());
        // Evaluate promising bounds first, so later candidates can be pruned.
        std::sort(seeds.begin(), seeds.end(),
                  [remaining_count](const SeedCandidate& a, const SeedCandidate& b) {
                      const int a_bound = std::min(a.upper_bound, remaining_count);
                      const int b_bound = std::min(b.upper_bound, remaining_count);
                      return a_bound != b_bound ? a_bound > b_bound : a.seed < b.seed;
                  });

        int best_cardinality = 0;
        int best_seed = std::numeric_limits<int>::max();
        const std::vector<int>* best_members = nullptr;
        for (SeedCandidate& seed : seeds) {
            const int bound = std::min(seed.upper_bound, remaining_count);
            if (!canBeat(bound, seed.seed, best_cardinality, best_seed)) break;

            if (seed.complete) {
                for (int member : seed.members) {
                    if (clustered[member]) {
                        seed.complete = false;
                        break;
                    }
                }
            }
            if (!seed.complete &&
                !generateCandidateCluster(seed, remaining, points, threshold,
                                          frontier, best_cardinality, best_seed)) {
                continue;
            }

            const int cardinality = static_cast<int>(seed.members.size());
            if (canBeat(cardinality, seed.seed, best_cardinality, best_seed)) {
                best_cardinality = cardinality;
                best_seed = seed.seed;
                best_members = &seed.members;
            }
        }

        // MINLOC on (-size, seed) selects largest size, then smallest seed,
        // exactly as the original ascending serial seed loop did.
        int local_best[2] = {-best_cardinality, best_seed};
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MINLOC, comm);
        const int cardinality = -global_best[0];
        const int seed = global_best[1];

        // A maximum size of one means no eligible pair remains. All subsequent
        // serial choices are singleton clusters in ascending point order.
        if (cardinality == 1) {
            if (rank == 0) {
                for (int point : remaining) clusters.push_back({{point}, point});
            }
            break;
        }

        winning_members.resize(cardinality);
        const int owner = seed % rank_count;
        if (rank == owner) {
            std::copy(best_members->begin(), best_members->end(), winning_members.begin());
        }
        MPI_Bcast(winning_members.data(), cardinality, MPI_INT, owner, comm);
        for (int member : winning_members) clustered[member] = 1;
        remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                       [&clustered](int point) {
                                           return clustered[point] != 0;
                                       }), remaining.end());

        if (rank == 0) {
            clusters.push_back({std::move(winning_members), seed});
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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse once and distribute the configuration, including early exits.
    int startup_status = 0;
    if (rank == 0) {
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
                startup_status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                startup_status = 2;
                break;
            }
        }

        if (startup_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points,
                   threshold);
            startup_status = 2;
        }
    }

    MPI_Bcast(&startup_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (startup_status != 0) {
        MPI_Finalize();
        return startup_status == 1 ? 0 : 1;
    }
    int options[3] = {num_points, validate, printResults};
    MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    num_points = options[0];
    validate = options[1] != 0;
    printResults = options[2] != 0;

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0)
        generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    const double local_time = MPI_Wtime() - cluster_start;
    double cluster_seconds = 0.0;
    MPI_Reduce(&local_time, &cluster_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int exit_status = 0;
    if (rank == 0) {
        const long long cluster_time_ms = static_cast<long long>(cluster_seconds * 1000.0);

        printf("Clustering time: %lld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        const double avg_cluster_size =
            clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = cluster_seconds;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

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
                exit_status = 1;
            }
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_status;
}

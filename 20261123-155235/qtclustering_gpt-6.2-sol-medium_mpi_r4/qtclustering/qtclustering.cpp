// QT Clustering Benchmark
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
        // The original group size cannot reliably reach one for small inputs.
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

// Each rank keeps the candidates for its own seeds. The x-sorted window
// bounds the points that could be within threshold of the seed.
struct SeedCandidate {
    int seed;
    int first;
    int last;
    int size_bound;
    std::vector<int> members;
};

struct Neighbor {
    int index;
    double max_distance;
};

bool closer(const Neighbor& a, const Neighbor& b) {
    return a.max_distance < b.max_distance ||
           (a.max_distance == b.max_distance && a.index < b.index);
}

void growCandidate(SeedCandidate& candidate, const std::vector<int>& sorted_by_x,
                   const std::vector<unsigned char>& clustered,
                   const std::vector<Point>& points, const double threshold,
                   std::vector<Neighbor>& neighbors) {
    neighbors.clear();
    const Point& seed = points[candidate.seed];
    for (int i = candidate.first; i < candidate.last; ++i) {
        const int index = sorted_by_x[i];
        if (index == candidate.seed || clustered[index]) continue;

        const double dist = std::max(0.0, distance(points[index], seed));
        if (dist < threshold) neighbors.push_back({index, dist});
    }
    candidate.size_bound = static_cast<int>(neighbors.size()) + 1;
    candidate.members.clear();
    candidate.members.push_back(candidate.seed);

    size_t closest = 0;
    for (size_t i = 1; i < neighbors.size(); ++i) {
        if (closer(neighbors[i], neighbors[closest])) closest = i;
    }

    while (!neighbors.empty()) {
        const int member = neighbors[closest].index;
        candidate.members.push_back(member);
        neighbors[closest] = neighbors.back();
        neighbors.pop_back();

        // A point's maximum distance can only increase as members are added.
        // Update it using the new member, compact away ineligible points,
        // and find the next closest point in the same pass.
        size_t kept = 0;
        closest = 0;
        for (Neighbor neighbor : neighbors) {
            neighbor.max_distance =
                std::max(neighbor.max_distance, distance(points[neighbor.index], points[member]));
            if (neighbor.max_distance < threshold) {
                neighbors[kept] = neighbor;
                if (closer(neighbor, neighbors[closest])) closest = kept;
                ++kept;
            }
        }
        neighbors.resize(kept);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                 const double threshold, const int rank, const int ranks) {
    const int n = static_cast<int>(points.size());
    std::vector<int> sorted_by_x(n);
    std::iota(sorted_by_x.begin(), sorted_by_x.end(), 0);
    const bool finite_points = std::all_of(points.begin(), points.end(), [](const Point& p) {
        return std::isfinite(p.x) && std::isfinite(p.y);
    });
    if (finite_points) {
        std::sort(sorted_by_x.begin(), sorted_by_x.end(), [&points](int a, int b) {
            return points[a].x < points[b].x;
        });
    }

    std::vector<SeedCandidate> candidates;
    // Cyclic ownership spreads the groups of nearby points across ranks.
    for (size_t index = rank; index < points.size(); index += ranks) {
        const int seed = static_cast<int>(index);
        int first = 0;
        int last = n;
        if (finite_points && threshold > 0.0) {
            const double x = points[seed].x;
            const auto within_x = [x, threshold, &points](int i) {
                const double dx = points[i].x - x;
                return std::sqrt(dx * dx) < threshold;
            };
            first = static_cast<int>(std::partition_point(
                sorted_by_x.begin(), sorted_by_x.end(), [&](int i) {
                    return points[i].x < x && !within_x(i);
                }) - sorted_by_x.begin());
            last = static_cast<int>(std::partition_point(
                sorted_by_x.begin() + first, sorted_by_x.end(), [&](int i) {
                    return points[i].x <= x || within_x(i);
                }) - sorted_by_x.begin());
        } else if (!(threshold > 0.0)) {
            // An unordered threshold makes every strict distance comparison fail.
            last = 0;
        }
        candidates.push_back({seed, first, last, std::max(1, last - first), {}});
    }

    std::vector<unsigned char> clustered(n, 0);
    std::vector<Cluster> clusters;
    std::vector<Neighbor> neighbors;
    std::vector<int> received_members;
    int remaining = n;
    while (remaining > 0) {
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        const SeedCandidate* local_winner = nullptr;
        for (SeedCandidate& candidate : candidates) {
            // Seeds are visited in ascending order, so an equal size cannot win.
            if (candidate.size_bound <= local_best[0]) continue;
            // Removing points outside the previous greedy sequence leaves every
            // choice in that sequence unchanged.
            if (candidate.members.empty() ||
                std::any_of(candidate.members.begin(), candidate.members.end(),
                            [&clustered](int i) { return clustered[i]; })) {
                growCandidate(candidate, sorted_by_x, clustered, points, threshold, neighbors);
            }

            const int size = static_cast<int>(candidate.members.size());
            if (size > local_best[0]) {
                local_best[0] = size;
                local_best[1] = candidate.seed;
                local_winner = &candidate;
                if (size == remaining) break;
            }
        }

        // MAXLOC selects the greatest cardinality and then the smallest seed.
        int best[2];
        MPI_Allreduce(local_best, best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int size = best[0];
        const int seed = best[1];
        if (size == 1) {
            // No eligible pair remains. The serial order is increasing seed.
            if (rank == 0) {
                for (int i = 0; i < n; ++i) {
                    if (!clustered[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }

        const int owner = seed % ranks;
        received_members.resize(size);
        if (rank == owner) {
            std::copy(local_winner->members.begin(), local_winner->members.end(),
                      received_members.begin());
        }
        MPI_Bcast(received_members.data(), size, MPI_INT, owner, MPI_COMM_WORLD);
        if (rank == 0) clusters.push_back({received_members, seed});

        remaining -= size;
        if (remaining == 0) break;
        for (const int member : received_members) clustered[member] = 1;
        std::erase_if(candidates, [&clustered](const SeedCandidate& candidate) {
            return clustered[candidate.seed];
        });
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    int exit_status = -1;
    if (rank == 0) {
        // Parse once so all ranks follow the same collective calls.
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
                exit_status = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_status = 1;
                break;
            }
        }

        if (exit_status < 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            exit_status = 1;
        }
    }

    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_status >= 0) return exit_status;
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }

    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double time_sec = 0.0;
    MPI_Reduce(&elapsed, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    printf("Clustering time: %lld ms\n", static_cast<long long>(time_sec * 1000.0));
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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}

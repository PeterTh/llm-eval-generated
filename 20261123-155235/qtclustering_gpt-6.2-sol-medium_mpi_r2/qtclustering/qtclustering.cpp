// QT Clustering Benchmark - MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        if (N <= 30) {
            // Small inputs still need to generate at least one point per group.
            group_cnt = 1;
        }
        
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

struct CandidatePoint {
    int index;
    double diameter;

    bool isCloserThan(const CandidatePoint& other) const {
        return diameter < other.diameter || (diameter == other.diameter && index < other.index);
    }
};

// Keep each eligible point's maximum distance to the members added so far.
void generateCandidateCluster(Cluster& cluster,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const std::vector<int>& points_by_x,
                              const bool use_x_bounds,
                              const double threshold,
                              std::vector<CandidatePoint>& candidates) {
    const int seed = cluster.seed_point;
    const Point& seed_point = points[seed];
    cluster.members.clear();
    cluster.members.push_back(seed);
    candidates.clear();

    auto first = points_by_x.begin();
    auto last = points_by_x.end();
    if (use_x_bounds) {
        // The rounded distance along x is a lower bound on the Euclidean distance.
        first = std::partition_point(first, last, [&](int index) {
            const double dx = points[index].x - seed_point.x;
            return dx < 0.0 && std::sqrt(dx * dx) >= threshold;
        });
        last = std::partition_point(first, last, [&](int index) {
            const double dx = points[index].x - seed_point.x;
            return dx <= 0.0 || std::sqrt(dx * dx) < threshold;
        });
    }

    size_t closest = 0;
    for (auto it = first; it != last; ++it) {
        const int index = *it;
        if (index == seed || clustered[index]) continue;
        const double diameter = std::max(0.0, distance(points[index], seed_point));
        if (diameter < threshold) {
            const CandidatePoint candidate{index, diameter};
            if (candidates.empty() || candidate.isCloserThan(candidates[closest])) {
                closest = candidates.size();
            }
            candidates.push_back(candidate);
        }
    }

    while (!candidates.empty()) {
        const size_t selected = closest;
        const int member = candidates[selected].index;
        cluster.members.push_back(member);
        const size_t count = candidates.size();
        size_t kept = 0;

        for (size_t i = 0; i < count; ++i) {
            if (i == selected) continue;
            const int index = candidates[i].index;
            const double diameter = std::max(candidates[i].diameter, distance(points[index], points[member]));
            if (diameter < threshold) {
                const CandidatePoint candidate{index, diameter};
                if (kept == 0 || candidate.isCloserThan(candidates[closest])) {
                    closest = kept;
                }
                candidates[kept++] = candidate;
            }
        }
        candidates.resize(kept);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, const int rank, const int ranks) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> points_by_x(point_count);
    std::iota(points_by_x.begin(), points_by_x.end(), 0);

    // An unordered distance leaves the running diameter unchanged in the serial algorithm.
    const bool use_x_bounds = std::all_of(points.begin(), points.end(), [](const Point& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
    });
    if (use_x_bounds) {
        std::sort(points_by_x.begin(), points_by_x.end(), [&](int a, int b) {
            return points[a].x < points[b].x;
        });
    }

    // Cyclic assignment spreads neighboring seed indices across ranks.
    std::vector<Cluster> local_clusters;
    const int local_count = point_count / ranks + (rank < point_count % ranks);
    local_clusters.reserve(local_count);
    for (int i = 0; i < local_count; ++i) {
        local_clusters.push_back({{}, rank + i * ranks});
    }

    std::vector<Cluster> clusters;
    std::vector<CandidatePoint> candidates;
    std::vector<int> members;
    int remaining = point_count;
    while (remaining > 0) {
        std::array<int, 2> local_best{0, point_count};
        const Cluster* best_cluster = nullptr;
        for (auto& cluster : local_clusters) {
            // Removing points outside this greedy sequence leaves every choice unchanged.
            if (cluster.members.empty() ||
                std::any_of(cluster.members.begin() + 1, cluster.members.end(),
                            [&](int index) { return clustered[index]; })) {
                generateCandidateCluster(cluster, clustered, points, points_by_x,
                                         use_x_bounds, threshold, candidates);
            }
            const int size = static_cast<int>(cluster.members.size());
            if (size > local_best[0]) {
                local_best = {size, cluster.seed_point};
                best_cluster = &cluster;
            }
            // Later seeds cannot beat a cluster containing every remaining point.
            if (size == remaining) break;
        }

        // MAXLOC selects the largest cardinality and then the lowest seed index.
        std::array<int, 2> winner;
        MPI_Allreduce(local_best.data(), winner.data(), 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const auto [cardinality, seed] = winner;

        if (cardinality == 1) {
            // Every remaining seed is a singleton, so all remaining choices are fixed.
            if (rank == 0) {
                for (int index = 0; index < point_count; ++index) {
                    if (!clustered[index]) clusters.push_back({{index}, index});
                }
            }
            break;
        }

        const int owner = seed % ranks;
        if (rank == owner) {
            members = best_cluster->members;
        } else {
            members.resize(cardinality);
        }
        MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD);

        if (rank == 0) clusters.push_back({members, seed});
        for (int index : members) clustered[index] = 1;
        remaining -= cardinality;
        std::erase_if(local_clusters, [&](const Cluster& cluster) {
            return clustered[cluster.seed_point];
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

int runBenchmark(int argc, char** argv, const int rank, const int ranks) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    const auto parseArguments = [&] {
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
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                return -1;
            }
        }

        if (num_points <= 0 || threshold <= 0.0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            return -1;
        }
        return num_points;
    };
    if (rank == 0) num_points = parseArguments();

    // Zero signals help and a negative value signals an argument error.
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_points <= 0) return num_points < 0;
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
    MPI_Datatype mpi_point;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point);
    MPI_Type_commit(&mpi_point);
    MPI_Bcast(points.data(), num_points, mpi_point, 0, MPI_COMM_WORLD);
    MPI_Type_free(&mpi_point);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_time_sec = MPI_Wtime() - start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;

    printf("Clustering time: %ld ms\n", static_cast<long>(time_sec * 1000.0));
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}

// QT Clustering Benchmark - MPI Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstddef>
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
    // At these sizes the original group scale cannot reliably generate points.
    const double group_scale = N <= 30 ? 2.0 : N / 30.0;
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * group_scale);
        
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
};

struct PointWindow {
    int begin;
    int end;
};

void generateCandidateCluster(int seed, const std::vector<Point>& points,
                              double threshold, const std::vector<int>& x_order,
                              const PointWindow window,
                              const std::vector<unsigned char>& clustered,
                              std::vector<CandidatePoint>& candidates,
                              std::vector<int>& members) {
    members.clear();
    members.push_back(seed);
    candidates.clear();

    for (int pos = window.begin; pos < window.end; ++pos) {
        const int index = x_order[pos];
        if (index != seed && !clustered[index]) candidates.push_back({index, 0.0});
    }

    // Maintain each candidate's maximum distance incrementally. Once it fails
    // the threshold, adding more members cannot make it eligible again.
    while (!candidates.empty()) {
        const Point& member = points[members.back()];
        CandidatePoint best{-1, std::numeric_limits<double>::max()};
        size_t best_position = 0;
        size_t eligible = 0;
        for (const auto& candidate : candidates) {
            const double diameter = std::max(candidate.diameter,
                                             distance(points[candidate.index], member));
            if (!(diameter < threshold)) continue;

            if (diameter < best.diameter ||
                (diameter == best.diameter && candidate.index < best.index)) {
                best = {candidate.index, diameter};
                best_position = eligible;
            }
            candidates[eligible++] = {candidate.index, diameter};
        }

        if (best.index < 0) break;
        members.push_back(best.index);
        candidates[best_position] = candidates[eligible - 1];
        candidates.resize(eligible - 1);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int rank_count) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, false);
    std::vector<Cluster> clusters;

    std::vector<int> x_order(point_count);
    std::iota(x_order.begin(), x_order.end(), 0);
    std::sort(x_order.begin(), x_order.end(), [&points](int a, int b) {
        return points[a].x < points[b].x;
    });

    // A pair outside the x window cannot pass the full distance check.
    // With a nonfinite y coordinate the distance may be NaN, which the
    // original maximum-distance update ignores, so use the full range then.
    std::vector<PointWindow> windows(point_count, {0, point_count});
    if (std::all_of(points.begin(), points.end(),
                    [](const Point& point) { return std::isfinite(point.y); })) {
        const auto outside_window = [&](int left, int right) {
            const double dx = points[x_order[right]].x - points[x_order[left]].x;
            return dx >= threshold && std::sqrt(dx * dx) >= threshold;
        };
        int left = 0;
        int right = 0;
        for (int pos = 0; pos < point_count; ++pos) {
            while (outside_window(left, pos)) ++left;
            while (right < point_count && !outside_window(pos, right)) ++right;
            windows[x_order[pos]] = {left, right};
        }
    }

    // Nearby indices from the generator tend to have similar densities.
    // Cyclic assignment spreads those neighborhoods across ranks.
    struct Seed {
        int index;
        std::vector<int> members;
    };
    std::vector<Seed> seeds;
    seeds.reserve(point_count / rank_count + (rank < point_count % rank_count));
    for (size_t i = rank; i < points.size(); i += rank_count) {
        seeds.push_back({static_cast<int>(i), {}});
    }

    std::vector<CandidatePoint> candidates;
    std::vector<int> received_members;
    int remaining = point_count;
    while (remaining > 0) {
        struct {
            int cardinality;
            int seed;
        } best{0, std::numeric_limits<int>::max()};
        std::vector<int>* local_members = nullptr;

        for (auto& seed : seeds) {
            // If all old members are available, removing other points cannot
            // change any choice in this seed's greedy sequence.
            if (seed.members.empty() ||
                std::any_of(seed.members.begin(), seed.members.end(),
                            [&clustered](int i) { return clustered[i]; })) {
                generateCandidateCluster(seed.index, points, threshold, x_order,
                                         windows[seed.index], clustered, candidates, seed.members);
            }

            const int cardinality = static_cast<int>(seed.members.size());
            if (cardinality > best.cardinality) {
                best = {cardinality, seed.index};
                local_members = &seed.members;
            }
            if (best.cardinality == remaining) break;
        }

        // MAXLOC selects the largest cluster and then the lowest seed index.
        MPI_Allreduce(MPI_IN_PLACE, &best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (best.cardinality == 1) {
            // No remaining pair fits the threshold. The serial loop would now
            // emit the remaining points in index order.
            if (rank == 0) {
                for (int i = 0; i < point_count; ++i) {
                    if (!clustered[i]) clusters.push_back({{i}, i});
                }
            }
            break;
        }

        const int owner = best.seed % rank_count;
        if (rank != owner) received_members.resize(best.cardinality);
        auto& winner = rank == owner ? *local_members : received_members;
        MPI_Bcast(winner.data(), best.cardinality, MPI_INT, owner, MPI_COMM_WORLD);

        if (rank == 0) clusters.push_back({winner, best.seed});
        remaining -= best.cardinality;
        if (remaining == 0) break;

        for (const int member : winner) clustered[member] = true;
        std::erase_if(seeds, [&clustered](const Seed& seed) { return clustered[seed.index]; });
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

struct Options {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results = false;
};

int parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            options.num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            options.threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.print_results = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (options.num_points <= 0 || options.threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
               options.num_points, options.threshold);
        return 1;
    }

    return -1;
}

int runBenchmark(int argc, char** argv, int rank, int rank_count) {
    Options options;
    // A positive count starts the benchmark. Otherwise, all ranks exit with
    // status zero for help and status one for invalid arguments.
    int num_points = 0;
    if (rank == 0) {
        const int status = parseOptions(argc, argv, options);
        num_points = status < 0 ? options.num_points : -status;
    }
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_points <= 0) return -num_points;

    double& threshold = options.threshold;
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Point> points(num_points);
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        generateSyntheticData(points, num_points);
    }

    static_assert(offsetof(Point, y) == sizeof(double) && sizeof(Point) == 2 * sizeof(double));
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, rank_count);
    const double elapsed = MPI_Wtime() - cluster_start;
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
    if (options.print_results) {
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
    if (options.validate) {
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;

    int rank;
    int rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    int status = runBenchmark(argc, argv, rank, rank_count);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

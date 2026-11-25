// QT Clustering Benchmark - MPI Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        
        // For small inputs the random group size can round down to zero every time.
        if (N <= 30) {
            group_cnt = std::max(1, group_cnt);
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

struct Candidate {
    int index;
    double diameter;
};

// Keep the maximum distance to members already visited, then update it once
// for each new member. A point that fails the threshold cannot qualify later.
static void generateCandidateCluster(int seed, const std::vector<int>& unclustered,
                                     const std::vector<Point>& points, double threshold,
                                     bool spatial_order,
                                     std::vector<Candidate>& candidates,
                                     std::vector<int>& members) {
    const Point& origin = points[seed];
    auto first = unclustered.begin();
    auto last = unclustered.end();
    if (spatial_order) {
        // The distance along the x axis alone bounds the full distance below.
        auto middle = std::lower_bound(first, last, origin.x, [&](int index, double x) {
            return points[index].x < x;
        });
        auto in_range = [&](int index) {
            const double dx = points[index].x - origin.x;
            return std::sqrt(dx * dx) < threshold;
        };
        first = std::partition_point(first, middle, [&](int index) { return !in_range(index); });
        last = std::partition_point(middle, last, in_range);
    }

    candidates.clear();
    double best_diameter = threshold;
    int best_index = -1;
    size_t best_position = 0;
    for (; first != last; ++first) {
        const int index = *first;
        if (index == seed) {
            continue;
        }
        const double diameter = std::max(0.0, distance(points[index], origin));
        if (diameter < threshold) {
            if (diameter < best_diameter || (diameter == best_diameter && index < best_index)) {
                best_diameter = diameter;
                best_index = index;
                best_position = candidates.size();
            }
            candidates.push_back({index, diameter});
        }
    }

    members.clear();
    members.push_back(seed);
    while (best_index >= 0) {
        members.push_back(best_index);
        const Point& newest = points[best_index];
        candidates[best_position] = candidates.back();
        candidates.pop_back();
        best_diameter = threshold;
        best_index = -1;
        size_t next_size = 0;
        for (const Candidate& candidate : candidates) {
            const double diameter = std::max(candidate.diameter,
                                             distance(points[candidate.index], newest));
            if (diameter < threshold) {
                if (diameter < best_diameter ||
                    (diameter == best_diameter && candidate.index < best_index)) {
                    best_diameter = diameter;
                    best_index = candidate.index;
                    best_position = next_size;
                }
                candidates[next_size++] = {candidate.index, diameter};
            }
        }
        candidates.resize(next_size);
    }
}

struct SeedCandidate {
    int seed;
    int size = 0;
    Point lower{};
    Point upper{};

    void remember(const std::vector<int>& members, const std::vector<Point>& points) {
        size = static_cast<int>(members.size());
        lower = upper = points[seed];
        for (int member : members) {
            const Point& point = points[member];
            lower.x = std::min(lower.x, point.x);
            lower.y = std::min(lower.y, point.y);
            upper.x = std::max(upper.x, point.x);
            upper.y = std::max(upper.y, point.y);
        }
    }

    void invalidate(const std::vector<int>& removed, const std::vector<Point>& points) {
        // Deleting a point outside the previous candidate cannot change any
        // greedy choice. The bounds allow reuse with constant storage per seed.
        if (size > 1 && std::any_of(removed.begin(), removed.end(), [&](int index) {
                const Point& point = points[index];
                return !(point.x < lower.x || point.x > upper.x ||
                         point.y < lower.y || point.y > upper.y);
            })) {
            size = 0;
        }
    }
};

static std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                         int rank, int rank_count) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered(point_count);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    const bool spatial_order = threshold < MAX_WIDTH &&
        std::all_of(points.begin(), points.end(), [](const Point& point) {
            return std::isfinite(point.x) && std::isfinite(point.y);
        });
    if (spatial_order) {
        std::sort(unclustered.begin(), unclustered.end(), [&](int a, int b) {
            return points[a].x < points[b].x;
        });
    }

    // Cyclic ownership spreads nearby seeds across ranks while keeping each
    // seed's cached evaluation on the same rank throughout clustering.
    std::vector<SeedCandidate> local_seeds;
    for (std::int64_t seed = rank; seed < point_count; seed += rank_count) {
        local_seeds.push_back({static_cast<int>(seed)});
    }

    std::vector<Candidate> candidates;
    std::vector<int> members;
    std::vector<int> best_members;
    std::vector<Cluster> clusters;
    while (!unclustered.empty()) {
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        bool have_best_members = false;
        for (SeedCandidate& candidate : local_seeds) {
            const bool evaluated = candidate.size == 0;
            if (evaluated) {
                generateCandidateCluster(candidate.seed, unclustered, points, threshold,
                                         spatial_order, candidates, members);
                candidate.remember(members, points);
            }
            if (candidate.size > local_best[0]) {
                local_best[0] = candidate.size;
                local_best[1] = candidate.seed;
                have_best_members = evaluated;
                if (evaluated) {
                    best_members.swap(members);
                }
                if (static_cast<size_t>(candidate.size) == unclustered.size()) {
                    break;
                }
            }
        }

        // MAXLOC prefers the lowest seed index among equal cardinalities,
        // matching the original ascending seed scan.
        int winner[2];
        MPI_Allreduce(local_best, winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (winner[0] == 1) {
            // Every remaining seed is a singleton, so the rest of the serial
            // selection order is already known.
            if (rank == 0) {
                for (int seed = 0; seed < point_count; ++seed) {
                    if (!clustered[seed]) {
                        clusters.push_back({{seed}, seed});
                    }
                }
            }
            break;
        }

        const int owner = winner[1] % rank_count;
        if (rank == owner) {
            if (!have_best_members) {
                generateCandidateCluster(winner[1], unclustered, points, threshold,
                                         spatial_order, candidates, best_members);
            }
        } else {
            best_members.resize(winner[0]);
        }
        MPI_Bcast(best_members.data(), winner[0], MPI_INT, owner, MPI_COMM_WORLD);

        if (rank == 0) {
            clusters.push_back({best_members, winner[1]});
        }
        if (best_members.size() == unclustered.size()) {
            break;
        }
        for (int member : best_members) {
            clustered[member] = 1;
        }
        std::erase_if(unclustered, [&](int index) { return clustered[index]; });
        std::erase_if(local_seeds, [&](const SeedCandidate& candidate) {
            return clustered[candidate.seed];
        });
        for (SeedCandidate& candidate : local_seeds) {
            candidate.invalidate(best_members, points);
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

static int parseArguments(int argc, char** argv, int num_points, double& threshold,
                          bool& validate, bool& printResults) {
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
}

static int runBenchmark(int argc, char** argv, int rank, int rank_count) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    if (rank == 0) {
        num_points = parseArguments(argc, argv, num_points, threshold, validate, printResults);
    }
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_points <= 0) {
        return num_points < 0 ? 1 : 0;
    }
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

    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, rank_count);
    const double elapsed = MPI_Wtime() - start;
    double time_sec = 0.0;
    MPI_Reduce(&elapsed, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        return 0;
    }

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
    int rank, rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    int status = runBenchmark(argc, argv, rank, rank_count);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

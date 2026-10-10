// QT Clustering Benchmark - Distributed MPI Version
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

struct Neighbor {
    int point;
    double max_distance;
};

struct SeedState {
    int seed;
    std::vector<Neighbor> neighbors;
    std::vector<int> members;
};

// Only the owner of a seed stores its threshold neighbors and cached cluster.
// A uniform grid avoids scanning all points when the threshold is small. Its
// cell width is at least the threshold and its number of cells is O(N).
std::vector<SeedState> buildSeedStates(const std::vector<Point>& points,
                                     const double threshold,
                                     const int rank, const int process_count) {
    const int N = static_cast<int>(points.size());
    std::vector<SeedState> seeds;
    seeds.reserve(N / process_count + (rank < N % process_count));
    for (size_t seed = rank; seed < points.size(); seed += process_count) {
        seeds.push_back({static_cast<int>(seed), {}, {}});
    }
    if (points.empty() || !(threshold > 0.0)) return seeds;

    double min_x = points[0].x, max_x = points[0].x;
    double min_y = points[0].y, max_y = points[0].y;
    for (const Point& point : points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    const double width = max_x - min_x;
    const double height = max_y - min_y;
    // Keep distances whose squared components underflow in the same search
    // neighborhood, as the original Euclidean calculation can round them to zero.
    const double minimum_cell_width = std::max(std::max(width, height) / std::sqrt(double(N)),
                                              std::sqrt(std::numeric_limits<double>::min()));
    const double cell_width = std::nextafter(std::max(minimum_cell_width, threshold),
                                           std::numeric_limits<double>::infinity());
    const int columns = static_cast<int>(width / cell_width) + 1;
    const int rows = static_cast<int>(height / cell_width) + 1;
    const size_t cell_count = static_cast<size_t>(columns) * rows;
    auto cell_x = [&](const Point& point) {
        return std::min(columns - 1, static_cast<int>((point.x - min_x) / cell_width));
    };
    auto cell_y = [&](const Point& point) {
        return std::min(rows - 1, static_cast<int>((point.y - min_y) / cell_width));
    };
    auto cell_id = [&](const Point& point) {
        return static_cast<size_t>(cell_y(point)) * columns + cell_x(point);
    };

    std::vector<size_t> offsets(cell_count + 1, 0);
    for (const Point& point : points) ++offsets[cell_id(point) + 1];
    for (size_t cell = 1; cell <= cell_count; ++cell) offsets[cell] += offsets[cell - 1];
    std::vector<size_t> next = offsets;
    std::vector<int> cell_points(N);
    for (int point = 0; point < N; ++point) {
        cell_points[next[cell_id(points[point])]++] = point;
    }

    for (SeedState& state : seeds) {
        const Point& seed = points[state.seed];
        const int x = cell_x(seed), y = cell_y(seed);
        for (int row = std::max(0, y - 1); row <= std::min(rows - 1, y + 1); ++row) {
            for (int column = std::max(0, x - 1); column <= std::min(columns - 1, x + 1); ++column) {
                const size_t cell = static_cast<size_t>(row) * columns + column;
                for (size_t pos = offsets[cell]; pos < offsets[cell + 1]; ++pos) {
                    const int candidate = cell_points[pos];
                    if (candidate == state.seed) continue;
                    const double dist = distance(points[candidate], seed);
                    if (dist < threshold) state.neighbors.push_back({candidate, dist});
                }
            }
        }
    }
    return seeds;
}

// Maintain each feasible point's maximum distance incrementally, updating it
// only against the newly added member. Removed points can never become feasible
// again. Explicit point-index ties preserve the original ascending scan even
// though the workspace uses swap-and-pop to discard candidates.
int generateCandidateCluster(SeedState& state,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             std::vector<Neighbor>& candidates,
                             const int best_cardinality) {
    candidates.clear();
    size_t closest_index = 0;
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (const Neighbor& neighbor : state.neighbors) {
        if (clustered[neighbor.point]) continue;
        if (neighbor.max_distance < min_diameter ||
            (neighbor.max_distance == min_diameter && neighbor.point < closest_point)) {
            min_diameter = neighbor.max_distance;
            closest_point = neighbor.point;
            closest_index = candidates.size();
        }
        candidates.push_back(neighbor);
    }
    // Seeds are visited in ascending order, so a tie cannot improve the winner.
    if (candidates.size() + 1 <= static_cast<size_t>(best_cardinality)) return 0;

    state.members.clear();
    state.members.push_back(state.seed);
    while (closest_point >= 0) {
        const int member = closest_point;
        state.members.push_back(member);
        candidates[closest_index] = candidates.back();
        candidates.pop_back();

        closest_point = -1;
        min_diameter = std::numeric_limits<double>::max();
        for (size_t i = 0; i < candidates.size();) {
            Neighbor& candidate = candidates[i];
            candidate.max_distance = std::max(candidate.max_distance,
                                             distance(points[candidate.point], points[member]));
            if (!(candidate.max_distance < threshold)) {
                candidate = candidates.back();
                candidates.pop_back();
                continue;
            }
            if (candidate.max_distance < min_diameter ||
                (candidate.max_distance == min_diameter && candidate.point < closest_point)) {
                min_diameter = candidate.max_distance;
                closest_point = candidate.point;
                closest_index = i;
            }
            ++i;
        }
    }
    return static_cast<int>(state.members.size());
}

// All ranks participate. Candidate construction is local; each iteration only
// reduces (cardinality, seed) and broadcasts the winning membership. Rank zero
// retains the final clusters for statistics, validation and result hashing.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<SeedState> seeds = buildSeedStates(points, threshold, rank, process_count);
    std::vector<size_t> active_seeds;
    active_seeds.reserve(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i) active_seeds.push_back(i);
    std::vector<Neighbor> candidates;
    std::vector<int> best_cluster_members;
    std::vector<Cluster> clusters;
    int remaining = N;

    while (remaining > 0) {
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        for (size_t index : active_seeds) {
            SeedState& state = seeds[index];
            if (state.neighbors.size() + 1 <= static_cast<size_t>(local_best[0])) continue;

            // If no cached member was removed, removing other candidates cannot
            // change any greedy choice or the point at which construction ends.
            bool reusable = !state.members.empty();
            for (int member : state.members) {
                if (clustered[member]) {
                    reusable = false;
                    break;
                }
            }
            const int cardinality = reusable ? static_cast<int>(state.members.size()) :
                generateCandidateCluster(state, clustered, points, threshold, candidates, local_best[0]);
            if (cardinality > local_best[0]) {
                local_best[0] = cardinality;
                local_best[1] = state.seed;
                if (cardinality == remaining) break;
            }
        }

        int global_best[2];
        // MPI_MAXLOC chooses the greatest size and the smallest seed on a tie.
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global_best[0] == 1) {
            // All remaining points are isolated: the sequential algorithm emits
            // singleton clusters in index order, without needing more elections.
            if (rank == 0) {
                for (int seed = 0; seed < N; ++seed) {
                    if (!clustered[seed]) clusters.push_back({{seed}, seed});
                }
            }
            break;
        }
        const int best_seed = global_best[1];
        const int owner = best_seed % process_count;
        if (rank == owner) {
            best_cluster_members = std::move(seeds[best_seed / process_count].members);
        } else {
            best_cluster_members.resize(global_best[0]);
        }
        MPI_Bcast(best_cluster_members.data(), global_best[0], MPI_INT, owner, MPI_COMM_WORLD);
        for (int member : best_cluster_members) {
            clustered[member] = 1;
            if (member % process_count == rank) {
                SeedState& state = seeds[member / process_count];
                std::vector<Neighbor>().swap(state.neighbors);
                std::vector<int>().swap(state.members);
            }
        }
        remaining -= global_best[0];
        active_seeds.erase(std::remove_if(active_seeds.begin(), active_seeds.end(),
            [&](size_t index) { return clustered[seeds[index].seed] != 0; }), active_seeds.end());
        if (rank == 0) clusters.push_back({std::move(best_cluster_members), best_seed});
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int command_status = 0; // 0: run, 1: help, 2: invalid arguments
    
    // Rank zero parses and reports arguments; distribute one configuration to
    // every rank so all early exits and collective calls follow the same path.
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
                command_status = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                command_status = 2;
                break;
            }
        }
    
        if (command_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
            command_status = 2;
        }
    }
    int configuration[4] = {num_points, validate, printResults, command_status};
    MPI_Bcast(configuration, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    num_points = configuration[0];
    validate = configuration[1] != 0;
    printResults = configuration[2] != 0;
    command_status = configuration[3];
    if (command_status != 0) {
        MPI_Finalize();
        return command_status == 1 ? 0 : 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate the original random stream once and replicate only coordinates.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two packed doubles");
    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);
    
    // Include distributed neighbor preparation and report the slowest rank.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int exit_status = 0;
    if (rank == 0) {
    
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

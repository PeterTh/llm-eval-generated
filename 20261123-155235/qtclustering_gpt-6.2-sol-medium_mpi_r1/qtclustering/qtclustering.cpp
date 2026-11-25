// QT Clustering Benchmark - MPI Version
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
        // Small inputs need a positive group size to make progress.
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

// Power-of-two cell widths keep cell boundaries exact. A neighbor within
// the threshold can only lie in the seed's cell or an adjacent cell.
class PointGrid {
public:
    PointGrid(const std::vector<Point>& points, double threshold) {
        while (width_ * 0.5 >= threshold &&
               4ULL * side_ * side_ <= points.size()) {
            width_ *= 0.5;
            side_ *= 2;
        }
        cells_.resize(side_ * side_);
        for (size_t i = 0; i < points.size(); ++i) {
            const Point& point = points[i];
            if (std::isfinite(point.x) && std::isfinite(point.y)) {
                cells_[cellCoordinate(point.y) * side_ + cellCoordinate(point.x)]
                    .push_back(static_cast<int>(i));
            } else {
                nonfinite_points_.push_back(static_cast<int>(i));
            }
        }
    }

    template <typename Visit>
    void visitNeighbors(const Point& point, Visit visit) const {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
            const int x = cellCoordinate(point.x);
            const int y = cellCoordinate(point.y);
            for (int row = std::max(0, y - 1); row <= std::min(side_ - 1, y + 1); ++row) {
                for (int col = std::max(0, x - 1); col <= std::min(side_ - 1, x + 1); ++col) {
                    for (int index : cells_[row * side_ + col]) visit(index);
                }
            }
        } else {
            for (const auto& cell : cells_) {
                for (int index : cell) visit(index);
            }
        }
        for (int index : nonfinite_points_) visit(index);
    }

private:
    int cellCoordinate(double coordinate) const {
        return static_cast<int>(coordinate / width_);
    }

    // The generated points lie in [0, 20] on both axes.
    double width_ = 32.0;
    int side_ = 1;
    std::vector<std::vector<int>> cells_;
    std::vector<int> nonfinite_points_;
};

struct SeedCandidate {
    int seed;
    int upper_bound;
    std::vector<int> members;
};

struct Neighbor {
    int index;
    double diameter;
};

bool closer(const Neighbor& a, const Neighbor& b) {
    return a.diameter == b.diameter ? a.index < b.index : a.diameter < b.diameter;
}

void generateCandidateCluster(SeedCandidate& candidate,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const PointGrid& grid,
                              double threshold, int best_size,
                              std::vector<Neighbor>& neighbors) {
    candidate.members.clear();
    neighbors.clear();
    const int seed = candidate.seed;
    size_t closest = 0;
    if (threshold > 0.0) {
        grid.visitNeighbors(points[seed], [&](int index) {
            if (index == seed || clustered[index]) return;
            // Match the original maximum starting at zero even if a distance is NaN.
            const Neighbor neighbor{index, std::max(0.0, distance(points[index], points[seed]))};
            if (neighbor.diameter < threshold) {
                if (neighbors.empty() || closer(neighbor, neighbors[closest])) {
                    closest = neighbors.size();
                }
                neighbors.push_back(neighbor);
            }
        });
    }
    // The remaining seed neighbors bound any future cluster from this seed.
    candidate.upper_bound = static_cast<int>(neighbors.size()) + 1;
    if (candidate.upper_bound <= best_size) return;

    candidate.members.push_back(seed);
    while (!neighbors.empty()) {
        const int member = neighbors[closest].index;
        candidate.members.push_back(member);
        neighbors[closest] = neighbors.back();
        neighbors.pop_back();

        // Update each maximum using only the new member. A rejected neighbor
        // cannot fit after the cluster grows further.
        size_t kept = 0;
        closest = 0;
        for (Neighbor neighbor : neighbors) {
            neighbor.diameter = std::max(neighbor.diameter,
                                         distance(points[neighbor.index], points[member]));
            if (neighbor.diameter < threshold) {
                if (kept == 0 || closer(neighbor, neighbors[closest])) closest = kept;
                neighbors[kept++] = neighbor;
            }
        }
        neighbors.resize(kept);
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int rank_count) {
    const int point_count = static_cast<int>(points.size());
    const PointGrid grid(points, threshold);
    std::vector<unsigned char> clustered(point_count, false);
    std::vector<SeedCandidate> local_seeds;
    local_seeds.reserve(point_count / rank_count + (rank < point_count % rank_count));
    // Cyclic ownership spreads seeds from the same generated group across ranks.
    for (size_t seed = rank; seed < points.size(); seed += rank_count) {
        local_seeds.push_back({static_cast<int>(seed), point_count, {}});
    }

    std::vector<Neighbor> neighbors;
    std::vector<int> winning_members;
    std::vector<Cluster> clusters;
    int remaining = point_count;
    while (remaining > 0) {
        int local_best[2] = {0, std::numeric_limits<int>::max()};
        std::vector<int>* local_members = nullptr;
        for (SeedCandidate& candidate : local_seeds) {
            if (candidate.upper_bound <= local_best[0]) continue;
            // Removing points outside the previous greedy sequence cannot
            // change any of its choices.
            if (candidate.members.empty() ||
                std::any_of(candidate.members.begin(), candidate.members.end(),
                            [&](int index) { return clustered[index]; })) {
                generateCandidateCluster(candidate, clustered, points, grid, threshold,
                                         local_best[0], neighbors);
            }
            const int size = static_cast<int>(candidate.members.size());
            if (size > local_best[0]) {
                local_best[0] = size;
                local_best[1] = candidate.seed;
                local_members = &candidate.members;
                if (size == remaining) break;
            }
        }

        // MAXLOC prefers the smallest seed when cardinalities tie.
        int global_best[2];
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int size = global_best[0];
        const int seed = global_best[1];
        if (size == 1) {
            // With no eligible pair left, the remaining seeds win in index order.
            if (rank == 0) {
                for (int index = 0; index < point_count; ++index) {
                    if (!clustered[index]) clusters.push_back({{index}, index});
                }
            }
            break;
        }

        const int owner = seed % rank_count;
        if (rank != owner) {
            winning_members.resize(size);
            local_members = &winning_members;
        }
        const auto& members = *local_members;
        MPI_Bcast(local_members->data(), size, MPI_INT, owner, MPI_COMM_WORLD);

        if (rank == 0) clusters.push_back({members, seed});
        remaining -= size;
        if (remaining == 0) break;
        for (int member : members) clustered[member] = true;
        std::erase_if(local_seeds, [&](const SeedCandidate& candidate) {
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

int runBenchmark(int argc, char** argv, int rank, int rank_count) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

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
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, rank_count);
    const double elapsed = MPI_Wtime() - start;
    double time_sec = 0.0;
    MPI_Reduce(&elapsed, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
    int rank, rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    const int result = runBenchmark(argc, argv, rank, rank_count);
    MPI_Finalize();
    return result;
}

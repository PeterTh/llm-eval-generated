// QT Clustering Benchmark - OpenMP Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <omp.h>

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
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// A full, row-major table makes every candidate update a contiguous read.  It
// also removes the large amount of duplicate square-root work done while
// evaluating different seeds.  Rows are initialized in parallel so that the
// storage is first-touched by the OpenMP team on NUMA machines.
class DistanceMatrix {
public:
    DistanceMatrix(const std::vector<Point>& points, const int point_count)
        : point_count_(point_count),
          values_(new double[static_cast<size_t>(point_count) * point_count]) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < point_count_; ++i) {
            double* const row = values_.get() + static_cast<size_t>(i) * point_count_;
            for (int j = 0; j < point_count_; ++j) {
                row[j] = distance(points[i], points[j]);
            }
        }
    }

    const double* row(const int point) const {
        return values_.get() + static_cast<size_t>(point) * point_count_;
    }

private:
    int point_count_;
    std::unique_ptr<double[]> values_;
};

// Each OpenMP worker retains its scratch array for the entire clustering
// operation, avoiding allocation in the seed-evaluation hot path.
class CandidateWorkspace {
public:
    explicit CandidateWorkspace(const int point_count)
        : max_distances_(point_count) {}

    std::vector<double>& maxDistances() { return max_distances_; }

private:
    std::vector<double> max_distances_;
};

// Generate the same greedy candidate as the sequential implementation.  Once
// a member has been added, only its distance can increase a candidate's
// diameter, so retaining each running maximum changes the work from repeated
// scans of all members to one scan of the points per added member.
int generateCandidateCluster(const int seed_point,
                             const int seed_position,
                             const std::vector<int>& unclustered_indices,
                             const int unclustered_count,
                             const DistanceMatrix& distances,
                             const double threshold,
                             CandidateWorkspace& workspace,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<double>& max_distances = workspace.maxDistances();
    int cardinality = 1;
    int newest_member = seed_point;

    if (cluster_members != nullptr) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    // Infinity marks members already selected into this candidate.  Keeping
    // scratch values in active-list order makes them dense and SIMD-friendly.
    const double infinity = std::numeric_limits<double>::infinity();
    const double* new_distances = distances.row(newest_member);
    double min_diameter = threshold;
#pragma omp simd reduction(min:min_diameter)
    for (int i = 0; i < unclustered_count; ++i) {
        const double max_dist = i == seed_position
            ? infinity
            : new_distances[unclustered_indices[i]];
        max_distances[i] = max_dist;
        min_diameter = std::min(min_diameter, max_dist);
    }

    while (cardinality < unclustered_count && min_diameter < threshold) {
        // The reduction determines the value; this ordered scan implements
        // the original first-candidate-wins tie-break.
        int closest_position = 0;
        while (max_distances[closest_position] != min_diameter) {
            ++closest_position;
        }

        const int closest_point = unclustered_indices[closest_position];
        max_distances[closest_position] = infinity;
        newest_member = closest_point;
        ++cardinality;
        if (cluster_members != nullptr) {
            cluster_members->push_back(closest_point);
        }

        if (cardinality == unclustered_count) {
            break;
        }

        new_distances = distances.row(newest_member);
        min_diameter = threshold;
#pragma omp simd reduction(min:min_diameter)
        for (int i = 0; i < unclustered_count; ++i) {
            const double max_dist = std::max(
                max_distances[i], new_distances[unclustered_indices[i]]);
            max_distances[i] = max_dist;
            min_diameter = std::min(min_diameter, max_dist);
        }
    }

    return cardinality;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const DistanceMatrix distances(points, N);
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    int unclustered_count = N;
    bool finished = false;
    std::uint64_t best_key = 0;

    // Keep one team alive across all QT rounds.  Candidate clusters from the
    // current unclustered set are independent; only the deterministic winner
    // is committed between rounds.  Encoding cardinality and inverse seed in
    // one reduction key preserves the sequential cardinality/seed tie-break.
#pragma omp parallel shared(best_key, clustered, clusters, finished, unclustered_count, unclustered_indices)
    {
        CandidateWorkspace workspace(N);

        while (true) {
#pragma omp single
            best_key = 0;

#pragma omp for schedule(guided, 1) reduction(max:best_key)
            for (int i = 0; i < unclustered_count; ++i) {
                const int seed = unclustered_indices[i];
                const int cardinality = generateCandidateCluster(
                    seed, i, unclustered_indices, unclustered_count,
                    distances, threshold, workspace);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cardinality)) << 32) |
                    (std::numeric_limits<std::uint32_t>::max() -
                     static_cast<std::uint32_t>(seed));
                best_key = std::max(best_key, key);
            }

#pragma omp single
            {
                if (best_key == 0) {
                    finished = true;
                } else {
                    const int best_seed = static_cast<int>(
                        std::numeric_limits<std::uint32_t>::max() -
                        static_cast<std::uint32_t>(best_key));
                    const int best_position = static_cast<int>(std::lower_bound(
                        unclustered_indices.begin(),
                        unclustered_indices.begin() + unclustered_count,
                        best_seed) - unclustered_indices.begin());

                    Cluster best_cluster;
                    best_cluster.seed_point = best_seed;
                    generateCandidateCluster(best_seed, best_position,
                                             unclustered_indices,
                                             unclustered_count, distances,
                                             threshold, workspace,
                                             &best_cluster.members);

                    for (const int member : best_cluster.members) {
                        clustered[member] = 1;
                    }
                    clusters.push_back(std::move(best_cluster));

                    int next_count = 0;
                    for (int i = 0; i < unclustered_count; ++i) {
                        const int point = unclustered_indices[i];
                        if (!clustered[point]) {
                            unclustered_indices[next_count++] = point;
                        }
                    }
                    unclustered_count = next_count;
                    finished = unclustered_count == 0;
                }
            }

            if (finished) {
                break;
            }
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

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

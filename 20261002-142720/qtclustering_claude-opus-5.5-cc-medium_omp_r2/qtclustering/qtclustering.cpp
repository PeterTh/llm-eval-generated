// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Per-thread scratch space for growing a candidate cluster.
// Candidates are stored in structure-of-arrays form, ordered by point index,
// together with the running maximum distance to the current cluster members.
struct GrowScratch {
    std::vector<int> idx;
    std::vector<double> cx, cy, maxd;

    void reserve(size_t n) {
        idx.resize(n);
        cx.resize(n);
        cy.resize(n);
        maxd.resize(n);
    }
};

// Generate a candidate cluster starting from a seed point.
// Returns the cardinality (size) of the cluster.
//
// Equivalent to repeatedly selecting the unclustered candidate (lowest index on
// ties) whose maximum distance to all current members is minimal and below the
// threshold. The max distance of each candidate is maintained incrementally;
// since it can only grow, candidates reaching the threshold are discarded.
// Only points closer than threshold to the seed can ever join.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             GrowScratch& s,
                             std::vector<int>* cluster_members = nullptr) {
    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    int* __restrict idx = s.idx.data();
    double* __restrict cx = s.cx.data();
    double* __restrict cy = s.cy.data();
    double* __restrict maxd = s.maxd.data();

    // Initial candidate set: unclustered points within threshold of the seed
    const Point sp = points[seed_point];
    int n = 0;
    int best = -1;
    double best_d = std::numeric_limits<double>::max();
    const int U = static_cast<int>(unclustered.size());
    for (int u = 0; u < U; ++u) {
        const int c = unclustered[u];
        if (c == seed_point) continue;
        const Point& p = points[c];
        const double dist = distance(p, sp);
        const double md = std::max(0.0, dist);
        if (md < threshold) {
            if (md < best_d) {
                best_d = md;
                best = n;
            }
            idx[n] = c;
            cx[n] = p.x;
            cy[n] = p.y;
            maxd[n] = md;
            ++n;
        }
    }

    int size = 1;
    while (best >= 0) {
        // Add the best candidate to the cluster
        const double mx = cx[best];
        const double my = cy[best];
        if (cluster_members) cluster_members->push_back(idx[best]);
        ++size;

        // Update max distances, drop the chosen point and points that exceed
        // the threshold, and find the next best candidate (stable order).
        const int chosen = best;
        best = -1;
        best_d = std::numeric_limits<double>::max();
        int m = 0;
        for (int k = 0; k < n; ++k) {
            if (k == chosen) continue;
            const double dx = cx[k] - mx;
            const double dy = cy[k] - my;
            const double dist = std::sqrt(dx * dx + dy * dy);
            const double md = std::max(maxd[k], dist);
            if (md < threshold) {
                if (md < best_d) {
                    best_d = md;
                    best = m;
                }
                idx[m] = idx[k];
                cx[m] = cx[k];
                cy[m] = cy[k];
                maxd[m] = md;
                ++m;
            }
        }
        n = m;
    }

    return size;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;

    // Cached cluster cardinality per seed; valid until a point within
    // threshold of the seed (i.e. a potential member) becomes clustered.
    std::vector<int> cardinality(N, 0);
    std::vector<char> dirty(N, 1);

    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    std::vector<int> best_cluster_members;
    best_cluster_members.reserve(N);
    bool done = false;

    // Avoid oversubscribing tiny problems with more threads than useful work
    const int num_threads = std::max(1, std::min(omp_get_max_threads(), N / 32));

    #pragma omp parallel num_threads(num_threads)
    {
        GrowScratch scratch;
        scratch.reserve(N);

        while (true) {
            // Invalidate seeds that had a point of the most recently formed
            // cluster within threshold (a potential member), and recompute
            // the cardinality of all invalidated seeds.
            const int U = static_cast<int>(unclustered_indices.size());
            const int B = static_cast<int>(best_cluster_members.size());
            #pragma omp for schedule(dynamic, 1)
            for (int u = 0; u < U; ++u) {
                const int seed = unclustered_indices[u];
                if (!dirty[seed]) {
                    const Point sp = points[seed];
                    for (int b = 0; b < B; ++b) {
                        if (distance(points[best_cluster_members[b]], sp) < threshold) {
                            dirty[seed] = 1;
                            break;
                        }
                    }
                }
                if (dirty[seed]) {
                    cardinality[seed] = generateCandidateCluster(seed, unclustered_indices, points,
                                                                 threshold, scratch);
                    dirty[seed] = 0;
                }
            }

            #pragma omp single
            {
                // Pick the seed with maximum cardinality (first one on ties)
                int max_cardinality = -1;
                int best_seed = -1;
                for (const int seed : unclustered_indices) {
                    if (cardinality[seed] > max_cardinality) {
                        max_cardinality = cardinality[seed];
                        best_seed = seed;
                    }
                }

                if (best_seed >= 0 && max_cardinality > 0) {
                    generateCandidateCluster(best_seed, unclustered_indices, points,
                                             threshold, scratch, &best_cluster_members);
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = best_cluster_members;
                    clusters.push_back(std::move(cluster));

                    for (const int m : best_cluster_members) {
                        clustered[m] = 1;
                    }

                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](int idx) { return clustered[idx]; }),
                        unclustered_indices.end());
                    done = unclustered_indices.empty();
                } else {
                    done = true;
                }
            }
            // Implicit barrier above makes 'done' visible to all threads
            if (done) break;
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

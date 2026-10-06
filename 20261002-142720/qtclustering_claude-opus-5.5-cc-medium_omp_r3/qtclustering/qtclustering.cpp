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
#include <set>
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

// Uniform grid over the points with cell size >= threshold, so every point
// within distance < threshold of a point lies in its 3x3 cell neighbourhood.
struct Grid {
    double min_x = 0.0, min_y = 0.0, cell = 1.0;
    int gx = 1, gy = 1;
    std::vector<int> cell_start; // size gx*gy+1
    std::vector<int> cell_pts;   // point indices sorted by cell

    void build(const std::vector<Point>& points, const double threshold) {
        const int N = static_cast<int>(points.size());
        double max_x = 0.0, max_y = 0.0;
        min_x = min_y = 0.0;
        if (N > 0) {
            min_x = max_x = points[0].x;
            min_y = max_y = points[0].y;
        }
        for (int i = 1; i < N; ++i) {
            min_x = std::min(min_x, points[i].x); max_x = std::max(max_x, points[i].x);
            min_y = std::min(min_y, points[i].y); max_y = std::max(max_y, points[i].y);
        }
        const double extent = std::max(max_x - min_x, max_y - min_y);
        const int max_dim = std::max(1, std::min(2048, static_cast<int>(2.0 * std::sqrt(static_cast<double>(N))) + 1));
        cell = std::max(threshold, extent / max_dim);
        if (!(cell > 0.0)) cell = 1.0;
        gx = std::min(max_dim, static_cast<int>((max_x - min_x) / cell) + 1);
        gy = std::min(max_dim, static_cast<int>((max_y - min_y) / cell) + 1);

        std::vector<int> cell_of(N);
        cell_start.assign(static_cast<size_t>(gx) * gy + 1, 0);
        for (int i = 0; i < N; ++i) {
            int cx, cy;
            cellCoords(points[i], cx, cy);
            cell_of[i] = cy * gx + cx;
            cell_start[cell_of[i] + 1]++;
        }
        for (size_t c = 1; c < cell_start.size(); ++c) cell_start[c] += cell_start[c - 1];
        cell_pts.resize(N);
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int i = 0; i < N; ++i) cell_pts[fill[cell_of[i]]++] = i;
    }

    inline void cellCoords(const Point& p, int& cx, int& cy) const {
        cx = std::clamp(static_cast<int>((p.x - min_x) / cell), 0, gx - 1);
        cy = std::clamp(static_cast<int>((p.y - min_y) / cell), 0, gy - 1);
    }

    // Call f(q) for every point q in the 3x3 neighbourhood of p's cell
    template <typename F>
    inline void forNeighbourhood(const Point& p, F&& f) const {
        int cx, cy;
        cellCoords(p, cx, cy);
        for (int y = std::max(0, cy - 1); y <= std::min(gy - 1, cy + 1); ++y) {
            const int row = y * gx;
            const int x0 = std::max(0, cx - 1), x1 = std::min(gx - 1, cx + 1);
            for (int k = cell_start[row + x0]; k < cell_start[row + x1 + 1]; ++k) {
                f(cell_pts[k]);
            }
        }
    }
};

// Per-thread scratch buffers for candidate cluster generation
struct CandidateScratch {
    std::vector<int> idx;
    std::vector<double> x, y, max_dist;
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster.
//
// Equivalent to iteratively adding the unclustered point whose maximum distance
// to all current members is smallest (ties -> smallest point index), as long as
// that maximum distance is < threshold. The maximum distance of each candidate
// is maintained incrementally, and candidates are restricted to points within
// threshold of the seed (all others can never be added).
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const Grid& grid,
                             const double threshold,
                             CandidateScratch& s,
                             std::vector<int>* cluster_members = nullptr) {
    s.idx.clear(); s.x.clear(); s.y.clear(); s.max_dist.clear();
    const Point& sp = points[seed_point];
    grid.forNeighbourhood(sp, [&](int q) {
        if (q == seed_point || clustered[q]) return;
        const double d = distance(points[q], sp);
        if (d < threshold) {
            s.idx.push_back(q);
            s.x.push_back(points[q].x);
            s.y.push_back(points[q].y);
            s.max_dist.push_back(d);
        }
    });

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    int size = 1;
    int m = static_cast<int>(s.idx.size());
    int* __restrict ci = s.idx.data();
    double* __restrict cx = s.x.data();
    double* __restrict cy = s.y.data();
    double* __restrict cd = s.max_dist.data();

    // Candidate with minimal max distance (ties -> smallest point index)
    auto better = [](double d, int i, double bd, int bi) {
        return d < bd || (d == bd && i < bi);
    };
    int best = 0;
    for (int j = 1; j < m; ++j) {
        if (better(cd[j], ci[j], cd[best], ci[best])) best = j;
    }

    while (m > 0) {
        ++size;
        if (cluster_members) cluster_members->push_back(ci[best]);
        const double nx = cx[best], ny = cy[best];

        // Update max distances with the new member, drop candidates that can
        // no longer be added (max distance only grows) and track the next best.
        int w = 0;
        int next = -1;
        double next_d = 0.0;
        int next_i = 0;
        for (int j = 0; j < m; ++j) {
            if (j == best) continue;
            const double dx = cx[j] - nx;
            const double dy = cy[j] - ny;
            const double nd = std::max(cd[j], std::sqrt(dx * dx + dy * dy));
            if (nd < threshold) {
                const int id = ci[j];
                ci[w] = id; cx[w] = cx[j]; cy[w] = cy[j]; cd[w] = nd;
                if (next < 0 || better(nd, id, next_d, next_i)) {
                    next = w; next_d = nd; next_i = id;
                }
                ++w;
            }
        }
        m = w;
        best = next;
    }
    return size;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    Grid grid;
    grid.build(points, threshold);

    std::vector<char> clustered(N, 0);
    std::vector<char> dirty(N, 0);
    std::vector<int> cardinality(N, 0);

    // Candidate cardinalities of a seed only depend on the clustered state of
    // points within threshold of the seed; they are cached and recomputed only
    // when such a point becomes clustered.
    std::vector<int> work(N);
    for (int i = 0; i < N; ++i) work[i] = i;

    // Ordered by (largest cardinality, smallest seed index)
    auto key = [](int card, int seed) { return (static_cast<long long>(-card) << 32) | static_cast<unsigned>(seed); };
    std::set<long long> ranking;

    std::vector<std::vector<int>> thread_dirty(omp_get_max_threads());
    CandidateScratch main_scratch;
    Cluster cluster;
    int remaining = N;

    // A single persistent parallel region: work-sharing loops for the parallel
    // phases, 'single' blocks (with implicit barriers) for the serial updates.
    #pragma omp parallel
    {
        CandidateScratch scratch;
        std::vector<int>& local = thread_dirty[omp_get_thread_num()];

        while (remaining > 0) {
            // (Re)compute candidate cluster cardinalities of dirty seeds
            const int W = static_cast<int>(work.size());
            #pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < W; ++k) {
                const int seed = work[k];
                cardinality[seed] = generateCandidateCluster(seed, clustered, points, grid,
                                                             threshold, scratch);
            }

            #pragma omp single
            {
                for (int k = 0; k < W; ++k) {
                    const int seed = work[k];
                    ranking.insert(key(cardinality[seed], seed));
                    dirty[seed] = 0;
                }

                // Best seed: maximum cardinality, first (smallest index) on ties
                const int best_seed = static_cast<int>(static_cast<unsigned>(*ranking.begin() & 0xffffffffLL));

                cluster.seed_point = best_seed;
                generateCandidateCluster(best_seed, clustered, points, grid, threshold,
                                         main_scratch, &cluster.members);

                for (const int p : cluster.members) {
                    clustered[p] = 1;
                    ranking.erase(key(cardinality[p], p));
                }
                remaining -= static_cast<int>(cluster.members.size());
            }

            // Mark unclustered seeds within threshold of new members as dirty
            const int M = static_cast<int>(cluster.members.size());
            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < M; ++i) {
                const Point& pp = points[cluster.members[i]];
                grid.forNeighbourhood(pp, [&](int q) {
                    if (clustered[q]) return;
                    if (distance(points[q], pp) < threshold) {
                        char old;
                        #pragma omp atomic capture
                        { old = dirty[q]; dirty[q] = 1; }
                        if (!old) local.push_back(q);
                    }
                });
            }

            #pragma omp single
            {
                work.clear();
                for (auto& td : thread_dirty) {
                    for (const int q : td) {
                        ranking.erase(key(cardinality[q], q));
                        work.push_back(q);
                    }
                    td.clear();
                }
                clusters.push_back(std::move(cluster));
                cluster = Cluster();
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

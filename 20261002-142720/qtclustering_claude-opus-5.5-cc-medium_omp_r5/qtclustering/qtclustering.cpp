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

// Per-thread scratch space for building candidate clusters.
// Candidates are kept in increasing index order (structure of arrays) together
// with their current maximum distance to all cluster members.
struct CandidateWorkspace {
    std::vector<int> idx;
    std::vector<double> cx, cy, maxd;
    std::vector<int> members;

    void reserve(const int n) {
        idx.resize(n);
        cx.resize(n);
        cy.resize(n);
        maxd.resize(n);
        members.reserve(n);
    }
};

// Optional precomputed neighbor lists (points within threshold, increasing index order)
struct NeighborLists {
    bool available = false;
    std::vector<size_t> offsets;
    std::vector<int> nbrs;
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster.
//
// Equivalent to iteratively picking the unclustered point (lowest index on ties)
// with the smallest maximum distance to all current members, as long as that
// distance is below the threshold. The maximum distance per candidate is
// maintained incrementally, and candidates whose maximum distance reaches the
// threshold are dropped (it can only grow).
int generateCandidateCluster(const int seed_point,
                              const std::vector<char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              const NeighborLists& nl,
                              CandidateWorkspace& ws,
                              std::vector<int>* cluster_members = nullptr) {
    int* __restrict idx = ws.idx.data();
    double* __restrict cx = ws.cx.data();
    double* __restrict cy = ws.cy.data();
    double* __restrict maxd = ws.maxd.data();

    const Point sp = points[seed_point];
    int m = 0;
    auto consider = [&](const int j) {
        if (j == seed_point || clustered[j]) return;
        const double d = distance(points[j], sp);
        if (d < threshold) {
            idx[m] = j;
            cx[m] = points[j].x;
            cy[m] = points[j].y;
            maxd[m] = d;
            ++m;
        }
    };
    if (nl.available) {
        for (size_t k = nl.offsets[seed_point]; k < nl.offsets[seed_point + 1]; ++k) {
            consider(nl.nbrs[k]);
        }
    } else {
        for (int j = 0; j < point_count; ++j) consider(j);
    }

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }
    int size = 1;

    while (m > 0 && size < point_count) {
        // Find the candidate with the smallest max distance (first on ties)
        int best = 0;
        double best_d = maxd[0];
        for (int k = 1; k < m; ++k) {
            if (maxd[k] < best_d) {
                best_d = maxd[k];
                best = k;
            }
        }

        const int chosen = idx[best];
        const Point np = {cx[best], cy[best]};
        if (cluster_members) cluster_members->push_back(chosen);
        ++size;

        // Remove chosen, update max distances and drop candidates that can no
        // longer be added, preserving index order
        int w = 0;
        for (int k = 0; k < m; ++k) {
            if (k == best) continue;
            const double dx = cx[k] - np.x;
            const double dy = cy[k] - np.y;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double nd = std::max(maxd[k], d);
            if (nd < threshold) {
                idx[w] = idx[k];
                cx[w] = cx[k];
                cy[w] = cy[k];
                maxd[w] = nd;
                ++w;
            }
        }
        m = w;
    }

    return size;
}

// Build neighbor lists if they fit into a reasonable amount of memory
NeighborLists buildNeighborLists(const std::vector<Point>& points, const double threshold) {
    NeighborLists nl;
    const int N = static_cast<int>(points.size());
    std::vector<size_t> counts(N + 1, 0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        size_t c = 0;
        const Point p = points[i];
        for (int j = 0; j < N; ++j) {
            c += (j != i && distance(points[j], p) < threshold) ? 1 : 0;
        }
        counts[i] = c;
    }

    size_t total = 0;
    nl.offsets.resize(N + 1);
    for (int i = 0; i < N; ++i) {
        nl.offsets[i] = total;
        total += counts[i];
    }
    nl.offsets[N] = total;

    static const size_t MAX_NEIGHBOR_ENTRIES = size_t(1) << 28; // 1 GiB of ints
    if (total > MAX_NEIGHBOR_ENTRIES) {
        nl.offsets.clear();
        return nl;
    }

    nl.nbrs.resize(total);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        size_t pos = nl.offsets[i];
        const Point p = points[i];
        for (int j = 0; j < N; ++j) {
            if (j != i && distance(points[j], p) < threshold) nl.nbrs[pos++] = j;
        }
    }
    nl.available = true;
    return nl;
}

// Main QT clustering algorithm
//
// A seed's candidate cluster only depends on the unclustered points within the
// threshold of the seed, so its cardinality is cached and only recomputed when
// one of those points has been clustered. Seeds are processed in parallel.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    const NeighborLists nl = buildNeighborLists(points, threshold);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    const int nthreads = omp_get_max_threads();
    std::vector<CandidateWorkspace> workspaces(nthreads);
    std::vector<int> card_cache(N, 0);
    std::vector<char> cache_valid(N, 0);
    std::vector<int> dirty, bounds, order;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Collect seeds whose cached cardinality is stale; best cached cardinality
        dirty.clear();
        int global_best = -1;
        for (int i = 0; i < U; ++i) {
            const int seed = unclustered_indices[i];
            if (cache_valid[seed]) {
                global_best = std::max(global_best, card_cache[seed]);
            } else {
                dirty.push_back(seed);
            }
        }
        const int D = static_cast<int>(dirty.size());

        // Upper bound on each stale seed's cardinality: the seed plus all
        // unclustered points within the threshold. Stale seeds are processed in
        // decreasing bound order so that seeds which cannot reach the best
        // cardinality found so far can be skipped (does not change the result).
        bounds.resize(D);
        order.resize(D);
        const int bound_work = nl.available ? D / 256 : D / 8;
        #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads) \
            if (bound_work > 1)
        for (int i = 0; i < D; ++i) {
            const int seed = dirty[i];
            const Point sp = points[seed];
            int c = 1;
            if (nl.available) {
                for (size_t k = nl.offsets[seed]; k < nl.offsets[seed + 1]; ++k) {
                    c += clustered[nl.nbrs[k]] ? 0 : 1;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    c += (j != seed && !clustered[j] && distance(points[j], sp) < threshold) ? 1 : 0;
                }
            }
            bounds[i] = c;
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return bounds[a] != bounds[b] ? bounds[a] > bounds[b] : a < b;
        });

        // Rough estimate of the work (greedy cluster growth is ~quadratic in the bound)
        long long card_work = 0;
        for (int i = 0; i < D; ++i) card_work += static_cast<long long>(bounds[i]) * bounds[i];

        // Compute cardinalities of stale seeds in parallel
        #pragma omp parallel num_threads(nthreads) if (D > 1 && card_work > 100000)
        {
            CandidateWorkspace& ws = workspaces[omp_get_thread_num()];
            if (static_cast<int>(ws.idx.size()) < N) ws.reserve(N);

            #pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < D; ++k) {
                const int i = order[k];
                int gb;
                #pragma omp atomic read relaxed
                gb = global_best;
                if (bounds[i] < gb) continue; // cannot be the best; stays stale

                const int seed = dirty[i];
                const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                                 threshold, N, nl, ws);
                card_cache[seed] = cardinality;
                cache_valid[seed] = 1;
                if (cardinality > gb) {
                    #pragma omp atomic compare relaxed
                    if (global_best < cardinality) { global_best = cardinality; }
                }
            }
        }

        // The winner is the seed with maximum cardinality, earliest on ties
        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < U; ++i) {
            const int seed = unclustered_indices[i];
            if (cache_valid[seed] && card_cache[seed] > max_cardinality) {
                max_cardinality = card_cache[seed];
                best_seed = seed;
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            generateCandidateCluster(best_seed, clustered, points, threshold, N, nl,
                                     workspaces[0], &cluster.members);

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = 1;
            }

            // Invalidate cached cardinalities of seeds near the new cluster
            if (nl.available) {
                for (const int mbr : cluster.members) {
                    for (size_t k = nl.offsets[mbr]; k < nl.offsets[mbr + 1]; ++k) {
                        cache_valid[nl.nbrs[k]] = 0;
                    }
                }
            } else {
                const std::vector<int>& mbrs = cluster.members;
                const int M = static_cast<int>(mbrs.size());
                const long long inval_work = static_cast<long long>(U) * M / 4096;
                #pragma omp parallel for schedule(static) num_threads(nthreads) \
                    if (inval_work > 1)
                for (int i = 0; i < U; ++i) {
                    const int j = unclustered_indices[i];
                    if (clustered[j] || !cache_valid[j]) continue;
                    const Point pj = points[j];
                    for (int k = 0; k < M; ++k) {
                        if (distance(points[mbrs[k]], pj) < threshold) {
                            cache_valid[j] = 0;
                            break;
                        }
                    }
                }
            }
            clusters.push_back(std::move(cluster));

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
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

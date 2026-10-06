// QT Clustering Benchmark - MPI Parallel Version
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

// Generate a candidate cluster starting from a seed point.
// Semantically identical to the original greedy procedure (repeatedly add the
// lowest-index unclustered point minimizing the max distance to the current
// members, as long as that max distance is < threshold), but implemented
// incrementally: each candidate keeps its running max distance to the cluster,
// and candidates that ever reach >= threshold are dropped permanently (the max
// can only grow). Only points within threshold of the seed can ever qualify.
// Returns the cardinality (size) of the cluster.
struct CandidateWork {
    std::vector<int> cand;
    std::vector<double> maxd;
};

int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             CandidateWork& work,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<int>& cand = work.cand;
    std::vector<double>& maxd = work.maxd;
    cand.clear();
    maxd.clear();

    const Point sp = points[seed_point];
    // Candidates in increasing index order (unclustered is sorted ascending)
    for (const int idx : unclustered) {
        if (idx == seed_point) continue;
        const double d = distance(points[idx], sp);
        if (d < threshold) {
            cand.push_back(idx);
            maxd.push_back(d);
        }
    }

    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }
    int size = 1;
    int n = static_cast<int>(cand.size());

    while (n > 0) {
        // Select the first candidate with minimal max distance
        int best = 0;
        double bestd = maxd[0];
        for (int k = 1; k < n; ++k) {
            if (maxd[k] < bestd) {
                bestd = maxd[k];
                best = k;
            }
        }
        const int chosen = cand[best];
        ++size;
        if (cluster_members) cluster_members->push_back(chosen);

        // Update running max distances, drop chosen and disqualified candidates
        const Point cp = points[chosen];
        int w = 0;
        for (int k = 0; k < n; ++k) {
            if (k == best) continue;
            const double d = std::max(maxd[k], distance(points[cand[k]], cp));
            if (d < threshold) {
                cand[w] = cand[k];
                maxd[w] = d;
                ++w;
            }
        }
        n = w;
    }

    return size;
}

// Main QT clustering algorithm (MPI-parallel).
// Every rank holds the full point set and identical clustering state. Candidate
// cluster cardinalities are cached per seed; a seed's candidate cluster can only
// change if a newly clustered point lies within threshold of the seed, so only
// such seeds are recomputed. Recomputation is distributed across ranks and the
// results are exchanged with MPI_Allgatherv. Ties are broken toward the lowest
// seed index, exactly as in the sequential version.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<char> dirty(N, 1);
    std::vector<int> card(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    CandidateWork work;
    std::vector<int> todo, my_results, all_results, counts(nprocs), displs(nprocs);
    // Slightly enlarged radius for conservative invalidation
    const double inval_r2 = threshold * threshold * (1.0 + 1e-9) + 1e-12;

    while (!unclustered_indices.empty()) {
        // Collect seeds needing recomputation (identical on all ranks)
        todo.clear();
        for (const int s : unclustered_indices)
            if (dirty[s]) todo.push_back(s);
        const int T = static_cast<int>(todo.size());

        if (T > 0) {
            // Block-cyclic distribution over the todo list for load balance
            my_results.clear();
            for (int k = rank; k < T; k += nprocs) {
                my_results.push_back(generateCandidateCluster(
                    todo[k], unclustered_indices, points, threshold, work));
            }
            for (int r = 0; r < nprocs; ++r) {
                counts[r] = (T > r) ? (T - r + nprocs - 1) / nprocs : 0;
                displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
            }
            all_results.resize(T);
            MPI_Allgatherv(my_results.data(), static_cast<int>(my_results.size()), MPI_INT,
                           all_results.data(), counts.data(), displs.data(), MPI_INT,
                           MPI_COMM_WORLD);
            for (int r = 0; r < nprocs; ++r) {
                int pos = displs[r];
                for (int k = r; k < T; k += nprocs) {
                    card[todo[k]] = all_results[pos++];
                    dirty[todo[k]] = 0;
                }
            }
        }

        // Select the best seed: max cardinality, first (lowest index) on ties
        int max_cardinality = -1;
        int best_seed = -1;
        for (const int s : unclustered_indices) {
            if (card[s] > max_cardinality) {
                max_cardinality = card[s];
                best_seed = s;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            generateCandidateCluster(best_seed, unclustered_indices, points, threshold,
                                     work, &cluster.members);

            double minx = std::numeric_limits<double>::max(), miny = minx;
            double maxx = -minx, maxy = -minx;
            for (const int m : cluster.members) {
                clustered[m] = 1;
                minx = std::min(minx, points[m].x);
                maxx = std::max(maxx, points[m].x);
                miny = std::min(miny, points[m].y);
                maxy = std::max(maxy, points[m].y);
            }

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());

            // Invalidate cached results of seeds near the newly clustered points
            const double pad = threshold * (1.0 + 1e-9) + 1e-12;
            for (const int s : unclustered_indices) {
                if (dirty[s]) continue;
                const Point p = points[s];
                if (p.x < minx - pad || p.x > maxx + pad ||
                    p.y < miny - pad || p.y > maxy + pad) continue;
                for (const int m : cluster.members) {
                    const double dx = p.x - points[m].x;
                    const double dy = p.y - points[m].y;
                    if (dx * dx + dy * dy <= inval_r2) {
                        dirty[s] = 1;
                        break;
                    }
                }
            }

            clusters.push_back(std::move(cluster));
        } else {
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

static int runMain(int argc, char** argv, const int rank) {
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
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
    if (printResults && rank == 0) {
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
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 produces output; all ranks run the same computation
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) {
            fclose(stdout);
        }
    }

    const int ret = runMain(argc, argv, rank);
    fflush(stdout);
    MPI_Finalize();
    return ret;
}

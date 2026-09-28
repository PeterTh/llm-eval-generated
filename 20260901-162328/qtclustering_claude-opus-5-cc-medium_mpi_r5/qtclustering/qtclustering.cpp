// QT Clustering Benchmark - MPI Distributed-Memory Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (MPI):
//   The outer loop of the algorithm is inherently sequential (each round
//   commits exactly one cluster, which changes the input of the next round).
//   The expensive part of every round is the independent evaluation of one
//   candidate cluster per remaining unclustered seed point.  Those candidate
//   clusters are distributed over the MPI ranks (cyclic seed ownership, which
//   spreads the spatially correlated -- and therefore cost-correlated -- point
//   groups evenly).  Each rank reduces its seeds to a local best candidate,
//   an MPI_MAXLOC allreduce selects the global winner with exactly the same
//   tie-breaking rule as the sequential code (largest cardinality, smallest
//   seed index), and the owning rank broadcasts the winning member list so
//   that every rank keeps a consistent global clustering state.
//
// Serial optimizations that preserve bit-identical results:
//   * Incremental cluster growth: the maximum distance of every candidate to
//     the current cluster is carried along and updated with a single distance
//     evaluation per step instead of being recomputed from scratch.
//   * Candidate pruning: a point can only ever join if it is within the
//     threshold of the seed, and once its running maximum distance reaches the
//     threshold it can never drop again, so it is removed from the work list.
//   * Candidate-cluster memoization: a cached candidate cluster stays exactly
//     valid as long as none of its members has been claimed by a committed
//     cluster (the greedy choices only depend on the points that are actually
//     picked), so only invalidated seeds are recomputed in later rounds.

#include <algorithm>
#include <chrono>
#include <climits>
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

// Per-rank scratch buffers for the candidate cluster construction.
struct GrowScratch {
    std::vector<double> cx, cy, cmax;
    std::vector<int> cidx;

    void resize(int n) {
        cx.resize(n);
        cy.resize(n);
        cmax.resize(n);
        cidx.resize(n);
    }
};

// Generate a candidate cluster starting from a seed point.
//
// This reproduces the sequential reference exactly: in every step the still
// unclustered point with the smallest maximum distance to all current members
// is added, provided that distance stays strictly below the threshold; ties
// are won by the smallest point index.  The maximum distances are maintained
// incrementally (max() is order independent, so the values are bit-identical
// to recomputing them over the whole member list).
static int generateCandidateCluster(const int seed_point,
                                    const double* __restrict px,
                                    const double* __restrict py,
                                    const unsigned char* __restrict clustered,
                                    const double threshold,
                                    const int point_count,
                                    GrowScratch& s,
                                    std::vector<int>& members) {
    const double INF = std::numeric_limits<double>::max();

    members.clear();
    members.push_back(seed_point);

    double* __restrict cx = s.cx.data();
    double* __restrict cy = s.cy.data();
    double* __restrict cmax = s.cmax.data();
    int* __restrict cidx = s.cidx.data();

    // Initial candidate set: unclustered points within the threshold of the
    // seed (the seed is a member, so anything farther can never be added).
    const double sx = px[seed_point];
    const double sy = py[seed_point];
    int n = 0;
    for (int i = 0; i < point_count; ++i) {
        if (clustered[i] || i == seed_point) continue;
        const double dx = px[i] - sx;
        const double dy = py[i] - sy;
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            cx[n] = px[i];
            cy[n] = py[i];
            cmax[n] = d;
            cidx[n] = i;
            ++n;
        }
    }

    // Pick the first point to add (smallest max distance, smallest index).
    int best = -1;
    double best_d = INF;
    for (int k = 0; k < n; ++k) {
        if (cmax[k] < best_d) {
            best_d = cmax[k];
            best = k;
        }
    }

    while (best >= 0) {
        const double mx = cx[best];
        const double my = cy[best];
        members.push_back(cidx[best]);
        // Force the freshly added member out of the candidate list below.
        cmax[best] = INF;

        // Update the running maximum distances (vectorizable).
        for (int k = 0; k < n; ++k) {
            const double dx = cx[k] - mx;
            const double dy = cy[k] - my;
            const double d = std::sqrt(dx * dx + dy * dy);
            cmax[k] = cmax[k] > d ? cmax[k] : d;
        }

        // Compact out the candidates that can never join anymore and pick the
        // next best one in a single pass.  Compaction preserves the ascending
        // index order, so the tie-breaking stays identical to the reference.
        int n2 = 0;
        best = -1;
        best_d = INF;
        for (int k = 0; k < n; ++k) {
            const double md = cmax[k];
            if (md >= threshold) continue;
            cx[n2] = cx[k];
            cy[n2] = cy[k];
            cmax[n2] = md;
            cidx[n2] = cidx[k];
            if (md < best_d) {
                best_d = md;
                best = n2;
            }
            ++n2;
        }
        n = n2;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel over candidate seeds)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());

    // Structure-of-arrays copy of the coordinates for cache/SIMD friendliness.
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Cyclic seed ownership: rank r owns seeds r, r+nprocs, r+2*nprocs, ...
    const int owned = (N - rank + nprocs - 1) / nprocs;
    std::vector<std::vector<int>> cache(owned > 0 ? owned : 0);
    std::vector<unsigned char> cache_valid(owned > 0 ? owned : 0, 0);

    GrowScratch scratch;
    scratch.resize(N);

    std::vector<int> recv_members;
    recv_members.reserve(N);

    int remaining = N;
    while (remaining > 0) {
        struct { int card; int seed; } local, global;
        local.card = -1;
        local.seed = INT_MAX;

        for (int slot = 0; slot < owned; ++slot) {
            const int seed = rank + slot * nprocs;
            if (clustered[seed]) continue;

            // A cached candidate cluster is still exactly valid as long as
            // none of its members has been claimed by a committed cluster.
            if (cache_valid[slot]) {
                for (const int m : cache[slot]) {
                    if (clustered[m]) {
                        cache_valid[slot] = 0;
                        break;
                    }
                }
            }
            if (!cache_valid[slot]) {
                generateCandidateCluster(seed, px.data(), py.data(),
                                         clustered.data(), threshold, N,
                                         scratch, cache[slot]);
                cache_valid[slot] = 1;
            }

            const int card = static_cast<int>(cache[slot].size());
            if (card > local.card) {
                local.card = card;
                local.seed = seed;
            }
        }

        // Largest cardinality wins; MPI_MAXLOC breaks ties by the smallest
        // seed index, exactly like the sequential scan in ascending order.
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global.card <= 0) break;

        const int owner = global.seed % nprocs;
        recv_members.resize(global.card);
        if (rank == owner) {
            const int slot = (global.seed - rank) / nprocs;
            std::copy(cache[slot].begin(), cache[slot].end(), recv_members.begin());
        }
        MPI_Bcast(recv_members.data(), global.card, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global.seed;
        cluster.members = recv_members;
        for (const int m : recv_members) {
            clustered[m] = 1;
        }
        remaining -= global.card;
        clusters.push_back(std::move(cluster));
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic; replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int exit_code = 0;

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", max_cluster_time);
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
        const double time_sec = max_cluster_time / 1000.0;
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
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering parallelized with MPI.
// Seed evaluation is distributed across ranks; the best cluster
// is selected via Allreduce and broadcast each iteration.

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

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Precomputed pairwise distance matrix (flat, row-major)
static std::vector<double> dist_matrix;
static int dist_N;

inline double getDist(int i, int j) {
    return dist_matrix[(size_t)i * dist_N + j];
}

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

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

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void precomputeDistances(const std::vector<Point>& points) {
    dist_N = static_cast<int>(points.size());
    dist_matrix.resize((size_t)dist_N * dist_N);
    for (int i = 0; i < dist_N; ++i) {
        dist_matrix[(size_t)i * dist_N + i] = 0.0;
        for (int j = i + 1; j < dist_N; ++j) {
            double d = distance(points[i], points[j]);
            dist_matrix[(size_t)i * dist_N + j] = d;
            dist_matrix[(size_t)j * dist_N + i] = d;
        }
    }
}

// Find closest unclustered point that keeps cluster diameter < threshold
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        bool exceeded = false;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const double dist = getDist(candidate, cluster_members[i]);
            if (dist >= threshold || dist >= min_diameter) { exceeded = true; break; }
            if (dist > max_dist) max_dist = dist;
        }

        if (!exceeded && max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// Generate a candidate cluster from a seed; returns cluster size
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const double threshold,
                              const int point_count,
                              std::vector<int>& members,
                              std::vector<bool>& in_cluster) {
    // Reset reusable buffers
    std::fill(in_cluster.begin(), in_cluster.end(), false);
    members.clear();

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster,
                                             threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm — parallelized with MPI
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Reusable per-candidate buffers
    std::vector<int> members_buf;
    std::vector<bool> in_cluster_buf(N, false);

    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = N; // sentinel larger than any valid index
        std::vector<int> local_best_members;

        const int total_seeds = static_cast<int>(unclustered_indices.size());

        // Round-robin distribution of seed evaluation across ranks
        for (int i = rank; i < total_seeds; i += nprocs) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(seed, clustered,
                                        threshold, N, members_buf, in_cluster_buf);

            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = members_buf;
            }
        }

        // Global reduction: max cardinality, ties broken by min seed index.
        // Encode as cardinality*(N+1) - seed so higher value = better.
        struct { double val; int idx; } local_pair, global_pair;

        if (local_max_cardinality > 0) {
            local_pair.val = (double)local_max_cardinality * (N + 1) - local_best_seed;
        } else {
            local_pair.val = -1.0;
        }
        local_pair.idx = rank;

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (global_pair.val < 0) break;

        int winning_rank = global_pair.idx;

        // Broadcast winning cluster from its owner
        int cluster_size = (rank == winning_rank) ? static_cast<int>(local_best_members.size()) : 0;
        MPI_Bcast(&cluster_size, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);

        if (cluster_size <= 0) break;

        std::vector<int> best_members(cluster_size);
        int best_seed;
        if (rank == winning_rank) {
            best_members = local_best_members;
            best_seed = local_best_seed;
        }
        MPI_Bcast(best_members.data(), cluster_size, MPI_INT, winning_rank, MPI_COMM_WORLD);
        MPI_Bcast(&best_seed, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);

        // All ranks update shared state identically
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) {
            clustered[m] = true;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

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
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks generate identical data (deterministic seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Precompute pairwise distance matrix for fast lookups
    precomputeDistances(points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_cluster_time_ms =
        std::chrono::duration<double, std::milli>(cluster_end - cluster_start).count();
    double cluster_time_ms = 0.0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %.3f ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

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

        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
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

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

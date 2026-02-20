// QT Clustering Benchmark - MPI Distributed Memory Version
//
// Parallelization strategy:
// - Outer clustering loop is inherently sequential.
// - Within each iteration, evaluation of candidate clusters for each seed is
//   distributed across MPI ranks; the best seed is selected deterministically.
// - The winning cluster members are broadcast so all ranks update state identically.

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

// Calculate Euclidean distance between two points (used for validation/printing)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline double distance2(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

inline double getDist2(const std::vector<double>* dist2_mat,
                       const std::vector<Point>& points,
                       const int i,
                       const int j) {
    if (dist2_mat && !dist2_mat->empty()) {
        const size_t N = points.size();
        return (*dist2_mat)[static_cast<size_t>(i) * N + static_cast<size_t>(j)];
    }
    return distance2(points[i], points[j]);
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<unsigned char>& clustered,
                     const std::vector<unsigned char>& in_cluster,
                     const std::vector<Point>& points,
                     const std::vector<double>* dist2_mat,
                     const double threshold2,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter2 = std::numeric_limits<double>::max();

    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;

        // Calculate the maximum distance^2 from candidate to all cluster members
        double max_dist2 = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double d2 = getDist2(dist2_mat, points, candidate, member);
            if (d2 > max_dist2) max_dist2 = d2;
            // Early exits: cannot improve current best or violates threshold
            if (max_dist2 >= threshold2 || max_dist2 >= min_diameter2) {
                break;
            }
        }

        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist2 < threshold2 && max_dist2 < min_diameter2) {
            min_diameter2 = max_dist2;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points,
                             const std::vector<double>* dist2_mat,
                             const double threshold2,
                             const int point_count,
                             std::vector<int>* cluster_members = nullptr) {
    std::vector<unsigned char> in_cluster(static_cast<size_t>(point_count), 0);
    std::vector<int> members;
    members.reserve(static_cast<size_t>(point_count));

    // Add seed point
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             dist2_mat, threshold2, point_count);

        if (closest < 0) break; // No more points can be added

        in_cluster[closest] = 1;
        members.push_back(closest);
    }

    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

static int ownerOfSeed(const std::vector<int>& unclustered_indices,
                       const int seed,
                       const int comm_size) {
    for (size_t pos = 0; pos < unclustered_indices.size(); ++pos) {
        if (unclustered_indices[pos] == seed) {
            return static_cast<int>(pos % static_cast<size_t>(comm_size));
        }
    }
    // Should never happen if seed came from unclustered_indices
    return 0;
}

static void maybeBuildDistanceMatrixMPI(const std::vector<Point>& points,
                                       MPI_Comm comm,
                                       std::vector<double>* dist2_mat) {
    const int N = static_cast<int>(points.size());
    if (N <= 0) {
        dist2_mat->clear();
        return;
    }

    // Cap memory to avoid pathological allocations.
    constexpr size_t kCapBytes = 256ull * 1024ull * 1024ull;
    const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(double);
    if (bytes > kCapBytes) {
        dist2_mat->clear();
        return;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int rows_per_rank = (N + size - 1) / size;
    const int row_start = std::min(N, rank * rows_per_rank);
    const int row_end = std::min(N, row_start + rows_per_rank);
    const int local_rows = std::max(0, row_end - row_start);

    std::vector<double> local(static_cast<size_t>(local_rows) * static_cast<size_t>(N));
    for (int i = row_start; i < row_end; ++i) {
        const Point pi = points[i];
        double* row = local.data() + static_cast<size_t>(i - row_start) * static_cast<size_t>(N);
        for (int j = 0; j < N; ++j) {
            row[j] = distance2(pi, points[j]);
        }
    }

    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const int rs = std::min(N, r * rows_per_rank);
        const int re = std::min(N, rs + rows_per_rank);
        const int rr = std::max(0, re - rs);
        const size_t cnt = static_cast<size_t>(rr) * static_cast<size_t>(N);
        recvcounts[r] = static_cast<int>(cnt);
    }
    for (int r = 1; r < size; ++r) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    dist2_mat->assign(static_cast<size_t>(N) * static_cast<size_t>(N), 0.0);
    MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                   dist2_mat->data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);
}

// MPI-distributed QT clustering algorithm (clusters returned on rank 0 only)
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int N = static_cast<int>(points.size());
    const double threshold2 = threshold * threshold;

    std::vector<double> dist2_mat;
    maybeBuildDistanceMatrixMPI(points, comm, &dist2_mat);
    const std::vector<double>* dist2_ptr = dist2_mat.empty() ? nullptr : &dist2_mat;

    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<Cluster> clusters;
    if (rank == 0) clusters.reserve(static_cast<size_t>(N));

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_best_card = -1;
        int local_best_seed = -1;

        // Try each unclustered point as a seed (distributed over ranks)
        for (size_t pos = static_cast<size_t>(rank); pos < unclustered_indices.size(); pos += static_cast<size_t>(size)) {
            const int seed = unclustered_indices[pos];
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                            dist2_ptr, threshold2, N,
                                                            nullptr);

            if (cardinality > local_best_card ||
                (cardinality == local_best_card && local_best_seed >= 0 && seed < local_best_seed) ||
                (cardinality == local_best_card && local_best_seed < 0)) {
                local_best_card = cardinality;
                local_best_seed = seed;
            }
        }

        int best_seed = -1;
        int best_card = -1;

        // Gather local bests to rank 0 and select global best deterministically
        std::vector<int> all_cards;
        std::vector<int> all_seeds;
        if (rank == 0) {
            all_cards.resize(static_cast<size_t>(size));
            all_seeds.resize(static_cast<size_t>(size));
        }

        MPI_Gather(&local_best_card, 1, MPI_INT,
                   rank == 0 ? all_cards.data() : nullptr, 1, MPI_INT, 0, comm);
        MPI_Gather(&local_best_seed, 1, MPI_INT,
                   rank == 0 ? all_seeds.data() : nullptr, 1, MPI_INT, 0, comm);

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                const int seed = all_seeds[static_cast<size_t>(r)];
                const int card = all_cards[static_cast<size_t>(r)];
                if (seed < 0) continue;
                if (card > best_card || (card == best_card && (best_seed < 0 || seed < best_seed))) {
                    best_card = card;
                    best_seed = seed;
                }
            }
        }

        MPI_Bcast(&best_seed, 1, MPI_INT, 0, comm);
        MPI_Bcast(&best_card, 1, MPI_INT, 0, comm);

        if (best_seed < 0 || best_card <= 0) {
            break;
        }

        // The rank that owns this seed builds the winning cluster members and broadcasts them
        const int owner = ownerOfSeed(unclustered_indices, best_seed, size);

        std::vector<int> best_members;
        int member_count = 0;
        if (rank == owner) {
            member_count = generateCandidateCluster(best_seed, clustered, points,
                                                    dist2_ptr, threshold2, N,
                                                    &best_members);
        }

        MPI_Bcast(&member_count, 1, MPI_INT, owner, comm);
        if (rank != owner) {
            best_members.resize(static_cast<size_t>(member_count));
        }
        MPI_Bcast(best_members.data(), member_count, MPI_INT, owner, comm);

        // Update clustered state on all ranks
        for (int i = 0; i < member_count; ++i) {
            clustered[best_members[static_cast<size_t>(i)]] = 1;
        }

        // Remove clustered points from unclustered list (preserve order)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[static_cast<size_t>(idx)] != 0; }),
            unclustered_indices.end());

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = std::move(best_members);
            clusters.push_back(std::move(cluster));
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;

    int run = 1;
    int exit_code = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                num_points = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                threshold = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                run = 0;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                run = 0;
                exit_code = 1;
                break;
            }
        }

        if (run && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            run = 0;
            exit_code = 1;
        }
    }

    MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!run) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        if (size > 1) {
            printf("MPI ranks: %d\n", size);
        }
    }

    // Generate synthetic data on rank 0 then broadcast to all ranks
    std::vector<Point> points(static_cast<size_t>(num_points));
    std::vector<double> coords(static_cast<size_t>(num_points) * 2);

    if (rank == 0) {
        generateSyntheticData(points, num_points);
        for (int i = 0; i < num_points; ++i) {
            coords[static_cast<size_t>(2 * i + 0)] = points[static_cast<size_t>(i)].x;
            coords[static_cast<size_t>(2 * i + 1)] = points[static_cast<size_t>(i)].y;
        }
    }

    MPI_Bcast(coords.data(), static_cast<int>(coords.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int i = 0; i < num_points; ++i) {
            points[static_cast<size_t>(i)] = {coords[static_cast<size_t>(2 * i + 0)],
                                              coords[static_cast<size_t>(2 * i + 1)]};
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
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
            std::vector<double> membershipData;
            membershipData.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[static_cast<size_t>(clusters[c].members[i])] = static_cast<int>(c);
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
            exit_code = valid ? 0 : 1;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}

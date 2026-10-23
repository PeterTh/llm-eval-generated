// QT Clustering Benchmark - MPI Distributed Memory Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * The synthetic data set and the "clustered"/"unclustered" state are
//     replicated on every rank (they are small compared to the candidate
//     generation work and are deterministically identical).
//   * In every outer iteration all unclustered points are candidate seeds.
//     The seeds are distributed cyclically over the ranks; each rank builds
//     the candidate clusters for its own seeds concurrently.
//   * A reduction selects the global winner with exactly the sequential
//     tie-break rule (largest cardinality, then smallest seed index). The
//     winning cluster is broadcast so that every rank advances its state
//     identically. No global synchronization is needed inside a candidate
//     cluster build.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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

// Per-rank scratch storage reused across all candidate cluster builds.
struct Workspace {
    std::vector<double> max_sq;    // max squared distance to current members
    std::vector<int> mark;         // generation stamp for "in current cluster"
    int stamp = 0;

    void init(int n) {
        max_sq.resize(n);
        mark.assign(n, 0);
        stamp = 0;
    }
};

// Reducer payload: (cardinality, seed). Better = larger cardinality, and on
// a tie the smaller seed index, matching the sequential loop order exactly.
struct Best {
    int card;
    int seed;
};

static void best_reduce(void* invec, void* inoutvec, int* len, MPI_Datatype*) {
    Best* in = static_cast<Best*>(invec);
    Best* out = static_cast<Best*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        if (in[i].card > out[i].card ||
            (in[i].card == out[i].card && in[i].seed < out[i].seed)) {
            out[i] = in[i];
        }
    }
}

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
// Returns the cardinality (size) of the cluster.
//
// Only unclustered points are considered, and the maximum distance to the
// current cluster is maintained incrementally. The candidate scan order is
// ascending point index, exactly as in the sequential implementation, so the
// selected points and tie-breaks are bit-for-bit identical.
int generateCandidateCluster(const int seed_point,
                             const std::vector<Point>& points,
                             const std::vector<int>& unclustered,
                             const double threshold,
                             Workspace& ws,
                             std::vector<int>& members) {
    const int U = static_cast<int>(unclustered.size());

    // Advance the generation stamp instead of clearing the mark array.
    if (++ws.stamp <= 0) {
        std::fill(ws.mark.begin(), ws.mark.end(), 0);
        ws.stamp = 1;
    }
    const int stamp = ws.stamp;

    double* max_sq = ws.max_sq.data();
    int* mark = ws.mark.data();
    const Point* pts = points.data();
    const int* unc = unclustered.data();

    // Distance from the seed to every unclustered point.
    const Point sp = pts[seed_point];
    for (int idx = 0; idx < U; ++idx) {
        const int p = unc[idx];
        const double dx = pts[p].x - sp.x;
        const double dy = pts[p].y - sp.y;
        max_sq[p] = dx * dx + dy * dy;
    }

    mark[seed_point] = stamp;
    members.clear();
    members.push_back(seed_point);

    // Iteratively add the closest point that keeps the diameter below threshold.
    while (static_cast<int>(members.size()) < U) {
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int idx = 0; idx < U; ++idx) {
            const int candidate = unc[idx];
            if (mark[candidate] == stamp) continue;

            // sqrt is monotonic, so the maximum over the rounded member
            // distances equals the rounded sqrt of the maximum squared
            // distance. One sqrt per candidate is enough and preserves the
            // exact floating-point comparison semantics.
            const double max_dist = std::sqrt(max_sq[candidate]);
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) break; // No more points can be added

        mark[closest_point] = stamp;
        members.push_back(closest_point);

        // Incrementally update the maximum distance to the enlarged cluster.
        const Point cp = pts[closest_point];
        for (int idx = 0; idx < U; ++idx) {
            const int p = unc[idx];
            if (mark[p] == stamp) continue;
            const double dx = pts[p].x - cp.x;
            const double dy = pts[p].y - cp.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 > max_sq[p]) max_sq[p] = d2;
        }
    }

    const int cardinality = static_cast<int>(members.size());
    return cardinality;
}

// Main QT clustering algorithm (MPI parallel).
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs,
                                  const MPI_Datatype best_type,
                                  const MPI_Op best_op) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    Workspace ws;
    ws.init(N);

    std::vector<int> local_members;
    std::vector<int> candidate_members;
    std::vector<int> global_members;

    // Main clustering loop
    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());

        Best local{-1, -1};
        local_members.clear();

        // Each rank builds candidate clusters for its cyclically assigned seeds.
        for (int pos = rank; pos < U; pos += nprocs) {
            const int seed = unclustered[pos];

            const int cardinality = generateCandidateCluster(seed, points, unclustered,
                                                             threshold, ws,
                                                             candidate_members);

            if (cardinality > local.card ||
                (cardinality == local.card && seed < local.seed)) {
                local.card = cardinality;
                local.seed = seed;
                // Swap so both buffers keep their capacity for reuse.
                local_members.swap(candidate_members);
            }
        }

        // Global winner: largest cardinality, then smallest seed index.
        Best best;
        MPI_Allreduce(&local, &best, 1, best_type, best_op, MPI_COMM_WORLD);

        // The rank that owns the winning seed already holds its member list.
        const int best_pos = static_cast<int>(
            std::lower_bound(unclustered.begin(), unclustered.end(), best.seed) -
            unclustered.begin());
        const int root = best_pos % nprocs;

        int member_count = (rank == root) ? static_cast<int>(local_members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, root, MPI_COMM_WORLD);

        global_members.resize(member_count);
        if (rank == root && member_count > 0) {
            std::copy(local_members.begin(), local_members.end(), global_members.begin());
        }
        if (member_count > 0) {
            MPI_Bcast(global_members.data(), member_count, MPI_INT, root, MPI_COMM_WORLD);
        }

        // If we found a cluster, add it
        if (best.seed >= 0 && best.card > 0) {
            Cluster cluster;
            cluster.seed_point = best.seed;
            cluster.members = global_members;
            clusters.push_back(std::move(cluster));

            // Mark all members as clustered
            for (int member : global_members) {
                clustered[member] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered.erase(
                std::remove_if(unclustered.begin(), unclustered.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered.end()
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Rank-local reduction type/op (identical on every rank).
    MPI_Datatype best_type;
    MPI_Type_contiguous(2, MPI_INT, &best_type);
    MPI_Type_commit(&best_type);
    MPI_Op best_op;
    MPI_Op_create(best_reduce, 1, &best_op);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;

    // Parse command line arguments (identical on all ranks).
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
            MPI_Op_free(&best_op);
            MPI_Type_free(&best_type);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Op_free(&best_op);
            MPI_Type_free(&best_type);
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Op_free(&best_op);
        MPI_Type_free(&best_type);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data once on rank 0 and replicate it so that every
    // rank works on bit-for-bit identical input.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), static_cast<int>(num_points * sizeof(Point)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nprocs,
                                                       best_type, best_op);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long local_cluster_time = cluster_time.count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(max_cluster_time);

    if (rank == 0) {
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
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
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
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Op_free(&best_op);
    MPI_Type_free(&best_type);
    MPI_Finalize();
    return exit_code;
}

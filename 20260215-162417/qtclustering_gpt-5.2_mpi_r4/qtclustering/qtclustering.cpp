// QT Clustering Benchmark - MPI Parallel Version
//
// Parallelization strategy:
// - All ranks hold the full point set.
// - In each QT iteration, candidate seed evaluations are distributed across ranks.
// - A deterministic all-reduce selects the globally best seed (max cardinality, tie -> smallest seed).
// - The winning rank broadcasts the chosen cluster membership; all ranks update state identically.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct CandidateScratch {
    std::vector<int> mark;   // mark[i] == epoch means i is in current candidate cluster
    int epoch = 1;
    std::vector<int> members;
};

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<unsigned char>& clustered,
                     const std::vector<int>& mark,
                     const int epoch,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || (mark[candidate] == epoch)) continue;

        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
            if (max_dist >= min_diameter) break; // can't beat current best
        }

        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
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
                             const double threshold,
                             const int point_count,
                             CandidateScratch& scratch,
                             std::vector<int>* cluster_members = nullptr) {
    // epoch wraparound is extremely unlikely, but handle deterministically.
    scratch.epoch++;
    if (scratch.epoch == 0) {
        std::fill(scratch.mark.begin(), scratch.mark.end(), 0);
        scratch.epoch = 1;
    }

    scratch.members.clear();

    // Add seed point
    scratch.mark[seed_point] = scratch.epoch;
    scratch.members.push_back(seed_point);

    // Iteratively add closest points
    while (static_cast<int>(scratch.members.size()) < point_count) {
        const int closest = findClosestPoint(scratch.members, clustered, scratch.mark,
                                             scratch.epoch, points, threshold, point_count);
        if (closest < 0) break; // No more points can be added

        scratch.mark[closest] = scratch.epoch;
        scratch.members.push_back(closest);
    }

    // Copy members if requested
    if (cluster_members) {
        *cluster_members = scratch.members;
    }

    return static_cast<int>(scratch.members.size());
}

struct BestSeed {
    int cardinality;
    int seed;
    int rank;
};

static inline bool betterBestSeed(const BestSeed& a, const BestSeed& b) {
    // Higher cardinality wins. For ties, smaller seed wins (matches sequential scan).
    // Seed < 0 means "no candidate".
    if (a.seed < 0) return false;
    if (b.seed < 0) return true;
    if (a.cardinality != b.cardinality) return a.cardinality > b.cardinality;
    if (a.seed != b.seed) return a.seed < b.seed;
    return a.rank < b.rank;
}

static void bestSeedReduce(void* invec, void* inoutvec, int* len, MPI_Datatype*) {
    auto* in = static_cast<BestSeed*>(invec);
    auto* inout = static_cast<BestSeed*>(inoutvec);
    for (int i = 0; i < *len; ++i) {
        if (betterBestSeed(in[i], inout[i])) {
            inout[i] = in[i];
        }
    }
}

// Main QT clustering algorithm (MPI-parallel). Returns clusters only on rank 0.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                    const double threshold,
                                    MPI_Comm comm) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    CandidateScratch scratch;
    scratch.mark.assign(static_cast<size_t>(N), 0);

    std::vector<Cluster> clusters;

    MPI_Datatype bestType;
    {
        int blocklen[3] = {1, 1, 1};
        MPI_Aint disps[3];
        disps[0] = static_cast<MPI_Aint>(offsetof(BestSeed, cardinality));
        disps[1] = static_cast<MPI_Aint>(offsetof(BestSeed, seed));
        disps[2] = static_cast<MPI_Aint>(offsetof(BestSeed, rank));
        MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
        MPI_Type_create_struct(3, blocklen, disps, types, &bestType);
        MPI_Type_commit(&bestType);
    }

    MPI_Op bestOp;
    MPI_Op_create(&bestSeedReduce, /*commute=*/1, &bestOp);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        BestSeed local{ -1, -1, rank };
        std::vector<int> local_best_members;
        std::vector<int> candidate_members;

        // Distribute seeds across ranks by index into unclustered_indices.
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size(); i += static_cast<size_t>(nprocs)) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                            threshold, N, scratch,
                                                            &candidate_members);

            BestSeed cand{ cardinality, seed, rank };
            if (betterBestSeed(cand, local)) {
                local = cand;
                local_best_members.swap(candidate_members);
            }
        }

        BestSeed global{ -1, -1, rank };
        MPI_Allreduce(&local, &global, 1, bestType, bestOp, comm);

        if (global.seed < 0 || global.cardinality <= 0) {
            break;
        }

        // Broadcast winning membership from the winning rank.
        int member_count = 0;
        if (rank == global.rank) {
            member_count = static_cast<int>(local_best_members.size());
        }
        MPI_Bcast(&member_count, 1, MPI_INT, global.rank, comm);

        std::vector<int> best_members(static_cast<size_t>(member_count));
        if (rank == global.rank) {
            best_members = local_best_members;
        }
        if (member_count > 0) {
            MPI_Bcast(best_members.data(), member_count, MPI_INT, global.rank, comm);
        }

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = global.seed;
            cluster.members = best_members;
            clusters.push_back(std::move(cluster));
        }

        // Mark all members as clustered (all ranks)
        for (int idx : best_members) {
            clustered[idx] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    MPI_Op_free(&bestOp);
    MPI_Type_free(&bestType);

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_points = 1000;
    double threshold = 2.0;
    int validate = 0;
    int printResults = 0;
    int arg_error = 0;

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
                arg_error = 2; // special: help
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                arg_error = 1;
                break;
            }
        }

        if (arg_error == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            arg_error = 1;
        }
    }

    // Broadcast parsed options / error state
    MPI_Bcast(&arg_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (arg_error != 0) {
        MPI_Finalize();
        return (arg_error == 2) ? 0 : 1;
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
    }

    // Generate/broadcast synthetic data (rank 0 is source of truth for determinism)
    std::vector<Point> points(static_cast<size_t>(num_points));
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), static_cast<int>(points.size() * sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const double local_time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();

    double max_time_sec = 0.0;
    MPI_Reduce(&local_time_sec, &max_time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;

    if (rank == 0) {
        const long cluster_ms = static_cast<long>(max_time_sec * 1000.0);
        printf("Clustering time: %ld ms\n", cluster_ms);
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
        const double time_sec = (max_time_sec > 0.0) ? max_time_sec : 1e-12;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        // Print results for external validation
        if (printResults) {
            // Serialize cluster membership for hashing
            std::vector<double> membershipData;
            membershipData.reserve(static_cast<size_t>(num_points));
            std::vector<int> membership(static_cast<size_t>(num_points), -1);
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
                rc = 0;
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}

// QT clustering benchmark -- distributed-memory MPI implementation.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

static void generateSyntheticData(std::vector<Point>& points, int n,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < n) {
        const double cx = frand() * MAX_WIDTH;
        const double cy = frand() * MAX_HEIGHT;
        const double radius = frand() * min_dim / 2.0;
        int group_count = static_cast<int>(frand() * (n / 30.0));
        // For n < 30 the original expression always truncates to zero.
        // Ensure small, otherwise valid benchmark inputs make progress.
        if (n < 30) group_count = 1;
        group_count = std::min(group_count, n - count);
        while (group_count > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double r = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cx + dx, y = cy + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT)
                continue;
            points[count++] = {x, y};
            --group_count;
        }
    }
}

static inline double squaredDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Build exactly the same greedy candidate as the original algorithm.  max_d2
// caches each point's distance to the farthest current member, avoiding a
// complete rescan of the cluster on every greedy step.
static void generateCandidateCluster(int seed,
                                     const std::vector<unsigned char>& clustered,
                                     const std::vector<Point>& points,
                                     double threshold2,
                                     std::vector<int>& members,
                                     std::vector<double>& max_d2,
                                     std::vector<unsigned char>& in_cluster) {
    const int n = static_cast<int>(points.size());
    std::fill(in_cluster.begin(), in_cluster.end(), 0);
    std::fill(max_d2.begin(), max_d2.end(), 0.0);
    members.clear();
    members.push_back(seed);
    in_cluster[seed] = 1;

    int newest = seed;
    for (;;) {
        int closest = -1;
        double minimum_diameter2 = std::numeric_limits<double>::max();
        const Point& added = points[newest];
        for (int candidate = 0; candidate < n; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            max_d2[candidate] = std::max(max_d2[candidate],
                                         squaredDistance(points[candidate], added));
            if (max_d2[candidate] < threshold2 &&
                max_d2[candidate] < minimum_diameter2) {
                minimum_diameter2 = max_d2[candidate];
                closest = candidate;
            }
        }
        if (closest < 0) break;
        in_cluster[closest] = 1;
        members.push_back(closest);
        newest = closest;
    }
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         double threshold, MPI_Comm comm,
                                         int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    const double threshold2 = threshold * threshold;
    std::vector<unsigned char> clustered(n, 0), in_cluster(n);
    std::vector<double> max_d2(n);
    std::vector<int> local_members, winning_members;
    std::vector<Cluster> clusters;
    int remaining = n;

    while (remaining > 0) {
        int local_cardinality = -1;
        int local_seed = std::numeric_limits<int>::max();
        winning_members.clear();

        // Cyclic ownership remains well balanced as clustered points disappear.
        for (int seed = rank; seed < n; seed += ranks) {
            if (clustered[seed]) continue;
            generateCandidateCluster(seed, clustered, points, threshold2,
                                     local_members, max_d2, in_cluster);
            const int cardinality = static_cast<int>(local_members.size());
            if (cardinality > local_cardinality ||
                (cardinality == local_cardinality && seed < local_seed)) {
                local_cardinality = cardinality;
                local_seed = seed;
                winning_members = local_members;
            }
        }

        // MPI_MAXLOC supplies the original algorithm's deterministic tie break:
        // for equal cardinalities, the smallest seed index wins.
        struct { int cardinality; int seed; } local{local_cardinality, local_seed}, global{};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, comm);
        if (global.cardinality <= 0 || global.seed == std::numeric_limits<int>::max()) break;

        const int owner = global.seed % ranks;
        if (rank != owner) winning_members.resize(global.cardinality);
        MPI_Bcast(winning_members.data(), global.cardinality, MPI_INT, owner, comm);

        clusters.push_back({winning_members, global.seed});
        for (int member : winning_members) {
            if (!clustered[member]) { clustered[member] = 1; --remaining; }
        }
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points, double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double diameter2 = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i)
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                diameter2 = std::max(diameter2, squaredDistance(
                    points[cluster.members[i]], points[cluster.members[j]]));
        const double diameter = std::sqrt(diameter2);
        if (c < 10) std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                                c, cluster.members.size(), cluster.seed_point, diameter);
        if (diameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        c, diameter, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    int clustered_count = 0;
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], c);
                valid = false;
            } else ++clustered_count;
            membership[member] = static_cast<int>(c);
        }
    }
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count, points.size() - clustered_count);
    return valid;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false, print_results_flag = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) num_points = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threshold = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); parse_status = 2; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parse_status = 1; }
    }
    if (parse_status) { MPI_Finalize(); return parse_status == 2 ? 0 : 1; }
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                                   num_points, threshold);
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\nNumber of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\nValidation: %s\nMPI processes: %d\n",
                    threshold, validate ? "enabled" : "disabled", ranks);
    }
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, MPI_COMM_WORLD, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int return_code = 0;
    if (rank == 0) {
        const long elapsed_ms = static_cast<long>(max_elapsed * 1000.0);
        std::printf("Clustering time: %ld ms\nClusters found: %zu\n", elapsed_ms, clusters.size());
        int total = 0, maximum = 0;
        for (const auto& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total += size; maximum = std::max(maximum, size);
        }
        const double average = clusters.empty() ? 0.0 : static_cast<double>(total) / clusters.size();
        std::printf("Points clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\n",
                    total, num_points, 100.0 * total / num_points, average, maximum);
        const double cps = max_elapsed > 0.0 ? clusters.size() / max_elapsed : 0.0;
        const double pps = max_elapsed > 0.0 ? num_points / max_elapsed : 0.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n", cps, pps);
        if (print_results_flag) {
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members) membership[member] = static_cast<int>(c);
            std::vector<double> data(membership.begin(), membership.end());
            print_results(data, "ClusterMembership");
        }
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return_code = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&return_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return return_code;
}

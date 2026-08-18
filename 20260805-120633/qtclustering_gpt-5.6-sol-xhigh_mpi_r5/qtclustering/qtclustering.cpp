// QT Clustering Benchmark - distributed-memory MPI version
//
// Candidate clusters are independent within each QT round.  MPI ranks own
// seeds cyclically; a MAXLOC reduction selects the largest candidate while
// retaining the sequential implementation's lowest-seed tie break.

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters.  Rank zero calls this and
// broadcasts the result, so the input is identical on every MPI rank.
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

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

            points[count++] = {x, y};
            --group_cnt;
        }
    }
}

inline double pointDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Construct a rank-local distance table.  Avoiding an all-to-all table
// exchange saves O(N^2) network traffic per rank; the replicated computation
// is small compared with candidate generation and scales independently on
// distributed nodes.  Candidate growth then reads one contiguous row per
// newly admitted member.
std::vector<double> buildDistanceTable(const std::vector<Point>& points,
                                       MPI_Comm comm, const int rank) {
    const int N = static_cast<int>(points.size());
    const std::size_t element_count =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(N);

    if (element_count > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Error: distance table exceeds supported size\n");
        }
        MPI_Abort(comm, 2);
    }

    std::vector<double> distances(element_count);
    for (int i = 0; i < N; ++i) {
        double* const row = distances.data() + static_cast<std::size_t>(i) * N;
        const Point& p = points[i];
        for (int j = 0; j < N; ++j) {
            row[j] = pointDistance(p, points[j]);
        }
    }
    return distances;
}

struct CandidateScratch {
    explicit CandidateScratch(const int point_count)
        : max_distance(point_count) {
        members.reserve(point_count);
        active.reserve(point_count);
        next_active.reserve(point_count);
    }

    std::vector<double> max_distance;
    std::vector<int> members;
    std::vector<int> active;
    std::vector<int> next_active;
};

// Grow one candidate with exactly the same ordering rules as the sequential
// algorithm.  max_distance is updated only for the newest member, avoiding
// the sequential implementation's repeated scan over every existing member.
int generateCandidateCluster(const int seed_point,
                             const std::vector<int>& unclustered,
                             const std::vector<double>& distances,
                             const double threshold, const int point_count,
                             CandidateScratch& scratch) {
    scratch.members.clear();
    scratch.active.clear();
    scratch.members.push_back(seed_point);

    const double* row = distances.data() +
                        static_cast<std::size_t>(seed_point) * point_count;
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    for (const int candidate : unclustered) {
        const double diameter = row[candidate];
        scratch.max_distance[candidate] = diameter;
        if (candidate != seed_point && diameter < threshold) {
            scratch.active.push_back(candidate);
            if (diameter < min_diameter) {
                min_diameter = diameter;
                closest_point = candidate;
            }
        }
    }

    while (closest_point >= 0) {
        scratch.members.push_back(closest_point);

        row = distances.data() +
              static_cast<std::size_t>(closest_point) * point_count;
        closest_point = -1;
        min_diameter = std::numeric_limits<double>::max();
        scratch.next_active.clear();

        // The active list contains only points that can still satisfy the
        // threshold.  Compaction preserves increasing point-index order and
        // therefore preserves the sequential tie break.
        for (const int candidate : scratch.active) {
            if (candidate == scratch.members.back()) {
                continue;
            }

            double diameter = scratch.max_distance[candidate];
            const double new_distance = row[candidate];
            if (new_distance > diameter) {
                diameter = new_distance;
                scratch.max_distance[candidate] = diameter;
            }

            if (diameter < threshold) {
                scratch.next_active.push_back(candidate);
                if (diameter < min_diameter) {
                    min_diameter = diameter;
                    closest_point = candidate;
                }
            }
        }
        scratch.active.swap(scratch.next_active);
    }

    return static_cast<int>(scratch.members.size());
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, MPI_Comm comm,
                                  const int rank, const int process_count) {
    const int N = static_cast<int>(points.size());
    const std::vector<double> distances =
        buildDistanceTable(points, comm, rank);

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) {
        unclustered[i] = i;
    }

    std::vector<Cluster> clusters;
    clusters.reserve(N);
    CandidateScratch scratch(N);
    std::vector<int> local_best_members;
    std::vector<int> winning_members;

    while (!unclustered.empty()) {
        int local_winner[2] = {-1, INT_MAX};  // cardinality, seed
        local_best_members.clear();

        // Cyclic ownership by current position gives every rank either floor
        // or ceil candidates even after earlier clusters remove many seeds.
        for (std::size_t position = static_cast<std::size_t>(rank);
             position < unclustered.size(); position += process_count) {
            const int seed = unclustered[position];
            const int cardinality = generateCandidateCluster(
                seed, unclustered, distances, threshold, N, scratch);
            if (cardinality > local_winner[0]) {
                local_winner[0] = cardinality;
                local_winner[1] = seed;
                local_best_members = scratch.members;
            }
        }

        int global_winner[2] = {-1, INT_MAX};
        // MPI_MAXLOC maximizes cardinality and selects the lowest seed on ties,
        // matching the original ordered scan through unclustered points.
        MPI_Allreduce(local_winner, global_winner, 1, MPI_2INT, MPI_MAXLOC,
                      comm);

        const int cardinality = global_winner[0];
        const int best_seed = global_winner[1];
        if (best_seed == INT_MAX || cardinality <= 0) {
            break;
        }

        const auto winner_position =
            std::lower_bound(unclustered.begin(), unclustered.end(), best_seed);
        const int owner = static_cast<int>(
            std::distance(unclustered.begin(), winner_position) % process_count);
        winning_members.resize(cardinality);
        if (rank == owner) {
            winning_members = local_best_members;
        }
        MPI_Bcast(winning_members.data(), cardinality, MPI_INT, owner, comm);

        clusters.push_back({winning_members, best_seed});
        for (const int member : winning_members) {
            clustered[member] = 1;
        }
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](const int point) {
                               return clustered[point] != 0;
                           }),
            unclustered.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");

    for (std::size_t c = 0; c < clusters.size(); ++c) {
        const Cluster& cluster = clusters[c];
        double max_diameter = 0.0;
        for (std::size_t i = 0; i < cluster.members.size(); ++i) {
            for (std::size_t j = i + 1; j < cluster.members.size(); ++j) {
                max_diameter = std::max(
                    max_diameter,
                    pointDistance(points[cluster.members[i]],
                                  points[cluster.members[j]]));
            }
        }

        if (c < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                        c, cluster.members.size(), cluster.seed_point,
                        max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            std::printf(
                "ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c,
                max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (std::size_t c = 0; c < clusters.size(); ++c) {
        for (const int member : clusters[c].members) {
            if (membership[member] >= 0) {
                std::printf(
                    "ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                    member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    const int clustered_count = static_cast<int>(std::count_if(
        membership.begin(), membership.end(), [](const int cluster) {
            return cluster >= 0;
        }));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
                points.size(), clustered_count,
                points.size() - static_cast<std::size_t>(clustered_count));
    return valid;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool print_results_requested = false;
    int parse_status = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parse_status = 2;
            break;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parse_status = 1;
            break;
        }
    }

    if (parse_status != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            std::printf(
                "Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("QT Clustering Benchmark\n");
        std::printf("Number of points: %d\n", num_points);
        std::printf("Distance threshold: %.2f\n", threshold);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", process_count);
    }

    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    static_assert(std::is_standard_layout_v<Point> &&
                      sizeof(Point) == 2 * sizeof(double),
                  "Point must be two contiguous doubles");
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, MPI_COMM_WORLD, rank, process_count);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int return_code = 0;
    if (rank == 0) {
        const long long cluster_milliseconds =
            static_cast<long long>(cluster_time * 1000.0);
        std::printf("Clustering time: %lld ms\n", cluster_milliseconds);
        std::printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        const double average_cluster_size = clusters.empty()
                                                ? 0.0
                                                : static_cast<double>(
                                                      total_clustered) /
                                                      clusters.size();

        std::printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
                    num_points, 100.0 * total_clustered / num_points);
        std::printf("Average cluster size: %.2f\n", average_cluster_size);
        std::printf("Maximum cluster size: %d\n", max_cluster_size);

        const double clusters_per_second =
            cluster_time > 0.0 ? clusters.size() / cluster_time : 0.0;
        const double points_per_second =
            cluster_time > 0.0 ? num_points / cluster_time : 0.0;
        std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                    clusters_per_second, points_per_second);

        if (print_results_requested) {
            std::vector<double> membership_data;
            membership_data.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (std::size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int cluster : membership) {
                membership_data.push_back(static_cast<double>(cluster));
            }
            print_results(membership_data, "ClusterMembership");
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

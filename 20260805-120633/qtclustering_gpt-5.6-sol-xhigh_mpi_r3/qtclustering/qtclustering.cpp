// QT Clustering Benchmark - MPI distributed-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

        // The original expression cannot make practical progress for N <= 30.
        // Retain its data stream for normal benchmark sizes while making the
        // small inputs useful for correctness testing.
        if (N <= 30 && group_cnt == 0) {
            group_cnt = 1;
        }
        
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

// Every rank needs arbitrary point pairs while growing its assigned candidate
// clusters.  Build rows in parallel, then replicate the read-only matrix once.
// The matrix turns repeated square roots in the hot loop into contiguous loads.
std::vector<double> buildDistanceMatrix(const std::vector<Point>& points,
                                        MPI_Comm communicator,
                                        const int rank,
                                        const int process_count) {
    const size_t point_count = points.size();
    if (point_count != 0 && point_count > SIZE_MAX / point_count) {
        throw std::length_error("distance matrix size overflow");
    }

    const size_t element_count = point_count * point_count;
    std::vector<double> distances(element_count);
    const size_t first_row = point_count * static_cast<size_t>(rank) /
                             static_cast<size_t>(process_count);
    const size_t last_row = point_count * static_cast<size_t>(rank + 1) /
                            static_cast<size_t>(process_count);

    auto calculate_rows = [&](const size_t begin, const size_t end) {
        for (size_t row = begin; row < end; ++row) {
            double* const row_data = distances.data() + row * point_count;
            for (size_t column = 0; column < point_count; ++column) {
                row_data[column] = distance(points[row], points[column]);
            }
        }
    };
    calculate_rows(first_row, last_row);

    // MPI_Allgatherv uses int counts through MPI 3.1. Inputs too large for
    // that interface are already far beyond practical replicated-matrix
    // sizes, but computing all rows locally keeps behavior well-defined.
    if (element_count <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> receive_counts(process_count);
        std::vector<int> displacements(process_count);
        for (int process = 0; process < process_count; ++process) {
            const size_t begin = point_count * static_cast<size_t>(process) /
                                 static_cast<size_t>(process_count);
            const size_t end = point_count * static_cast<size_t>(process + 1) /
                               static_cast<size_t>(process_count);
            receive_counts[process] = static_cast<int>((end - begin) * point_count);
            displacements[process] = static_cast<int>(begin * point_count);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                       distances.data(), receive_counts.data(), displacements.data(),
                       MPI_DOUBLE, communicator);
    } else {
        calculate_rows(0, first_row);
        calculate_rows(last_row, point_count);
    }

    return distances;
}

// Scratch space is reused for every seed. Epoch marking avoids clearing an
// N-element membership array for each candidate cluster.
class CandidateWorkspace {
  public:
    explicit CandidateWorkspace(const int point_count)
        : max_diameter_(point_count), membership_epoch_(point_count, 0) {
        members_.reserve(point_count);
    }

    const std::vector<int>& generate(const int seed_point,
                                     const std::vector<int>& unclustered_indices,
                                     const double* const distances,
                                     const double threshold,
                                     const int point_count) {
        advanceEpoch();
        members_.clear();
        members_.push_back(seed_point);
        membership_epoch_[seed_point] = epoch_;

        int newest_member = seed_point;
        bool first_member = true;
        while (members_.size() < unclustered_indices.size()) {
            const double* const distance_row =
                distances + static_cast<size_t>(newest_member) * point_count;
            int closest_point = -1;
            double minimum_diameter = std::numeric_limits<double>::max();

            // Updating with only the newly added member produces exactly the
            // same maximum diameter as rescanning all members, in the same
            // ascending candidate order used by the sequential algorithm.
            for (const int candidate : unclustered_indices) {
                if (membership_epoch_[candidate] == epoch_) {
                    continue;
                }

                const double candidate_diameter = first_member
                    ? distance_row[candidate]
                    : std::max(max_diameter_[candidate], distance_row[candidate]);
                max_diameter_[candidate] = candidate_diameter;

                if (candidate_diameter < threshold &&
                    candidate_diameter < minimum_diameter) {
                    minimum_diameter = candidate_diameter;
                    closest_point = candidate;
                }
            }

            if (closest_point < 0) {
                break;
            }
            membership_epoch_[closest_point] = epoch_;
            members_.push_back(closest_point);
            newest_member = closest_point;
            first_member = false;
        }
        return members_;
    }

  private:
    void advanceEpoch() {
        ++epoch_;
        if (epoch_ == 0) {
            std::fill(membership_epoch_.begin(), membership_epoch_.end(), 0);
            epoch_ = 1;
        }
    }

    std::vector<double> max_diameter_;
    std::vector<uint32_t> membership_epoch_;
    std::vector<int> members_;
    uint32_t epoch_ = 0;
};

struct CandidateChoice {
    int cardinality;
    int seed;
    int owner;
};

static_assert(sizeof(CandidateChoice) == 3 * sizeof(int));

// Select greatest cardinality, then the lowest seed. The final owner tie-break
// is defensive; a seed is assigned to exactly one rank.
void reduceCandidateChoice(void* input_buffer, void* output_buffer,
                           int* length, MPI_Datatype*) {
    const auto* input = static_cast<const CandidateChoice*>(input_buffer);
    auto* output = static_cast<CandidateChoice*>(output_buffer);
    for (int i = 0; i < *length; ++i) {
        const bool input_is_better =
            input[i].cardinality > output[i].cardinality ||
            (input[i].cardinality == output[i].cardinality &&
             (input[i].seed < output[i].seed ||
              (input[i].seed == output[i].seed && input[i].owner < output[i].owner)));
        if (input_is_better) {
            output[i] = input[i];
        }
    }
}

// Main QT clustering algorithm. Candidate seeds are distributed cyclically to
// keep uneven cluster-growth work balanced. All ranks maintain identical
// clustered state; only rank zero retains the completed clusters for output.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  MPI_Comm communicator,
                                  const int rank,
                                  const int process_count) {
    const int point_count = static_cast<int>(points.size());
    const std::vector<double> distances =
        buildDistanceMatrix(points, communicator, rank, process_count);
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    for (int i = 0; i < point_count; ++i) {
        unclustered_indices[i] = i;
    }

    std::vector<Cluster> clusters;
    CandidateWorkspace workspace(point_count);
    std::vector<int> local_best_members;
    std::vector<int> winning_members;
    local_best_members.reserve(point_count);
    winning_members.reserve(point_count);

    MPI_Datatype choice_type;
    MPI_Op choice_operation;
    MPI_Type_contiguous(3, MPI_INT, &choice_type);
    MPI_Type_commit(&choice_type);
    MPI_Op_create(reduceCandidateChoice, 1, &choice_operation);

    while (!unclustered_indices.empty()) {
        CandidateChoice local_choice{-1, INT_MAX, rank};
        local_best_members.clear();

        // Cyclic ownership over the current compacted list avoids systematic
        // imbalance when neighboring points generate similarly costly seeds.
        for (size_t position = static_cast<size_t>(rank);
             position < unclustered_indices.size();
             position += static_cast<size_t>(process_count)) {
            const int seed = unclustered_indices[position];
            const std::vector<int>& candidate_members = workspace.generate(
                seed, unclustered_indices, distances.data(), threshold, point_count);
            const int cardinality = static_cast<int>(candidate_members.size());
            if (cardinality > local_choice.cardinality ||
                (cardinality == local_choice.cardinality && seed < local_choice.seed)) {
                local_choice.cardinality = cardinality;
                local_choice.seed = seed;
                local_best_members.assign(candidate_members.begin(), candidate_members.end());
            }
        }

        CandidateChoice global_choice{-1, INT_MAX, INT_MAX};
        MPI_Allreduce(&local_choice, &global_choice, 1, choice_type,
                      choice_operation, communicator);
        if (global_choice.cardinality <= 0 || global_choice.seed == INT_MAX) {
            break;
        }

        if (rank == global_choice.owner) {
            winning_members = local_best_members;
        } else {
            winning_members.resize(static_cast<size_t>(global_choice.cardinality));
        }
        MPI_Bcast(winning_members.data(), global_choice.cardinality, MPI_INT,
                  global_choice.owner, communicator);

        if (rank == 0) {
            clusters.push_back(Cluster{winning_members, global_choice.seed});
        }
        for (const int member : winning_members) {
            clustered[member] = 1;
        }

        size_t write_position = 0;
        for (const int index : unclustered_indices) {
            if (!clustered[index]) {
                unclustered_indices[write_position++] = index;
            }
        }
        unclustered_indices.resize(write_position);
    }

    MPI_Op_free(&choice_operation);
    MPI_Type_free(&choice_type);
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0; // 0 = run, 1 = error, 2 = help

    // Rank zero owns command-line diagnostics, then distributes one canonical
    // configuration so launchers never produce duplicate benchmark output.
    if (rank == 0) {
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
                parse_status = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 1;
                break;
            }
        }

        if (parse_status == 0 && (num_points <= 0 || threshold <= 0.0)) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            parse_status = 1;
        }
    }

    int configuration[4] = {
        num_points, validate ? 1 : 0, printResults ? 1 : 0, parse_status
    };
    MPI_Bcast(configuration, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    num_points = configuration[0];
    validate = configuration[1] != 0;
    printResults = configuration[2] != 0;
    parse_status = configuration[3];
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", process_count);
    }

    // Generate one deterministic input on rank zero and replicate it. This
    // avoids depending on per-node C library random-number implementations.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }

    MPI_Datatype point_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&point_type);

    // A barrier excludes launcher skew. Report the slowest rank, which is the
    // meaningful elapsed time for a distributed collective algorithm.
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold, MPI_COMM_WORLD,
                                rank, process_count);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: clustering failed: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        printf("Clustering time: %lld ms\n",
               static_cast<long long>(cluster_time * 1000.0));
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (const Cluster& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
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

        const double clusters_per_sec = clusters.size() / cluster_time;
        const double points_per_sec = num_points / cluster_time;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
            // Serialize cluster membership in the original point order.
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (const int member : clusters[c].members) {
                    membership[member] = static_cast<int>(c);
                }
            }
            for (const int membership_index : membership) {
                membershipData.push_back(static_cast<double>(membership_index));
            }
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}

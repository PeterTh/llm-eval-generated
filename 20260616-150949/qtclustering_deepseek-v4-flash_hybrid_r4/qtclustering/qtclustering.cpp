// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Parallel Version
//
// QT (Quality Threshold) clustering builds clusters by starting with a seed
// point and iteratively adding the closest point that maintains the cluster's
// diameter below a threshold. This hybrid implementation leverages MPI for
// inter-node parallelism (distributing seed evaluation across ranks), OpenMP
// for intra-node parallel reductions, and CUDA for GPU-accelerated distance
// computations in the inner loop.

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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

// GPU state management: device pointers and host-side mirrors for efficient
// data transfer to/from the GPU accelerator
struct GPUState {
    double *d_points_x = nullptr;
    double *d_points_y = nullptr;
    uint8_t *d_clustered = nullptr;
    uint8_t *d_in_cluster = nullptr;
    int *d_cluster_members = nullptr;
    double *d_max_dists = nullptr;
    uint8_t *d_valid = nullptr;

    std::vector<double> h_max_dists;
    std::vector<uint8_t> h_valid;
    std::vector<uint8_t> h_clustered;
    std::vector<uint8_t> h_in_cluster;

    int num_points = 0;
    int max_members = 0;
};

// ---------------------------------------------------------------------------
// CUDA kernel: computes, for every candidate point, the maximum distance to
// all current cluster members. Candidates already clustered or in the current
// cluster are skipped. Output arrays are scanned on the host to find the best
// (minimum max-distance) valid candidate.
// ---------------------------------------------------------------------------
__global__ void computeCandidateMaxDists(
    const double* points_x,
    const double* points_y,
    const uint8_t* clustered,
    const uint8_t* in_cluster,
    const int* cluster_members,
    int num_members,
    int num_points,
    double threshold,
    double* max_dists,
    uint8_t* valid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    if (clustered[idx] || in_cluster[idx]) {
        valid[idx] = 0;
        return;
    }

    double px = points_x[idx];
    double py = points_y[idx];
    double max_d = 0.0;

    for (int i = 0; i < num_members; ++i) {
        int m = cluster_members[i];
        double dx = px - points_x[m];
        double dy = py - points_y[m];
        double dist = sqrt(dx * dx + dy * dy);
        if (dist > max_d) max_d = dist;
    }

    if (max_d < threshold) {
        valid[idx] = 1;
        max_dists[idx] = max_d;
    } else {
        valid[idx] = 0;
    }
}

// ---------------------------------------------------------------------------
// GPU state management helpers
// ---------------------------------------------------------------------------

// Allocate device memory and upload point coordinates
void initGPUState(GPUState& state, const std::vector<Point>& points) {
    int N = static_cast<int>(points.size());
    state.num_points = N;
    state.max_members = N;

    CUDA_CHECK(cudaMalloc(&state.d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_points_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_clustered, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&state.d_in_cluster, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&state.d_cluster_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_max_dists, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state.d_valid, N * sizeof(uint8_t)));

    state.h_max_dists.resize(N);
    state.h_valid.resize(N);
    state.h_clustered.assign(N, 0);
    state.h_in_cluster.assign(N, 0);

    // Transpose points into separate X/Y arrays for coalesced GPU access
    std::vector<double> h_x(N), h_y(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        h_x[i] = points[i].x;
        h_y[i] = points[i].y;
    }
    CUDA_CHECK(cudaMemcpy(state.d_points_x, h_x.data(),
                N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_points_y, h_y.data(),
                N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_clustered, 0, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(state.d_in_cluster, 0, N * sizeof(uint8_t)));
}

void updateGPUClustered(GPUState& state, const std::vector<uint8_t>& clustered) {
    #pragma omp parallel for
    for (int i = 0; i < state.num_points; ++i) {
        state.h_clustered[i] = clustered[i];
    }
    CUDA_CHECK(cudaMemcpy(state.d_clustered, state.h_clustered.data(),
                state.num_points * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void updateGPUInCluster(GPUState& state, const std::vector<uint8_t>& in_cluster) {
    #pragma omp parallel for
    for (int i = 0; i < state.num_points; ++i) {
        state.h_in_cluster[i] = in_cluster[i];
    }
    CUDA_CHECK(cudaMemcpy(state.d_in_cluster, state.h_in_cluster.data(),
                state.num_points * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void destroyGPUState(GPUState& state) {
    if (state.d_points_x)      CUDA_CHECK(cudaFree(state.d_points_x));
    if (state.d_points_y)      CUDA_CHECK(cudaFree(state.d_points_y));
    if (state.d_clustered)     CUDA_CHECK(cudaFree(state.d_clustered));
    if (state.d_in_cluster)    CUDA_CHECK(cudaFree(state.d_in_cluster));
    if (state.d_cluster_members) CUDA_CHECK(cudaFree(state.d_cluster_members));
    if (state.d_max_dists)     CUDA_CHECK(cudaFree(state.d_max_dists));
    if (state.d_valid)         CUDA_CHECK(cudaFree(state.d_valid));
    state.d_points_x = nullptr;
    state.d_points_y = nullptr;
    state.d_clustered = nullptr;
    state.d_in_cluster = nullptr;
    state.d_cluster_members = nullptr;
    state.d_max_dists = nullptr;
    state.d_valid = nullptr;
}

// ---------------------------------------------------------------------------
// Data generation (unchanged semantics)
// ---------------------------------------------------------------------------

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

            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// ---------------------------------------------------------------------------
// Distance computation (host-only; GPU kernels inline the same arithmetic)
// ---------------------------------------------------------------------------

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// GPU-accelerated findClosestPoint
// ---------------------------------------------------------------------------

int findClosestPointCUDA(const std::vector<int>& cluster_members,
                          const std::vector<uint8_t>& clustered,
                          const std::vector<uint8_t>& in_cluster,
                          const std::vector<Point>& points,
                          const double threshold,
                          const int point_count,
                          GPUState& gpu_state) {
    int num_members = static_cast<int>(cluster_members.size());

    CUDA_CHECK(cudaMemcpy(gpu_state.d_cluster_members, cluster_members.data(),
                num_members * sizeof(int), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;

    computeCandidateMaxDists<<<blocks, threads>>>(
        gpu_state.d_points_x, gpu_state.d_points_y,
        gpu_state.d_clustered, gpu_state.d_in_cluster,
        gpu_state.d_cluster_members, num_members,
        point_count, threshold,
        gpu_state.d_max_dists, gpu_state.d_valid);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gpu_state.h_max_dists.data(), gpu_state.d_max_dists,
                point_count * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_state.h_valid.data(), gpu_state.d_valid,
                point_count * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Host-side reduction with OpenMP: find the candidate with smallest
    // max-dist among all valid points
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        int local_best = -1;
        double local_best_dist = std::numeric_limits<double>::max();

        #pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (gpu_state.h_valid[candidate] &&
                gpu_state.h_max_dists[candidate] < local_best_dist) {
                local_best_dist = gpu_state.h_max_dists[candidate];
                local_best = candidate;
            }
        }

        #pragma omp critical
        {
            if (local_best >= 0 && local_best_dist < best_dist) {
                best_dist = local_best_dist;
                best = local_best;
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// CPU+OpenMP fallback for findClosestPoint
// ---------------------------------------------------------------------------

int findClosestPointCPU(const std::vector<int>& cluster_members,
                         const std::vector<uint8_t>& clustered,
                         const std::vector<uint8_t>& in_cluster,
                         const std::vector<Point>& points,
                         const double threshold,
                         const int point_count) {
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();
    int num_members = static_cast<int>(cluster_members.size());

    #pragma omp parallel
    {
        int local_best = -1;
        double local_best_dist = std::numeric_limits<double>::max();

        #pragma omp for nowait
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            double max_dist = 0.0;
            for (int i = 0; i < num_members; ++i) {
                const int member = cluster_members[i];
                const double dist = distance(points[candidate], points[member]);
                if (dist > max_dist) max_dist = dist;
            }

            if (max_dist < threshold && max_dist < local_best_dist) {
                local_best_dist = max_dist;
                local_best = candidate;
            }
        }

        #pragma omp critical
        {
            if (local_best >= 0 && local_best_dist < best_dist) {
                best_dist = local_best_dist;
                best = local_best;
            }
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Candidate cluster generation (CUDA-accelerated inner loop)
// ---------------------------------------------------------------------------

int generateCandidateCluster(const int seed_point,
                              const std::vector<uint8_t>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members,
                              GPUState& gpu_state) {
    std::vector<uint8_t> in_cluster(point_count, 0);
    std::vector<int> members;

    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    updateGPUInCluster(gpu_state, in_cluster);

    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointCUDA(members, clustered, in_cluster,
                                                  points, threshold, point_count,
                                                  gpu_state);
        if (closest < 0) break;

        in_cluster[closest] = 1;
        members.push_back(closest);

        gpu_state.h_in_cluster[closest] = 1;
        CUDA_CHECK(cudaMemcpy(gpu_state.d_in_cluster + closest,
                    gpu_state.h_in_cluster.data() + closest,
                    1, cudaMemcpyHostToDevice));
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------------
// MPI communication structure for exchanging local best results
// ---------------------------------------------------------------------------

struct LocalBestInfo {
    int seed;
    int cardinality;
};

// ---------------------------------------------------------------------------
// Main QT clustering with hybrid MPI+OpenMP+CUDA
// ---------------------------------------------------------------------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold,
                                   GPUState& gpu_state) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const int N = static_cast<int>(points.size());
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    updateGPUClustered(gpu_state, clustered);

    while (!unclustered_indices.empty()) {
        int local_best_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // MPI: distribute seed evaluation across ranks (round-robin)
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            if (static_cast<int>(i) % mpi_size != mpi_rank) continue;

            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, N,
                &candidate_members, gpu_state);

            if (cardinality > local_best_cardinality) {
                local_best_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // MPI_Allgather: share local best info across all ranks
        LocalBestInfo local_info = {local_best_seed, local_best_cardinality};
        std::vector<LocalBestInfo> all_info(mpi_size);
        MPI_Allgather(&local_info, sizeof(LocalBestInfo), MPI_BYTE,
                      all_info.data(), sizeof(LocalBestInfo), MPI_BYTE,
                      MPI_COMM_WORLD);

        // Deterministic tie-breaking: first (lowest rank) with the largest
        // cardinality wins, matching sequential round-robin ordering.
        int global_best_seed = -1;
        int global_best_cardinality = -1;
        int global_best_rank = -1;
        for (int r = 0; r < mpi_size; ++r) {
            if (all_info[r].cardinality > global_best_cardinality) {
                global_best_cardinality = all_info[r].cardinality;
                global_best_seed = all_info[r].seed;
                global_best_rank = r;
            }
        }

        if (global_best_seed < 0 || global_best_cardinality <= 0) break;

        // Broadcast winning cluster members from owner rank
        int members_size = (mpi_rank == global_best_rank)
            ? static_cast<int>(local_best_members.size()) : 0;
        MPI_Bcast(&members_size, 1, MPI_INT, global_best_rank, MPI_COMM_WORLD);

        std::vector<int> global_members;
        if (mpi_rank == global_best_rank) {
            global_members = std::move(local_best_members);
        } else {
            global_members.resize(members_size);
        }
        if (members_size > 0) {
            MPI_Bcast(global_members.data(), members_size, MPI_INT,
                      global_best_rank, MPI_COMM_WORLD);
        }

        // Only rank 0 records the cluster for output
        if (mpi_rank == 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members = global_members;
            clusters.push_back(cluster);
        }

        // All ranks update their local state identically
        #pragma omp parallel for
        for (int i = 0; i < members_size; ++i) {
            clustered[global_members[i]] = 1;
        }
        updateGPUClustered(gpu_state, clustered);

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation (unchanged logic, runs only on rank 0)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Main: MPI init, CUDA device setup, data generation, hybrid clustering
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Assign a GPU to each MPI rank (round-robin if more ranks than GPUs)
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_id = mpi_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Rank 0 parses command line, then broadcasts to all ranks
    if (mpi_rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("CUDA device: %d (of %d)\n", device_id, num_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks generate the same data (deterministic seed)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Allocate GPU memory and upload point coordinates
    GPUState gpu_state;
    initGPUState(gpu_state, points);

    MPI_Barrier(MPI_COMM_WORLD);

    // Hybrid parallel clustering (MPI + CUDA + OpenMP within)
    auto cluster_start = std::chrono::high_resolution_clock::now();

    std::vector<Cluster> clusters = qtClustering(points, threshold, gpu_state);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    destroyGPUState(gpu_state);

    // Rank 0 prints results and validates
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
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

        const double time_sec = cluster_time.count() / 1000.0;
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
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return 0;
}

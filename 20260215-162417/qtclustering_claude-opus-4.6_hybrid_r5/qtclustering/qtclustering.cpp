// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// CUDA kernel: each block generates a full candidate cluster for one seed.
// Uses strided access so blockDim.x threads can handle N > blockDim.x candidates.
__global__ void generateClustersKernel(
    const double* __restrict__ px, const double* __restrict__ py,
    const int* __restrict__ clustered,
    int N, double threshold,
    const int* __restrict__ seeds, int num_seeds,
    int* g_members,        // [num_seeds * N]
    int* g_in_cluster,     // [num_seeds * N]
    int* out_cardinality)  // [num_seeds]
{
    extern __shared__ char smem[];
    double* s_dist = (double*)smem;
    int* s_idx = (int*)(s_dist + blockDim.x);

    int bid = blockIdx.x;
    if (bid >= num_seeds) return;

    int seed = seeds[bid];
    int* my_members = g_members + (long long)bid * N;
    int* my_in_cluster = g_in_cluster + (long long)bid * N;
    int tid = threadIdx.x;

    // Initialize in_cluster to 0
    for (int c = tid; c < N; c += blockDim.x)
        my_in_cluster[c] = 0;
    __syncthreads();

    if (tid == 0) {
        my_members[0] = seed;
        my_in_cluster[seed] = 1;
    }
    __syncthreads();

    int num_members = 1;

    while (num_members < N) {
        // Each thread finds its local best candidate via striding
        double local_min = DBL_MAX;
        int local_idx = -1;

        for (int c = tid; c < N; c += blockDim.x) {
            if (clustered[c] || my_in_cluster[c]) continue;

            double cx = px[c], cy = py[c];
            double max_d = 0.0;
            for (int i = 0; i < num_members; i++) {
                int m = my_members[i];
                double dx2 = cx - px[m];
                double dy2 = cy - py[m];
                double d = sqrt(dx2 * dx2 + dy2 * dy2);
                if (d > max_d) max_d = d;
            }

            if (max_d < threshold) {
                if (max_d < local_min ||
                    (max_d == local_min && (local_idx < 0 || c < local_idx))) {
                    local_min = max_d;
                    local_idx = c;
                }
            }
        }

        // Block-level argmin reduction
        s_dist[tid] = local_min;
        s_idx[tid] = local_idx;
        __syncthreads();

        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_dist[tid + s] < s_dist[tid] ||
                    (s_dist[tid + s] == s_dist[tid] && s_idx[tid + s] >= 0 &&
                     (s_idx[tid] < 0 || s_idx[tid + s] < s_idx[tid]))) {
                    s_dist[tid] = s_dist[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }

        // All threads see the same result — uniform branch
        if (s_idx[0] < 0) break;

        if (tid == 0) {
            my_members[num_members] = s_idx[0];
            my_in_cluster[s_idx[0]] = 1;
        }
        num_members++;
        __syncthreads();
    }

    if (tid == 0) out_cardinality[bid] = num_members;
}

// Main QT clustering algorithm — hybrid MPI+OpenMP+CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int rank, int num_procs) {
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered_h(N, 0);
    std::vector<int> unclustered;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; i++) unclustered.push_back(i);

    // SoA layout for GPU
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) { px[i] = points[i].x; py[i] = points[i].y; }

    // GPU: allocate shared point/clustered arrays
    double *d_px, *d_py;
    int *d_clustered;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N * sizeof(int)));

    // Determine batch size based on available GPU memory
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    long long per_seed_bytes = 2LL * N * sizeof(int); // members + in_cluster
    int max_batch = static_cast<int>(std::min((long long)(free_mem * 0.8) / std::max(per_seed_bytes, 1LL),
                                              (long long)N));
    max_batch = std::max(max_batch, 1);

    // Pre-allocate batch GPU memory
    int *d_seeds, *d_g_members, *d_g_in_cluster, *d_cardinality;
    int *h_cardinality;
    CUDA_CHECK(cudaMalloc(&d_seeds, max_batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g_members, (long long)max_batch * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g_in_cluster, (long long)max_batch * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinality, max_batch * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_cardinality, max_batch * sizeof(int)));

    const int BLOCK_SIZE = 256;
    int smem_size = BLOCK_SIZE * (int)(sizeof(double) + sizeof(int));

    // Main clustering loop
    while (!unclustered.empty()) {
        int total_seeds = static_cast<int>(unclustered.size());
        int per_rank = (total_seeds + num_procs - 1) / num_procs;
        int my_start = std::min(rank * per_rank, total_seeds);
        int my_end = std::min(my_start + per_rank, total_seeds);
        int my_count = my_end - my_start;

        int local_max_card = -1;
        int local_best_seed = -1;
        int local_best_batch_start = -1;
        int local_best_offset = -1;

        // Process seeds in batches on GPU, use OMP for CPU-side reduction
        for (int b_start = 0; b_start < my_count; b_start += max_batch) {
            int b_count = std::min(max_batch, my_count - b_start);
            if (b_count <= 0) continue;

            // Upload seeds for this batch
            CUDA_CHECK(cudaMemcpy(d_seeds,
                unclustered.data() + my_start + b_start,
                b_count * sizeof(int), cudaMemcpyHostToDevice));

            // Launch kernel: one block per seed
            generateClustersKernel<<<b_count, BLOCK_SIZE, smem_size>>>(
                d_px, d_py, d_clustered, N, threshold,
                d_seeds, b_count,
                d_g_members, d_g_in_cluster, d_cardinality);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Download cardinalities
            CUDA_CHECK(cudaMemcpy(h_cardinality, d_cardinality,
                b_count * sizeof(int), cudaMemcpyDeviceToHost));

            // OpenMP: find best seed in this batch
            #pragma omp parallel
            {
                int t_max_card = -1;
                int t_best_seed = -1;
                int t_best_offset = -1;

                #pragma omp for nowait
                for (int i = 0; i < b_count; i++) {
                    int seed = unclustered[my_start + b_start + i];
                    int card = h_cardinality[i];
                    if (card > t_max_card ||
                        (card == t_max_card && seed < t_best_seed)) {
                        t_max_card = card;
                        t_best_seed = seed;
                        t_best_offset = i;
                    }
                }

                #pragma omp critical
                {
                    if (t_max_card > local_max_card ||
                        (t_max_card == local_max_card && t_best_seed >= 0 &&
                         (local_best_seed < 0 || t_best_seed < local_best_seed))) {
                        local_max_card = t_max_card;
                        local_best_seed = t_best_seed;
                        local_best_batch_start = b_start;
                        local_best_offset = t_best_offset;
                    }
                }
            }
        }

        // MPI: find global best cluster
        int local_info[2] = {local_max_card, local_best_seed};
        std::vector<int> all_info(num_procs * 2);
        MPI_Allgather(local_info, 2, MPI_INT, all_info.data(), 2, MPI_INT, MPI_COMM_WORLD);

        int global_best_card = -1;
        int global_best_seed = -1;
        int winning_rank = -1;
        for (int r = 0; r < num_procs; r++) {
            int rc = all_info[r * 2];
            int rs = all_info[r * 2 + 1];
            if (rc > global_best_card ||
                (rc == global_best_card && rs >= 0 &&
                 (global_best_seed < 0 || rs < global_best_seed))) {
                global_best_card = rc;
                global_best_seed = rs;
                winning_rank = r;
            }
        }

        if (global_best_seed < 0 || global_best_card <= 0) break;

        // Retrieve best cluster members
        std::vector<int> best_members(global_best_card);
        if (rank == winning_rank) {
            // Re-run the winning seed's batch to get its members
            // (The batch data may have been overwritten by later batches)
            int b_start = local_best_batch_start;
            int b_count = std::min(max_batch, my_count - b_start);

            CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data() + my_start + b_start,
                b_count * sizeof(int), cudaMemcpyHostToDevice));

            generateClustersKernel<<<b_count, BLOCK_SIZE, smem_size>>>(
                d_px, d_py, d_clustered, N, threshold,
                d_seeds, b_count,
                d_g_members, d_g_in_cluster, d_cardinality);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy the winning seed's members from GPU
            CUDA_CHECK(cudaMemcpy(best_members.data(),
                d_g_members + (long long)local_best_offset * N,
                global_best_card * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // Broadcast members
        MPI_Bcast(best_members.data(), global_best_card, MPI_INT,
                  winning_rank, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        // Update clustered status on CPU and GPU
        for (int m : best_members) clustered_h[m] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered_h.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice));

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered_h](int idx) { return clustered_h[idx]; }),
            unclustered.end());
    }

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_clustered);
    cudaFree(d_seeds);
    cudaFree(d_g_members);
    cudaFree(d_g_in_cluster);
    cudaFree(d_cardinality);
    cudaFreeHost(h_cardinality);

    return clusters;
}

// Validation
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
                double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                double dist = std::sqrt(dx * dx + dy * dy);
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
            int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) clustered_count++;

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
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Select GPU based on rank (round-robin across available devices)
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

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

    // All ranks generate identical data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long long local_cluster_time_ms = static_cast<long long>(cluster_time.count());
    long long global_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &global_cluster_time_ms, 1,
               MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %lld ms\n", global_cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }

        double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        double time_sec = global_cluster_time_ms / 1000.0;
        double clusters_per_sec = clusters.size() / time_sec;
        double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (size_t i = 0; i < clusters[c].members.size(); ++i)
                    membership[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : membership)
                membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}

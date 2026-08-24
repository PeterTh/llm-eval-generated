// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering parallelized with:
// - MPI: distributes seed points across ranks
// - OpenMP: parallelizes seed trials within each rank
// - CUDA: accelerates findClosestPoint distance computations on GPU

#include <algorithm>
#include <chrono>
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

__host__ __device__ inline double point_distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel: For each candidate point, compute the max distance to all cluster members.
// Store result_idx[c] = candidate index, result_dist[c] = max distance to cluster members.
// Only considers points where clustered[c]==0 and in_cluster[c]==0.
__global__ void findClosestPointKernel(
    const double* __restrict__ px, const double* __restrict__ py,
    const int* __restrict__ member_indices, int num_members,
    const int* __restrict__ clustered, const int* __restrict__ in_cluster,
    int point_count, double threshold,
    int* __restrict__ result_idx, double* __restrict__ result_dist)
{
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= point_count) return;

    if (clustered[candidate] || in_cluster[candidate]) {
        result_dist[candidate] = 1e30;
        result_idx[candidate] = -1;
        return;
    }

    double cx = px[candidate];
    double cy = py[candidate];
    double max_dist = 0.0;

    for (int i = 0; i < num_members; ++i) {
        int m = member_indices[i];
        double d = point_distance(cx, cy, px[m], py[m]);
        if (d > max_dist) max_dist = d;
    }

    if (max_dist < threshold) {
        result_dist[candidate] = max_dist;
        result_idx[candidate] = candidate;
    } else {
        result_dist[candidate] = 1e30;
        result_idx[candidate] = -1;
    }
}

// GPU context for one CUDA stream/device per OpenMP thread
struct GPUContext {
    double *d_px, *d_py;
    int *d_clustered, *d_in_cluster;
    int *d_members;
    int *d_result_idx;
    double *d_result_dist;
    int *h_result_idx;
    double *h_result_dist;
    cudaStream_t stream;
    int point_count;
};

void initGPUContext(GPUContext& ctx, int N) {
    ctx.point_count = N;
    cudaMalloc(&ctx.d_px, N * sizeof(double));
    cudaMalloc(&ctx.d_py, N * sizeof(double));
    cudaMalloc(&ctx.d_clustered, N * sizeof(int));
    cudaMalloc(&ctx.d_in_cluster, N * sizeof(int));
    cudaMalloc(&ctx.d_members, N * sizeof(int));
    cudaMalloc(&ctx.d_result_idx, N * sizeof(int));
    cudaMalloc(&ctx.d_result_dist, N * sizeof(double));
    cudaMallocHost(&ctx.h_result_idx, N * sizeof(int));
    cudaMallocHost(&ctx.h_result_dist, N * sizeof(double));
    cudaStreamCreate(&ctx.stream);
}

void freeGPUContext(GPUContext& ctx) {
    cudaFree(ctx.d_px);
    cudaFree(ctx.d_py);
    cudaFree(ctx.d_clustered);
    cudaFree(ctx.d_in_cluster);
    cudaFree(ctx.d_members);
    cudaFree(ctx.d_result_idx);
    cudaFree(ctx.d_result_dist);
    cudaFreeHost(ctx.h_result_idx);
    cudaFreeHost(ctx.h_result_dist);
    cudaStreamDestroy(ctx.stream);
}

// Upload points (once) and clustered state (each outer iteration) to GPU context
void uploadPoints(GPUContext& ctx, const double* px, const double* py, int N) {
    cudaMemcpyAsync(ctx.d_px, px, N * sizeof(double), cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_py, py, N * sizeof(double), cudaMemcpyHostToDevice, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
}

void uploadClustered(GPUContext& ctx, const int* clustered, int N) {
    cudaMemcpyAsync(ctx.d_clustered, clustered, N * sizeof(int), cudaMemcpyHostToDevice, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
}

// GPU-accelerated findClosestPoint
int findClosestPointGPU(GPUContext& ctx,
                        const int* members, int num_members,
                        const int* in_cluster,
                        double threshold, int point_count) {
    cudaMemcpyAsync(ctx.d_members, members, num_members * sizeof(int), cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_in_cluster, in_cluster, point_count * sizeof(int), cudaMemcpyHostToDevice, ctx.stream);

    int blockSize = 256;
    int gridSize = (point_count + blockSize - 1) / blockSize;
    findClosestPointKernel<<<gridSize, blockSize, 0, ctx.stream>>>(
        ctx.d_px, ctx.d_py, ctx.d_members, num_members,
        ctx.d_clustered, ctx.d_in_cluster, point_count, threshold,
        ctx.d_result_idx, ctx.d_result_dist);

    cudaMemcpyAsync(ctx.h_result_idx, ctx.d_result_idx, point_count * sizeof(int), cudaMemcpyDeviceToHost, ctx.stream);
    cudaMemcpyAsync(ctx.h_result_dist, ctx.d_result_dist, point_count * sizeof(double), cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    int closest_point = -1;
    double min_diameter = 1e30;
    for (int i = 0; i < point_count; ++i) {
        if (ctx.h_result_idx[i] >= 0 && ctx.h_result_dist[i] < min_diameter) {
            min_diameter = ctx.h_result_dist[i];
            closest_point = ctx.h_result_idx[i];
        }
    }
    return closest_point;
}

// Generate candidate cluster using GPU-accelerated closest point search
int generateCandidateClusterGPU(int seed_point, const int* clustered,
                                 double threshold, int point_count,
                                 GPUContext& ctx,
                                 std::vector<int>* cluster_members = nullptr) {
    std::vector<int> in_cluster_vec(point_count, 0);
    std::vector<int> members;
    in_cluster_vec[seed_point] = 1;
    members.push_back(seed_point);

    while ((int)members.size() < point_count) {
        int closest = findClosestPointGPU(ctx, members.data(), (int)members.size(),
                                          in_cluster_vec.data(), threshold, point_count);
        if (closest < 0) break;
        in_cluster_vec[closest] = 1;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return (int)members.size();
}

// Main QT clustering algorithm - hybrid MPI + OpenMP + CUDA
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  int mpi_rank, int mpi_size) {
    const int N = (int)points.size();
    std::vector<int> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Separate x,y for GPU-friendly layout
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) { px[i] = points[i].x; py[i] = points[i].y; }

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    // Create one GPU context per OpenMP thread
    int num_threads = omp_get_max_threads();
    std::vector<GPUContext> gpu_contexts(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        initGPUContext(gpu_contexts[t], N);
        uploadPoints(gpu_contexts[t], px.data(), py.data(), N);
    }

    while (!unclustered_indices.empty()) {
        // Upload current clustered state to all GPU contexts
        for (int t = 0; t < num_threads; ++t) {
            uploadClustered(gpu_contexts[t], clustered.data(), N);
        }

        int num_unclustered = (int)unclustered_indices.size();

        // Compute how many seeds this rank handles
        // Strided distribution: rank r gets indices r, r+mpi_size, r+2*mpi_size, ...
        int local_count = 0;
        for (int i = mpi_rank; i < num_unclustered; i += mpi_size) local_count++;

        int local_best_cardinality = -1;
        int local_best_seed = N; // use high value so lower seed wins ties
        std::vector<int> local_best_members;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            GPUContext& my_ctx = gpu_contexts[tid];
            int thread_best_cardinality = -1;
            int thread_best_seed = N;
            std::vector<int> thread_best_members;

            #pragma omp for schedule(dynamic, 1) nowait
            for (int li = 0; li < local_count; ++li) {
                int i = mpi_rank + li * mpi_size;
                int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                std::vector<int> candidate_members;
                int cardinality = generateCandidateClusterGPU(seed, clustered.data(),
                                                              threshold, N, my_ctx,
                                                              &candidate_members);
                // Tie-break: higher cardinality wins; on tie, lower seed wins
                if (cardinality > thread_best_cardinality ||
                    (cardinality == thread_best_cardinality && seed < thread_best_seed)) {
                    thread_best_cardinality = cardinality;
                    thread_best_seed = seed;
                    thread_best_members = candidate_members;
                }
            }

            #pragma omp critical
            {
                if (thread_best_cardinality > local_best_cardinality ||
                    (thread_best_cardinality == local_best_cardinality && thread_best_seed < local_best_seed)) {
                    local_best_cardinality = thread_best_cardinality;
                    local_best_seed = thread_best_seed;
                    local_best_members = thread_best_members;
                }
            }
        }

        // MPI reduction: gather best {cardinality, seed} from all ranks
        // Use custom reduction: max cardinality, then min seed for tie-break
        int local_pair[2] = {local_best_cardinality, local_best_seed};
        int global_pair[2];
        // First: allreduce max cardinality
        MPI_Allreduce(&local_pair[0], &global_pair[0], 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_pair[0] <= 0) break;

        // Among ranks with max cardinality, find min seed
        int my_seed_for_min = (local_best_cardinality == global_pair[0]) ? local_best_seed : N + 1;
        int global_min_seed;
        MPI_Allreduce(&my_seed_for_min, &global_min_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // The winning rank is the one with matching cardinality and seed
        int winner = (local_best_cardinality == global_pair[0] && local_best_seed == global_min_seed) ? mpi_rank : mpi_size;
        int global_winner;
        MPI_Allreduce(&winner, &global_winner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // The winning rank broadcasts its cluster members
        int best_count = (global_winner == mpi_rank) ? (int)local_best_members.size() : 0;
        MPI_Bcast(&best_count, 1, MPI_INT, global_winner, MPI_COMM_WORLD);

        std::vector<int> best_members(best_count);
        int best_seed = 0;
        if (global_winner == mpi_rank) {
            best_members = local_best_members;
            best_seed = local_best_seed;
        }
        MPI_Bcast(best_members.data(), best_count, MPI_INT, global_winner, MPI_COMM_WORLD);
        MPI_Bcast(&best_seed, 1, MPI_INT, global_winner, MPI_COMM_WORLD);

        // All ranks update clustered state consistently
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int idx : best_members) clustered[idx] = 1;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    for (int t = 0; t < num_threads; ++t) freeGPUContext(gpu_contexts[t]);

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
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = (int)c;
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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", mpi_size, omp_get_max_threads());
    }

    // All ranks generate identical data (deterministic)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    long local_cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long global_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &global_cluster_time_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", global_cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0;
        int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int size = (int)clusters[i].members.size();
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        double avg_cluster_size = clusters.empty() ? 0.0 :
            (double)total_clustered / clusters.size();

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
                    membership[clusters[c].members[i]] = (int)c;
            for (int m : membership)
                membershipData.push_back((double)m);
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

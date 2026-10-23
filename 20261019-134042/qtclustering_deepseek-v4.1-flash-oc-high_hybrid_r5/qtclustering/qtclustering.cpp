// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

static constexpr int QT_BLOCK = 256;
static constexpr int QT_MAX_OMP_THREADS = 8;
static constexpr long long QT_WS_BUDGET_BYTES = 64LL * 1024 * 1024;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

static void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
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

__device__ __forceinline__ double deviceDistance(const double* __restrict__ px,
                                                 const double* __restrict__ py,
                                                 int a, int b) {
    const double dx = px[a] - px[b];
    const double dy = py[a] - py[b];
    return sqrt(fma(dx, dx, dy * dy));
}

__device__ __forceinline__ double deviceDistanceXY(const double* __restrict__ px,
                                                   const double* __restrict__ py,
                                                   int a, double bx, double by) {
    const double dx = px[a] - bx;
    const double dy = py[a] - by;
    return sqrt(fma(dx, dx, dy * dy));
}

__global__ void qtCandidateKernel(const double* __restrict__ px,
                                  const double* __restrict__ py,
                                  const unsigned char* __restrict__ clustered,
                                  int N,
                                  double threshold,
                                  const int* __restrict__ seeds,
                                  double* __restrict__ dist_ws,
                                  int* __restrict__ members_ws,
                                  int* __restrict__ card_out) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[b];
    double* dist = dist_ws + static_cast<size_t>(b) * static_cast<size_t>(N);

    for (int c = tid; c < N; c += blockDim.x) {
        if (c == seed || clustered[c]) {
            dist[c] = -1.0;
        } else {
            const double d = deviceDistance(px, py, c, seed);
            dist[c] = (d < threshold) ? d : -1.0;
        }
    }

    if (members_ws != nullptr && tid == 0) {
        members_ws[static_cast<size_t>(b) * static_cast<size_t>(N)] = seed;
    }
    __syncthreads();

    __shared__ double sval[QT_BLOCK];
    __shared__ int sidx[QT_BLOCK];

    int count = 1;
    while (count < N) {
        double best = threshold;
        int best_idx = INT_MAX;

        for (int c = tid; c < N; c += blockDim.x) {
            const double d = dist[c];
            if (d >= 0.0 && d < best) {
                best = d;
                best_idx = c;
            }
        }

        sval[tid] = best;
        sidx[tid] = best_idx;
        __syncthreads();

        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                const double ov = sval[tid + s];
                const int oi = sidx[tid + s];
                if (ov < sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                    sval[tid] = ov;
                    sidx[tid] = oi;
                }
            }
            __syncthreads();
        }

        const int winner = sidx[0];
        if (winner == INT_MAX) break;

        if (members_ws != nullptr && tid == 0) {
            members_ws[static_cast<size_t>(b) * static_cast<size_t>(N) + count] = winner;
        }
        ++count;

        const double wx = px[winner];
        const double wy = py[winner];
        for (int c = tid; c < N; c += blockDim.x) {
            if (c == winner) {
                dist[c] = -1.0;
                continue;
            }
            const double dcur = dist[c];
            if (dcur >= 0.0) {
                const double d = deviceDistanceXY(px, py, c, wx, wy);
                if (d > dcur) {
                    dist[c] = (d < threshold) ? d : -1.0;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        card_out[b] = count;
    }
}

struct GpuState {
    double* d_px = nullptr;
    double* d_py = nullptr;
    unsigned char* d_clustered = nullptr;
    double* d_b_dist = nullptr;
    int* d_b_members = nullptr;
    int* d_b_seeds = nullptr;
    int* d_b_card = nullptr;
    cudaStream_t b_stream = nullptr;
};

struct ThreadWorkspace {
    double* d_dist = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    cudaStream_t stream = nullptr;
    int capacity = 0;
};

static void ensureWorkspace(ThreadWorkspace& ws, int capacity, int N) {
    if (capacity <= ws.capacity) return;

    if (ws.d_dist != nullptr) cudaFree(ws.d_dist);
    if (ws.d_seeds != nullptr) cudaFree(ws.d_seeds);
    if (ws.d_cards != nullptr) cudaFree(ws.d_cards);
    if (ws.stream != nullptr) cudaStreamDestroy(ws.stream);

    cudaMalloc(reinterpret_cast<void**>(&ws.d_dist),
               static_cast<size_t>(capacity) * static_cast<size_t>(N) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&ws.d_seeds),
               static_cast<size_t>(capacity) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&ws.d_cards),
               static_cast<size_t>(capacity) * sizeof(int));
    cudaStreamCreateWithFlags(&ws.stream, cudaStreamNonBlocking);
    ws.capacity = capacity;
}

static void destroyWorkspace(ThreadWorkspace& ws) {
    if (ws.d_dist != nullptr) cudaFree(ws.d_dist);
    if (ws.d_seeds != nullptr) cudaFree(ws.d_seeds);
    if (ws.d_cards != nullptr) cudaFree(ws.d_cards);
    if (ws.stream != nullptr) cudaStreamDestroy(ws.stream);
    ws = ThreadWorkspace{};
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         const double threshold,
                                         const GpuState& gpu,
                                         int mpi_rank,
                                         int mpi_size,
                                         int device_id) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<ThreadWorkspace> workspaces(QT_MAX_OMP_THREADS);

    while (!unclustered_indices.empty()) {
        cudaMemcpy(gpu.d_clustered, clustered.data(), static_cast<size_t>(N),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        const long long total_seeds = static_cast<long long>(unclustered_indices.size());
        const long long seed_begin = total_seeds * static_cast<long long>(mpi_rank) /
                                     static_cast<long long>(mpi_size);
        const long long seed_end = total_seeds * static_cast<long long>(mpi_rank + 1) /
                                   static_cast<long long>(mpi_size);
        std::vector<int> seeds(unclustered_indices.begin() + static_cast<size_t>(seed_begin),
                               unclustered_indices.begin() + static_cast<size_t>(seed_end));
        const int local_seed_count = static_cast<int>(seeds.size());

        int threads = omp_get_max_threads();
        if (threads > QT_MAX_OMP_THREADS) threads = QT_MAX_OMP_THREADS;
        if (threads < 1) threads = 1;

        long long mem_chunk = QT_WS_BUDGET_BYTES /
                              (static_cast<long long>(N) * static_cast<long long>(sizeof(double)));
        if (mem_chunk < 1) mem_chunk = 1;

        int chunk = 1;
        int num_chunks = 0;
        if (local_seed_count > 0) {
            const int target_chunks = std::max(1, threads);
            chunk = (local_seed_count + target_chunks - 1) / target_chunks;
            if (chunk < 1) chunk = 1;
            if (static_cast<long long>(chunk) > mem_chunk) chunk = static_cast<int>(mem_chunk);
            num_chunks = (local_seed_count + chunk - 1) / chunk;
            if (threads > num_chunks) threads = num_chunks;
            if (threads < 1) threads = 1;
        }

        int rank_best_card = -1;
        int rank_best_seed = INT_MAX;

        if (local_seed_count > 0) {
            for (int t = 0; t < threads; ++t) {
                ensureWorkspace(workspaces[t], chunk, N);
            }

            std::vector<int> thread_card(threads, -1);
            std::vector<int> thread_seed(threads, INT_MAX);

            #pragma omp parallel num_threads(threads)
            {
                const int tid = omp_get_thread_num();
                ThreadWorkspace& ws = workspaces[tid];
                std::vector<int> host_cards(static_cast<size_t>(chunk));
                int lb_card = -1;
                int lb_seed = INT_MAX;

                cudaSetDevice(device_id);

                #pragma omp for schedule(dynamic, 1)
                for (int ci = 0; ci < num_chunks; ++ci) {
                    const int start = ci * chunk;
                    const int cnt = std::min(chunk, local_seed_count - start);

                    cudaMemcpyAsync(ws.d_seeds, seeds.data() + start,
                                    static_cast<size_t>(cnt) * sizeof(int),
                                    cudaMemcpyHostToDevice, ws.stream);
                    qtCandidateKernel<<<cnt, QT_BLOCK, 0, ws.stream>>>(
                        gpu.d_px, gpu.d_py, gpu.d_clustered, N, threshold,
                        ws.d_seeds, ws.d_dist, nullptr, ws.d_cards);
                    cudaMemcpyAsync(host_cards.data(), ws.d_cards,
                                    static_cast<size_t>(cnt) * sizeof(int),
                                    cudaMemcpyDeviceToHost, ws.stream);
                    cudaStreamSynchronize(ws.stream);

                    for (int j = 0; j < cnt; ++j) {
                        const int card = host_cards[j];
                        const int sd = seeds[start + j];
                        if (card > lb_card || (card == lb_card && sd < lb_seed)) {
                            lb_card = card;
                            lb_seed = sd;
                        }
                    }
                }

                thread_card[tid] = lb_card;
                thread_seed[tid] = lb_seed;
            }

            for (int t = 0; t < threads; ++t) {
                if (thread_card[t] > rank_best_card ||
                    (thread_card[t] == rank_best_card && thread_seed[t] < rank_best_seed)) {
                    rank_best_card = thread_card[t];
                    rank_best_seed = thread_seed[t];
                }
            }
        }

        int local_pair[2];
        local_pair[0] = rank_best_card;
        local_pair[1] = rank_best_seed;

        std::vector<int> all_pairs(static_cast<size_t>(2 * mpi_size));
        MPI_Allgather(local_pair, 2, MPI_INT, all_pairs.data(), 2, MPI_INT,
                      MPI_COMM_WORLD);

        int best_card = -1;
        int best_seed = INT_MAX;
        for (int r = 0; r < mpi_size; ++r) {
            const int card = all_pairs[static_cast<size_t>(2 * r)];
            const int sd = all_pairs[static_cast<size_t>(2 * r + 1)];
            if (card > best_card || (card == best_card && sd < best_seed)) {
                best_card = card;
                best_seed = sd;
            }
        }

        if (best_card <= 0 || best_seed < 0 || best_seed == INT_MAX) break;

        cudaMemcpyAsync(gpu.d_b_seeds, &best_seed, sizeof(int),
                        cudaMemcpyHostToDevice, gpu.b_stream);
        qtCandidateKernel<<<1, QT_BLOCK, 0, gpu.b_stream>>>(
            gpu.d_px, gpu.d_py, gpu.d_clustered, N, threshold,
            gpu.d_b_seeds, gpu.d_b_dist, gpu.d_b_members, gpu.d_b_card);

        int regenerated_card = 0;
        cudaMemcpyAsync(&regenerated_card, gpu.d_b_card, sizeof(int),
                        cudaMemcpyDeviceToHost, gpu.b_stream);
        cudaStreamSynchronize(gpu.b_stream);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(static_cast<size_t>(regenerated_card));
        if (regenerated_card > 0) {
            cudaMemcpyAsync(cluster.members.data(), gpu.d_b_members,
                            static_cast<size_t>(regenerated_card) * sizeof(int),
                            cudaMemcpyDeviceToHost, gpu.b_stream);
            cudaStreamSynchronize(gpu.b_stream);
        }
        clusters.push_back(std::move(cluster));

        for (int m : clusters.back().members) {
            clustered[m] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    for (ThreadWorkspace& ws : workspaces) {
        destroyWorkspace(ws);
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

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int local_rank = 0;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                        MPI_INFO_NULL, &local_comm);
    if (local_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(local_comm, &local_rank);
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 1) device_count = 1;
    const int device_id = local_rank % device_count;
    cudaSetDevice(device_id);
    cudaFree(nullptr);

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
            if (local_comm != MPI_COMM_NULL) MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            if (local_comm != MPI_COMM_NULL) MPI_Comm_free(&local_comm);
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        if (local_comm != MPI_COMM_NULL) MPI_Comm_free(&local_comm);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Point> points(static_cast<size_t>(num_points));
    generateSyntheticData(points, num_points);

    std::vector<double> px(static_cast<size_t>(num_points));
    std::vector<double> py(static_cast<size_t>(num_points));
    for (int i = 0; i < num_points; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    GpuState gpu;
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_px),
               static_cast<size_t>(num_points) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_py),
               static_cast<size_t>(num_points) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_clustered),
               static_cast<size_t>(num_points) * sizeof(unsigned char));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_b_dist),
               static_cast<size_t>(num_points) * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_b_members),
               static_cast<size_t>(num_points) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_b_seeds), sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&gpu.d_b_card), sizeof(int));
    cudaStreamCreateWithFlags(&gpu.b_stream, cudaStreamNonBlocking);

    cudaMemcpy(gpu.d_px, px.data(), static_cast<size_t>(num_points) * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.d_py, py.data(), static_cast<size_t>(num_points) * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, gpu, mpi_rank, mpi_size, device_id);
    const double cluster_end = MPI_Wtime();

    cudaDeviceSynchronize();

    cudaFree(gpu.d_px);
    cudaFree(gpu.d_py);
    cudaFree(gpu.d_clustered);
    cudaFree(gpu.d_b_dist);
    cudaFree(gpu.d_b_members);
    cudaFree(gpu.d_b_seeds);
    cudaFree(gpu.d_b_card);
    cudaStreamDestroy(gpu.b_stream);

    int exit_code = 0;
    if (mpi_rank == 0) {
        const long long cluster_ms =
            static_cast<long long>((cluster_end - cluster_start) * 1000.0);
        printf("Clustering time: %lld ms\n", cluster_ms);
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

        const double time_sec = (cluster_end - cluster_start);
        const double safe_time = (time_sec > 0.0) ? time_sec : 1e-9;
        const double clusters_per_sec = clusters.size() / safe_time;
        const double points_per_sec = num_points / safe_time;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);

        if (printResults) {
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

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    if (local_comm != MPI_COMM_NULL) MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return exit_code;
}

// QT Clustering Benchmark - CUDA Parallelized Version
//
// QT (Quality Threshold) clustering algorithm parallelized with CUDA.
// Each outer iteration launches one kernel; each block cooperatively
// processes one seed candidate via parallel reduction (block-per-seed).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point { double x, y; };

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
            const double x = cntr_x + dx, y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++; group_cnt--;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x, dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Cooperative kernel: one block per seed.
// Shared memory layout:
//   [0 .. nthreads)       double  smax  (per-thread max distances)
//   [nthreads .. 2*N)    int     sidx  (per-thread best candidate indices)
//   [2*N]                int     scount (current cluster size, shared)
//   [2*N + 1]            char    sdone  (done flag, shared)
// ---------------------------------------------------------------------------

__global__ void evaluateAllSeedsKernel(
    const double* d_x, const double* d_y,
    const char* d_clustered_global,
    const int* d_unclustered_indices, int num_unclustered,
    int num_points, double threshold,
    char* d_in_cluster_all,
    int* d_members_all,
    int* d_cardinalities)
{
    int sid = blockIdx.x;
    if (sid >= num_unclustered) return;

    int seed = d_unclustered_indices[sid];
    if (d_clustered_global[seed]) { d_cardinalities[sid] = 0; return; }

    char* in_cluster = d_in_cluster_all + (size_t)sid * num_points;
    int*  members    = d_members_all    + (size_t)sid * num_points;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    extern __shared__ char _smem[];
    double* smax   = (double*)_smem;
    int*    sidx    = (int*)(smax + nthreads);
    int*    scount  = (int*)(sidx + nthreads);
    char*   sdone   = (char*)(scount + 1);

    // Thread 0 initialises shared state
    if (tid == 0) {
        in_cluster[seed] = 1;
        members[0] = seed;
        *scount = 1;
        *sdone = 0;
    }
    __syncthreads();

    while (!*sdone) {
        // Each thread scans a chunk of candidate points
        double best_dist = threshold;
        int    best_cand = -1;

        int cur_count = *scount;
        if (cur_count >= num_points) break; // should not happen, but safety

        for (int cand = tid; cand < num_points; cand += nthreads) {
            if (d_clustered_global[cand] || in_cluster[cand]) continue;

            double max_d = 0.0;
            for (int i = 0; i < cur_count; ++i) {
                int m = members[i];
                double dx = d_x[cand] - d_x[m];
                double dy = d_y[cand] - d_y[m];
                double d = sqrt(dx * dx + dy * dy);
                if (d > max_d) max_d = d;
            }
            if (max_d < best_dist) {
                best_dist = max_d;
                best_cand = cand;
            }
        }

        smax[tid] = best_dist;
        sidx[tid] = best_cand;
        __syncthreads();

        // Parallel min-reduction
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (smax[tid + s] < smax[tid]) {
                    smax[tid] = smax[tid + s];
                    sidx[tid] = sidx[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (smax[0] < threshold && sidx[0] >= 0) {
                int best = sidx[0];
                in_cluster[best] = 1;
                members[*scount] = best;
                ++(*scount);
            } else {
                *sdone = 1;
            }
        }
        __syncthreads();
    }

    if (tid == 0) d_cardinalities[sid] = *scount;
}

// ---------------------------------------------------------------------------
// GPU memory management
// ---------------------------------------------------------------------------

struct GPUData {
    double* d_x = nullptr;
    double* d_y = nullptr;
    char* d_clustered = nullptr;
    int* d_unclustered = nullptr;
    char* d_in_cluster_all = nullptr;
    int* d_members_all = nullptr;
    int* d_cardinalities = nullptr;
    int cap_points = 0;
    bool points_loaded = false;
} g_gpu;

static void ensureGPUAlloc(int num_points) {
    if (num_points <= g_gpu.cap_points) return;
    if (g_gpu.d_x)              cudaFree(g_gpu.d_x);
    if (g_gpu.d_y)              cudaFree(g_gpu.d_y);
    if (g_gpu.d_clustered)      cudaFree(g_gpu.d_clustered);
    if (g_gpu.d_unclustered)    cudaFree(g_gpu.d_unclustered);
    if (g_gpu.d_in_cluster_all) cudaFree(g_gpu.d_in_cluster_all);
    if (g_gpu.d_members_all)    cudaFree(g_gpu.d_members_all);
    if (g_gpu.d_cardinalities)  cudaFree(g_gpu.d_cardinalities);

    cudaMalloc(&g_gpu.d_x,              num_points * sizeof(double));
    cudaMalloc(&g_gpu.d_y,              num_points * sizeof(double));
    cudaMalloc(&g_gpu.d_clustered,      num_points * sizeof(char));
    cudaMalloc(&g_gpu.d_unclustered,    num_points * sizeof(int));
    cudaMalloc(&g_gpu.d_in_cluster_all, (size_t)num_points * num_points * sizeof(char));
    cudaMalloc(&g_gpu.d_members_all,    (size_t)num_points * num_points * sizeof(int));
    cudaMalloc(&g_gpu.d_cardinalities,  num_points * sizeof(int));

    g_gpu.cap_points = num_points;
    g_gpu.points_loaded = false;
}

static void loadPointsToGPU(const std::vector<Point>& points, int n) {
    if (g_gpu.points_loaded) return;
    std::vector<double> h_x(n), h_y(n);
    for (int i = 0; i < n; ++i) { h_x[i] = points[i].x; h_y[i] = points[i].y; }
    cudaMemcpy(g_gpu.d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_gpu.d_y, h_y.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    g_gpu.points_loaded = true;
}

static void cleanupGPU() {
    if (g_gpu.d_x)              cudaFree(g_gpu.d_x);
    if (g_gpu.d_y)              cudaFree(g_gpu.d_y);
    if (g_gpu.d_clustered)      cudaFree(g_gpu.d_clustered);
    if (g_gpu.d_unclustered)    cudaFree(g_gpu.d_unclustered);
    if (g_gpu.d_in_cluster_all) cudaFree(g_gpu.d_in_cluster_all);
    if (g_gpu.d_members_all)    cudaFree(g_gpu.d_members_all);
    if (g_gpu.d_cardinalities)  cudaFree(g_gpu.d_cardinalities);
    g_gpu = GPUData();
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm
// ---------------------------------------------------------------------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    ensureGPUAlloc(N);
    loadPointsToGPU(points, N);

    const int tpb = 256;  // threads per block
    const size_t smem = tpb * (sizeof(double) + sizeof(int)) + 8;

    while (!unclustered_indices.empty()) {
        const int num_u = static_cast<int>(unclustered_indices.size());

        cudaMemcpy(g_gpu.d_clustered, clustered.data(),
                   N * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(g_gpu.d_unclustered, unclustered_indices.data(),
                   num_u * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(g_gpu.d_in_cluster_all, 0,
                   (size_t)num_u * N * sizeof(char));

        evaluateAllSeedsKernel<<<num_u, tpb, smem>>>(
            g_gpu.d_x, g_gpu.d_y,
            g_gpu.d_clustered,
            g_gpu.d_unclustered, num_u,
            N, threshold,
            g_gpu.d_in_cluster_all,
            g_gpu.d_members_all,
            g_gpu.d_cardinalities);
        cudaDeviceSynchronize();

        std::vector<int> h_card(num_u);
        cudaMemcpy(h_card.data(), g_gpu.d_cardinalities,
                   num_u * sizeof(int), cudaMemcpyDeviceToHost);

        int best_idx = -1, max_card = -1;
        for (int i = 0; i < num_u; ++i) {
            if (h_card[i] > max_card) { max_card = h_card[i]; best_idx = i; }
        }

        if (best_idx < 0 || max_card <= 0) break;

        std::vector<int> best_members(max_card);
        cudaMemcpy(best_members.data(),
                   g_gpu.d_members_all + (size_t)best_idx * N,
                   max_card * sizeof(int), cudaMemcpyDeviceToHost);

        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_idx];
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int p : best_members) clustered[p] = 1;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation and reporting
// ---------------------------------------------------------------------------

bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i)
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                max_diameter = std::max(max_diameter,
                    distance(points[cluster.members[i]], points[cluster.members[j]]));
        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu diameter %.4f > %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c)
        for (int m : clusters[c].members) {
            if (membership[m] >= 0) {
                printf("ERROR: Point %d in multiple clusters\n", m);
                valid = false;
            }
            membership[m] = (int)c;
        }
    int cc = 0;
    for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) cc++;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), cc, points.size() - cc);
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("  -n <num>     Points (default: 1000)\n");
    printf("  -t <float>   Threshold (default: 2.0)\n");
    printf("  -v           Validate\n");
    printf("  -r           Print results\n");
    printf("  -h           Help\n");
}

int main(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: invalid params\n"); return 1;
    }

    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    auto t0 = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    printf("Clustering time: %ld ms\n", dt_ms.count());
    printf("Clusters found: %zu\n", clusters.size());

    int total_clustered = 0, max_sz = 0;
    for (auto& c : clusters) {
        total_clustered += (int)c.members.size();
        max_sz = std::max(max_sz, (int)c.members.size());
    }
    double avg = clusters.empty() ? 0.0 : (double)total_clustered / clusters.size();
    printf("Points clustered: %d / %d (%.1f%%)\n",
           total_clustered, num_points, 100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg);
    printf("Maximum cluster size: %d\n", max_sz);

    double sec = dt_ms.count() / 1000.0;
    printf("Performance: %.1f clusters/s, %.1f points/s\n",
           clusters.size() / sec, num_points / sec);

    if (printResults) {
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c)
            for (int m : clusters[c].members) membership[m] = (int)c;
        std::vector<double> md;
        md.reserve(num_points);
        for (int m : membership) md.push_back((double)m);
        print_results(md, "ClusterMembership");
    }

    if (validate) {
        bool ok = validateClusters(clusters, points, threshold);
        cleanupGPU();
        printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        return ok ? 0 : 1;
    }

    cleanupGPU();
    return 0;
}

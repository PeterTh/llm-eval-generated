// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// This file unconditionally uses MPI, OpenMP and CUDA for maximum parallelism.

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
#if defined(HAS_CUDA) && (HAS_CUDA==1)
#include <cuda_runtime.h>
#endif

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

// Global MPI info
static int g_mpi_rank = 0;
static int g_mpi_size = 1;

// Generate synthetic 2D point data in clusters (thread-safe PRNG per thread)
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
            const double dy = std::sqrt(std::max(0.0, r * r - dx * dx)) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++; group_cnt--;
        }
    }
}

inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

#if defined(HAS_CUDA) && (HAS_CUDA==1)
// CUDA helpers: atomicMax for double using atomicCAS
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    while (true) {
        double oldd = __longlong_as_double(old);
        if (oldd >= val) break;
        unsigned long long assumed = old;
        unsigned long long newval = __double_as_longlong(val);
        unsigned long long prev = atomicCAS(address_as_ull, assumed, newval);
        if (prev == assumed) { old = prev; break; }
        old = prev;
    }
    return __longlong_as_double(old);
}

// Kernel computes distances for all candidate-member pairs and reduces max per candidate
__global__ void computeMaxDistancesKernel(const double* xs, const double* ys,
                                          const int* candidates, int numCandidates,
                                          const int* members, int numMembers,
                                          double* outMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numCandidates * numMembers;
    if (idx >= total) return;
    int ci = idx / numMembers;
    int mi = idx % numMembers;
    int cand = candidates[ci];
    int mem = members[mi];
    double dx = xs[cand] - xs[mem];
    double dy = ys[cand] - ys[mem];
    double d = sqrt(dx*dx + dy*dy);
    atomicMaxDouble(&outMax[ci], d);
}
#endif

// Compute max distances for each candidate to the set of members using CUDA; falls back to CPU if no device
void gpuComputeMaxDistances(const std::vector<Point>& points,
                            const std::vector<int>& candidates,
                            const std::vector<int>& members,
                            std::vector<double>& outMax) {
    const int numC = static_cast<int>(candidates.size());
    const int numM = static_cast<int>(members.size());
    outMax.assign(numC, 0.0);
    if (numC == 0) return;

    // Prepare host arrays
    std::vector<double> xs(points.size()), ys(points.size());
    for (size_t i = 0; i < points.size(); ++i) { xs[i] = points[i].x; ys[i] = points[i].y; }

    double *d_xs=nullptr, *d_ys=nullptr, *d_out=nullptr;
    int *d_cand=nullptr, *d_mem=nullptr;
    size_t sX = xs.size()*sizeof(double);
    size_t sC = numC*sizeof(int);
    size_t sM = numM*sizeof(int);
    size_t sOut = numC*sizeof(double);

#if defined(HAS_CUDA) && (HAS_CUDA==1)
    // Try GPU path; if any CUDA call fails, fall back to CPU path
    cudaError_t err;
    err = cudaMalloc(&d_xs, sX);
    if (err == cudaSuccess) {
        if (cudaMalloc(&d_ys, sX) == cudaSuccess && cudaMalloc(&d_cand, sC) == cudaSuccess && cudaMalloc(&d_mem, sM) == cudaSuccess && cudaMalloc(&d_out, sOut) == cudaSuccess) {
            if (cudaMemcpy(d_xs, xs.data(), sX, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(d_ys, ys.data(), sX, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(d_cand, candidates.data(), sC, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(d_mem, members.data(), sM, cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemset(d_out, 0, sOut) == cudaSuccess) {

                int total = numC * numM;
                int block = 256;
                int grid = (total + block - 1) / block;
                computeMaxDistancesKernel<<<grid, block>>>(d_xs, d_ys, d_cand, numC, d_mem, numM, d_out);
                if (cudaDeviceSynchronize() == cudaSuccess) {
                    if (cudaMemcpy(outMax.data(), d_out, sOut, cudaMemcpyDeviceToHost) == cudaSuccess) {
                        cudaFree(d_xs); cudaFree(d_ys); cudaFree(d_cand); cudaFree(d_mem); cudaFree(d_out);
                        return; // GPU path succeeded
                    }
                }
            }
        }
    }
    // If we reach here GPU path failed; free any allocated device memory
    if (d_xs) cudaFree(d_xs); if (d_ys) cudaFree(d_ys); if (d_cand) cudaFree(d_cand); if (d_mem) cudaFree(d_mem); if (d_out) cudaFree(d_out);
#endif

    // CPU fallback
    for (int ci = 0; ci < numC; ++ci) {
        double maxd = 0.0;
        int cand = candidates[ci];
        for (int mi = 0; mi < numM; ++mi) {
            int mem = members[mi];
            double dx = xs[cand] - xs[mem];
            double dy = ys[cand] - ys[mem];
            double d = sqrt(dx*dx + dy*dy);
            if (d > maxd) maxd = d;
        }
        outMax[ci] = maxd;
    }
}


// Generate a candidate cluster starting from a seed point using CUDA to accelerate distance evaluation
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<char> in_cluster(point_count, 0);
    std::vector<int> members;
    members.reserve(256);
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);

    std::vector<int> candidates; candidates.reserve(point_count);
    while (static_cast<int>(members.size()) < point_count) {
        candidates.clear();
        for (int c = 0; c < point_count; ++c) {
            if (!clustered[c] && !in_cluster[c]) candidates.push_back(c);
        }
        if (candidates.empty()) break;

        // Compute max distances candidate->members using GPU
        std::vector<double> maxd;
        gpuComputeMaxDistances(points, candidates, members, maxd);

        // Find best candidate
        int best_idx = -1; double best_val = std::numeric_limits<double>::max();
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (maxd[i] < threshold && maxd[i] < best_val) {
                best_val = maxd[i]; best_idx = static_cast<int>(candidates[i]);
            }
        }
        if (best_idx < 0) break;
        in_cluster[best_idx] = 1;
        members.push_back(best_idx);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm with MPI + OpenMP parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // Loop until no unclustered points
    while (true) {
        // Build list of seeds (unclustered points)
        std::vector<int> seeds;
        for (int i = 0; i < N; ++i) if (!clustered[i]) seeds.push_back(i);
        if (seeds.empty()) break;

        // Partition seeds among MPI ranks by index
        int local_best_card = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // Each rank processes a subset of seeds
        #pragma omp parallel
        {
            int thread_best_card = -1;
            int thread_best_seed = -1;
            std::vector<int> thread_best_members;

            #pragma omp for schedule(dynamic)
            for (int si = 0; si < (int)seeds.size(); ++si) {
                if ((si % g_mpi_size) != g_mpi_rank) continue; // distribute among ranks
                int seed = seeds[si];
                if (clustered[seed]) continue;
                std::vector<int> cand_members;
                int card = generateCandidateCluster(seed, std::vector<bool>(clustered.begin(), clustered.end()), points, threshold, N, &cand_members);
                if (card > thread_best_card) {
                    thread_best_card = card; thread_best_seed = seed; thread_best_members.swap(cand_members);
                }
            }

            #pragma omp critical
            {
                if (thread_best_card > local_best_card) {
                    local_best_card = thread_best_card;
                    local_best_seed = thread_best_seed;
                    local_best_members = std::move(thread_best_members);
                }
            }
        }

        // Gather local bests to determine global best
        std::vector<int> all_local_cards(g_mpi_size);
        MPI_Allgather(&local_best_card, 1, MPI_INT, all_local_cards.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int global_best_card = -1;
        for (int v : all_local_cards) if (v > global_best_card) global_best_card = v;
        if (global_best_card <= 0) break; // no more clusters

        int owner = -1;
        for (int r = 0; r < g_mpi_size; ++r) {
            if (all_local_cards[r] == global_best_card) { owner = r; break; }
        }

        // Owner rank will broadcast the best seed and members
        int members_size = 0;
        if (g_mpi_rank == owner) members_size = static_cast<int>(local_best_members.size());
        MPI_Bcast(&members_size, 1, MPI_INT, owner, MPI_COMM_WORLD);
        std::vector<int> members_buf(members_size);
        if (g_mpi_rank == owner) {
            for (int i = 0; i < members_size; ++i) members_buf[i] = local_best_members[i];
        }
        if (members_size > 0) MPI_Bcast(members_buf.data(), members_size, MPI_INT, owner, MPI_COMM_WORLD);

        // Add cluster (only ranks will keep full cluster list)
        Cluster cluster; cluster.seed_point = (g_mpi_rank == owner ? local_best_seed : -1);
        cluster.members = members_buf;
        clusters.push_back(cluster);

        // Mark members as clustered
        for (int m : members_buf) clustered[m] = 1;
    }

    return clusters;
}

// Validation: same as original but uses char vector
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
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold); valid = false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) { printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c); valid = false; }
            membership[member] = static_cast<int>(c);
        }
    }
    int clustered_count = 0; for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) clustered_count++;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
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
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Initialize MPI and OpenMP
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);

    // Parse command line arguments (only rank 0 prints usage)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) { num_points = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { threshold = atof(argv[++i]); }
        else if (strcmp(argv[i], "-v") == 0) { validate = true; }
        else if (strcmp(argv[i], "-r") == 0) { printResults = true; }
        else if (strcmp(argv[i], "-h") == 0) { if (g_mpi_rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (g_mpi_rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (g_mpi_rank==0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize(); return 1;
    }

    if (g_mpi_rank==0) {
        printf("QT Clustering Benchmark (hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    omp_set_num_threads(omp_get_max_threads());

    // Generate synthetic data (all ranks create identical dataset)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (g_mpi_rank==0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
        int total_clustered = 0; int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) { int s = (int)clusters[i].members.size(); total_clustered += s; max_cluster_size = std::max(max_cluster_size, s); }
        const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData; membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) for (size_t i = 0; i < clusters[c].members.size(); ++i) membership[clusters[c].members[i]] = (int)c;
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) { printf("Validation: PASSED\n"); MPI_Finalize(); return 0; }
            else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}

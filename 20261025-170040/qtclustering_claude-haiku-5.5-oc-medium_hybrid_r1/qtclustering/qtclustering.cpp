// QT Clustering Benchmark - hybrid MPI + OpenMP + CUDA implementation
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Each outer step grows a candidate cluster from every unclustered seed,
// commits the largest one (ties broken by lowest seed index), and removes its
// members from consideration. The result is identical to the original
// sequential algorithm; only the execution is parallel:
//
//  - CUDA:   one thread block grows the candidate cluster for one seed. The
//            threads cooperate on the per-step argmin and the distance updates.
//  - OpenMP: one host thread drives each GPU owned by the rank, and the
//            host-side neighbour counting and bound maintenance are parallel loops.
//  - MPI:    every rank holds the replicated clustering state. Seed evaluations
//            are split across ranks and combined with an allreduce, so all ranks
//            make identical decisions without exchanging cluster data.
//
// Two exact optimizations keep the search equivalent to the original:
//  - Incremental diameter: each candidate's maximum distance to the current
//    members is updated in O(1) per added member instead of being recomputed
//    from scratch. Candidates whose maximum reaches the threshold are dropped,
//    since that value only grows and such a candidate can never be admitted.
//  - Branch and bound over seeds: a seed's cluster can contain only unclustered
//    points within the threshold of it, so that count is an upper bound on its
//    cardinality. Seeds are evaluated in decreasing bound order and the search
//    stops once no remaining bound can reach the best cardinality found.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block of the candidate-growing kernel (power of two).
static const int BLOCK_THREADS = 256;
// Upper bound on concurrently resident candidate slots per GPU.
static const int MAX_SLOTS = 2048;
// Device memory budget per GPU for per-slot working sets.
static const size_t SCRATCH_BUDGET = size_t(1) << 30;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

static void fatalAbort(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

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

// Euclidean distance between two points. The fused multiply-add reproduces the
// contraction of dx*dx + dy*dy used by the original build. It is shared by host
// and device code so that both compute bit-identical values.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return sqrt(fma(dx, dx, dy * dy));
}

// Grow the candidate cluster for a seed on the host (equivalent to the original
// step-by-step argmin, with incremental max-distance maintenance).
// Returns the cardinality of the cluster; members are appended in addition order.
int generateCandidateCluster(const int seed_point,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             std::vector<int>* cluster_members) {
    const int N = static_cast<int>(points.size());
    std::vector<int> members;
    std::vector<double> max_dist(N, 0.0);  // max distance from candidate to members
    std::vector<int> active;               // admissible candidates, ascending index

    members.push_back(seed_point);
    for (int c = 0; c < N; ++c) {
        if (c == seed_point || clustered[c]) continue;
        const double d = distance(points[c], points[seed_point]);
        if (d < threshold) {
            max_dist[c] = d;
            active.push_back(c);
        }
    }

    while (!active.empty()) {
        // Smallest max distance; strict '<' keeps the lowest index on ties
        size_t best_k = 0;
        for (size_t k = 1; k < active.size(); ++k) {
            if (max_dist[active[k]] < max_dist[active[best_k]]) best_k = k;
        }
        const int chosen = active[best_k];
        members.push_back(chosen);

        // Update remaining candidates and drop those that can no longer qualify
        size_t out = 0;
        for (size_t k = 0; k < active.size(); ++k) {
            const int c = active[k];
            if (c == chosen) continue;
            const double d = distance(points[c], points[chosen]);
            const double m = std::max(max_dist[c], d);
            if (m < threshold) {
                max_dist[c] = m;
                active[out++] = c;
            }
        }
        active.resize(out);
    }

    if (cluster_members) {
        *cluster_members = members;
    }
    return static_cast<int>(members.size());
}

// Device kernel: one block grows the candidate cluster for each seed in
// seeds[0..nseeds). Block b handles seeds b, b+gridDim, ... and uses scratch slot b.
__global__ void candidateCardinalityKernel(const Point* __restrict__ pts,
                                           const unsigned char* __restrict__ clustered,
                                           const int* __restrict__ seeds,
                                           const int nseeds,
                                           const int N,
                                           const double threshold,
                                           int* __restrict__ cards,
                                           double* __restrict__ scratch_d,
                                           unsigned char* __restrict__ scratch_s) {
    __shared__ double s_val[BLOCK_THREADS];
    __shared__ int s_idx[BLOCK_THREADS];
    __shared__ Point s_cp;

    const int tid = threadIdx.x;
    double* max_dist = scratch_d + static_cast<size_t>(blockIdx.x) * N;
    unsigned char* state = scratch_s + static_cast<size_t>(blockIdx.x) * N;  // 0 = active

    for (int q = blockIdx.x; q < nseeds; q += gridDim.x) {
        const int seed = seeds[q];
        const Point sp = pts[seed];

        for (int i = tid; i < N; i += BLOCK_THREADS) {
            if (i == seed || clustered[i]) {
                state[i] = 1;
                continue;
            }
            const double d = distance(pts[i], sp);
            max_dist[i] = d;
            state[i] = (d < threshold) ? 0 : 1;
        }
        __syncthreads();

        int members = 1;
        while (true) {
            // Block-wide argmin of max_dist over active candidates
            double bv = 0.0;
            int bi = -1;
            for (int i = tid; i < N; i += BLOCK_THREADS) {
                if (state[i] == 0 && (bi < 0 || max_dist[i] < bv)) {
                    bv = max_dist[i];
                    bi = i;
                }
            }
            s_val[tid] = bv;
            s_idx[tid] = bi;
            __syncthreads();
            for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    const int j = tid + s;
                    const int bj = s_idx[j];
                    if (bj >= 0 && (s_idx[tid] < 0 || s_val[j] < s_val[tid] ||
                                    (s_val[j] == s_val[tid] && bj < s_idx[tid]))) {
                        s_val[tid] = s_val[j];
                        s_idx[tid] = bj;
                    }
                }
                __syncthreads();
            }
            const int chosen = s_idx[0];
            if (chosen < 0) break;  // no admissible candidate left (uniform across block)

            if (tid == 0) {
                s_cp = pts[chosen];
                state[chosen] = 1;
            }
            __syncthreads();

            // Fold the chosen member into every remaining candidate
            const Point cp = s_cp;
            for (int i = tid; i < N; i += BLOCK_THREADS) {
                if (state[i] == 0) {
                    const double d = distance(pts[i], cp);
                    const double m = (d > max_dist[i]) ? d : max_dist[i];
                    max_dist[i] = m;
                    if (m >= threshold) state[i] = 1;
                }
            }
            __syncthreads();
            ++members;
        }

        if (tid == 0) cards[q] = members;
        __syncthreads();
    }
}

// Per-GPU resources owned by this rank
struct DeviceCtx {
    int device = -1;
    Point* pts = nullptr;
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    int* cards = nullptr;
    double* scratch_d = nullptr;
    unsigned char* scratch_s = nullptr;
    int slots = 0;
};

// Allocate device resources; slot count bounded by the scratch budget
static void setupDevice(DeviceCtx& c, const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    CUDA_CHECK(cudaSetDevice(c.device));
    CUDA_CHECK(cudaMalloc(&c.pts, sizeof(Point) * N));
    CUDA_CHECK(cudaMemcpy(c.pts, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&c.clustered, N));
    CUDA_CHECK(cudaMalloc(&c.seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&c.cards, sizeof(int) * N));

    const size_t per_slot = static_cast<size_t>(N) * (sizeof(double) + sizeof(unsigned char));
    size_t slots = std::max<size_t>(1, SCRATCH_BUDGET / per_slot);
    slots = std::min<size_t>(slots, std::min<size_t>(MAX_SLOTS, static_cast<size_t>(N)));
    c.slots = static_cast<int>(slots);
    CUDA_CHECK(cudaMalloc(&c.scratch_d, sizeof(double) * slots * N));
    CUDA_CHECK(cudaMalloc(&c.scratch_s, slots * N));
}

static void releaseDevice(DeviceCtx& c) {
    CUDA_CHECK(cudaSetDevice(c.device));
    CUDA_CHECK(cudaFree(c.pts));
    CUDA_CHECK(cudaFree(c.clustered));
    CUDA_CHECK(cudaFree(c.seeds));
    CUDA_CHECK(cudaFree(c.cards));
    CUDA_CHECK(cudaFree(c.scratch_d));
    CUDA_CHECK(cudaFree(c.scratch_s));
}

// Evaluate candidate cardinalities for a batch of seeds. Entries are split across
// ranks (k % nranks) and then across this rank's GPUs. Entries owned by other
// ranks are left at zero, so the caller must sum over ranks.
static void evaluateCardinalities(std::vector<DeviceCtx>& ctxs,
                                  const std::vector<unsigned char>& clustered,
                                  const double threshold,
                                  const std::vector<int>& batch,
                                  const int rank,
                                  const int nranks,
                                  std::vector<int>& cards) {
    const int N = static_cast<int>(clustered.size());
    std::vector<size_t> mine;
    for (size_t k = rank; k < batch.size(); k += nranks) mine.push_back(k);

    const int nd = static_cast<int>(ctxs.size());
    #pragma omp parallel for num_threads(nd) schedule(static, 1)
    for (int t = 0; t < nd; ++t) {
        DeviceCtx& c = ctxs[t];
        std::vector<size_t> pos;
        std::vector<int> seeds;
        for (size_t j = t; j < mine.size(); j += nd) {
            pos.push_back(mine[j]);
            seeds.push_back(batch[mine[j]]);
        }
        if (seeds.empty()) continue;

        const int count = static_cast<int>(seeds.size());
        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaMemcpy(c.clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.seeds, seeds.data(), sizeof(int) * count, cudaMemcpyHostToDevice));

        const int grid = std::min(count, c.slots);
        candidateCardinalityKernel<<<grid, BLOCK_THREADS>>>(c.pts, c.clustered, c.seeds, count, N,
                                                            threshold, c.cards, c.scratch_d,
                                                            c.scratch_s);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), c.cards, sizeof(int) * count, cudaMemcpyDeviceToHost));
        for (int j = 0; j < count; ++j) {
            cards[pos[j]] = host[j];
        }
    }
}

// Main QT clustering algorithm. Every rank executes this identically and
// returns the same clusters.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nranks,
                                  std::vector<DeviceCtx>& ctxs) {
    const int N = static_cast<int>(points.size());
    // Neighbour-count bound uses a slightly enlarged radius. This is a superset of
    // the admissible test, so the bound stays valid under rounding differences.
    const double bound_thr = threshold * (1.0 + 1e-9);

    std::vector<unsigned char> clustered(N, 0);

    // cnt[s] = number of unclustered points j (including s) with dist(s, j) < bound_thr.
    // Each rank computes its cyclic share of rows; an allreduce merges them.
    std::vector<int> cnt(N, 0);
    #pragma omp parallel for schedule(dynamic, 16) if (static_cast<long long>(N) * N >= (1LL << 20))
    for (int s = rank; s < N; s += nranks) {
        int c = 0;
        for (int j = 0; j < N; ++j) {
            if (distance(points[s], points[j]) < bound_thr) ++c;
        }
        cnt[s] = c;
    }
    MPI_Allreduce(MPI_IN_PLACE, cnt.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    const size_t batch_min = 32 * static_cast<size_t>(nranks);
    const size_t batch_max = 4096 * static_cast<size_t>(nranks);

    // Main clustering loop
    while (!unclustered.empty()) {
        // Visit seeds by decreasing bound, then by index
        std::vector<int> order = unclustered;
        std::sort(order.begin(), order.end(), [&cnt](int a, int b) {
            return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b;
        });

        int best_card = -1;
        int best_seed = -1;
        size_t pos = 0;
        size_t batch_size = batch_min;

        while (pos < order.size()) {
            // Remaining seeds cannot reach the best cardinality
            if (best_card >= 0 && cnt[order[pos]] < best_card) break;

            size_t end = std::min(order.size(), pos + batch_size);
            if (best_card >= 0) {
                size_t e = pos;
                while (e < end && cnt[order[e]] >= best_card) ++e;
                end = e;
            }

            const std::vector<int> batch(order.begin() + pos, order.begin() + end);
            std::vector<int> cards(batch.size(), 0);
            evaluateCardinalities(ctxs, clustered, threshold, batch, rank, nranks, cards);
            MPI_Allreduce(MPI_IN_PLACE, cards.data(), static_cast<int>(cards.size()),
                          MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            // Largest cardinality wins; ties go to the lowest seed index
            for (size_t k = 0; k < batch.size(); ++k) {
                if (cards[k] > best_card ||
                    (cards[k] == best_card && batch[k] < best_seed)) {
                    best_card = cards[k];
                    best_seed = batch[k];
                }
            }

            pos = end;
            batch_size = std::min(batch_size * 2, batch_max);
        }

        if (best_seed < 0 || best_card <= 0) {
            // No more clusters can be formed
            break;
        }

        // Rebuild the winning cluster on the host (deterministic, identical on all ranks)
        std::vector<int> members;
        const int card = generateCandidateCluster(best_seed, clustered, points, threshold, &members);
        if (card != best_card) {
            fatalAbort("Internal error: GPU and host cluster cardinalities disagree");
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = members;
        clusters.push_back(cluster);

        // Mark members clustered and remove their contributions from the bounds
        for (int m : members) clustered[m] = 1;
        #pragma omp parallel for schedule(static) \
            if (static_cast<long long>(N) * members.size() >= (1LL << 18))
        for (int s = 0; s < N; ++s) {
            int dec = 0;
            for (int m : members) {
                if (distance(points[s], points[m]) < bound_thr) ++dec;
            }
            cnt[s] -= dec;
        }

        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

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

// Select the GPUs this rank drives. GPUs are shared among the ranks on a node
// in round-robin fashion; a rank with no assigned GPU shares one.
static std::vector<int> selectDevices(const int local_rank, const int local_size, const int ndev) {
    std::vector<int> devices;
    for (int d = 0; d < ndev; ++d) {
        if (d % local_size == local_rank) devices.push_back(d);
    }
    if (devices.empty()) devices.push_back(local_rank % ndev);
    return devices;
}

// Run the benchmark on all ranks. Returns the process exit status.
static int runBenchmark(const int num_points, const double threshold, const bool validate,
                        const bool printResults, const int rank, const int nranks) {
    const bool root = (rank == 0);

    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Bind this rank to its GPUs
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    MPI_Comm_free(&node_comm);

    // Share the node's cores among its ranks. Idle OpenMP threads spin between
    // parallel regions, so an oversized team starves the CUDA host threads.
    omp_set_num_threads(std::max(1, omp_get_num_procs() / local_size));

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev <= 0) {
        fatalAbort("Error: no CUDA devices available");
    }

    std::vector<DeviceCtx> ctxs;
    for (int d : selectDevices(local_rank, local_size, ndev)) {
        DeviceCtx c;
        c.device = d;
        setupDevice(c, points);
        ctxs.push_back(c);
    }

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks, ctxs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(max_cluster_time);

    for (auto& c : ctxs) releaseDevice(c);

    int valid = 1;
    if (root) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
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
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);

        // Print results for external validation
        if (printResults) {
            // Serialize cluster membership for hashing
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

        // Validation
        if (validate) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!valid) return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int status = -1;  // -1: continue running the benchmark

    // Parse command line arguments
    for (int i = 1; i < argc && status < 0; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            status = 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
        }
    }

    if (status < 0 && (num_points <= 0 || threshold <= 0.0)) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        status = 1;
    }

    if (status < 0) {
        status = runBenchmark(num_points, threshold, validate, printResults, rank, nranks);
    }

    MPI_Finalize();
    return status;
}

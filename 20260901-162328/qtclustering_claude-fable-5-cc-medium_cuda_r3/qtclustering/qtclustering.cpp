// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - Each outer round evaluates every unclustered point as a candidate seed.
//    One CUDA thread block grows one candidate cluster (grid-stride over
//    seeds), so all seeds are evaluated concurrently on the GPU.
//  - Within a block, threads cooperatively maintain a per-point array holding
//    the maximum distance from that point to the cluster grown so far. Adding
//    a member is an O(n) parallel update, and the next member is chosen with
//    a block-wide argmin reduction that reproduces the sequential
//    tie-breaking (smallest distance, then lowest point index).
//  - The host selects the seed with the largest cardinality (earliest seed
//    wins ties, as in the sequential scan) and a single-block kernel re-grows
//    that cluster to record its members.

#include <algorithm>
#include <chrono>
#include <cfloat>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

static const int BLOCK_SIZE = 256;
static const int MAX_GROW_BLOCKS = 512;

// Grow the candidate cluster for `seed` cooperatively with all threads of the
// calling block. Only points strictly closer than `threshold` to the seed can
// ever join (the max-distance to the cluster includes the seed and never
// shrinks), so those are compacted into a candidate list first. `maxdist[j]`
// tracks the maximum distance from candidate slot j to the cluster members
// added so far; slots that become unavailable (added to the cluster, or whose
// max-distance reached the threshold) are marked with DBL_MAX. If `members`
// is non-null the members are recorded in insertion order. Returns the
// cardinality of the grown cluster.
__device__ int growCluster(const double2* pts, const unsigned char* clustered,
                           const int seed, double* maxdist, int* cand,
                           const int n, const double threshold, int* members) {
    __shared__ double sdist[BLOCK_SIZE];
    __shared__ int sidx[BLOCK_SIZE];
    __shared__ int sslot[BLOCK_SIZE];
    __shared__ int num_cand;
    __shared__ int num_dead;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    if (tid == 0) {
        num_cand = 0;
        num_dead = 0;
    }
    __syncthreads();

    // Build the candidate list: unclustered points within threshold of seed.
    // The `maxdist`/`cand` buffers are 2*n large so the list can be compacted
    // by ping-ponging between the two halves.
    double* md_a = maxdist;
    double* md_b = maxdist + n;
    int* cand_a = cand;
    int* cand_b = cand + n;

    const double2 sp = pts[seed];
    for (int i = tid; i < n; i += nt) {
        if (clustered[i] || i == seed) continue;
        const double dx = pts[i].x - sp.x;
        const double dy = pts[i].y - sp.y;
        const double d = sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            const int slot = atomicAdd(&num_cand, 1);
            cand_a[slot] = i;
            md_a[slot] = d;
        }
    }
    __syncthreads();
    int m = num_cand;

    if (members && tid == 0) {
        members[0] = seed;
    }

    int count = 1;
    for (;;) {
        // Per-thread argmin over its strided slice of candidate slots; ties
        // keep the lowest point index, matching the sequential scan.
        double best_d = DBL_MAX;
        int best_i = -1;
        int best_slot = -1;
        for (int j = tid; j < m; j += nt) {
            const double d = md_a[j];
            const int i = cand_a[j];
            if (d < threshold && (d < best_d || (d == best_d && i < best_i))) {
                best_d = d;
                best_i = i;
                best_slot = j;
            }
        }
        sdist[tid] = best_d;
        sidx[tid] = best_i;
        sslot[tid] = best_slot;
        __syncthreads();

        // Block-wide argmin reduction
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const double od = sdist[tid + s];
                const int oi = sidx[tid + s];
                if (oi >= 0 && (sidx[tid] < 0 || od < sdist[tid] ||
                                (od == sdist[tid] && oi < sidx[tid]))) {
                    sdist[tid] = od;
                    sidx[tid] = oi;
                    sslot[tid] = sslot[tid + s];
                }
            }
            __syncthreads();
        }

        const int chosen = sidx[0];
        const int chosen_slot = sslot[0];
        __syncthreads();

        if (chosen < 0) break; // No more points can be added

        if (members && tid == 0) {
            members[count] = chosen;
        }
        count++;

        // Fold the new member into the per-candidate maximum distances,
        // pruning slots whose max-distance can no longer pass the threshold.
        const double2 cp = pts[chosen];
        for (int j = tid; j < m; j += nt) {
            if (j == chosen_slot) continue;
            const double d = md_a[j];
            if (d != DBL_MAX) {
                const double dx = pts[cand_a[j]].x - cp.x;
                const double dy = pts[cand_a[j]].y - cp.y;
                const double dd = sqrt(dx * dx + dy * dy);
                if (dd >= threshold) {
                    md_a[j] = DBL_MAX;
                    atomicAdd(&num_dead, 1);
                } else if (dd > d) {
                    md_a[j] = dd;
                }
            }
        }
        if (tid == 0) {
            md_a[chosen_slot] = DBL_MAX;
            atomicAdd(&num_dead, 1);
        }
        __syncthreads();

        // Compact the candidate list once at least half its slots are dead.
        // The list order is arbitrary; correctness relies only on point
        // indices, which the argmin tie-break uses.
        if (2 * num_dead >= m) {
            if (tid == 0) num_cand = 0;
            __syncthreads();
            for (int j = tid; j < m; j += nt) {
                const double d = md_a[j];
                if (d != DBL_MAX) {
                    const int slot = atomicAdd(&num_cand, 1);
                    md_b[slot] = d;
                    cand_b[slot] = cand_a[j];
                }
            }
            __syncthreads();
            m = num_cand;
            double* mdt = md_a; md_a = md_b; md_b = mdt;
            int* ct = cand_a; cand_a = cand_b; cand_b = ct;
            if (tid == 0) num_dead = 0;
            __syncthreads();
        }
    }

    return count;
}

// One block per candidate seed (grid-stride): compute the cardinality of the
// cluster grown from each unclustered seed.
__global__ void candidateCardinalityKernel(const double2* pts,
                                           const unsigned char* clustered,
                                           const int* seeds, const int num_seeds,
                                           double* workspace, int* cand_workspace,
                                           const int n, const double threshold,
                                           int* cardinalities) {
    double* maxdist = workspace + static_cast<size_t>(blockIdx.x) * 2 * n;
    int* cand = cand_workspace + static_cast<size_t>(blockIdx.x) * 2 * n;
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        const int card = growCluster(pts, clustered, seeds[s], maxdist, cand,
                                     n, threshold, nullptr);
        if (threadIdx.x == 0) {
            cardinalities[s] = card;
        }
        __syncthreads();
    }
}

// Mark unclustered seeds whose candidate cluster may have changed after the
// points in `members` were removed. A grown cluster only depends on the
// availability of points strictly within `threshold` of its seed, so a seed
// stays clean unless a removed point lies inside that radius.
__global__ void markDirtySeedsKernel(const double2* pts, const int* seeds,
                                     const int num_seeds, const int* members,
                                     const int num_members,
                                     const double threshold,
                                     unsigned char* dirty) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;
    const double2 sp = pts[seeds[s]];
    unsigned char is_dirty = 0;
    for (int i = 0; i < num_members; ++i) {
        const double2 mp = pts[members[i]];
        const double dx = mp.x - sp.x;
        const double dy = mp.y - sp.y;
        if (sqrt(dx * dx + dy * dy) < threshold) {
            is_dirty = 1;
            break;
        }
    }
    dirty[s] = is_dirty;
}

// Single-block kernel: re-grow the winning cluster, recording its members and
// marking them as clustered on the device.
__global__ void growBestClusterKernel(const double2* pts,
                                      unsigned char* clustered, const int seed,
                                      double* workspace, int* cand_workspace,
                                      const int n, const double threshold,
                                      int* members, int* cardinality) {
    const int card = growCluster(pts, clustered, seed, workspace,
                                 cand_workspace, n, threshold, members);
    if (threadIdx.x == 0) {
        *cardinality = card;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < card; i += blockDim.x) {
        clustered[members[i]] = 1;
    }
}

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    if (ngpus < 1) {
        fprintf(stderr, "No CUDA devices available\n");
        exit(1);
    }

    const int grow_blocks = std::min(N, MAX_GROW_BLOCKS);

    // Per-GPU device buffers: points and clustered flags are replicated so
    // seed evaluation can be split across all GPUs.
    std::vector<double2*> d_points(ngpus);
    std::vector<unsigned char*> d_clustered(ngpus);
    std::vector<int*> d_seeds(ngpus);
    std::vector<int*> d_cardinalities(ngpus);
    std::vector<double*> d_workspace(ngpus);
    std::vector<int*> d_cand_workspace(ngpus);
    // GPU 0 additionally hosts the winning-cluster growth and dirty marking
    int* d_members;
    int* d_best_card;
    unsigned char* d_dirty;

    std::vector<double2> h_points(N);
    for (int i = 0; i < N; ++i) {
        h_points[i] = {points[i].x, points[i].y};
    }

    for (int g = 0; g < ngpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&d_points[g], N * sizeof(double2)));
        CUDA_CHECK(cudaMalloc(&d_clustered[g], N * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_seeds[g], N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cardinalities[g], N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace[g],
                              static_cast<size_t>(grow_blocks) * 2 * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cand_workspace[g],
                              static_cast<size_t>(grow_blocks) * 2 * N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_points[g], h_points.data(), N * sizeof(double2),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_clustered[g], 0, N * sizeof(unsigned char)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_card, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dirty, N * sizeof(unsigned char)));

    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<int> h_cardinalities(N);
    std::vector<int> h_members(N);
    std::vector<bool> clustered(N, false);
    std::vector<unsigned char> h_clustered(N, 0);

    // Cached cardinality of the cluster grown from each point as a seed. A
    // cached value only becomes stale when a point within `threshold` of the
    // seed gets clustered, so each round recomputes only those "dirty" seeds.
    std::vector<int> card_cache(N, 0);
    std::vector<unsigned char> h_dirty(N);
    std::vector<int> dirty_seeds;
    dirty_seeds.reserve(N);
    bool first_round = true;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Determine which seeds need (re-)evaluation
        dirty_seeds.clear();
        if (first_round) {
            dirty_seeds = unclustered_indices;
            first_round = false;
        } else {
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(d_seeds[0], unclustered_indices.data(),
                                  num_seeds * sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int prev_card =
                static_cast<int>(clusters.back().members.size());
            const int dirty_blocks = (num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE;
            markDirtySeedsKernel<<<dirty_blocks, BLOCK_SIZE>>>(
                d_points[0], d_seeds[0], num_seeds, d_members, prev_card,
                threshold, d_dirty);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_dirty.data(), d_dirty,
                                  num_seeds * sizeof(unsigned char),
                                  cudaMemcpyDeviceToHost));
            for (int s = 0; s < num_seeds; ++s) {
                if (h_dirty[s]) dirty_seeds.push_back(unclustered_indices[s]);
            }
        }

        // Evaluate the affected seeds in parallel, one block per seed, split
        // across all GPUs.
        const int num_dirty = static_cast<int>(dirty_seeds.size());
        if (num_dirty > 0) {
            const int chunk = (num_dirty + ngpus - 1) / ngpus;
            for (int g = 0; g < ngpus; ++g) {
                const int begin = std::min(g * chunk, num_dirty);
                const int cnt = std::min(chunk, num_dirty - begin);
                if (cnt <= 0) continue;
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyAsync(d_seeds[g],
                                           dirty_seeds.data() + begin,
                                           cnt * sizeof(int),
                                           cudaMemcpyHostToDevice));
                const int blocks = std::min(cnt, grow_blocks);
                candidateCardinalityKernel<<<blocks, BLOCK_SIZE>>>(
                    d_points[g], d_clustered[g], d_seeds[g], cnt,
                    d_workspace[g], d_cand_workspace[g], N, threshold,
                    d_cardinalities[g]);
                CUDA_CHECK(cudaGetLastError());
            }
            for (int g = 0; g < ngpus; ++g) {
                const int begin = std::min(g * chunk, num_dirty);
                const int cnt = std::min(chunk, num_dirty - begin);
                if (cnt <= 0) continue;
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpy(h_cardinalities.data() + begin,
                                      d_cardinalities[g], cnt * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
            for (int s = 0; s < num_dirty; ++s) {
                card_cache[dirty_seeds[s]] = h_cardinalities[s];
            }
        }

        // Pick the seed with maximum cardinality; the earliest seed in the
        // unclustered order wins ties, as in the sequential version.
        int max_cardinality = -1;
        int best_seed = -1;
        for (int s = 0; s < num_seeds; ++s) {
            const int seed = unclustered_indices[s];
            if (card_cache[seed] > max_cardinality) {
                max_cardinality = card_cache[seed];
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        // Re-grow the winning cluster to obtain its members and mark them
        // clustered on device 0.
        CUDA_CHECK(cudaSetDevice(0));
        growBestClusterKernel<<<1, BLOCK_SIZE>>>(d_points[0], d_clustered[0],
                                                 best_seed, d_workspace[0],
                                                 d_cand_workspace[0], N,
                                                 threshold, d_members,
                                                 d_best_card);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_members.data(), d_members,
                              max_cardinality * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(h_members.begin(),
                               h_members.begin() + max_cardinality);
        clusters.push_back(cluster);

        // Mark all members as clustered on the host and the remaining GPUs
        for (int i = 0; i < max_cardinality; ++i) {
            clustered[cluster.members[i]] = true;
            h_clustered[cluster.members[i]] = 1;
        }
        for (int g = 1; g < ngpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(d_clustered[g], h_clustered.data(),
                                  N * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaSetDevice(0));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    for (int g = 0; g < ngpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(d_points[g]));
        CUDA_CHECK(cudaFree(d_clustered[g]));
        CUDA_CHECK(cudaFree(d_seeds[g]));
        CUDA_CHECK(cudaFree(d_cardinalities[g]));
        CUDA_CHECK(cudaFree(d_workspace[g]));
        CUDA_CHECK(cudaFree(d_cand_workspace[g]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_best_card));
    CUDA_CHECK(cudaFree(d_dirty));

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
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
               num_points, threshold);
        return 1;
    }

    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Initialize the CUDA contexts before timing
    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    for (int g = 0; g < ngpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(nullptr));
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

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
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

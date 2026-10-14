// QT Clustering Benchmark - CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The candidate-cluster construction for every unclustered seed runs on the
// GPU, one thread block per seed. Within a block, threads cooperate on the
// candidate scan and an argmin reduction.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block used for one seed's cluster construction
static constexpr int BLOCK_SIZE = 256;
static constexpr int NUM_WARPS = BLOCK_SIZE / 32;

// Upper bound on device scratch (candidate lists per seed) per launch batch
static constexpr size_t WORKSPACE_BYTES = size_t{256} << 20;

// Upper bound on grid cells along each axis, so the grid stays small for tiny thresholds
static constexpr double MAX_GRID_DIM = 1024.0;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
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

// Device-side counterpart of distance(), using the same expression
__device__ inline double deviceDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Returns true if (v, i) is a better argmin key than (bv, bi): smaller value,
// or equal value with a lower index (matches the sequential first-wins scan)
__device__ inline bool isBetterKey(double v, int i, double bv, int bi) {
    return v < bv || (v == bv && i < bi);
}

// Block-wide argmin over per-thread (value, index) pairs.
// Every thread receives the winning index. Must be called by all threads.
__device__ int blockArgmin(double v, int i, double* s_val, int* s_idx) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Reduce within each warp
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (isBetterKey(ov, oi, v, i)) { v = ov; i = oi; }
    }
    if (lane == 0) {
        s_val[warp] = v;
        s_idx[warp] = i;
    }
    __syncthreads();

    // Reduce the per-warp results in the first warp
    if (warp == 0) {
        v = (lane < NUM_WARPS) ? s_val[lane] : static_cast<double>(INFINITY);
        i = (lane < NUM_WARPS) ? s_idx[lane] : INT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, v, off);
            const int oi = __shfl_down_sync(0xffffffffu, i, off);
            if (isBetterKey(ov, oi, v, i)) { v = ov; i = oi; }
        }
        if (lane == 0) {
            s_val[NUM_WARPS] = v;
            s_idx[NUM_WARPS] = i;
        }
    }
    __syncthreads();
    return s_idx[NUM_WARPS];
}

// Uniform grid over the points. Cells are at least as wide as the threshold, so
// any point within the threshold of another lies in the same or an adjacent cell.
struct GridView {
    int gx, gy;
    int* cell_of;     // Per point: cell id = cy * gx + cx
    int* cell_start;  // CSR offsets into cell_points, size gx * gy + 1
    int* cell_points; // Point indices grouped by cell
};

// One block grows the candidate cluster of seeds[blockIdx.x].
//
// A candidate can only ever join while its max distance to the cluster members
// is below the threshold, and that max distance never decreases as members are
// added. So the candidate list is seeded with the unclustered points within the
// threshold of the seed, and entries are retired once they fail the test. The
// max distance is maintained incrementally, which gives exactly the value a
// scan over all members would produce.
//
// If members_out is non-null (single-block launch), the cluster members are
// written there in insertion order.
__global__ void growClustersKernel(const Point* __restrict__ points,
                                   const unsigned char* __restrict__ clustered,
                                   const int N,
                                   const double threshold,
                                   const GridView grid,
                                   const int* __restrict__ seeds,
                                   double* __restrict__ ws_val,
                                   int* __restrict__ ws_idx,
                                   int* __restrict__ cardinalities,
                                   int* __restrict__ members_out) {
    __shared__ double s_val[NUM_WARPS + 1];
    __shared__ int s_idx[NUM_WARPS + 1];
    __shared__ int s_count;

    const int seed = seeds[blockIdx.x];
    const Point seedPt = points[seed];
    double* candVal = ws_val + static_cast<size_t>(blockIdx.x) * N; // Max distance to cluster
    int* candIdx = ws_idx + static_cast<size_t>(blockIdx.x) * N;    // Point index, -1 once retired

    // Gather candidates: unclustered points within the threshold of the seed.
    // Such points can only lie in the seed's cell or one of its eight neighbours.
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    const int seed_cell = grid.cell_of[seed];
    const int sx = seed_cell % grid.gx;
    const int sy = seed_cell / grid.gx;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int cx = sx + dx;
            const int cy = sy + dy;
            if (cx < 0 || cx >= grid.gx || cy < 0 || cy >= grid.gy) continue;
            const int cell = cy * grid.gx + cx;
            const int end = grid.cell_start[cell + 1];
            for (int k = grid.cell_start[cell] + threadIdx.x; k < end; k += BLOCK_SIZE) {
                const int c = grid.cell_points[k];
                if (c == seed || clustered[c]) continue;
                const double d = deviceDistance(points[c], seedPt);
                if (d < threshold) {
                    const int slot = atomicAdd(&s_count, 1);
                    candVal[slot] = d;
                    candIdx[slot] = c;
                }
            }
        }
    }
    __syncthreads();
    const int count = s_count;

    int last = seed;
    int card = 1;
    if (members_out && threadIdx.x == 0) members_out[0] = seed;

    while (true) {
        // Fold the most recently added member into each live candidate's max distance,
        // retire candidates that can no longer pass, and find the best remaining one
        const Point lastPt = points[last];
        double bestVal = static_cast<double>(INFINITY);
        int bestIdx = INT_MAX;
        for (int k = threadIdx.x; k < count; k += BLOCK_SIZE) {
            const int c = candIdx[k];
            if (c < 0) continue;
            if (c == last) {
                candIdx[k] = -1; // Newly added member, now retired
                continue;
            }
            double m = candVal[k];
            const double d = deviceDistance(points[c], lastPt);
            if (d > m) m = d;
            candVal[k] = m;
            if (m < threshold) {
                if (isBetterKey(m, c, bestVal, bestIdx)) {
                    bestVal = m;
                    bestIdx = c;
                }
            } else {
                candIdx[k] = -1; // Max distance only grows, so it can never pass again
            }
        }

        const int sel = blockArgmin(bestVal, bestIdx, s_val, s_idx);
        if (sel == INT_MAX) break; // No point can be added while keeping diameter < threshold

        // Record the selected point; it is retired from the candidates on the next scan
        if (threadIdx.x == 0 && members_out) members_out[card] = sel;
        last = sel;
        ++card;
    }

    if (threadIdx.x == 0) cardinalities[blockIdx.x] = card;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Bucket points into a uniform grid for neighbour lookup. Cells are slightly wider
    // than the threshold (the margin absorbs rounding) and never narrower than
    // span / MAX_GRID_DIM. A NaN or non-positive cell width falls back to the minimum.
    double min_x = points[0].x, max_x = points[0].x;
    double min_y = points[0].y, max_y = points[0].y;
    for (const Point& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    const double span = std::max(max_x - min_x, max_y - min_y);
    double cell = threshold * (1.0 + 1e-6);
    if (!(cell >= span / MAX_GRID_DIM)) cell = span / MAX_GRID_DIM;
    if (!(cell > 0.0)) cell = 1.0;
    const int gx = static_cast<int>((max_x - min_x) / cell) + 1;
    const int gy = static_cast<int>((max_y - min_y) / cell) + 1;
    const int num_cells = gx * gy;
    auto axisCell = [cell](double v, double lo, int g) {
        return std::min(static_cast<int>((v - lo) / cell), g - 1);
    };

    std::vector<int> cell_of(N);
    std::vector<int> cell_start(num_cells + 1, 0);
    std::vector<int> cell_points(N);
    for (int i = 0; i < N; ++i) {
        cell_of[i] = axisCell(points[i].y, min_y, gy) * gx + axisCell(points[i].x, min_x, gx);
        ++cell_start[cell_of[i] + 1];
    }
    std::partial_sum(cell_start.begin(), cell_start.end(), cell_start.begin());
    std::vector<int> write_pos(cell_start.begin(), cell_start.end() - 1);
    for (int i = 0; i < N; ++i) {
        cell_points[write_pos[cell_of[i]]++] = i;
    }

    // Device buffers. Per-seed candidate scratch is processed in batches so that
    // memory stays bounded for large N.
    const size_t bytesPerSeed = static_cast<size_t>(N) * (sizeof(double) + sizeof(int));
    const int maxBatch = std::max(1, std::min(N, static_cast<int>(WORKSPACE_BYTES / bytesPerSeed)));

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    double* d_ws_val = nullptr;
    int* d_ws_idx = nullptr;
    int* d_cell_of = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_points = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ws_val, static_cast<size_t>(maxBatch) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ws_idx, static_cast<size_t>(maxBatch) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_of, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, (num_cells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_points, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_of, cell_of.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_start, cell_start.data(), (num_cells + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell_points, cell_points.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    const GridView grid{gx, gy, d_cell_of, d_cell_start, d_cell_points};

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<int> cardinalities(N);
    std::vector<int> best_members(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int S = static_cast<int>(unclustered_indices.size());
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), S * sizeof(int), cudaMemcpyHostToDevice));

        // Build the candidate cluster for every unclustered seed in parallel
        for (int off = 0; off < S; off += maxBatch) {
            const int batch = std::min(maxBatch, S - off);
            growClustersKernel<<<batch, BLOCK_SIZE>>>(d_points, d_clustered, N, threshold, grid,
                                                      d_seeds + off, d_ws_val, d_ws_idx, d_card + off, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_card, S * sizeof(int), cudaMemcpyDeviceToHost));

        // Pick the seed with the largest cluster; the first such seed wins ties
        int best = 0;
        for (int i = 1; i < S; ++i) {
            if (cardinalities[i] > cardinalities[best]) best = i;
        }
        const int best_seed = unclustered_indices[best];

        // Rebuild the winning cluster's member list
        CUDA_CHECK(cudaMemcpy(d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
        growClustersKernel<<<1, BLOCK_SIZE>>>(d_points, d_clustered, N, threshold, grid,
                                              d_seeds, d_ws_val, d_ws_idx, d_card, d_members);
        CUDA_CHECK(cudaGetLastError());
        const int best_card = cardinalities[best];
        CUDA_CHECK(cudaMemcpy(best_members.data(), d_members, best_card * sizeof(int), cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(best_members.begin(), best_members.begin() + best_card);
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (int member : cluster.members) {
            clustered[member] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_ws_val));
    CUDA_CHECK(cudaFree(d_ws_idx));
    CUDA_CHECK(cudaFree(d_cell_of));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_points));

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

    // Create the CUDA context up front so the one-time driver setup is not timed as clustering
    CUDA_CHECK(cudaFree(nullptr));

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

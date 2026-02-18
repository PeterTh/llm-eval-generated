// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering parallelized with CUDA.
// Precomputes N×N distance matrix on GPU, then uses block-parallel
// candidate cluster building (one thread block per seed, threads
// cooperate via shared-memory reduction to find closest point).

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

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
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

// --------------- CUDA Kernels ---------------

static constexpr int BLOCK_SIZE = 256;

// Compute pairwise distance matrix
__global__ void distanceMatrixKernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    double* __restrict__ dist,
    int N)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * N;
    if (idx >= total) return;
    int i = (int)(idx / N);
    int j = (int)(idx % N);
    double dx = px[i] - px[j];
    double dy = py[i] - py[j];
    dist[idx] = sqrt(dx * dx + dy * dy);
}

// Build candidate clusters — one block per seed, threads cooperate on findClosestPoint
__global__ void buildClustersKernel(
    const double* __restrict__ dist_matrix,
    const int* __restrict__ clustered,
    const int* __restrict__ seed_indices,
    int num_seeds,
    int N,
    double threshold,
    int* __restrict__ cardinalities,
    int* __restrict__ all_members,
    unsigned char* __restrict__ all_in_cluster)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int tid = threadIdx.x;
    int seed = seed_indices[seed_idx];
    int* my_members = all_members + (long long)seed_idx * N;
    unsigned char* my_flags = all_in_cluster + (long long)seed_idx * N;

    // Cooperatively zero the flags
    for (int i = tid; i < N; i += BLOCK_SIZE)
        my_flags[i] = 0;
    __syncthreads();

    __shared__ int s_member_count;
    __shared__ int s_closest;
    __shared__ double s_diameters[BLOCK_SIZE];
    __shared__ int s_candidates[BLOCK_SIZE];

    if (tid == 0) {
        my_flags[seed] = 1;
        my_members[0] = seed;
        s_member_count = 1;
    }
    __syncthreads();

    while (s_member_count < N) {
        // Each thread evaluates a stripe of candidates
        double my_min_diam = 1e308;
        int my_closest = -1;

        for (int c = tid; c < N; c += BLOCK_SIZE) {
            if (clustered[c] || my_flags[c]) continue;

            double max_d = 0.0;
            int mc = s_member_count;
            for (int m = 0; m < mc; m++) {
                double d = dist_matrix[(long long)c * N + my_members[m]];
                if (d > max_d) max_d = d;
            }

            if (max_d < threshold) {
                if (max_d < my_min_diam || (max_d == my_min_diam && (my_closest < 0 || c < my_closest))) {
                    my_min_diam = max_d;
                    my_closest = c;
                }
            }
        }

        // Store in shared memory for reduction
        s_diameters[tid] = my_min_diam;
        s_candidates[tid] = my_closest;
        __syncthreads();

        // Tree reduction — pick smallest diameter, tie-break by smallest index
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                int rc = s_candidates[tid + stride];
                if (rc >= 0) {
                    int lc = s_candidates[tid];
                    double rd = s_diameters[tid + stride];
                    double ld = s_diameters[tid];
                    if (lc < 0 || rd < ld || (rd == ld && rc < lc)) {
                        s_diameters[tid] = rd;
                        s_candidates[tid] = rc;
                    }
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_closest = s_candidates[0];
        }
        __syncthreads();

        if (s_closest < 0) break;

        if (tid == 0) {
            my_flags[s_closest] = 1;
            my_members[s_member_count] = s_closest;
            s_member_count++;
        }
        __syncthreads();
    }

    if (tid == 0)
        cardinalities[seed_idx] = s_member_count;
}

// --------------- Host QT Clustering ---------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> h_clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; i++)
        unclustered_indices.push_back(i);

    // SoA layout
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; i++) { px[i] = points[i].x; py[i] = points[i].y; }

    long long NN = (long long)N * N;

    // GPU allocations
    double *d_px, *d_py, *d_dist;
    int *d_clustered, *d_seeds, *d_card, *d_members;
    unsigned char *d_flags;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, NN * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, NN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, NN * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Precompute distance matrix
    {
        int bs = 256;
        int gs = (int)((NN + bs - 1) / bs);
        distanceMatrixKernel<<<gs, bs>>>(d_px, d_py, d_dist, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));

    std::vector<int> h_card(N);

    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));

        // One block per seed
        buildClustersKernel<<<num_seeds, BLOCK_SIZE>>>(
            d_dist, d_clustered, d_seeds, num_seeds, N, threshold,
            d_card, d_members, d_flags);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_card.data(), d_card, num_seeds * sizeof(int), cudaMemcpyDeviceToHost));

        // Find first seed with maximum cardinality (matches sequential tie-breaking)
        int max_card = -1, best_idx = -1;
        for (int i = 0; i < num_seeds; i++) {
            if (h_card[i] > max_card) {
                max_card = h_card[i];
                best_idx = i;
            }
        }

        if (best_idx < 0 || max_card <= 0) break;

        std::vector<int> best_members(max_card);
        CUDA_CHECK(cudaMemcpy(best_members.data(),
                              d_members + (long long)best_idx * N,
                              max_card * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_idx];
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members)
            h_clustered[m] = 1;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&h_clustered](int idx) { return h_clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_flags));

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

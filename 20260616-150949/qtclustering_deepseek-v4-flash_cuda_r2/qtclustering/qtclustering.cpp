// QT Clustering Benchmark - CUDA Parallelized Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ======================= CUDA Kernels =======================

// Kernel: compute full N x N matrix of squared pairwise distances.
// 2D grid: each thread computes one (i, j) element.
__global__ void computeDistMatrixKernel(
    const double* points_x, const double* points_y,
    double* dist_matrix, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    double dx = points_x[i] - points_x[j];
    double dy = points_y[i] - points_y[j];
    dist_matrix[i * N + j] = dx * dx + dy * dy;
}

// Batch kernel: each block evaluates ONE seed completely.
// Uses 256 threads with a stride pattern to support any N.
// Coalesced __ldg loads from the symmetric distance matrix.
__launch_bounds__(256)
__global__ void evaluateSeedBatchKernel(
    const double* dist_matrix, int N,
    const char* __restrict__ clustered,
    const int* __restrict__ seeds,
    int num_seeds,
    double threshold_sq,
    int* out_cardinalities,
    int* out_members)
{
    int b = blockIdx.x;
    if (b >= num_seeds) return;
    int seed_point = seeds[b];

    // Shared memory layout (B = 256):
    // [0, N)             – in_cluster flags (char)
    // [N, N+4*N)         – members array (int)
    // [N+4*N, N+4*N+8*B) – reduction distances (double)
    // [N+4*N+8*B, N+4*N+8*B+4*B) – reduction indices (int)
    // [+4]               – cluster_size (int)
    enum { B = 256 };
    extern __shared__ char smem[];

    char*   in_cluster = smem;
    int*    members    = (int*)(smem + N);
    double* sh_dist    = (double*)(smem + N + N * sizeof(int));
    int*    sh_idx     = (int*)(smem + N + N * sizeof(int) + B * sizeof(double));
    int*    p_cs       = (int*)(smem + N + N * sizeof(int) + B * sizeof(double) + B * sizeof(int));

    int tid = threadIdx.x;

    // Zero in_cluster
    for (int i = tid; i < N; i += blockDim.x) in_cluster[i] = 0;
    __syncthreads();

    if (tid == 0) {
        in_cluster[seed_point] = 1;
        members[0] = seed_point;
        *p_cs = 1;
    }
    __syncthreads();

    // --- Iterative cluster growth ---
    while (*p_cs < N) {
        int cs = *p_cs;

        // Each thread evaluates a stride of candidates.
        // Coalesced __ldg: read column c of each member's row.
        double best_d = 1e30;
        int    best_c = -1;

        for (int c = tid; c < N; c += blockDim.x) {
            if (clustered[c] || in_cluster[c]) continue;
            double msq = 0.0;
            for (int i = 0; i < cs; ++i) {
                double d = __ldg(dist_matrix + (size_t)members[i] * N + c);
                if (d > msq) msq = d;
            }
            if (msq < threshold_sq && msq < best_d) {
                best_d = msq;
                best_c = c;
            }
        }

        // Store per-thread best into shared-memory reduction buffer
        sh_dist[tid] = best_d;
        sh_idx[tid]  = best_c;
        __syncthreads();

        // Tree reduction (8 steps for B=256)
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (sh_dist[tid + s] < sh_dist[tid]) {
                    sh_dist[tid] = sh_dist[tid + s];
                    sh_idx[tid]  = sh_idx[tid + s];
                }
            }
            __syncthreads();
        }

        int best_id = sh_idx[0];
        __syncthreads();

        if (best_id < 0) break;

        if (tid == 0) {
            in_cluster[best_id] = 1;
            members[*p_cs] = best_id;
            ++(*p_cs);
        }
        __syncthreads();
    }

    // --- Write outputs ---
    if (tid == 0) {
        int cs = *p_cs;
        out_cardinalities[b] = cs;
        for (int i = 0; i < cs; ++i)
            out_members[b * N + i] = members[i];
    }
}

// ===================== CUDA Context =====================

// Manages persistent device memory for CUDA-accelerated operations.
struct CUDAContext {
    double *d_points_x, *d_points_y;
    double *d_dist_matrix;  // N x N matrix of squared pairwise distances
    char *d_clustered;
    int capacity;

    CUDAContext() : d_points_x(nullptr), d_points_y(nullptr),
        d_dist_matrix(nullptr), d_clustered(nullptr), capacity(0) {}

    void init(int N) {
        capacity = N;
        auto chk = [](cudaError_t e) {
            if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); }
        };
        chk(cudaMalloc(&d_points_x,    N * sizeof(double)));
        chk(cudaMalloc(&d_points_y,    N * sizeof(double)));
        chk(cudaMalloc(&d_dist_matrix, (size_t)N * N * sizeof(double)));
        chk(cudaMalloc(&d_clustered,   N * sizeof(char)));
    }

    void destroy() {
        cudaFree(d_points_x); cudaFree(d_points_y);
        cudaFree(d_dist_matrix);
        cudaFree(d_clustered);
    }

    // Compute the N x N pairwise distance matrix on the GPU.
    void computeDistanceMatrix() {
        const int b = 16;
        dim3 block(b, b);
        dim3 grid((capacity + b - 1) / b, (capacity + b - 1) / b);
        computeDistMatrixKernel<<<grid, block>>>(d_points_x, d_points_y, d_dist_matrix, capacity);
        cudaDeviceSynchronize();
    }

    void uploadPoints(const std::vector<Point>& pts) {
        std::vector<double> hx(capacity), hy(capacity);
        for (int i = 0; i < capacity; ++i) { hx[i] = pts[i].x; hy[i] = pts[i].y; }
        cudaMemcpy(d_points_x, hx.data(), capacity * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_points_y, hy.data(), capacity * sizeof(double), cudaMemcpyHostToDevice);
    }

    void uploadClustered(const std::vector<bool>& cl) {
        std::vector<char> h(capacity);
        for (int i = 0; i < capacity; ++i) h[i] = cl[i] ? 1 : 0;
        cudaMemcpy(d_clustered, h.data(), capacity * sizeof(char), cudaMemcpyHostToDevice);
    }
};

// File-scope CUDA context (initialised once in qtClustering)
static CUDAContext g_cuda;

// ==========================================================

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
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

// Main QT clustering algorithm (CUDA-accelerated with batch kernel)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                   const double threshold) {
    const int N = static_cast<int>(points.size());
    
    // Initialize CUDA context, upload points, and precompute distance matrix
    g_cuda.init(N);
    g_cuda.uploadPoints(points);
    g_cuda.computeDistanceMatrix();
    
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);
    
    // Device memory for batched seed evaluation
    int *d_seeds = nullptr, *d_card = nullptr, *d_members = nullptr;
    cudaMalloc(&d_seeds,   N * sizeof(int));
    cudaMalloc(&d_card,    N * sizeof(int));
    cudaMalloc(&d_members, (size_t)N * N * sizeof(int));
    
    // Host buffers for copying results back
    std::vector<int> h_card(N);
    std::vector<int> h_members((size_t)N * N);
    
    const double threshold_sq = threshold * threshold;
    // Use 256 threads; each thread handles stride candidates.
    const int block_dim = 256;
    // Shared mem = in_cluster(N) + members(N*4) + sh_dist(B*8) + sh_idx(B*4) + cs(4)
    const size_t shmem_sz = N + 4*N + 8*block_dim + 4*block_dim + 4;
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Upload clustered state to device
        g_cuda.uploadClustered(clustered);
        
        // Collect seeds to evaluate (all unclustered points)
        std::vector<int> seeds;
        for (int idx : unclustered_indices) {
            if (!clustered[idx]) seeds.push_back(idx);
        }
        int num_seeds = static_cast<int>(seeds.size());
        if (num_seeds == 0) break;
        
        // Upload seeds to device
        cudaMemcpy(d_seeds, seeds.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch batch kernel: each block evaluates one seed
        // RTX 3090 supports up to (2^31 - 1) blocks per grid dimension
        int best_cluster_size = 0;
        int best_seed = -1;
        std::vector<int> best_members;
        
        for (int batch_start = 0; batch_start < num_seeds; batch_start += 256) {
            int batch_count = std::min(256, num_seeds - batch_start);
            
            evaluateSeedBatchKernel<<<batch_count, block_dim, shmem_sz>>>(
                g_cuda.d_dist_matrix, N,
                g_cuda.d_clustered,
                d_seeds + batch_start, batch_count,
                threshold_sq,
                d_card, d_members
            );
            cudaDeviceSynchronize();
            
            // Copy results for this batch
            cudaMemcpy(h_card.data(), d_card, batch_count * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_members.data(), d_members, (size_t)batch_count * N * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Find best seed in this batch
            for (int i = 0; i < batch_count; ++i) {
                int sz = h_card[i];
                if (sz > best_cluster_size) {
                    best_cluster_size = sz;
                    best_seed = seeds[batch_start + i];
                    best_members.assign(h_members.begin() + i * N,
                                        h_members.begin() + i * N + sz);
                }
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && best_cluster_size > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_members;
            clusters.push_back(cluster);
            
            for (int m : best_members) clustered[m] = true;
            
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    cudaFree(d_seeds); cudaFree(d_card); cudaFree(d_members);
    g_cuda.destroy();
    
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

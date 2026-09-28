// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy: for every round of the algorithm, every
// still-unclustered point is tried as a cluster seed *simultaneously*.
// Each seed's candidate cluster is grown by one CUDA thread block: the
// threads of the block cooperatively scan every remaining point in
// parallel to find the next point to add (the point whose maximum
// distance to the current cluster members is smallest while still below
// the threshold), using a block-wide reduction. Once every block has
// finished growing its candidate cluster, the host picks the seed that
// produced the largest cluster (ties broken by lowest index, exactly as
// the sequential algorithm does) and commits it, then the next round is
// launched for the remaining unclustered points.

#include <algorithm>
#include <cfloat>
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                              \
            exit(1);                                                        \
        }                                                                    \
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

// CUDA kernel: one block per candidate seed. Grows the candidate cluster for
// that seed by repeatedly finding, in parallel across all threads, the
// unclustered point whose maximum distance to the current cluster members is
// smallest while remaining below the threshold, exactly mirroring the
// sequential findClosestPoint()/generateCandidateCluster() logic (including
// its "first minimum wins ties" semantics).
__global__ void growClustersKernel(const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const int* __restrict__ seed_list,
                                    int num_seeds,
                                    const unsigned char* __restrict__ clustered,
                                    double threshold,
                                    int N,
                                    int* __restrict__ out_cardinality,
                                    int* __restrict__ out_members,
                                    unsigned char* __restrict__ in_cluster_scratch) {
    const int b = blockIdx.x;
    if (b >= num_seeds) return;

    unsigned char* in_cluster = in_cluster_scratch + static_cast<size_t>(b) * N;
    int* members = out_members + static_cast<size_t>(b) * N;

    extern __shared__ unsigned char smem[];
    double* sdist = reinterpret_cast<double*>(smem);
    int* sidx = reinterpret_cast<int*>(sdist + blockDim.x);

    __shared__ int member_count;
    __shared__ int best_idx;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        in_cluster[i] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int seed = seed_list[b];
        members[0] = seed;
        in_cluster[seed] = 1;
        member_count = 1;
    }
    __syncthreads();

    while (member_count < N) {
        double local_best_val = DBL_MAX;
        int local_best_idx = -1;
        const int mc = member_count;

        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            if (clustered[c] || in_cluster[c]) continue;

            const double cx = px[c];
            const double cy = py[c];
            double max_dist = 0.0;
            for (int m = 0; m < mc; ++m) {
                const int mi = members[m];
                const double dx = cx - px[mi];
                const double dy = cy - py[mi];
                const double d = sqrt(dx * dx + dy * dy);
                if (d > max_dist) max_dist = d;
            }

            if (max_dist < threshold && max_dist < local_best_val) {
                local_best_val = max_dist;
                local_best_idx = c;
            }
        }

        sdist[threadIdx.x] = local_best_val;
        sidx[threadIdx.x] = local_best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double ov = sdist[threadIdx.x + stride];
                const int oi = sidx[threadIdx.x + stride];
                const double mv = sdist[threadIdx.x];
                const int mi = sidx[threadIdx.x];
                // Prefer the candidate with the smaller distance; break ties
                // by choosing the smaller point index, matching the
                // sequential algorithm's first-minimum-wins behavior.
                const bool take_other =
                    (oi >= 0) && (mi < 0 || ov < mv || (ov == mv && oi < mi));
                if (take_other) {
                    sdist[threadIdx.x] = ov;
                    sidx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            best_idx = sidx[0];
        }
        __syncthreads();

        if (best_idx < 0) break;

        if (threadIdx.x == 0) {
            in_cluster[best_idx] = 1;
            members[member_count] = best_idx;
            member_count++;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_cardinality[b] = member_count;
    }
}

// Main QT clustering algorithm (CUDA-parallel)
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    std::vector<double> h_px(N), h_py(N);
    for (int i = 0; i < N; ++i) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
    }

    double *d_px = nullptr, *d_py = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seed_list = nullptr;
    int* d_out_cardinality = nullptr;
    int* d_out_members = nullptr;
    unsigned char* d_in_cluster_scratch = nullptr;

    CUDA_CHECK(cudaMalloc(&d_px, static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_seed_list, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_cardinality, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_members, static_cast<size_t>(N) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster_scratch, static_cast<size_t>(N) * N));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), static_cast<size_t>(N) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), static_cast<size_t>(N) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, static_cast<size_t>(N)));

    std::vector<unsigned char> h_clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    const int threads = 256;
    const size_t shmem_bytes = static_cast<size_t>(threads) * (sizeof(double) + sizeof(int));

    std::vector<int> h_cardinality;
    std::vector<int> h_members_row;

    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_seed_list, unclustered_indices.data(),
                              static_cast<size_t>(num_seeds) * sizeof(int),
                              cudaMemcpyHostToDevice));

        growClustersKernel<<<num_seeds, threads, shmem_bytes>>>(
            d_px, d_py, d_seed_list, num_seeds, d_clustered, threshold, N,
            d_out_cardinality, d_out_members, d_in_cluster_scratch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        h_cardinality.resize(num_seeds);
        CUDA_CHECK(cudaMemcpy(h_cardinality.data(), d_out_cardinality,
                              static_cast<size_t>(num_seeds) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_b = -1;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cardinality[i] > max_cardinality) {
                max_cardinality = h_cardinality[i];
                best_b = i;
            }
        }

        if (best_b < 0 || max_cardinality <= 0) break;

        h_members_row.resize(max_cardinality);
        CUDA_CHECK(cudaMemcpy(h_members_row.data(),
                              d_out_members + static_cast<size_t>(best_b) * N,
                              static_cast<size_t>(max_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_b];
        cluster.members.assign(h_members_row.begin(), h_members_row.end());
        clusters.push_back(cluster);

        for (const int m : cluster.members) h_clustered[m] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), static_cast<size_t>(N),
                              cudaMemcpyHostToDevice));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&h_clustered](int idx) { return h_clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_clustered);
    cudaFree(d_seed_list);
    cudaFree(d_out_cardinality);
    cudaFree(d_out_members);
    cudaFree(d_in_cluster_scratch);

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

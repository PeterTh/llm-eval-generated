// QT Clustering Benchmark - CUDA-accelerated version
// The core algorithm semantics are preserved; distance computations over
// many candidates are accelerated on the GPU to improve parallel scalability.

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

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Generate synthetic 2D point data in clusters
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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// GPU buffers (allocated once per run)
static double* d_x = nullptr;
static double* d_y = nullptr;
static unsigned char* d_clustered = nullptr;
static unsigned char* d_in_cluster = nullptr;
static double* d_maxdist = nullptr;
static int* d_members = nullptr; // workspace for cluster member indices

// Kernel: for each candidate compute max distance to members (early exit using threshold_sq)
__global__ void compute_max_distances_kernel(const double* xs, const double* ys, int point_count,
                                             const int* members, int member_count,
                                             const unsigned char* clustered_mask,
                                             const unsigned char* in_cluster_mask,
                                             double threshold_sq, double* out_maxdist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;
    if (clustered_mask[idx] || in_cluster_mask[idx]) {
        out_maxdist[idx] = 1e300; // effectively infinity
        return;
    }
    double maxd2 = 0.0;
    for (int i = 0; i < member_count; ++i) {
        int m = members[i];
        double dx = xs[idx] - xs[m];
        double dy = ys[idx] - ys[m];
        double d2 = dx * dx + dy * dy;
        if (d2 > maxd2) maxd2 = d2;
        if (maxd2 > threshold_sq) break; // no need to continue, candidate invalid
    }
    out_maxdist[idx] = sqrt(maxd2);
}

// Initialize GPU buffers for points and masks
void gpu_init_points(const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    std::vector<double> xs(N), ys(N);
    for (int i = 0; i < N; ++i) { xs[i] = points[i].x; ys[i] = points[i].y; }
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(d_x, xs.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, ys.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_maxdist, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
}

void gpu_free() {
    if (d_x) CUDA_CHECK(cudaFree(d_x));
    if (d_y) CUDA_CHECK(cudaFree(d_y));
    if (d_clustered) CUDA_CHECK(cudaFree(d_clustered));
    if (d_in_cluster) CUDA_CHECK(cudaFree(d_in_cluster));
    if (d_maxdist) CUDA_CHECK(cudaFree(d_maxdist));
    if (d_members) CUDA_CHECK(cudaFree(d_members));
    d_x = d_y = nullptr;
    d_clustered = d_in_cluster = nullptr;
    d_maxdist = nullptr;
    d_members = nullptr;
}

// GPU-accelerated version of findClosestPoint. Returns -1 if none found.
int findClosestPointGPU(const std::vector<int>& cluster_members,
                        const std::vector<bool>& clustered,
                        const std::vector<bool>& in_cluster,
                        const int point_count,
                        const double threshold) {
    // Copy masks to device (clustered: global state, in_cluster: local to this candidate cluster)
    std::vector<unsigned char> clustered_mask(point_count), in_cluster_mask(point_count);
    for (int i = 0; i < point_count; ++i) { clustered_mask[i] = clustered[i] ? 1 : 0; in_cluster_mask[i] = in_cluster[i] ? 1 : 0; }
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered_mask.data(), sizeof(unsigned char) * point_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_cluster, in_cluster_mask.data(), sizeof(unsigned char) * point_count, cudaMemcpyHostToDevice));

    // Copy members to device
    const int member_count = static_cast<int>(cluster_members.size());
    if (member_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_members, cluster_members.data(), sizeof(int) * member_count, cudaMemcpyHostToDevice));
    }

    // Launch kernel
    const int threads = 256;
    const int blocks = (point_count + threads - 1) / threads;
    const double threshold_sq = threshold * threshold;
    compute_max_distances_kernel<<<blocks, threads>>>(d_x, d_y, point_count, d_members, member_count, d_clustered, d_in_cluster, threshold_sq, d_maxdist);
    CUDA_CHECK(cudaGetLastError());

    // Copy max distances back
    std::vector<double> maxd(point_count);
    CUDA_CHECK(cudaMemcpy(maxd.data(), d_maxdist, sizeof(double) * point_count, cudaMemcpyDeviceToHost));

    // Find the best candidate on host (min max distance < threshold)
    int best = -1;
    double best_d = std::numeric_limits<double>::max();
    for (int i = 0; i < point_count; ++i) {
        if (clustered[i] || in_cluster[i]) continue;
        double md = maxd[i];
        if (md < threshold && md < best_d) { best_d = md; best = i; }
    }
    return best;
}

// Generate a candidate cluster starting from a seed point (uses GPU to accelerate candidate evaluation)
int generateCandidateClusterGPU(const int seed_point,
                                 const std::vector<bool>& clustered,
                                 const std::vector<Point>& points,
                                 const double threshold,
                                 const int point_count,
                                 std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < point_count) {
        int closest = findClosestPointGPU(members, clustered, in_cluster, point_count, threshold);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (uses GPU-accelerated candidate evaluation)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    // Initialize GPU data
    gpu_init_points(points);

    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;

        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateClusterGPU(seed, clustered, points, threshold, N, &candidate_members);
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = std::move(candidate_members);
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            for (size_t i = 0; i < best_cluster_members.size(); ++i) clustered[best_cluster_members[i]] = true;
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end());
        } else {
            break;
        }
    }

    gpu_free();
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
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
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) clustered_count++;
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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (num_points <= 0 || threshold <= 0.0) { printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold); return 1; }

    printf("QT Clustering Benchmark (CUDA)\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());

    int total_clustered = 0; int max_cluster_size = 0;
    for (size_t i = 0; i < clusters.size(); ++i) { const int size = static_cast<int>(clusters[i].members.size()); total_clustered += size; max_cluster_size = std::max(max_cluster_size, size); }
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
        for (size_t c = 0; c < clusters.size(); ++c) for (size_t i = 0; i < clusters[c].members.size(); ++i) membership[clusters[c].members[i]] = static_cast<int>(c);
        for (int m : membership) membershipData.push_back(static_cast<double>(m));
        print_results(membershipData, "ClusterMembership");
    }

    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        if (valid) { printf("Validation: PASSED\n"); return 0; } else { printf("Validation: FAILED\n"); return 1; }
    }
    return 0;
}

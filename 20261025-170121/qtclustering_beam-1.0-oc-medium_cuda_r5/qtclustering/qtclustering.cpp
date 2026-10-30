// QT Clustering Benchmark - CUDA Parallel Version
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
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
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

__global__ void computeCardinalities(const double* px, const double* py,
                                    const char* clustered, double threshold, int N,
                                    int* cardOut, int* membersOut, char* inClusterBase) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= N) return;
    if (clustered[seed]) { cardOut[seed] = 0; return; }
    char* inC = inClusterBase + (size_t)seed * N;
    for (int i = 0; i < N; ++i) inC[i] = 0;
    int* myMembers = membersOut + (size_t)seed * N;
    int mc = 0;
    inC[seed] = 1;
    myMembers[mc++] = seed;
    while (mc < N) {
        int closest = -1;
        double minDiam = 1e300;
        for (int cand = 0; cand < N; ++cand) {
            if (clustered[cand] || inC[cand]) continue;
            double maxDist = 0.0;
            for (int m = 0; m < mc; ++m) {
                int member = myMembers[m];
                double dx = px[cand] - px[member];
                double dy = py[cand] - py[member];
                double d = sqrt(dx*dx + dy*dy);
                if (d > maxDist) maxDist = d;
            }
            if (maxDist < threshold && maxDist < minDiam) {
                minDiam = maxDist;
                closest = cand;
            }
        }
        if (closest < 0) break;
        inC[closest] = 1;
        myMembers[mc++] = closest;
    }
    cardOut[seed] = mc;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);
    std::vector<Cluster> clusters;

    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) { px[i] = points[i].x; py[i] = points[i].y; }

    double *d_px, *d_py;
    char *d_clustered;
    int *d_card, *d_members;
    char *d_inC;
    cudaMalloc(&d_px, N*sizeof(double));
    cudaMalloc(&d_py, N*sizeof(double));
    cudaMalloc(&d_clustered, N*sizeof(char));
    cudaMalloc(&d_card, N*sizeof(int));
    cudaMalloc(&d_members, (size_t)N*N*sizeof(int));
    cudaMalloc(&d_inC, (size_t)N*N*sizeof(char));
    cudaMemcpy(d_px, px.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    while (!unclustered_indices.empty()) {
        cudaMemcpy(d_clustered, clustered.data(), N*sizeof(char), cudaMemcpyHostToDevice);
        computeCardinalities<<<blocks, threads>>>(d_px, d_py, d_clustered, threshold, N, d_card, d_members, d_inC);
        cudaDeviceSynchronize();
        std::vector<int> cards(N);
        cudaMemcpy(cards.data(), d_card, N*sizeof(int), cudaMemcpyDeviceToHost);

        int max_card = -1;
        int best_seed = -1;
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            if (cards[seed] > max_card) {
                max_card = cards[seed];
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_card > 0) {
            std::vector<int> members(max_card);
            cudaMemcpy(members.data(), d_members + (size_t)best_seed * N, max_card*sizeof(int), cudaMemcpyDeviceToHost);
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = members;
            clusters.push_back(cluster);
            for (int m : members) clustered[m] = 1;
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else break;
    }

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_members); cudaFree(d_inC);
    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, const double threshold) {
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
        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c);
                valid = false;
            }
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
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        return 1;
    }
    printf("QT Clustering Benchmark\n");
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

    int total_clustered = 0;
    int max_cluster_size = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }
    const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
    printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);

    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

    if (printResults) {
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) membershipData.push_back(static_cast<double>(m));
        print_results(membershipData, "ClusterMembership");
    }

    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        if (valid) { printf("Validation: PASSED\n"); return 0; }
        else { printf("Validation: FAILED\n"); return 1; }
    }
    return 0;
}

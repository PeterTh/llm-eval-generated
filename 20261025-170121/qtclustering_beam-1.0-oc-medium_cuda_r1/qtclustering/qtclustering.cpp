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

__device__ inline double distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

__global__ void generateClustersKernel(
    const double* px, const double* py,
    const char* clustered,
    const int* seeds, int M,
    int N, double threshold,
    char* inClusterWs, int* membersWs, int* cardOut) {
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= M) return;
    int seed = seeds[si];
    if (clustered[seed]) { cardOut[si] = 0; return; }

    char* inCluster = inClusterWs + (size_t)si * N;
    int* members = membersWs + (size_t)si * N;
    for (int i = 0; i < N; ++i) inCluster[i] = 0;

    inCluster[seed] = 1;
    members[0] = seed;
    int size = 1;

    while (size < N) {
        int closest = -1;
        double minDiam = 1e300;
        for (int cand = 0; cand < N; ++cand) {
            if (clustered[cand] || inCluster[cand]) continue;
            double maxDist = 0.0;
            for (int k = 0; k < size; ++k) {
                int member = members[k];
                double d = distance(px[cand], py[cand], px[member], py[member]);
                if (d > maxDist) maxDist = d;
            }
            if (maxDist < threshold && maxDist < minDiam) {
                minDiam = maxDist;
                closest = cand;
            }
        }
        if (closest < 0) break;
        inCluster[closest] = 1;
        members[size++] = closest;
    }
    cardOut[si] = size;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) { px[i] = points[i].x; py[i] = points[i].y; }

    double *d_px, *d_py;
    char *d_clustered;
    cudaMalloc(&d_px, N*sizeof(double));
    cudaMalloc(&d_py, N*sizeof(double));
    cudaMalloc(&d_clustered, N*sizeof(char));
    cudaMemcpy(d_px, px.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    std::vector<Cluster> clusters;

    while (!unclustered_indices.empty()) {
        int M = static_cast<int>(unclustered_indices.size());
        cudaMemcpy(d_clustered, clustered.data(), N*sizeof(char), cudaMemcpyHostToDevice);

        int *d_seeds, *d_card, *d_membersWs;
        char *d_inClusterWs;
        cudaMalloc(&d_seeds, M*sizeof(int));
        cudaMalloc(&d_card, M*sizeof(int));
        cudaMemcpy(d_seeds, unclustered_indices.data(), M*sizeof(int), cudaMemcpyHostToDevice);

        size_t ws = (size_t)M * N;
        cudaMalloc(&d_inClusterWs, ws*sizeof(char));
        cudaMalloc(&d_membersWs, ws*sizeof(int));

        int threads = 256;
        int blocks = (M + threads - 1) / threads;
        generateClustersKernel<<<blocks, threads>>>(d_px, d_py, d_clustered, d_seeds, M, N, threshold, d_inClusterWs, d_membersWs, d_card);
        cudaDeviceSynchronize();

        std::vector<int> cards(M);
        cudaMemcpy(cards.data(), d_card, M*sizeof(int), cudaMemcpyDeviceToHost);

        int bestIdx = -1;
        int maxCard = -1;
        for (int i = 0; i < M; ++i) {
            if (cards[i] > maxCard) { maxCard = cards[i]; bestIdx = i; }
        }

        if (bestIdx >= 0 && maxCard > 0) {
            int bestSeed = unclustered_indices[bestIdx];
            std::vector<int> bestMembers(maxCard);
            cudaMemcpy(bestMembers.data(), d_membersWs + (size_t)bestIdx * N, maxCard*sizeof(int), cudaMemcpyDeviceToHost);

            Cluster cluster;
            cluster.seed_point = bestSeed;
            cluster.members = bestMembers;
            clusters.push_back(cluster);

            for (int m : bestMembers) clustered[m] = 1;

            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }

        cudaFree(d_seeds);
        cudaFree(d_card);
        cudaFree(d_inClusterWs);
        cudaFree(d_membersWs);
    }

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_clustered);

    return clusters;
}

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
                const double dist = std::hypot(points[cluster.members[i]].x - points[cluster.members[j]].x,
                                                points[cluster.members[i]].y - points[cluster.members[j]].y);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) {
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
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
    
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    
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
    
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", 
           clusters_per_sec, points_per_sec);
    
    if (printResults) {
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
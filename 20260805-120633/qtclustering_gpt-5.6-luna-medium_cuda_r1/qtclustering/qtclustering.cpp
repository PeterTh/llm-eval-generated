// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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
        // The original distribution can produce zero here for small inputs,
        // which would leave the generator spinning forever.
        int group_cnt = std::max(1, static_cast<int>(frand() * (N / 30.0)));
        
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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct CandidateChoice {
    double diameter;
    int index;
};

// One block handles one seed. Threads scan candidate points in parallel and
// the shared-memory reduction deliberately prefers the lower index on equal
// diameters, matching the sequential loop's tie behavior.
__global__ void chooseCandidates(const Point* points, const unsigned char* clustered,
                                  const int* members, const int* member_counts,
                                  const unsigned char* active, CandidateChoice* choices,
                                  const int point_count, const double threshold) {
    const int seed_slot = static_cast<int>(blockIdx.x);
    if (active[seed_slot] == 0) return;

    extern __shared__ CandidateChoice reduction[];
    CandidateChoice best{DBL_MAX, -1};
    const int member_count = member_counts[seed_slot];
    const size_t member_base = static_cast<size_t>(seed_slot) * point_count;

    for (int candidate = static_cast<int>(threadIdx.x); candidate < point_count;
         candidate += static_cast<int>(blockDim.x)) {
        if (clustered[candidate]) continue;

        bool already_member = false;
        double max_dist = 0.0;
        for (int i = 0; i < member_count; ++i) {
            const int member = members[member_base + i];
            if (member == candidate) {
                already_member = true;
                break;
            }
            max_dist = fmax(max_dist, distance(points[candidate], points[member]));
        }

        if (!already_member && max_dist < threshold &&
            (max_dist < best.diameter ||
             (max_dist == best.diameter && candidate < best.index))) {
            best = {max_dist, candidate};
        }
    }

    reduction[threadIdx.x] = best;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const CandidateChoice other = reduction[threadIdx.x + stride];
            if (other.diameter < reduction[threadIdx.x].diameter ||
                (other.diameter == reduction[threadIdx.x].diameter &&
                 other.index >= 0 &&
                 (reduction[threadIdx.x].index < 0 ||
                  other.index < reduction[threadIdx.x].index))) {
                reduction[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) choices[seed_slot] = reduction[0];
}

__global__ void appendCandidates(int* members, int* member_counts,
                                  unsigned char* active, const CandidateChoice* choices,
                                  const int point_count) {
    const int seed_slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const CandidateChoice choice = choices[seed_slot];
    if (active[seed_slot] == 0 || choice.index < 0) {
        active[seed_slot] = 0;
        return;
    }
    const int count = member_counts[seed_slot];
    members[static_cast<size_t>(seed_slot) * point_count + count] = choice.index;
    member_counts[seed_slot] = count + 1;
    if (count + 1 >= point_count) active[seed_slot] = 0;
}

static void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices. Their order is significant for the
    // deterministic seed tie-break and remains ascending after erasure.
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_members = nullptr;
    int* d_member_counts = nullptr;
    unsigned char* d_active = nullptr;
    CandidateChoice* d_choices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    while (!unclustered_indices.empty()) {
        const int seed_count = static_cast<int>(unclustered_indices.size());
        const size_t matrix_size = static_cast<size_t>(seed_count) * N;
        std::vector<int> initial_members(matrix_size, 0);
        std::vector<int> member_counts(seed_count, 1);
        std::vector<unsigned char> active(seed_count, 1);
        for (int slot = 0; slot < seed_count; ++slot) {
            initial_members[static_cast<size_t>(slot) * N] = unclustered_indices[slot];
        }

        CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * matrix_size));
        CUDA_CHECK(cudaMalloc(&d_member_counts, sizeof(int) * seed_count));
        CUDA_CHECK(cudaMalloc(&d_active, sizeof(unsigned char) * seed_count));
        CUDA_CHECK(cudaMalloc(&d_choices, sizeof(CandidateChoice) * seed_count));
        CUDA_CHECK(cudaMemcpy(d_members, initial_members.data(), sizeof(int) * matrix_size,
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_member_counts, member_counts.data(), sizeof(int) * seed_count,
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_active, active.data(), sizeof(unsigned char) * seed_count,
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), sizeof(unsigned char) * N,
                             cudaMemcpyHostToDevice));

        int active_count = seed_count;
        while (active_count != 0) {
            chooseCandidates<<<seed_count, 256, 256 * sizeof(CandidateChoice)>>>(
                d_points, d_clustered, d_members, d_member_counts, d_active, d_choices,
                N, threshold);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<CandidateChoice> choices(seed_count);
            CUDA_CHECK(cudaMemcpy(choices.data(), d_choices,
                                  sizeof(CandidateChoice) * seed_count,
                                  cudaMemcpyDeviceToHost));

            active_count = 0;
            for (int slot = 0; slot < seed_count; ++slot) {
                if (choices[slot].index >= 0 &&
                    member_counts[slot] + 1 < N) ++active_count;
            }
            appendCandidates<<<(seed_count + 255) / 256, 256>>>(
                d_members, d_member_counts, d_active, d_choices, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int slot = 0; slot < seed_count; ++slot) {
                if (choices[slot].index < 0) active[slot] = 0;
                else member_counts[slot]++;
            }
        }

        CUDA_CHECK(cudaMemcpy(member_counts.data(), d_member_counts,
                              sizeof(int) * seed_count, cudaMemcpyDeviceToHost));
        int max_cardinality = -1;
        int best_seed = -1;
        int best_slot = -1;
        std::vector<int> best_cluster_members;

        for (int slot = 0; slot < seed_count; ++slot) {
            const int cardinality = member_counts[slot];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = unclustered_indices[slot];
                best_slot = slot;
            }
        }

        best_cluster_members.resize(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                              d_members + static_cast<size_t>(best_slot) * N,
                              sizeof(int) * max_cardinality, cudaMemcpyDeviceToHost));
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }

        cudaFree(d_members); d_members = nullptr;
        cudaFree(d_member_counts); d_member_counts = nullptr;
        cudaFree(d_active); d_active = nullptr;
        cudaFree(d_choices); d_choices = nullptr;
    }

    cudaFree(d_points);
    cudaFree(d_clustered);
    
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

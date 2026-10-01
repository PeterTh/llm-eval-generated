// QT Clustering Benchmark - MPI, OpenMP and CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <vector>

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

// Each block evaluates one seed. Candidate diameters are updated incrementally
// as points join the cluster. Reductions prefer the lowest point index on ties.
static int selected_device = 0;
static constexpr int BLOCK_SIZE = 256;
static constexpr int SEED_BATCH = 128;

__global__ void evaluateSeeds(const Point* points, const int* clustered,
                              const int* seeds, int count, int n, double threshold,
                              double* diameters, int* cardinalities, int* members) {
    const int slot = blockIdx.x;
    if (slot >= count) return;
    const int tid = threadIdx.x;
    const int seed = seeds[slot];
    double* diameter = diameters + static_cast<size_t>(slot) * n;
    int* member = members + static_cast<size_t>(slot) * n;
    __shared__ double min_values[BLOCK_SIZE];
    __shared__ int min_indices[BLOCK_SIZE];
    __shared__ int current, size;
    for (int c = tid; c < n; c += BLOCK_SIZE)
        diameter[c] = (c == seed || clustered[c]) ? -1.0 : 0.0;
    if (tid == 0) {
        member[0] = seed;
        current = seed;
        size = 1;
    }
    __syncthreads();
    while (true) {
        double best_value = DBL_MAX;
        int best_index = INT_MAX;
        const Point last = points[current];
        for (int c = tid; c < n; c += BLOCK_SIZE) {
            double value = diameter[c];
            if (value < 0.0) continue;
            const double dx = points[c].x - last.x;
            const double dy = points[c].y - last.y;
            const double dist = sqrt(dx * dx + dy * dy);
            if (dist > value) value = dist;
            diameter[c] = value;
            if (value < threshold &&
                (value < best_value || (value == best_value && c < best_index))) {
                best_value = value;
                best_index = c;
            }
        }
        min_values[tid] = best_value;
        min_indices[tid] = best_index;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride; stride >>= 1) {
            if (tid < stride &&
                (min_values[tid + stride] < min_values[tid] ||
                 (min_values[tid + stride] == min_values[tid] &&
                  min_indices[tid + stride] < min_indices[tid]))) {
                min_values[tid] = min_values[tid + stride];
                min_indices[tid] = min_indices[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            current = min_indices[0];
            if (current != INT_MAX) {
                diameter[current] = -1.0;
                member[size++] = current;
            }
        }
        __syncthreads();
        if (current == INT_MAX) break;
    }
    if (tid == 0) cardinalities[slot] = size;
}

static void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

struct Worker {
    cudaStream_t stream{};
    int *seeds{}, *cardinalities{}, *members{};
    double* diameters{};
    std::vector<int> host_cardinalities;
    std::vector<int> best_members;
    int best_seed = INT_MAX;
    int best_size = 0;

    explicit Worker(int n) : host_cardinalities(SEED_BATCH), best_members(n) {
        checkCuda(cudaStreamCreate(&stream));
        checkCuda(cudaMalloc(&seeds, SEED_BATCH * sizeof(int)));
        checkCuda(cudaMalloc(&cardinalities, SEED_BATCH * sizeof(int)));
        checkCuda(cudaMalloc(&members, static_cast<size_t>(SEED_BATCH) * n * sizeof(int)));
        checkCuda(cudaMalloc(&diameters, static_cast<size_t>(SEED_BATCH) * n * sizeof(double)));
    }
    ~Worker() {
        cudaFree(seeds);
        cudaFree(cardinalities);
        cudaFree(members);
        cudaFree(diameters);
        cudaStreamDestroy(stream);
    }
    void run(const std::vector<int>& local_seeds, size_t begin, int count,
             int n, double threshold, const Point* points, const int* clustered) {
        checkCuda(cudaMemcpyAsync(seeds, local_seeds.data() + begin,
                                  count * sizeof(int), cudaMemcpyHostToDevice, stream));
        evaluateSeeds<<<count, BLOCK_SIZE, 0, stream>>>(
            points, clustered, seeds, count, n, threshold,
            diameters, cardinalities, members);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpyAsync(host_cardinalities.data(), cardinalities,
                                  count * sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        int best_slot = -1;
        for (int i = 0; i < count; ++i) {
            const int seed = local_seeds[begin + i];
            const int size = host_cardinalities[i];
            if (size > best_size || (size == best_size && seed < best_seed)) {
                best_size = size;
                best_seed = seed;
                best_slot = i;
            }
        }
        if (best_slot >= 0) {
            checkCuda(cudaMemcpyAsync(best_members.data(),
                      members + static_cast<size_t>(best_slot) * n,
                      best_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaStreamSynchronize(stream));
        }
    }
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  double threshold, int rank, int ranks) {
    const int n = static_cast<int>(points.size());
    Point* device_points = nullptr;
    int* device_clustered = nullptr;
    checkCuda(cudaMalloc(&device_points, static_cast<size_t>(n) * sizeof(Point)));
    checkCuda(cudaMalloc(&device_clustered, static_cast<size_t>(n) * sizeof(int)));
    checkCuda(cudaMemcpy(device_points, points.data(), static_cast<size_t>(n) * sizeof(Point),
                         cudaMemcpyHostToDevice));
    std::vector<int> clustered(n, 0), local_seeds;
    std::vector<Cluster> clusters;
    const int workers = std::min(4, omp_get_max_threads());
    std::vector<int> thread_sizes(workers), thread_seeds(workers);
    std::vector<std::vector<int>> thread_members(workers);
    bool done = false;

    #pragma omp parallel num_threads(workers) shared(done, local_seeds, clustered, clusters)
    {
        const int tid = omp_get_thread_num();
        checkCuda(cudaSetDevice(selected_device));
        Worker worker(n);
        while (true) {
            #pragma omp master
            {
                local_seeds.clear();
                for (int i = rank; i < n; i += ranks)
                    if (!clustered[i]) local_seeds.push_back(i);
                int remaining = 0;
                for (int v : clustered) remaining += !v;
                done = remaining == 0;
                if (!done)
                    checkCuda(cudaMemcpy(device_clustered, clustered.data(),
                                         static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));
            }
            #pragma omp barrier
            if (done) break;
            worker.best_size = 0;
            worker.best_seed = INT_MAX;
            const int batches = (static_cast<int>(local_seeds.size()) + SEED_BATCH - 1) / SEED_BATCH;
            #pragma omp for schedule(dynamic, 1)
            for (int batch = 0; batch < batches; ++batch) {
                const size_t begin = static_cast<size_t>(batch) * SEED_BATCH;
                const int count = std::min(SEED_BATCH, static_cast<int>(local_seeds.size() - begin));
                worker.run(local_seeds, begin, count, n, threshold,
                           device_points, device_clustered);
            }
            thread_sizes[tid] = worker.best_size;
            thread_seeds[tid] = worker.best_seed;
            thread_members[tid].assign(worker.best_members.begin(),
                                       worker.best_members.begin() + worker.best_size);
            #pragma omp barrier
            #pragma omp master
            {
                int local_size = 0, local_seed = INT_MAX, best_thread = -1;
                for (int t = 0; t < workers; ++t)
                    if (thread_sizes[t] > local_size ||
                        (thread_sizes[t] == local_size && thread_seeds[t] < local_seed)) {
                        local_size = thread_sizes[t];
                        local_seed = thread_seeds[t];
                        best_thread = t;
                    }
                int global_size = 0;
                MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                int candidate_seed = local_size == global_size ? local_seed : INT_MAX;
                int global_seed = INT_MAX;
                MPI_Allreduce(&candidate_seed, &global_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                const int owner = global_seed % ranks;
                std::vector<int> winning_members(global_size);
                if (rank == owner)
                    std::copy_n(thread_members[best_thread].begin(), global_size,
                                winning_members.begin());
                MPI_Bcast(winning_members.data(), global_size, MPI_INT, owner, MPI_COMM_WORLD);
                clusters.push_back({std::move(winning_members), global_seed});
                for (int member : clusters.back().members) clustered[member] = 1;
            }
            #pragma omp barrier
        }
    }
    checkCuda(cudaFree(device_points));
    checkCuda(cudaFree(device_clustered));
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) fprintf(stderr, "CUDA device required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    selected_device = local_rank % device_count;
    checkCuda(cudaSetDevice(selected_device));
    MPI_Comm_free(&local_comm);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false, printResults = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) parse_status = 2;
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parse_status = 1;
        }
    }
    if (num_points <= 0 || !(threshold > 0.0)) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        parse_status = 1;
    }
    if (parse_status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), num_points * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto cluster_start = std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time);
        printf("Clusters found: %zu\n", clusters.size());
        int total_clustered = 0, max_cluster_size = 0;
        for (const auto& cluster : clusters) {
            const int size = static_cast<int>(cluster.members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        const double average = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered,
               num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", average);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        const double seconds = static_cast<double>(cluster_time) / 1000.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters.size() / seconds, num_points / seconds);
        if (printResults) {
            std::vector<double> membershipData(num_points, -1.0);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (int member : clusters[c].members)
                    membershipData[member] = static_cast<double>(c);
            print_results(membershipData, "ClusterMembership");
        }
        if (validate) {
            result = validateClusters(clusters, points, threshold) ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Finalize();
    return result;
}

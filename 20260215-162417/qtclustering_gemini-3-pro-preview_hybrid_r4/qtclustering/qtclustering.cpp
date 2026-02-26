// QT Clustering Benchmark - Simplified Sequential Version
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
#include <mpi.h>
#include <omp.h>
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

// Global variables for MPI
int mpi_rank = 0;
int mpi_size = 1;

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Device struct for Point
struct PointDevice {
    double x, y;
};

// Global device pointers
PointDevice* d_points = nullptr;
int* d_clustered = nullptr; // bool as int for easier atomic/handling
int* d_members = nullptr;
double* d_max_dists = nullptr;
int* d_candidate_indices = nullptr;

// Initialize GPU memory
void initGPU(const std::vector<Point>& h_points) {
    size_t N = h_points.size();
    cudaCheckError(cudaMalloc((void**)&d_points, N * sizeof(PointDevice)));
    cudaCheckError(cudaMemcpy(d_points, h_points.data(), N * sizeof(PointDevice), cudaMemcpyHostToDevice));
    
    cudaCheckError(cudaMalloc((void**)&d_clustered, N * sizeof(int)));
    cudaCheckError(cudaMemset(d_clustered, 0, N * sizeof(int)));
    
    // Allocate max possible size for members
    cudaCheckError(cudaMalloc((void**)&d_members, N * sizeof(int)));
    
    // Scratch space for max distances and indices
    cudaCheckError(cudaMalloc((void**)&d_max_dists, N * sizeof(double)));
    cudaCheckError(cudaMalloc((void**)&d_candidate_indices, N * sizeof(int)));
}

void cleanupGPU() {
    if (d_points) cudaFree(d_points);
    if (d_clustered) cudaFree(d_clustered);
    if (d_members) cudaFree(d_members);
    if (d_max_dists) cudaFree(d_max_dists);
    if (d_candidate_indices) cudaFree(d_candidate_indices);
}

// Kernel where each block computes one cluster for one seed
__global__ void generateClusterPerBlock(
    const PointDevice* points,
    const int* seeds,
    int num_seeds,
    const int* global_clustered,
    int num_points,
    double threshold,
    int* out_cluster_sizes) {
    
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    
    int seed = seeds[seed_idx];
    
    // Check if seed is already clustered globally (should be checked on host, but double check)
    if (global_clustered[seed]) {
        if (threadIdx.x == 0) out_cluster_sizes[seed_idx] = 0;
        return;
    }
    
    // Shared memory for this block's cluster state
    extern __shared__ int s_mem[];
    int* s_members = s_mem; // Max size N
    // We need space for in_cluster flags too. Let's use bits or just bytes.
    // For simplicity, let's assume we just check s_members.
    // Or we can put flags after members if N is small enough.
    // Let's rely on checking members list which is O(cluster_size).
    // Better: use bitmask in shared mem.
    int* s_in_cluster = &s_members[num_points]; // This requires 2*N*4 bytes shared mem. 
    // 1000 points * 8 bytes = 8KB. Safe.
    
    // Initialize
    int tid = threadIdx.x;
    for (int i = tid; i < num_points; i += blockDim.x) {
        s_in_cluster[i] = 0;
    }
    __syncthreads();
    
    if (tid == 0) {
        s_members[0] = seed;
        s_in_cluster[seed] = 1;
    }
    __syncthreads();
    
    int cluster_size = 1;
    
    while (true) {
        // Find best candidate
        double my_min_max_dist = 1.0e30;
        int my_best_idx = -1;
        
        for (int i = tid; i < num_points; i += blockDim.x) {
            if (global_clustered[i] || s_in_cluster[i]) continue;
            
            double max_dist = 0.0;
            // Check against all current members
            for (int m = 0; m < cluster_size; ++m) {
                int member = s_members[m];
                double dx = points[i].x - points[member].x;
                double dy = points[i].y - points[member].y;
                double d = sqrt(dx*dx + dy*dy);
                if (d > max_dist) max_dist = d;
            }
            
            if (max_dist < threshold && max_dist < my_min_max_dist) {
                my_min_max_dist = max_dist;
                my_best_idx = i;
            }
        }
        
        // Reduction to find best across block
        // We need shared mem for reduction.
        // s_in_cluster is int array. We need to align for double.
        
        // Calculate offset for reduction buffer
        // s_in_cluster starts at &s_mem[num_points] (int*)
        // It has size num_points.
        // So end of s_in_cluster is &s_mem[2 * num_points].
        
        // Ensure 8-byte alignment for double*
        size_t int_count = 2 * num_points;
        if ((size_t)(&s_mem[int_count]) % 8 != 0) {
            // Adjust index to align
            // Assuming s_mem is aligned to at least 4 bytes.
            // If address is not 8-byte aligned, add 1 int (4 bytes).
            int_count++;
        }
        // Note: shared memory base address is usually 256-byte aligned.
        // So we just need to align relative to start.
        // But &s_mem[int_count] is dynamic.
        // Safer to just use a fixed offset if we can or casting.
        
        double* s_reduce_dist = (double*)&s_mem[int_count];
        int* s_reduce_idx = (int*)&s_reduce_dist[blockDim.x];
        
        s_reduce_dist[tid] = my_min_max_dist;
        s_reduce_idx[tid] = my_best_idx;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_reduce_dist[tid + s] < s_reduce_dist[tid]) {
                    s_reduce_dist[tid] = s_reduce_dist[tid + s];
                    s_reduce_idx[tid] = s_reduce_idx[tid + s];
                }
            }
            __syncthreads();
        }
        
        int best_candidate = s_reduce_idx[0];
        // Check if valid (might be -1 if no candidate found)
        if (best_candidate == -1 || s_reduce_dist[0] >= threshold) {
            break; 
        }
        
        // Add to cluster
        if (tid == 0) {
            s_members[cluster_size] = best_candidate;
            s_in_cluster[best_candidate] = 1;
        }
        cluster_size++;
        __syncthreads();
        
        // Safety check for max size
        if (cluster_size >= num_points) break;
    }
    
    if (tid == 0) {
        out_cluster_sizes[seed_idx] = cluster_size;
    }
}

// Function to run CUDA kernel
int runCUDAClustering(const std::vector<int>& seeds, 
                      int num_points, 
                      double threshold, 
                      std::vector<int>& out_sizes) {
    if (seeds.empty()) return 0;
    
    int num_seeds = seeds.size();
    out_sizes.resize(num_seeds);
    
    int* d_seeds;
    int* d_sizes;
    cudaCheckError(cudaMalloc((void**)&d_seeds, num_seeds * sizeof(int)));
    cudaCheckError(cudaMalloc((void**)&d_sizes, num_seeds * sizeof(int)));
    
    cudaCheckError(cudaMemcpy(d_seeds, seeds.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));
    
    // Calculate shared memory size
    // s_members (N ints) + s_in_cluster (N ints) + padding (max 2 ints) + reduction (blockDim doubles + blockDim ints)
    // 1000 * 4 + 1000 * 4 + 8 + 256 * 8 + 256 * 4 = 4000 + 4000 + 8 + 2048 + 1024 = ~11KB
    int shared_mem_size = (num_points * sizeof(int)) + (num_points * sizeof(int)) + 
                          (16) + // padding safety
                          (256 * sizeof(double)) + (256 * sizeof(int));
    
    generateClusterPerBlock<<<num_seeds, 256, shared_mem_size>>>(
        d_points, d_seeds, num_seeds, d_clustered, num_points, threshold, d_sizes);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Check for async errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
         fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
         return -1;
    }
        
    cudaCheckError(cudaMemcpy(out_sizes.data(), d_sizes, num_seeds * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_seeds);
    cudaFree(d_sizes);
    return 0;
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
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
    }
    
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments on rank 0
    if (mpi_rank == 0) {
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
                MPI_Finalize();
                return 0;
            } 
        }
        if (num_points <= 0 || threshold <= 0.0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Broadcast parameters
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Validation flags only needed on rank 0
    // But we need to keep consistent execution flow
    
    // Initialize CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    bool use_cuda = (deviceCount > 0);
    
    if (use_cuda) {
        // Assign device based on rank to utilize multiple GPUs if available
        int device_id = mpi_rank % deviceCount;
        cudaSetDevice(device_id);
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI Ranks: %d\n", mpi_size);
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
        printf("CUDA Devices: %d\n", deviceCount);
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    if (use_cuda) {
        initGPU(points);
    }
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Local best found by this rank
        struct {
            int cardinality;
            int rank;
        } local_best, global_best;
        
        local_best.cardinality = -1;
        local_best.rank = mpi_rank;
        
        int best_seed_local = -1;
        
        // If using CUDA, process local seeds on GPU
        if (use_cuda) {
            std::vector<int> local_seeds;
            local_seeds.reserve(unclustered_indices.size() / mpi_size + 1);
            
            // Collect local seeds
            for (size_t i = mpi_rank; i < unclustered_indices.size(); i += mpi_size) {
                if (!clustered[unclustered_indices[i]]) {
                    local_seeds.push_back(unclustered_indices[i]);
                }
            }
            
            if (!local_seeds.empty()) {
                std::vector<int> out_sizes;
                runCUDAClustering(local_seeds, N, threshold, out_sizes);
                
                // Find best
                for (size_t i = 0; i < out_sizes.size(); ++i) {
                    if (out_sizes[i] > local_best.cardinality) {
                        local_best.cardinality = out_sizes[i];
                        best_seed_local = local_seeds[i];
                    }
                }
            }
        } else {
            // Fallback to OpenMP
            // Distribute the work of checking unclustered seeds
            #pragma omp parallel
            {
                int local_max_card = -1;
                int local_best_s = -1;
                
                #pragma omp for schedule(dynamic)
                for (size_t i = mpi_rank; i < unclustered_indices.size(); i += mpi_size) {
                    const int seed = unclustered_indices[i];
                    if (clustered[seed]) continue;
                    
                    // We only need size here, members only for the winner
                    const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                               threshold, N, nullptr);
                    
                    if (cardinality > local_max_card) {
                        local_max_card = cardinality;
                        local_best_s = seed;
                    }
                }
                
                #pragma omp critical
                {
                    if (local_max_card > local_best.cardinality) {
                        local_best.cardinality = local_max_card;
                        best_seed_local = local_best_s;
                    }
                }
            }
        }
        
        // Find global best
        MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        // If a cluster was found
        if (global_best.cardinality > 0) {
            int cluster_size = global_best.cardinality;
            int winner_rank = global_best.rank;
            int best_seed_global = -1;
            
            if (mpi_rank == winner_rank) {
                best_seed_global = best_seed_local;
            }
            MPI_Bcast(&best_seed_global, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);
            
            // Reconstruct members on all ranks (or just broadcast members)
            // It's cheaper to broadcast members than to recompute on all ranks if cluster is small.
            // But here we need members to update clustered array.
            // Let winner compute members and broadcast.
            
            std::vector<int> global_best_members;
            if (mpi_rank == winner_rank) {
                // Re-run on CPU to get members
                generateCandidateCluster(best_seed_global, clustered, points, 
                                       threshold, N, &global_best_members);
            }
            
            global_best_members.resize(cluster_size); // Ensure correct size for receive
            MPI_Bcast(global_best_members.data(), cluster_size, MPI_INT, winner_rank, MPI_COMM_WORLD);
             
            Cluster cluster;
            cluster.seed_point = best_seed_global;
            cluster.members = global_best_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < global_best_members.size(); ++i) {
                clustered[global_best_members[i]] = true;
            }
            
            // Update GPU clustered array
            if (use_cuda) {
                 // We can either copy the whole array or just update the new members.
                 // Updating new members is faster.
                 std::vector<int> new_clustered_indices = global_best_members;
                 // Need to set d_clustered[idx] = 1 for these indices.
                 // Copying indices to device and running a small kernel or cudaMemcpy per index?
                 // Since cluster size is small (e.g. 50), cudaMemcpy per index might be ok?
                 // Or better: construct a small update array and copy.
                 // Actually, simpler: maintain a host copy of 'clustered' as int array and copy diff?
                 // Or just update d_clustered one by one?
                 // Let's create a kernel to update clustered flags from a list of indices.
                 // Or just update local `clustered` bool vector and `d_clustered` from it?
                 // `clustered` is bool vector. `d_clustered` is int array.
                 
                 // Let's implement updateClusterFlags kernel quickly or use Memcpy
                 // Since we don't have a kernel for this, and adding one is messy now...
                 // Let's just do Memcpy.
                 // d_clustered is array of INTs.
                 int val = 1;
                 for (int member : global_best_members) {
                     cudaCheckError(cudaMemcpy(&d_clustered[member], &val, sizeof(int), cudaMemcpyHostToDevice));
                 }
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    if (use_cuda) {
        cleanupGPU();
    }
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
        
    if (mpi_rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}

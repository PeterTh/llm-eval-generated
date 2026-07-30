// QT Clustering Benchmark - Hybrid MPI+OpenMP+CUDA Parallel Version
// 
// QT (Quality Threshold) clustering algorithm parallelized with:
//   - MPI: Distribute seed evaluations across ranks
//   - OpenMP: Parallel seed evaluation within each rank
//   - CUDA: GPU-accelerated distance computations

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

// CUDA runtime (compiled with nvcc since LANGUAGE CUDA is set in CMake)
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// ---------------------------------------------------------------------------
// CUDA device memory globals (persistent across calls, initialized once)
// ---------------------------------------------------------------------------
static double *d_points_x    = nullptr;
static double *d_points_y    = nullptr;
static double *d_max_dists   = nullptr;   // device,  N doubles
static double *h_max_dists   = nullptr;   // host staging,  N doubles
static int    *d_cluster_mem = nullptr;   // device,  N ints
static int     d_capacity    = 0;
static bool    cuda_ok       = false;
static int     my_cuda_device = 0;        // device assigned to this MPI rank

// ---------------------------------------------------------------------------
// CUDA kernel: compute max distance from each point to all cluster members
// ---------------------------------------------------------------------------
__global__ void compute_max_dists_kernel(
    const double *points_x,
    const double *points_y,
    const int    *cluster_members,
    int           num_members,
    int           point_count,
    double       *max_dists_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;

    double max_dist = 0.0;
    for (int i = 0; i < num_members; ++i) {
        int m = cluster_members[i];
        double dx = points_x[idx] - points_x[m];
        double dy = points_y[idx] - points_y[m];
        double dist = sqrt(dx * dx + dy * dy);
        if (dist > max_dist) max_dist = dist;
    }
    max_dists_out[idx] = max_dist;
}

// ---------------------------------------------------------------------------
// CUDA cleanup
// ---------------------------------------------------------------------------
static void cleanup_cuda()
{
    if (!cuda_ok) return;
    cudaSetDevice(my_cuda_device);
    if (d_points_x)    { cudaFree(d_points_x);    d_points_x = nullptr; }
    if (d_points_y)    { cudaFree(d_points_y);    d_points_y = nullptr; }
    if (d_max_dists)   { cudaFree(d_max_dists);   d_max_dists = nullptr; }
    if (d_cluster_mem) { cudaFree(d_cluster_mem); d_cluster_mem = nullptr; }
    delete[] h_max_dists;  h_max_dists = nullptr;
    d_capacity = 0;
    cuda_ok = false;
}

// ---------------------------------------------------------------------------
// CUDA initialization – allocate device & host staging memory
// ---------------------------------------------------------------------------
static bool init_cuda(int n)
{
    if (cuda_ok && d_capacity >= n) return true;

    cleanup_cuda();

    // Ensure we're on the assigned device (cudaSetDevice is thread-local in CUDA 12+)
    cudaSetDevice(my_cuda_device);

    cudaError_t err;

    err = cudaMalloc(&d_points_x, n * sizeof(double));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_points_y, n * sizeof(double));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_max_dists, n * sizeof(double));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_cluster_mem, n * sizeof(int));
    if (err != cudaSuccess) return false;

    h_max_dists = new double[n];
    d_capacity = n;
    cuda_ok = true;
    return true;
}

// ---------------------------------------------------------------------------
// Structure to represent a point in 2D space
// ---------------------------------------------------------------------------
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ---------------------------------------------------------------------------
// Generate synthetic 2D point data in clusters
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Calculate Euclidean distance between two points
// ---------------------------------------------------------------------------
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Find the closest unclustered point to the current cluster.
// Uses CUDA for GPU-accelerated distance computation.
// Returns -1 if no such point exists.
// ---------------------------------------------------------------------------
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    const int num_members = static_cast<int>(cluster_members.size());

    // --- CUDA accelerated path ----------------------------------------------
    // GPU is a shared resource; protect with critical section.  Both the
    // kernel launch and the host-side reduction happen inside the critical
    // section to avoid races on h_max_dists.
    if (cuda_ok && num_members > 0) {
        int   closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        #pragma omp critical (cuda)
        {
            // cudaSetDevice is thread-local in CUDA 12+; ensure this thread
            // operates on the correct device (the one where memory was
            // allocated).
            int dev = 0;
            if (cudaGetDevice(&dev) == cudaSuccess && dev != my_cuda_device)
                cudaSetDevice(my_cuda_device);

            cudaMemcpy(d_cluster_mem, cluster_members.data(),
                       num_members * sizeof(int), cudaMemcpyHostToDevice);

            const int block_size = 256;
            const int grid_size  = (point_count + block_size - 1) / block_size;
            compute_max_dists_kernel<<<grid_size, block_size>>>(
                d_points_x, d_points_y, d_cluster_mem, num_members,
                point_count, d_max_dists);
            cudaDeviceSynchronize();

            cudaMemcpy(h_max_dists, d_max_dists,
                       point_count * sizeof(double), cudaMemcpyDeviceToHost);

            // Reduction inside critical section – h_max_dists is stable here
            for (int candidate = 0; candidate < point_count; ++candidate) {
                if (clustered[candidate] || in_cluster[candidate]) continue;
                double md = h_max_dists[candidate];
                if (md < threshold && md < min_diameter) {
                    min_diameter = md;
                    closest_point = candidate;
                }
            }
        } // #pragma omp critical (cuda)

        return closest_point;
    }

    // --- CPU / OpenMP accelerated path --------------------------------------
    std::vector<double> all_max_dists(point_count,
                                      std::numeric_limits<double>::max());

    #pragma omp parallel for schedule(guided)
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        double max_dist = 0.0;
        for (int i = 0; i < num_members; ++i) {
            const int member = cluster_members[i];
            double dx = points[candidate].x - points[member].x;
            double dy = points[candidate].y - points[member].y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist > max_dist) max_dist = dist;
        }
        all_max_dists[candidate] = max_dist;
    }

    // Sequential reduction
    int   closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;
        double md = all_max_dists[candidate];
        if (md < threshold && md < min_diameter) {
            min_diameter = md;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// ---------------------------------------------------------------------------
// Generate a candidate cluster starting from a seed point
// ---------------------------------------------------------------------------
int generateCandidateCluster(const int seed_point,
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
        const int closest = findClosestPoint(members, clustered, in_cluster, points,
                                             threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    int sz = static_cast<int>(members.size());
    if (cluster_members) {
        *cluster_members = std::move(members);
    }
    return sz;
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm – hybrid MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;

    // Main clustering loop – synchronised across ranks via MPI broadcasts
    while (true) {
        // ---- 1.  Build list of unclustered indices (identical on all ranks) ----
        std::vector<int> unclustered_indices;
        unclustered_indices.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) unclustered_indices.push_back(i);
        }
        const int num_unclustered = static_cast<int>(unclustered_indices.size());
        if (num_unclustered == 0) break;

        // ---- 2.  Divide seeds among MPI ranks --------------------------------
        int base      = num_unclustered / mpi_size;
        int remainder = num_unclustered % mpi_size;
        int my_start  = mpi_rank * base + std::min(mpi_rank, remainder);
        int my_count  = base + (mpi_rank < remainder ? 1 : 0);

        // ---- 3.  Each rank evaluates its seeds (OpenMP + CUDA) ---------------
        int local_best_card = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        #pragma omp parallel
        {
            int   thr_best_card = -1;
            int   thr_best_seed = -1;
            std::vector<int> thr_best_members;

            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < my_count; ++i) {
                const int seed = unclustered_indices[my_start + i];
                std::vector<int> cand_members;
                const int card = generateCandidateCluster(seed, clustered, points,
                                                          threshold, N, &cand_members);
                if (card > thr_best_card) {
                    thr_best_card = card;
                    thr_best_seed = seed;
                    thr_best_members = std::move(cand_members);
                }
            }

            #pragma omp critical
            {
                if (thr_best_card > local_best_card) {
                    local_best_card = thr_best_card;
                    local_best_seed = thr_best_seed;
                    local_best_members = std::move(thr_best_members);
                }
            }
        }

        // ---- 4.  MPI: gather best results to rank 0 --------------------------
        int sendbuf[2] = { local_best_card, local_best_seed };
        std::vector<int> recvbuf(2 * mpi_size);
        MPI_Gather(sendbuf, 2, MPI_INT,
                   recvbuf.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

        // ---- 5.  Rank 0 picks the globally best cluster ----------------------
        int   global_best_card = -1;
        int   global_best_seed = -1;
        int   global_best_rank = -1;
        if (mpi_rank == 0) {
            for (int r = 0; r < mpi_size; ++r) {
                if (recvbuf[2 * r] > global_best_card) {
                    global_best_card = recvbuf[2 * r];
                    global_best_seed = recvbuf[2 * r + 1];
                    global_best_rank = r;
                }
            }
        }

        // ---- 6.  Broadcast global best info to all ranks ---------------------
        int best_info[3] = { global_best_seed, global_best_card, global_best_rank };
        MPI_Bcast(best_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        global_best_seed  = best_info[0];
        global_best_card  = best_info[1];
        global_best_rank  = best_info[2];

        if (global_best_seed < 0 || global_best_card <= 0) break;

        // ---- 7.  Rank owning best sends members to rank 0 --------------------
        // Avoid self-sends (rank 0 already holds its own members).
        std::vector<int> global_members;
        if (mpi_rank == 0) {
            if (global_best_rank == 0) {
                global_members = std::move(local_best_members);
            } else {
                int count;
                MPI_Recv(&count, 1, MPI_INT, global_best_rank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_members.resize(count);
                MPI_Recv(global_members.data(), count, MPI_INT,
                         global_best_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (mpi_rank == global_best_rank) {
            int count = static_cast<int>(local_best_members.size());
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(local_best_members.data(), count, MPI_INT,
                     0, 1, MPI_COMM_WORLD);
        }

        // ---- 8.  Broadcast cluster members to all ranks ----------------------
        int bcast_count = (mpi_rank == 0) ? static_cast<int>(global_members.size()) : 0;
        MPI_Bcast(&bcast_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank != 0) global_members.resize(bcast_count);
        MPI_Bcast(global_members.data(), bcast_count, MPI_INT, 0, MPI_COMM_WORLD);

        // ---- 9.  All ranks update their local clustered array ----------------
        for (int i = 0; i < bcast_count; ++i) {
            clustered[global_members[i]] = true;
        }

        // ---- 10. Rank 0 records the cluster ---------------------------------
        if (mpi_rank == 0) {
            Cluster cl;
            cl.seed_point = global_best_seed;
            cl.members = global_members;
            clusters.push_back(std::move(cl));
        }
    }

    // Broadcast final cluster list from rank 0 to all ranks (for validation)
    int final_count = static_cast<int>(clusters.size());
    MPI_Bcast(&final_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        clusters.resize(final_count);
    }
    for (int ci = 0; ci < final_count; ++ci) {
        int sz = (mpi_rank == 0) ? static_cast<int>(clusters[ci].members.size()) : 0;
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank != 0) clusters[ci].members.resize(sz);
        MPI_Bcast(clusters[ci].members.data(), sz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&clusters[ci].seed_point, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation: check that clusters satisfy the QT clustering properties
// ---------------------------------------------------------------------------
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
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
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

// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // -----------------------------------------------------------------------
    // Rank 0 parses arguments, then broadcasts to all ranks
    // -----------------------------------------------------------------------
    int    num_points  = 1000;
    double threshold   = 2.0;
    bool   validate    = false;
    bool   printResults = false;
    bool   parse_ok    = true;

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
                parse_ok = false;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_ok = false;
            }
        }
    }

    // Broadcast args to all ranks
    int args_int[3]  = { num_points, static_cast<int>(validate), static_cast<int>(printResults) };
    double args_dbl  = threshold;
    MPI_Bcast(&parse_ok, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!parse_ok) { MPI_Finalize(); return 1; }
    MPI_Bcast(args_int, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&args_dbl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    num_points  = args_int[0];
    validate    = args_int[1] != 0;
    printResults = args_int[2] != 0;
    threshold   = args_dbl;

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    // -----------------------------------------------------------------------
    // Print banner from rank 0 only
    // -----------------------------------------------------------------------
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        #pragma omp parallel
        {
            #pragma omp single
            printf("OpenMP threads per rank: %d\n", omp_get_num_threads());
        }
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        int cuda_device_count = 0;
        cudaGetDeviceCount(&cuda_device_count);
        printf("CUDA devices available: %d\n", cuda_device_count);
    }

    // -----------------------------------------------------------------------
    // Each rank generates data (identical due to fixed seed)
    // -----------------------------------------------------------------------
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // -----------------------------------------------------------------------
    // Upload GPU point data on each rank
    // -----------------------------------------------------------------------
    {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count > 0) {
            int dev = mpi_rank % dev_count;
            cudaSetDevice(dev);
            my_cuda_device = dev;
            bool ok = init_cuda(num_points);
            if (ok && cuda_ok) {
                std::vector<double> px(num_points), py(num_points);
                for (int i = 0; i < num_points; ++i) {
                    px[i] = points[i].x;
                    py[i] = points[i].y;
                }
                cudaMemcpy(d_points_x, px.data(), num_points * sizeof(double),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_points_y, py.data(), num_points * sizeof(double),
                           cudaMemcpyHostToDevice);
                if (mpi_rank == 0)
                    printf("CUDA initialised (device %d, %d pts)\n", dev, num_points);
            } else {
                if (mpi_rank == 0)
                    printf("CUDA init failed - using CPU+OpenMP only\n");
            }
        } else {
            if (mpi_rank == 0)
                printf("No CUDA devices found - using CPU+OpenMP only\n");
        }
    }

    // -----------------------------------------------------------------------
    // Time and run the clustering
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();

    if (mpi_rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start);

        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }

    int total_clustered = 0;
    int max_cluster_size = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const int sz = static_cast<int>(clusters[i].members.size());
        total_clustered += sz;
        if (sz > max_cluster_size) max_cluster_size = sz;
    }

    if (mpi_rank == 0) {
        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        auto ct = std::chrono::duration_cast<std::chrono::milliseconds>(
            cluster_end - cluster_start);
        const double time_sec = ct.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // -----------------------------------------------------------------------
    // Print results for external validation (rank 0 only)
    // -----------------------------------------------------------------------
    if (printResults && mpi_rank == 0) {
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

    // -----------------------------------------------------------------------
    // Validation (rank 0 only – clusters match across ranks)
    // -----------------------------------------------------------------------
    if (validate) {
        int valid_int = 0;
        if (mpi_rank == 0) {
            const bool valid = validateClusters(clusters, points, threshold);
            valid_int = valid ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cleanup_cuda();
        MPI_Finalize();
        return (valid_int == 1) ? 0 : 1;
    }

    cleanup_cuda();
    MPI_Finalize();
    return 0;
}

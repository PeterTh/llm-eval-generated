// QT Clustering Benchmark - Hybrid MPI/OpenMP/CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Hybrid parallelization strategy:
//   MPI:    Distribute seed evaluation across ranks within each outer iteration.
//           Allreduces determine the global best seed; Bcast shares cluster members.
//   OpenMP: Parallelize seed evaluation within each MPI rank. Each thread owns
//           its own GPU device buffers so CUDA kernels can run concurrently.
//   CUDA:   Accelerate findClosestPoint — each thread computes the max distance
//           from one candidate point to all current cluster members.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
// Each thread handles one candidate point.  It computes the maximum distance
// from that candidate to every current cluster member.  Candidates that are
// already clustered or already inside the growing cluster are marked invalid
// with -1.0.  Valid candidates whose max distance exceeds the threshold are
// also marked invalid.
__global__ void maxDistKernel(
    const double* __restrict__ px, const double* __restrict__ py,
    const int* __restrict__ members, int n_members,
    const unsigned char* __restrict__ clustered,
    const unsigned char* __restrict__ in_cluster,
    int n_points, double threshold,
    double* __restrict__ results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    if (clustered[idx] || in_cluster[idx]) {
        results[idx] = -1.0;
        return;
    }

    double max_d = 0.0;
    for (int i = 0; i < n_members; i++) {
        int m = members[i];
        double dx = px[idx] - px[m];
        double dy = py[idx] - py[m];
        max_d = fmax(max_d, sqrt(dx * dx + dy * dy));
    }

    results[idx] = (max_d < threshold) ? max_d : -1.0;
}

// ---------------------------------------------------------------------------
// GPU memory management — one context per OpenMP thread
// ---------------------------------------------------------------------------
struct DeviceContext {
    double*        d_px        = nullptr;
    double*        d_py        = nullptr;
    unsigned char* d_clustered = nullptr;
    unsigned char* d_in_cluster= nullptr;
    double*        d_max_dists = nullptr;
    int*           d_members   = nullptr;
    int            capacity    = 0;

    void init(int n, const Point* h_points, const std::vector<bool>& h_clustered) {
        capacity = n;
        cudaMalloc(&d_px,           n * sizeof(double));
        cudaMalloc(&d_py,           n * sizeof(double));
        cudaMalloc(&d_clustered,    n * sizeof(unsigned char));
        cudaMalloc(&d_in_cluster,   n * sizeof(unsigned char));
        cudaMalloc(&d_max_dists,    n * sizeof(double));
        cudaMalloc(&d_members,      n * sizeof(int));

        // Transfer point coordinates
        std::vector<double> h_px(n), h_py(n);
        for (int i = 0; i < n; i++) {
            h_px[i] = h_points[i].x;
            h_py[i] = h_points[i].y;
        }
        cudaMemcpy(d_px, h_px.data(), n * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_py, h_py.data(), n * sizeof(double),
                   cudaMemcpyHostToDevice);

        // Transfer clustered flags
        std::vector<unsigned char> h_cc(n);
        for (int i = 0; i < n; i++) h_cc[i] = h_clustered[i] ? 1 : 0;
        cudaMemcpy(d_clustered, h_cc.data(), n * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
    }

    void updateClustered(const std::vector<bool>& h_clustered, int n) {
        std::vector<unsigned char> h_cc(n);
        for (int i = 0; i < n; i++) h_cc[i] = h_clustered[i] ? 1 : 0;
        cudaMemcpy(d_clustered, h_cc.data(), n * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
    }

    ~DeviceContext() {
        if (d_px)          cudaFree(d_px);
        if (d_py)          cudaFree(d_py);
        if (d_clustered)   cudaFree(d_clustered);
        if (d_in_cluster)  cudaFree(d_in_cluster);
        if (d_max_dists)   cudaFree(d_max_dists);
        if (d_members)     cudaFree(d_members);
    }
};

// ---------------------------------------------------------------------------
// CUDA-accelerated findClosestPoint
// ---------------------------------------------------------------------------
int findClosestPointCUDA(
    DeviceContext& dev,
    const std::vector<int>& members,
    const std::vector<bool>& in_cluster,
    double threshold, int n_points)
{
    // Upload in_cluster flags
    std::vector<unsigned char> h_ic(n_points);
    for (int i = 0; i < n_points; i++)
        h_ic[i] = in_cluster[i] ? 1 : 0;
    cudaMemcpy(dev.d_in_cluster, h_ic.data(),
               n_points * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Upload cluster member indices
    cudaMemcpy(dev.d_members, members.data(),
               members.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel — one thread per candidate
    const int block_size = 256;
    const int grid_size  = (n_points + block_size - 1) / block_size;
    maxDistKernel<<<grid_size, block_size>>>(
        dev.d_px, dev.d_py, dev.d_members,
        static_cast<int>(members.size()),
        dev.d_clustered, dev.d_in_cluster,
        n_points, threshold, dev.d_max_dists);
    cudaDeviceSynchronize();

    // Download results and find best candidate on host
    std::vector<double> h_md(n_points);
    cudaMemcpy(h_md.data(), dev.d_max_dists,
               n_points * sizeof(double), cudaMemcpyDeviceToHost);

    int best = -1;
    double best_d = threshold;
    for (int i = 0; i < n_points; i++) {
        if (h_md[i] >= 0.0 && h_md[i] < best_d) {
            best_d = h_md[i];
            best = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// CUDA-accelerated generateCandidateCluster
// ---------------------------------------------------------------------------
int generateCandidateClusterCUDA(
    DeviceContext& dev,
    int seed_point,
    double threshold, int n_points,
    std::vector<int>* cluster_members = nullptr)
{
    std::vector<bool> in_cluster(n_points, false);
    std::vector<int>  members;
    members.reserve(n_points);

    in_cluster[seed_point] = true;
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < n_points) {
        int closest = findClosestPointCUDA(dev, members, in_cluster,
                                           threshold, n_points);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------------
// Data generation (sequential — not the bottleneck)
// ---------------------------------------------------------------------------
void generateSyntheticData(std::vector<Point>& points, const int N,
                           unsigned int seed = 42)
{
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R      = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

        if (group_cnt > (N - count)) group_cnt = N - count;

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r    = frand() * R;
            const double dx   = (2.0 * frand() - 1.0) * r;
            const double dy   = std::sqrt(r * r - dx * dx) * sign;
            const double x    = cntr_x + dx;
            const double y    = cntr_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;

            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering — MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(
    const std::vector<Point>& points, double threshold,
    int mpi_rank, int mpi_size)
{
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int>  unclustered_indices;
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; i++)
        unclustered_indices.push_back(i);

    // ---- main clustering loop ----
    while (!unclustered_indices.empty()) {
        int n_seeds = static_cast<int>(unclustered_indices.size());

        // Distribute seed indices across MPI ranks (block distribution)
        int my_start = (mpi_rank * n_seeds) / mpi_size;
        int my_end   = ((mpi_rank + 1) * n_seeds) / mpi_size;

        int local_max_card    = -1;
        int local_best_seed   = -1;
        std::vector<int> local_best_members;

        // OpenMP: each thread gets its own GPU buffers so kernels run
        // concurrently without contention.
        #pragma omp parallel
        {
            DeviceContext dev;
            dev.init(N, points.data(), clustered);

            int thread_max_card  = -1;
            int thread_best_seed = -1;
            std::vector<int> thread_best_members;

            #pragma omp for schedule(dynamic, 4)
            for (int i = my_start; i < my_end; i++) {
                int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                std::vector<int> candidate_members;
                int cardinality = generateCandidateClusterCUDA(
                    dev, seed, threshold, N,
                    &candidate_members);

                // Tie-break: smallest seed index wins
                if (cardinality > thread_max_card ||
                    (cardinality == thread_max_card &&
                     seed < thread_best_seed)) {
                    thread_max_card  = cardinality;
                    thread_best_seed = seed;
                    thread_best_members = std::move(candidate_members);
                }
            }

            // Reduce thread-local bests into rank-local best
            #pragma omp critical
            {
                if (thread_max_card > local_max_card ||
                    (thread_max_card == local_max_card &&
                     thread_best_seed < local_best_seed)) {
                    local_max_card    = thread_max_card;
                    local_best_seed   = thread_best_seed;
                    local_best_members = std::move(thread_best_members);
                }
            }
        } // OpenMP parallel region ends — per-thread GPU memory freed

        // ---- MPI: find global best cardinality ----
        int global_max_card;
        MPI_Allreduce(&local_max_card, &global_max_card, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_card <= 0) break;

        // Among ranks with the max cardinality, pick the smallest seed
        int my_seed_val = (local_max_card == global_max_card)
                              ? local_best_seed : INT_MAX;
        int global_best_seed;
        MPI_Allreduce(&my_seed_val, &global_best_seed, 1,
                      MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Determine which MPI rank owns the best seed
        int my_root = (local_best_seed == global_best_seed)
                          ? mpi_rank : -1;
        int root_rank;
        MPI_Allreduce(&my_root, &root_rank, 1,
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Broadcast cluster members from the root rank
        int members_count = 0;
        if (mpi_rank == root_rank)
            members_count = static_cast<int>(local_best_members.size());
        MPI_Bcast(&members_count, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

        std::vector<int> best_cluster_members;
        if (mpi_rank == root_rank) {
            best_cluster_members = std::move(local_best_members);
        } else {
            best_cluster_members.resize(members_count);
        }
        MPI_Bcast(best_cluster_members.data(), members_count, MPI_INT,
                  root_rank, MPI_COMM_WORLD);

        // ---- commit the cluster ----
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members    = std::move(best_cluster_members);
        clusters.push_back(cluster);

        for (int m : cluster.members)
            clustered[m] = true;

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation (sequential — only used with -v)
// ---------------------------------------------------------------------------
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold)
{
    bool valid = true;
    printf("Validating clusters:\n");

    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(
                    points[cluster.members[i]], points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10)
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point,
                   max_diameter);

        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d in clusters %d and %zu\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i)
        if (membership[i] >= 0) clustered_count++;

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count,
           points.size() - clustered_count);

    return valid;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int    num_points = 1000;
    double threshold  = 2.0;
    bool   validate   = false;
    bool   printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            num_points = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            threshold = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0)
            printf("Error: Invalid parameters (num_points=%d, "
                   "threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate data on rank 0, then broadcast to all ranks
    std::vector<Point> points(num_points);
    if (mpi_rank == 0)
        generateSyntheticData(points, num_points);

    // Point is {double x, double y} — contiguous, so broadcast as doubles
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- clustering ----
    double t0 = MPI_Wtime();
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, mpi_rank, mpi_size);
    double local_cluster_time = MPI_Wtime() - t0;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    // ---- results (rank 0 only) ----
    if (mpi_rank == 0) {
        printf("Clustering time: %.3f s\n", cluster_time);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0, max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }

        double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        if (cluster_time > 0.0)
            printf("Performance: %.1f clusters/s, %.1f points/s\n",
                   clusters.size() / cluster_time,
                   num_points / cluster_time);

        if (printResults) {
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c)
                for (size_t i = 0; i < clusters[c].members.size(); ++i)
                    membership[clusters[c].members[i]] = static_cast<int>(c);
            for (int m : membership)
                membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            bool ok = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return 0;
}

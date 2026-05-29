// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   MPI    - distribute seed trials across ranks; collective reduce for best cluster
//   OpenMP - parallelize seed trials within each rank
//   CUDA   - accelerate max-distance computation (inner loop of findClosestPoint)

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

// ---------------------------------------------------------------------------
// CUDA kernel: compute max distance from each candidate to all cluster members
// ---------------------------------------------------------------------------
__global__ void maxDistKernel(
    const double* d_candX, const double* d_candY,
    const int*     d_candIdx,
    const double* d_memX,  const double* d_memY,
    double* d_maxDist,
    int num_candidates, int num_members)
{
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_candidates) return;

    double cx = d_candX[cid];
    double cy = d_candY[cid];
    double md = 0.0;
    for (int m = 0; m < num_members; ++m) {
        double dx = cx - d_memX[m];
        double dy = cy - d_memY[m];
        double d = sqrt(dx * dx + dy * dy);
        if (d > md) md = d;
    }
    d_maxDist[cid] = md;
}

// CUDA-accelerated findClosestPoint
static int findClosestPointCUDA(
    const std::vector<int>& cluster_members,
    const std::vector<bool>& clustered,
    const std::vector<bool>& in_cluster,
    const std::vector<Point>& points,
    const double threshold,
    const int point_count)
{
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    // Collect valid candidates
    std::vector<int> cand_idx;
    cand_idx.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        if (!clustered[i] && !in_cluster[i])
            cand_idx.push_back(i);
    }
    if (cand_idx.empty()) return -1;

    int ncand = static_cast<int>(cand_idx.size());
    int nmem  = static_cast<int>(cluster_members.size());

    // Gather candidate and member coordinates
    std::vector<double> h_cx(ncand), h_cy(ncand);
    for (int i = 0; i < ncand; ++i) {
        h_cx[i] = points[cand_idx[i]].x;
        h_cy[i] = points[cand_idx[i]].y;
    }
    std::vector<double> h_mx(nmem), h_my(nmem);
    for (int i = 0; i < nmem; ++i) {
        h_mx[i] = points[cluster_members[i]].x;
        h_my[i] = points[cluster_members[i]].y;
    }

    // Device buffers
    double *d_cx, *d_cy, *d_mx, *d_my, *d_md;
    int    *d_ci;
    cudaMalloc(&d_cx, ncand * sizeof(double));
    cudaMalloc(&d_cy, ncand * sizeof(double));
    cudaMalloc(&d_ci, ncand * sizeof(int));
    cudaMalloc(&d_mx, nmem  * sizeof(double));
    cudaMalloc(&d_my, nmem  * sizeof(double));
    cudaMalloc(&d_md, ncand * sizeof(double));

    cudaMemcpy(d_cx, h_cx.data(), ncand * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, h_cy.data(), ncand * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, cand_idx.data(), ncand * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mx, h_mx.data(), nmem  * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_my, h_my.data(), nmem  * sizeof(double), cudaMemcpyHostToDevice);

    int blk = 256;
    int grid = (ncand + blk - 1) / blk;
    maxDistKernel<<<grid, blk>>>(d_cx, d_cy, d_ci, d_mx, d_my, d_md, ncand, nmem);

    std::vector<double> h_md(ncand);
    cudaMemcpy(h_md.data(), d_md, ncand * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_ci);
    cudaFree(d_mx); cudaFree(d_my); cudaFree(d_md);

    for (int i = 0; i < ncand; ++i) {
        if (h_md[i] < threshold && h_md[i] < min_diameter) {
            min_diameter = h_md[i];
            closest_point = cand_idx[i];
        }
    }
    return closest_point;
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

// Generate a candidate cluster starting from a seed point (CUDA-accelerated)
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
    
    // Iteratively add closest points using CUDA kernel
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointCUDA(members, clustered, in_cluster, points,
                                                 threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    if (cluster_members) {
        *cluster_members = members;
    }
    return static_cast<int>(members.size());
}

// Hybrid MPI + OpenMP + CUDA main clustering
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    for (int i = 0; i < N; ++i)
        unclustered_indices.push_back(i);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    while (!unclustered_indices.empty()) {
        // --- distribute seeds across MPI ranks ---
        int n_seeds = static_cast<int>(unclustered_indices.size());
        std::vector<int> local_seeds;
        if (mpi_size == 1) {
            local_seeds = unclustered_indices;
        } else {
            // Scatter unclustered indices to all ranks
            std::vector<int> all_counts(mpi_size), all_disps(mpi_size);
            int base = n_seeds / mpi_size;
            int rem  = n_seeds % mpi_size;
            int offset = 0;
            for (int r = 0; r < mpi_size; ++r) {
                int cnt = base + (r < rem ? 1 : 0);
                all_counts[r] = cnt;
                all_disps[r]  = offset;
                offset += cnt;
            }
            std::vector<int> recv_buf(all_counts[mpi_rank]);
            MPI_Scatterv(unclustered_indices.data(), all_counts.data(),
                         all_disps.data(), MPI_INT,
                         recv_buf.data(), all_counts[mpi_rank],
                         MPI_INT, 0, MPI_COMM_WORLD);
            local_seeds = std::move(recv_buf);
        }

        // --- OpenMP: parallel seed trials on each rank ---
        int local_best_card  = -1;
        int local_best_seed  = -1;
        std::vector<int> local_best_members;

#pragma omp parallel
        {
            int       my_card  = -1;
            int       my_seed  = -1;
            std::vector<int> my_members;

#pragma omp for nowait
            for (int si = 0; si < static_cast<int>(local_seeds.size()); ++si) {
                int seed = local_seeds[si];
                if (clustered[seed]) continue;

                std::vector<int> cm;
                int card = generateCandidateCluster(seed, clustered, points,
                                                    threshold, N, &cm);
                if (card <= my_card) continue;

                // Check whether this seed is still unclustered (atomic guard)
                bool still = true;
#pragma omp critical
                {
                    if (clustered[seed]) {
                        still = false;
                    }
                }
                if (!still) continue;

                if (card > my_card) {
                    my_card    = card;
                    my_seed    = seed;
                    my_members = std::move(cm);
                }
            }

            // Reduction into thread-local best
#pragma omp critical
            {
                if (my_card > local_best_card) {
                    local_best_card  = my_card;
                    local_best_seed  = my_seed;
                    local_best_members = std::move(my_members);
                }
            }
        }

        // --- MPI: reduce to global best ---
        int global_best_card = -1;
        int global_best_seed = -1;
        std::vector<int> global_best_members;

        if (mpi_size == 1) {
            global_best_card  = local_best_card;
            global_best_seed  = local_best_seed;
            global_best_members = local_best_members;
        } else {
            // Pack: (cardinality, seed, member_count, members...)
            int local_mcount = local_best_members.empty() ? 0
                                     : static_cast<int>(local_best_members.size());
            int local_nints  = 3 + local_mcount;
            std::vector<int> send_buf(local_nints);
            send_buf[0] = local_best_card;
            send_buf[1] = local_best_seed;
            send_buf[2] = local_mcount;
            for (int i = 0; i < local_mcount; ++i)
                send_buf[3 + i] = local_best_members[i];

            // Gather member counts
            std::vector<int> all_mcounts(mpi_size);
            MPI_Allgather(&local_mcount, 1, MPI_INT,
                          all_mcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // Compute receive sizes / displacements
            std::vector<int> recv_counts(mpi_size), recv_disps(mpi_size);
            int r_off = 0;
            int total = 0;
            for (int r = 0; r < mpi_size; ++r) {
                int cnt = 3 + all_mcounts[r];
                recv_counts[r] = cnt;
                recv_disps[r]  = r_off;
                r_off += cnt;
                total += cnt;
            }

            std::vector<int> recv_all(total);
            MPI_Gatherv(send_buf.data(), local_nints, MPI_INT,
                        recv_all.data(), recv_counts.data(),
                        recv_disps.data(), MPI_INT, 0, MPI_COMM_WORLD);

            // Rank 0 finds global best
            if (mpi_rank == 0) {
                for (int r = 0; r < mpi_size; ++r) {
                    int off = recv_disps[r];
                    int card = recv_all[off];
                    int seed = recv_all[off + 1];
                    int mc   = recv_all[off + 2];
                    if (card > global_best_card) {
                        global_best_card  = card;
                        global_best_seed  = seed;
                        global_best_members.resize(mc);
                        for (int i = 0; i < mc; ++i)
                            global_best_members[i] = recv_all[off + 3 + i];
                    }
                }
            }

            // Broadcast the winner
            if (mpi_rank != 0) {
                global_best_card  = -1;
                global_best_seed  = -1;
            }
            MPI_Bcast(&global_best_card, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&global_best_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (global_best_card > 0) {
                int gmc = static_cast<int>(global_best_members.size());
                MPI_Bcast(&gmc, 1, MPI_INT, 0, MPI_COMM_WORLD);
                global_best_members.resize(gmc);
                MPI_Bcast(global_best_members.data(), gmc, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        // --- apply winning cluster (all ranks) ---
        if (global_best_seed >= 0 && global_best_card > 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members    = global_best_members;
            clusters.push_back(cluster);

            for (int m : global_best_members)
                clustered[m] = true;

            auto it = std::remove_if(unclustered_indices.begin(),
                                     unclustered_indices.end(),
                                     [&clustered](int idx){ return clustered[idx]; });
            unclustered_indices.erase(it, unclustered_indices.end());
        } else {
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
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Broadcast parameters
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (same seed on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
    
    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
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

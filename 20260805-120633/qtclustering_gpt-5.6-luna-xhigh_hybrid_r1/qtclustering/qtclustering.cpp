// QT Clustering Benchmark - Simplified Sequential Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
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

struct LocalBestCluster {
  int cardinality = 0;
  int seed = -1;
  std::vector<int> members;
};

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

// Each block evaluates one seed.  The cluster growth itself is sequential for
// a seed by definition, but all candidate points and all rank-local seeds are
// evaluated concurrently on the GPU.
__global__ void generateCandidateClustersKernel(
    const Point *points, const unsigned char *clustered, const int *seeds,
    const int point_count, const double threshold, unsigned char *in_cluster,
    int *members, int *member_counts) {
  const int seed_slot = static_cast<int>(blockIdx.x);
  if (seed_slot >= gridDim.x)
    return;

  const size_t base =
      static_cast<size_t>(seed_slot) * static_cast<size_t>(point_count);

  __shared__ double best_distances[CUDA_BLOCK_SIZE];
  __shared__ int best_candidates[CUDA_BLOCK_SIZE];
  __shared__ int finished;

  if (threadIdx.x == 0) {
    const int seed = seeds[seed_slot];
    member_counts[seed_slot] = 1;
    members[base] = seed;
    in_cluster[base + static_cast<size_t>(seed)] = 1;
    finished = 0;
  }
  __syncthreads();

  while (true) {
    const int current_count = member_counts[seed_slot];
    double local_best_distance = DBL_MAX;
    int local_best_candidate = -1;

    // The strided scan covers candidates in increasing order overall.
    // The reduction below explicitly resolves equal distances by the
    // smallest candidate index, matching the original serial loop.
    for (int candidate = static_cast<int>(threadIdx.x); candidate < point_count;
         candidate += CUDA_BLOCK_SIZE) {
      if (clustered[candidate] ||
          in_cluster[base + static_cast<size_t>(candidate)]) {
        continue;
      }

      double max_distance = 0.0;
      const Point candidate_point = points[candidate];
      for (int member_index = 0; member_index < current_count; ++member_index) {
        const Point member_point =
            points[members[base + static_cast<size_t>(member_index)]];
        const double dx = candidate_point.x - member_point.x;
        const double dy = candidate_point.y - member_point.y;
        const double current_distance = sqrt(dx * dx + dy * dy);
        if (current_distance > max_distance) {
          max_distance = current_distance;
        }
      }

      if (max_distance < threshold && max_distance < local_best_distance) {
        local_best_distance = max_distance;
        local_best_candidate = candidate;
      }
    }

    best_distances[threadIdx.x] = local_best_distance;
    best_candidates[threadIdx.x] = local_best_candidate;
    __syncthreads();

    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        const int other = threadIdx.x + offset;
        const int other_candidate = best_candidates[other];
        const bool other_is_better =
            other_candidate >= 0 &&
            (best_candidates[threadIdx.x] < 0 ||
             best_distances[other] < best_distances[threadIdx.x] ||
             (best_distances[other] == best_distances[threadIdx.x] &&
              other_candidate < best_candidates[threadIdx.x]));
        if (other_is_better) {
          best_distances[threadIdx.x] = best_distances[other];
          best_candidates[threadIdx.x] = other_candidate;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const int selected_candidate = best_candidates[0];
      if (selected_candidate < 0) {
        finished = 1;
      } else {
        const int write_index = member_counts[seed_slot];
        members[base + static_cast<size_t>(write_index)] = selected_candidate;
        in_cluster[base + static_cast<size_t>(selected_candidate)] = 1;
        member_counts[seed_slot] = write_index + 1;
      }
    }
    __syncthreads();

    if (finished)
      break;
  }
}

void checkCuda(const cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return;

  std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
               cudaGetErrorString(status));
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::abort();
}

bool betterCluster(const int cardinality, const int seed,
                   const int other_cardinality, const int other_seed) {
  return cardinality > other_cardinality ||
         (cardinality == other_cardinality && seed < other_seed);
}

class CudaClusterEngine {
public:
  explicit CudaClusterEngine(const std::vector<Point> &points)
      : point_count_(static_cast<int>(points.size())) {
    static_assert(sizeof(Point) == sizeof(double) * 2,
                  "Point must be a tightly packed pair of doubles");

    const size_t point_bytes = points.size() * sizeof(Point);
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&device_points_), point_bytes),
        "cudaMalloc(points)");
    checkCuda(cudaMemcpy(device_points_, points.data(), point_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(points)");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&device_clustered_),
                         points.size() * sizeof(unsigned char)),
              "cudaMalloc(clustered)");
  }

  ~CudaClusterEngine() {
    releaseBatchBuffers();
    if (device_clustered_ != nullptr)
      cudaFree(device_clustered_);
    if (device_points_ != nullptr)
      cudaFree(device_points_);
  }

  CudaClusterEngine(const CudaClusterEngine &) = delete;
  CudaClusterEngine &operator=(const CudaClusterEngine &) = delete;

  LocalBestCluster evaluate(const std::vector<int> &seed_indices,
                            const std::vector<unsigned char> &clustered,
                            const double threshold) {
    LocalBestCluster best;
    if (seed_indices.empty())
      return best;

    checkCuda(cudaMemcpy(device_clustered_, clustered.data(),
                         clustered.size() * sizeof(unsigned char),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(clustered)");
    ensureBatchCapacity(seed_indices.size());

    const int openmp_threads = std::max(1, omp_get_max_threads());
    for (size_t batch_begin = 0; batch_begin < seed_indices.size();
         batch_begin += batch_capacity_) {
      const size_t batch_size =
          std::min(batch_capacity_, seed_indices.size() - batch_begin);

      checkCuda(cudaMemcpy(device_seeds_, seed_indices.data() + batch_begin,
                           batch_size * sizeof(int), cudaMemcpyHostToDevice),
                "cudaMemcpy(seeds)");
      checkCuda(cudaMemset(device_in_cluster_, 0,
                           batch_size * static_cast<size_t>(point_count_)),
                "cudaMemset(in_cluster)");

      generateCandidateClustersKernel<<<static_cast<unsigned int>(batch_size),
                                        CUDA_BLOCK_SIZE>>>(
          device_points_, device_clustered_, device_seeds_, point_count_,
          threshold, device_in_cluster_, device_members_,
          device_member_counts_);
      checkCuda(cudaGetLastError(), "generateCandidateClustersKernel launch");
      checkCuda(cudaDeviceSynchronize(), "generateCandidateClustersKernel");

      std::vector<int> cardinalities(batch_size);
      checkCuda(cudaMemcpy(cardinalities.data(), device_member_counts_,
                           batch_size * sizeof(int), cudaMemcpyDeviceToHost),
                "cudaMemcpy(member_counts)");

      struct BatchChoice {
        int cardinality = 0;
        int seed = INT_MAX;
        int slot = -1;
      };
      std::vector<BatchChoice> thread_choices(openmp_threads);

// OpenMP performs the rank-local reduction while the next batch's
// data remains on the host.  The explicit tie-break retains QT's
// first-seed semantics regardless of scheduling.
#pragma omp parallel
      {
        const int thread_id = omp_get_thread_num();
        BatchChoice &thread_choice = thread_choices[thread_id];

#pragma omp for schedule(static)
        for (int slot = 0; slot < static_cast<int>(batch_size); ++slot) {
          const int seed =
              seed_indices[batch_begin + static_cast<size_t>(slot)];
          if (betterCluster(cardinalities[slot], seed,
                            thread_choice.cardinality, thread_choice.seed)) {
            thread_choice.cardinality = cardinalities[slot];
            thread_choice.seed = seed;
            thread_choice.slot = slot;
          }
        }
      }

      BatchChoice batch_best;
      for (const BatchChoice &choice : thread_choices) {
        if (betterCluster(choice.cardinality, choice.seed,
                          batch_best.cardinality, batch_best.seed)) {
          batch_best = choice;
        }
      }

      if (betterCluster(batch_best.cardinality, batch_best.seed,
                        best.cardinality, best.seed)) {
        best.cardinality = batch_best.cardinality;
        best.seed = batch_best.seed;
        best.members.resize(static_cast<size_t>(best.cardinality));
        const size_t member_offset = static_cast<size_t>(batch_best.slot) *
                                     static_cast<size_t>(point_count_);
        checkCuda(cudaMemcpy(best.members.data(),
                             device_members_ + member_offset,
                             best.members.size() * sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(best_members)");
      }
    }

    return best;
  }

private:
  void releaseBatchBuffers() {
    if (device_seeds_ != nullptr)
      cudaFree(device_seeds_);
    if (device_in_cluster_ != nullptr)
      cudaFree(device_in_cluster_);
    if (device_members_ != nullptr)
      cudaFree(device_members_);
    if (device_member_counts_ != nullptr)
      cudaFree(device_member_counts_);
    device_seeds_ = nullptr;
    device_in_cluster_ = nullptr;
    device_members_ = nullptr;
    device_member_counts_ = nullptr;
    batch_capacity_ = 0;
  }

  void ensureBatchCapacity(const size_t requested) {
    if (requested <= batch_capacity_)
      return;

    const size_t bytes_per_seed = static_cast<size_t>(point_count_) *
                                      (sizeof(unsigned char) + sizeof(int)) +
                                  sizeof(int);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;

    size_t capacity = requested;
    if (bytes_per_seed > 0) {
      const size_t memory_limited_capacity =
          (free_bytes / 4 * 3) / bytes_per_seed;
      if (memory_limited_capacity > 0) {
        capacity = std::min(capacity, memory_limited_capacity);
      } else {
        capacity = 1;
      }
    }

    while (capacity > 0) {
      releaseBatchBuffers();

      const size_t point_slots = capacity * static_cast<size_t>(point_count_);
      cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&device_seeds_),
                                      capacity * sizeof(int));
      if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void **>(&device_in_cluster_),
                            point_slots * sizeof(unsigned char));
      }
      if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void **>(&device_members_),
                            point_slots * sizeof(int));
      }
      if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void **>(&device_member_counts_),
                            capacity * sizeof(int));
      }

      if (status == cudaSuccess) {
        batch_capacity_ = capacity;
        return;
      }

      releaseBatchBuffers();
      capacity /= 2;
    }

    checkCuda(cudaErrorMemoryAllocation, "allocating CUDA cluster buffers");
  }

  int point_count_ = 0;
  Point *device_points_ = nullptr;
  unsigned char *device_clustered_ = nullptr;
  int *device_seeds_ = nullptr;
  unsigned char *device_in_cluster_ = nullptr;
  int *device_members_ = nullptr;
  int *device_member_counts_ = nullptr;
  size_t batch_capacity_ = 0;
};

} // namespace

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point> &points, const int N,
                           unsigned int seed = 42) {
  auto frand = [&seed]() mutable {
    return rand_r(&seed) / static_cast<double>(RAND_MAX);
  };

  const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
  int count = 0;

  while (count < N) {
    // Create group_cnt points within a circle of radius R
    // around center point (cntr_x, cntr_y)
    const double cntr_x = frand() * MAX_WIDTH;
    const double cntr_y = frand() * MAX_HEIGHT;
    const double R = frand() * min_dim / 2.0;
    int group_cnt = static_cast<int>(frand() * (N / 30.0));
    // The original generator cannot produce a nonzero group for N <= 30;
    // keep its RNG stream unchanged for benchmark-sized inputs while
    // making this degenerate small-input case terminate.
    if (N <= 30)
      group_cnt = 1;

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
inline double distance(const Point &p1, const Point &p2) {
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return std::sqrt(dx * dx + dy * dy);
}

// Main QT clustering algorithm.  MPI partitions the seed candidates in every
// greedy outer iteration.  CUDA evaluates all candidates for each seed and
// OpenMP performs the rank-local reductions and seed-list preparation.
std::vector<Cluster> qtClustering(const std::vector<Point> &points,
                                  const double threshold, MPI_Comm communicator,
                                  const int rank, const int world_size) {
  const int point_count = static_cast<int>(points.size());
  std::vector<unsigned char> clustered(point_count, 0);
  std::vector<int> unclustered_indices(point_count);
  std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
  std::vector<Cluster> clusters;
  clusters.reserve(point_count);

  CudaClusterEngine cuda_engine(points);

  while (!unclustered_indices.empty()) {
    const size_t local_count =
        unclustered_indices.size() <= static_cast<size_t>(rank)
            ? 0
            : (unclustered_indices.size() - static_cast<size_t>(rank) +
               static_cast<size_t>(world_size) - 1) /
                  static_cast<size_t>(world_size);
    std::vector<int> local_seeds(local_count);

#pragma omp parallel for schedule(static)
    for (int local_index = 0; local_index < static_cast<int>(local_count);
         ++local_index) {
      const size_t global_index =
          static_cast<size_t>(rank) +
          static_cast<size_t>(local_index) * static_cast<size_t>(world_size);
      local_seeds[local_index] = unclustered_indices[global_index];
    }

    const LocalBestCluster local_best =
        cuda_engine.evaluate(local_seeds, clustered, threshold);

    // MPI_MAXLOC maximizes cardinality and, for equal cardinality, uses
    // the lowest location.  The location is therefore the seed itself,
    // exactly matching the serial first-winner rule.
    const int local_pair[2] = {local_best.cardinality, local_best.seed >= 0
                                                           ? local_best.seed
                                                           : INT_MAX};
    int global_pair[2] = {0, INT_MAX};
    MPI_Allreduce(local_pair, global_pair, 1, MPI_2INT, MPI_MAXLOC,
                  communicator);

    if (global_pair[0] <= 0)
      break;

    const int best_seed = global_pair[1];
    const int local_winner = (local_best.cardinality == global_pair[0] &&
                              local_best.seed == best_seed)
                                 ? rank
                                 : world_size;
    int winner = world_size;
    MPI_Allreduce(&local_winner, &winner, 1, MPI_INT, MPI_MIN, communicator);

    int selected_count = rank == winner ? local_best.cardinality : 0;
    MPI_Bcast(&selected_count, 1, MPI_INT, winner, communicator);
    std::vector<int> selected_members(static_cast<size_t>(selected_count));
    if (rank == winner)
      selected_members = local_best.members;
    MPI_Bcast(selected_members.data(), selected_count, MPI_INT, winner,
              communicator);

    Cluster cluster;
    cluster.seed_point = best_seed;
    cluster.members = selected_members;
    clusters.push_back(cluster);

    for (const int member : selected_members)
      clustered[member] = 1;
    unclustered_indices.erase(std::remove_if(unclustered_indices.begin(),
                                             unclustered_indices.end(),
                                             [&clustered](const int index) {
                                               return clustered[index] != 0;
                                             }),
                              unclustered_indices.end());
  }

  return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster> &clusters,
                      const std::vector<Point> &points,
                      const double threshold) {
  bool valid = true;

  printf("Validating clusters:\n");

  // Check each cluster
  for (size_t c = 0; c < clusters.size(); ++c) {
    const auto &cluster = clusters[c];
    double max_diameter = 0.0;

    // Check diameter (max distance between any two points)
    for (size_t i = 0; i < cluster.members.size(); ++i) {
      for (size_t j = i + 1; j < cluster.members.size(); ++j) {
        const double dist =
            distance(points[cluster.members[i]], points[cluster.members[j]]);
        max_diameter = std::max(max_diameter, dist);
      }
    }

    if (c < 10) { // Print first 10 clusters
      printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c,
             cluster.members.size(), cluster.seed_point, max_diameter);
    }

    // Validate diameter is within threshold
    if (max_diameter > threshold * 1.001) { // Allow small numerical error
      printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c,
             max_diameter, threshold);
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
    if (membership[i] >= 0)
      clustered_count++;
  }

  printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
         clustered_count, points.size() - clustered_count);

  return valid;
}

void printUsage(const char *progName) {
  printf("Usage: %s [options]\n", progName);
  printf("Options:\n");
  printf("  -n <num>     Number of points (default: 1000)\n");
  printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
  printf("  -v           Enable validation\n");
  printf("  -r           Print results for external validation\n");
  printf("  -h           Show this help message\n");
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (provided < MPI_THREAD_FUNNELED) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return 1;
  }

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
      if (rank == 0)
        printUsage(argv[0]);
      MPI_Finalize();
      return 0;
    } else {
      if (rank == 0) {
        printf("Unknown option: %s\n", argv[i]);
        printUsage(argv[0]);
      }
      MPI_Finalize();
      return 1;
    }
  }

  if (num_points <= 0 || threshold <= 0.0) {
    if (rank == 0) {
      printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
             num_points, threshold);
    }
    MPI_Finalize();
    return 1;
  }

  // One rank selects one accelerator.  MPI ranks on the same node are
  // mapped to local devices, which is the standard accelerator-cluster
  // launch model (one or more ranks per node).
  MPI_Comm local_communicator = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &local_communicator);
  int local_rank = 0;
  MPI_Comm_rank(local_communicator, &local_rank);

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count <= 0) {
    if (rank == 0) {
      std::fprintf(stderr, "CUDA accelerator is required: %s\n",
                   cudaGetErrorString(device_status));
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return 1;
  }
  checkCuda(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
  MPI_Comm_free(&local_communicator);

  if (rank == 0) {
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("MPI ranks: %d\n", world_size);
    printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
  }

  // Generate synthetic data
  std::vector<Point> points(num_points);
  if (rank == 0)
    generateSyntheticData(points, num_points);

  // Broadcast as bytes in MPI-count-sized chunks so the data remains
  // replicated and directly addressable by every rank and accelerator.
  const size_t point_bytes = points.size() * sizeof(Point);
  unsigned char *point_data = reinterpret_cast<unsigned char *>(points.data());
  for (size_t offset = 0; offset < point_bytes;) {
    const size_t chunk =
        std::min(point_bytes - offset,
                 static_cast<size_t>(std::numeric_limits<int>::max()));
    MPI_Bcast(point_data + offset, static_cast<int>(chunk), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    offset += chunk;
  }

  // Perform QT clustering
  MPI_Barrier(MPI_COMM_WORLD);
  const double cluster_start = MPI_Wtime();
  const std::vector<Cluster> clusters =
      qtClustering(points, threshold, MPI_COMM_WORLD, rank, world_size);
  MPI_Barrier(MPI_COMM_WORLD);
  const double local_cluster_time = MPI_Wtime() - cluster_start;
  double cluster_time = 0.0;
  MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Clustering time: %ld ms\n",
           static_cast<long>(cluster_time * 1000.0));
    printf("Clusters found: %zu\n", clusters.size());

    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
      const int size = static_cast<int>(clusters[i].members.size());
      total_clustered += size;
      max_cluster_size = std::max(max_cluster_size, size);
    }

    const double avg_cluster_size =
        clusters.empty()
            ? 0.0
            : static_cast<double>(total_clustered) / clusters.size();

    printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = std::max(cluster_time, 1.0e-9);
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec,
           points_per_sec);

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
  }

  // Validation
  int validation_result = 1;
  if (validate && rank == 0) {
    const bool valid = validateClusters(clusters, points, threshold);

    if (valid) {
      printf("Validation: PASSED\n");
      validation_result = 0;
    } else {
      printf("Validation: FAILED\n");
      validation_result = 1;
    }
  }
  if (!validate)
    validation_result = 0;
  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize();
  return validation_result;
}

// Hybrid MPI/OpenMP/CUDA Quality Threshold clustering.
// MPI distributes seeds; OpenMP builds sparse neighborhoods and drives the
// GPUs owned by each rank; CUDA grows independent candidates cooperatively.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static constexpr int BLOCK_SIZE = 128;

struct Point {
    double x, y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Preserve the original random sequence, including its native host arithmetic.
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        // The original generator cannot make progress for N <= 30.
        if (N <= 30) group_cnt = std::max(1, group_cnt);
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count++] = {x, y};
            --group_cnt;
        }
    }
}

// Match native host FMA arithmetic explicitly on the device. Rounded double
// square roots matter: equal rounded distances must prefer the lower index.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
#else
    return std::sqrt(std::fma(dx, dx, dy * dy));
#endif
}

static void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("GPU allocation size overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedMembers {
    int* data = nullptr;
    void allocate(size_t count) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(int)));
    }
    ~PinnedMembers() { if (data) cudaFreeHost(data); }
};

// A compact uniform grid avoids an all-pairs host neighborhood search. Cell
// widths are at least the threshold, so only the adjacent nine cells matter.
struct Neighborhoods {
    std::vector<size_t> offsets;
    std::vector<int> neighbors;

    Neighborhoods(const std::vector<Point>& points, double threshold, int rank, int ranks) {
        const int N = static_cast<int>(points.size());
        const int local_count = N / ranks + (rank < N % ranks);
        int side = 1;
        if (std::isfinite(threshold) && threshold < MAX_WIDTH) {
            const int limit = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N))));
            side = static_cast<int>(std::min(static_cast<double>(limit),
                                           std::floor(MAX_WIDTH / threshold)));
            side = std::max(1, side);
        }
        const double cell_width = MAX_WIDTH / side;
        const size_t cell_count = static_cast<size_t>(side) * side;
        std::vector<size_t> cell_offsets(cell_count + 1, 0);
        std::vector<int> cell_points(N);
        auto cell = [&](const Point& p) {
            const int x = std::min(side - 1, static_cast<int>(p.x / cell_width));
            const int y = std::min(side - 1, static_cast<int>(p.y / cell_width));
            return static_cast<size_t>(y) * side + x;
        };
        for (const Point& p : points) ++cell_offsets[cell(p) + 1];
        std::partial_sum(cell_offsets.begin(), cell_offsets.end(), cell_offsets.begin());
        std::vector<size_t> cursor = cell_offsets;
        for (int i = 0; i < N; ++i) cell_points[cursor[cell(points[i])]++] = i;

        offsets.resize(static_cast<size_t>(local_count) + 1, 0);
        const int threads = std::max(1, std::min(omp_get_max_threads(),
                                               (local_count + 63) / 64));
        // Count, prefix-sum, then fill CSR without per-seed allocations or locks.
        for (int pass = 0; pass < 2; ++pass) {
            #pragma omp parallel for schedule(dynamic, 32) num_threads(threads)
            for (int local = 0; local < local_count; ++local) {
                const int seed = rank + local * ranks;
                const Point& p = points[seed];
                const int cx = std::min(side - 1, static_cast<int>(p.x / cell_width));
                const int cy = std::min(side - 1, static_cast<int>(p.y / cell_width));
                size_t out = pass ? offsets[local] : 0;
                if (pass) neighbors[out] = seed;
                ++out;
                for (int y = std::max(0, cy - 1); y <= std::min(side - 1, cy + 1); ++y) {
                    for (int x = std::max(0, cx - 1); x <= std::min(side - 1, cx + 1); ++x) {
                        const size_t bin = static_cast<size_t>(y) * side + x;
                        for (size_t k = cell_offsets[bin]; k < cell_offsets[bin + 1]; ++k) {
                            const int candidate = cell_points[k];
                            if (candidate != seed && distance(points[candidate], p) < threshold) {
                                if (pass) neighbors[out] = candidate;
                                ++out;
                            }
                        }
                    }
                }
                if (!pass) offsets[local + 1] = out;
            }
            if (!pass) {
                std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
                neighbors.resize(offsets.back());
            }
        }
    }
};

struct Choice {
    double diameter;
    int point;
    int remaining;
};

struct CombineChoices {
    __device__ Choice operator()(Choice a, Choice b) const {
        Choice result = (b.diameter < a.diameter ||
                        (b.diameter == a.diameter && b.point < a.point)) ? b : a;
        result.remaining = a.remaining + b.remaining;
        return result;
    }
};

__device__ inline unsigned long long clusterKey(int cardinality, int seed) {
    return (static_cast<unsigned long long>(cardinality) << 32) |
           (0xffffffffu - static_cast<unsigned int>(seed));
}

// One block grows one seed. Small neighborhoods keep coordinates and running
// maxima in registers; the general path uses CSR-sized global scratch space.
// Cached candidates remain exact until a previously chosen member is removed:
// removing only unchosen points cannot change any choice in the greedy sequence.
template <int ITEMS>
__global__ void growCandidates(const Point* __restrict__ points,
                               const unsigned char* __restrict__ clustered,
                               const int* __restrict__ seeds,
                               const size_t* __restrict__ offsets,
                               const int* __restrict__ neighbors,
                               const int* __restrict__ tasks,
                               int* __restrict__ members,
                               int* __restrict__ sizes,
                               double* __restrict__ scratch,
                               double threshold,
                               unsigned long long* best) {
    const int slot = tasks[blockIdx.x];
    const int seed = seeds[slot];
    if (clustered[seed]) return;
    const size_t begin = offsets[slot];
    const size_t degree = offsets[slot + 1] - begin;
    const double infinity = 1.7976931348623157e308;
    using Reduction = cub::BlockReduce<Choice, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage reduction;
    __shared__ Point last_point;
    __shared__ int last_id;
    __shared__ int cardinality;
    __shared__ int action;

    const int cached_size = sizes[slot];
    if (cached_size > 0) {
        bool changed = false;
        for (int i = threadIdx.x; i < cached_size; i += BLOCK_SIZE)
            changed = changed || clustered[members[begin + i]];
        if (!__syncthreads_or(changed)) {
            if (threadIdx.x == 0) atomicMax(best, clusterKey(cached_size, seed));
            return;
        }
    }
    if (threadIdx.x == 0) {
        sizes[slot] = 0;
        members[begin] = seed;
        last_point = points[seed];
        last_id = seed;
        cardinality = 1;
    }

    int ids[ITEMS ? ITEMS : 1];
    Point candidates[ITEMS ? ITEMS : 1];
    double diameters[ITEMS ? ITEMS : 1];
    if constexpr (ITEMS > 0) {
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            const size_t position = threadIdx.x + static_cast<size_t>(i) * BLOCK_SIZE;
            int id = position < degree ? neighbors[begin + position] : -1;
            if (id >= 0 && (id == seed || clustered[id])) id = -1;
            ids[i] = id;
            candidates[i] = id >= 0 ? points[id] : Point{0.0, 0.0};
            diameters[i] = 0.0;
        }
    } else {
        for (size_t i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            const int id = neighbors[begin + i];
            scratch[begin + i] = (id == seed || clustered[id]) ? infinity : 0.0;
        }
    }
    __syncthreads();

    while (true) {
        Choice choice{infinity, 0x7fffffff, 0};
        if constexpr (ITEMS > 0) {
            #pragma unroll
            for (int i = 0; i < ITEMS; ++i) {
                if (ids[i] == last_id) ids[i] = -1;
                if (ids[i] >= 0 && diameters[i] < threshold) {
                    const double d = distance(candidates[i], last_point);
                    diameters[i] = d > diameters[i] ? d : diameters[i];
                    if (diameters[i] < threshold && diameters[i] < infinity) {
                        const Choice candidate{diameters[i], ids[i], 1};
                        choice = CombineChoices{}(choice, candidate);
                    }
                }
            }
        } else {
            for (size_t i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
                const int id = neighbors[begin + i];
                double diameter = scratch[begin + i];
                if (id == last_id) diameter = infinity;
                if (diameter < threshold && diameter < infinity) {
                    const double d = distance(points[id], last_point);
                    diameter = d > diameter ? d : diameter;
                    if (diameter < threshold && diameter < infinity)
                        choice = CombineChoices{}(choice, Choice{diameter, id, 1});
                }
                scratch[begin + i] = diameter;
            }
        }
        const Choice next = Reduction(reduction).Reduce(choice, CombineChoices{});
        if (threadIdx.x == 0) {
            if (next.remaining == 0) {
                sizes[slot] = cardinality;
                atomicMax(best, clusterKey(cardinality, seed));
                action = 0;
            } else if (clusterKey(cardinality + next.remaining, seed) <= atomicAdd(best, 0ull)) {
                // This exact upper bound cannot beat an already completed seed.
                // Leave the cache invalid so a subsequent round can revisit it.
                action = 0;
            } else {
                members[begin + cardinality++] = next.point;
                last_id = next.point;
                last_point = points[next.point];
                action = 1;
            }
        }
        __syncthreads();
        if (!action) return;
    }
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        clustered[members[i]] = 1;
}

class GpuWorker {
    int device;
    cudaStream_t stream = nullptr;
    DeviceBuffer<Point> points;
    DeviceBuffer<unsigned char> clustered;
    DeviceBuffer<int> seeds, neighbors, tasks, members, sizes, selected;
    DeviceBuffer<size_t> offsets;
    DeviceBuffer<double> scratch;
    DeviceBuffer<unsigned long long> best;
    PinnedMembers transfer;
    std::vector<size_t> host_offsets;
    int group_begin[6]{};
    unsigned long long host_best = 0;

    template <int ITEMS> void launch(int group, double threshold) {
        const int count = group_begin[group + 1] - group_begin[group];
        if (!count) return;
        growCandidates<ITEMS><<<count, BLOCK_SIZE, 0, stream>>>(
            points.data, clustered.data, seeds.data, offsets.data, neighbors.data,
            tasks.data + group_begin[group], members.data, sizes.data, scratch.data,
            threshold, best.data);
        cudaCheck(cudaPeekAtLastError());
    }

public:
    GpuWorker(int gpu, int worker, int workers, int rank, int ranks,
              const std::vector<Point>& input, const Neighborhoods& graph) : device(gpu) {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const int local_count = static_cast<int>(graph.offsets.size() - 1);
        const int count = local_count / workers + (worker < local_count % workers);
        std::vector<int> host_seeds(count);
        host_offsets.resize(static_cast<size_t>(count) + 1, 0);
        for (int i = 0; i < count; ++i) {
            const int local = worker + i * workers;
            host_seeds[i] = rank + local * ranks;
            host_offsets[i + 1] = host_offsets[i] + graph.offsets[local + 1] - graph.offsets[local];
        }
        std::vector<int> host_neighbors(host_offsets.back());
        for (int i = 0; i < count; ++i) {
            const int local = worker + i * workers;
            std::copy(graph.neighbors.begin() + graph.offsets[local],
                      graph.neighbors.begin() + graph.offsets[local + 1],
                      host_neighbors.begin() + host_offsets[i]);
        }
        std::vector<int> host_tasks(count);
        std::iota(host_tasks.begin(), host_tasks.end(), 0);
        auto degree = [&](int i) { return host_offsets[i + 1] - host_offsets[i]; };
        std::sort(host_tasks.begin(), host_tasks.end(), [&](int a, int b) {
            return degree(a) != degree(b) ? degree(a) > degree(b) : a < b;
        });
        const size_t limits[4] = {1024, 512, 256, 128};
        for (int group = 0; group < 4; ++group) {
            group_begin[group + 1] = group_begin[group];
            while (group_begin[group + 1] < count &&
                   degree(host_tasks[group_begin[group + 1]]) > limits[group])
                ++group_begin[group + 1];
        }
        group_begin[5] = count;

        points.allocate(input.size());
        clustered.allocate(input.size());
        selected.allocate(input.size());
        seeds.allocate(count);
        sizes.allocate(count);
        tasks.allocate(count);
        offsets.allocate(host_offsets.size());
        neighbors.allocate(host_neighbors.size());
        members.allocate(host_neighbors.size());
        if (group_begin[1]) scratch.allocate(host_neighbors.size());
        best.allocate(1);
        transfer.allocate(input.size());
        cudaCheck(cudaMemcpyAsync(points.data, input.data(), input.size() * sizeof(Point),
                                  cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemsetAsync(clustered.data, 0, input.size(), stream));
        cudaCheck(cudaMemcpyAsync(offsets.data, host_offsets.data(), host_offsets.size() * sizeof(size_t),
                                  cudaMemcpyHostToDevice, stream));
        if (count) {
            cudaCheck(cudaMemsetAsync(sizes.data, 0, count * sizeof(int), stream));
            cudaCheck(cudaMemcpyAsync(seeds.data, host_seeds.data(), count * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaMemcpyAsync(tasks.data, host_tasks.data(), count * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaMemcpyAsync(neighbors.data, host_neighbors.data(),
                                      host_neighbors.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        cudaCheck(cudaStreamSynchronize(stream));
    }

    ~GpuWorker() {
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        // Buffer destructors run while this worker's device is still current.
    }

    unsigned long long evaluate(double threshold) {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaMemsetAsync(best.data, 0, sizeof(unsigned long long), stream));
        launch<0>(0, threshold);
        launch<8>(1, threshold);
        launch<4>(2, threshold);
        launch<2>(3, threshold);
        launch<1>(4, threshold);
        cudaCheck(cudaMemcpyAsync(&host_best, best.data, sizeof(host_best), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return host_best;
    }

    void fetch(int slot, std::vector<int>& output) {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaMemcpyAsync(output.data(), members.data + host_offsets[slot],
                                  output.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void remove(const std::vector<int>& cluster) {
        cudaCheck(cudaSetDevice(device));
        std::copy(cluster.begin(), cluster.end(), transfer.data);
        cudaCheck(cudaMemcpyAsync(selected.data, transfer.data, cluster.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
        const int blocks = std::min(1024, static_cast<int>((cluster.size() + 255) / 256));
        markClustered<<<blocks, 256, 0, stream>>>(clustered.data, selected.data,
                                                static_cast<int>(cluster.size()));
        cudaCheck(cudaPeekAtLastError());
        // The next evaluation synchronizes this stream before reusing transfer.
    }
};

// Exceptions must leave an OpenMP region before MPI's funneled main thread
// reports the failure. Every iteration writes a distinct exception slot.
template <typename Operation>
static void onGpus(int count, Operation operation) {
    std::vector<std::exception_ptr> errors(count);
    #pragma omp parallel for num_threads(count) schedule(static)
    for (int i = 0; i < count; ++i) {
        try { operation(i); }
        catch (...) { errors[i] = std::current_exception(); }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  int rank, int ranks) {
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, local_ranks;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_ranks);
    MPI_Comm_free(&local_comm);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(32, omp_get_num_procs() / local_ranks)));

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));
    if (!device_count) throw std::runtime_error("MPI/OpenMP/CUDA clustering requires a CUDA GPU on every rank");
    std::vector<int> devices;
    if (local_ranks <= device_count) {
        for (int device = local_rank; device < device_count; device += local_ranks)
            devices.push_back(device);
    } else {
        devices.push_back(local_rank % device_count);
    }
    const int workers = static_cast<int>(devices.size());
    std::vector<std::unique_ptr<GpuWorker>> gpus(workers);
    {
        const Neighborhoods graph(points, threshold, rank, ranks);
        onGpus(workers, [&](int i) {
            gpus[i] = std::make_unique<GpuWorker>(devices[i], i, workers, rank, ranks, points, graph);
        });
    }
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;
    std::vector<unsigned long long> keys(workers);
    int remaining = N;
    while (remaining > 0) {
        onGpus(workers, [&](int i) { keys[i] = gpus[i]->evaluate(threshold); });
        const unsigned long long key = *std::max_element(keys.begin(), keys.end());
        struct { int cardinality; int seed; } local{
            static_cast<int>(key >> 32),
            key ? static_cast<int>(0xffffffffu - static_cast<unsigned int>(key)) : 0x7fffffff
        }, global{};
        MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (global.cardinality <= 0) throw std::runtime_error("No candidate for remaining points");
        if (global.cardinality == 1) {
            // All remaining candidates are singletons. Their exact selection
            // order is increasing seed index, without further collectives.
            if (rank == 0) {
                for (int seed = 0; seed < N; ++seed)
                    if (!clustered[seed]) clusters.push_back(Cluster{{seed}, seed});
            }
            break;
        }
        const int owner = global.seed % ranks;
        const int local_slot = global.seed / ranks;
        std::vector<int> chosen(global.cardinality);
        if (rank == owner)
            gpus[local_slot % workers]->fetch(local_slot / workers, chosen);
        MPI_Bcast(chosen.data(), global.cardinality, MPI_INT, owner, MPI_COMM_WORLD);
        for (int point : chosen) clustered[point] = 1;
        onGpus(workers, [&](int i) { gpus[i]->remove(chosen); });
        remaining -= global.cardinality;
        if (rank == 0) clusters.push_back(Cluster{std::move(chosen), global.seed});
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

static int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate once, then replicate coordinates. Chunking keeps MPI counts in
    // range even when the number of doubles exceeds INT_MAX.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    double* coordinates = reinterpret_cast<double*>(points.data());
    const size_t coordinate_count = static_cast<size_t>(num_points) * 2;
    for (size_t offset = 0; offset < coordinate_count;) {
        const int count = static_cast<int>(std::min(coordinate_count - offset,
                                                   static_cast<size_t>(1 << 26)));
        MPI_Bcast(coordinates + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, ranks);
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank != 0) return 0;
    const long long cluster_ms = static_cast<long long>(time_sec * 1000.0);
    printf("Clustering time: %lld ms\n", cluster_ms);
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


int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int status = 0;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

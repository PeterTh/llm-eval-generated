// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
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
        // The original generator otherwise never makes progress for N <= 30.
        if (N <= 30) group_cnt = 1;
        
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

namespace {
constexpr int BLOCK_SIZE = 128;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        // This can be called by an OpenMP worker; MPI calls stay on the main thread.
        std::abort();
    }
}

void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        std::fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            std::fprintf(stderr, "CUDA allocation size overflow\n");
            std::abort();
        }
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "allocate");
    }
    ~DeviceBuffer() { if (data) cudaCheck(cudaFree(data), "free"); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Best {
    int size;
    int seed;
};
static_assert(sizeof(Best) == 2 * sizeof(int), "MPI_MAXLOC requires two adjacent ints");

__host__ __device__ bool better(Best a, Best b) {
    return a.size > b.size || (a.size == b.size && a.seed < b.seed);
}

// Counting first gives a sparse, distributed threshold graph rather than a
// replicated N-by-N distance matrix. Every cluster member must neighbor its seed.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                              int first_seed, int stride, int* counts) {
    const int row = blockIdx.x;
    const int seed = first_seed + row * stride;
    int count = 0;
    const Point origin = points[seed];
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        if (i != seed && distance(origin, points[i]) < threshold) ++count;
    }
    __shared__ int partial[BLOCK_SIZE];
    partial[threadIdx.x] = count;
    __syncthreads();
    for (int step = BLOCK_SIZE / 2; step; step /= 2) {
        if (threadIdx.x < step) partial[threadIdx.x] += partial[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[row] = partial[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                             int first_seed, int stride, const size_t* offsets,
                             int* neighbors, double* initial_distances) {
    const int row = blockIdx.x;
    const int seed = first_seed + row * stride;
    const Point origin = points[seed];
    int count = 0;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        if (i != seed && distance(origin, points[i]) < threshold) ++count;
    }
    // Each lane gets a disjoint output range without per-edge atomics.
    __shared__ int prefix[BLOCK_SIZE];
    prefix[threadIdx.x] = count;
    __syncthreads();
    for (int step = 1; step < BLOCK_SIZE; step *= 2) {
        const int add = threadIdx.x >= step ? prefix[threadIdx.x - step] : 0;
        __syncthreads();
        prefix[threadIdx.x] += add;
        __syncthreads();
    }
    size_t output = offsets[row] + prefix[threadIdx.x] - count;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        if (i == seed) continue;
        const double d = distance(origin, points[i]);
        if (d < threshold) {
            neighbors[output] = i;
            initial_distances[output++] = d;
        }
    }
}

// A cached greedy candidate is unchanged if none of its members was removed:
// every choice it made is still available, and removing other choices cannot
// alter a deterministic minimum. This avoids rebuilding unaffected candidates.
__global__ void invalidateCandidates(const unsigned char* clustered, int first_seed,
                                     int stride, const size_t* offsets,
                                     const int* members, int* sizes) {
    const int row = blockIdx.x;
    const int seed = first_seed + row * stride;
    const int count = sizes[row];
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[row] = 0;
        return;
    }
    if (count <= 0) return;
    const size_t begin = offsets[row] + row;
    int removed = 0;
    for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        removed |= clustered[members[begin + i]];
    }
    if (__syncthreads_or(removed) && threadIdx.x == 0) sizes[row] = -1;
}

// Reduce (maximum distance, point id) in lexicographic order. Explicit ids
// preserve the sequential scan's first-point tie break even in an unsorted CSR.
__device__ void blockMinimum(double& value, int& id, double* values, int* ids) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    for (int step = 16; step; step /= 2) {
        const double other_value = __shfl_down_sync(0xffffffffu, value, step);
        const int other_id = __shfl_down_sync(0xffffffffu, id, step);
        if (other_value < value || (other_value == value && other_id < id)) {
            value = other_value;
            id = other_id;
        }
    }
    if (lane == 0) {
        values[warp] = value;
        ids[warp] = id;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < BLOCK_SIZE / 32 ? values[lane] : CUDART_INF;
        id = lane < BLOCK_SIZE / 32 ? ids[lane] : INT_MAX;
        for (int step = 16; step; step /= 2) {
            const double other_value = __shfl_down_sync(0xffffffffu, value, step);
            const int other_id = __shfl_down_sync(0xffffffffu, id, step);
            if (other_value < value || (other_value == value && other_id < id)) {
                value = other_value;
                id = other_id;
            }
        }
        if (lane == 0) {
            values[0] = value;
            ids[0] = id;
        }
    }
    __syncthreads();
    value = values[0];
    id = ids[0];
    // All lanes must read the broadcast before a faster warp reuses scratch.
    __syncthreads();
}

__global__ void buildCandidates(const Point* points, double threshold,
                                const unsigned char* clustered, int first_seed,
                                int stride, const size_t* offsets,
                                const int* neighbors, const double* initial_distances,
                                double* maxima, int* members, int* sizes) {
    const int row = blockIdx.x;
    const int seed = first_seed + row * stride;
    if (sizes[row] >= 0) return;
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[row] = 0;
        return;
    }
    const size_t begin = offsets[row], end = offsets[row + 1];
    const size_t member_begin = begin + row;
    __shared__ double values[BLOCK_SIZE / 32];
    __shared__ int ids[BLOCK_SIZE / 32];
    if (threadIdx.x == 0) members[member_begin] = seed;
    double minimum = CUDART_INF;
    int closest = INT_MAX;
    for (size_t i = begin + threadIdx.x; i < end; i += BLOCK_SIZE) {
        const int candidate = neighbors[i];
        const double d = clustered[candidate] ? CUDART_INF : initial_distances[i];
        maxima[i] = d;
        if (d < minimum || (d == minimum && candidate < closest)) {
            minimum = d;
            closest = candidate;
        }
    }
    blockMinimum(minimum, closest, values, ids);
    int cardinality = 1;
    while (closest != INT_MAX && minimum < threshold) {
        const int chosen = closest;
        if (threadIdx.x == 0) members[member_begin + cardinality] = chosen;
        ++cardinality;
        const Point added = points[chosen];
        minimum = CUDART_INF;
        closest = INT_MAX;
        // Update only the contribution of the new member. Previously the
        // entire growing cluster was scanned again for every candidate.
        for (size_t i = begin + threadIdx.x; i < end; i += BLOCK_SIZE) {
            const int candidate = neighbors[i];
            double d = maxima[i];
            if (candidate == chosen) {
                d = CUDART_INF;
            } else if (d < threshold) {
                const double next = distance(points[candidate], added);
                d = next > d ? next : d;
                if (!(d < threshold)) d = CUDART_INF;
            }
            maxima[i] = d;
            if (d < minimum || (d == minimum && candidate < closest)) {
                minimum = d;
                closest = candidate;
            }
        }
        blockMinimum(minimum, closest, values, ids);
    }
    if (threadIdx.x == 0) sizes[row] = cardinality;
}

__global__ void selectBest(const int* sizes, int rows, int first_seed,
                           int stride, Best* result) {
    Best best{0, INT_MAX};
    for (int row = threadIdx.x; row < rows; row += BLOCK_SIZE) {
        Best candidate{sizes[row], first_seed + row * stride};
        if (better(candidate, best)) best = candidate;
    }
    __shared__ Best partial[BLOCK_SIZE];
    partial[threadIdx.x] = best;
    __syncthreads();
    for (int step = BLOCK_SIZE / 2; step; step /= 2) {
        if (threadIdx.x < step && better(partial[threadIdx.x + step], partial[threadIdx.x])) {
            partial[threadIdx.x] = partial[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = partial[0];
}

class GpuWorker {
    int device_, n_, first_seed_, stride_, rows_;
    double threshold_;
    cudaStream_t stream_{};
    DeviceBuffer<Point> points_;
    DeviceBuffer<unsigned char> clustered_;
    DeviceBuffer<size_t> offsets_;
    DeviceBuffer<int> neighbors_, members_, sizes_;
    DeviceBuffer<double> initial_distances_, maxima_;
    DeviceBuffer<Best> result_;
    std::vector<size_t> host_offsets_;
    std::vector<int> batch_starts_;
    std::vector<size_t> batch_offsets_;
    std::vector<int> saved_members_;
    bool persistent_;
    Best best_{0, INT_MAX};

    void loadBatch(int first, int last) {
        batch_offsets_.resize(last - first + 1);
        for (int i = first; i <= last; ++i) {
            batch_offsets_[i - first] = host_offsets_[i] - host_offsets_[first];
        }
        cudaCheck(cudaMemcpyAsync(offsets_.data, batch_offsets_.data(),
                                  batch_offsets_.size() * sizeof(size_t),
                                  cudaMemcpyHostToDevice, stream_), "copy offsets");
        fillNeighbors<<<last - first, BLOCK_SIZE, 0, stream_>>>(
            points_.data, n_, threshold_, first_seed_ + first * stride_, stride_,
            offsets_.data, neighbors_.data, initial_distances_.data);
        cudaCheck(cudaGetLastError(), "fill neighbors");
        cudaCheck(cudaMemsetAsync(sizes_.data, 0xff, (last - first) * sizeof(int), stream_), "initialize candidates");
    }

    void copyMembers(int seed, int size, int batch_first, std::vector<int>& output) {
        const int row = (seed - first_seed_) / stride_ - batch_first;
        const size_t offset = host_offsets_[row + batch_first] - host_offsets_[batch_first] + row;
        output.resize(size);
        cudaCheck(cudaMemcpyAsync(output.data(), members_.data + offset, size * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream_), "copy cluster");
        cudaCheck(cudaStreamSynchronize(stream_), "finish cluster copy");
    }

public:
    GpuWorker(int device, int first_seed, int stride, const std::vector<Point>& points, double threshold)
        : device_(device), n_(static_cast<int>(points.size())), first_seed_(first_seed),
          stride_(stride), rows_(first_seed < n_ ? 1 + (n_ - 1 - first_seed) / stride : 0),
          threshold_(threshold), persistent_(true) {
        cudaCheck(cudaSetDevice(device_), "select GPU");
        cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
        points_.allocate(n_);
        clustered_.allocate(n_);
        result_.allocate(1);
        cudaCheck(cudaMemcpyAsync(points_.data, points.data(), n_ * sizeof(Point),
                                  cudaMemcpyHostToDevice, stream_), "copy points");
        if (!rows_) {
            cudaCheck(cudaStreamSynchronize(stream_), "initialize idle GPU");
            return;
        }
        std::vector<int> counts(rows_);
        {
            DeviceBuffer<int> device_counts;
            device_counts.allocate(rows_);
            countNeighbors<<<rows_, BLOCK_SIZE, 0, stream_>>>(
                points_.data, n_, threshold_, first_seed_, stride_, device_counts.data);
            cudaCheck(cudaGetLastError(), "count neighbors");
            cudaCheck(cudaMemcpyAsync(counts.data(), device_counts.data, rows_ * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_), "copy neighbor counts");
            cudaCheck(cudaStreamSynchronize(stream_), "finish neighbor counts");
        }
        host_offsets_.resize(rows_ + 1, 0);
        for (int row = 0; row < rows_; ++row) host_offsets_[row + 1] = host_offsets_[row] + counts[row];
        size_t free_bytes, total_bytes;
        cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "query GPU memory");
        // Bounded workspace also supports dense threshold graphs. If the full
        // graph fits, keep it and the candidate cache resident across rounds.
        // Otherwise stream seed batches through the same CUDA algorithm.
        const size_t budget = free_bytes / 10 * 7;
        const size_t row_bytes = (static_cast<size_t>(rows_) + 1) * (sizeof(size_t) + 2 * sizeof(int));
        if (budget <= row_bytes) {
            std::fprintf(stderr, "Insufficient GPU memory for %d seeds\n", rows_);
            std::abort();
        }
        const size_t edge_bytes = sizeof(int) * 2 + sizeof(double) * 2;
        const size_t edge_limit = (budget - row_bytes) / edge_bytes;
        batch_starts_.push_back(0);
        size_t max_edges = 0;
        int max_rows = 0;
        int first = 0;
        while (first < rows_) {
            int last = first;
            while (last < rows_ && host_offsets_[last + 1] - host_offsets_[first] <= edge_limit) ++last;
            if (last == first) {
                std::fprintf(stderr, "Insufficient GPU memory for a single candidate\n");
                std::abort();
            }
            max_edges = std::max(max_edges, host_offsets_[last] - host_offsets_[first]);
            max_rows = std::max(max_rows, last - first);
            batch_starts_.push_back(last);
            first = last;
        }
        persistent_ = batch_starts_.size() == 2;
        offsets_.allocate(max_rows + 1);
        neighbors_.allocate(max_edges);
        initial_distances_.allocate(max_edges);
        maxima_.allocate(max_edges);
        members_.allocate(max_edges + max_rows);
        sizes_.allocate(max_rows);
        if (persistent_) loadBatch(0, rows_);
        cudaCheck(cudaStreamSynchronize(stream_), "initialize graph");
    }

    ~GpuWorker() {
        cudaCheck(cudaSetDevice(device_), "select GPU for cleanup");
        cudaCheck(cudaStreamDestroy(stream_), "destroy stream");
    }

    Best evaluate(const std::vector<unsigned char>& clustered) {
        best_ = {0, INT_MAX};
        if (!rows_) return best_;
        cudaCheck(cudaSetDevice(device_), "select GPU for candidates");
        cudaCheck(cudaMemcpyAsync(clustered_.data, clustered.data(), n_ * sizeof(unsigned char),
                                  cudaMemcpyHostToDevice, stream_), "copy clustered flags");
        for (size_t batch = 1; batch < batch_starts_.size(); ++batch) {
            const int first = batch_starts_[batch - 1], last = batch_starts_[batch];
            const int seed = first_seed_ + first * stride_;
            if (!persistent_) loadBatch(first, last);
            invalidateCandidates<<<last - first, BLOCK_SIZE, 0, stream_>>>(
                clustered_.data, seed, stride_, offsets_.data, members_.data, sizes_.data);
            cudaCheck(cudaGetLastError(), "invalidate candidates");
            buildCandidates<<<last - first, BLOCK_SIZE, 0, stream_>>>(
                points_.data, threshold_, clustered_.data, seed, stride_, offsets_.data,
                neighbors_.data, initial_distances_.data, maxima_.data, members_.data, sizes_.data);
            cudaCheck(cudaGetLastError(), "build candidates");
            selectBest<<<1, BLOCK_SIZE, 0, stream_>>>(sizes_.data, last - first, seed, stride_, result_.data);
            cudaCheck(cudaGetLastError(), "select best candidate");
            Best batch_best;
            cudaCheck(cudaMemcpyAsync(&batch_best, result_.data, sizeof(Best),
                                      cudaMemcpyDeviceToHost, stream_), "copy best candidate");
            cudaCheck(cudaStreamSynchronize(stream_), "finish candidates");
            if (better(batch_best, best_)) {
                best_ = batch_best;
                if (!persistent_ && best_.size > 0) copyMembers(best_.seed, best_.size, first, saved_members_);
            }
        }
        return best_;
    }

    void winningMembers(std::vector<int>& output) {
        cudaCheck(cudaSetDevice(device_), "select winning GPU");
        if (persistent_) copyMembers(best_.seed, best_.size, 0, output);
        else output = saved_members_;
    }
};
} // namespace

// All ranks follow identical greedy rounds; only the winner and its ordered
// members cross MPI. Independent seed candidates stay distributed on GPUs.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_Comm local_comm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank, local_ranks;
    mpiCheck(MPI_Comm_rank(local_comm, &local_rank));
    mpiCheck(MPI_Comm_size(local_comm, &local_ranks));
    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "enumerate GPUs");
    if (device_count == 0) {
        std::fprintf(stderr, "Rank %d: qtclustering requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    for (int device = local_rank; device < device_count; device += local_ranks) devices.push_back(device);
    if (devices.empty()) devices.push_back(local_rank % device_count);
    mpiCheck(MPI_Comm_free(&local_comm));
    const int worker_count = static_cast<int>(devices.size());
    std::vector<int> rank_workers(ranks), worker_offsets(ranks + 1, 0);
    mpiCheck(MPI_Allgather(&worker_count, 1, MPI_INT, rank_workers.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int r = 0; r < ranks; ++r) worker_offsets[r + 1] = worker_offsets[r] + rank_workers[r];
    const int workers = worker_offsets.back();
    const int n = static_cast<int>(points.size());
    std::vector<std::unique_ptr<GpuWorker>> gpus(worker_count);
    // OpenMP owns independent CUDA contexts/streams; MPI is FUNNELED and is
    // called only outside these regions. One rank can drive all node GPUs.
    const int host_threads = std::max(1, std::min(worker_count, omp_get_max_threads()));
    #pragma omp parallel for schedule(static) num_threads(host_threads)
    for (int gpu = 0; gpu < worker_count; ++gpu) {
        gpus[gpu] = std::make_unique<GpuWorker>(devices[gpu], worker_offsets[rank] + gpu,
                                               workers, points, threshold);
    }
    std::vector<unsigned char> clustered(n, 0);
    std::vector<Best> gpu_best(worker_count);
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        #pragma omp parallel for schedule(static) num_threads(host_threads)
        for (int gpu = 0; gpu < worker_count; ++gpu) gpu_best[gpu] = gpus[gpu]->evaluate(clustered);
        Best local_best{0, INT_MAX}, global_best;
        for (Best best : gpu_best) if (better(best, local_best)) local_best = best;
        // MAXLOC selects the highest cardinality and then the lowest seed id.
        mpiCheck(MPI_Allreduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD));
        if (global_best.size <= 0) throw std::runtime_error("No candidate for remaining points");
        if (global_best.size == 1) {
            // Size one for every seed means no remaining pair is closer than
            // the threshold. Emit the remaining singletons in seed order.
            if (rank == 0) {
                for (int seed = 0; seed < n; ++seed) {
                    if (!clustered[seed]) clusters.push_back(Cluster{{seed}, seed});
                }
            }
            break;
        }
        const int owner_worker = global_best.seed % workers;
        const int owner_rank = static_cast<int>(std::upper_bound(worker_offsets.begin(), worker_offsets.end(),
                                                               owner_worker) - worker_offsets.begin()) - 1;
        std::vector<int> members;
        if (rank == owner_rank) gpus[owner_worker - worker_offsets[rank]]->winningMembers(members);
        else members.resize(global_best.size);
        mpiCheck(MPI_Bcast(members.data(), global_best.size, MPI_INT, owner_rank, MPI_COMM_WORLD));
        #pragma omp parallel for schedule(static) if(global_best.size >= 4096)
        for (int i = 0; i < global_best.size; ++i) clustered[members[i]] = 1;
        remaining -= global_best.size;
        if (rank == 0) clusters.push_back(Cluster{std::move(members), global_best.seed});
    }
    #pragma omp parallel for schedule(static) num_threads(host_threads)
    for (int gpu = 0; gpu < worker_count; ++gpu) gpus[gpu].reset();
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1) if(clusters.size() >= 32)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& members = clusters[c].members;
        double maximum = 0.0;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                maximum = std::max(maximum, distance(points[members[i]], points[members[j]]));
            }
        }
        diameters[c] = maximum;
    }
    // Keep diagnostic output in the original cluster order.
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
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

int runBenchmark(int argc, char** argv, int rank) {
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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                              num_points, threshold);
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two doubles");
    // Chunk the broadcast to stay within MPI's int count limit.
    const size_t coordinate_count = static_cast<size_t>(num_points) * 2;
    double* coordinates = reinterpret_cast<double*>(points.data());
    for (size_t first = 0; first < coordinate_count;) {
        const int count = static_cast<int>(std::min(coordinate_count - first, static_cast<size_t>(INT_MAX)));
        mpiCheck(MPI_Bcast(coordinates + first, count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        first += count;
    }
    
    // Perform QT clustering
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    mpiCheck(MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank != 0) return 0;
    
    printf("Clustering time: %lld ms\n", static_cast<long long>(time_sec * 1000.0));
    printf("Clusters found: %zu\n", clusters.size());
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    #pragma omp parallel for reduction(+:total_clustered) reduction(max:max_cluster_size) if(clusters.size() >= 4096)
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
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int local_status = 0;
    try {
        local_status = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int status = 0;
    mpiCheck(MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return status;
}

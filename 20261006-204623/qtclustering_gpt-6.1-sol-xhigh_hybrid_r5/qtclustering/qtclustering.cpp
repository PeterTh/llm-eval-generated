// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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

// Keep the same double-precision distances, strict threshold, and lowest-index
// tie breaking as the sequential algorithm. In particular, comparing squared
// distances instead would change ties caused by square-root rounding.
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(dx * dx + dy * dy);
#else
    return std::sqrt(dx * dx + dy * dy);
#endif
}

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void checkMpi(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

// MPI calls stay on the initial thread; OpenMP workers manage independent CUDA
// devices. A rank can use several GPUs, and local ranks divide visible devices.
struct ParallelRuntime {
    int rank = 0, ranks = 1, local_rank = 0, local_ranks = 1;
    std::vector<int> devices;
    std::vector<int> device_sharers;

    ParallelRuntime(int& argc, char**& argv) {
        int provided = MPI_THREAD_SINGLE;
        checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
        if (provided < MPI_THREAD_FUNNELED) {
            if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    void initializeDevices() {
        MPI_Comm local_comm;
        checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                    MPI_INFO_NULL, &local_comm));
        checkMpi(MPI_Comm_rank(local_comm, &local_rank));
        checkMpi(MPI_Comm_size(local_comm, &local_ranks));
        int count = 0;
        checkCuda(cudaGetDeviceCount(&count));
        if (count == 0) throw std::runtime_error("CUDA requires at least one visible GPU per rank");
        if (local_ranks <= count) {
            for (int device = local_rank; device < count; device += local_ranks)
                devices.push_back(device);
        } else {
            devices.push_back(local_rank % count);
        }
        // CUDA_VISIBLE_DEVICES may expose a different GPU to each local rank.
        // Count sharing by physical UUID, rather than by visible device number.
        std::vector<cudaUUID_t> local_uuids(devices.size());
        for (size_t i = 0; i < devices.size(); ++i) {
            cudaDeviceProp properties;
            checkCuda(cudaGetDeviceProperties(&properties, devices[i]));
            local_uuids[i] = properties.uuid;
        }
        const int uuid_bytes = static_cast<int>(sizeof(cudaUUID_t));
        const int local_bytes = static_cast<int>(local_uuids.size()) * uuid_bytes;
        std::vector<int> counts(local_ranks), displacements(local_ranks);
        checkMpi(MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, local_comm));
        int total_bytes = 0;
        for (int r = 0; r < local_ranks; ++r) {
            displacements[r] = total_bytes;
            total_bytes += counts[r];
        }
        std::vector<unsigned char> all_uuids(total_bytes);
        checkMpi(MPI_Allgatherv(local_uuids.data(), local_bytes, MPI_BYTE,
                              all_uuids.data(), counts.data(), displacements.data(), MPI_BYTE, local_comm));
        device_sharers.assign(devices.size(), 0);
        for (size_t i = 0; i < devices.size(); ++i)
            for (int offset = 0; offset < total_bytes; offset += uuid_bytes)
                if (memcmp(&local_uuids[i], all_uuids.data() + offset, sizeof(cudaUUID_t)) == 0)
                    ++device_sharers[i];
        checkMpi(MPI_Comm_free(&local_comm));
        std::vector<std::string> errors(devices.size());
        // Establish CUDA contexts before the benchmark timer starts.
        #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
        for (int i = 0; i < static_cast<int>(devices.size()); ++i) {
            try {
                checkCuda(cudaSetDevice(devices[i]));
                checkCuda(cudaFree(nullptr));
            } catch (const std::exception& e) { errors[i] = e.what(); }
        }
        for (const auto& error : errors)
            if (!error.empty()) throw std::runtime_error(error);
    }

    ~ParallelRuntime() { MPI_Finalize(); }
};

constexpr int BLOCK_SIZE = 128;
constexpr int SHARED_CANDIDATES = 1024;
using EdgeIndex = unsigned long long;

// A cell is at least threshold wide. Search an extra cell beyond the adjacent
// cells to cover floating-point rounding at cell and distance boundaries. Bound
// the number of cells by N even for tiny thresholds, avoiding an all-pairs search.
struct NeighborhoodGrid {
    int axis = 1;
    double inverse_width = 1.0 / MAX_WIDTH;
    std::vector<int> offsets, indices;

    NeighborhoodGrid(const std::vector<Point>& points, double threshold) {
        if (std::isfinite(threshold) && threshold > 0.0)
            axis = static_cast<int>(std::max(1.0, std::min(std::floor(MAX_WIDTH / threshold),
                                                        std::sqrt(static_cast<double>(points.size())))));
        inverse_width = axis / MAX_WIDTH;
        auto cell = [&](const Point& p) {
            const int x = std::clamp(static_cast<int>(p.x * inverse_width), 0, axis - 1);
            const int y = std::clamp(static_cast<int>(p.y * inverse_width), 0, axis - 1);
            return y * axis + x;
        };
        offsets.assign(static_cast<size_t>(axis) * axis + 1, 0);
        indices.resize(points.size());
        for (const auto& p : points) ++offsets[cell(p) + 1];
        for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
        std::vector<int> cursor(offsets);
        for (int i = 0; i < static_cast<int>(points.size()); ++i) indices[cursor[cell(points[i])]++] = i;
    }
};

__device__ inline int cellCoordinate(double coordinate, double inverse_width, int axis) {
    return min(axis - 1, max(0, static_cast<int>(coordinate * inverse_width)));
}

// Warp/block reductions compare both distance and point index, independently
// of graph storage order and GPU scheduling.
__device__ inline void reduceClosest(double& value, int& point) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int shift = 16; shift; shift >>= 1) {
        const double other_value = __shfl_down_sync(0xffffffff, value, shift);
        const int other_point = __shfl_down_sync(0xffffffff, point, shift);
        if (other_value < value || (other_value == value && other_point < point)) {
            value = other_value;
            point = other_point;
        }
    }
    __shared__ double warp_values[BLOCK_SIZE / 32];
    __shared__ int warp_points[BLOCK_SIZE / 32];
    __shared__ double block_value;
    __shared__ int block_point;
    if (lane == 0) { warp_values[warp] = value; warp_points[warp] = point; }
    __syncthreads();
    if (warp == 0) {
        value = lane < BLOCK_SIZE / 32 ? warp_values[lane] : DBL_MAX;
        point = lane < BLOCK_SIZE / 32 ? warp_points[lane] : INT_MAX;
        for (int shift = 16; shift; shift >>= 1) {
            const double other_value = __shfl_down_sync(0xffffffff, value, shift);
            const int other_point = __shfl_down_sync(0xffffffff, point, shift);
            if (other_value < value || (other_value == value && other_point < point)) {
                value = other_value;
                point = other_point;
            }
        }
        if (lane == 0) { block_value = value; block_point = point; }
    }
    __syncthreads();
    // Separate result storage keeps the next iteration's warp writes from
    // racing with slower warps reading this iteration's result.
    value = block_value;
    point = block_point;
}

__global__ void countNeighbors(const Point* points, double threshold,
                               long long first_seed, long long stride,
                               const int* cell_offsets, const int* cell_points,
                               int axis, double inverse_width,
                               int* counts, int rows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int seed = static_cast<int>(first_seed + row * stride);
    const Point center = points[seed];
    const int cell_x = cellCoordinate(center.x, inverse_width, axis);
    const int cell_y = cellCoordinate(center.y, inverse_width, axis);
    int count = 0;
    for (int y = max(0, cell_y - 2); y <= min(axis - 1, cell_y + 2); ++y)
        for (int x = max(0, cell_x - 2); x <= min(axis - 1, cell_x + 2); ++x) {
            const int cell = y * axis + x;
            for (long long i = static_cast<long long>(cell_offsets[cell]) + threadIdx.x;
                 i < cell_offsets[cell + 1]; i += BLOCK_SIZE) {
                const int p = cell_points[i];
                count += p != seed && distance(center, points[p]) < threshold;
            }
        }
    for (int shift = 16; shift; shift >>= 1)
        count += __shfl_down_sync(0xffffffff, count, shift);
    __shared__ int warp_counts[BLOCK_SIZE / 32];
    if ((threadIdx.x & 31) == 0) warp_counts[threadIdx.x >> 5] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < BLOCK_SIZE / 32; ++w) total += warp_counts[w];
        counts[row] = total;
    }
}

// Construct just the threshold neighborhood of each owned seed. Warp ballots
// compact each tile without a global atomic or a dense N x N distance matrix.
__global__ void fillNeighbors(const Point* points, double threshold,
                              long long first_seed, long long stride,
                              const int* cell_offsets, const int* cell_points,
                              int axis, double inverse_width,
                              const EdgeIndex* offsets, int first_row, int rows,
                              EdgeIndex edge_base, int* neighbors) {
    const int row = first_row + blockIdx.x;
    if (row >= first_row + rows) return;
    const int seed = static_cast<int>(first_seed + row * stride);
    const Point center = points[seed];
    const int cell_x = cellCoordinate(center.x, inverse_width, axis);
    const int cell_y = cellCoordinate(center.y, inverse_width, axis);
    const EdgeIndex start = offsets[row] - edge_base;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ int warp_offsets[BLOCK_SIZE / 32];
    __shared__ int written;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    for (int y = max(0, cell_y - 2); y <= min(axis - 1, cell_y + 2); ++y)
        for (int x = max(0, cell_x - 2); x <= min(axis - 1, cell_x + 2); ++x) {
            const int cell = y * axis + x;
            for (long long tile = cell_offsets[cell]; tile < cell_offsets[cell + 1]; tile += BLOCK_SIZE) {
                const long long i = tile + threadIdx.x;
                const int p = i < cell_offsets[cell + 1] ? cell_points[i] : -1;
                const bool eligible = p >= 0 && p != seed && distance(center, points[p]) < threshold;
                const unsigned mask = __ballot_sync(0xffffffff, eligible);
                if (lane == 0) warp_offsets[warp] = __popc(mask);
                __syncthreads();
                if (threadIdx.x == 0) {
                    int next = written;
                    for (int w = 0; w < BLOCK_SIZE / 32; ++w) {
                        const int count = warp_offsets[w];
                        warp_offsets[w] = next;
                        next += count;
                    }
                    written = next;
                }
                __syncthreads();
                if (eligible) {
                    const unsigned lower_lanes = (1u << lane) - 1u;
                    neighbors[start + warp_offsets[warp] + __popc(mask & lower_lanes)] = static_cast<int>(p);
                }
                __syncthreads();
            }
        }
}

// One cooperative block grows a seed's candidate. Scores are updated against
// only the newest member, changing repeated O(k^2) distance work to O(k).
// A negative score records a selected member's position, preserving its order
// without another membership array. DBL_MAX denotes a permanently pruned point.
// Small neighborhoods stay in shared memory throughout candidate growth.
template<bool use_shared>
__global__ void growCandidates(const Point* points, double threshold,
                               long long first_seed, long long stride,
                               const unsigned char* clustered,
                               const EdgeIndex* offsets, int first_row, int rows,
                               EdgeIndex edge_base, const int* neighbors,
                               double* scores, int* sizes, int shared_capacity,
                               bool reuse) {
    const int row = first_row + blockIdx.x;
    if (row >= first_row + rows) return;
    const EdgeIndex start = offsets[row] - edge_base;
    const int count = static_cast<int>(offsets[row + 1] - offsets[row]);
    if (use_shared ? count > shared_capacity : count <= shared_capacity) return;
    const int seed = static_cast<int>(first_seed + row * stride);
    if (clustered[seed]) {
        if (threadIdx.x == 0) sizes[row] = 0;
        return;
    }

    // Removing points outside a cached candidate leaves every greedy choice
    // unchanged. Rebuild only when one of that candidate's members was removed.
    if (reuse && sizes[row] >= 0) {
        bool removed = false;
        for (int j = threadIdx.x; j < count; j += BLOCK_SIZE)
            if (scores[start + j] < 0.0 && clustered[neighbors[start + j]]) removed = true;
        if (!__syncthreads_or(removed)) return;
    }

    extern __shared__ double shared_data[];
    double* values = use_shared ? shared_data : scores + start;
    double* x = shared_data + shared_capacity;
    double* y = x + shared_capacity;
    int* ids = reinterpret_cast<int*>(y + shared_capacity);
    for (int j = threadIdx.x; j < count; j += BLOCK_SIZE) {
        const int p = neighbors[start + j];
        values[j] = clustered[p] ? DBL_MAX : 0.0;
        if (use_shared) { ids[j] = p; x[j] = points[p].x; y[j] = points[p].y; }
    }
    __syncthreads();

    int cardinality = 1, newest = seed;
    for (;;) {
        double closest_distance = DBL_MAX;
        int closest_point = INT_MAX;
        const Point member = points[newest];
        for (int j = threadIdx.x; j < count; j += BLOCK_SIZE) {
            double score = values[j];
            if (score < 0.0 || score == DBL_MAX) continue;
            const int p = use_shared ? ids[j] : neighbors[start + j];
            if (p == newest) {
                values[j] = -static_cast<double>(cardinality - 1);
                continue;
            }
            Point candidate;
            if (use_shared) candidate = {x[j], y[j]};
            else candidate = points[p];
            const double d = distance(candidate, member);
            score = fmax(score, d);
            if (!(score < threshold)) { values[j] = DBL_MAX; continue; }
            values[j] = score;
            if (score < closest_distance || (score == closest_distance && p < closest_point)) {
                closest_distance = score;
                closest_point = p;
            }
        }
        reduceClosest(closest_distance, closest_point);
        if (closest_point == INT_MAX) break;
        newest = closest_point;
        ++cardinality;
    }
    if (use_shared)
        for (int j = threadIdx.x; j < count; j += BLOCK_SIZE) scores[start + j] = values[j];
    if (threadIdx.x == 0) sizes[row] = cardinality;
}

__global__ void extractMembers(long long first_seed, long long stride,
                               const EdgeIndex* offsets, int row, EdgeIndex edge_base,
                               const int* neighbors, const double* scores, int* members) {
    if (threadIdx.x == 0) members[0] = static_cast<int>(first_seed + row * stride);
    const EdgeIndex start = offsets[row] - edge_base;
    const EdgeIndex end = offsets[row + 1] - edge_base;
    for (EdgeIndex j = start + threadIdx.x; j < end; j += BLOCK_SIZE)
        if (scores[j] < 0.0) members[static_cast<int>(-scores[j])] = neighbors[j];
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

struct SeedChoice {
    int size = 0;
    int seed = INT_MAX;
};

// Every device owns cyclically distributed seeds, so input generation order
// and the changing number of unclustered points do not concentrate work on one
// rank. Only points and the accepted cluster are replicated across ranks.
class DeviceCandidates {
    struct Batch { int first, rows; };
    int device_, n_, rows_, shared_capacity_ = 0;
    int grid_axis_ = 1;
    double inverse_width_ = 1.0 / MAX_WIDTH;
    long long first_seed_, stride_;
    double threshold_;
    cudaStream_t stream_ = nullptr;
    Point* points_ = nullptr;
    unsigned char* clustered_ = nullptr;
    EdgeIndex* offsets_ = nullptr;
    int* neighbors_ = nullptr;
    double* scores_ = nullptr;
    int* sizes_ = nullptr;
    int* members_ = nullptr;
    int* cell_offsets_ = nullptr;
    int* cell_points_ = nullptr;
    int* host_members_ = nullptr;
    int* host_sizes_ = nullptr;
    std::vector<EdgeIndex> host_offsets_;
    std::vector<Batch> batches_;
    bool resident_ = true;

    void fill(const Batch& batch) {
        if (host_offsets_[batch.first + batch.rows] == host_offsets_[batch.first]) return;
        fillNeighbors<<<batch.rows, BLOCK_SIZE, 0, stream_>>>(
            points_, threshold_, first_seed_, stride_, cell_offsets_, cell_points_,
            grid_axis_, inverse_width_, offsets_, batch.first,
            batch.rows, host_offsets_[batch.first], neighbors_);
        checkCuda(cudaGetLastError());
    }

    void grow(const Batch& batch, bool reuse) {
        const EdgeIndex base = host_offsets_[batch.first];
        growCandidates<true><<<batch.rows, BLOCK_SIZE, shared_capacity_ * (3 * sizeof(double) + sizeof(int)), stream_>>>(
            points_, threshold_, first_seed_, stride_, clustered_, offsets_, batch.first,
            batch.rows, base, neighbors_, scores_, sizes_, shared_capacity_, reuse);
        checkCuda(cudaGetLastError());
        if (shared_capacity_ == SHARED_CANDIDATES) {
            growCandidates<false><<<batch.rows, BLOCK_SIZE, 0, stream_>>>(
                points_, threshold_, first_seed_, stride_, clustered_, offsets_, batch.first,
                batch.rows, base, neighbors_, scores_, sizes_, shared_capacity_, reuse);
            checkCuda(cudaGetLastError());
        }
    }

public:
    DeviceCandidates(int device, int n, int rows, long long first_seed,
                     long long stride, double threshold)
        : device_(device), n_(n), rows_(rows), first_seed_(first_seed),
          stride_(stride), threshold_(threshold) {}

    void initialize(const std::vector<Point>& points, const NeighborhoodGrid& grid, int sharers) {
        checkCuda(cudaSetDevice(device_));
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        checkCuda(cudaMalloc(&points_, static_cast<size_t>(n_) * sizeof(Point)));
        checkCuda(cudaMalloc(&clustered_, static_cast<size_t>(n_)));
        checkCuda(cudaMalloc(&members_, static_cast<size_t>(n_) * sizeof(int)));
        checkCuda(cudaMallocHost(&host_members_, static_cast<size_t>(n_) * sizeof(int)));
        checkCuda(cudaMemcpyAsync(points_, points.data(), static_cast<size_t>(n_) * sizeof(Point), cudaMemcpyHostToDevice, stream_));
        checkCuda(cudaMemsetAsync(clustered_, 0, static_cast<size_t>(n_), stream_));
        if (rows_ == 0) { checkCuda(cudaStreamSynchronize(stream_)); return; }
        grid_axis_ = grid.axis;
        inverse_width_ = grid.inverse_width;
        checkCuda(cudaMalloc(&cell_offsets_, grid.offsets.size() * sizeof(int)));
        checkCuda(cudaMalloc(&cell_points_, grid.indices.size() * sizeof(int)));
        checkCuda(cudaMemcpyAsync(cell_offsets_, grid.offsets.data(), grid.offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
        checkCuda(cudaMemcpyAsync(cell_points_, grid.indices.data(), grid.indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
        checkCuda(cudaMalloc(&sizes_, static_cast<size_t>(rows_) * sizeof(int)));
        checkCuda(cudaMallocHost(&host_sizes_, static_cast<size_t>(rows_) * sizeof(int)));
        countNeighbors<<<rows_, BLOCK_SIZE, 0, stream_>>>(points_, threshold_, first_seed_, stride_,
            cell_offsets_, cell_points_, grid_axis_, inverse_width_, sizes_, rows_);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpyAsync(host_sizes_, sizes_, static_cast<size_t>(rows_) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        checkCuda(cudaStreamSynchronize(stream_));
        host_offsets_.resize(static_cast<size_t>(rows_) + 1, 0);
        int max_degree = 0;
        for (int row = 0; row < rows_; ++row) {
            host_offsets_[row + 1] = host_offsets_[row] + host_sizes_[row];
            max_degree = std::max(max_degree, host_sizes_[row]);
        }
        shared_capacity_ = std::min(max_degree, SHARED_CANDIDATES);
        checkCuda(cudaMalloc(&offsets_, host_offsets_.size() * sizeof(EdgeIndex)));
        checkCuda(cudaMemcpyAsync(offsets_, host_offsets_.data(), host_offsets_.size() * sizeof(EdgeIndex), cudaMemcpyHostToDevice, stream_));
        size_t free_bytes = 0, total_bytes = 0;
        checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes));
        const EdgeIndex capacity = (free_bytes / sharers / 4 * 3) / (sizeof(int) + sizeof(double));
        if (static_cast<EdgeIndex>(max_degree) > capacity)
            throw std::runtime_error("Insufficient GPU memory for a single seed's neighborhood");
        EdgeIndex largest_batch = 0;
        int first = 0;
        for (int row = 0; row < rows_; ++row) {
            if (host_offsets_[row + 1] - host_offsets_[first] > capacity) {
                batches_.push_back({first, row - first});
                largest_batch = std::max(largest_batch, host_offsets_[row] - host_offsets_[first]);
                first = row;
            }
        }
        batches_.push_back({first, rows_ - first});
        largest_batch = std::max(largest_batch, host_offsets_.back() - host_offsets_[first]);
        resident_ = batches_.size() == 1;
        // Allocate at least one element for zero-neighbor datasets.
        const size_t allocation = static_cast<size_t>(std::max<EdgeIndex>(1, largest_batch));
        checkCuda(cudaMalloc(&neighbors_, allocation * sizeof(int)));
        checkCuda(cudaMalloc(&scores_, allocation * sizeof(double)));
        checkCuda(cudaMemsetAsync(sizes_, 0xff, static_cast<size_t>(rows_) * sizeof(int), stream_));
        if (resident_) fill(batches_[0]);
        checkCuda(cudaStreamSynchronize(stream_));
    }

    SeedChoice evaluate() {
        checkCuda(cudaSetDevice(device_));
        if (rows_ == 0) return {};
        for (const auto& batch : batches_) {
            if (!resident_) fill(batch);
            grow(batch, resident_);
        }
        checkCuda(cudaMemcpyAsync(host_sizes_, sizes_, static_cast<size_t>(rows_) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        checkCuda(cudaStreamSynchronize(stream_));
        int best_size = 0;
        #pragma omp simd reduction(max:best_size)
        for (int row = 0; row < rows_; ++row) {
            best_size = std::max(best_size, host_sizes_[row]);
        }
        int best_row = INT_MAX;
        #pragma omp simd reduction(min:best_row)
        for (int row = 0; row < rows_; ++row)
            if (host_sizes_[row] == best_size) best_row = std::min(best_row, row);
        return best_size > 0 ? SeedChoice{best_size, static_cast<int>(first_seed_ + best_row * stride_)} : SeedChoice{};
    }

    void getMembers(int seed, std::vector<int>& members) {
        checkCuda(cudaSetDevice(device_));
        const int row = static_cast<int>((seed - first_seed_) / stride_);
        EdgeIndex base = 0;
        if (!resident_) {
            // A streamed batch has overwritten scratch storage. Rebuild only
            // the winning row, rather than retaining all candidates on the host.
            const Batch winner{row, 1};
            fill(winner);
            grow(winner, false);
            base = host_offsets_[row];
        }
        extractMembers<<<1, BLOCK_SIZE, 0, stream_>>>(first_seed_, stride_, offsets_, row, base, neighbors_, scores_, members_);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpyAsync(members.data(), members_, members.size() * sizeof(int), cudaMemcpyDeviceToHost, stream_));
        checkCuda(cudaStreamSynchronize(stream_));
    }

    void remove(const std::vector<int>& members) {
        checkCuda(cudaSetDevice(device_));
        if (rows_ == 0) return;
        // Retain the source until the next evaluate() synchronizes this stream;
        // non-root ranks release their broadcast vector after this round.
        std::copy(members.begin(), members.end(), host_members_);
        checkCuda(cudaMemcpyAsync(members_, host_members_, members.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
        markClustered<<<(members.size() + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream_>>>(
            clustered_, members_, static_cast<int>(members.size()));
        checkCuda(cudaGetLastError());
        // The same stream orders this update before the next candidate round.
    }

    ~DeviceCandidates() {
        cudaSetDevice(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        cudaFree(points_); cudaFree(clustered_); cudaFree(offsets_);
        cudaFree(neighbors_); cudaFree(scores_); cudaFree(sizes_); cudaFree(members_);
        cudaFree(cell_offsets_); cudaFree(cell_points_);
        if (host_members_) cudaFreeHost(host_members_);
        if (host_sizes_) cudaFreeHost(host_sizes_);
        if (stream_) cudaStreamDestroy(stream_);
    }
};

// Workers never call MPI, including on CUDA failures. Exceptions are collected
// and propagated on the initial thread, where the caller can abort all ranks.
template<class Function>
static void forDevices(int count, Function function) {
    std::vector<std::string> errors(count);
    #pragma omp parallel for num_threads(count) schedule(static)
    for (int i = 0; i < count; ++i) {
        try { function(i); }
        catch (const std::exception& e) { errors[i] = e.what(); }
    }
    for (const auto& error : errors)
        if (!error.empty()) throw std::runtime_error(error);
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold,
                                  const ParallelRuntime& runtime) {
    const int n = static_cast<int>(points.size());
    const int device_count = static_cast<int>(runtime.devices.size());
    const int local_seeds = runtime.rank < n ? (n - 1 - runtime.rank) / runtime.ranks + 1 : 0;
    const NeighborhoodGrid grid(points, threshold);
    std::vector<std::unique_ptr<DeviceCandidates>> devices(device_count);
    for (int i = 0; i < device_count; ++i) {
        const int rows = i < local_seeds ? (local_seeds - 1 - i) / device_count + 1 : 0;
        devices[i] = std::make_unique<DeviceCandidates>(runtime.devices[i], n, rows,
            runtime.rank + static_cast<long long>(runtime.ranks) * i,
            static_cast<long long>(runtime.ranks) * device_count, threshold);
    }
    forDevices(device_count, [&](int i) { devices[i]->initialize(points, grid, runtime.device_sharers[i]); });
    std::vector<SeedChoice> choices(device_count);
    std::vector<Cluster> clusters;
    std::vector<unsigned char> clustered(runtime.rank == 0 ? n : 0, 0);
    int remaining = n;
    while (remaining > 0) {
        forDevices(device_count, [&](int i) { choices[i] = devices[i]->evaluate(); });
        // MPI_MINLOC selects the largest size and then the smallest global seed,
        // exactly matching the sequential traversal through unclustered indices.
        int local_best[2] = {0, INT_MAX}, global_best[2];
        for (const auto& choice : choices)
            if (-choice.size < local_best[0] || (-choice.size == local_best[0] && choice.seed < local_best[1])) {
                local_best[0] = -choice.size;
                local_best[1] = choice.seed;
            }
        checkMpi(MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD));
        const int cardinality = -global_best[0], seed = global_best[1];
        if (cardinality <= 0 || cardinality > remaining || seed < 0 || seed >= n)
            throw std::runtime_error("Invalid global candidate selection");
        if (cardinality == 1) {
            // No remaining pair meets the threshold. The sequential loop now
            // emits singleton clusters in point order, without further searches.
            if (runtime.rank == 0)
                for (int p = 0; p < n; ++p)
                    if (!clustered[p]) clusters.push_back({{p}, p});
            break;
        }
        const int owner = seed % runtime.ranks;
        std::vector<int> members(cardinality);
        if (runtime.rank == owner)
            devices[(seed / runtime.ranks) % device_count]->getMembers(seed, members);
        checkMpi(MPI_Bcast(members.data(), cardinality, MPI_INT, owner, MPI_COMM_WORLD));
        forDevices(device_count, [&](int i) { devices[i]->remove(members); });
        remaining -= cardinality;
        if (runtime.rank == 0) {
            for (int p : members) clustered[p] = 1;
            clusters.push_back({std::move(members), seed});
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
    
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic) num_threads(std::min(omp_get_max_threads(), 16))
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
        
        diameters[c] = max_diameter;
    }

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

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
    try {
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
                if (runtime.rank == 0) printUsage(argv[0]);
                return 0;
            } else {
                if (runtime.rank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
        }
    
        if (num_points <= 0 || threshold <= 0.0) {
            if (runtime.rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
            return 1;
        }
    
        runtime.initializeDevices();
        if (runtime.rank == 0) {
            printf("QT Clustering Benchmark\n");
            printf("Number of points: %d\n", num_points);
            printf("Distance threshold: %.2f\n", threshold);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
        }
    
        // Generate synthetic data
        std::vector<Point> points(num_points);
        if (runtime.rank == 0) generateSyntheticData(points, num_points);
        // MPI counts are ints; chunking also supports datasets beyond INT_MAX bytes.
        unsigned char* point_bytes = reinterpret_cast<unsigned char*>(points.data());
        const size_t point_bytes_count = points.size() * sizeof(Point);
        for (size_t offset = 0; offset < point_bytes_count;) {
            const int count = static_cast<int>(std::min<size_t>(point_bytes_count - offset, INT_MAX));
            checkMpi(MPI_Bcast(point_bytes + offset, count, MPI_BYTE, 0, MPI_COMM_WORLD));
            offset += count;
        }
        checkMpi(MPI_Barrier(MPI_COMM_WORLD));
    
        // Perform QT clustering
        auto cluster_start = std::chrono::high_resolution_clock::now();
    
        const std::vector<Cluster> clusters = qtClustering(points, threshold, runtime);
    
        auto cluster_end = std::chrono::high_resolution_clock::now();
    
        const double local_seconds = std::chrono::duration<double>(cluster_end - cluster_start).count();
        double elapsed_seconds = 0.0;
        checkMpi(MPI_Reduce(&local_seconds, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (runtime.rank != 0) return 0;
        printf("Clustering time: %lld ms\n", static_cast<long long>(elapsed_seconds * 1000.0));
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
        const double time_sec = elapsed_seconds;
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
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", runtime.rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

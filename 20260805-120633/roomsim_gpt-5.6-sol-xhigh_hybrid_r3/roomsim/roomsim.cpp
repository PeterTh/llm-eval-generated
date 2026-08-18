/**
 * Hybrid MPI/OpenMP/CUDA Room Response Simulation Benchmark
 *
 * One MPI rank drives one GPU.  Receiver triangles are divided into contiguous
 * row blocks across ranks, CUDA computes and retains the corresponding Kij/Tau
 * rows, and MPI_Allgatherv exchanges the newly computed radiosity after every
 * timestep.  OpenMP parallelizes host initialization and validation work.
 */

#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / static_cast<val_t>(NUM_RAYS);
constexpr val_t EPSILON = 1.0e-6f;
constexpr int CUDA_BLOCK_SIZE = 256;

static int worldRank = 0;

[[noreturn]] static void failMpi(const char* expression, int error, const char* file, int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: MPI failure at %s:%d: %s: %.*s\n",
                 worldRank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

[[noreturn]] static void failCuda(const char* expression, cudaError_t error,
                                  const char* file, int line) {
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d: %s: %s\n",
                 worldRank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        const int mpi_check_error = (call);                                    \
        if (mpi_check_error != MPI_SUCCESS)                                    \
            failMpi(#call, mpi_check_error, __FILE__, __LINE__);               \
    } while (false)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t cuda_check_error = (call);                           \
        if (cuda_check_error != cudaSuccess)                                   \
            failCuda(#call, cuda_check_error, __FILE__, __LINE__);             \
    } while (false)

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x_, val_t y_, val_t z_)
        : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }
    __host__ __device__ Vec3 operator-(const Vec3& o) const {
        return {x - o.x, y - o.y, z - o.z};
    }
    __host__ __device__ Vec3 operator*(val_t s) const {
        return {x * s, y * s, z * s};
    }
    __host__ __device__ Vec3 operator/(val_t s) const {
        return {x / s, y / s, z / s};
    }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }
    __host__ __device__ val_t dot(const Vec3& o) const {
        return x * o.x + y * o.y + z * o.z;
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z,
                x * o.y - y * o.x};
    }
    __host__ __device__ val_t squaredNorm() const {
        return x * x + y * y + z * z;
    }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        const val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 normal;

    Triangle() = default;
    Triangle(const Vec3& a_, const Vec3& b_, const Vec3& c_)
        : a(a_), b(b_), c(c_), normal((b_ - a_).cross(c_ - a_).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    val_t area() const { return 0.5f * (b - a).cross(c - a).norm(); }
};

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        for (int level = 0; level < subdivisions; ++level) {
            std::vector<std::array<idx_t, 3>> newFaces;
            newFaces.reserve(faces.size() * 4);
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto midpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                const auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                const auto found = midpointCache.find(key);
                if (found != midpointCache.end()) return found->second;
                const Vec3 point = ((vertices[i1] + vertices[i2]) / 2.0f).normalized() * radius;
                const idx_t index = static_cast<idx_t>(vertices.size());
                vertices.push_back(point);
                midpointCache.emplace(key, index);
                return index;
            };

            for (const auto& face : faces) {
                const idx_t a = midpoint(face[0], face[1]);
                const idx_t b = midpoint(face[1], face[2]);
                const idx_t c = midpoint(face[2], face[0]);
                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        triangles.reserve(faces.size());
        for (const auto& face : faces)
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
    }
};

struct RowPartition {
    size_t begin = 0;
    size_t count = 0;
    std::vector<int> counts;
    std::vector<int> displacements;
};

static RowPartition partitionRows(size_t n, int rank, int ranks) {
    RowPartition result;
    result.counts.resize(static_cast<size_t>(ranks));
    result.displacements.resize(static_cast<size_t>(ranks));
    const size_t quotient = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = quotient + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t first = static_cast<size_t>(r) * quotient +
                             std::min(static_cast<size_t>(r), remainder);
        result.counts[static_cast<size_t>(r)] = static_cast<int>(rows);
        result.displacements[static_cast<size_t>(r)] = static_cast<int>(first);
    }
    result.begin = static_cast<size_t>(result.displacements[static_cast<size_t>(rank)]);
    result.count = static_cast<size_t>(result.counts[static_cast<size_t>(rank)]);
    return result;
}

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    val_t reflectivity = 0.8f;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> distances;
    RowPartition rows;
};

template <typename T>
static T* deviceAllocate(size_t elements) {
    T* pointer = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                          std::max<size_t>(elements, 1) * sizeof(T)));
    return pointer;
}

template <typename T>
static T* pinnedAllocate(size_t elements) {
    T* pointer = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pointer),
                             std::max<size_t>(elements, 1) * sizeof(T),
                             cudaHostAllocPortable));
    return pointer;
}

struct DeviceData {
    Triangle* triangles = nullptr;
    val_t* areas = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* localStep = nullptr;
    val_t* distances = nullptr;
    unsigned long long* counter = nullptr;
    val_t* hostLocal = nullptr;
    val_t* hostGlobal = nullptr;

    void release() noexcept {
        cudaFree(triangles);
        cudaFree(areas);
        cudaFree(kij);
        cudaFree(tau);
        cudaFree(radB);
        cudaFree(localStep);
        cudaFree(distances);
        cudaFree(counter);
        cudaFreeHost(hostLocal);
        cudaFreeHost(hostGlobal);
        triangles = nullptr;
        areas = nullptr;
        kij = nullptr;
        tau = nullptr;
        radB = nullptr;
        localStep = nullptr;
        distances = nullptr;
        counter = nullptr;
        hostLocal = nullptr;
        hostGlobal = nullptr;
    }
};

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ val_t uniform01(uint64_t key) {
    return static_cast<val_t>(mix64(key) >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ Vec3 sampleTriangle(const Triangle& triangle,
                                               val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u +
           (triangle.c - triangle.a) * v;
}

/*
 * Each half warp evaluates one triangle pair: its 16 lanes are the original
 * 16 visibility samples.  The generated icosphere is a convex boundary, hence
 * a chord between two boundary points is wholly inside it and cannot intersect
 * any third boundary triangle.  This makes the old octree visibility query
 * identically false while avoiding an expensive divergent traversal.
 */
__global__ void precomputeKernel(const Triangle* __restrict__ triangles,
                                 size_t n, size_t rowBegin, size_t localRows,
                                 val_t* __restrict__ kij,
                                 int* __restrict__ tau) {
    const size_t rayTask = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pair = rayTask / NUM_RAYS;
    const size_t pairCount = localRows * n;
    if (pair >= pairCount) return;

    const int lane = threadIdx.x & (NUM_RAYS - 1);
    const size_t localI = pair / n;
    const size_t i = rowBegin + localI;
    const size_t j = pair - localI * n;
    const Triangle triI = triangles[i];
    const Triangle triJ = triangles[j];

    if (lane == 0) {
        if (i == j) {
            tau[pair] = 0;
        } else {
            const Vec3 ci = (triI.a + triI.b + triI.c) / 3.0f;
            const Vec3 cj = (triJ.a + triJ.b + triJ.c) / 3.0f;
            tau[pair] = static_cast<int>(ceilf((ci - cj).norm() * INV_WAVE_SPEED));
        }
    }

    if (i == j || triI.normal.dot(triJ.normal) > 0.99f) {
        if (lane == 0) kij[pair] = ZERO;
        return;
    }

    const uint64_t pairKey = (static_cast<uint64_t>(i) * n + j) *
                             0xd2b74407b1ce6e93ULL + 42ULL;
    const uint64_t rayKey = pairKey ^
                            (static_cast<uint64_t>(lane) * 0x9e3779b97f4a7c15ULL);
    const val_t uI = uniform01(rayKey ^ 0x243f6a8885a308d3ULL);
    const val_t vI = uniform01(rayKey ^ 0x13198a2e03707344ULL);
    const val_t uJ = uniform01(rayKey ^ 0xa4093822299f31d0ULL);
    const val_t vJ = uniform01(rayKey ^ 0x082efa98ec4e6c89ULL);
    const Vec3 pI = sampleTriangle(triI, uI, vI);
    const Vec3 pJ = sampleTriangle(triJ, uJ, vJ);
    const Vec3 direction = pJ - pI;
    const val_t distanceSquared = direction.squaredNorm();
    val_t contribution = ZERO;
    if (distanceSquared >= EPSILON) {
        const val_t inverseDistance = 1.0f / sqrtf(distanceSquared);
        const val_t cosI = fmaxf(ZERO, direction.dot(triI.normal) * inverseDistance);
        const val_t cosJ = fmaxf(ZERO, (-direction).dot(triJ.normal) * inverseDistance);
        contribution = (cosI * cosJ) / (PI * distanceSquared);
    }

    const unsigned int halfWarpMask = (threadIdx.x & 16) ? 0xffff0000U : 0x0000ffffU;
#pragma unroll
    for (int offset = NUM_RAYS / 2; offset > 0; offset >>= 1)
        contribution += __shfl_down_sync(halfWarpMask, contribution, offset, NUM_RAYS);
    if (lane == 0) kij[pair] = contribution * INV_NUM_RAYS;
}

__global__ void propagationKernel(const val_t* __restrict__ kij,
                                  const int* __restrict__ tau,
                                  const val_t* __restrict__ areas,
                                  const val_t* __restrict__ radB,
                                  size_t n, size_t rowBegin, size_t localRows,
                                  size_t sourceIndex, int timestep,
                                  int numTimesteps, val_t reflectivity,
                                  val_t* __restrict__ localStep) {
    const size_t localI = blockIdx.x;
    if (localI >= localRows) return;
    const size_t i = rowBegin + localI;
    const size_t rowOffset = localI * n;
    val_t partial = ZERO;

    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        const int delay = tau[rowOffset + j];
        if (i == j || timestep < delay) continue;
        const val_t factor = kij[rowOffset + j];
        if (factor <= ZERO) continue;
        const size_t sourceTime = static_cast<size_t>(timestep - delay);
        const val_t sourceRadiosity = radB[sourceTime * n + j];
        if (sourceRadiosity <= ZERO) continue;
        partial += fminf(factor * areas[j], ONE) * sourceRadiosity;
    }

    __shared__ val_t sums[CUDA_BLOCK_SIZE];
    sums[threadIdx.x] = partial;
    __syncthreads();
#pragma unroll
    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < numTimesteps / 2)
                                   ? ONE : ZERO;
        localStep[localI] = reflectivity * sums[0] + emission;
    }
}

__global__ void distanceKernel(const val_t* __restrict__ radB,
                               size_t n, int timesteps, size_t sourceIndex,
                               size_t rowBegin, size_t localRows,
                               val_t* __restrict__ distances) {
    const size_t localI = blockIdx.x;
    if (localI >= localRows) return;
    const size_t i = rowBegin + localI;
    val_t threadBestCorrelation = ZERO;
    int threadBestTime = 0;

    for (int lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (int tt = lag; tt < timesteps; ++tt) {
            const val_t pB = radB[static_cast<size_t>(tt) * n + i];
            const val_t pS = radB[static_cast<size_t>(tt - lag) * n + sourceIndex];
            correlation += pS * pB;
        }
        if (correlation > threadBestCorrelation ||
            (correlation == threadBestCorrelation && lag < threadBestTime)) {
            threadBestCorrelation = correlation;
            threadBestTime = lag;
        }
    }

    __shared__ val_t correlations[CUDA_BLOCK_SIZE];
    __shared__ int times[CUDA_BLOCK_SIZE];
    correlations[threadIdx.x] = threadBestCorrelation;
    times[threadIdx.x] = threadBestTime;
    __syncthreads();
    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const val_t otherCorrelation = correlations[threadIdx.x + offset];
            const int otherTime = times[threadIdx.x + offset];
            if (otherCorrelation > correlations[threadIdx.x] ||
                (otherCorrelation == correlations[threadIdx.x] &&
                 otherTime < times[threadIdx.x])) {
                correlations[threadIdx.x] = otherCorrelation;
                times[threadIdx.x] = otherTime;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        distances[localI] = WAVE_SPEED * static_cast<val_t>(times[0]);
}

__global__ void countPositiveKernel(const val_t* values, size_t elements,
                                    unsigned long long* result) {
    unsigned long long localCount = 0;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = first; index < elements; index += stride)
        localCount += values[index] > EPSILON ? 1ULL : 0ULL;

    __shared__ unsigned long long counts[CUDA_BLOCK_SIZE];
    counts[threadIdx.x] = localCount;
    __syncthreads();
    for (int offset = CUDA_BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) counts[threadIdx.x] += counts[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, counts[0]);
}

__global__ void countReceivedKernel(const val_t* radB, size_t n, int timesteps,
                                    size_t rowBegin, size_t localRows,
                                    unsigned long long* result) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localRows) return;
    const size_t i = rowBegin + localI;
    for (int timestep = 0; timestep < timesteps; ++timestep) {
        if (radB[static_cast<size_t>(timestep) * n + i] > EPSILON) {
            atomicAdd(result, 1ULL);
            return;
        }
    }
}

static void initializeSimulation(SimulationState& state, int subdivisions,
                                 size_t timesteps, size_t sourceIndex,
                                 val_t reflectivity, int rank, int ranks) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIndex % state.numTriangles;
    state.reflectivity = reflectivity;
    state.rows = partitionRows(state.numTriangles, rank, ranks);
    state.areas.resize(state.numTriangles);

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i)
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();

    if (rank == 0) state.distances.resize(state.numTriangles, ZERO);
}

static void allocateDeviceData(const SimulationState& state, DeviceData& data) {
    const size_t n = state.numTriangles;
    const size_t localRows = state.rows.count;
    const size_t localPairs = localRows * n;
    data.triangles = deviceAllocate<Triangle>(n);
    data.areas = deviceAllocate<val_t>(n);
    data.kij = deviceAllocate<val_t>(localPairs);
    data.tau = deviceAllocate<int>(localPairs);
    data.radB = deviceAllocate<val_t>(state.numTimesteps * n);
    data.localStep = deviceAllocate<val_t>(localRows);
    data.distances = deviceAllocate<val_t>(localRows);
    data.counter = deviceAllocate<unsigned long long>(1);
    data.hostLocal = pinnedAllocate<val_t>(localRows);
    data.hostGlobal = pinnedAllocate<val_t>(n);

    CUDA_CHECK(cudaMemcpy(data.triangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.areas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(data.radB, 0, state.numTimesteps * n * sizeof(val_t)));
}

static void warmUpCuda(const SimulationState& state, DeviceData& data) {
    // Force context/module loading outside the benchmark's computation phases.
    precomputeKernel<<<1, CUDA_BLOCK_SIZE>>>(
        data.triangles, state.numTriangles, state.rows.begin, 0, data.kij, data.tau);
    propagationKernel<<<1, CUDA_BLOCK_SIZE>>>(
        data.kij, data.tau, data.areas, data.radB, state.numTriangles,
        state.rows.begin, 0, state.sourceIndex, 0,
        static_cast<int>(state.numTimesteps), state.reflectivity, data.localStep);
    distanceKernel<<<1, CUDA_BLOCK_SIZE>>>(
        data.radB, state.numTriangles, static_cast<int>(state.numTimesteps),
        state.sourceIndex, state.rows.begin, 0, data.distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static bool queryCudaAwareMpi() {
#if defined(OMPI_HAVE_MPI_EXT_CUDA)
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

static void computePrecomputation(const SimulationState& state, DeviceData& data) {
    const size_t pairs = state.rows.count * state.numTriangles;
    const size_t tasks = pairs * NUM_RAYS;
    if (tasks == 0) return;
    const size_t blocks = (tasks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    precomputeKernel<<<static_cast<unsigned int>(blocks), CUDA_BLOCK_SIZE>>>(
        data.triangles, state.numTriangles, state.rows.begin, state.rows.count,
        data.kij, data.tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void runSimulation(const SimulationState& state, DeviceData& data,
                          int rank, bool cudaAwareMpi) {
    const int localCount = static_cast<int>(state.rows.count);
    for (int timestep = 0; timestep < static_cast<int>(state.numTimesteps); ++timestep) {
        if (state.rows.count != 0) {
            propagationKernel<<<static_cast<unsigned int>(state.rows.count), CUDA_BLOCK_SIZE>>>(
                data.kij, data.tau, data.areas, data.radB, state.numTriangles,
                state.rows.begin, state.rows.count, state.sourceIndex, timestep,
                static_cast<int>(state.numTimesteps), state.reflectivity,
                data.localStep);
            CUDA_CHECK(cudaGetLastError());
        }

        if (cudaAwareMpi) {
            // CUDA-aware MPI avoids both PCIe staging copies and can use GPUDirect
            // transports between nodes.  Synchronization makes the send buffer
            // ready for MPI implementations that do not inspect CUDA streams.
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_CHECK(MPI_Allgatherv(
                data.localStep, localCount, MPI_FLOAT,
                data.radB + static_cast<size_t>(timestep) * state.numTriangles,
                state.rows.counts.data(), state.rows.displacements.data(), MPI_FLOAT,
                MPI_COMM_WORLD));
        } else {
            if (state.rows.count != 0)
                CUDA_CHECK(cudaMemcpy(data.hostLocal, data.localStep,
                                      state.rows.count * sizeof(val_t),
                                      cudaMemcpyDeviceToHost));
            MPI_CHECK(MPI_Allgatherv(data.hostLocal, localCount, MPI_FLOAT,
                                     data.hostGlobal, state.rows.counts.data(),
                                     state.rows.displacements.data(), MPI_FLOAT,
                                     MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpy(
                data.radB + static_cast<size_t>(timestep) * state.numTriangles,
                data.hostGlobal, state.numTriangles * sizeof(val_t),
                cudaMemcpyHostToDevice));
        }

        if (rank == 0 && ((timestep + 1) % 10 == 0 ||
                          timestep + 1 == static_cast<int>(state.numTimesteps)))
            std::printf("  Timestep %d/%zu\n", timestep + 1, state.numTimesteps);
    }
}

static void computeDistances(SimulationState& state, DeviceData& data, int rank) {
    if (state.rows.count != 0) {
        distanceKernel<<<static_cast<unsigned int>(state.rows.count), CUDA_BLOCK_SIZE>>>(
            data.radB, state.numTriangles, static_cast<int>(state.numTimesteps),
            state.sourceIndex, state.rows.begin, state.rows.count, data.distances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(data.hostLocal, data.distances,
                              state.rows.count * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    MPI_CHECK(MPI_Gatherv(data.hostLocal, static_cast<int>(state.rows.count), MPI_FLOAT,
                          rank == 0 ? state.distances.data() : nullptr,
                          state.rows.counts.data(), state.rows.displacements.data(),
                          MPI_FLOAT, 0, MPI_COMM_WORLD));
}

static unsigned long long countLocalPositiveKij(const SimulationState& state,
                                                DeviceData& data) {
    const size_t elements = state.rows.count * state.numTriangles;
    CUDA_CHECK(cudaMemset(data.counter, 0, sizeof(unsigned long long)));
    if (elements != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>((elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, 4096));
        countPositiveKernel<<<blocks, CUDA_BLOCK_SIZE>>>(data.kij, elements, data.counter);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long result = 0;
    CUDA_CHECK(cudaMemcpy(&result, data.counter, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

static unsigned long long countLocalReceived(const SimulationState& state,
                                             DeviceData& data) {
    CUDA_CHECK(cudaMemset(data.counter, 0, sizeof(unsigned long long)));
    if (state.rows.count != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            (state.rows.count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        countReceivedKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            data.radB, state.numTriangles, static_cast<int>(state.numTimesteps),
            state.rows.begin, state.rows.count, data.counter);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long result = 0;
    CUDA_CHECK(cudaMemcpy(&result, data.counter, sizeof(result), cudaMemcpyDeviceToHost));
    return result;
}

static bool validateResults(const SimulationState& state, DeviceData& data,
                            int rank) {
    val_t localMin = std::numeric_limits<val_t>::max();
    val_t localMax = std::numeric_limits<val_t>::lowest();
    double localSum = 0.0;
    long long localNonZero = 0;
    int localValid = 1;

#pragma omp parallel for reduction(min:localMin) reduction(max:localMax) \
    reduction(+:localSum,localNonZero) reduction(&:localValid) schedule(static)
    for (long long localI = 0; localI < static_cast<long long>(state.rows.count); ++localI) {
        const val_t distance = data.hostLocal[static_cast<size_t>(localI)];
        localMin = std::min(localMin, distance);
        localMax = std::max(localMax, distance);
        localSum += static_cast<double>(distance);
        localNonZero += distance > EPSILON ? 1 : 0;
        localValid &= (distance >= ZERO && std::isfinite(distance)) ? 1 : 0;
    }

    val_t globalMin = ZERO;
    val_t globalMax = ZERO;
    double globalSum = 0.0;
    long long globalNonZero = 0;
    int globalValid = 0;
    MPI_CHECK(MPI_Reduce(&localMin, &globalMin, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localMax, &globalMax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localNonZero, &globalNonZero, 1, MPI_LONG_LONG, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, 0,
                         MPI_COMM_WORLD));

    const unsigned long long localKij = countLocalPositiveKij(state, data);
    const unsigned long long localReceived = countLocalReceived(state, data);
    unsigned long long globalKij = 0;
    unsigned long long globalReceived = 0;
    MPI_CHECK(MPI_Reduce(&localKij, &globalKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localReceived, &globalReceived, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, 0, MPI_COMM_WORLD));

    int passed = 1;
    if (rank == 0) {
        std::printf("\nValidation:\n");
        std::printf("  Distance range: [%.4f, %.4f]\n", globalMin, globalMax);
        std::printf("  Average distance: %.4f\n",
                    globalSum / static_cast<double>(state.numTriangles));
        std::printf("  Non-zero distances: %lld/%zu\n", globalNonZero,
                    state.numTriangles);
        const val_t sourceDistance = state.distances[state.sourceIndex];
        if (sourceDistance > WAVE_SPEED * 2)
            std::printf("  WARNING: Source triangle distance is non-zero: %.4f\n",
                        sourceDistance);
        std::printf("  Triangles receiving energy: %llu/%zu\n", globalReceived,
                    state.numTriangles);
        const unsigned long long matrixElements =
            static_cast<unsigned long long>(state.numTriangles) * state.numTriangles;
        std::printf("  Non-zero form factors: %llu/%llu (%.2f%%)\n",
                    globalKij, matrixElements,
                    100.0 * static_cast<double>(globalKij) /
                        static_cast<double>(matrixElements));
        passed = globalValid && globalReceived != 0 && globalKij != 0;
        std::printf("  Validation: %s\n", passed ? "PASSED" : "FAILED");
    }
    MPI_CHECK(MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return passed != 0;
}

static uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

static int getSubdivisionsForTriangleCount(int targetTriangles) {
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        ++subdivisions;
        triangles *= 4;
    }
    return subdivisions;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Target number of triangles (default: 320)\n");
    std::printf("  -t <num>     Number of timesteps (default: 50)\n");
    std::printf("  -s <num>     Source triangle index (default: 0)\n");
    std::printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -o           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static double maxPhaseTime(double localSeconds, int rank) {
    double maximum = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    return rank == 0 ? maximum : localSeconds;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0)
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIndex = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResultsFlag = false;
    bool argumentsValid = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIndex = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-o") == 0) {
            printResultsFlag = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (worldRank == 0) std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            break;
        }
    }

    if (targetTriangles <= 0 || timesteps <= 0) argumentsValid = false;
    if (showHelp || !argumentsValid) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localSize));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) std::fprintf(stderr, "CUDA device required but none was found\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int selectedDevice = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(selectedDevice));
    CUDA_CHECK(cudaFree(nullptr));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, selectedDevice));

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIndex), reflectivity,
                         worldRank, worldSize);

    if (worldRank == 0) {
        std::printf("Room Response Simulation Benchmark\n");
        std::printf("===================================\n");
        std::printf("Target triangles: %d (using %d subdivisions)\n",
                    targetTriangles, subdivisions);
        std::printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        std::printf("Timesteps: %d\n", timesteps);
        std::printf("Source triangle: %d\n", sourceIndex);
        std::printf("Reflectivity: %.2f\n", reflectivity);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid runtime: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "%d CUDA device(s) on rank-0 node\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("Rank 0 GPU: %s; rank/GPU mapping: local_rank %% device_count\n",
                    deviceProperties.name);
        if (localSize > deviceCount)
            std::printf("WARNING: %d local MPI ranks share %d GPU(s); one rank per GPU is recommended\n",
                        localSize, deviceCount);
        std::printf("Distributed matrix rows per rank: %zu..%zu\n\n",
                    state.numTriangles / static_cast<size_t>(worldSize),
                    (state.numTriangles + static_cast<size_t>(worldSize) - 1) /
                        static_cast<size_t>(worldSize));
    }

    DeviceData data;
    allocateDeviceData(state, data);
    warmUpCuda(state, data);
    const bool cudaAwareMpi = queryCudaAwareMpi();

    if (worldRank == 0)
        std::printf("MPI radiosity transport: %s\n\n",
                    cudaAwareMpi ? "CUDA-aware direct device buffers"
                                 : "portable pinned-host staging");

    if (worldRank == 0)
        std::printf("Computing distributed form factors and time delays on CUDA devices...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double phaseStart = MPI_Wtime();
    computePrecomputation(state, data);
    double precomputationSeconds = maxPhaseTime(MPI_Wtime() - phaseStart, worldRank);
    if (worldRank == 0)
        std::printf("Precomputation time: %ld ms\n\n",
                    static_cast<long>(precomputationSeconds * 1000.0));

    if (worldRank == 0)
        std::printf("Running CUDA wave propagation with MPI timestep exchange...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    phaseStart = MPI_Wtime();
    runSimulation(state, data, worldRank, cudaAwareMpi);
    CUDA_CHECK(cudaDeviceSynchronize());
    double simulationSeconds = maxPhaseTime(MPI_Wtime() - phaseStart, worldRank);
    if (worldRank == 0)
        std::printf("Simulation time: %ld ms\n\n",
                    static_cast<long>(simulationSeconds * 1000.0));

    if (worldRank == 0)
        std::printf("Computing distributed distances on CUDA devices...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    phaseStart = MPI_Wtime();
    computeDistances(state, data, worldRank);
    double distanceSeconds = maxPhaseTime(MPI_Wtime() - phaseStart, worldRank);
    const double totalSeconds = precomputationSeconds + simulationSeconds + distanceSeconds;

    int exitCode = 0;
    if (worldRank == 0) {
        const long distanceMs = static_cast<long>(distanceSeconds * 1000.0);
        const long totalMs = static_cast<long>(totalSeconds * 1000.0);
        const double n = static_cast<double>(state.numTriangles);
        const double t = static_cast<double>(state.numTimesteps);
        const size_t localMatrixBytes = state.rows.count * state.numTriangles *
                                        (sizeof(val_t) + sizeof(int));
        const size_t replicatedBytes = state.numTimesteps * state.numTriangles *
                                       sizeof(val_t);
        std::printf("Distance computation time: %ld ms\n\n", distanceMs);
        std::printf("Total computation time: %ld ms\n", totalMs);
        std::printf("\nPerformance:\n");
        std::printf("  Triangles: %zu\n", state.numTriangles);
        std::printf("  Timesteps: %zu\n", state.numTimesteps);
        std::printf("  Form factor computations: %.2e\n", n * n);
        std::printf("  Simulation operations: %.2e\n", n * n * t);
        std::printf("  Distance computations: %.2e\n", n * t * t);
        std::printf("  Total time per triangle: %.4f ms\n",
                    totalSeconds * 1000.0 / n);
        std::printf("  Rank 0 device memory (distributed matrices + radiosity): %.2f MB\n",
                    (localMatrixBytes + replicatedBytes) / (1024.0 * 1024.0));
        std::printf("  Result hash: %016" PRIX64 "\n\n", computeHash(state));

        if (printResultsFlag) {
            std::vector<double> output(state.distances.begin(), state.distances.end());
            print_results(output, "Distances");
        }
    }

    if (validate && !validateResults(state, data, worldRank)) exitCode = 1;
    data.release();
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}

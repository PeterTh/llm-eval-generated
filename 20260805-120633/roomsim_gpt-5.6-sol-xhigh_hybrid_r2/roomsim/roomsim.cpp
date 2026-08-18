/**
 * Room Response Simulation Benchmark
 *
 * Hybrid implementation:
 *   - MPI partitions receiver triangles between accelerator processes.
 *   - CUDA computes form factors, delays, radiosity, and correlations.
 *   - OpenMP prepares the device-friendly geometry on each process.
 *
 * Every MPI process retains the (small) radiosity history, while the dense
 * Kij/Tau rows are distributed.  A timestep all-gather supplies the emitter
 * values required by the next delayed propagation step.
 */

#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
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
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr int PAIR_THREADS = 256;
constexpr int REDUCE_THREADS = 256;

// ============================================================================
// Host geometry
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t xValue, val_t yValue, val_t zValue)
        : x(xValue), y(yValue), z(zValue) {}

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }
    Vec3 operator*(val_t scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Vec3 operator/(val_t scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    val_t dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
    Vec3 cross(const Vec3& other) const {
        return {y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x};
    }
    val_t squaredNorm() const { return dot(*this); }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        const val_t length = norm();
        return length > EPSILON ? *this / length : Vec3{};
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 normal;

    Triangle() = default;
    Triangle(const Vec3& av, const Vec3& bv, const Vec3& cv)
        : a(av), b(bv), c(cv), normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    val_t area() const { return 0.5f * (b - a).cross(c - a).norm(); }
};

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t golden = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<Vec3> vertices = {
            Vec3(-1,  golden,  0).normalized() * radius,
            Vec3( 1,  golden,  0).normalized() * radius,
            Vec3(-1, -golden,  0).normalized() * radius,
            Vec3( 1, -golden,  0).normalized() * radius,
            Vec3( 0, -1,  golden).normalized() * radius,
            Vec3( 0,  1,  golden).normalized() * radius,
            Vec3( 0, -1, -golden).normalized() * radius,
            Vec3( 0,  1, -golden).normalized() * radius,
            Vec3( golden,  0, -1).normalized() * radius,
            Vec3( golden,  0,  1).normalized() * radius,
            Vec3(-golden,  0, -1).normalized() * radius,
            Vec3(-golden,  0,  1).normalized() * radius
        };

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        for (int level = 0; level < subdivisions; ++level) {
            std::vector<std::array<idx_t, 3>> refined;
            refined.reserve(faces.size() * 4);
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto midpoint = [&](idx_t first, idx_t second) -> idx_t {
                const auto key = std::make_pair(std::min(first, second),
                                                std::max(first, second));
                const auto found = midpointCache.find(key);
                if (found != midpointCache.end()) return found->second;
                Vec3 point = ((vertices[first] + vertices[second]) / 2.0f).normalized() * radius;
                const idx_t index = static_cast<idx_t>(vertices.size());
                vertices.push_back(point);
                midpointCache.emplace(key, index);
                return index;
            };

            for (const auto& face : faces) {
                const idx_t ab = midpoint(face[0], face[1]);
                const idx_t bc = midpoint(face[1], face[2]);
                const idx_t ca = midpoint(face[2], face[0]);
                refined.push_back({face[0], ab, ca});
                refined.push_back({face[1], bc, ab});
                refined.push_back({face[2], ca, bc});
                refined.push_back({ab, bc, ca});
            }
            faces = std::move(refined);
        }

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reversed winding gives the room inward-facing normals.
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// A 64-byte, naturally coalesced record used by the CUDA kernels.
struct DeviceTriangle {
    float3 a;
    float3 b;
    float3 c;
    float3 normal;
    float3 center;
    val_t area;
};

// ============================================================================
// Hybrid runtime and error handling
// ============================================================================

struct HybridContext {
    int rank = 0;
    int ranks = 1;
    int localRank = 0;
    int device = 0;
    bool cudaAwareMpi = false;
    cudaStream_t stream = nullptr;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
};

[[noreturn]] void fatalError(const HybridContext& context, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", context.rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression,
               const char* file, int line, const HybridContext& context) {
    if (status == cudaSuccess) return;
    char message[1024];
    std::snprintf(message, sizeof(message), "CUDA failure at %s:%d for %s: %s",
                  file, line, expression, cudaGetErrorString(status));
    fatalError(context, message);
}

#define CUDA_CHECK(context, expression) \
    checkCuda((expression), #expression, __FILE__, __LINE__, (context))

void initializeHybridRuntime(HybridContext& context) {
    MPI_Comm_rank(MPI_COMM_WORLD, &context.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.ranks);

    MPI_Comm shared = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, context.rank,
                        MPI_INFO_NULL, &shared);
    MPI_Comm_rank(shared, &context.localRank);

    int deviceCount = 0;
    CUDA_CHECK(context, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) fatalError(context, "the hybrid benchmark requires a CUDA device");
    context.device = context.localRank % deviceCount;
    CUDA_CHECK(context, cudaSetDevice(context.device));
    CUDA_CHECK(context, cudaFree(nullptr));
    CUDA_CHECK(context, cudaStreamCreateWithFlags(&context.stream, cudaStreamNonBlocking));

#if defined(MPIX_CUDA_AWARE_SUPPORT)
    context.cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif

    MPI_Comm_free(&shared);
}

// ============================================================================
// Simulation storage
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    size_t localBegin = 0;
    size_t localRows = 0;
    val_t reflectivity = 0.8f;

    std::vector<Triangle> triangles;
    std::vector<val_t> distances;  // Complete only on rank zero.

    DeviceTriangle* dTriangles = nullptr;
    val_t* dKij = nullptr;
    uint16_t* dTau = nullptr;
    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;

    val_t* hLocalRad = nullptr;   // Pinned MPI staging when MPI is not CUDA-aware.
    val_t* hGlobalRad = nullptr;
};

template <typename T>
void allocateDevice(T*& pointer, size_t elements, const HybridContext& context) {
    const size_t count = std::max<size_t>(elements, 1);
    CUDA_CHECK(context, cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
}

void releaseSimulation(SimulationState& state, HybridContext& context) {
    if (state.hLocalRad) CUDA_CHECK(context, cudaFreeHost(state.hLocalRad));
    if (state.hGlobalRad) CUDA_CHECK(context, cudaFreeHost(state.hGlobalRad));
    if (state.dDistances) CUDA_CHECK(context, cudaFree(state.dDistances));
    if (state.dRadB) CUDA_CHECK(context, cudaFree(state.dRadB));
    if (state.dTau) CUDA_CHECK(context, cudaFree(state.dTau));
    if (state.dKij) CUDA_CHECK(context, cudaFree(state.dKij));
    if (state.dTriangles) CUDA_CHECK(context, cudaFree(state.dTriangles));
    if (context.stream) CUDA_CHECK(context, cudaStreamDestroy(context.stream));
}

void initializeSimulation(SimulationState& state, HybridContext& context,
                          int subdivisions, size_t timesteps,
                          size_t sourceIndex, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIndex % state.numTriangles;
    state.reflectivity = reflectivity;

    const size_t base = state.numTriangles / static_cast<size_t>(context.ranks);
    const size_t remainder = state.numTriangles % static_cast<size_t>(context.ranks);
    state.localRows = base + (static_cast<size_t>(context.rank) < remainder ? 1 : 0);
    state.localBegin = static_cast<size_t>(context.rank) * base +
                       std::min(static_cast<size_t>(context.rank), remainder);

    context.rowCounts.resize(context.ranks);
    context.rowDisplacements.resize(context.ranks);
    for (int rank = 0; rank < context.ranks; ++rank) {
        const size_t rankRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        const size_t rankBegin = static_cast<size_t>(rank) * base +
                                 std::min(static_cast<size_t>(rank), remainder);
        if (rankRows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            rankBegin > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fatalError(context, "MPI row counts exceed the MPI_Allgatherv integer limit");
        }
        context.rowCounts[rank] = static_cast<int>(rankRows);
        context.rowDisplacements[rank] = static_cast<int>(rankBegin);
    }

    std::vector<DeviceTriangle> packed(state.numTriangles);
    const long long triangleCount = static_cast<long long>(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long index = 0; index < triangleCount; ++index) {
        const Triangle& triangle = state.triangles[static_cast<size_t>(index)];
        const Vec3 center = triangle.center();
        packed[static_cast<size_t>(index)] = {
            make_float3(triangle.a.x, triangle.a.y, triangle.a.z),
            make_float3(triangle.b.x, triangle.b.y, triangle.b.z),
            make_float3(triangle.c.x, triangle.c.y, triangle.c.z),
            make_float3(triangle.normal.x, triangle.normal.y, triangle.normal.z),
            make_float3(center.x, center.y, center.z),
            triangle.area()
        };
    }

    const size_t pairElements = state.localRows * state.numTriangles;
    const size_t historyElements = state.numTimesteps * state.numTriangles;
    const size_t requiredBytes = state.numTriangles * sizeof(DeviceTriangle) +
                                 pairElements * (sizeof(val_t) + sizeof(uint16_t)) +
                                 historyElements * sizeof(val_t) +
                                 state.localRows * sizeof(val_t);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CUDA_CHECK(context, cudaMemGetInfo(&freeBytes, &totalBytes));
    if (requiredBytes + 64ULL * 1024ULL * 1024ULL > freeBytes) {
        char message[512];
        std::snprintf(message, sizeof(message),
                      "device %d needs %.2f GiB but only %.2f GiB is free; use more MPI ranks/GPUs",
                      context.device, requiredBytes / 1073741824.0, freeBytes / 1073741824.0);
        fatalError(context, message);
    }

    allocateDevice(state.dTriangles, state.numTriangles, context);
    allocateDevice(state.dKij, pairElements, context);
    allocateDevice(state.dTau, pairElements, context);
    allocateDevice(state.dRadB, historyElements, context);
    allocateDevice(state.dDistances, state.localRows, context);

    CUDA_CHECK(context, cudaMemcpyAsync(state.dTriangles, packed.data(),
                                        packed.size() * sizeof(DeviceTriangle),
                                        cudaMemcpyHostToDevice, context.stream));
    CUDA_CHECK(context, cudaMemsetAsync(state.dRadB, 0,
                                        historyElements * sizeof(val_t), context.stream));
    CUDA_CHECK(context, cudaStreamSynchronize(context.stream));

    if (!context.cudaAwareMpi) {
        CUDA_CHECK(context, cudaMallocHost(reinterpret_cast<void**>(&state.hLocalRad),
                                           std::max<size_t>(state.localRows, 1) * sizeof(val_t)));
        CUDA_CHECK(context, cudaMallocHost(reinterpret_cast<void**>(&state.hGlobalRad),
                                           state.numTriangles * sizeof(val_t)));
    }

    if (context.rank == 0) {
        state.distances.resize(state.numTriangles);
        std::printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        std::printf("Hybrid layout: %d MPI rank%s, up to %d OpenMP threads/rank, CUDA device %d on rank 0\n",
                    context.ranks, context.ranks == 1 ? "" : "s",
                    omp_get_max_threads(), context.device);
        std::printf("MPI GPU buffers: %s\n",
                    context.cudaAwareMpi ? "CUDA-aware direct" : "pinned-host staged");
        std::printf("Dense pair storage per rank (maximum): %.2f MiB\n",
                    pairElements * (sizeof(val_t) + sizeof(uint16_t)) / 1048576.0);
    }
}

// ============================================================================
// CUDA geometry and random sampling helpers
// ============================================================================

__device__ __forceinline__ float3 add3(float3 first, float3 second) {
    return make_float3(first.x + second.x, first.y + second.y, first.z + second.z);
}

__device__ __forceinline__ float3 sub3(float3 first, float3 second) {
    return make_float3(first.x - second.x, first.y - second.y, first.z - second.z);
}

__device__ __forceinline__ float3 scale3(float3 value, val_t scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ __forceinline__ val_t dot3(float3 first, float3 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

__device__ __forceinline__ uint32_t randomBits(uint32_t& state) {
    // PCG RXS-M-XS: independent counter-seeded streams make results invariant
    // to MPI rank count, OpenMP scheduling, and CUDA launch scheduling.
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ val_t randomUnit(uint32_t& state) {
    return static_cast<val_t>(randomBits(state) >> 8u) * 0x1.0p-24f;
}

__device__ __forceinline__ float3 randomPoint(const DeviceTriangle& triangle,
                                               uint32_t& randomState) {
    val_t u = randomUnit(randomState);
    val_t v = randomUnit(randomState);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return add3(triangle.a,
                add3(scale3(sub3(triangle.b, triangle.a), u),
                     scale3(sub3(triangle.c, triangle.a), v)));
}

__device__ __forceinline__ uint32_t pairSeed(size_t receiver, size_t emitter,
                                             size_t triangleCount) {
    const unsigned long long pair = static_cast<unsigned long long>(receiver) * triangleCount + emitter;
    uint32_t value = static_cast<uint32_t>(pair) ^ static_cast<uint32_t>(pair >> 32u) ^ 0x9e3779b9u;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

// The procedural icosphere is a convex closed polyhedron.  A segment joining
// points on any two facets lies within that polyhedron and therefore cannot be
// intercepted by a third facet.  This is exactly the visibility result of the
// original octree/ray test, without its divergent pointer traversal.
__global__ void computePairsKernel(const DeviceTriangle* __restrict__ triangles,
                                   size_t triangleCount, size_t localBegin,
                                   size_t pairCount, val_t* __restrict__ kij,
                                   uint16_t* __restrict__ tau) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= pairCount) return;

    const size_t localReceiver = linear / triangleCount;
    const size_t emitter = linear - localReceiver * triangleCount;
    const size_t receiver = localBegin + localReceiver;
    const DeviceTriangle triI = triangles[receiver];
    const DeviceTriangle triJ = triangles[emitter];

    if (receiver == emitter) {
        kij[linear] = ZERO;
        tau[linear] = 0;
        return;
    }

    const float3 centerDelta = sub3(triI.center, triJ.center);
    const val_t distance = sqrtf(dot3(centerDelta, centerDelta));
    tau[linear] = static_cast<uint16_t>(ceilf(distance * INV_WAVE_SPEED));

    if (dot3(triI.normal, triJ.normal) > 0.99f) {
        kij[linear] = ZERO;
        return;
    }

    uint32_t randomState = pairSeed(receiver, emitter, triangleCount);
    val_t factor = ZERO;
#pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const float3 pointI = randomPoint(triI, randomState);
        const float3 pointJ = randomPoint(triJ, randomState);
        const float3 direction = sub3(pointJ, pointI);
        const val_t distanceSquared = dot3(direction, direction);
        if (distanceSquared < EPSILON) continue;
        const val_t inverseDistance = ONE / sqrtf(distanceSquared);
        const val_t cosI = fmaxf(ZERO, dot3(direction, triI.normal) * inverseDistance);
        const val_t cosJ = fmaxf(ZERO, -dot3(direction, triJ.normal) * inverseDistance);
        if (cosI > ZERO && cosJ > ZERO) {
            factor += cosI * cosJ / (PI * distanceSquared);
        }
    }
    kij[linear] = factor * INV_NUM_RAYS;
}

void computeFormFactorsAndDelays(SimulationState& state, HybridContext& context) {
    if (context.rank == 0) std::printf("Computing distributed form factors and time delays on CUDA...\n");
    const size_t pairCount = state.localRows * state.numTriangles;
    if (pairCount != 0) {
        const size_t blocks = (pairCount + PAIR_THREADS - 1) / PAIR_THREADS;
        computePairsKernel<<<static_cast<unsigned int>(blocks), PAIR_THREADS, 0, context.stream>>>(
            state.dTriangles, state.numTriangles, state.localBegin, pairCount,
            state.dKij, state.dTau);
        CUDA_CHECK(context, cudaGetLastError());
        CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
    }
}

// ============================================================================
// Delayed wave propagation
// ============================================================================

__global__ void propagateKernel(size_t triangleCount, size_t localBegin,
                                size_t localRows, size_t timestep,
                                size_t sourceIndex, size_t timeOn,
                                val_t reflectivity,
                                const DeviceTriangle* __restrict__ triangles,
                                const val_t* __restrict__ kij,
                                const uint16_t* __restrict__ tau,
                                val_t* __restrict__ radB) {
    const size_t localReceiver = blockIdx.x;
    if (localReceiver >= localRows) return;
    const size_t rowOffset = localReceiver * triangleCount;
    val_t partial = ZERO;

    for (size_t emitter = threadIdx.x; emitter < triangleCount; emitter += blockDim.x) {
        const uint16_t delay = tau[rowOffset + emitter];
        if (delay == 0 || timestep < delay) continue;
        const val_t factor = kij[rowOffset + emitter];
        if (factor <= ZERO) continue;
        const size_t sourceTime = timestep - delay;
        const val_t emitterRadiosity = radB[sourceTime * triangleCount + emitter];
        if (emitterRadiosity > ZERO) {
            partial += fminf(factor * triangles[emitter].area, ONE) * emitterRadiosity;
        }
    }

    __shared__ val_t sums[REDUCE_THREADS];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const size_t receiver = localBegin + localReceiver;
        const val_t emission = (receiver == sourceIndex && timestep < timeOn) ? ONE : ZERO;
        radB[timestep * triangleCount + receiver] = reflectivity * sums[0] + emission;
    }
}

void exchangeRadiositySlice(SimulationState& state, HybridContext& context, size_t timestep) {
    val_t* const deviceSlice = state.dRadB + timestep * state.numTriangles;
    if (context.cudaAwareMpi) {
        CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
        const int status = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                                          deviceSlice, context.rowCounts.data(),
                                          context.rowDisplacements.data(), MPI_FLOAT,
                                          MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) fatalError(context, "CUDA-aware MPI_Allgatherv failed");
    } else {
        CUDA_CHECK(context, cudaMemcpyAsync(state.hLocalRad,
                                            deviceSlice + state.localBegin,
                                            state.localRows * sizeof(val_t),
                                            cudaMemcpyDeviceToHost, context.stream));
        CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
        const int status = MPI_Allgatherv(state.hLocalRad,
                                          static_cast<int>(state.localRows), MPI_FLOAT,
                                          state.hGlobalRad, context.rowCounts.data(),
                                          context.rowDisplacements.data(), MPI_FLOAT,
                                          MPI_COMM_WORLD);
        if (status != MPI_SUCCESS) fatalError(context, "staged MPI_Allgatherv failed");
        CUDA_CHECK(context, cudaMemcpyAsync(deviceSlice, state.hGlobalRad,
                                            state.numTriangles * sizeof(val_t),
                                            cudaMemcpyHostToDevice, context.stream));
        CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
    }
}

void runSimulation(SimulationState& state, HybridContext& context) {
    if (context.rank == 0) std::printf("Running CUDA wave propagation with MPI timestep exchange...\n");
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        if (state.localRows != 0) {
            propagateKernel<<<static_cast<unsigned int>(state.localRows), REDUCE_THREADS,
                              0, context.stream>>>(
                state.numTriangles, state.localBegin, state.localRows, timestep,
                state.sourceIndex, state.numTimesteps / 2, state.reflectivity,
                state.dTriangles, state.dKij, state.dTau, state.dRadB);
            CUDA_CHECK(context, cudaGetLastError());
        }
        exchangeRadiositySlice(state, context, timestep);
        if (context.rank == 0 &&
            ((timestep + 1) % 10 == 0 || timestep + 1 == state.numTimesteps)) {
            std::printf("  Timestep %zu/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Cross-correlation and validation reductions
// ============================================================================

__device__ __forceinline__ bool betterCorrelation(val_t candidateValue, int candidateLag,
                                                   val_t currentValue, int currentLag) {
    return candidateValue > currentValue ||
           (candidateValue == currentValue && candidateLag < currentLag);
}

__global__ void distanceKernel(size_t triangleCount, size_t timesteps,
                               size_t localBegin, size_t localRows,
                               size_t sourceIndex, const val_t* __restrict__ radB,
                               val_t* __restrict__ distances) {
    const size_t localReceiver = blockIdx.x;
    if (localReceiver >= localRows) return;
    const size_t receiver = localBegin + localReceiver;
    val_t threadBest = ZERO;
    int threadLag = 0;

    for (size_t lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (size_t time = lag; time < timesteps; ++time) {
            correlation += radB[(time - lag) * triangleCount + sourceIndex] *
                           radB[time * triangleCount + receiver];
        }
        if (betterCorrelation(correlation, static_cast<int>(lag), threadBest, threadLag)) {
            threadBest = correlation;
            threadLag = static_cast<int>(lag);
        }
    }

    __shared__ val_t bestValues[REDUCE_THREADS];
    __shared__ int bestLags[REDUCE_THREADS];
    bestValues[threadIdx.x] = threadBest;
    bestLags[threadIdx.x] = threadLag;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            betterCorrelation(bestValues[threadIdx.x + offset],
                              bestLags[threadIdx.x + offset],
                              bestValues[threadIdx.x], bestLags[threadIdx.x])) {
            bestValues[threadIdx.x] = bestValues[threadIdx.x + offset];
            bestLags[threadIdx.x] = bestLags[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[localReceiver] = WAVE_SPEED * static_cast<val_t>(bestLags[0]);
    }
}

void computeDistances(SimulationState& state, HybridContext& context) {
    if (context.rank == 0) std::printf("Computing distributed cross-correlations on CUDA...\n");
    if (state.localRows != 0) {
        distanceKernel<<<static_cast<unsigned int>(state.localRows), REDUCE_THREADS,
                         0, context.stream>>>(
            state.numTriangles, state.numTimesteps, state.localBegin, state.localRows,
            state.sourceIndex, state.dRadB, state.dDistances);
        CUDA_CHECK(context, cudaGetLastError());
    }

    std::vector<val_t> localDistances(state.localRows);
    CUDA_CHECK(context, cudaMemcpyAsync(localDistances.data(), state.dDistances,
                                        state.localRows * sizeof(val_t),
                                        cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
    const int status = MPI_Gatherv(localDistances.data(), static_cast<int>(state.localRows),
                                   MPI_FLOAT,
                                   context.rank == 0 ? state.distances.data() : nullptr,
                                   context.rowCounts.data(), context.rowDisplacements.data(),
                                   MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (status != MPI_SUCCESS) fatalError(context, "MPI_Gatherv of distances failed");
}

__global__ void countKijKernel(const val_t* __restrict__ kij, size_t elements,
                               unsigned long long* count) {
    unsigned long long local = 0;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        local += kij[index] > EPSILON;
    }
    __shared__ unsigned long long sums[REDUCE_THREADS];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(count, sums[0]);
}

__global__ void countEnergyKernel(const val_t* __restrict__ radB,
                                  size_t triangleCount, size_t timesteps,
                                  size_t localBegin, size_t localRows,
                                  unsigned int* count) {
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= localRows) return;
    const size_t receiver = localBegin + local;
    for (size_t time = 0; time < timesteps; ++time) {
        if (radB[time * triangleCount + receiver] > EPSILON) {
            atomicAdd(count, 1u);
            return;
        }
    }
}

std::pair<uint64_t, uint64_t> collectValidationCounts(const SimulationState& state,
                                                       HybridContext& context) {
    unsigned long long* dKijCount = nullptr;
    unsigned int* dEnergyCount = nullptr;
    allocateDevice(dKijCount, 1, context);
    allocateDevice(dEnergyCount, 1, context);
    CUDA_CHECK(context, cudaMemsetAsync(dKijCount, 0, sizeof(*dKijCount), context.stream));
    CUDA_CHECK(context, cudaMemsetAsync(dEnergyCount, 0, sizeof(*dEnergyCount), context.stream));

    const size_t pairElements = state.localRows * state.numTriangles;
    if (pairElements != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>(4096, (pairElements + REDUCE_THREADS - 1) / REDUCE_THREADS));
        countKijKernel<<<blocks, REDUCE_THREADS, 0, context.stream>>>(
            state.dKij, pairElements, dKijCount);
        CUDA_CHECK(context, cudaGetLastError());
    }
    if (state.localRows != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            (state.localRows + REDUCE_THREADS - 1) / REDUCE_THREADS);
        countEnergyKernel<<<blocks, REDUCE_THREADS, 0, context.stream>>>(
            state.dRadB, state.numTriangles, state.numTimesteps,
            state.localBegin, state.localRows, dEnergyCount);
        CUDA_CHECK(context, cudaGetLastError());
    }

    unsigned long long localKij = 0;
    unsigned int localEnergy = 0;
    CUDA_CHECK(context, cudaMemcpyAsync(&localKij, dKijCount, sizeof(localKij),
                                        cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(context, cudaMemcpyAsync(&localEnergy, dEnergyCount, sizeof(localEnergy),
                                        cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(context, cudaStreamSynchronize(context.stream));
    CUDA_CHECK(context, cudaFree(dKijCount));
    CUDA_CHECK(context, cudaFree(dEnergyCount));

    uint64_t globalKij = 0;
    uint64_t globalEnergy = 0;
    const uint64_t localKij64 = static_cast<uint64_t>(localKij);
    const uint64_t localEnergy64 = static_cast<uint64_t>(localEnergy);
    MPI_Reduce(&localKij64, &globalKij, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localEnergy64, &globalEnergy, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    return {globalKij, globalEnergy};
}

bool validateResults(const SimulationState& state, uint64_t nonZeroKij,
                     uint64_t receivedEnergy) {
    std::printf("\nValidation:\n");
    bool valid = true;
    val_t minDistance = std::numeric_limits<val_t>::max();
    val_t maxDistance = std::numeric_limits<val_t>::lowest();
    val_t distanceSum = ZERO;
    size_t nonZeroDistances = 0;

    for (size_t index = 0; index < state.numTriangles; ++index) {
        const val_t distance = state.distances[index];
        if (distance < ZERO || !std::isfinite(distance)) {
            std::printf("  ERROR: Invalid distance at triangle %zu: %f\n", index, distance);
            valid = false;
        }
        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
        distanceSum += distance;
        nonZeroDistances += distance > EPSILON;
    }

    std::printf("  Distance range: [%.4f, %.4f]\n", minDistance, maxDistance);
    std::printf("  Average distance: %.4f\n",
                distanceSum / static_cast<val_t>(state.numTriangles));
    std::printf("  Non-zero distances: %zu/%zu\n", nonZeroDistances, state.numTriangles);
    std::printf("  Triangles receiving energy: %llu/%zu\n",
                static_cast<unsigned long long>(receivedEnergy), state.numTriangles);

    const uint64_t totalPairs = static_cast<uint64_t>(state.numTriangles) * state.numTriangles;
    std::printf("  Non-zero form factors: %llu/%llu (%.2f%%)\n",
                static_cast<unsigned long long>(nonZeroKij),
                static_cast<unsigned long long>(totalPairs),
                100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalPairs));
    if (receivedEnergy == 0 || nonZeroKij == 0) {
        std::printf("  ERROR: wave propagation or form-factor computation produced no energy\n");
        valid = false;
    }
    std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

uint64_t computeHash(const SimulationState& state) {
    uint64_t value = 0;
    for (size_t index = 0; index < state.numTriangles; ++index) {
        uint32_t bits = 0;
        std::memcpy(&bits, &state.distances[index], sizeof(bits));
        value ^= (static_cast<uint64_t>(bits) + index) * 0x9e3779b97f4a7c15ULL;
    }
    return value;
}

int getSubdivisionsForTriangleCount(int targetTriangles) {
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        ++subdivisions;
        triangles *= 4;
    }
    return subdivisions;
}

void printUsage(const char* program) {
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

double reduceMaximumTime(double localTime, const HybridContext& context) {
    double maximum = 0.0;
    MPI_Reduce(&localTime, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return context.rank == 0 ? maximum : localTime;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    HybridContext context;
    MPI_Comm_rank(MPI_COMM_WORLD, &context.rank);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        fatalError(context, "MPI does not provide the required MPI_THREAD_FUNNELED support");
    }
    initializeHybridRuntime(context);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIndex = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
            targetTriangles = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-t") == 0 && argument + 1 < argc) {
            timesteps = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-s") == 0 && argument + 1 < argc) {
            sourceIndex = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-r") == 0 && argument + 1 < argc) {
            reflectivity = static_cast<val_t>(std::atof(argv[++argument]));
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-o") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (context.rank == 0) std::printf("Unknown or incomplete option: %s\n", argv[argument]);
        }
    }

    if (showHelp || !argumentsValid || targetTriangles <= 0 || timesteps <= 0 ||
        reflectivity < ZERO || reflectivity > ONE) {
        if (context.rank == 0) printUsage(argv[0]);
        CUDA_CHECK(context, cudaStreamDestroy(context.stream));
        MPI_Finalize();
        return showHelp && argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    if (context.rank == 0) {
        std::printf("Room Response Simulation Benchmark\n");
        std::printf("===================================\n");
        std::printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        std::printf("Timesteps: %d\n", timesteps);
        std::printf("Source triangle: %d\n", sourceIndex);
        std::printf("Reflectivity: %.2f\n", reflectivity);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    SimulationState state;
    initializeSimulation(state, context, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIndex), reflectivity);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    computeFormFactorsAndDelays(state, context);
    MPI_Barrier(MPI_COMM_WORLD);
    const double preSeconds = reduceMaximumTime(MPI_Wtime() - start, context);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    runSimulation(state, context);
    MPI_Barrier(MPI_COMM_WORLD);
    const double simulationSeconds = reduceMaximumTime(MPI_Wtime() - start, context);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    computeDistances(state, context);
    MPI_Barrier(MPI_COMM_WORLD);
    const double distanceSeconds = reduceMaximumTime(MPI_Wtime() - start, context);

    std::pair<uint64_t, uint64_t> validationCounts{0, 0};
    if (validate) validationCounts = collectValidationCounts(state, context);

    int exitCode = EXIT_SUCCESS;
    if (context.rank == 0) {
        const long preMilliseconds = static_cast<long>(preSeconds * 1000.0);
        const long simulationMilliseconds = static_cast<long>(simulationSeconds * 1000.0);
        const long distanceMilliseconds = static_cast<long>(distanceSeconds * 1000.0);
        const long totalMilliseconds = preMilliseconds + simulationMilliseconds + distanceMilliseconds;
        const size_t n = state.numTriangles;
        const size_t t = state.numTimesteps;

        std::printf("Precomputation time: %ld ms\n\n", preMilliseconds);
        std::printf("Simulation time: %ld ms\n\n", simulationMilliseconds);
        std::printf("Distance computation time: %ld ms\n\n", distanceMilliseconds);
        std::printf("Total computation time: %ld ms\n", totalMilliseconds);
        std::printf("\nPerformance:\n");
        std::printf("  Triangles: %zu\n", n);
        std::printf("  Timesteps: %zu\n", t);
        std::printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        std::printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        std::printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        std::printf("  Total time per triangle: %.4f ms\n",
                    static_cast<double>(totalMilliseconds) / n);
        const size_t distributedMemory = n * n * (sizeof(val_t) + sizeof(uint16_t));
        const size_t replicatedMemory = t * n * sizeof(val_t) + n * sizeof(DeviceTriangle);
        std::printf("  Distributed matrix memory: %.2f MiB total\n", distributedMemory / 1048576.0);
        std::printf("  Replicated memory: %.2f MiB per rank\n", replicatedMemory / 1048576.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(computeHash(state)));

        if (printResults) {
            const std::vector<double> distanceData(state.distances.begin(), state.distances.end());
            print_results(distanceData, "Distances");
        }
        if (validate && !validateResults(state, validationCounts.first, validationCounts.second)) {
            exitCode = EXIT_FAILURE;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    releaseSimulation(state, context);
    MPI_Finalize();
    return exitCode;
}

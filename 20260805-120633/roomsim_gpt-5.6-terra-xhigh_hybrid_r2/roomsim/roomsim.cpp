/**
 * Room Response Simulation Benchmark
 *
 * Hybrid implementation:
 *   - MPI partitions receiver-triangle rows across accelerator ranks.
 *   - CUDA computes form factors, propagation, and correlation on each GPU.
 *   - OpenMP prepares rank-local host data and validates the final result.
 *
 * The generated room is a convex icosphere.  Consequently, a segment joining
 * two points on its surface is inside the room (apart from its endpoints), so
 * no third triangle can occlude it.  This is the same visibility result as the
 * original octree traversal, while allowing the O(N^2) form-factor phase to
 * execute efficiently on the GPU.
 */

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
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
constexpr int CUDA_THREADS = 256;

static void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure at %s:%d while evaluating %s: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

static void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(status, error, &length);
        std::fprintf(stderr, "MPI failure at %s:%d while evaluating %s: %.*s\n",
                     file, line, expression, length, error);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

// ============================================================================
// Host mesh types and procedural icosphere construction
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t xIn, val_t yIn, val_t zIn) : x(xIn), y(yIn), z(zIn) {}

    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(val_t scale) const { return {x * scale, y * scale, z * scale}; }
    Vec3 operator/(val_t scale) const { return {x / scale, y / scale, z / scale}; }
    Vec3 cross(const Vec3& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z,
                x * other.y - y * other.x};
    }
    val_t dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
    val_t squaredNorm() const { return dot(*this); }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        const val_t length = norm();
        return length > EPSILON ? *this / length : Vec3();
    }
};

struct Triangle {
    Vec3 a, b, c, normal;

    Triangle() = default;
    Triangle(const Vec3& aIn, const Vec3& bIn, const Vec3& cIn)
        : a(aIn), b(bIn), c(cIn), normal((b - a).cross(c - a).normalized()) {}
};

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<Vec3> vertices = {
            Vec3(-1,  goldenRatio,  0).normalized() * radius,
            Vec3( 1,  goldenRatio,  0).normalized() * radius,
            Vec3(-1, -goldenRatio,  0).normalized() * radius,
            Vec3( 1, -goldenRatio,  0).normalized() * radius,
            Vec3( 0, -1,  goldenRatio).normalized() * radius,
            Vec3( 0,  1,  goldenRatio).normalized() * radius,
            Vec3( 0, -1, -goldenRatio).normalized() * radius,
            Vec3( 0,  1, -goldenRatio).normalized() * radius,
            Vec3( goldenRatio,  0, -1).normalized() * radius,
            Vec3( goldenRatio,  0,  1).normalized() * radius,
            Vec3(-goldenRatio,  0, -1).normalized() * radius,
            Vec3(-goldenRatio,  0,  1).normalized() * radius
        };
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        for (int level = 0; level < subdivisions; ++level) {
            std::vector<std::array<idx_t, 3>> nextFaces;
            nextFaces.reserve(faces.size() * 4);
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;
            auto midpoint = [&](idx_t first, idx_t second) {
                const auto key = std::make_pair(std::min(first, second), std::max(first, second));
                const auto found = midpointCache.find(key);
                if (found != midpointCache.end()) return found->second;
                const idx_t index = static_cast<idx_t>(vertices.size());
                vertices.push_back(((vertices[first] + vertices[second]) / 2.0f).normalized() * radius);
                midpointCache.emplace(key, index);
                return index;
            };
            for (const auto& face : faces) {
                const idx_t a = midpoint(face[0], face[1]);
                const idx_t b = midpoint(face[1], face[2]);
                const idx_t c = midpoint(face[2], face[0]);
                nextFaces.push_back({face[0], a, c});
                nextFaces.push_back({face[1], b, a});
                nextFaces.push_back({face[2], c, b});
                nextFaces.push_back({a, b, c});
            }
            faces = std::move(nextFaces);
        }

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reversed winding points the normals into the room, as in the original solver.
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// A tightly packed representation that can be broadcast with MPI_BYTE and read by CUDA.
struct DeviceTriangle {
    val_t ax, ay, az;
    val_t bx, by, bz;
    val_t cx, cy, cz;
    val_t nx, ny, nz;
};

struct DevicePoint {
    val_t x, y, z;
};

static DeviceTriangle packTriangle(const Triangle& triangle) {
    return {triangle.a.x, triangle.a.y, triangle.a.z,
            triangle.b.x, triangle.b.y, triangle.b.z,
            triangle.c.x, triangle.c.y, triangle.c.z,
            triangle.normal.x, triangle.normal.y, triangle.normal.z};
}

static val_t triangleArea(const DeviceTriangle& triangle) {
    const val_t abx = triangle.bx - triangle.ax;
    const val_t aby = triangle.by - triangle.ay;
    const val_t abz = triangle.bz - triangle.az;
    const val_t acx = triangle.cx - triangle.ax;
    const val_t acy = triangle.cy - triangle.ay;
    const val_t acz = triangle.cz - triangle.az;
    const val_t crossX = aby * acz - abz * acy;
    const val_t crossY = abz * acx - abx * acz;
    const val_t crossZ = abx * acy - aby * acx;
    return 0.5f * std::sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
}

// ============================================================================
// Distributed simulation state
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    size_t localBegin = 0;
    size_t localCount = 0;

    std::vector<DeviceTriangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    // Every rank keeps the time history because each local row depends on all
    // source triangles.  Kij and Tau remain distributed on their owning GPU.
    std::vector<val_t> radB;
    std::vector<val_t> distances;
    std::vector<int> mpiCounts;
    std::vector<int> mpiDisplacements;

    size_t idxTN(size_t timestep, size_t triangle) const {
        return timestep * numTriangles + triangle;
    }
};

struct DeviceBuffers {
    DeviceTriangle* triangles = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    DevicePoint* samples = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* localRadB = nullptr;
    val_t* localDistances = nullptr;
    unsigned long long* nonZeroCounter = nullptr;

    void release() {
        cudaFree(nonZeroCounter);
        cudaFree(localDistances);
        cudaFree(localRadB);
        cudaFree(radB);
        cudaFree(tau);
        cudaFree(kij);
        cudaFree(rho);
        cudaFree(areas);
        cudaFree(samples);
        cudaFree(triangles);
        nonZeroCounter = nullptr;
        localDistances = nullptr;
        localRadB = nullptr;
        radB = nullptr;
        tau = nullptr;
        kij = nullptr;
        rho = nullptr;
        areas = nullptr;
        samples = nullptr;
        triangles = nullptr;
    }
};

static void computePartition(SimulationState& state, int rank, int ranks) {
    const size_t base = state.numTriangles / static_cast<size_t>(ranks);
    const size_t remainder = state.numTriangles % static_cast<size_t>(ranks);
    state.localCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    state.localBegin = static_cast<size_t>(rank) * base +
                       std::min(static_cast<size_t>(rank), remainder);
    state.mpiCounts.resize(ranks);
    state.mpiDisplacements.resize(ranks);
    for (int process = 0; process < ranks; ++process) {
        const size_t count = base + (static_cast<size_t>(process) < remainder ? 1 : 0);
        const size_t displacement = static_cast<size_t>(process) * base +
                                    std::min(static_cast<size_t>(process), remainder);
        state.mpiCounts[process] = static_cast<int>(count);
        state.mpiDisplacements[process] = static_cast<int>(displacement);
    }
}

static void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                                 size_t requestedSource, val_t reflectivity, int rank,
                                 MPI_Comm communicator) {
    if (rank == 0) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        state.numTriangles = mesh.triangles.size();
        state.triangles.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
            state.triangles[static_cast<size_t>(i)] = packTriangle(mesh.triangles[static_cast<size_t>(i)]);
        }
    }

    unsigned long long triangleCount = static_cast<unsigned long long>(state.numTriangles);
    MPI_CHECK(MPI_Bcast(&triangleCount, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator));
    state.numTriangles = static_cast<size_t>(triangleCount);
    if (rank != 0) state.triangles.resize(state.numTriangles);
    MPI_CHECK(MPI_Bcast(state.triangles.data(),
                        static_cast<int>(state.numTriangles * sizeof(DeviceTriangle)), MPI_BYTE,
                        0, communicator));

    state.numTimesteps = timesteps;
    state.sourceIndex = requestedSource % state.numTriangles;
    state.areas.resize(state.numTriangles);
    state.rho.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const size_t index = static_cast<size_t>(i);
        state.areas[index] = triangleArea(state.triangles[index]);
        state.rho[index] = reflectivity;
    }
    state.radB.assign(state.numTimesteps * state.numTriangles, ZERO);
}

static void allocateDeviceBuffers(const SimulationState& state, DeviceBuffers& device) {
    const size_t localElements = std::max<size_t>(1, state.localCount * state.numTriangles);
    const size_t localCount = std::max<size_t>(1, state.localCount);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.triangles),
                          state.numTriangles * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.areas), state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.rho), state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.samples),
                          state.numTriangles * NUM_RAYS * sizeof(DevicePoint)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.kij), localElements * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.tau), localElements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.radB),
                          state.numTimesteps * state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.localRadB), localCount * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.localDistances), localCount * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.nonZeroCounter), sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(device.triangles, state.triangles.data(),
                          state.numTriangles * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.areas, state.areas.data(), state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.rho, state.rho.data(), state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device.radB, 0,
                          state.numTimesteps * state.numTriangles * sizeof(val_t)));
}

// ============================================================================
// CUDA kernels
// ============================================================================

__device__ __forceinline__ unsigned long long mixRandom(unsigned long long value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ val_t randomUnit(unsigned long long counter) {
    // Use the top 24 bits, exactly representable in a float, for reproducible
    // per-ray samples regardless of MPI process count or GPU scheduling.
    return static_cast<val_t>(mixRandom(counter) >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ void randomPoint(const DeviceTriangle& triangle, val_t u, val_t v,
                                             val_t& x, val_t& y, val_t& z) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    x = triangle.ax + (triangle.bx - triangle.ax) * u + (triangle.cx - triangle.ax) * v;
    y = triangle.ay + (triangle.by - triangle.ay) * u + (triangle.cy - triangle.ay) * v;
    z = triangle.az + (triangle.bz - triangle.az) * u + (triangle.cz - triangle.az) * v;
}

__global__ void generateSamplePointsKernel(const DeviceTriangle* triangles, size_t numTriangles,
                                           DevicePoint* samples) {
    const size_t sampleIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t totalSamples = numTriangles * NUM_RAYS;
    if (sampleIndex >= totalSamples) return;
    const size_t triangleIndex = sampleIndex / NUM_RAYS;
    const unsigned long long randomIndex = static_cast<unsigned long long>(sampleIndex) * 2ULL;
    DevicePoint point;
    randomPoint(triangles[triangleIndex], randomUnit(randomIndex), randomUnit(randomIndex + 1ULL),
                point.x, point.y, point.z);
    samples[sampleIndex] = point;
}

__global__ void precomputeKernel(const DeviceTriangle* triangles, size_t numTriangles,
                                 size_t localBegin, size_t localCount,
                                 const DevicePoint* __restrict__ samples, val_t* kij, int* tau) {
    const size_t localI = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localCount || j >= numTriangles) return;

    const size_t i = localBegin + localI;
    const size_t output = localI * numTriangles + j;
    if (i == j) {
        kij[output] = ZERO;
        tau[output] = 0;
        return;
    }

    const DeviceTriangle triI = triangles[i];
    const DeviceTriangle triJ = triangles[j];
    const val_t centerIx = (triI.ax + triI.bx + triI.cx) / 3.0f;
    const val_t centerIy = (triI.ay + triI.by + triI.cy) / 3.0f;
    const val_t centerIz = (triI.az + triI.bz + triI.cz) / 3.0f;
    const val_t centerJx = (triJ.ax + triJ.bx + triJ.cx) / 3.0f;
    const val_t centerJy = (triJ.ay + triJ.by + triJ.cy) / 3.0f;
    const val_t centerJz = (triJ.az + triJ.bz + triJ.cz) / 3.0f;
    const val_t centerDx = centerIx - centerJx;
    const val_t centerDy = centerIy - centerJy;
    const val_t centerDz = centerIz - centerJz;
    tau[output] = static_cast<int>(ceilf(sqrtf(centerDx * centerDx + centerDy * centerDy + centerDz * centerDz) *
                                         INV_WAVE_SPEED));

    const val_t normalDot = triI.nx * triJ.nx + triI.ny * triJ.ny + triI.nz * triJ.nz;
    if (normalDot > 0.99f) {
        kij[output] = ZERO;
        return;
    }

    val_t factor = ZERO;
#pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        // Points for i and j are independently sampled.  Reusing the samples
        // for other matrix entries preserves the estimator while removing RNG
        // work from the O(N^2) kernel.
        const DevicePoint pointI = samples[i * NUM_RAYS + ray];
        const DevicePoint pointJ = samples[j * NUM_RAYS + ray];
        const val_t vx = pointJ.x - pointI.x;
        const val_t vy = pointJ.y - pointI.y;
        const val_t vz = pointJ.z - pointI.z;
        const val_t distanceSquared = vx * vx + vy * vy + vz * vz;
        if (distanceSquared < EPSILON) continue;
        const val_t inverseDistance = rsqrtf(distanceSquared);
        const val_t cosineI = fmaxf(ZERO, (vx * triI.nx + vy * triI.ny + vz * triI.nz) * inverseDistance);
        const val_t cosineJ = fmaxf(ZERO, (-vx * triJ.nx - vy * triJ.ny - vz * triJ.nz) * inverseDistance);
        if (cosineI > ZERO && cosineJ > ZERO) {
            factor += (cosineI * cosineJ) / (PI * distanceSquared);
        }
    }
    kij[output] = factor * INV_NUM_RAYS;
}

__global__ void propagateKernel(size_t numTriangles, size_t localBegin, size_t localCount,
                                size_t timestep, size_t sourceIndex, size_t timeOff,
                                const val_t* __restrict__ areas, const val_t* __restrict__ rho,
                                const val_t* __restrict__ kij, const int* __restrict__ tau,
                                const val_t* __restrict__ history, val_t* __restrict__ output) {
    const size_t localI = blockIdx.x;
    if (localI >= localCount) return;
    const size_t i = localBegin + localI;
    const size_t row = localI * numTriangles;
    val_t partial = ZERO;
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        const int delay = tau[row + j];
        if (delay > 0 && timestep >= static_cast<size_t>(delay)) {
            const val_t formFactor = kij[row + j];
            if (formFactor > ZERO) {
                const val_t sourceRadiosity = history[(timestep - static_cast<size_t>(delay)) * numTriangles + j];
                if (sourceRadiosity > ZERO) {
                    partial += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
                }
            }
        }
    }

    extern __shared__ val_t partialSums[];
    partialSums[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) partialSums[threadIdx.x] += partialSums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < timeOff) ? ONE : ZERO;
        output[localI] = rho[i] * partialSums[0] + emission;
    }
}

__global__ void distanceKernel(size_t numTriangles, size_t numTimesteps, size_t sourceIndex,
                               size_t localBegin, size_t localCount,
                               const val_t* __restrict__ radiosity, val_t* __restrict__ distances) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localCount) return;
    const size_t i = localBegin + localI;
    val_t maxCorrelation = ZERO;
    int bestTime = 0;
    for (size_t delay = 0; delay < numTimesteps; ++delay) {
        val_t sum = ZERO;
        for (size_t time = delay; time < numTimesteps; ++time) {
            sum += radiosity[time * numTriangles + sourceIndex] *
                   radiosity[(time - delay) * numTriangles + i];
        }
        if (sum > maxCorrelation) {
            maxCorrelation = sum;
            bestTime = static_cast<int>(delay);
        }
    }
    distances[localI] = WAVE_SPEED * static_cast<val_t>(bestTime);
}

__global__ void countPositiveKernel(const val_t* values, size_t count, unsigned long long* total) {
    unsigned long long local = 0;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        local += values[index] > EPSILON;
    }
    extern __shared__ unsigned long long counts[];
    counts[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) counts[threadIdx.x] += counts[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, counts[0]);
}

// ============================================================================
// GPU/MPI execution phases
// ============================================================================

static void computePrecomputation(const SimulationState& state, const DeviceBuffers& device) {
    const size_t sampleCount = state.numTriangles * NUM_RAYS;
    const unsigned int sampleGrid = static_cast<unsigned int>((sampleCount + CUDA_THREADS - 1) / CUDA_THREADS);
    generateSamplePointsKernel<<<sampleGrid, CUDA_THREADS>>>(device.triangles, state.numTriangles,
                                                             device.samples);
    CUDA_CHECK(cudaGetLastError());
    if (state.localCount == 0) return;
    constexpr int blockWidth = 16;
    constexpr int blockHeight = 16;
    const dim3 block(blockWidth, blockHeight);
    const dim3 grid(static_cast<unsigned int>((state.numTriangles + blockWidth - 1) / blockWidth),
                    static_cast<unsigned int>((state.localCount + blockHeight - 1) / blockHeight));
    precomputeKernel<<<grid, block>>>(device.triangles, state.numTriangles, state.localBegin,
                                      state.localCount, device.samples, device.kij, device.tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void runSimulation(SimulationState& state, const DeviceBuffers& device, MPI_Comm communicator) {
    std::vector<val_t> localValues(state.localCount);
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        if (state.localCount != 0) {
            propagateKernel<<<static_cast<unsigned int>(state.localCount), CUDA_THREADS,
                              CUDA_THREADS * sizeof(val_t)>>>(
                state.numTriangles, state.localBegin, state.localCount, timestep, state.sourceIndex,
                state.numTimesteps / 2, device.areas, device.rho, device.kij, device.tau,
                device.radB, device.localRadB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localValues.data(), device.localRadB,
                                  state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        val_t* row = state.radB.data() + state.idxTN(timestep, 0);
        MPI_CHECK(MPI_Allgatherv(localValues.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                                 row, state.mpiCounts.data(), state.mpiDisplacements.data(), MPI_FLOAT,
                                 communicator));
        CUDA_CHECK(cudaMemcpy(device.radB + state.idxTN(timestep, 0), row,
                              state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    }
}

static void computeDistances(SimulationState& state, const DeviceBuffers& device, MPI_Comm communicator,
                             int rank) {
    std::vector<val_t> localDistances(state.localCount);
    if (state.localCount != 0) {
        const unsigned int grid = static_cast<unsigned int>((state.localCount + CUDA_THREADS - 1) / CUDA_THREADS);
        distanceKernel<<<grid, CUDA_THREADS>>>(state.numTriangles, state.numTimesteps, state.sourceIndex,
                                               state.localBegin, state.localCount, device.radB,
                                               device.localDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDistances.data(), device.localDistances,
                              state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    if (rank == 0) state.distances.resize(state.numTriangles);
    MPI_CHECK(MPI_Gatherv(localDistances.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                          rank == 0 ? state.distances.data() : nullptr, state.mpiCounts.data(),
                          state.mpiDisplacements.data(), MPI_FLOAT, 0, communicator));
}

static unsigned long long countNonZeroFormFactors(const SimulationState& state,
                                                  const DeviceBuffers& device, MPI_Comm communicator) {
    const size_t localElements = state.localCount * state.numTriangles;
    unsigned long long localCount = 0;
    if (localElements != 0) {
        CUDA_CHECK(cudaMemset(device.nonZeroCounter, 0, sizeof(unsigned long long)));
        const unsigned int grid = static_cast<unsigned int>(
            std::min<size_t>(65535, (localElements + CUDA_THREADS - 1) / CUDA_THREADS));
        countPositiveKernel<<<grid, CUDA_THREADS, CUDA_THREADS * sizeof(unsigned long long)>>>(
            device.kij, localElements, device.nonZeroCounter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&localCount, device.nonZeroCounter, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    unsigned long long globalCount = 0;
    MPI_CHECK(MPI_Reduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator));
    return globalCount;
}

// ============================================================================
// Validation and reporting
// ============================================================================

static bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
    std::printf("\nValidation:\n");
    bool allNonNegative = true;
    val_t minDistance = std::numeric_limits<val_t>::max();
    val_t maxDistance = std::numeric_limits<val_t>::lowest();
    val_t sumDistance = ZERO;
    int nonZeroDistances = 0;

#pragma omp parallel for reduction(min:minDistance) reduction(max:maxDistance) reduction(+:sumDistance,nonZeroDistances)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const val_t distance = state.distances[static_cast<size_t>(i)];
        minDistance = std::min(minDistance, distance);
        maxDistance = std::max(maxDistance, distance);
        sumDistance += distance;
        nonZeroDistances += distance > EPSILON;
    }
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const val_t distance = state.distances[i];
        if (distance < ZERO) {
            allNonNegative = false;
            std::printf("  ERROR: Negative distance at triangle %zu: %f\n", i, distance);
        }
        if (!std::isfinite(distance)) {
            std::printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, distance);
            return false;
        }
    }
    std::printf("  Distance range: [%.4f, %.4f]\n", minDistance, maxDistance);
    std::printf("  Average distance: %.4f\n", sumDistance / static_cast<val_t>(state.numTriangles));
    std::printf("  Non-zero distances: %d/%zu\n", nonZeroDistances, state.numTriangles);

    if (state.distances[state.sourceIndex] > WAVE_SPEED * 2) {
        std::printf("  WARNING: Source triangle distance is non-zero: %.4f\n",
                    state.distances[state.sourceIndex]);
    }

    int receivedEnergy = 0;
#pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, static_cast<size_t>(i))] > EPSILON) {
                ++receivedEnergy;
                break;
            }
        }
    }
    std::printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
    if (receivedEnergy == 0) {
        std::printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const size_t totalPairs = state.numTriangles * state.numTriangles;
    std::printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij, totalPairs,
                100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalPairs));
    if (nonZeroKij == 0) {
        std::printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }
    if (!allNonNegative) return false;
    std::printf("  Validation: PASSED\n");
    return true;
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
    std::printf("               Actual count: 20, 80, 320, 1280, 5120, 20480\n");
    std::printf("  -t <num>     Number of timesteps (default: 50)\n");
    std::printf("  -s <num>     Source triangle index (default: 0)\n");
    std::printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -o           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIndex = 0;
    val_t reflectivity = 0.8f;
    int validate = 0;
    int printResults = 0;
    int help = 0;
    int valid = 1;
};

static Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            options.targetTriangles = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            options.timesteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            options.sourceIndex = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            options.reflectivity = static_cast<val_t>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (std::strcmp(argv[i], "-o") == 0) {
            options.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = 1;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            options.valid = 0;
        }
    }
    if (options.targetTriangles <= 0 || options.timesteps <= 0 || options.sourceIndex < 0 ||
        options.reflectivity < ZERO || options.reflectivity > ONE) {
        std::printf("Invalid simulation dimensions, source index, or reflectivity.\n");
        options.valid = 0;
    }
    return options;
}

static int selectGpu(MPI_Comm communicator) {
    int localRank = 0;
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &localCommunicator));
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA device is visible to this MPI rank.\n");
        MPI_Abort(communicator, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    return device;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));
    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Options options;
    if (rank == 0) options = parseOptions(argc, argv);
    MPI_CHECK(MPI_Bcast(&options, static_cast<int>(sizeof(options)), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (options.help || !options.valid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return options.valid ? 0 : 1;
    }

    const int device = selectGpu(MPI_COMM_WORLD);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (rank == 0) {
        std::printf("Room Response Simulation Benchmark\n");
        std::printf("===================================\n");
        std::printf("Target triangles: %d (using %d subdivisions)\n", options.targetTriangles,
                    getSubdivisionsForTriangleCount(options.targetTriangles));
        std::printf("Timesteps: %d\n", options.timesteps);
        std::printf("Source triangle: %d\n", options.sourceIndex);
        std::printf("Reflectivity: %.2f\n", options.reflectivity);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI ranks, OpenMP (%d host threads/rank), CUDA (%s)\n\n",
                    ranks, omp_get_max_threads(), properties.name);
    }

    SimulationState state;
    initializeSimulation(state, getSubdivisionsForTriangleCount(options.targetTriangles),
                         static_cast<size_t>(options.timesteps), static_cast<size_t>(options.sourceIndex),
                         options.reflectivity, rank, MPI_COMM_WORLD);
    computePartition(state, rank, ranks);
    if (rank == 0) {
        std::printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        std::printf("MPI receiver-row partitioning enabled; rank-local rows remain on each GPU.\n\n");
    }

    DeviceBuffers deviceBuffers;
    allocateDeviceBuffers(state, deviceBuffers);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto preStart = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::printf("Computing time delays and form factors (CUDA)...\n");
    computePrecomputation(state, deviceBuffers);
    const auto preEnd = std::chrono::high_resolution_clock::now();
    const double preLocal = std::chrono::duration<double, std::milli>(preEnd - preStart).count();
    double preDuration = 0.0;
    MPI_CHECK(MPI_Reduce(&preLocal, &preDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) std::printf("Precomputation time: %.0f ms\n\n", preDuration);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto simulationStart = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::printf("Running wave propagation simulation (CUDA + MPI)...\n");
    runSimulation(state, deviceBuffers, MPI_COMM_WORLD);
    const auto simulationEnd = std::chrono::high_resolution_clock::now();
    const double simulationLocal = std::chrono::duration<double, std::milli>(simulationEnd - simulationStart).count();
    double simulationDuration = 0.0;
    MPI_CHECK(MPI_Reduce(&simulationLocal, &simulationDuration, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) std::printf("Simulation time: %.0f ms\n\n", simulationDuration);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto distanceStart = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::printf("Computing distances via cross-correlation (CUDA)...\n");
    computeDistances(state, deviceBuffers, MPI_COMM_WORLD, rank);
    const auto distanceEnd = std::chrono::high_resolution_clock::now();
    const double distanceLocal = std::chrono::duration<double, std::milli>(distanceEnd - distanceStart).count();
    double distanceDuration = 0.0;
    MPI_CHECK(MPI_Reduce(&distanceLocal, &distanceDuration, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    unsigned long long nonZeroKij = 0;
    if (options.validate) nonZeroKij = countNonZeroFormFactors(state, deviceBuffers, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        std::printf("Distance computation time: %.0f ms\n\n", distanceDuration);
        const double totalTime = preDuration + simulationDuration + distanceDuration;
        const size_t n = state.numTriangles;
        const size_t t = state.numTimesteps;
        std::printf("Total computation time: %.0f ms\n", totalTime);
        std::printf("\nPerformance:\n");
        std::printf("  Triangles: %zu\n", n);
        std::printf("  Timesteps: %zu\n", t);
        std::printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        std::printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        std::printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        std::printf("  Total time per triangle: %.4f ms\n", totalTime / n);
        const size_t localMatrixBytes = state.localCount * n * (sizeof(val_t) + sizeof(int));
        const size_t replicatedHistoryBytes = t * n * sizeof(val_t);
        std::printf("  Per-rank GPU working memory: %.2f MB\n",
                    (localMatrixBytes + replicatedHistoryBytes) / (1024.0 * 1024.0));
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(computeHash(state)));
        if (options.printResults) {
            std::vector<double> distances(state.distances.begin(), state.distances.end());
            print_results(distances, "Distances");
        }
        if (options.validate && !validateResults(state, nonZeroKij)) result = 1;
    }

    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    deviceBuffers.release();
    MPI_CHECK(MPI_Finalize());
    return result;
}

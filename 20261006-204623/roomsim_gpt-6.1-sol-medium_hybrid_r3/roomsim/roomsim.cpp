/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
#include <memory>
#include <vector>
#include <cstdarg>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Only the MPI main thread calls MPI (MPI_THREAD_FUNNELED).
int worldRank = 0, worldSize = 1;
void rootPrintf(const char* format, ...) {
    if (worldRank != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
#define printf rootPrintf

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)
void mpiCheck(int status) {
    if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, status);
}
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                     MPI_INFO_NULL, &local));
        int localRank, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "roomsim requires a CUDA device on every rank\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&local));
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

// Allocation ownership stays on the host; there is no CPU execution fallback.
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax, int depth = 0) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                Octree* child = children[i].get();
                #pragma omp task firstprivate(child, childMin, childMax, i, depth) shared(childIndices) if(depth < 3)
                child->buildNode(childIndices[i], childMin, childMax, depth + 1);
            }
        }
        #pragma omp taskwait
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
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

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < faces.size(); ++f) {
            const auto& face = faces[f];
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return 0x1.fffffep127f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 0x1.fffffep127f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 0x1.fffffep127f;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> kij;         // Local receiver rows, downloaded only for validation
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t rowBegin = 0, localRows = 0;
    std::vector<int> counts, displacements;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<val_t> deviceAreas, deviceKij, deviceRadB, deviceDistances;
    DeviceBuffer<int> deviceTau, deviceMinDelay;
    DeviceBuffer<val_t> devicePacked;
    int communicationBatch = 1;
    uint64_t nonZeroFactors = 0;
    val_t reflectivity = 0;
    val_t* exchange = nullptr; // Pinned MPI staging; avoids requiring CUDA-aware MPI.
    ~SimulationState() { if (exchange) cudaFreeHost(exchange); }

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    state.reflectivity = reflectivity;
    state.counts.resize(worldSize);
    state.displacements.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t begin = state.numTriangles * rank / worldSize;
        size_t end = state.numTriangles * (rank + 1) / worldSize;
        state.counts[rank] = static_cast<int>(end - begin);
        state.displacements[rank] = static_cast<int>(begin);
    }
    state.rowBegin = state.displacements[worldRank];
    state.localRows = state.counts[worldRank];

    printf("Building octree...\n");
    #pragma omp parallel
    #pragma omp single
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i)
        state.areas[i] = state.triangles[i].area();
    state.distances.resize(state.numTriangles, ZERO);
    state.deviceTriangles.allocate(state.numTriangles);
    state.deviceAreas.allocate(state.numTriangles);
    state.deviceKij.allocate(state.localRows * state.numTriangles);
    state.deviceTau.allocate(state.localRows * state.numTriangles);
    state.deviceRadB.allocate(timesteps * state.numTriangles);
    state.deviceDistances.allocate(state.localRows);
    CUDA_CHECK(cudaMemcpy(state.deviceTriangles.data, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.deviceAreas.data, state.areas.data(),
                          state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    if (timesteps)
        CUDA_CHECK(cudaMemset(state.deviceRadB.data, 0, timesteps * state.numTriangles * sizeof(val_t)));
    state.deviceMinDelay.allocate(1);
    int initialDelay = static_cast<int>(std::max<size_t>(1, std::min<size_t>(timesteps, 64)));
    CUDA_CHECK(cudaMemcpy(state.deviceMinDelay.data, &initialDelay, sizeof(int), cudaMemcpyHostToDevice));

}

// ============================================================================
// Precomputation Phase
// ============================================================================

// A plain MT19937 state lets each receiver row start at exactly the same
// stream position as the sequential benchmark. Culling consumes no samples.
// Checkpoints are O(N), not O(N^2), and are scattered only once.
struct MTState {
    uint32_t words[624];
    int position = 624;
    explicit MTState(uint32_t seed = 42) {
        words[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253u * (words[i-1] ^ (words[i-1] >> 30)) + i;
    }
    void twist() {
        // Split ranges remove modulus operations and allow host vectorization.
        for (int i = 0; i < 227; ++i) {
            uint32_t y = (words[i]&0x80000000u) | (words[i+1]&0x7fffffffu);
            words[i] = words[i+397] ^ (y >> 1) ^ ((y&1u) ? 0x9908b0dfu : 0u);
        }
        for (int i = 227; i < 623; ++i) {
            uint32_t y = (words[i]&0x80000000u) | (words[i+1]&0x7fffffffu);
            words[i] = words[i-227] ^ (y >> 1) ^ ((y&1u) ? 0x9908b0dfu : 0u);
        }
        uint32_t y = (words[623]&0x80000000u) | (words[0]&0x7fffffffu);
        words[623] = words[396] ^ (y >> 1) ^ ((y&1u) ? 0x9908b0dfu : 0u);
        position = 0;
    }
    void discard(uint64_t count) {
        while (count) {
            if (position == 624) twist();
            uint64_t take = std::min(count, static_cast<uint64_t>(624 - position));
            position += static_cast<int>(take);
            count -= take;
        }
    }
};

struct FlatNode {
    Vec3 center, halfExtent;
    size_t first, count, escape;
};
void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                   std::vector<size_t>& indices) {
    size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(),
                     tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[here].escape = nodes.size();
}

__device__ bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > node.halfExtent.y*ad.z+node.halfExtent.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > node.halfExtent.z*ad.x+node.halfExtent.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > node.halfExtent.x*ad.y+node.halfExtent.y*ad.x+EPSILON) return false;
    return true;
}

// Preorder escape links replace recursive device calls and per-ray stacks.
__device__ bool blocked(const Vec3& from, const Vec3& to, size_t src, size_t dst,
                        const Triangle* triangles, const FlatNode* nodes,
                        const size_t* indices, size_t nodeCount) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    size_t node = 0;
    while (node < nodeCount) {
        const FlatNode& n = nodes[node];
        if (node != 0 && !intersectsBox(n, from, to)) {
            node = n.escape;
            continue;
        }
        for (size_t k = 0; k < n.count; ++k) {
            size_t tri = indices[n.first + k];
            if (tri == src || tri == dst) continue;
            const Triangle& t = triangles[tri];
            val_t distance = rayTriangleIntersect(from, dir, t.a, t.b, t.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}
__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}

// The in-place MT recurrence has three dependency waves, followed by the
// wraparound element. All threads participate in every barrier.
__device__ void twistParallel(uint32_t* mt, uint32_t* old) {
    for (int k = threadIdx.x; k < 624; k += blockDim.x) old[k] = mt[k];
    __syncthreads();
    for (int phase = 0; phase < 3; ++phase) {
        int end = min((phase+1)*227, 623);
        for (int k = phase*227 + threadIdx.x; k < end; k += blockDim.x) {
            uint32_t y = (old[k]&0x80000000u) | (old[k+1]&0x7fffffffu);
            uint32_t tap = k < 227 ? old[k+397] : mt[k-227];
            mt[k] = tap ^ (y >> 1) ^ ((y&1u) ? 0x9908b0dfu : 0u);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        uint32_t y = (old[623]&0x80000000u) | (mt[0]&0x7fffffffu);
        mt[623] = mt[396] ^ (y >> 1) ^ ((y&1u) ? 0x9908b0dfu : 0u);
    }
    __syncthreads();
}

__global__ void formFactorsKernel(const Triangle* triangles, size_t n, size_t rowBegin,
                                  const MTState* seeds, const FlatNode* nodes,
                                  const size_t* indices, size_t nodeCount, val_t* kij,
                                  const int* tau, int* minDelay) {
    constexpr int PAIRS = 32;
    __shared__ uint32_t mt[624], old[624];
    __shared__ val_t samples[PAIRS*NUM_RAYS*4];
    __shared__ val_t contributions[PAIRS*NUM_RAYS];
    __shared__ int offsets[PAIRS], sampleCount;
    size_t row = blockIdx.x, i = rowBegin + row;
    const Triangle& triI = triangles[i];
    for (int k = threadIdx.x; k < 624; k += blockDim.x) mt[k] = seeds[row].words[k];
    int position = seeds[row].position;
    int smallestDelay = 2147483647;
    __syncthreads();

    for (size_t first = 0; first < n; first += PAIRS) {
        if (threadIdx.x == 0) {
            int count = 0;
            for (int p = 0; p < PAIRS; ++p) {
                size_t j = first+p;
                bool active = j < n && j != i && triI.normal().dot(triangles[j].normal()) <= 0.99f;
                offsets[p] = active ? count++ * NUM_RAYS*4 : -1;
            }
            sampleCount = count * NUM_RAYS*4;
        }
        __syncthreads();
        int written = 0;
        while (written < sampleCount) {
            if (position == 624) { twistParallel(mt, old); position = 0; }
            int take = min(624-position, sampleCount-written);
            for (int k = threadIdx.x; k < take; k += blockDim.x) {
                uint32_t y = mt[position+k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                // libstdc++ generate_canonical<float,24>: one tempered word.
                samples[written+k] = fminf(static_cast<val_t>(y) * (1.0f/4294967296.0f),
                                           0x1.fffffep-1f);
            }
            __syncthreads();
            position += take;
            written += take;
        }
        for (int k = threadIdx.x; k < PAIRS*NUM_RAYS; k += blockDim.x) {
            int p = k / NUM_RAYS, ray = k % NUM_RAYS;
            val_t contribution = ZERO;
            if (offsets[p] >= 0) {
                size_t j = first+p;
                const Triangle& triJ = triangles[j];
                int off = offsets[p] + ray*4;
                Vec3 pI = samplePoint(triI, samples[off], samples[off+1]);
                Vec3 pJ = samplePoint(triJ, samples[off+2], samples[off+3]);
                if (!blocked(pI, pJ, i, j, triangles, nodes, indices, nodeCount)) {
                    Vec3 v = pJ-pI;
                    val_t squared = v.squaredNorm();
                    if (squared >= EPSILON) {
                        val_t cosI = cosPhi(v, triI.normal());
                        val_t cosJ = cosPhi(-v, triJ.normal());
                        if (cosI > ZERO && cosJ > ZERO)
                            contribution = (cosI*cosJ)/(PI*squared);
                    }
                }
            }
            contributions[k] = contribution;
        }
        __syncthreads();
        if (threadIdx.x < PAIRS && first+threadIdx.x < n) {
            val_t sum = ZERO;
            // Ordered reduction maintains the original 16-ray accumulation.
            for (int r = 0; r < NUM_RAYS; ++r)
                sum += contributions[threadIdx.x*NUM_RAYS+r];
            val_t factor = sum * INV_NUM_RAYS;
            kij[row*n + first+threadIdx.x] = factor;
            if (factor > ZERO) smallestDelay = min(smallestDelay, tau[row*n+first+threadIdx.x]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        for (int delta = 16; delta > 0; delta >>= 1)
            smallestDelay = min(smallestDelay, __shfl_down_sync(0xffffffffu, smallestDelay, delta));
        if (threadIdx.x == 0) atomicMin(minDelay, smallestDelay);
    }
}

__global__ void timeDelaysKernel(const Triangle* triangles, size_t n, size_t begin,
                                 size_t entries, int* tau) {
    for (size_t k = blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
         k < entries; k += static_cast<size_t>(gridDim.x)*blockDim.x) {
        size_t i = begin+k/n, j = k%n;
        tau[k] = i == j ? 0 : computeTau(triangles[i], triangles[j]);
    }
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on %d MPI ranks...\n", worldSize);
    const size_t n = state.numTriangles;
    std::vector<MTState> allSeeds;
    if (worldRank == 0) {
        std::vector<uint64_t> draws(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            uint64_t active = 0;
            for (size_t j = 0; j < n; ++j)
                if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f)
                    ++active;
            draws[i] = active * NUM_RAYS * 4;
        }
        allSeeds.resize(n);
        MTState stream(42);
        for (size_t i = 0; i < n; ++i) {
            allSeeds[i] = stream;
            stream.discard(draws[i]);
        }
    }
    std::vector<MTState> seeds(state.localRows);
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = state.counts[r] * sizeof(MTState);
        offsets[r] = state.displacements[r] * sizeof(MTState);
    }
    mpiCheck(MPI_Scatterv(allSeeds.data(), counts.data(), offsets.data(), MPI_BYTE,
                          seeds.data(), counts[worldRank], MPI_BYTE, 0, MPI_COMM_WORLD));
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    DeviceBuffer<MTState> deviceSeeds;
    DeviceBuffer<FlatNode> deviceNodes;
    DeviceBuffer<size_t> deviceIndices;
    deviceSeeds.allocate(seeds.size());
    deviceNodes.allocate(nodes.size());
    deviceIndices.allocate(indices.size());
    if (state.localRows) {
        CUDA_CHECK(cudaMemcpy(deviceSeeds.data, seeds.data(), seeds.size()*sizeof(MTState), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));
        formFactorsKernel<<<state.localRows, 256>>>(state.deviceTriangles.data, n, state.rowBegin,
            deviceSeeds.data, deviceNodes.data, deviceIndices.data, nodes.size(), state.deviceKij.data,
            state.deviceTau.data, state.deviceMinDelay.data);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    int localDelay;
    CUDA_CHECK(cudaMemcpy(&localDelay, state.deviceMinDelay.data, sizeof(int), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Allreduce(&localDelay, &state.communicationBatch, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    // Distinct icosphere triangle centers give strictly positive delays.
    if (state.communicationBatch < 1) MPI_Abort(MPI_COMM_WORLD, 1);
    if (worldSize > 1) {
        size_t count = n * state.communicationBatch;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&state.exchange), count*sizeof(val_t)));
        state.devicePacked.allocate(count);
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    size_t entries = state.localRows * state.numTriangles;
    if (entries) {
        unsigned blocks = static_cast<unsigned>(std::min<size_t>((entries+255)/256, 65535));
        timeDelaysKernel<<<blocks, 256>>>(state.deviceTriangles.data, state.numTriangles,
                                         state.rowBegin, entries, state.deviceTau.data);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Each block computes one receiver. Contributions are evaluated cooperatively,
// then summed in emitter order to avoid changing floating-point/tie semantics.
__global__ void propagationKernel(size_t n, size_t begin, size_t firstTime, size_t timeOff,
                                  size_t source, val_t rho, const val_t* kij,
                                  const int* tau, const val_t* areas, val_t* radB) {
    __shared__ val_t contributions[128];
    size_t row = blockIdx.x, i = begin+row, time = firstTime+blockIdx.y;
    val_t sum = ZERO;
    for (size_t first = 0; first < n; first += blockDim.x) {
        size_t j = first+threadIdx.x;
        val_t contribution = ZERO;
        if (j < n && j != i) {
            int delay = tau[row*n+j];
            val_t factor = kij[row*n+j];
            if (time >= static_cast<size_t>(delay) && factor > ZERO) {
                val_t incoming = radB[(time-delay)*n+j];
                if (incoming > ZERO) contribution = fminf(factor*areas[j], ONE)*incoming;
            }
        }
        contributions[threadIdx.x] = contribution;
        __syncthreads();
        if (threadIdx.x == 0)
            for (unsigned k = 0; k < blockDim.x && first+k < n; ++k) sum += contributions[k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        radB[time*n+i] = rho*sum + ((i == source && time < timeOff) ? ONE : ZERO);
}
// MPI gathers rank-major batches; unpack directly into time-major history.
__global__ void unpackHistory(const val_t* packed, val_t* history, size_t n,
                              size_t first, size_t batch, int ranks) {
    size_t entries = n*batch;
    for (size_t k = blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
         k < entries; k += static_cast<size_t>(gridDim.x)*blockDim.x) {
        size_t time = k/n, j = k%n;
        size_t rank = ((j+1)*ranks-1)/n;
        size_t begin = n*rank/ranks, count = n*(rank+1)/ranks-begin;
        history[(first+time)*n+j] = packed[begin*batch+time*count+j-begin];
    }
}
void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t first = 0; first < state.numTimesteps; first += state.communicationBatch) {
        size_t batch = std::min<size_t>(state.communicationBatch, state.numTimesteps-first);
        // No row depends on another row inside this batch: every positive
        // interaction's delay is at least communicationBatch.
        if (state.localRows) {
            dim3 grid(static_cast<unsigned>(state.localRows), static_cast<unsigned>(batch));
            propagationKernel<<<grid, 128>>>(n, state.rowBegin, first, state.numTimesteps/2,
                state.sourceIndex, state.reflectivity, state.deviceKij.data,
                state.deviceTau.data, state.deviceAreas.data, state.deviceRadB.data);
            CUDA_CHECK(cudaGetLastError());
        }
        for (size_t t = first; t < first+batch; ++t) {
            if ((t+1)%10 == 0 || t+1 == state.numTimesteps)
                printf("  Timestep %zu/%zu\n", t+1, state.numTimesteps);
        }
        if (worldSize > 1) {
            for (int r = 0; r < worldSize; ++r) {
                counts[r] = state.counts[r]*batch;
                offsets[r] = state.displacements[r]*batch;
            }
            if (state.localRows)
                CUDA_CHECK(cudaMemcpy2D(state.exchange+state.rowBegin*batch,
                    state.localRows*sizeof(val_t), state.deviceRadB.data+first*n+state.rowBegin,
                    n*sizeof(val_t), state.localRows*sizeof(val_t), batch, cudaMemcpyDeviceToHost));
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.exchange,
                                    counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpy(state.devicePacked.data, state.exchange, n*batch*sizeof(val_t), cudaMemcpyHostToDevice));
            unsigned blocks = static_cast<unsigned>(std::min<size_t>((n*batch+255)/256, 65535));
            unpackHistory<<<blocks, 256>>>(state.devicePacked.data, state.deviceRadB.data,
                                           n, first, batch, worldSize);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void distancesKernel(size_t n, size_t begin, size_t times, size_t source,
                                const val_t* radB, val_t* distances) {
    __shared__ val_t correlations[128];
    size_t i = begin+blockIdx.x;
    val_t maximum = ZERO;
    size_t best = 0;
    for (size_t first = 0; first < times; first += blockDim.x) {
        size_t lag = first+threadIdx.x;
        val_t sum = ZERO;
        for (size_t tt = lag; tt < times; ++tt)
            sum += radB[(tt-lag)*n+source] * radB[tt*n+i];
        correlations[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned k = 0; k < blockDim.x && first+k < times; ++k) {
                if (correlations[k] > maximum) { maximum = correlations[k]; best = first+k; }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[blockIdx.x] = WAVE_SPEED*static_cast<val_t>(best);
}
void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    if (state.localRows) {
        distancesKernel<<<state.localRows, 128>>>(state.numTriangles, state.rowBegin,
            state.numTimesteps, state.sourceIndex, state.deviceRadB.data, state.deviceDistances.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data()+state.rowBegin, state.deviceDistances.data,
                               state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    // All ranks retain the same small result vector for hashes and exit status.
    mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(),
                            state.counts.data(), state.displacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
}
void collectValidation(SimulationState& state) {
    size_t entries = state.localRows * state.numTriangles;
    state.kij.resize(entries);
    if (entries)
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.deviceKij.data, entries*sizeof(val_t), cudaMemcpyDeviceToHost));
    uint64_t nonZero = 0;
    #pragma omp parallel for reduction(+:nonZero) schedule(static)
    for (size_t k = 0; k < entries; ++k) if (state.kij[k] > EPSILON) ++nonZero;
    mpiCheck(MPI_Reduce(&nonZero, &state.nonZeroFactors, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD));
    if (worldRank == 0) {
        state.radB.resize(state.numTriangles * state.numTimesteps);
        if (!state.radB.empty())
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.deviceRadB.data,
                state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}


// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Counts have already been reduced across all receiver-row owners.
    uint64_t nonZeroKij = state.nonZeroFactors;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
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

    if (timesteps < 0) {
        fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    int64_t preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &preDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    int64_t simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &simDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    int64_t distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &distDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", n);
    printf("  Timesteps: %zu\n", t);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    printf("  MPI ranks: %d; OpenMP threads per rank: %d; exchange batch: %d\n",
           worldSize, omp_get_max_threads(), state.communicationBatch);

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Matrix memory per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int valid = 1;
    if (validate) {
        collectValidation(state);
        if (worldRank == 0) valid = validateResults(state) ? 1 : 0;
        mpiCheck(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    if (!valid) return 1;

    return 0;
}

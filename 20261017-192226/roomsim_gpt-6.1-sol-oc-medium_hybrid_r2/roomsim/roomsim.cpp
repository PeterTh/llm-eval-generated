/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI/OpenMP/CUDA implementation of room impulse response
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
#include <random>
#include <vector>
#include <cstdarg>
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

int worldRank = 0, worldSize = 1;

void rootPrintf(const char* format, ...) {
    if (worldRank != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
#define printf rootPrintf
#define HD __host__ __device__

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: a CUDA device is required\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
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

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
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
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
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
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            // Reverse winding to make normals point inward
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(size_t count) { rng.discard(count); }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
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
    std::vector<val_t> radB;        // Host copy only when validation is requested
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t rowBegin, localRows;
    std::vector<int> counts, offsets;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<val_t> deviceKij, deviceRadB, deviceAreas, deviceDistances;
    DeviceBuffer<int> deviceTau;
    unsigned long long nonZeroKij = 0;
    val_t reflectivity;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
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
    state.reflectivity = reflectivity;
    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        state.offsets[r] = state.numTriangles * r / worldSize;
        state.counts[r] = state.numTriangles * (r + 1) / worldSize - state.offsets[r];
    }
    state.rowBegin = state.offsets[worldRank];
    state.localRows = state.counts[worldRank];

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize matrices
    state.deviceTriangles.allocate(state.numTriangles);
    state.deviceAreas.allocate(state.numTriangles);
    state.deviceKij.allocate(state.localRows * state.numTriangles);
    state.deviceTau.allocate(state.localRows * state.numTriangles);
    state.deviceRadB.allocate(timesteps * state.numTriangles);
    state.deviceDistances.allocate(state.localRows);
    CUDA_CHECK(cudaMemcpy(state.deviceTriangles.data, state.triangles.data(), state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.deviceAreas.data, state.areas.data(), state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.deviceRadB.data, 0, timesteps * state.numTriangles * sizeof(val_t)));
    state.distances.resize(state.numTriangles, ZERO);

    // Emission is added directly by the propagation kernel.
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Preorder nodes with escape links give the GPU a stackless traversal with
// exactly the original octree's boxes, leaf contents, and intersection tests.
struct FlatNode {
    Vec3 center, halfExtent;
    int escape, first, count;
};

void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                   std::vector<size_t>& indices) {
    int here = static_cast<int>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<int>(indices.size()), static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[here].escape = static_cast<int>(nodes.size());
}

__device__ bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(const Vec3& from, const Vec3& to, size_t src, size_t dst,
                        const Triangle* triangles, const FlatNode* nodes,
                        const size_t* indices, int nodeCount) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    for (int n = 0; n < nodeCount;) {
        const FlatNode& node = nodes[n];
        if (n && !intersectsBox(node, from, to)) { n = node.escape; continue; }
        for (int k = 0; k < node.count; ++k) {
            size_t index = indices[node.first + k];
            if (index == src || index == dst) continue;
            const Triangle& t = triangles[index];
            val_t distance = rayTriangleIntersect(from, dir, t.a, t.b, t.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++n;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__global__ void formFactorKernel(const Triangle* triangles, const FlatNode* nodes,
                                 const size_t* indices, int nodeCount,
                                 const val_t* samples, val_t* matrix, size_t n,
                                 size_t begin, size_t localRows, size_t batchBegin,
                                 size_t batchRows) {
    size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= batchRows * n) return;
    size_t row = pair / n, j = pair % n, i = begin + batchBegin + row;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    val_t kij = ZERO;
    if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
        const val_t* random = samples + pair * (4 * NUM_RAYS);
        for (int r = 0; r < NUM_RAYS; ++r) {
            Vec3 p = sampledPoint(a, random[4*r], random[4*r+1]);
            Vec3 q = sampledPoint(b, random[4*r+2], random[4*r+3]);
            if (blocked(p, q, i, j, triangles, nodes, indices, nodeCount)) continue;
            Vec3 v = q - p;
            val_t distSqr = v.squaredNorm();
            if (distSqr < EPSILON) continue;
            val_t ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
            if (ci <= ZERO || cj <= ZERO) continue;
            kij += (ci * cj) / (PI * distSqr);
        }
    }
    // Column-major local rows coalesce accesses during propagation.
    matrix[j * localRows + batchBegin + row] = kij * INV_NUM_RAYS;
}

__global__ void countFactors(const val_t* matrix, size_t entries, unsigned long long* count) {
    unsigned long long local = 0;
    for (size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x; i < entries;
         i += size_t(gridDim.x)*blockDim.x)
        local += matrix[i] > EPSILON;
    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(count, sums[0]);
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    size_t n = state.numTriangles;
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    DeviceBuffer<FlatNode> deviceNodes;
    DeviceBuffer<size_t> deviceIndices;
    deviceNodes.allocate(nodes.size());
    deviceIndices.allocate(indices.size());
    CUDA_CHECK(cudaMemcpy(deviceNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));

    // Snapshot the ORIGINAL serial stream at row boundaries. Culling consumes
    // no random numbers; all other pairs consume exactly 4*NUM_RAYS values.
    // This makes sampling independent of both MPI and OpenMP scheduling.
    std::vector<size_t> draws(state.rowBegin + state.localRows);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < draws.size(); ++i) {
        size_t pairs = 0;
        for (size_t j = 0; j < n; ++j)
            pairs += i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f;
        draws[i] = pairs * (4 * NUM_RAYS);
    }
    std::vector<RandomGenerator> rowRng;
    rowRng.reserve(state.localRows);
    RandomGenerator rng(42);
    for (size_t i = 0; i < draws.size(); ++i) {
        if (i >= state.rowBegin) rowRng.push_back(rng);
        if (i + 1 < draws.size()) rng.discard(draws[i]);
    }

    // Bounded, double-buffered staging overlaps CPU RNG and PCIe transfers
    // with the much more expensive visibility kernels. No CUDA-aware MPI needed.
    const size_t batchRows = std::max(size_t(1), std::min(size_t(32), size_t(32*1024*1024)/(n*4*NUM_RAYS*sizeof(val_t))));
    const size_t capacity = batchRows*n*4*NUM_RAYS;
    val_t* hostSamples[2];
    DeviceBuffer<val_t> deviceSamples[2];
    cudaStream_t streams[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostSamples[b]), capacity*sizeof(val_t)));
        deviceSamples[b].allocate(capacity);
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[b], cudaStreamNonBlocking));
    }
    size_t batch = 0;
    for (size_t start = 0; start < state.localRows; start += batchRows, ++batch) {
        int b = batch % 2;
        CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        size_t rows = std::min(batchRows, state.localRows-start);
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            RandomGenerator random = rowRng[start+r];
            size_t i = state.rowBegin + start + r;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                val_t* out = hostSamples[b] + (r*n+j)*4*NUM_RAYS;
                for (int k = 0; k < 4*NUM_RAYS; ++k) out[k] = random.rand();
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceSamples[b].data, hostSamples[b], rows*n*4*NUM_RAYS*sizeof(val_t), cudaMemcpyHostToDevice, streams[b]));
        formFactorKernel<<<(rows*n+127)/128, 128, 0, streams[b]>>>(state.deviceTriangles.data,
            deviceNodes.data, deviceIndices.data, static_cast<int>(nodes.size()), deviceSamples[b].data,
            state.deviceKij.data, n, state.rowBegin, state.localRows, start, rows);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFreeHost(hostSamples[b]));
    }
    DeviceBuffer<unsigned long long> count;
    count.allocate(1);
    CUDA_CHECK(cudaMemset(count.data, 0, sizeof(unsigned long long)));
    countFactors<<<256, 256>>>(state.deviceKij.data, n*state.localRows, count.data);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long localCount;
    CUDA_CHECK(cudaMemcpy(&localCount, count.data, sizeof(localCount), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

__global__ void timeDelayKernel(const Triangle* triangles, int* tau, size_t n,
                                size_t begin, size_t rows) {
    for (size_t k = size_t(blockIdx.x)*blockDim.x+threadIdx.x; k < n*rows;
         k += size_t(blockDim.x)*gridDim.x) {
        size_t i = begin+k%rows, j = k/rows;
        tau[k] = i == j ? 0 : computeTau(triangles[i], triangles[j]);
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    timeDelayKernel<<<256, 256>>>(state.deviceTriangles.data, state.deviceTau.data,
        state.numTriangles, state.rowBegin, state.localRows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

__global__ void propagationKernel(const val_t* kij, const int* tau, const val_t* areas,
                                  val_t* radB, size_t n, size_t rows, size_t begin,
                                  size_t t, size_t timesteps, size_t source, val_t rho) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    size_t i = begin+row;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j*rows+row];
        if (t < static_cast<size_t>(delay)) continue;
        val_t factor = kij[j*rows+row];
        if (factor <= ZERO) continue;
        val_t rad = radB[(t-delay)*n+j];
        if (rad <= ZERO) continue;
        sum += fminf(factor*areas[j], ONE)*rad;
    }
    radB[t*n+i] = rho*sum + ((i == source && t < timesteps/2) ? ONE : ZERO);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    val_t* exchange = nullptr;
    if (worldSize > 1) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&exchange), state.numTriangles*sizeof(val_t)));
    // Distinct triangle centers imply strictly positive off-diagonal delays,
    // so receivers within one timestep are independent.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows) {
            propagationKernel<<<(state.localRows+127)/128, 128>>>(state.deviceKij.data,
                state.deviceTau.data, state.deviceAreas.data, state.deviceRadB.data,
                state.numTriangles, state.localRows, state.rowBegin, t, state.numTimesteps,
                state.sourceIndex, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
        }
        if (worldSize > 1) {
            val_t* timeSlice = state.deviceRadB.data+t*state.numTriangles;
            CUDA_CHECK(cudaMemcpy(exchange+state.rowBegin, timeSlice+state.rowBegin,
                state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange, state.counts.data(),
                state.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(timeSlice, exchange, state.numTriangles*sizeof(val_t), cudaMemcpyHostToDevice));
        }
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (exchange) CUDA_CHECK(cudaFreeHost(exchange));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distanceKernel(const val_t* radB, val_t* distances, size_t n,
                               size_t rows, size_t begin, size_t timesteps, size_t source) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row < rows) {
        size_t i = begin+row;
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < timesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < timesteps; ++tt) {
                val_t pB = radB[tt*n+i];
                val_t pS = radB[(tt-t)*n+source];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        distances[row] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    if (state.localRows) {
        distanceKernel<<<(state.localRows+127)/128, 128>>>(state.deviceRadB.data,
            state.deviceDistances.data, state.numTriangles, state.localRows, state.rowBegin,
            state.numTimesteps, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<val_t> local(state.localRows);
    CUDA_CHECK(cudaMemcpy(local.data(), state.deviceDistances.data, state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_Gatherv(local.data(), static_cast<int>(state.localRows), MPI_FLOAT, state.distances.data(),
        state.counts.data(), state.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    if (timesteps <= 0) {
        printf("Number of timesteps must be positive\n");
        return 1;
    }

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("Parallel execution: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n", worldSize, omp_get_max_threads());
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    long localTotalTime = preDuration + simDuration + distDuration;
    long maxTotalTime = 0;
    MPI_Reduce(&localTotalTime, &maxTotalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank != 0) {
        int status;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return status;
    }

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = maxTotalTime;
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

    // Memory usage
    size_t maxRows = (n + worldSize - 1) / worldSize;
    size_t memKij = maxRows * n * sizeof(val_t);
    size_t memTau = maxRows * n * sizeof(int);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Largest rank matrix/history storage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        state.radB.resize(state.numTriangles*state.numTimesteps);
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.deviceRadB.data,
            state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
        status = validateResults(state) ? 0 : 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

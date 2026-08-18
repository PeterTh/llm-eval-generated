/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * MPI distributes receiver rows, OpenMP parallelizes irregular visibility work,
 * and CUDA executes the dense delay, propagation, and correlation kernels.
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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

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

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
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

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

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
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Deterministic, Parallel-Safe Sampling
// ============================================================================

// A counter based generator makes every (triangle pair, ray) sample independent.
// Unlike a shared state generator it is reproducible regardless of MPI rank count
// or OpenMP scheduling, while retaining the same uniform barycentric sampling.
uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

val_t random01(uint64_t counter) {
    // Use the high 24 bits: exactly representable in float and in [0, 1).
    return static_cast<val_t>(splitmix64(counter) >> 40U) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates.
Vec3 randomPointInTriangle(const Triangle& t, uint64_t sampleCounter) {
    val_t u = random01(sampleCounter);
    val_t v = random01(sampleCounter + 1ULL);
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

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

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
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const uint64_t pair = static_cast<uint64_t>(idxI) * triangles.size() + idxJ;
        const uint64_t ray = pair * NUM_RAYS + static_cast<uint64_t>(r);
        Vec3 pI = randomPointInTriangle(triI, 42ULL + 4ULL * ray);
        Vec3 pJ = randomPointInTriangle(triJ, 42ULL + 4ULL * ray + 2ULL);

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
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t localFirst;
    size_t localCount;
    size_t maxDelay;
    size_t historyLength;

    int rank;
    int ranks;
    int device;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    // A rank owns contiguous receiver rows, avoiding replicated O(N^2) state.
    std::vector<val_t> kij;         // Local form-factor rows: localCount x N
    std::vector<val_t> localRadB;   // Local radiosity history: T x localCount
    std::vector<val_t> sourceRadB;  // Globally replicated source signal
    std::vector<val_t> distances;   // Local computed distances
    std::vector<val_t> globalDistances; // Root-only output and validation data
    std::vector<int> recvCounts;
    std::vector<int> recvDisplacements;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    val_t reflectivity;

    size_t localIdx2d(size_t localI, size_t j) const { return localI * numTriangles + j; }
    size_t localIdxTN(size_t t, size_t localI) const { return t * localCount + localI; }
};

// ============================================================================
// CUDA Storage and Kernels
// ============================================================================

int cudaRank = 0;

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "MPI failure at %s:%d while executing %s: %.*s\n",
            file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
}

#define MPI_CHECK(expression) checkMpi((expression), #expression, __FILE__, __LINE__)

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "Rank %d: CUDA failure at %s:%d while executing %s: %s\n",
            cudaRank, file, line, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { reset(); }

    void allocate(size_t newCount) {
        reset();
        count = newCount;
        if (count != 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
    }

    void reset() {
        if (pointer != nullptr) cudaFree(pointer);
        pointer = nullptr;
        count = 0;
    }

    T* data() { return pointer; }
    const T* data() const { return pointer; }

private:
    T* pointer = nullptr;
    size_t count = 0;
};

struct DeviceState {
    DeviceBuffer<Vec3> centers;
    DeviceBuffer<val_t> areas;
    DeviceBuffer<val_t> kij;
    DeviceBuffer<int> tau;
    DeviceBuffer<int> maxDelay;
    DeviceBuffer<val_t> history;
    DeviceBuffer<val_t> localRadB;
    DeviceBuffer<val_t> sourceRadB;
    DeviceBuffer<val_t> correlations;
    DeviceBuffer<val_t> distances;
};

__global__ void computeTimeDelayKernel(const Vec3* centers, int* tau, int* maxDelay,
                                       size_t numTriangles, size_t localFirst,
                                       size_t localCount) {
    const size_t localI = blockIdx.x;
    if (localI >= localCount) return;

    const size_t i = localFirst + localI;
    const Vec3 ci = centers[i];
    int threadMaxDelay = 0;
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (i == j) { tau[localI * numTriangles + j] = 0; continue; }
        const Vec3 cj = centers[j];
        const val_t dx = ci.x - cj.x;
        const val_t dy = ci.y - cj.y;
        const val_t dz = ci.z - cj.z;
        const int delay = static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy + dz * dz) * INV_WAVE_SPEED));
        tau[localI * numTriangles + j] = delay;
        threadMaxDelay = delay > threadMaxDelay ? delay : threadMaxDelay;
    }
    __shared__ int blockMaxDelay;
    if (threadIdx.x == 0) blockMaxDelay = 0;
    __syncthreads();
    atomicMax(&blockMaxDelay, threadMaxDelay);
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(maxDelay, blockMaxDelay);
}

__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
                                const val_t* history, val_t* localRadB,
                                size_t numTriangles, size_t localFirst,
                                size_t localCount, size_t historyLength,
                                size_t timestep, size_t sourceIndex,
                                size_t timeOff, val_t reflectivity) {
    const size_t localI = blockIdx.x;
    if (localI >= localCount) return;

    const size_t i = localFirst + localI;
    val_t sum = ZERO;
    const size_t rowOffset = localI * numTriangles;
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (i == j) continue;
        const int delay = tau[rowOffset + j];
        if (static_cast<int>(timestep) < delay) continue;
        const val_t formFactor = kij[rowOffset + j];
        if (formFactor <= ZERO) continue;
        const size_t sourceTime = timestep - static_cast<size_t>(delay);
        const val_t sourceRadiosity = history[(sourceTime % historyLength) * numTriangles + j];
        if (sourceRadiosity > ZERO) sum += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }

    extern __shared__ val_t partial[];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < timeOff) ? ONE : ZERO;
        localRadB[timestep * localCount + localI] = reflectivity * partial[0] + emission;
    }
}

__global__ void correlationKernel(const val_t* localRadB, const val_t* sourceRadB,
                                  val_t* correlations, size_t localCount,
                                  size_t numTimesteps) {
    const size_t block = blockIdx.x;
    const size_t localI = block / numTimesteps;
    const size_t lag = block % numTimesteps;
    if (localI >= localCount) return;

    val_t sum = ZERO;
    for (size_t t = lag + threadIdx.x; t < numTimesteps; t += blockDim.x) {
        sum += sourceRadB[t - lag] * localRadB[t * localCount + localI];
    }

    extern __shared__ val_t partial[];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) correlations[localI * numTimesteps + lag] = partial[0];
}

__global__ void selectDistanceKernel(const val_t* correlations, val_t* distances,
                                     size_t localCount, size_t numTimesteps) {
    const size_t localI = blockIdx.x;
    if (localI >= localCount) return;

    val_t bestValue = ZERO;
    int bestLag = 0;
    for (size_t lag = threadIdx.x; lag < numTimesteps; lag += blockDim.x) {
        const val_t value = correlations[localI * numTimesteps + lag];
        if (value > bestValue || (value == bestValue && static_cast<int>(lag) < bestLag)) {
            bestValue = value;
            bestLag = static_cast<int>(lag);
        }
    }

    extern __shared__ val_t values[];
    int* lags = reinterpret_cast<int*>(values + blockDim.x);
    values[threadIdx.x] = bestValue;
    lags[threadIdx.x] = bestLag;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t rightValue = values[threadIdx.x + stride];
            const int rightLag = lags[threadIdx.x + stride];
            if (rightValue > values[threadIdx.x] ||
                (rightValue == values[threadIdx.x] && rightLag < lags[threadIdx.x])) {
                values[threadIdx.x] = rightValue;
                lags[threadIdx.x] = rightLag;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[localI] = WAVE_SPEED * static_cast<val_t>(lags[0]);
}

void initializeCudaAndLaunchTimeDelays(const SimulationState& state, DeviceState& deviceState) {
    std::vector<Vec3> centers(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) centers[i] = state.triangles[i].center();

    deviceState.centers.allocate(state.numTriangles);
    deviceState.areas.allocate(state.numTriangles);
    deviceState.tau.allocate(state.localCount * state.numTriangles);
    deviceState.maxDelay.allocate(1);
    CUDA_CHECK(cudaMemcpy(deviceState.centers.data(), centers.data(),
                          centers.size() * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceState.areas.data(), state.areas.data(),
                          state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceState.maxDelay.data(), 0, sizeof(int)));
    if (state.localCount != 0) {
        constexpr unsigned int threads = 256;
        computeTimeDelayKernel<<<static_cast<unsigned int>(state.localCount), threads>>>(
            deviceState.centers.data(), deviceState.tau.data(), deviceState.maxDelay.data(), state.numTriangles,
            state.localFirst, state.localCount);
        CUDA_CHECK(cudaGetLastError());
    }
}

void finalizeTimeDelays(SimulationState& state, DeviceState& deviceState) {
    CUDA_CHECK(cudaDeviceSynchronize());
    int localMaxDelay = 0;
    CUDA_CHECK(cudaMemcpy(&localMaxDelay, deviceState.maxDelay.data(), sizeof(int), cudaMemcpyDeviceToHost));
    int globalMaxDelay = 0;
    MPI_CHECK(MPI_Allreduce(&localMaxDelay, &globalMaxDelay, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    state.maxDelay = static_cast<size_t>(globalMaxDelay);
    state.historyLength = std::max<size_t>(1, std::min(state.numTimesteps, state.maxDelay + 1));
}

void prepareSimulationDevice(const SimulationState& state, DeviceState& deviceState) {
    deviceState.kij.allocate(state.kij.size());
    deviceState.history.allocate(state.historyLength * state.numTriangles);
    deviceState.localRadB.allocate(state.localRadB.size());
    deviceState.sourceRadB.allocate(state.sourceRadB.size());
    deviceState.correlations.allocate(state.localCount * state.numTimesteps);
    deviceState.distances.allocate(state.localCount);
    if (!state.kij.empty()) {
        CUDA_CHECK(cudaMemcpy(deviceState.kij.data(), state.kij.data(),
                              state.kij.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(deviceState.history.data(), ZERO,
                          state.historyLength * state.numTriangles * sizeof(val_t)));
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int ranks, int device) {
    state.rank = rank;
    state.ranks = ranks;
    state.device = device;
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.localFirst = state.numTriangles * static_cast<size_t>(rank) / static_cast<size_t>(ranks);
    const size_t localEnd = state.numTriangles * static_cast<size_t>(rank + 1) / static_cast<size_t>(ranks);
    state.localCount = localEnd - state.localFirst;
    state.recvCounts.resize(ranks);
    state.recvDisplacements.resize(ranks);
    for (int process = 0; process < ranks; ++process) {
        const size_t begin = state.numTriangles * static_cast<size_t>(process) / static_cast<size_t>(ranks);
        const size_t end = state.numTriangles * static_cast<size_t>(process + 1) / static_cast<size_t>(ranks);
        state.recvCounts[process] = static_cast<int>(end - begin);
        state.recvDisplacements[process] = static_cast<int>(begin);
    }

    if (rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.kij.assign(state.localCount * state.numTriangles, ZERO);
    state.localRadB.assign(timesteps * state.localCount, ZERO);
    state.sourceRadB.assign(timesteps, ZERO);
    state.distances.assign(state.localCount, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) {
        printf("Computing form factors (Kij) with %d MPI ranks and %d OpenMP threads/rank...\n",
               state.ranks, omp_get_max_threads());
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t localI = 0; localI < static_cast<int64_t>(state.localCount); ++localI) {
        const size_t i = state.localFirst + static_cast<size_t>(localI);
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.localIdx2d(static_cast<size_t>(localI), j)] = computeKij(
                i, j, state.triangles, state.octree);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, DeviceState& deviceState) {
    if (state.rank == 0) printf("Running wave propagation simulation on CUDA devices...\n");
    std::vector<val_t> globalFrame(state.numTriangles, ZERO);
    constexpr unsigned int threads = 256;
    const size_t timeOff = state.numTimesteps / 2;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localCount != 0) {
            propagateKernel<<<static_cast<unsigned int>(state.localCount), threads, threads * sizeof(val_t)>>>(
                deviceState.kij.data(), deviceState.tau.data(), deviceState.areas.data(),
                deviceState.history.data(), deviceState.localRadB.data(), state.numTriangles,
                state.localFirst, state.localCount, state.historyLength, t, state.sourceIndex,
                timeOff, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(state.localRadB.data() + t * state.localCount,
                                  deviceState.localRadB.data() + t * state.localCount,
                                  state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        val_t* localFrame = state.localCount == 0 ? nullptr : state.localRadB.data() + t * state.localCount;
        MPI_CHECK(MPI_Allgatherv(localFrame,
                                 static_cast<int>(state.localCount), MPI_FLOAT,
                                 globalFrame.data(), state.recvCounts.data(),
                                 state.recvDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
        state.sourceRadB[t] = globalFrame[state.sourceIndex];
        CUDA_CHECK(cudaMemcpy(deviceState.history.data() + (t % state.historyLength) * state.numTriangles,
                              globalFrame.data(), state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, DeviceState& deviceState) {
    if (state.rank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    if (state.localCount == 0 || state.numTimesteps == 0) return;

    constexpr unsigned int threads = 256;
    CUDA_CHECK(cudaMemcpy(deviceState.sourceRadB.data(), state.sourceRadB.data(),
                          state.numTimesteps * sizeof(val_t), cudaMemcpyHostToDevice));
    const size_t correlationBlocks = state.localCount * state.numTimesteps;
    correlationKernel<<<static_cast<unsigned int>(correlationBlocks), threads, threads * sizeof(val_t)>>>(
        deviceState.localRadB.data(), deviceState.sourceRadB.data(), deviceState.correlations.data(),
        state.localCount, state.numTimesteps);
    CUDA_CHECK(cudaGetLastError());
    selectDistanceKernel<<<static_cast<unsigned int>(state.localCount), threads,
                           threads * (sizeof(val_t) + sizeof(int))>>>(
        deviceState.correlations.data(), deviceState.distances.data(),
        state.localCount, state.numTimesteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), deviceState.distances.data(),
                          state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
}

void gatherDistances(SimulationState& state) {
    if (state.rank == 0) state.globalDistances.resize(state.numTriangles);
    MPI_CHECK(MPI_Gatherv(state.distances.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                          state.rank == 0 ? state.globalDistances.data() : nullptr,
                          state.recvCounts.data(), state.recvDisplacements.data(), MPI_FLOAT,
                          0, MPI_COMM_WORLD));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    bool localValid = true;
    val_t localMin = std::numeric_limits<val_t>::max();
    val_t localMax = std::numeric_limits<val_t>::lowest();
    val_t localSum = ZERO;
    int localNonZeroDistances = 0;
    for (val_t distance : state.distances) {
        if (distance < ZERO || !std::isfinite(distance)) localValid = false;
        localMin = std::min(localMin, distance);
        localMax = std::max(localMax, distance);
        localSum += distance;
        if (distance > EPSILON) ++localNonZeroDistances;
    }

    int localReceivedEnergy = 0;
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.localRadB[state.localIdxTN(t, localI)] > EPSILON) {
                ++localReceivedEnergy;
                break;
            }
        }
    }

    long long localNonZeroKij = 0;
    for (val_t formFactor : state.kij) if (formFactor > EPSILON) ++localNonZeroKij;

    val_t minDistance = ZERO;
    val_t maxDistance = ZERO;
    val_t sumDistance = ZERO;
    int nonZeroDistances = 0;
    int receivedEnergy = 0;
    long long nonZeroKij = 0;
    int validAsInt = localValid ? 1 : 0;
    int globallyValidAsInt = 0;
    MPI_CHECK(MPI_Reduce(&localMin, &minDistance, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localMax, &maxDistance, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localSum, &sumDistance, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localNonZeroDistances, &nonZeroDistances, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&validAsInt, &globallyValidAsInt, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD));

    bool valid = false;
    if (state.rank == 0) {
        printf("\nValidation:\n");
        printf("  Distance range: [%.4f, %.4f]\n", minDistance, maxDistance);
        printf("  Average distance: %.4f\n", sumDistance / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroDistances, state.numTriangles);
        const val_t sourceDistance = state.globalDistances[state.sourceIndex];
        if (sourceDistance > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", sourceDistance);
        }
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n", nonZeroKij,
               state.numTriangles * state.numTriangles,
               100.0f * static_cast<val_t>(nonZeroKij) /
                   static_cast<val_t>(state.numTriangles * state.numTriangles));
        valid = globallyValidAsInt != 0 && receivedEnergy != 0 && nonZeroKij != 0;
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    int validInt = valid ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return validInt != 0;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
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

int selectCudaDevice(int rank) {
    MPI_Comm localCommunicator;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device is visible; roomsim requires CUDA.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    return device;
}

long maximumElapsedMilliseconds(std::chrono::high_resolution_clock::time_point start) {
    const auto end = std::chrono::high_resolution_clock::now();
    const long localElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maximumElapsed = 0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &maximumElapsed, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
    return maximumElapsed;
}

int main(int argc, char** argv) {
    int providedThreadSupport = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadSupport);
    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    cudaRank = rank;
    if (providedThreadSupport < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide required MPI_THREAD_FUNNELED support.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_dynamic(0);

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
            if (rank == 0) printUsage(argv[0]);
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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const int device = selectCudaDevice(rank);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        printf("CUDA execution: enabled (rank-local GPU mapping)\n\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, ranks, device);
    DeviceState deviceState;

    if (rank == 0) printf("\n");

    // Launch the regular Tau kernel, then overlap it with the CPU/OpenMP
    // visibility calculation while each rank builds its local Kij rows.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();
    initializeCudaAndLaunchTimeDelays(state, deviceState);
    computeFormFactors(state);
    finalizeTimeDelays(state, deviceState);
    prepareSimulationDevice(state, deviceState);
    const long preDuration = maximumElapsedMilliseconds(startPre);

    if (rank == 0) {
        printf("Precomputation time (max rank): %ld ms\n", preDuration);
        printf("Maximum propagation delay: %zu timesteps\n\n", state.maxDelay);
    }

    // Simulation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulation(state, deviceState);
    const long simDuration = maximumElapsedMilliseconds(startSim);

    if (rank == 0) {
        printf("Simulation time (max rank): %ld ms\n\n", simDuration);
    }

    // Distance computation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistances(state, deviceState);
    gatherDistances(state);
    const long distDuration = maximumElapsedMilliseconds(startDist);

    if (rank == 0) {
        printf("Distance computation time (max rank): %ld ms\n\n", distDuration);
    }

    int exitCode = 0;
    if (rank == 0) {
        const long totalTime = preDuration + simDuration + distDuration;
        const size_t n = state.numTriangles;
        const size_t t = state.numTimesteps;
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = kijOps * static_cast<double>(t);
        const double distOps = static_cast<double>(n) * static_cast<double>(t) * static_cast<double>(t);

        printf("Total computation time (sum of max phases): %ld ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        const size_t localMatrixBytes = state.localCount * n * (sizeof(val_t) + sizeof(int));
        const size_t localHistoryBytes = state.localRadB.size() * sizeof(val_t);
        printf("  Rank 0 distributed working set: %.2f MB\n",
               (localMatrixBytes + localHistoryBytes) / (1024.0 * 1024.0));
        printf("  Result hash: %016lX\n\n", computeHash(state.globalDistances));

        if (printResults) {
            std::vector<double> distData(state.globalDistances.begin(), state.globalDistances.end());
            print_results(distData, "Distances");
        }
    }

    if (validate && !validateResults(state)) exitCode = 1;
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}

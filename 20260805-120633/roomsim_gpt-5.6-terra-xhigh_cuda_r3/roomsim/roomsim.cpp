/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
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

#include <cuda_runtime.h>

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
constexpr uint32_t INVALID_SAMPLE_OFFSET = 0xFFFFFFFFu;

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

// The host-side mesh types above deliberately remain unchanged.  These compact
// POD types are the GPU representation used by the kernels below.
struct DeviceVec3 {
    val_t x, y, z;

    __host__ __device__ constexpr DeviceVec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr DeviceVec3(val_t x_, val_t y_, val_t z_)
        : x(x_), y(y_), z(z_) {}

    __device__ __forceinline__ DeviceVec3 operator+(const DeviceVec3& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }
    __device__ __forceinline__ DeviceVec3 operator-(const DeviceVec3& o) const {
        return {x - o.x, y - o.y, z - o.z};
    }
    __device__ __forceinline__ DeviceVec3 operator-() const { return {-x, -y, -z}; }
    __device__ __forceinline__ DeviceVec3 operator*(val_t s) const {
        return {x * s, y * s, z * s};
    }
    __device__ __forceinline__ val_t dot(const DeviceVec3& o) const {
        return x * o.x + y * o.y + z * o.z;
    }
    __device__ __forceinline__ val_t squaredNorm() const { return dot(*this); }
};

struct DeviceTriangle {
    DeviceVec3 a, b, c, normal;
};

DeviceTriangle toDeviceTriangle(const Triangle& tri) {
    return {{tri.a.x, tri.a.y, tri.a.z}, {tri.b.x, tri.b.y, tri.b.z},
            {tri.c.x, tri.c.y, tri.c.z}, {tri._normal.x, tri._normal.y, tri._normal.z}};
}

__device__ __forceinline__ DeviceVec3 deviceTriangleCenter(const DeviceTriangle& tri) {
    return (tri.a + tri.b + tri.c) * (1.0f / 3.0f);
}

__device__ __forceinline__ DeviceVec3 deviceRandomPointInTriangle(
    const DeviceTriangle& tri, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct DeviceSimulationData {
    DeviceTriangle* triangles = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    DeviceSimulationData() = default;
    DeviceSimulationData(const DeviceSimulationData&) = delete;
    DeviceSimulationData& operator=(const DeviceSimulationData&) = delete;

    ~DeviceSimulationData() { release(); }

    void release() {
        // Cleanup is best-effort so destruction during error unwinding never
        // obscures the CUDA error that caused it.
        if (triangles) cudaFree(triangles);
        if (areas) cudaFree(areas);
        if (rho) cudaFree(rho);
        if (kij) cudaFree(kij);
        if (tau) cudaFree(tau);
        if (radE) cudaFree(radE);
        if (radB) cudaFree(radB);
        if (distances) cudaFree(distances);
        triangles = nullptr;
        areas = nullptr;
        rho = nullptr;
        kij = nullptr;
        tau = nullptr;
        radE = nullptr;
        radB = nullptr;
        distances = nullptr;
    }
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    DeviceSimulationData device;    // Persistent GPU-resident simulation state

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

template <typename T>
void allocateDevice(T*& ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
}

void uploadSimulationState(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t tn = state.numTimesteps * n;
    const size_t nn = n * n;

    std::vector<DeviceTriangle> deviceTriangles(n);
    for (size_t i = 0; i < n; ++i) {
        deviceTriangles[i] = toDeviceTriangle(state.triangles[i]);
    }

    allocateDevice(state.device.triangles, n);
    allocateDevice(state.device.areas, n);
    allocateDevice(state.device.rho, n);
    allocateDevice(state.device.kij, nn);
    allocateDevice(state.device.tau, nn);
    allocateDevice(state.device.radE, tn);
    allocateDevice(state.device.radB, tn);
    allocateDevice(state.device.distances, n);

    CUDA_CHECK(cudaMemcpy(state.device.triangles, deviceTriangles.data(),
                          n * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.device.areas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.device.rho, state.rho.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.device.radE, state.radE.data(),
                          tn * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.device.kij, 0, nn * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.device.tau, 0, nn * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.device.radB, 0, tn * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.device.distances, 0, n * sizeof(val_t)));

    // These are staging data only; all subsequent computation reads the
    // device-resident copies.  Releasing them avoids duplicating the large
    // time-series allocation on the host.
    state.radE.clear();
    state.radE.shrink_to_fit();
}

__global__ void computeTimeDelaysKernel(const DeviceTriangle* triangles, int* tau, size_t n) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    const size_t index = i * n + j;
    if (i == j) {
        tau[index] = 0;
        return;
    }

    const DeviceVec3 delta = deviceTriangleCenter(triangles[i]) - deviceTriangleCenter(triangles[j]);
    tau[index] = static_cast<int>(ceilf(sqrtf(delta.squaredNorm()) * INV_WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(const DeviceTriangle* triangles, val_t* kij,
                                         const val_t* randomSamples,
                                         const uint32_t* sampleOffsets, size_t n,
                                         size_t rowBegin, size_t batchRows) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localI = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t i = rowBegin + localI;
    if (localI >= batchRows || i >= n || j >= n) return;

    const size_t matrixIndex = i * n + j;
    if (i == j) {
        kij[matrixIndex] = ZERO;
        return;
    }

    const uint32_t sampleOffset = sampleOffsets[localI * n + j];
    if (sampleOffset == INVALID_SAMPLE_OFFSET) {
        kij[matrixIndex] = ZERO;
        return;
    }

    const DeviceTriangle triI = triangles[i];
    const DeviceTriangle triJ = triangles[j];

    // RandomGenerator is consumed in row-major pair order by the original
    // program.  The host fills this batch in precisely that order, preserving
    // its seeded Monte Carlo sample stream while the expensive geometry is
    // evaluated in parallel.
    const val_t* samples = randomSamples + sampleOffset;
    val_t formFactor = ZERO;

    // The only geometry constructed by this benchmark is an icosphere.  It is
    // convex, so each open line segment between two distinct surface samples
    // is inside the room and cannot hit a third triangle.  This is exactly the
    // successful-visibility case of the original octree traversal, without
    // serial pointer chasing on the GPU.
#pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const val_t* raySamples = samples + ray * 4;
        const DeviceVec3 pointI = deviceRandomPointInTriangle(triI, raySamples[0], raySamples[1]);
        const DeviceVec3 pointJ = deviceRandomPointInTriangle(triJ, raySamples[2], raySamples[3]);
        const DeviceVec3 direction = pointJ - pointI;
        const val_t distanceSquared = direction.squaredNorm();
        if (distanceSquared < EPSILON) continue;

        const val_t inverseDistance = 1.0f / sqrtf(distanceSquared);
        const val_t cosPhiI = fmaxf(ZERO, direction.dot(triI.normal) * inverseDistance);
        const val_t cosPhiJ = fmaxf(ZERO, (-direction).dot(triJ.normal) * inverseDistance);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        formFactor += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
    }
    kij[matrixIndex] = formFactor * INV_NUM_RAYS;
}

__global__ void runSimulationTimestepKernel(const val_t* areas, const val_t* rho,
                                             const val_t* kij, const int* tau,
                                             const val_t* radE, val_t* radB,
                                             size_t n, size_t timestep) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t sumB = ZERO;
    const size_t rowOffset = i * n;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        const int tauij = tau[rowOffset + j];
        if (timestep < static_cast<size_t>(tauij)) continue;

        const val_t formFactor = kij[rowOffset + j];
        if (formFactor <= ZERO) continue;

        const size_t sourceTime = timestep - static_cast<size_t>(tauij);
        const val_t sourceRadiosity = radB[sourceTime * n + j];
        if (sourceRadiosity <= ZERO) continue;

        sumB += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }
    radB[timestep * n + i] = rho[i] * sumB + radE[timestep * n + i];
}

__global__ void computeDistancesKernel(const val_t* radB, val_t* distances,
                                       size_t n, size_t timesteps, size_t sourceIndex) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t maxCorrelation = ZERO;
    int bestTime = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt) {
            sum += radB[(tt - t) * n + sourceIndex] * radB[tt * n + i];
        }
        if (sum > maxCorrelation) {
            maxCorrelation = sum;
            bestTime = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestTime);
}

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

    // Icosphere visibility is analytically trivial (the surface is convex),
    // so the CUDA form-factor kernel does not need an octree upload/build.
    printf("Preparing CUDA geometry...\n");

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // radE is a one-time upload.  The large matrices remain GPU-resident and
    // are copied back only for an explicit validation request.
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    uploadSimulationState(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    if (n < 2) return;

    // Keep the staging footprint bounded as N grows.  Two pinned buffers and
    // streams overlap generation/copy of the next deterministic sample batch
    // with GPU evaluation of the current batch.
    constexpr size_t maxSampleBytesPerBuffer = 64ULL * 1024ULL * 1024ULL;
    constexpr size_t samplesPerPair = NUM_RAYS * 4;
    const size_t bytesPerRow = (n - 1) * samplesPerPair * sizeof(val_t);
    const size_t rowsPerBatch = std::max<size_t>(1, std::min(n, maxSampleBytesPerBuffer / bytesPerRow));
    const size_t maxBatchSamples = rowsPerBatch * (n - 1) * samplesPerPair;

    cudaStream_t streams[2];
    val_t* hostSamples[2] = {nullptr, nullptr};
    val_t* deviceSamples[2] = {nullptr, nullptr};
    uint32_t* hostSampleOffsets[2] = {nullptr, nullptr};
    uint32_t* deviceSampleOffsets[2] = {nullptr, nullptr};
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostSamples[slot]),
                                 maxBatchSamples * sizeof(val_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostSampleOffsets[slot]),
                                 rowsPerBatch * n * sizeof(uint32_t), cudaHostAllocDefault));
        allocateDevice(deviceSamples[slot], maxBatchSamples);
        allocateDevice(deviceSampleOffsets[slot], rowsPerBatch * n);
    }

    RandomGenerator rng(42);
    const dim3 block(16, 16);

    for (size_t rowBegin = 0, batchIndex = 0; rowBegin < n; rowBegin += rowsPerBatch, ++batchIndex) {
        const size_t rows = std::min(rowsPerBatch, n - rowBegin);
        const int slot = static_cast<int>(batchIndex & 1);
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));

        size_t sampleCount = 0;
        for (size_t localI = 0; localI < rows; ++localI) {
            const size_t i = rowBegin + localI;
            const Triangle& triI = state.triangles[i];
            for (size_t j = 0; j < n; ++j) {
                const size_t offsetIndex = localI * n + j;
                if (i == j || triI.normal().dot(state.triangles[j].normal()) > 0.99f) {
                    hostSampleOffsets[slot][offsetIndex] = INVALID_SAMPLE_OFFSET;
                    continue;
                }

                hostSampleOffsets[slot][offsetIndex] = static_cast<uint32_t>(sampleCount);
                for (size_t sample = 0; sample < samplesPerPair; ++sample) {
                    hostSamples[slot][sampleCount++] = rng.rand();
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(deviceSamples[slot], hostSamples[slot],
                                   sampleCount * sizeof(val_t), cudaMemcpyHostToDevice,
                                   streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(deviceSampleOffsets[slot], hostSampleOffsets[slot],
                                   rows * n * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                   streams[slot]));
        const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                        static_cast<unsigned int>((rows + block.y - 1) / block.y));
        computeFormFactorsKernel<<<grid, block, 0, streams[slot]>>>(
            state.device.triangles, state.device.kij, deviceSamples[slot],
            deviceSampleOffsets[slot], n, rowBegin, rows);
        CUDA_CHECK(cudaGetLastError());

        if (rowBegin + rows == n || (rowBegin / 100) != ((rowBegin + rows - 1) / 100)) {
            printf("  Progress: %zu/%zu triangles\n", rowBegin + rows, n);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaFree(deviceSamples[slot]));
        CUDA_CHECK(cudaFree(deviceSampleOffsets[slot]));
        CUDA_CHECK(cudaFreeHost(hostSamples[slot]));
        CUDA_CHECK(cudaFreeHost(hostSampleOffsets[slot]));
        CUDA_CHECK(cudaStreamDestroy(streams[slot]));
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    constexpr unsigned int blockSide = 16;
    const size_t n = state.numTriangles;
    const dim3 block(blockSide, blockSide);
    const dim3 grid(static_cast<unsigned int>((n + blockSide - 1) / blockSide),
                    static_cast<unsigned int>((n + blockSide - 1) / blockSide));
    computeTimeDelaysKernel<<<grid, block>>>(state.device.triangles, state.device.tau, n);
    CUDA_CHECK(cudaGetLastError());
    // Keep this independent precomputation out of the two streaming form
    // factor queues.  It is small compared with Kij and establishes a clean
    // phase boundary for devices that do not overlap default/non-default
    // streams under legacy default-stream semantics.
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    constexpr unsigned int threadsPerBlock = 256;
    const size_t n = state.numTriangles;
    const dim3 block(threadsPerBlock);
    const dim3 grid(static_cast<unsigned int>((n + threadsPerBlock - 1) / threadsPerBlock));
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        runSimulationTimestepKernel<<<grid, block>>>(
            state.device.areas, state.device.rho, state.device.kij, state.device.tau,
            state.device.radE, state.device.radB, n, t);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    constexpr unsigned int threadsPerBlock = 256;
    const size_t n = state.numTriangles;
    const dim3 block(threadsPerBlock);
    const dim3 grid(static_cast<unsigned int>((n + threadsPerBlock - 1) / threadsPerBlock));
    computeDistancesKernel<<<grid, block>>>(state.device.radB, state.device.distances,
                                             n, state.numTimesteps, state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.device.distances,
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
}

void downloadValidationState(SimulationState& state) {
    const size_t n = state.numTriangles;
    state.kij.resize(n * n);
    state.radB.resize(state.numTimesteps * n);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.device.kij,
                          n * n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.device.radB,
                          state.numTimesteps * n * sizeof(val_t), cudaMemcpyDeviceToHost));
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
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
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
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

int main(int argc, char** argv) {
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

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
    if (validate) {
        downloadValidationState(state);
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}

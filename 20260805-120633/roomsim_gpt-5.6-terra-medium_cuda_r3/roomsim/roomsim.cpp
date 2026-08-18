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

    // CUDA allocations are kept alive across all phases.  Host copies remain for
    // validation, hashing, and the established output interface.
    struct DeviceData {
        struct DeviceTriangle { float3 a, b, c, normal; };
        struct DeviceNode {
            float3 center, halfExtent;
            int children[8];
            int begin, count;
        };
        DeviceTriangle* triangles = nullptr;
        DeviceNode* nodes = nullptr;
        int* leafIndices = nullptr;
        val_t *areas = nullptr, *rho = nullptr, *kij = nullptr, *radE = nullptr, *radB = nullptr, *distances = nullptr;
        int* tau = nullptr;
        int nodeCount = 0;
        int leafIndexCount = 0;

        void release() {
            cudaFree(triangles); cudaFree(nodes); cudaFree(leafIndices);
            cudaFree(areas); cudaFree(rho); cudaFree(kij); cudaFree(tau);
            cudaFree(radE); cudaFree(radB); cudaFree(distances);
            triangles = nullptr; nodes = nullptr; leafIndices = nullptr;
            areas = rho = kij = radE = radB = distances = nullptr; tau = nullptr;
        }
        ~DeviceData() { release(); }
        DeviceData() = default;
        DeviceData(const DeviceData&) = delete;
        DeviceData& operator=(const DeviceData&) = delete;
    } device;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// CUDA implementation
// ============================================================================

inline void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure during %s: %s\n", what, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ float3 dsub(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 dadd(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 dmul(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ float ddot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ float3 dcross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float dnorm(float3 a) { return sqrtf(ddot(a, a)); }

__device__ __forceinline__ uint32_t mixBits(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float random01(uint32_t seed) {
    return (mixBits(seed) >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ bool rayIntersectsBox(float3 p1, float3 p2,
                                                   const SimulationState::DeviceData::DeviceNode& node) {
    float3 d = dmul(dsub(p2, p1), 0.5f);
    float3 c = dsub(dadd(p1, d), node.center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    float3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z - d.z*c.y) > h.y*ad.z + h.z*ad.y + EPSILON) return false;
    if (fabsf(d.z*c.x - d.x*c.z) > h.z*ad.x + h.x*ad.z + EPSILON) return false;
    return fabsf(d.x*c.y - d.y*c.x) <= h.x*ad.y + h.y*ad.x + EPSILON;
}

__device__ __forceinline__ float rayTriangleDistance(float3 orig, float3 dir,
                                                       const SimulationState::DeviceData::DeviceTriangle& tri) {
    float3 e1 = dsub(tri.b, tri.a), e2 = dsub(tri.c, tri.a);
    float3 p = dcross(dir, e2);
    float det = ddot(e1, p);
    if (fabsf(det) < EPSILON) return 3.402823466e+38F;
    float inv = 1.0f / det;
    float u = ddot(dsub(orig, tri.a), p) * inv;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38F;
    float3 q = dcross(dsub(orig, tri.a), e1);
    float v = ddot(dir, q) * inv;
    return (v < 0.0f || u + v > 1.0f) ? 3.402823466e+38F : ddot(e2, q) * inv;
}

__device__ bool blockedOnDevice(float3 from, float3 to, int src, int dst,
                                 const SimulationState::DeviceData::DeviceTriangle* triangles,
                                 const SimulationState::DeviceData::DeviceNode* nodes,
                                 const int* leafIndices) {
    float3 delta = dsub(to, from);
    float length = dnorm(delta);
    if (length < EPSILON) return true;
    float3 dir = dmul(delta, 1.0f / length);
    // The generated octree is shallow; a fixed local stack avoids global-memory traversal overhead.
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top) {
        const auto& node = nodes[stack[--top]];
        if (node.count) {
            for (int q = 0; q < node.count; ++q) {
                int index = leafIndices[node.begin + q];
                if (index == src || index == dst) continue;
                float hit = rayTriangleDistance(from, dir, triangles[index]);
                if (hit > EPSILON && hit < length - EPSILON) return true;
            }
        } else {
            // Reverse push order preserves the original child visitation order.
            for (int c = 7; c >= 0; --c) {
                int child = node.children[c];
                if (child >= 0 && rayIntersectsBox(from, to, nodes[child])) stack[top++] = child;
            }
        }
    }
    return false;
}

__global__ void timeDelayKernel(const SimulationState::DeviceData::DeviceTriangle* triangles, int* tau, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    if (i == j) { tau[i*n+j] = 0; return; }
    float3 ci = dmul(dadd(dadd(triangles[i].a, triangles[i].b), triangles[i].c), 1.0f/3.0f);
    float3 cj = dmul(dadd(dadd(triangles[j].a, triangles[j].b), triangles[j].c), 1.0f/3.0f);
    tau[i*n+j] = __float2int_ru(dnorm(dsub(ci, cj)) * INV_WAVE_SPEED);
}

__global__ void formFactorKernel(const SimulationState::DeviceData::DeviceTriangle* triangles,
                                 const SimulationState::DeviceData::DeviceNode* nodes, const int* leafIndices,
                                 val_t* kij, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || i == j) return;
    const auto ti = triangles[i];
    const auto tj = triangles[j];
    if (ddot(ti.normal, tj.normal) > 0.99f) { kij[i*n+j] = ZERO; return; }
    float total = 0.0f;
    uint32_t pairSeed = (uint32_t)i * 0x9e3779b9U ^ (uint32_t)j * 0x85ebca6bU ^ 42U;
    #pragma unroll
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u = random01(pairSeed + 4U*r), v = random01(pairSeed + 4U*r + 1U);
        if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
        float3 pi = dadd(ti.a, dadd(dmul(dsub(ti.b, ti.a), u), dmul(dsub(ti.c, ti.a), v)));
        u = random01(pairSeed + 4U*r + 2U); v = random01(pairSeed + 4U*r + 3U);
        if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
        float3 pj = dadd(tj.a, dadd(dmul(dsub(tj.b, tj.a), u), dmul(dsub(tj.c, tj.a), v)));
        if (blockedOnDevice(pi, pj, i, j, triangles, nodes, leafIndices)) continue;
        float3 ray = dsub(pj, pi);
        float distanceSquared = ddot(ray, ray);
        if (distanceSquared < EPSILON) continue;
        float invDistance = rsqrtf(distanceSquared);
        float ci = fmaxf(0.0f, ddot(ray, ti.normal) * invDistance);
        float cj = fmaxf(0.0f, -ddot(ray, tj.normal) * invDistance);
        if (ci > 0.0f && cj > 0.0f) total += ci * cj / (PI * distanceSquared);
    }
    kij[i*n+j] = total * INV_NUM_RAYS;
}

__global__ void propagateKernel(const val_t* areas, const val_t* rho, const val_t* kij, const int* tau,
                                const val_t* radE, val_t* radB, int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    const int row = i * n;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[row+j];
        if (t < delay) continue;
        float factor = kij[row+j];
        if (factor <= 0.0f) continue;
        float source = radB[(t-delay)*n+j];
        if (source > 0.0f) sum += fminf(factor * areas[j], 1.0f) * source;
    }
    radB[t*n+i] = rho[i] * sum + radE[t*n+i];
}

__global__ void distanceKernel(const val_t* radB, val_t* distances, int n, int steps, int source) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float maxCorrelation = 0.0f;
    int best = 0;
    for (int t = 0; t < steps; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < steps; ++tt) sum += radB[tt*n+source] * radB[tt*n+i];
        if (sum > maxCorrelation) { maxCorrelation = sum; best = t; }
    }
    distances[i] = WAVE_SPEED * best;
}

struct FlattenedOctree {
    std::vector<SimulationState::DeviceData::DeviceNode> nodes;
    std::vector<int> leafIndices;
};

int flattenOctree(const Octree& source, FlattenedOctree& target) {
    int here = static_cast<int>(target.nodes.size());
    target.nodes.emplace_back();
    auto& node = target.nodes[here];
    node.center = make_float3(source.center.x, source.center.y, source.center.z);
    node.halfExtent = make_float3(source.halfExtent.x, source.halfExtent.y, source.halfExtent.z);
    std::fill(std::begin(node.children), std::end(node.children), -1);
    node.begin = static_cast<int>(target.leafIndices.size());
    node.count = static_cast<int>(source.triangleIndices.size());
    target.leafIndices.insert(target.leafIndices.end(), source.triangleIndices.begin(), source.triangleIndices.end());
    if (!node.count) {
        for (int i = 0; i < 8; ++i) {
            if (source.children[i]) {
                int child = flattenOctree(*source.children[i], target);
                // Recursive growth may reallocate nodes, so do not retain a reference here.
                target.nodes[here].children[i] = child;
            }
        }
    }
    return here;
}

void initializeDeviceState(SimulationState& state) {
    auto& d = state.device;
    std::vector<SimulationState::DeviceData::DeviceTriangle> triangles(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const auto& t = state.triangles[i];
        triangles[i] = {make_float3(t.a.x,t.a.y,t.a.z), make_float3(t.b.x,t.b.y,t.b.z), make_float3(t.c.x,t.c.y,t.c.z), make_float3(t._normal.x,t._normal.y,t._normal.z)};
    }
    FlattenedOctree flat;
    flattenOctree(state.octree, flat);
    d.nodeCount = static_cast<int>(flat.nodes.size()); d.leafIndexCount = static_cast<int>(flat.leafIndices.size());
    const size_t n = state.numTriangles, nn = n*n, tn = n*state.numTimesteps;
    cudaCheck(cudaMalloc(&d.triangles, n*sizeof(*d.triangles)), "allocating device triangles");
    cudaCheck(cudaMalloc(&d.nodes, d.nodeCount*sizeof(*d.nodes)), "allocating device octree");
    cudaCheck(cudaMalloc(&d.leafIndices, d.leafIndexCount*sizeof(*d.leafIndices)), "allocating octree leaves");
    cudaCheck(cudaMalloc(&d.areas, n*sizeof(val_t)), "allocating areas"); cudaCheck(cudaMalloc(&d.rho, n*sizeof(val_t)), "allocating rho");
    cudaCheck(cudaMalloc(&d.kij, nn*sizeof(val_t)), "allocating kij"); cudaCheck(cudaMalloc(&d.tau, nn*sizeof(int)), "allocating tau");
    cudaCheck(cudaMalloc(&d.radE, tn*sizeof(val_t)), "allocating radE"); cudaCheck(cudaMalloc(&d.radB, tn*sizeof(val_t)), "allocating radB");
    cudaCheck(cudaMalloc(&d.distances, n*sizeof(val_t)), "allocating distances");
    cudaCheck(cudaMemcpy(d.triangles, triangles.data(), n*sizeof(*d.triangles), cudaMemcpyHostToDevice), "uploading triangles");
    cudaCheck(cudaMemcpy(d.nodes, flat.nodes.data(), d.nodeCount*sizeof(*d.nodes), cudaMemcpyHostToDevice), "uploading octree");
    cudaCheck(cudaMemcpy(d.leafIndices, flat.leafIndices.data(), d.leafIndexCount*sizeof(*d.leafIndices), cudaMemcpyHostToDevice), "uploading octree leaves");
    cudaCheck(cudaMemcpy(d.areas, state.areas.data(), n*sizeof(val_t), cudaMemcpyHostToDevice), "uploading areas");
    cudaCheck(cudaMemcpy(d.rho, state.rho.data(), n*sizeof(val_t), cudaMemcpyHostToDevice), "uploading rho");
    cudaCheck(cudaMemcpy(d.radE, state.radE.data(), tn*sizeof(val_t), cudaMemcpyHostToDevice), "uploading emissions");
    cudaCheck(cudaMemset(d.kij, 0, nn*sizeof(val_t)), "clearing kij"); cudaCheck(cudaMemset(d.tau, 0, nn*sizeof(int)), "clearing tau"); cudaCheck(cudaMemset(d.radB, 0, tn*sizeof(val_t)), "clearing radB");
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

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    initializeDeviceState(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    constexpr int tile = 16;
    dim3 block(tile, tile), grid((n + tile - 1) / tile, (n + tile - 1) / tile);
    formFactorKernel<<<grid, block>>>(state.device.triangles, state.device.nodes, state.device.leafIndices,
                                      state.device.kij, n);
    cudaCheck(cudaGetLastError(), "launching form-factor kernel");
    cudaCheck(cudaDeviceSynchronize(), "computing form factors");
    cudaCheck(cudaMemcpy(state.kij.data(), state.device.kij, state.kij.size()*sizeof(val_t), cudaMemcpyDeviceToHost),
              "downloading form factors");
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    constexpr int tile = 16;
    dim3 block(tile, tile), grid((n + tile - 1) / tile, (n + tile - 1) / tile);
    timeDelayKernel<<<grid, block>>>(state.device.triangles, state.device.tau, n);
    cudaCheck(cudaGetLastError(), "launching time-delay kernel");
    cudaCheck(cudaDeviceSynchronize(), "computing time delays");
    cudaCheck(cudaMemcpy(state.tau.data(), state.device.tau, state.tau.size()*sizeof(int), cudaMemcpyDeviceToHost),
              "downloading time delays");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagateKernel<<<(n + 255) / 256, 256>>>(state.device.areas, state.device.rho, state.device.kij,
                                                   state.device.tau, state.device.radE, state.device.radB,
                                                   n, static_cast<int>(t));
        cudaCheck(cudaGetLastError(), "launching propagation kernel");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "running wave propagation");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    distanceKernel<<<(n + 255) / 256, 256>>>(state.device.radB, state.device.distances, n,
                                              static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex));
    cudaCheck(cudaGetLastError(), "launching distance kernel");
    cudaCheck(cudaDeviceSynchronize(), "computing distances");
    cudaCheck(cudaMemcpy(state.distances.data(), state.device.distances, state.distances.size()*sizeof(val_t), cudaMemcpyDeviceToHost),
              "downloading distances");
    cudaCheck(cudaMemcpy(state.radB.data(), state.device.radB, state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost),
              "downloading radiosity");
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
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}

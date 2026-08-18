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
#include <cfloat>
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

// Flat, pointer-free representations used by CUDA kernels.  The host-side
// Triangle and Octree types intentionally remain unchanged so mesh generation
// and the visibility partitioning retain their original semantics.
struct DeviceVec3 {
    val_t x, y, z;
};

struct DeviceTriangle {
    DeviceVec3 a, b, c, normal;
};

struct DeviceOctreeNode {
    DeviceVec3 center;
    DeviceVec3 halfExtent;
    int children[8];
    idx_t triangleOffset;
    idx_t triangleCount;
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

// Convert the pointer-based host octree into contiguous arrays suitable for
// device traversal.  Nodes are emitted in depth-first order and child links
// become integer indices, while leaf triangle lists are packed separately.
int flattenOctree(const Octree& source,
                  std::vector<DeviceOctreeNode>& nodes,
                  std::vector<idx_t>& leafTriangles) {
    const int nodeIndex = static_cast<int>(nodes.size());
    DeviceOctreeNode flat{};
    flat.center = {source.center.x, source.center.y, source.center.z};
    flat.halfExtent = {source.halfExtent.x, source.halfExtent.y, source.halfExtent.z};
    for (int i = 0; i < 8; ++i) flat.children[i] = -1;
    flat.triangleOffset = static_cast<idx_t>(leafTriangles.size());
    flat.triangleCount = static_cast<idx_t>(source.triangleIndices.size());
    nodes.push_back(flat);

    leafTriangles.reserve(leafTriangles.size() + source.triangleIndices.size());
    for (size_t index : source.triangleIndices) {
        leafTriangles.push_back(static_cast<idx_t>(index));
    }

    for (int child = 0; child < 8; ++child) {
        if (source.children[child]) {
            nodes[nodeIndex].children[child] = flattenOctree(
                *source.children[child], nodes, leafTriangles);
        }
    }
    return nodeIndex;
}

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
// CUDA kernels
// ============================================================================

__device__ __forceinline__ DeviceVec3 deviceSub(DeviceVec3 a, DeviceVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ DeviceVec3 deviceAdd(DeviceVec3 a, DeviceVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ DeviceVec3 deviceScale(DeviceVec3 a, val_t s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ val_t deviceDot(DeviceVec3 a, DeviceVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ DeviceVec3 deviceCross(DeviceVec3 a, DeviceVec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ val_t deviceSquaredNorm(DeviceVec3 v) {
    return deviceDot(v, v);
}

__device__ __forceinline__ val_t deviceUniform(uint32_t& state) {
    // A small, deterministic generator gives every (receiver, emitter) pair
    // an independent stream.  This removes the serial RNG dependency while
    // retaining the original uniform barycentric sampling semantics.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<val_t>(state >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ uint32_t pairSeed(size_t receiver, size_t emitter) {
    uint32_t x = 42u;
    x ^= static_cast<uint32_t>(receiver) * 0x9e3779b9u;
    x ^= static_cast<uint32_t>(emitter) * 0x85ebca6bu;
    x ^= static_cast<uint32_t>(receiver >> 32) * 0xc2b2ae35u;
    x ^= static_cast<uint32_t>(emitter >> 32) * 0x27d4eb2du;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x == 0 ? 0x6d2b79f5u : x;
}

__device__ __forceinline__ DeviceVec3 deviceRandomPoint(
    const DeviceTriangle& triangle, uint32_t& state) {
    val_t u = deviceUniform(state);
    val_t v = deviceUniform(state);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return deviceAdd(triangle.a,
                     deviceAdd(deviceScale(deviceSub(triangle.b, triangle.a), u),
                               deviceScale(deviceSub(triangle.c, triangle.a), v)));
}

__device__ val_t deviceRayTriangleIntersect(DeviceVec3 orig, DeviceVec3 dir,
                                             DeviceVec3 v0, DeviceVec3 v1,
                                             DeviceVec3 v2) {
    const DeviceVec3 e1 = deviceSub(v1, v0);
    const DeviceVec3 e2 = deviceSub(v2, v0);
    const DeviceVec3 pvec = deviceCross(dir, e2);
    const val_t det = deviceDot(e1, pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    const val_t invDet = 1.0f / det;
    const DeviceVec3 tvec = deviceSub(orig, v0);
    const val_t u = deviceDot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    const DeviceVec3 qvec = deviceCross(tvec, e1);
    const val_t v = deviceDot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return deviceDot(e2, qvec) * invDet;
}

__device__ __forceinline__ bool deviceRayIntersectsBox(
    DeviceVec3 p1, DeviceVec3 p2, const DeviceOctreeNode& node) {
    const DeviceVec3 d = deviceScale(deviceSub(p2, p1), 0.5f);
    const DeviceVec3 c = deviceSub(deviceAdd(p1, d), node.center);
    const DeviceVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool deviceRayBlocked(DeviceVec3 from, DeviceVec3 to,
                                  size_t sourceIndex, size_t destinationIndex,
                                  const DeviceTriangle* triangles,
                                  const DeviceOctreeNode* nodes,
                                  const idx_t* leafTriangles,
                                  int nodeCount) {
    const DeviceVec3 direction = deviceSub(to, from);
    const val_t rayLength = sqrtf(deviceSquaredNorm(direction));
    if (rayLength < EPSILON) return true;
    const DeviceVec3 directionNormal = deviceScale(direction, 1.0f / rayLength);

    // The generated octree has a bounded depth because construction stops at
    // MAX_OCTREE_LEAF_SIZE.  This fixed stack avoids device recursion and
    // keeps visibility traversal allocation-free.
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = 0;
    while (stackSize > 0) {
        const DeviceOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount != 0) {
            for (idx_t k = 0; k < node.triangleCount; ++k) {
                const size_t index = leafTriangles[node.triangleOffset + k];
                if (index == sourceIndex || index == destinationIndex) continue;
                const DeviceTriangle& triangle = triangles[index];
                const val_t distance = deviceRayTriangleIntersect(
                    from, directionNormal, triangle.a, triangle.b, triangle.c);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }

        for (int child = 7; child >= 0; --child) {
            const int childIndex = node.children[child];
            if (childIndex >= 0 && deviceRayIntersectsBox(from, to, nodes[childIndex])) {
                if (stackSize < static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
                    stack[stackSize++] = childIndex;
                }
            }
        }
    }
    return false;
}

__global__ void computeFormFactorsKernel(
    size_t rowStart, size_t localRows, size_t numTriangles,
    const DeviceTriangle* triangles, const DeviceOctreeNode* nodes,
    const idx_t* leafTriangles, int nodeCount, val_t* output) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = localRows * numTriangles;
    if (pair >= pairCount) return;

    const size_t localRow = pair / numTriangles;
    const size_t emitter = pair % numTriangles;
    const size_t receiver = rowStart + localRow;
    if (receiver == emitter) {
        output[pair] = ZERO;
        return;
    }

    const DeviceTriangle& triI = triangles[receiver];
    const DeviceTriangle& triJ = triangles[emitter];
    if (deviceDot(triI.normal, triJ.normal) > 0.99f) {
        output[pair] = ZERO;
        return;
    }

    uint32_t randomState = pairSeed(receiver, emitter);
    val_t kij = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const DeviceVec3 pI = deviceRandomPoint(triI, randomState);
        const DeviceVec3 pJ = deviceRandomPoint(triJ, randomState);
        if (deviceRayBlocked(pI, pJ, receiver, emitter, triangles, nodes,
                             leafTriangles, nodeCount)) continue;

        const DeviceVec3 v = deviceSub(pJ, pI);
        const val_t distanceSquared = deviceSquaredNorm(v);
        if (distanceSquared < EPSILON) continue;

        const val_t distance = sqrtf(distanceSquared);
        const val_t cosPhiI = fmaxf(ZERO, deviceDot(v, triI.normal) / distance);
        const val_t cosPhiJ = fmaxf(ZERO, -deviceDot(v, triJ.normal) / distance);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
    }
    output[pair] = kij * INV_NUM_RAYS;
}

__global__ void computeTimeDelaysKernel(
    size_t rowStart, size_t localRows, size_t numTriangles,
    const DeviceTriangle* triangles, int* output) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = localRows * numTriangles;
    if (pair >= pairCount) return;

    const size_t receiver = rowStart + pair / numTriangles;
    const size_t emitter = pair % numTriangles;
    if (receiver == emitter) {
        output[pair] = 0;
        return;
    }

    const DeviceTriangle& triI = triangles[receiver];
    const DeviceTriangle& triJ = triangles[emitter];
    const DeviceVec3 centerI = deviceScale(deviceAdd(deviceAdd(triI.a, triI.b), triI.c), 1.0f / 3.0f);
    const DeviceVec3 centerJ = deviceScale(deviceAdd(deviceAdd(triJ.a, triJ.b), triJ.c), 1.0f / 3.0f);
    const DeviceVec3 delta = deviceSub(centerI, centerJ);
    output[pair] = static_cast<int>(ceilf(sqrtf(deviceSquaredNorm(delta)) * INV_WAVE_SPEED));
}

// One block computes one receiver row.  Threads walk the emitter row in
// parallel, giving coalesced accesses to the row-major form-factor matrix and
// a deterministic reduction for each output radiosity value.
__global__ void propagateKernel(
    size_t timestep, size_t rowStart, size_t localRows, size_t numTriangles,
    size_t numTimesteps, size_t sourceIndex, const val_t* kij, const int* tau,
    const val_t* areas, const val_t* rho, const val_t* radB, val_t* output) {
    const size_t localRow = blockIdx.x;
    if (localRow >= localRows) return;
    const size_t receiver = rowStart + localRow;
    val_t sumB = ZERO;

    for (size_t emitter = threadIdx.x; emitter < numTriangles; emitter += blockDim.x) {
        if (receiver == emitter) continue;
        const size_t matrixIndex = localRow * numTriangles + emitter;
        const int delay = tau[matrixIndex];
        if (static_cast<int>(timestep) < delay) continue;
        const val_t formFactor = kij[matrixIndex];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radB[(timestep - static_cast<size_t>(delay)) * numTriangles + emitter];
        if (sourceRadiosity <= ZERO) continue;
        const val_t weighted = formFactor * areas[emitter];
        sumB += (weighted < ONE ? weighted : ONE) * sourceRadiosity;
    }

    extern __shared__ val_t reduction[];
    reduction[threadIdx.x] = sumB;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (timestep < numTimesteps / 2 && receiver == sourceIndex) ? ONE : ZERO;
        output[localRow] = rho[receiver] * reduction[0] + emission;
    }
}

struct DistanceCandidate {
    val_t correlation;
    int timestep;
};

__global__ void distanceKernel(
    size_t rowStart, size_t localRows, size_t numTriangles, size_t numTimesteps,
    size_t sourceIndex, const val_t* radB, val_t* distances) {
    const size_t localRow = blockIdx.x;
    if (localRow >= localRows) return;
    const size_t receiver = rowStart + localRow;

    DistanceCandidate best{ZERO, 0};
    for (size_t lag = threadIdx.x; lag < numTimesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (size_t time = lag; time < numTimesteps; ++time) {
            const val_t source = radB[(time - lag) * numTriangles + sourceIndex];
            const val_t response = radB[time * numTriangles + receiver];
            correlation += source * response;
        }
        const int lagInt = static_cast<int>(lag);
        if (correlation > best.correlation ||
            (correlation == best.correlation && lagInt < best.timestep)) {
            best = {correlation, lagInt};
        }
    }

    extern __shared__ DistanceCandidate candidates[];
    candidates[threadIdx.x] = best;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const DistanceCandidate other = candidates[threadIdx.x + stride];
            if (other.correlation > candidates[threadIdx.x].correlation ||
                (other.correlation == candidates[threadIdx.x].correlation &&
                 other.timestep < candidates[threadIdx.x].timestep)) {
                candidates[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[localRow] = WAVE_SPEED * static_cast<val_t>(candidates[0].timestep);
    }
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

    int mpiRank = 0;
    int mpiSize = 1;
    size_t rowStart = 0;
    size_t localRows = 0;

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

    size_t idx2d(size_t i, size_t j) const {
        return (i - rowStart) * numTriangles + j;
    }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

struct CudaContext {
    int mpiRank;
    int device;
    size_t numTriangles;
    size_t numTimesteps;
    size_t rowStart;
    size_t localRows;

    std::vector<DeviceTriangle> hostTriangles;
    std::vector<DeviceOctreeNode> hostNodes;
    std::vector<idx_t> hostLeafTriangles;

    DeviceTriangle* deviceTriangles = nullptr;
    DeviceOctreeNode* deviceNodes = nullptr;
    idx_t* deviceLeafTriangles = nullptr;
    val_t* deviceKij = nullptr;
    int* deviceTau = nullptr;
    val_t* deviceAreas = nullptr;
    val_t* deviceRho = nullptr;
    val_t* deviceRadB = nullptr;
    val_t* deviceStep = nullptr;
    val_t* deviceDistances = nullptr;

    CudaContext(const SimulationState& state, int localDevice)
        : mpiRank(state.mpiRank), device(localDevice), numTriangles(state.numTriangles),
          numTimesteps(state.numTimesteps), rowStart(state.rowStart), localRows(state.localRows) {
        check(cudaSetDevice(device), "cudaSetDevice");

        hostTriangles.resize(numTriangles);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(numTriangles); ++i) {
            const Triangle& source = state.triangles[static_cast<size_t>(i)];
            hostTriangles[static_cast<size_t>(i)] = {
                {source.a.x, source.a.y, source.a.z},
                {source.b.x, source.b.y, source.b.z},
                {source.c.x, source.c.y, source.c.z},
                {source._normal.x, source._normal.y, source._normal.z}};
        }
        flattenOctree(state.octree, hostNodes, hostLeafTriangles);

        allocateAndCopy(deviceTriangles, hostTriangles);
        allocateAndCopy(deviceNodes, hostNodes);
        allocateAndCopy(deviceLeafTriangles, hostLeafTriangles);
        allocateAndCopy(deviceAreas, state.areas);
        allocateAndCopy(deviceRho, state.rho);

        allocate(deviceKij, localRows * numTriangles);
        allocate(deviceTau, localRows * numTriangles);
        allocate(deviceStep, localRows);
        allocate(deviceDistances, localRows);
        allocate(deviceRadB, numTimesteps * numTriangles);
        if (deviceRadB) check(cudaMemset(deviceRadB, 0, numTimesteps * numTriangles * sizeof(val_t)),
                              "cudaMemset(radB)");
    }

    ~CudaContext() {
        cudaSetDevice(device);
        cudaFree(deviceTriangles);
        cudaFree(deviceNodes);
        cudaFree(deviceLeafTriangles);
        cudaFree(deviceKij);
        cudaFree(deviceTau);
        cudaFree(deviceAreas);
        cudaFree(deviceRho);
        cudaFree(deviceRadB);
        cudaFree(deviceStep);
        cudaFree(deviceDistances);
    }

    void check(cudaError_t error, const char* operation) const {
        if (error != cudaSuccess) {
            std::fprintf(stderr, "Rank %d CUDA error in %s: %s\n", mpiRank,
                         operation, cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    template <typename T>
    void allocate(T*& pointer, size_t count) {
        if (count == 0) return;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "cudaMalloc");
    }

    template <typename T>
    void allocateAndCopy(T*& pointer, const std::vector<T>& source) {
        allocate(pointer, source.size());
        if (!source.empty()) {
            check(cudaMemcpy(pointer, source.data(), source.size() * sizeof(T),
                             cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice");
        }
    }

    void synchronize(const char* operation) const {
        check(cudaGetLastError(), operation);
        check(cudaDeviceSynchronize(), operation);
    }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          int mpiRank, int mpiSize) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    const size_t baseRows = state.numTriangles / static_cast<size_t>(mpiSize);
    const size_t extraRows = state.numTriangles % static_cast<size_t>(mpiSize);
    state.localRows = baseRows + (static_cast<size_t>(mpiRank) < extraRows ? 1 : 0);
    state.rowStart = baseRows * static_cast<size_t>(mpiRank) +
                     std::min(static_cast<size_t>(mpiRank), extraRows);
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    state.rho.resize(state.numTriangles, reflectivity);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();
    }

    // Kij and Tau are row-distributed across MPI ranks.  The time histories
    // are replicated because every receiver needs every earlier emitter row.
    state.kij.resize(state.localRows * state.numTriangles, ZERO);
    state.tau.resize(state.localRows * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, CudaContext& cuda) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");

    constexpr unsigned int threads = 256;
    const size_t pairCount = state.localRows * state.numTriangles;
    if (pairCount != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            (pairCount + threads - 1) / threads);
        computeFormFactorsKernel<<<blocks, threads>>>(
            state.rowStart, state.localRows, state.numTriangles,
            cuda.deviceTriangles, cuda.deviceNodes, cuda.deviceLeafTriangles,
            static_cast<int>(cuda.hostNodes.size()), cuda.deviceKij);
        cuda.synchronize("computeFormFactorsKernel");
        cuda.check(cudaMemcpy(state.kij.data(), cuda.deviceKij,
                              pairCount * sizeof(val_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(Kij)");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (state.mpiRank == 0) {
        printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
    }
}

void computeTimeDelays(SimulationState& state, CudaContext& cuda) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    constexpr unsigned int threads = 256;
    const size_t pairCount = state.localRows * state.numTriangles;
    if (pairCount != 0) {
        const unsigned int blocks = static_cast<unsigned int>(
            (pairCount + threads - 1) / threads);
        computeTimeDelaysKernel<<<blocks, threads>>>(
            state.rowStart, state.localRows, state.numTriangles,
            cuda.deviceTriangles, cuda.deviceTau);
        cuda.synchronize("computeTimeDelaysKernel");
        cuda.check(cudaMemcpy(state.tau.data(), cuda.deviceTau,
                              pairCount * sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(Tau)");
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void buildMpiRowLayout(size_t numTriangles, int mpiSize,
                       std::vector<int>& counts, std::vector<int>& displacements) {
    counts.resize(static_cast<size_t>(mpiSize));
    displacements.resize(static_cast<size_t>(mpiSize));
    const size_t baseRows = numTriangles / static_cast<size_t>(mpiSize);
    const size_t extraRows = numTriangles % static_cast<size_t>(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        const size_t rows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
        const size_t start = baseRows * static_cast<size_t>(rank) +
                             std::min(static_cast<size_t>(rank), extraRows);
        counts[static_cast<size_t>(rank)] = static_cast<int>(rows);
        displacements[static_cast<size_t>(rank)] = static_cast<int>(start);
    }
}

void runSimulation(SimulationState& state, CudaContext& cuda) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    std::vector<val_t> localStep(state.localRows, ZERO);
    std::vector<val_t> globalStep(state.numTriangles, ZERO);
    std::vector<int> counts;
    std::vector<int> displacements;
    buildMpiRowLayout(state.numTriangles, state.mpiSize, counts, displacements);

    constexpr unsigned int threads = 256;
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        if (state.localRows != 0) {
            propagateKernel<<<static_cast<unsigned int>(state.localRows), threads,
                              threads * sizeof(val_t)>>>(
                timestep, state.rowStart, state.localRows, state.numTriangles,
                state.numTimesteps, state.sourceIndex, cuda.deviceKij,
                cuda.deviceTau, cuda.deviceAreas, cuda.deviceRho, cuda.deviceRadB,
                cuda.deviceStep);
            cuda.synchronize("propagateKernel");
            cuda.check(cudaMemcpy(localStep.data(), cuda.deviceStep,
                                  state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy(radiosity step)");
        }

        MPI_Allgatherv(localStep.empty() ? nullptr : localStep.data(),
                       static_cast<int>(state.localRows), MPI_FLOAT,
                       globalStep.data(), counts.data(), displacements.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        std::copy(globalStep.begin(), globalStep.end(),
                  state.radB.begin() + state.idxTN(timestep, 0));
        cuda.check(cudaMemcpy(cuda.deviceRadB + state.idxTN(timestep, 0),
                              globalStep.data(), state.numTriangles * sizeof(val_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(radiosity history)");

        if (state.mpiRank == 0 && ((timestep + 1) % 10 == 0 ||
                                   timestep + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, CudaContext& cuda) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    constexpr unsigned int threads = 256;
    if (state.localRows != 0) {
        distanceKernel<<<static_cast<unsigned int>(state.localRows), threads,
                         threads * sizeof(DistanceCandidate)>>>(
            state.rowStart, state.localRows, state.numTriangles, state.numTimesteps,
            state.sourceIndex, cuda.deviceRadB, cuda.deviceDistances);
        cuda.synchronize("distanceKernel");

        std::vector<val_t> localDistances(state.localRows, ZERO);
        cuda.check(cudaMemcpy(localDistances.data(), cuda.deviceDistances,
                              state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(distances)");
        std::vector<int> counts;
        std::vector<int> displacements;
        buildMpiRowLayout(state.numTriangles, state.mpiSize, counts, displacements);
        MPI_Allgatherv(localDistances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                       state.distances.data(), counts.data(), displacements.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
    } else {
        std::vector<int> counts;
        std::vector<int> displacements;
        buildMpiRowLayout(state.numTriangles, state.mpiSize, counts, displacements);
        MPI_Allgatherv(nullptr, 0, MPI_FLOAT, state.distances.data(), counts.data(),
                       displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    const int rank = state.mpiRank;

    // Check that distances are non-negative
    int localNegative = 0;
    int localNonFinite = 0;
    val_t localMinDist = std::numeric_limits<val_t>::max();
    val_t localMaxDist = std::numeric_limits<val_t>::lowest();
    val_t localSumDist = ZERO;
    int localNonZeroCount = 0;

    #pragma omp parallel for reduction(+:localNegative,localNonFinite,localNonZeroCount) \
        reduction(min:localMinDist) reduction(max:localMaxDist) reduction(+:localSumDist)
    for (long long index = 0; index < static_cast<long long>(state.localRows); ++index) {
        const size_t i = state.rowStart + static_cast<size_t>(index);
        val_t d = state.distances[i];
        if (d < 0) localNegative++;
        if (!std::isfinite(d)) localNonFinite++;
        localMinDist = std::min(localMinDist, d);
        localMaxDist = std::max(localMaxDist, d);
        localSumDist += d;
        if (d > EPSILON) localNonZeroCount++;
    }

    int negative = 0;
    int nonFinite = 0;
    int nonZeroCount = 0;
    val_t minDist = ZERO;
    val_t maxDist = ZERO;
    val_t sumDist = ZERO;
    MPI_Allreduce(&localNegative, &negative, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNonFinite, &nonFinite, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNonZeroCount, &nonZeroCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMinDist, &minDist, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMaxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localSumDist, &sumDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nValidation:\n");
        if (negative != 0) printf("  ERROR: Negative distances: %d\n", negative);
        if (nonFinite != 0) printf("  ERROR: Non-finite distances: %d\n", nonFinite);
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    }

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (rank == 0 && srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int localReceivedEnergy = 0;
    #pragma omp parallel for reduction(+:localReceivedEnergy)
    for (long long index = 0; index < static_cast<long long>(state.localRows); ++index) {
        const size_t i = state.rowStart + static_cast<size_t>(index);
        bool received = false;
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                received = true;
                break;
            }
        }
        if (received) localReceivedEnergy++;
    }
    int receivedEnergy = 0;
    MPI_Allreduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
    }

    if (receivedEnergy == 0) {
        if (rank == 0) printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int localNonZeroKij = 0;
    #pragma omp parallel for reduction(+:localNonZeroKij)
    for (long long index = 0; index < static_cast<long long>(state.kij.size()); ++index) {
        const size_t i = static_cast<size_t>(index);
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }
    int nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const size_t matrixElements = state.numTriangles * state.numTriangles;
    if (rank == 0) {
        printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
               nonZeroKij, matrixElements,
               100.0f * nonZeroKij / static_cast<val_t>(matrixElements));
    }

    if (nonZeroKij == 0) {
        if (rank == 0) printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (negative != 0 || nonFinite != 0) {
        return false;
    }

    if (rank == 0) printf("  Validation: PASSED\n");
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    bool parseError = false;
    // Parse command line arguments.  Every MPI rank parses the same argv so
    // all ranks enter the same collectives even when launched interactively.
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Comm_free(&localCommunicator);
            MPI_Finalize();
            return 0;
        } else {
            parseError = true;
        }
    }

    if (parseError) {
        if (mpiRank == 0) {
            printf("Unknown or incomplete command-line option\n");
            printUsage(argv[0]);
        }
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    int exitCode = 0;
    {
        if (mpiRank == 0) {
            printf("Room Response Simulation Benchmark\n");
            printf("===================================\n");
            printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
            printf("Timesteps: %d\n", timesteps);
            printf("Source triangle: %d\n", sourceIdx);
            printf("Reflectivity: %.2f\n", reflectivity);
            printf("MPI ranks: %d, OpenMP threads/rank: %d, local GPUs: %d\n",
                   mpiSize, omp_get_max_threads(), 1);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("\n");
        }

        SimulationState state;
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity,
                             mpiRank, mpiSize);

        int deviceCount = 0;
        const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
        if (deviceQuery != cudaSuccess || deviceCount == 0) {
            std::fprintf(stderr, "Rank %d requires at least one CUDA device: %s\n",
                         mpiRank, cudaGetErrorString(deviceQuery));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        CudaContext cuda(state, localRank % deviceCount);

        if (mpiRank == 0) printf("\n");

        auto timedPhase = [&](auto&& phase) -> long {
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            phase();
            MPI_Barrier(MPI_COMM_WORLD);
            const double localMilliseconds = (MPI_Wtime() - start) * 1000.0;
            double maximumMilliseconds = 0.0;
            MPI_Reduce(&localMilliseconds, &maximumMilliseconds, 1, MPI_DOUBLE,
                       MPI_MAX, 0, MPI_COMM_WORLD);
            return mpiRank == 0 ? static_cast<long>(std::llround(maximumMilliseconds)) : 0L;
        };

        const long preDuration = timedPhase([&] {
            computeTimeDelays(state, cuda);
            computeFormFactors(state, cuda);
        });
        if (mpiRank == 0) {
            printf("Precomputation time: %ld ms\n\n", preDuration);
        }

        const long simDuration = timedPhase([&] { runSimulation(state, cuda); });
        if (mpiRank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

        const long distDuration = timedPhase([&] { computeDistances(state, cuda); });
        if (mpiRank == 0) printf("Distance computation time: %ld ms\n\n", distDuration);

        if (mpiRank == 0) {
            const long totalTime = preDuration + simDuration + distDuration;
            printf("Total computation time: %ld ms\n", totalTime);

            const size_t n = state.numTriangles;
            const size_t t = state.numTimesteps;
            const double kijOps = static_cast<double>(n * n);
            const double simOps = static_cast<double>(n * n * t);
            const double distOps = static_cast<double>(n * t * t);

            printf("\nPerformance:\n");
            printf("  Triangles: %zu\n", n);
            printf("  Timesteps: %zu\n", t);
            printf("  Form factor computations: %.2e\n", kijOps);
            printf("  Simulation operations: %.2e\n", simOps);
            printf("  Distance computations: %.2e\n", distOps);
            printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

            const size_t memKij = n * n * sizeof(val_t);
            const size_t memTau = n * n * sizeof(int);
            const size_t memRad = 2 * t * n * sizeof(val_t);
            const size_t totalMem = memKij + memTau + memRad;
            printf("  Memory usage: %.2f MB logical (Kij/Tau rows are MPI-distributed)\n",
                   totalMem / (1024.0 * 1024.0));

            const uint64_t resultHash = computeHash(state);
            printf("  Result hash: %016lX\n", resultHash);
            printf("\n");

            if (printResults) {
                std::vector<double> distData(state.distances.begin(), state.distances.end());
                print_results(distData, "Distances");
            }
        }

        if (validate) {
            const bool valid = validateResults(state);
            int localFailure = valid ? 0 : 1;
            int globalFailure = 0;
            MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
            exitCode = globalFailure;
        }
    }

    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return exitCode;
}

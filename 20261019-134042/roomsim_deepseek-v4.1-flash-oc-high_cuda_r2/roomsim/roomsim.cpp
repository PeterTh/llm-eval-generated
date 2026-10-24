/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallelized implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * All numerically intensive phases (time delays, form factors, wave propagation,
 * and cross-correlation) run on the GPU.
 */

#include <algorithm>
#include <array>
#include <cfloat>
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

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

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
    HD val_t norm() const {
#ifdef __CUDA_ARCH__
        return sqrtf(squaredNorm());
#else
        return std::sqrt(squaredNorm());
#endif
    }
    HD Vec3 normalized() const {
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
// Flattened octree representation for the GPU
// ============================================================================

struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

void flattenOctree(const Octree& node, std::vector<FlatNode>& nodes,
                   std::vector<int>& triIndices) {
    int self = static_cast<int>(nodes.size());

    FlatNode flat;
    flat.center = node.center;
    flat.halfExtent = node.halfExtent;
    for (int i = 0; i < 8; ++i) flat.child[i] = -1;
    flat.triStart = 0;
    flat.triCount = 0;
    nodes.push_back(flat);

    for (int i = 0; i < 8; ++i) {
        if (node.children[i]) {
            int childIndex = static_cast<int>(nodes.size());
            flattenOctree(*node.children[i], nodes, triIndices);
            nodes[self].child[i] = childIndex;
        }
    }

    if (!node.triangleIndices.empty()) {
        nodes[self].triStart = static_cast<int>(triIndices.size());
        nodes[self].triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            triIndices.push_back(static_cast<int>(idx));
        }
    }
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

// MT19937 with an externally visible state.  The C++ standard engine state is
// not accessible, so this compatible implementation is used to snapshot the
// generator at the beginning of every row.  The row states are then used by the
// GPU to regenerate the exact same random sequence in parallel.
struct MTState {
    uint32_t x[624];
    uint32_t p;
};

class MT19937 {
    MTState s;
public:
    explicit MT19937(uint32_t seed = 5489) {
        s.x[0] = seed;
        for (uint32_t i = 1; i < 624; ++i) {
            s.x[i] = 1812433253u * (s.x[i - 1] ^ (s.x[i - 1] >> 30)) + i;
        }
        s.p = 624;
    }

    const MTState& state() const { return s; }

    void twist() {
        uint32_t* x = s.x;
        for (uint32_t i = 0; i < 227; ++i) {
            uint32_t y = (x[i] & 0x80000000u) | (x[i + 1] & 0x7fffffffu);
            x[i] = x[i + 397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        for (uint32_t i = 227; i < 623; ++i) {
            uint32_t y = (x[i] & 0x80000000u) | (x[i + 1] & 0x7fffffffu);
            x[i] = x[i - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        uint32_t y = (x[623] & 0x80000000u) | (x[0] & 0x7fffffffu);
        x[623] = x[396] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        s.p = 0;
    }

    uint32_t next() {
        if (s.p >= 624) twist();
        uint32_t y = s.x[s.p++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    void discard(uint64_t z) {
        while (z > static_cast<uint64_t>(624 - s.p)) {
            z -= static_cast<uint64_t>(624 - s.p);
            twist();
        }
        s.p += static_cast<uint32_t>(z);
    }
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
// CUDA device helpers and kernels
// ============================================================================

__device__ __forceinline__ val_t deviceAbs(val_t v) {
    return v < 0.0f ? -v : v;
}

__device__ __forceinline__ val_t deviceMax0(val_t v) {
    return v > 0.0f ? v : 0.0f;
}

__device__ __forceinline__ val_t deviceMin(val_t a, val_t b) {
    return b < a ? b : a;
}

__device__ __forceinline__ val_t rayTriangleIntersectDevice(
    const Vec3& orig, const Vec3& dir,
    const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (deviceAbs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

__device__ __forceinline__ bool rayIntersectsBoxDevice(
    const FlatNode* __restrict__ nodes, int nodeIndex,
    const Vec3& p1, const Vec3& p2) {
    const FlatNode& node = nodes[nodeIndex];
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {deviceAbs(d.x), deviceAbs(d.y), deviceAbs(d.z)};

    if (deviceAbs(c.x) > node.halfExtent.x + ad.x) return false;
    if (deviceAbs(c.y) > node.halfExtent.y + ad.y) return false;
    if (deviceAbs(c.z) > node.halfExtent.z + ad.z) return false;

    if (deviceAbs(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (deviceAbs(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (deviceAbs(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ bool isRayBlockedDevice(
    const Vec3& from, const Vec3& to,
    const Triangle* __restrict__ triangles,
    const FlatNode* __restrict__ nodes,
    const int* __restrict__ triIndices,
    size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nodeIndex = stack[--sp];
        const FlatNode& node = nodes[nodeIndex];

        if (node.triCount > 0) {
            int start = node.triStart;
            int count = node.triCount;
            for (int k = 0; k < count; ++k) {
                int idx = triIndices[start + k];
                if (static_cast<size_t>(idx) == srcTriIdx ||
                    static_cast<size_t>(idx) == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersectDevice(
                    from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int childIndex = node.child[c];
                if (childIndex >= 0 &&
                    rayIntersectsBoxDevice(nodes, childIndex, from, to)) {
                    if (sp < 64) stack[sp++] = childIndex;
                }
            }
        }
    }

    return false;
}

__device__ __forceinline__ Vec3 randomPointInTriangleDevice(
    const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ val_t cosPhiDevice(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return deviceMax0(v.dot(normal) / vNorm);
}

__device__ val_t computeKijDevice(
    uint32_t idxI, uint32_t idxJ,
    const Triangle* __restrict__ triangles,
    const FlatNode* __restrict__ nodes,
    const int* __restrict__ triIndices,
    const val_t* __restrict__ rngValues) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (triI._normal.dot(triJ._normal) > 0.99f) return ZERO;

    val_t kij = ZERO;
    uint32_t rngPos = 0;

    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t uI = rngValues[rngPos++];
        val_t vI = rngValues[rngPos++];
        val_t uJ = rngValues[rngPos++];
        val_t vJ = rngValues[rngPos++];

        Vec3 pI = randomPointInTriangleDevice(triI, uI, vI);
        Vec3 pJ = randomPointInTriangleDevice(triJ, uJ, vJ);

        if (isRayBlockedDevice(pI, pJ, triangles, nodes, triIndices, idxI, idxJ)) {
            continue;
        }

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhiDevice(v, triI._normal);
        val_t cosPhiJ = cosPhiDevice(-v, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

__device__ __forceinline__ void mtTwistDevice(MTState& s) {
    uint32_t* x = s.x;
    for (uint32_t i = 0; i < 227; ++i) {
        uint32_t y = (x[i] & 0x80000000u) | (x[i + 1] & 0x7fffffffu);
        x[i] = x[i + 397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    for (uint32_t i = 227; i < 623; ++i) {
        uint32_t y = (x[i] & 0x80000000u) | (x[i + 1] & 0x7fffffffu);
        x[i] = x[i - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    uint32_t y = (x[623] & 0x80000000u) | (x[0] & 0x7fffffffu);
    x[623] = x[396] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    s.p = 0;
}

__device__ __forceinline__ val_t mtNextFloatDevice(MTState& s) {
    if (s.p >= 624) mtTwistDevice(s);
    uint32_t y = s.x[s.p++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return static_cast<val_t>(y) / 4294967296.0f;
}

__device__ __forceinline__ void mtDiscardDevice(MTState& s, uint64_t z) {
    while (z > static_cast<uint64_t>(624u - s.p)) {
        z -= static_cast<uint64_t>(624u - s.p);
        mtTwistDevice(s);
    }
    s.p += static_cast<uint32_t>(z);
}

__global__ void generateRngKernel(
    const MTState* __restrict__ segmentStates,
    uint64_t firstSegment, uint64_t numSegments,
    uint64_t segmentLength,
    uint64_t batchStart, uint64_t batchEnd,
    val_t* __restrict__ rngValues) {
    uint64_t t = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= numSegments) return;

    uint64_t segmentIndex = firstSegment + t;
    uint64_t segmentStart = segmentIndex * segmentLength;

    MTState state = segmentStates[segmentIndex];

    uint64_t writeStart = segmentStart > batchStart ? segmentStart : batchStart;
    uint64_t segmentEnd = segmentStart + segmentLength;
    uint64_t writeEnd = segmentEnd < batchEnd ? segmentEnd : batchEnd;
    if (writeStart >= writeEnd) return;

    mtDiscardDevice(state, writeStart - segmentStart);

    for (uint64_t off = writeStart; off < writeEnd; ++off) {
        rngValues[off - batchStart] = mtNextFloatDevice(state);
    }
}

__global__ void timeDelaysKernel(
    const Triangle* __restrict__ triangles,
    int* __restrict__ tau, uint32_t n) {
    size_t total = static_cast<size_t>(n) * n;
    size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= total) return;

    uint32_t i = static_cast<uint32_t>(id / n);
    uint32_t j = static_cast<uint32_t>(id - static_cast<size_t>(i) * n);
    if (i == j) {
        tau[id] = 0;
        return;
    }

    Vec3 ci = (triangles[i].a + triangles[i].b + triangles[i].c) / 3.0f;
    Vec3 cj = (triangles[j].a + triangles[j].b + triangles[j].c) / 3.0f;
    val_t dist = (ci - cj).norm();
    tau[id] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void formFactorsKernel(
    const Triangle* __restrict__ triangles,
    const FlatNode* __restrict__ nodes,
    const int* __restrict__ triIndices,
    const val_t* __restrict__ rngValues,
    const uint32_t* __restrict__ pairRngOffset,
    val_t* __restrict__ kij,
    uint32_t n, uint32_t rowStart, uint32_t rows) {
    uint32_t total = rows * n;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint32_t r = tid / n;
    uint32_t j = tid - r * n;
    uint32_t i = rowStart + r;
    if (i == j) return;

    uint32_t offset = pairRngOffset[tid];
    val_t value = ZERO;
    if (offset != UINT32_MAX) {
        value = computeKijDevice(i, j, triangles, nodes, triIndices,
                                 rngValues + offset);
    }
    kij[static_cast<size_t>(i) * n + j] = value;
}

__global__ void simulationKernel(
    const val_t* __restrict__ radE,
    val_t* __restrict__ radB,
    const val_t* __restrict__ kij,
    const int* __restrict__ tau,
    const val_t* __restrict__ rho,
    const val_t* __restrict__ areas,
    uint32_t n, int t) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t sumB = ZERO;
    size_t row = static_cast<size_t>(i) * n;

    for (uint32_t j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tau[row + j];
        if (t < tauij) continue;

        val_t kijVal = kij[row + j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t - tauij);
        val_t radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        sumB += deviceMin(kijVal * areas[j], ONE) * radJ;
    }

    radB[static_cast<size_t>(t) * n + i] =
        rho[i] * sumB + radE[static_cast<size_t>(t) * n + i];
}

__global__ void distanceKernel(
    const val_t* __restrict__ radB,
    uint32_t n, uint32_t numTimesteps, uint32_t sourceIndex,
    val_t* __restrict__ distances) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (uint32_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;

        for (uint32_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countNonZeroKernel(
    const val_t* __restrict__ kij, size_t total,
    unsigned long long* __restrict__ counter) {
    unsigned int count = 0;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         id < total; id += stride) {
        if (kij[id] > EPSILON) ++count;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffffu, count, offset);
    }

    if ((threadIdx.x & 31u) == 0 && count > 0) {
        atomicAdd(counter, static_cast<unsigned long long>(count));
    }
}

// ============================================================================
// GPU resource management
// ============================================================================

struct GpuState {
    Triangle* d_triangles = nullptr;
    FlatNode* d_nodes = nullptr;
    int* d_triIndices = nullptr;
    int numNodes = 0;

    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

    val_t* d_rngValues = nullptr;
    uint32_t* d_pairRngOffset = nullptr;
    size_t rngCapacity = 0;
    size_t pairCapacity = 0;
};

void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
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

void uploadStaticState(SimulationState& state, GpuState& gpu) {
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    size_t nn = n * n;

    std::vector<FlatNode> nodes;
    std::vector<int> triIndices;
    nodes.reserve(2 * n + 64);
    flattenOctree(state.octree, nodes, triIndices);
    gpu.numNodes = static_cast<int>(nodes.size());

    checkCuda(cudaMalloc(&gpu.d_triangles, n * sizeof(Triangle)), "cudaMalloc triangles");
    checkCuda(cudaMalloc(&gpu.d_nodes, nodes.size() * sizeof(FlatNode)), "cudaMalloc nodes");
    checkCuda(cudaMalloc(&gpu.d_triIndices, triIndices.size() * sizeof(int)), "cudaMalloc triIndices");
    checkCuda(cudaMalloc(&gpu.d_areas, n * sizeof(val_t)), "cudaMalloc areas");
    checkCuda(cudaMalloc(&gpu.d_rho, n * sizeof(val_t)), "cudaMalloc rho");
    checkCuda(cudaMalloc(&gpu.d_kij, nn * sizeof(val_t)), "cudaMalloc kij");
    checkCuda(cudaMalloc(&gpu.d_tau, nn * sizeof(int)), "cudaMalloc tau");
    checkCuda(cudaMalloc(&gpu.d_radE, t * n * sizeof(val_t)), "cudaMalloc radE");
    checkCuda(cudaMalloc(&gpu.d_radB, t * n * sizeof(val_t)), "cudaMalloc radB");
    checkCuda(cudaMalloc(&gpu.d_distances, n * sizeof(val_t)), "cudaMalloc distances");

    checkCuda(cudaMemcpy(gpu.d_triangles, state.triangles.data(),
                         n * sizeof(Triangle), cudaMemcpyHostToDevice), "copy triangles");
    checkCuda(cudaMemcpy(gpu.d_nodes, nodes.data(),
                         nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice), "copy nodes");
    checkCuda(cudaMemcpy(gpu.d_triIndices, triIndices.data(),
                         triIndices.size() * sizeof(int), cudaMemcpyHostToDevice), "copy triIndices");
    checkCuda(cudaMemcpy(gpu.d_areas, state.areas.data(),
                         n * sizeof(val_t), cudaMemcpyHostToDevice), "copy areas");
    checkCuda(cudaMemcpy(gpu.d_rho, state.rho.data(),
                         n * sizeof(val_t), cudaMemcpyHostToDevice), "copy rho");
    checkCuda(cudaMemcpy(gpu.d_radE, state.radE.data(),
                         t * n * sizeof(val_t), cudaMemcpyHostToDevice), "copy radE");

    checkCuda(cudaMemset(gpu.d_kij, 0, nn * sizeof(val_t)), "memset kij");
    checkCuda(cudaMemset(gpu.d_tau, 0, nn * sizeof(int)), "memset tau");
    checkCuda(cudaMemset(gpu.d_radB, 0, t * n * sizeof(val_t)), "memset radB");
}

void releaseGpuState(GpuState& gpu) {
    cudaFree(gpu.d_triangles);
    cudaFree(gpu.d_nodes);
    cudaFree(gpu.d_triIndices);
    cudaFree(gpu.d_areas);
    cudaFree(gpu.d_rho);
    cudaFree(gpu.d_kij);
    cudaFree(gpu.d_tau);
    cudaFree(gpu.d_radE);
    cudaFree(gpu.d_radB);
    cudaFree(gpu.d_distances);
    cudaFree(gpu.d_rngValues);
    cudaFree(gpu.d_pairRngOffset);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, GpuState& gpu) {
    printf("Computing time delays (Tau)...\n");

    uint32_t n = static_cast<uint32_t>(state.numTriangles);
    size_t total = static_cast<size_t>(n) * n;
    int threads = 256;
    unsigned int blocks = static_cast<unsigned int>((total + threads - 1) / threads);

    timeDelaysKernel<<<blocks, threads>>>(gpu.d_triangles, gpu.d_tau, n);
    checkCuda(cudaGetLastError(), "timeDelaysKernel");
    checkCuda(cudaDeviceSynchronize(), "time delays sync");
}

void computeFormFactors(SimulationState& state, GpuState& gpu) {
    printf("Computing form factors (Kij)...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);

    // Number of random draws consumed by each row in the original loop order.
    std::vector<size_t> rowDraws(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3 ni = state.triangles[i].normal();
        size_t pairCount = 0;
        for (uint32_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (ni.dot(state.triangles[j].normal()) > 0.99f) continue;
            ++pairCount;
        }
        rowDraws[i] = pairCount * (2 * NUM_RAYS);
    }

    std::vector<uint64_t> rowBase(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) {
        rowBase[i + 1] = rowBase[i] + static_cast<uint64_t>(rowDraws[i]);
    }
    const uint64_t totalDraws = rowBase[n];

    // Split the global MT19937 sequence into many small segments.  A snapshot
    // of the generator is taken at the start of each segment, and each segment
    // is then generated independently by one GPU thread.  This preserves the
    // exact sequential random sequence while exposing a large amount of
    // parallelism.
    const uint64_t targetSegments = 32768;
    const uint64_t maxStateBytes = 256ull * 1024ull * 1024ull;
    const uint64_t maxSegments = maxStateBytes / sizeof(MTState);
    uint64_t segmentLength = totalDraws / targetSegments;
    if (segmentLength < 1024) segmentLength = 1024;
    uint64_t minSegmentLength =
        (maxSegments == 0) ? segmentLength : (totalDraws + maxSegments - 1) / maxSegments;
    if (segmentLength < minSegmentLength) segmentLength = minSegmentLength;
    if (segmentLength == 0) segmentLength = 1;

    uint64_t numSegments = (totalDraws + segmentLength - 1) / segmentLength;
    if (numSegments == 0) numSegments = 1;

    MT19937 rng(42);
    std::vector<MTState> segmentStates(numSegments);
    for (uint64_t s = 0; s < numSegments; ++s) {
        segmentStates[s] = rng.state();
        rng.discard(segmentLength);
    }

    MTState* d_segmentStates = nullptr;
    checkCuda(cudaMalloc(&d_segmentStates, numSegments * sizeof(MTState)),
              "cudaMalloc segmentStates");
    checkCuda(cudaMemcpy(d_segmentStates, segmentStates.data(),
                         numSegments * sizeof(MTState), cudaMemcpyHostToDevice),
              "copy segmentStates");

    const size_t MAX_RNG_FLOATS = 256u * 1024u * 1024u;
    const size_t MAX_PAIR_ENTRIES = 32u * 1024u * 1024u;

    size_t rowStart = 0;
    std::vector<uint32_t> hostPairOffsets;

    while (rowStart < static_cast<size_t>(n)) {
        size_t rows = 0;
        size_t rngTotal = 0;
        size_t mapTotal = 0;

        while (rowStart + rows < static_cast<size_t>(n)) {
            size_t addRng = rowDraws[rowStart + rows];
            size_t addMap = n;
            if (rows > 0 &&
                (rngTotal + addRng > MAX_RNG_FLOATS ||
                 mapTotal + addMap > MAX_PAIR_ENTRIES)) {
                break;
            }
            rngTotal += addRng;
            mapTotal += addMap;
            ++rows;
        }

        hostPairOffsets.assign(mapTotal, UINT32_MAX);

        size_t pos = 0;
        for (size_t r = 0; r < rows; ++r) {
            uint32_t i = static_cast<uint32_t>(rowStart + r);
            const Vec3 ni = state.triangles[i].normal();
            for (uint32_t j = 0; j < n; ++j) {
                if (i == j) continue;
                if (ni.dot(state.triangles[j].normal()) > 0.99f) continue;

                hostPairOffsets[r * n + j] = static_cast<uint32_t>(pos);
                pos += 2 * NUM_RAYS;
            }
        }

        size_t allocRng = rngTotal > 0 ? rngTotal : 1;
        if (allocRng > gpu.rngCapacity) {
            cudaFree(gpu.d_rngValues);
            checkCuda(cudaMalloc(&gpu.d_rngValues, allocRng * sizeof(val_t)),
                      "cudaMalloc rngValues");
            gpu.rngCapacity = allocRng;
        }
        if (mapTotal > gpu.pairCapacity) {
            cudaFree(gpu.d_pairRngOffset);
            checkCuda(cudaMalloc(&gpu.d_pairRngOffset, mapTotal * sizeof(uint32_t)),
                      "cudaMalloc pairRngOffset");
            gpu.pairCapacity = mapTotal;
        }

        checkCuda(cudaMemcpy(gpu.d_pairRngOffset, hostPairOffsets.data(),
                             mapTotal * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  "copy pairRngOffset");

        uint64_t batchStart = rowBase[rowStart];
        uint64_t batchEnd = batchStart + static_cast<uint64_t>(rngTotal);

        if (batchEnd > batchStart) {
            uint64_t firstSegment = batchStart / segmentLength;
            uint64_t lastSegment = (batchEnd + segmentLength - 1) / segmentLength;
            uint64_t segCount = lastSegment - firstSegment;

            int genThreads = 128;
            uint64_t genBlocks64 = (segCount + genThreads - 1) / genThreads;
            unsigned int genBlocks = static_cast<unsigned int>(genBlocks64);

            generateRngKernel<<<genBlocks, genThreads>>>(
                d_segmentStates, firstSegment, segCount, segmentLength,
                batchStart, batchEnd, gpu.d_rngValues);
            checkCuda(cudaGetLastError(), "generateRngKernel");
        }

        uint32_t totalEntries = static_cast<uint32_t>(mapTotal);
        int threads = 256;
        unsigned int blocks = (totalEntries + threads - 1) / threads;

        formFactorsKernel<<<blocks, threads>>>(
            gpu.d_triangles, gpu.d_nodes, gpu.d_triIndices,
            gpu.d_rngValues, gpu.d_pairRngOffset, gpu.d_kij,
            n, static_cast<uint32_t>(rowStart), static_cast<uint32_t>(rows));
        checkCuda(cudaGetLastError(), "formFactorsKernel");

        for (size_t r = 0; r < rows; ++r) {
            size_t done = rowStart + r + 1;
            if (done % 100 == 0 || done == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", done, state.numTriangles);
            }
        }

        rowStart += rows;
    }

    checkCuda(cudaDeviceSynchronize(), "form factors sync");
    cudaFree(d_segmentStates);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuState& gpu) {
    printf("Running wave propagation simulation...\n");

    uint32_t n = static_cast<uint32_t>(state.numTriangles);
    int threads = 256;
    unsigned int blocks = (n + threads - 1) / threads;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<blocks, threads>>>(
            gpu.d_radE, gpu.d_radB, gpu.d_kij, gpu.d_tau,
            gpu.d_rho, gpu.d_areas, n, static_cast<int>(t));
        checkCuda(cudaGetLastError(), "simulationKernel");

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    checkCuda(cudaDeviceSynchronize(), "simulation sync");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuState& gpu) {
    printf("Computing distances via cross-correlation...\n");

    uint32_t n = static_cast<uint32_t>(state.numTriangles);
    int threads = 256;
    unsigned int blocks = (n + threads - 1) / threads;

    distanceKernel<<<blocks, threads>>>(
        gpu.d_radB, n, static_cast<uint32_t>(state.numTimesteps),
        static_cast<uint32_t>(state.sourceIndex), gpu.d_distances);
    checkCuda(cudaGetLastError(), "distanceKernel");
    checkCuda(cudaDeviceSynchronize(), "distance sync");

    checkCuda(cudaMemcpy(state.distances.data(), gpu.d_distances,
                         n * sizeof(val_t), cudaMemcpyDeviceToHost),
              "copy distances");
}

unsigned long long countNonZeroKij(const GpuState& gpu, size_t total) {
    unsigned long long* d_counter = nullptr;
    checkCuda(cudaMalloc(&d_counter, sizeof(unsigned long long)),
              "cudaMalloc nonZero counter");
    checkCuda(cudaMemset(d_counter, 0, sizeof(unsigned long long)),
              "memset nonZero counter");

    int threads = 256;
    size_t needed = (total + threads - 1) / threads;
    unsigned int blocks = static_cast<unsigned int>(std::min<size_t>(needed, 65535));

    countNonZeroKernel<<<blocks, threads>>>(gpu.d_kij, total, d_counter);
    checkCuda(cudaGetLastError(), "countNonZeroKernel");
    checkCuda(cudaDeviceSynchronize(), "count non-zero sync");

    unsigned long long result = 0;
    checkCuda(cudaMemcpy(&result, d_counter, sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost), "copy nonZero counter");
    cudaFree(d_counter);
    return result;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
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
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * static_cast<double>(nonZeroKij) /
               static_cast<double>(state.numTriangles * state.numTriangles));

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

    GpuState gpu;
    uploadStaticState(state, gpu);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, gpu);
    computeFormFactors(state, gpu);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, gpu);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, gpu);

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
        checkCuda(cudaMemcpy(state.radB.data(), gpu.d_radB,
                             t * n * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "copy radB");
        unsigned long long nonZeroKij = countNonZeroKij(gpu, n * n);
        if (!validateResults(state, nonZeroKij)) {
            releaseGpuState(gpu);
            return 1;
        }
    }

    releaseGpuState(gpu);
    return 0;
}

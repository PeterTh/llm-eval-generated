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

#define HD __host__ __device__

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
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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

struct DeviceTriangle {
    Vec3 a, b, c;
    Vec3 normal;
    Vec3 center;
    val_t area;
};

struct DeviceNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triOffset;
    int triCount;
};

constexpr int SAMPLES_PER_PAIR = NUM_RAYS * 4;

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

    int flatten(std::vector<DeviceNode>& nodes, std::vector<int>& triIndices) const {
        int index = static_cast<int>(nodes.size());
        nodes.emplace_back();
        DeviceNode& node = nodes[index];
        node.center = center;
        node.halfExtent = halfExtent;
        for (int i = 0; i < 8; ++i) node.child[i] = -1;

        if (!triangleIndices.empty()) {
            node.triOffset = static_cast<int>(triIndices.size());
            node.triCount = static_cast<int>(triangleIndices.size());
            for (size_t idx : triangleIndices) {
                triIndices.push_back(static_cast<int>(idx));
            }
        } else {
            node.triOffset = -1;
            node.triCount = 0;
            for (int i = 0; i < 8; ++i) {
                if (children[i]) {
                    int childIndex = children[i]->flatten(nodes, triIndices);
                    nodes[index].child[i] = childIndex;
                }
            }
        }
        return index;
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
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed) {}
    val_t rand() {
        return static_cast<val_t>(rng()) * (1.0f / 4294967296.0f);
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
// CUDA Device Helpers and Kernels
// ============================================================================

HD Vec3 pointInTriangleDevice(const DeviceTriangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

HD bool rayIntersectsBoxDevice(const DeviceNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

HD val_t devRayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                 const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    constexpr val_t FLT_MAX_VAL = 3.402823466e+38f;

    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX_VAL;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX_VAL;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_VAL;

    return e2.dot(qvec) * invDet;
}

HD val_t devCosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

HD bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                           const DeviceNode* nodes, const int* triIndices,
                           const DeviceTriangle* triangles,
                           int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        const DeviceNode& node = nodes[stack[--stackSize]];

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triOffset + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DeviceTriangle& tri = triangles[idx];
                val_t dist = devRayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            int pending[8];
            int count = 0;
            for (int k = 0; k < 8; ++k) {
                int ci = node.child[k];
                if (ci >= 0 && rayIntersectsBoxDevice(nodes[ci], from, to)) {
                    pending[count++] = ci;
                }
            }
            for (int k = count - 1; k >= 0; --k) {
                stack[stackSize++] = pending[k];
            }
        }
    }

    return false;
}

__global__ void computeFormFactorsKernel(
        int numTriangles, int rowStart, int numRows,
        const DeviceTriangle* triangles,
        const DeviceNode* nodes,
        const int* triIndices,
        const float* samples,
        float* kij) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(numRows) * numTriangles;
    if (p >= total) return;

    int i = rowStart + static_cast<int>(p / numTriangles);
    int j = static_cast<int>(p % numTriangles);
    size_t out = static_cast<size_t>(i) * numTriangles + j;

    if (i == j) {
        kij[out] = ZERO;
        return;
    }

    const DeviceTriangle& triI = triangles[i];
    const DeviceTriangle& triJ = triangles[j];
    if (triI.normal.dot(triJ.normal) > 0.99f) {
        kij[out] = ZERO;
        return;
    }

    const float* s = samples + p * SAMPLES_PER_PAIR;
    val_t acc = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const float* rs = s + r * 4;
        Vec3 pI = pointInTriangleDevice(triI, rs[0], rs[1]);
        Vec3 pJ = pointInTriangleDevice(triJ, rs[2], rs[3]);

        if (isRayBlockedDevice(pI, pJ, nodes, triIndices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = devCosPhi(v, triI.normal);
        val_t cosPhiJ = devCosPhi(-v, triJ.normal);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        acc += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[out] = acc * INV_NUM_RAYS;
}

__global__ void runSimulationKernel(
        int numTriangles, int t,
        const float* kij,
        const int* tau,
        const float* areas,
        const float* rho,
        const float* radE,
        float* radB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t sumB = ZERO;
    size_t row = static_cast<size_t>(i) * numTriangles;

    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tau[row + j];
        if (t < tauij) continue;

        val_t kijVal = kij[row + j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t - tauij);
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        val_t coeff = kijVal * areas[j];
        if (coeff > ONE) coeff = ONE;
        sumB += coeff * radJ;
    }

    radB[static_cast<size_t>(t) * numTriangles + i] =
        rho[i] * sumB + radE[static_cast<size_t>(t) * numTriangles + i];
}

__global__ void computeDistancesKernel(
        int numTriangles, int numTimesteps, int sourceIndex,
        const float* radB, float* distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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

struct GpuContext {
    DeviceTriangle* dTriangles = nullptr;
    DeviceNode* dNodes = nullptr;
    int* dTriIndices = nullptr;
    float* dKij = nullptr;
    int* dTau = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;
    float* dDistances = nullptr;
    float* dSamples = nullptr;
    size_t sampleCapacity = 0;

    ~GpuContext() { release(); }

    void release() {
        cudaFree(dTriangles);
        cudaFree(dNodes);
        cudaFree(dTriIndices);
        cudaFree(dKij);
        cudaFree(dTau);
        cudaFree(dAreas);
        cudaFree(dRho);
        cudaFree(dRadE);
        cudaFree(dRadB);
        cudaFree(dDistances);
        cudaFree(dSamples);
        dTriangles = nullptr;
        dNodes = nullptr;
        dTriIndices = nullptr;
        dKij = nullptr;
        dTau = nullptr;
        dAreas = nullptr;
        dRho = nullptr;
        dRadE = nullptr;
        dRadB = nullptr;
        dDistances = nullptr;
        dSamples = nullptr;
        sampleCapacity = 0;
    }
};

inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
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
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void initializeGpu(SimulationState& state, GpuContext& gpu) {
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    std::vector<DeviceTriangle> hostTriangles(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& src = state.triangles[i];
        DeviceTriangle& dst = hostTriangles[i];
        dst.a = src.a;
        dst.b = src.b;
        dst.c = src.c;
        dst.normal = src.normal();
        dst.center = src.center();
        dst.area = state.areas[i];
    }

    std::vector<DeviceNode> hostNodes;
    std::vector<int> hostTriIndices;
    hostNodes.reserve(n);
    hostTriIndices.reserve(n);
    state.octree.flatten(hostNodes, hostTriIndices);

    cudaCheck(cudaMalloc(&gpu.dTriangles, n * sizeof(DeviceTriangle)), "triangle allocation");
    cudaCheck(cudaMemcpy(gpu.dTriangles, hostTriangles.data(),
                         n * sizeof(DeviceTriangle), cudaMemcpyHostToDevice), "triangle upload");

    cudaCheck(cudaMalloc(&gpu.dNodes, hostNodes.size() * sizeof(DeviceNode)), "node allocation");
    cudaCheck(cudaMemcpy(gpu.dNodes, hostNodes.data(),
                         hostNodes.size() * sizeof(DeviceNode), cudaMemcpyHostToDevice), "node upload");

    cudaCheck(cudaMalloc(&gpu.dTriIndices, hostTriIndices.size() * sizeof(int)), "index allocation");
    cudaCheck(cudaMemcpy(gpu.dTriIndices, hostTriIndices.data(),
                         hostTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice), "index upload");

    cudaCheck(cudaMalloc(&gpu.dKij, n * n * sizeof(float)), "Kij allocation");
    cudaCheck(cudaMalloc(&gpu.dTau, n * n * sizeof(int)), "Tau allocation");
    cudaCheck(cudaMemcpy(gpu.dTau, state.tau.data(), n * n * sizeof(int), cudaMemcpyHostToDevice), "Tau upload");

    size_t radSize = (t > 0 ? t : 1) * n;
    cudaCheck(cudaMalloc(&gpu.dAreas, n * sizeof(float)), "area allocation");
    cudaCheck(cudaMalloc(&gpu.dRho, n * sizeof(float)), "reflectivity allocation");
    cudaCheck(cudaMalloc(&gpu.dRadE, radSize * sizeof(float)), "emission allocation");
    cudaCheck(cudaMalloc(&gpu.dRadB, radSize * sizeof(float)), "radiosity allocation");
    cudaCheck(cudaMalloc(&gpu.dDistances, n * sizeof(float)), "distance allocation");

    cudaCheck(cudaMemcpy(gpu.dAreas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice), "area upload");
    cudaCheck(cudaMemcpy(gpu.dRho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice), "reflectivity upload");
    if (t > 0) {
        cudaCheck(cudaMemcpy(gpu.dRadE, state.radE.data(), radSize * sizeof(float), cudaMemcpyHostToDevice), "emission upload");
    }
    cudaCheck(cudaMemset(gpu.dRadB, 0, radSize * sizeof(float)), "radiosity clear");
}

void generateSampleChunk(const SimulationState& state, size_t rowStart, size_t rowEnd,
                         RandomGenerator& rng, float* buffer) {
    size_t n = state.numTriangles;
    for (size_t i = rowStart; i < rowEnd; ++i) {
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const Triangle& triJ = state.triangles[j];
            if (triI.normal().dot(triJ.normal()) > 0.99f) continue;

            float* samples = buffer + (((i - rowStart) * n + j) * SAMPLES_PER_PAIR);
            for (int r = 0; r < NUM_RAYS; ++r) {
                samples[r * 4 + 0] = rng.rand();
                samples[r * 4 + 1] = rng.rand();
                samples[r * 4 + 2] = rng.rand();
                samples[r * 4 + 3] = rng.rand();
            }
        }
    }
}

void computeFormFactors(SimulationState& state, GpuContext& gpu) {
    printf("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);

    size_t n = state.numTriangles;
    size_t bytesPerRow = n * SAMPLES_PER_PAIR * sizeof(float);
    constexpr size_t TARGET_SAMPLE_BYTES = 64ull * 1024ull * 1024ull;

    size_t chunkRows = TARGET_SAMPLE_BYTES / bytesPerRow;
    if (chunkRows < 1) chunkRows = 1;
    if (chunkRows > n) chunkRows = n;

    size_t capacity = chunkRows * n * SAMPLES_PER_PAIR;
    std::vector<float> hostSamples(capacity);

    if (gpu.sampleCapacity < capacity) {
        cudaFree(gpu.dSamples);
        gpu.dSamples = nullptr;
        cudaCheck(cudaMalloc(&gpu.dSamples, capacity * sizeof(float)), "sample allocation");
        gpu.sampleCapacity = capacity;
    }

    const int blockSize = 256;

    for (size_t rowStart = 0; rowStart < n; rowStart += chunkRows) {
        size_t rows = std::min(chunkRows, n - rowStart);
        size_t totalPairs = rows * n;

        generateSampleChunk(state, rowStart, rowStart + rows, rng, hostSamples.data());

        cudaCheck(cudaMemcpy(gpu.dSamples, hostSamples.data(),
                             totalPairs * SAMPLES_PER_PAIR * sizeof(float),
                             cudaMemcpyHostToDevice), "sample upload");

        int blocks = static_cast<int>((totalPairs + blockSize - 1) / blockSize);
        computeFormFactorsKernel<<<blocks, blockSize>>>(
            static_cast<int>(n), static_cast<int>(rowStart), static_cast<int>(rows),
            gpu.dTriangles, gpu.dNodes, gpu.dTriIndices, gpu.dSamples, gpu.dKij);
        cudaCheck(cudaGetLastError(), "form factor launch");

        for (size_t i = rowStart; i < rowStart + rows; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }

    cudaCheck(cudaDeviceSynchronize(), "form factor computation");
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuContext& gpu) {
    printf("Running wave propagation simulation...\n");

    int n = static_cast<int>(state.numTriangles);
    int timesteps = static_cast<int>(state.numTimesteps);
    const int blockSize = 256;
    int blocks = (n + blockSize - 1) / blockSize;

    for (int t = 0; t < timesteps; ++t) {
        runSimulationKernel<<<blocks, blockSize>>>(
            n, t, gpu.dKij, gpu.dTau, gpu.dAreas, gpu.dRho, gpu.dRadE, gpu.dRadB);
        cudaCheck(cudaGetLastError(), "simulation launch");

        if ((t + 1) % 10 == 0 || t + 1 == timesteps) {
            printf("  Timestep %d/%d\n", t + 1, timesteps);
        }
    }

    cudaCheck(cudaDeviceSynchronize(), "wave propagation");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuContext& gpu) {
    printf("Computing distances via cross-correlation...\n");

    int n = static_cast<int>(state.numTriangles);
    int timesteps = static_cast<int>(state.numTimesteps);
    const int blockSize = 256;
    int blocks = (n + blockSize - 1) / blockSize;

    computeDistancesKernel<<<blocks, blockSize>>>(
        n, timesteps, static_cast<int>(state.sourceIndex), gpu.dRadB, gpu.dDistances);
    cudaCheck(cudaGetLastError(), "distance launch");

    cudaCheck(cudaMemcpy(state.distances.data(), gpu.dDistances,
                         state.numTriangles * sizeof(float), cudaMemcpyDeviceToHost),
              "distance download");
    cudaCheck(cudaDeviceSynchronize(), "distance computation");
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

    if (std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        setenv("CUDA_VISIBLE_DEVICES", "0", 1);
    }
    cudaCheck(cudaFree(nullptr), "CUDA initialization");

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

    GpuContext gpu;

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    initializeGpu(state, gpu);
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
        state.kij.resize(n * n);
        cudaCheck(cudaMemcpy(state.kij.data(), gpu.dKij, n * n * sizeof(float),
                             cudaMemcpyDeviceToHost), "Kij download");
        if (t > 0) {
            cudaCheck(cudaMemcpy(state.radB.data(), gpu.dRadB, t * n * sizeof(float),
                                 cudaMemcpyDeviceToHost), "radiosity download");
        }
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}

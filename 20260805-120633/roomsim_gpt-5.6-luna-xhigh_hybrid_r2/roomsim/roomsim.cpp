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

#define RS_CUDA_CHECK(call) do { \
    cudaError_t rsError = (call); \
    if (rsError != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(rsError)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(rsError)); \
    } \
} while (false)

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

// Flat, trivially-copyable representations used by the CUDA kernels.  The
// host-side Triangle/Octree types intentionally remain unchanged so mesh
// generation and the original geometric semantics are retained.
struct GpuTriangle {
    val_t ax, ay, az;
    val_t bx, by, bz;
    val_t cx, cy, cz;
    val_t nx, ny, nz;
};

struct GpuOctreeNode {
    val_t cx, cy, cz;
    val_t hx, hy, hz;
    int children[8];
    int triangleStart;
    int triangleCount;
};

struct FlatOctree {
    std::vector<GpuOctreeNode> nodes;
    std::vector<int> triangleIndices;
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

// Convert the pointer-rich CPU octree to a compact breadth-independent layout
// that can be traversed by every CUDA thread without recursion or pointers.
int flattenOctreeNode(const Octree& node, FlatOctree& flat) {
    const int nodeIndex = static_cast<int>(flat.nodes.size());
    GpuOctreeNode flatNode{};
    flatNode.cx = node.center.x;
    flatNode.cy = node.center.y;
    flatNode.cz = node.center.z;
    flatNode.hx = node.halfExtent.x;
    flatNode.hy = node.halfExtent.y;
    flatNode.hz = node.halfExtent.z;
    flatNode.triangleStart = -1;
    flatNode.triangleCount = 0;
    for (int i = 0; i < 8; ++i) flatNode.children[i] = -1;
    flat.nodes.push_back(flatNode);

    if (!node.triangleIndices.empty()) {
        flat.nodes[nodeIndex].triangleStart = static_cast<int>(flat.triangleIndices.size());
        flat.nodes[nodeIndex].triangleCount = static_cast<int>(node.triangleIndices.size());
        for (size_t triangleIndex : node.triangleIndices) {
            flat.triangleIndices.push_back(static_cast<int>(triangleIndex));
        }
        return nodeIndex;
    }

    for (int i = 0; i < 8; ++i) {
        if (node.children[i]) {
            flat.nodes[nodeIndex].children[i] = flattenOctreeNode(*node.children[i], flat);
        }
    }
    return nodeIndex;
}

FlatOctree flattenOctree(const Octree& octree) {
    FlatOctree flat;
    flat.nodes.reserve(1024);
    flat.triangleIndices.reserve(octree.allTriangles ? octree.allTriangles->size() : 0);
    if (octree.allTriangles && !octree.allTriangles->empty()) {
        flattenOctreeNode(octree, flat);
    }
    return flat;
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
// CUDA Kernels
// ============================================================================

__device__ __forceinline__ uint32_t mixRandom(uint64_t value) {
    uint32_t x = static_cast<uint32_t>(value) ^ static_cast<uint32_t>(value >> 32);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ val_t pairRandom(uint64_t pairKey, uint32_t draw) {
    // A counter-based stream makes every (i,j) independent, deterministic,
    // and reproducible regardless of MPI rank or OpenMP scheduling.
    const uint64_t counter = pairKey * 0x9e3779b97f4a7c15ULL +
                             static_cast<uint64_t>(draw) * 0xd1b54a32d192ed03ULL + 42ULL;
    return static_cast<val_t>(mixRandom(counter)) * (1.0f / 4294967296.0f);
}

__device__ __forceinline__ float3 randomPointOnTriangle(const GpuTriangle& tri,
                                                         uint64_t pairKey,
                                                         uint32_t& draw) {
    val_t u = pairRandom(pairKey, draw++);
    val_t v = pairRandom(pairKey, draw++);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    const float3 a = make_float3(tri.ax, tri.ay, tri.az);
    const float3 ab = make_float3(tri.bx - tri.ax, tri.by - tri.ay, tri.bz - tri.az);
    const float3 ac = make_float3(tri.cx - tri.ax, tri.cy - tri.ay, tri.cz - tri.az);
    return make_float3(a.x + ab.x * u + ac.x * v,
                       a.y + ab.y * u + ac.y * v,
                       a.z + ab.z * u + ac.z * v);
}

__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float squaredNorm3(const float3& a) {
    return dot3(a, a);
}

__device__ __forceinline__ float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float rayTriangleIntersectDevice(const float3& orig,
                                                             const float3& dir,
                                                             const GpuTriangle& tri) {
    const float3 v0 = make_float3(tri.ax, tri.ay, tri.az);
    const float3 v1 = make_float3(tri.bx, tri.by, tri.bz);
    const float3 v2 = make_float3(tri.cx, tri.cy, tri.cz);
    const float3 e1 = sub3(v1, v0);
    const float3 e2 = sub3(v2, v0);
    const float3 pvec = cross3(dir, e2);
    const val_t det = dot3(e1, pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    const val_t invDet = ONE / det;
    const float3 tvec = sub3(orig, v0);
    const val_t u = dot3(tvec, pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;

    const float3 qvec = cross3(tvec, e1);
    const val_t v = dot3(dir, qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return dot3(e2, qvec) * invDet;
}

__device__ __forceinline__ bool rayIntersectsBoxDevice(const float3& p1,
                                                       const float3& p2,
                                                       const GpuOctreeNode& node) {
    const float3 d = make_float3((p2.x - p1.x) * 0.5f,
                                 (p2.y - p1.y) * 0.5f,
                                 (p2.z - p1.z) * 0.5f);
    const float3 c = make_float3(p1.x + d.x - node.cx,
                                 p1.y + d.y - node.cy,
                                 p1.z + d.z - node.cz);
    const float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > node.hx + ad.x || fabsf(c.y) > node.hy + ad.y ||
        fabsf(c.z) > node.hz + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.hy * ad.z + node.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.hz * ad.x + node.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.hx * ad.y + node.hy * ad.x + EPSILON) return false;
    return true;
}

__device__ bool rayBlockedDevice(const float3& from,
                                 const float3& to,
                                 const GpuTriangle* triangles,
                                 const GpuOctreeNode* nodes,
                                 const int* leafTriangles,
                                 int sourceTriangle,
                                 int destinationTriangle) {
    const float3 direction = sub3(to, from);
    const val_t rayLength = sqrtf(squaredNorm3(direction));
    if (rayLength < EPSILON) return true;
    const float3 directionNormal = make_float3(direction.x / rayLength,
                                               direction.y / rayLength,
                                               direction.z / rayLength);

    // The CPU octree visits children in index order.  Pushing in reverse
    // order preserves that traversal and its early-out behavior.
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = 0;
    while (stackSize > 0) {
        const GpuOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount > 0) {
            for (int k = 0; k < node.triangleCount; ++k) {
                const int triangleIndex = leafTriangles[node.triangleStart + k];
                if (triangleIndex == sourceTriangle || triangleIndex == destinationTriangle) continue;
                const val_t distance = rayTriangleIntersectDevice(
                    from, directionNormal, triangles[triangleIndex]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }
        for (int child = 7; child >= 0; --child) {
            const int childIndex = node.children[child];
            if (childIndex >= 0 && rayIntersectsBoxDevice(from, to, nodes[childIndex])) {
                if (stackSize < 64) stack[stackSize++] = childIndex;
            }
        }
    }
    return false;
}

__global__ void computeInteractionsKernel(const GpuTriangle* triangles,
                                          const GpuOctreeNode* nodes,
                                          const int* leafTriangles,
                                          int numTriangles,
                                          int rowStart,
                                          int localRows,
                                          val_t* localKij,
                                          int* localTau) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = static_cast<size_t>(localRows) * numTriangles;
    if (pair >= pairCount) return;

    const int localI = static_cast<int>(pair / numTriangles);
    const int j = static_cast<int>(pair % numTriangles);
    const int i = rowStart + localI;
    const GpuTriangle& triI = triangles[i];
    const GpuTriangle& triJ = triangles[j];
    const float3 normalI = make_float3(triI.nx, triI.ny, triI.nz);
    const float3 normalJ = make_float3(triJ.nx, triJ.ny, triJ.nz);

    if (i == j) {
        localKij[pair] = ZERO;
        localTau[pair] = 0;
        return;
    }

    const float3 centerI = make_float3((triI.ax + triI.bx + triI.cx) / 3.0f,
                                       (triI.ay + triI.by + triI.cy) / 3.0f,
                                       (triI.az + triI.bz + triI.cz) / 3.0f);
    const float3 centerJ = make_float3((triJ.ax + triJ.bx + triJ.cx) / 3.0f,
                                       (triJ.ay + triJ.by + triJ.cy) / 3.0f,
                                       (triJ.az + triJ.bz + triJ.cz) / 3.0f);
    const float3 centerDelta = sub3(centerI, centerJ);
    localTau[pair] = static_cast<int>(ceilf(sqrtf(squaredNorm3(centerDelta)) * INV_WAVE_SPEED));

    if (dot3(normalI, normalJ) > 0.99f) {
        localKij[pair] = ZERO;
        return;
    }

    val_t kij = ZERO;
    const uint64_t pairKey = static_cast<uint64_t>(i) * static_cast<uint64_t>(numTriangles) + j;
    uint32_t draw = 0;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const float3 pI = randomPointOnTriangle(triI, pairKey, draw);
        const float3 pJ = randomPointOnTriangle(triJ, pairKey, draw);
        if (rayBlockedDevice(pI, pJ, triangles, nodes, leafTriangles, i, j)) continue;

        const float3 v = sub3(pJ, pI);
        const val_t distanceSquared = squaredNorm3(v);
        if (distanceSquared < EPSILON) continue;
        const val_t distance = sqrtf(distanceSquared);
        const val_t cosPhiI = fmaxf(ZERO, dot3(v, normalI) / distance);
        const float3 reverseV = make_float3(-v.x, -v.y, -v.z);
        const val_t cosPhiJ = fmaxf(ZERO, dot3(reverseV, normalJ) / distance);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;
        kij += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
    }
    localKij[pair] = kij * INV_NUM_RAYS;
}

__global__ void countPositiveKernel(const val_t* values, size_t count,
                                    unsigned long long* positiveCount) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && values[index] > EPSILON) atomicAdd(positiveCount, 1ULL);
}

__global__ void propagateKernel(const val_t* localKij,
                                const int* localTau,
                                const val_t* areas,
                                const val_t* rho,
                                const val_t* radiosityHistory,
                                val_t* localRadiosity,
                                int rowStart,
                                int localRows,
                                int numTriangles,
                                int timestep,
                                int timeOff,
                                int sourceIndex) {
    const int localI = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localRows) return;

    val_t sumB = ZERO;
    const size_t rowOffset = static_cast<size_t>(localI) * numTriangles;
    for (int j = 0; j < numTriangles; ++j) {
        if (rowStart + localI == j) continue;
        const int tauij = localTau[rowOffset + j];
        if (timestep < tauij) continue;
        const val_t kij = localKij[rowOffset + j];
        if (kij <= ZERO) continue;
        const size_t sourceOffset = static_cast<size_t>(timestep - tauij) * numTriangles + j;
        const val_t radJ = radiosityHistory[sourceOffset];
        if (radJ <= ZERO) continue;
        const val_t weightedKij = fminf(kij * areas[j], ONE);
        sumB += weightedKij * radJ;
    }

    const int globalI = rowStart + localI;
    const val_t emission = (timestep < timeOff && globalI == sourceIndex) ? ONE : ZERO;
    localRadiosity[localI] = rho[globalI] * sumB + emission;
}

__global__ void distanceKernel(const val_t* radiosityHistory,
                               val_t* localDistances,
                               int rowStart,
                               int localRows,
                               int numTriangles,
                               int timesteps,
                               int sourceIndex) {
    const int localI = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localRows) return;
    const int triangle = rowStart + localI;
    val_t maxCorrelation = ZERO;
    int bestTimestep = 0;

    for (int lag = 0; lag < timesteps; ++lag) {
        val_t sum = ZERO;
        for (int time = lag; time < timesteps; ++time) {
            const val_t pB = radiosityHistory[static_cast<size_t>(time) * numTriangles + triangle];
            const val_t pS = radiosityHistory[
                static_cast<size_t>(time - lag) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorrelation) {
            maxCorrelation = sum;
            bestTimestep = lag;
        }
    }
    localDistances[localI] = WAVE_SPEED * static_cast<val_t>(bestTimestep);
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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source
    std::vector<val_t> localRadiosity; // Non-overlapping MPI send buffer

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;          // Device-friendly octree representation

    // MPI row decomposition.  Each rank owns a contiguous range of receiver
    // triangles; the distributed Kij/Tau matrices stay resident on that rank's
    // GPU instead of being replicated on every host and device.
    int mpiRank = 0;
    int mpiSize = 1;
    size_t localRowStart = 0;
    size_t localRows = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    size_t nonZeroKij = 0;

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    GpuTriangle* dTriangles = nullptr;
    GpuOctreeNode* dOctreeNodes = nullptr;
    int* dLeafTriangles = nullptr;
    val_t* dKij = nullptr;
    int* dTau = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadiosityHistory = nullptr;
    val_t* dLocalRadiosity = nullptr;
    val_t* dLocalDistances = nullptr;
    unsigned long long* dPositiveKij = nullptr;
};

// ============================================================================
// Initialization
// ============================================================================

void configureRowDecomposition(SimulationState& state) {
    const size_t baseRows = state.numTriangles / static_cast<size_t>(state.mpiSize);
    const size_t remainder = state.numTriangles % static_cast<size_t>(state.mpiSize);
    state.rowCounts.resize(state.mpiSize);
    state.rowDisplacements.resize(state.mpiSize);
    int displacement = 0;
    for (int rank = 0; rank < state.mpiSize; ++rank) {
        const size_t rows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        state.rowCounts[rank] = static_cast<int>(rows);
        state.rowDisplacements[rank] = displacement;
        displacement += state.rowCounts[rank];
    }
    state.localRowStart = static_cast<size_t>(state.rowDisplacements[state.mpiRank]);
    state.localRows = static_cast<size_t>(state.rowCounts[state.mpiRank]);
}

void setupGpu(SimulationState& state) {
    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.mpiRank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    RS_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Rank %d found no CUDA accelerator\n", state.mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    RS_CUDA_CHECK(cudaSetDevice(device));
    MPI_Comm_free(&localCommunicator);

    std::vector<GpuTriangle> gpuTriangles(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const Triangle& tri = state.triangles[static_cast<size_t>(i)];
        GpuTriangle& gpuTri = gpuTriangles[static_cast<size_t>(i)];
        gpuTri.ax = tri.a.x; gpuTri.ay = tri.a.y; gpuTri.az = tri.a.z;
        gpuTri.bx = tri.b.x; gpuTri.by = tri.b.y; gpuTri.bz = tri.b.z;
        gpuTri.cx = tri.c.x; gpuTri.cy = tri.c.y; gpuTri.cz = tri.c.z;
        gpuTri.nx = tri.normal().x; gpuTri.ny = tri.normal().y; gpuTri.nz = tri.normal().z;
    }

    state.flatOctree = flattenOctree(state.octree);
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dTriangles),
                             gpuTriangles.size() * sizeof(GpuTriangle)));
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dOctreeNodes),
                             state.flatOctree.nodes.size() * sizeof(GpuOctreeNode)));
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dLeafTriangles),
                             state.flatOctree.triangleIndices.size() * sizeof(int)));
    RS_CUDA_CHECK(cudaMemcpy(state.dTriangles, gpuTriangles.data(),
                             gpuTriangles.size() * sizeof(GpuTriangle), cudaMemcpyHostToDevice));
    RS_CUDA_CHECK(cudaMemcpy(state.dOctreeNodes, state.flatOctree.nodes.data(),
                             state.flatOctree.nodes.size() * sizeof(GpuOctreeNode), cudaMemcpyHostToDevice));
    RS_CUDA_CHECK(cudaMemcpy(state.dLeafTriangles, state.flatOctree.triangleIndices.data(),
                             state.flatOctree.triangleIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    const size_t localPairCount = state.localRows * state.numTriangles;
    if (localPairCount > 0) {
        RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dKij),
                                 localPairCount * sizeof(val_t)));
        RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dTau),
                                 localPairCount * sizeof(int)));
        RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dLocalRadiosity),
                                 state.localRows * sizeof(val_t)));
        RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dLocalDistances),
                                 state.localRows * sizeof(val_t)));
    }
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dAreas),
                             state.numTriangles * sizeof(val_t)));
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dRho),
                             state.numTriangles * sizeof(val_t)));
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dRadiosityHistory),
                             state.numTimesteps * state.numTriangles * sizeof(val_t)));
    RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dPositiveKij),
                             sizeof(unsigned long long)));

    RS_CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(),
                             state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    RS_CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(),
                             state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    RS_CUDA_CHECK(cudaMemset(state.dRadiosityHistory, 0,
                             state.numTimesteps * state.numTriangles * sizeof(val_t)));

    if (state.mpiRank == 0) {
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               state.mpiSize, omp_get_max_threads(), deviceCount);
    }
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    configureRowDecomposition(state);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.rho[i] = reflectivity;
    }

    // Initialize matrices
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
    state.localRadiosity.resize(state.localRows, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    #pragma omp parallel for schedule(static)
    for (long long t = static_cast<long long>(timeOn); t < static_cast<long long>(timeOff); ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
    setupGpu(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeInteractions(SimulationState& state) {
    if (state.mpiRank == 0) {
        printf("Computing form factors (Kij) and time delays (Tau) on CUDA...\n");
    }
    const size_t pairCount = state.localRows * state.numTriangles;
    if (pairCount > 0) {
        constexpr int blockSize = 256;
        const int gridSize = static_cast<int>((pairCount + blockSize - 1) / blockSize);
        computeInteractionsKernel<<<gridSize, blockSize>>>(
            state.dTriangles, state.dOctreeNodes, state.dLeafTriangles,
            static_cast<int>(state.numTriangles), static_cast<int>(state.localRowStart),
            static_cast<int>(state.localRows), state.dKij, state.dTau);
        RS_CUDA_CHECK(cudaGetLastError());
        RS_CUDA_CHECK(cudaDeviceSynchronize());

        RS_CUDA_CHECK(cudaMemset(state.dPositiveKij, 0, sizeof(unsigned long long)));
        const int countGrid = static_cast<int>((pairCount + blockSize - 1) / blockSize);
        countPositiveKernel<<<countGrid, blockSize>>>(state.dKij, pairCount, state.dPositiveKij);
        RS_CUDA_CHECK(cudaGetLastError());
        RS_CUDA_CHECK(cudaDeviceSynchronize());
    }

    unsigned long long localNonZero = 0;
    if (state.localRows > 0) {
        RS_CUDA_CHECK(cudaMemcpy(&localNonZero, state.dPositiveKij,
                                 sizeof(localNonZero), cudaMemcpyDeviceToHost));
    }
    unsigned long long globalNonZero = 0;
    MPI_Allreduce(&localNonZero, &globalNonZero, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = static_cast<size_t>(globalNonZero);
    if (state.mpiRank == 0) {
        printf("  Distributed interaction matrix: %zu receiver rows/rank\n", state.localRows);
        printf("  Form factors and time delays ready (%zu non-zero Kij entries)\n",
               state.nonZeroKij);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation on CUDA...\n");

    constexpr int blockSize = 256;
    const int gridSize = static_cast<int>((state.localRows + blockSize - 1) / blockSize);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows > 0) {
            propagateKernel<<<gridSize, blockSize>>>(
                state.dKij, state.dTau, state.dAreas, state.dRho,
                state.dRadiosityHistory, state.dLocalRadiosity,
                static_cast<int>(state.localRowStart), static_cast<int>(state.localRows),
                static_cast<int>(state.numTriangles), static_cast<int>(t),
                static_cast<int>(state.numTimesteps / 2), static_cast<int>(state.sourceIndex));
            RS_CUDA_CHECK(cudaGetLastError());
            RS_CUDA_CHECK(cudaDeviceSynchronize());
            RS_CUDA_CHECK(cudaMemcpy(state.localRadiosity.data(), state.dLocalRadiosity,
                                     state.localRows * sizeof(val_t),
                                     cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(state.localRows > 0 ? state.localRadiosity.data() : nullptr,
                       static_cast<int>(state.localRows), MPI_FLOAT,
                       state.radB.data() + state.idxTN(t, 0), state.rowCounts.data(),
                       state.rowDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        RS_CUDA_CHECK(cudaMemcpy(state.dRadiosityHistory + state.idxTN(t, 0),
                                 state.radB.data() + state.idxTN(t, 0),
                                 state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    if (state.localRows > 0) {
        constexpr int blockSize = 256;
        const int gridSize = static_cast<int>((state.localRows + blockSize - 1) / blockSize);
        distanceKernel<<<gridSize, blockSize>>>(
            state.dRadiosityHistory, state.dLocalDistances,
            static_cast<int>(state.localRowStart), static_cast<int>(state.localRows),
            static_cast<int>(state.numTriangles), static_cast<int>(state.numTimesteps),
            static_cast<int>(state.sourceIndex));
        RS_CUDA_CHECK(cudaGetLastError());
        RS_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<val_t> localDistances(state.localRows);
        RS_CUDA_CHECK(cudaMemcpy(localDistances.data(), state.dLocalDistances,
                                 state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(localDistances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                       state.distances.data(), state.rowCounts.data(), state.rowDisplacements.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
    } else {
        MPI_Allgatherv(nullptr, 0, MPI_FLOAT, state.distances.data(),
                       state.rowCounts.data(), state.rowDisplacements.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
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

    // Check Kij matrix (counted on the owning GPUs during precomputation).
    const size_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
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

void releaseGpu(SimulationState& state) {
    RS_CUDA_CHECK(cudaFree(state.dPositiveKij));
    RS_CUDA_CHECK(cudaFree(state.dLocalDistances));
    RS_CUDA_CHECK(cudaFree(state.dLocalRadiosity));
    RS_CUDA_CHECK(cudaFree(state.dRadiosityHistory));
    RS_CUDA_CHECK(cudaFree(state.dRho));
    RS_CUDA_CHECK(cudaFree(state.dAreas));
    RS_CUDA_CHECK(cudaFree(state.dTau));
    RS_CUDA_CHECK(cudaFree(state.dKij));
    RS_CUDA_CHECK(cudaFree(state.dLeafTriangles));
    RS_CUDA_CHECK(cudaFree(state.dOctreeNodes));
    RS_CUDA_CHECK(cudaFree(state.dTriangles));
    state.dPositiveKij = nullptr;
    state.dLocalDistances = nullptr;
    state.dLocalRadiosity = nullptr;
    state.dRadiosityHistory = nullptr;
    state.dRho = nullptr;
    state.dAreas = nullptr;
    state.dTau = nullptr;
    state.dKij = nullptr;
    state.dLeafTriangles = nullptr;
    state.dOctreeNodes = nullptr;
    state.dTriangles = nullptr;
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
    int providedThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI thread support: %d, OpenMP max threads: %d\n",
               providedThreadLevel, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeInteractions(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    long localPreDuration = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count());
    long preDuration = 0;
    MPI_Reduce(&localPreDuration, &preDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    long localSimDuration = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count());
    long simDuration = 0;
    MPI_Reduce(&localSimDuration, &simDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    long localDistDuration = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count());
    long distDuration = 0;
    MPI_Reduce(&localDistDuration, &distDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    if (mpiRank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (mpiRank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    }

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    if (mpiRank == 0) {
        printf("  Distributed matrix memory (aggregate): %.2f MB\n", totalMem / (1024.0 * 1024.0));
    }

    // Hash
    uint64_t hash = computeHash(state);
    if (mpiRank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults && mpiRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int valid = 1;
    if (validate && mpiRank == 0) {
        valid = validateResults(state) ? 1 : 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    releaseGpu(state);
    MPI_Finalize();

    return valid ? 0 : 1;
}

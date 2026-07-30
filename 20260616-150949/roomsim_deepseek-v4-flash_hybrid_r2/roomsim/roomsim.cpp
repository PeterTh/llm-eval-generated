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

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// MPI Globals
// ============================================================================

static int mpi_rank = 0;
static int mpi_size = 1;

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

// Annotation helper for host+device code
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

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
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() : a(), b(), c(), _normal() {}
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
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
    auto minMax3 = [](val_t a, val_t b, val_t c, val_t& mn, val_t& mx) {
        mn = std::min({a, b, c});
        mx = std::max({a, b, c});
    };

    val_t minX, maxX, minY, maxY, minZ, maxZ;
    minMax3(v0.x, v1.x, v2.x, minX, maxX);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    minMax3(v0.y, v1.y, v2.y, minY, maxY);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    minMax3(v0.z, v1.z, v2.z, minZ, maxZ);
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
        val_t minP, maxP;
        minMax3(p0, p1, p2, minP, maxP);
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
// GPU Structures and Device Functions (for CUDA acceleration)
// ============================================================================

#ifdef __CUDACC__

// Compact octree node for GPU traversal
struct GPUOctreeNode {
    val_t minX, minY, minZ, maxX, maxY, maxZ;
    int children[8];
    int triStart;   // For leaf: start index in triangle indices array
    int triCount;   // For leaf: number of triangles
};

// Simple fast PRNG for GPU (LCG)
__device__ __forceinline__ float gpu_rand(unsigned long long& state) {
    state = state * 6364136223846793005ULL + 1;
    return (state >> 33) * (1.0f / 8388608.0f);
}

// Generate random point in a triangle (GPU version)
__device__ __forceinline__ Vec3 gpu_randomPointInTriangle(const Triangle& t, unsigned long long& rngState) {
    float u = gpu_rand(rngState);
    float v = gpu_rand(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// GPU version of ray-box intersection for octree traversal
__device__ __forceinline__ bool gpu_rayBoxIntersect(const Vec3& p1, const Vec3& p2,
                                                     val_t minX, val_t minY, val_t minZ,
                                                     val_t maxX, val_t maxY, val_t maxZ) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 center = {(minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f};
    Vec3 half = {(maxX - minX) * 0.5f, (maxY - minY) * 0.5f, (maxZ - minZ) * 0.5f};
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > half.x + ad.x) return false;
    if (fabsf(c.y) > half.y + ad.y) return false;
    if (fabsf(c.z) > half.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > half.y * ad.z + half.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > half.z * ad.x + half.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > half.x * ad.y + half.y * ad.x + EPSILON) return false;

    return true;
}

// GPU ray-triangle intersection (Möller-Trumbore)
__device__ __forceinline__ float gpu_rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                           const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 1e30f;

    float invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;

    Vec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    return e2.dot(qvec) * invDet;
}

// Check if a ray between two triangles is blocked (GPU octree traversal)
__device__ bool gpu_isRayBlocked(const Vec3& from, const Vec3& to,
                                  const GPUOctreeNode* nodes, const int* triIndices,
                                  const Triangle* triangles,
                                  int srcTriIdx, int dstTriIdx, int numNodes) {
    Vec3 dir = to - from;
    float rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Stack-based octree traversal
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;  // start at root

    while (sp > 0) {
        int nodeIdx = stack[--sp];
        if (nodeIdx < 0 || nodeIdx >= numNodes) continue;

        const GPUOctreeNode& node = nodes[nodeIdx];

        if (!gpu_rayBoxIntersect(from, to, node.minX, node.minY, node.minZ,
                                           node.maxX, node.maxY, node.maxZ))
            continue;

        if (node.triCount > 0) {
            // Leaf: test triangles
            for (int t = 0; t < node.triCount; ++t) {
                int idx = triIndices[node.triStart + t];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                float dist = gpu_rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            // Internal: push children
            for (int c = 7; c >= 0; --c) {
                if (node.children[c] >= 0) {
                    stack[sp++] = node.children[c];
                }
            }
        }
    }
    return false;
}

// GPU form factor computation for a single (i,j) pair
__device__ float gpu_computeKij(int i, int j,
                                 const Triangle* triangles,
                                 const GPUOctreeNode* nodes,
                                 const int* triIndices,
                                 int numNodes,
                                 unsigned long long& rngState) {
    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    // Cull triangles facing same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return 0.0f;

    float kij = 0.0f;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = gpu_randomPointInTriangle(triI, rngState);
        Vec3 pJ = gpu_randomPointInTriangle(triJ, rngState);

        if (gpu_isRayBlocked(pI, pJ, nodes, triIndices, triangles, i, j, numNodes))
            continue;

        Vec3 v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        float cosPhiI = v.dot(triI.normal()) / sqrtf(distSqr);
        if (cosPhiI < 0.0f) cosPhiI = 0.0f;
        float cosPhiJ = (-v).dot(triJ.normal()) / sqrtf(distSqr);
        if (cosPhiJ < 0.0f) cosPhiJ = 0.0f;

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// CUDA kernel: compute form factors for a block of rows
__global__ void formFactorKernel(const Triangle* triangles,
                                  const GPUOctreeNode* nodes,
                                  const int* triIndices,
                                  int numNodes,
                                  int numTriangles,
                                  int globalStartRow, int localNumRows,
                                  float* kijOutput, unsigned long long baseSeed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = localNumRows * numTriangles;
    if (tid >= totalPairs) return;

    int localRow = tid / numTriangles;
    int col = tid % numTriangles;
    int globalRow = globalStartRow + localRow;

    if (globalRow == col) {
        kijOutput[localRow * numTriangles + col] = 0.0f;
        return;
    }

    // Deterministic seed based on (i,j) pair for reproducibility
    unsigned long long rngState = baseSeed + (unsigned long long)(globalRow * numTriangles + col) * 6364136223846793005ULL;
    float val = gpu_computeKij(globalRow, col, triangles, nodes, triIndices, numNodes, rngState);
    kijOutput[localRow * numTriangles + col] = val;
}

// CUDA kernel: compute distances via cross-correlation
__global__ void distanceKernel(const float* radB, int numTriangles, int numTimesteps,
                                float waveSpeed, int globalStartRow, int localNumRows,
                                int sourceIndex, float* distOutput) {
    int localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= localNumRows) return;

    int globalRow = globalStartRow + localRow;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[tt * numTriangles + globalRow];
            float pS = radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distOutput[localRow] = waveSpeed * static_cast<float>(bestT);
}

#endif // __CUDACC__

// ============================================================================
// Host helper: flatten octree for GPU transfer
// ============================================================================

#ifdef __CUDACC__
struct FlatOctree {
    std::vector<GPUOctreeNode> nodes;
    std::vector<int> triIndices;
};

static void flattenNode(const Octree& node, FlatOctree& flat) {
    GPUOctreeNode gpuNode;
    gpuNode.minX = node.minBound.x;
    gpuNode.minY = node.minBound.y;
    gpuNode.minZ = node.minBound.z;
    gpuNode.maxX = node.maxBound.x;
    gpuNode.maxY = node.maxBound.y;
    gpuNode.maxZ = node.maxBound.z;

    int idx = flat.nodes.size();
    flat.nodes.push_back(gpuNode);

    if (!node.triangleIndices.empty()) {
        // Leaf node
        flat.nodes[idx].triStart = static_cast<int>(flat.triIndices.size());
        flat.nodes[idx].triCount = static_cast<int>(node.triangleIndices.size());
        flat.triIndices.insert(flat.triIndices.end(), node.triangleIndices.begin(), node.triangleIndices.end());
        for (int c = 0; c < 8; ++c) flat.nodes[idx].children[c] = -1;
    } else {
        flat.nodes[idx].triStart = -1;
        flat.nodes[idx].triCount = 0;
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                flat.nodes[idx].children[c] = static_cast<int>(flat.nodes.size());
                flattenNode(*node.children[c], flat);
            } else {
                flat.nodes[idx].children[c] = -1;
            }
        }
    }
}

static FlatOctree flattenOctree(const Octree& octree) {
    FlatOctree flat;
    if (octree.allTriangles && !octree.allTriangles->empty()) {
        flattenNode(octree, flat);
    }
    return flat;
}
#endif // __CUDACC__

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
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const size_t N = state.numTriangles;

    // Distribute rows across MPI ranks
    size_t rowsPerRank = (N + mpi_size - 1) / mpi_size;
    size_t startRow = mpi_rank * rowsPerRank;
    size_t endRow = std::min(startRow + rowsPerRank, N);
    size_t localRows = (startRow < N) ? (endRow - startRow) : 0;

    // Local Kij buffer for this rank's rows
    std::vector<val_t> localKij(localRows * N, ZERO);

    if (mpi_rank == 0) {
        printf("Computing form factors (Kij) with MPI+CUDA+OpenMP...\n");
        printf("  MPI ranks: %d, triangles: %zu, rows per rank: %zu\n", mpi_size, N, rowsPerRank);
    }

#ifdef __CUDACC__
    // CUDA path: launch GPU kernel for this rank's rows
    if (localRows > 0) {
        // Flatten octree for GPU
        FlatOctree flat = flattenOctree(state.octree);

        // Allocate GPU memory
        Triangle* d_triangles = nullptr;
        GPUOctreeNode* d_nodes = nullptr;
        int* d_triIndices = nullptr;
        float* d_kij = nullptr;

        cudaMalloc(&d_triangles, N * sizeof(Triangle));
        cudaMalloc(&d_nodes, flat.nodes.size() * sizeof(GPUOctreeNode));
        cudaMalloc(&d_triIndices, flat.triIndices.size() * sizeof(int));
        cudaMalloc(&d_kij, localRows * N * sizeof(float));

        cudaMemcpy(d_triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nodes, flat.nodes.data(), flat.nodes.size() * sizeof(GPUOctreeNode), cudaMemcpyHostToDevice);
        cudaMemcpy(d_triIndices, flat.triIndices.data(), flat.triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

        int totalPairs = static_cast<int>(localRows * N);
        int threadsPerBlock = 256;
        int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;
        numBlocks = std::min(numBlocks, 1 << 30);

        unsigned long long baseSeed = 42ULL;

        formFactorKernel<<<numBlocks, threadsPerBlock>>>(
            d_triangles, d_nodes, d_triIndices,
            static_cast<int>(flat.nodes.size()),
            static_cast<int>(N),
            static_cast<int>(startRow),
            static_cast<int>(localRows),
            d_kij, baseSeed
        );

        cudaDeviceSynchronize();

        cudaMemcpy(localKij.data(), d_kij, localRows * N * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_triangles);
        cudaFree(d_nodes);
        cudaFree(d_triIndices);
        cudaFree(d_kij);
    }

    if (mpi_rank == 0) printf("  GPU kernels complete\n");
#else
    // Fallback: OpenMP parallel CPU path
    #pragma omp parallel for schedule(dynamic)
    for (size_t localI = 0; localI < localRows; ++localI) {
        size_t i = startRow + localI;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            // Deterministic seed based on (i,j) for reproducibility
            uint32_t pairSeed = 42 + static_cast<uint32_t>(i * 2654435761U + j);
            RandomGenerator localRng(pairSeed);
            localKij[localI * N + j] = computeKij(i, j, state.triangles, state.octree, localRng);
        }
        if (mpi_rank == 0 && ((i + 1) % 100 == 0 || i + 1 == N)) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }
#endif

    // Allgather: each rank contributes its rows to the full Kij matrix
    std::vector<int> recvCounts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t rStart = r * rowsPerRank;
        size_t rEnd = std::min(rStart + rowsPerRank, N);
        size_t rRows = (rStart < N) ? (rEnd - rStart) : 0;
        recvCounts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rStart * N);
    }

    MPI_Allgatherv(localKij.data(), static_cast<int>(localRows * N), MPI_FLOAT,
                   state.kij.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

void computeTimeDelays(SimulationState& state) {
    const size_t N = state.numTriangles;

    if (mpi_rank == 0) {
        printf("Computing time delays (Tau) with MPI+OpenMP...\n");
    }

    // Distribute rows across MPI ranks
    size_t rowsPerRank = (N + mpi_size - 1) / mpi_size;
    size_t startRow = mpi_rank * rowsPerRank;
    size_t endRow = std::min(startRow + rowsPerRank, N);
    size_t localRows = (startRow < N) ? (endRow - startRow) : 0;

    // Local Tau buffer
    std::vector<int> localTau(localRows * N, 0);

    #pragma omp parallel for schedule(static)
    for (size_t localI = 0; localI < localRows; ++localI) {
        size_t i = startRow + localI;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            localTau[localI * N + j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }

    // Allgather: each rank contributes its rows to the full Tau matrix
    std::vector<int> recvCounts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t rStart = r * rowsPerRank;
        size_t rEnd = std::min(rStart + rowsPerRank, N);
        size_t rRows = (rStart < N) ? (rEnd - rStart) : 0;
        recvCounts[r] = static_cast<int>(rRows * N);
        displs[r] = static_cast<int>(rStart * N);
    }

    MPI_Allgatherv(localTau.data(), static_cast<int>(localRows * N), MPI_INT,
                   state.tau.data(), recvCounts.data(), displs.data(), MPI_INT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    if (mpi_rank == 0) {
        printf("Running wave propagation simulation with MPI+OpenMP...\n");
    }

    // Distribute triangles across MPI ranks
    size_t rowsPerRank = (N + mpi_size - 1) / mpi_size;
    size_t startRow = mpi_rank * rowsPerRank;
    size_t endRow = std::min(startRow + rowsPerRank, N);
    size_t localRows = (startRow < N) ? (endRow - startRow) : 0;

    // Each rank stores a contiguous block of radB for the triangles it owns
    // Full radB is shared via MPI_Allgather after each timestep
    std::vector<val_t> localRadB(T * N, ZERO);

    for (size_t t = 0; t < T; ++t) {
        // Copy emission from global state
        for (size_t localI = 0; localI < localRows; ++localI) {
            size_t i = startRow + localI;
            localRadB[t * N + i] = state.radE[state.idxTN(t, i)];
        }

        // Compute reflected radiosity for this rank's triangles
        #pragma omp parallel for schedule(static)
        for (size_t localI = 0; localI < localRows; ++localI) {
            size_t i = startRow + localI;
            val_t sumB = ZERO;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;

                int tauij = state.tau[state.idx2d(i, j)];
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[state.idx2d(i, j)];
                if (kij <= ZERO) continue;

                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = localRadB[srcTime * N + j];
                if (radJ <= ZERO) continue;

                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            localRadB[t * N + i] += state.rho[i] * sumB;
        }

        // Share this timestep's radB values across all MPI ranks
        // We gather the full vector for triangle i at time t
        std::vector<val_t> tRadB(N, ZERO);
        std::vector<int> recvCounts(mpi_size);
        std::vector<int> displs(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            size_t rStart = r * rowsPerRank;
            size_t rEnd = std::min(rStart + rowsPerRank, N);
            recvCounts[r] = static_cast<int>((rStart < N) ? (rEnd - rStart) : 0);
            displs[r] = static_cast<int>(rStart);
        }

        // Extract local radB values for this timestep
        std::vector<val_t> localRadB_t(localRows);
        for (size_t localI = 0; localI < localRows; ++localI) {
            localRadB_t[localI] = localRadB[t * N + startRow + localI];
        }

        MPI_Allgatherv(localRadB_t.data(), static_cast<int>(localRows), MPI_FLOAT,
                       tRadB.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Copy gathered data into localRadB for next iteration
        for (size_t i = 0; i < N; ++i) {
            localRadB[t * N + i] = tRadB[i];
        }

        if (mpi_rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Copy localRadB back into state.radB (only rank 0 has full data for output)
    // Actually every rank needs full radB for distance computation
    // So we allgather the full radB matrix
    if (T * N > 0) {
        std::vector<val_t> fullRadB(T * N, ZERO);
        std::vector<int> recvCounts2(mpi_size);
        std::vector<int> displs2(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            size_t rStart = r * rowsPerRank;
            size_t rEnd = std::min(rStart + rowsPerRank, N);
            size_t rLocal = (rStart < N) ? (rEnd - rStart) : 0;
            recvCounts2[r] = static_cast<int>(rLocal * T);
            displs2[r] = static_cast<int>(rStart * T);
        }

        // Prepare local radB in contiguous form for MPI
        std::vector<val_t> localRadBFlat(localRows * T, ZERO);
        for (size_t t = 0; t < T; ++t) {
            for (size_t localI = 0; localI < localRows; ++localI) {
                localRadBFlat[localI * T + t] = localRadB[t * N + startRow + localI];
            }
        }

        MPI_Allgatherv(localRadBFlat.data(), static_cast<int>(localRows * T), MPI_FLOAT,
                       fullRadB.data(), recvCounts2.data(), displs2.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Now store back: each rank's localRadB = fullRadB (transposed)
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < N; ++i) {
                state.radB[state.idxTN(t, i)] = fullRadB[i * T + t];
            }
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    if (mpi_rank == 0) {
        printf("Computing distances via cross-correlation with MPI+OpenMP+CUDA...\n");
    }

    // Distribute across MPI ranks
    size_t rowsPerRank = (N + mpi_size - 1) / mpi_size;
    size_t startRow = mpi_rank * rowsPerRank;
    size_t endRow = std::min(startRow + rowsPerRank, N);
    size_t localRows = (startRow < N) ? (endRow - startRow) : 0;

    std::vector<val_t> localDist(localRows, ZERO);

#ifdef __CUDACC__
    if (localRows > 0) {
        // CUDA kernel for distance computation
        float* d_radB = nullptr;
        float* d_dist = nullptr;

        cudaMalloc(&d_radB, T * N * sizeof(float));
        cudaMalloc(&d_dist, localRows * sizeof(float));

        cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(float), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int numBlocks = (static_cast<int>(localRows) + threadsPerBlock - 1) / threadsPerBlock;

        distanceKernel<<<numBlocks, threadsPerBlock>>>(
            d_radB, static_cast<int>(N), static_cast<int>(T),
            WAVE_SPEED, static_cast<int>(startRow), static_cast<int>(localRows),
            static_cast<int>(state.sourceIndex), d_dist
        );

        cudaDeviceSynchronize();

        cudaMemcpy(localDist.data(), d_dist, localRows * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_radB);
        cudaFree(d_dist);
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t localI = 0; localI < localRows; ++localI) {
        size_t i = startRow + localI;
        val_t maxCorr = ZERO;
        int bestT = 0;

        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDist[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
#endif

    // Allgather distances
    std::vector<int> recvCounts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t rStart = r * rowsPerRank;
        size_t rEnd = std::min(rStart + rowsPerRank, N);
        recvCounts[r] = static_cast<int>((rStart < N) ? (rEnd - rStart) : 0);
        displs[r] = static_cast<int>(rStart);
    }

    MPI_Allgatherv(localDist.data(), static_cast<int>(localRows), MPI_FLOAT,
                   state.distances.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0, then broadcast)
    if (mpi_rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&targetTriangles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sourceIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflectivity, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    // Only rank 0 prints output
    if (mpi_rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
#ifdef __CUDACC__
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA devices available: %d\n", deviceCount);
        if (deviceCount > 0) {
            cudaSetDevice(mpi_rank % deviceCount);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, mpi_rank % deviceCount);
            printf("  Using GPU: %s\n", prop.name);
        }
#endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize simulation on all ranks (each gets a copy)
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpi_rank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpi_rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpi_rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (mpi_rank == 0) {
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
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        // Validation
        if (validate) {
            if (!validateResults(state)) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
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

#include <mpi.h>
#include <omp.h>
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
constexpr int OCTREE_STACK_SIZE = 16;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit Vec3(val_t v) : x(v), y(v), z(v) {}

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

    __host__ __device__ Triangle() = default;
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
// Flattened Octree for CUDA
// ============================================================================

struct FlatNode {
    Vec3 minBound;
    Vec3 maxBound;
    int child[8];
    int triStart;
    int triCount;
};

struct FlatOctree {
    std::vector<FlatNode> nodes;
    std::vector<int> leafTriangleIndices;
};

static int flattenOctreeRec(const Octree& node, FlatOctree& flat) {
    int idx = static_cast<int>(flat.nodes.size());
    flat.nodes.push_back(FlatNode{});
    flat.nodes[idx].minBound = node.minBound;
    flat.nodes[idx].maxBound = node.maxBound;
    flat.nodes[idx].triStart = 0;
    flat.nodes[idx].triCount = 0;
    for (int c = 0; c < 8; ++c) flat.nodes[idx].child[c] = -1;

    if (!node.triangleIndices.empty()) {
        flat.nodes[idx].triStart = static_cast<int>(flat.leafTriangleIndices.size());
        flat.nodes[idx].triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t t : node.triangleIndices) {
            flat.leafTriangleIndices.push_back(static_cast<int>(t));
        }
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                int childIdx = flattenOctreeRec(*node.children[c], flat);
                flat.nodes[idx].child[c] = childIdx;
            }
        }
    }
    return idx;
}

static void flattenOctree(const Octree& root, FlatOctree& flat) {
    flat.nodes.clear();
    flat.leafTriangleIndices.clear();
    flattenOctreeRec(root, flat);
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
// Deterministic Counter-Based Random Numbers
// ============================================================================

__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct RandStream {
    uint64_t state;

    __host__ __device__ explicit RandStream(uint64_t seed) : state(seed) {}

    __host__ __device__ inline uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    __host__ __device__ inline val_t rand() {
        return static_cast<val_t>(next() >> 8) * 0x1.0p-24f;
    }
};

__host__ __device__ inline uint64_t pairSeed(size_t i, size_t j, size_t n) {
    return splitmix64(static_cast<uint64_t>(i) * static_cast<uint64_t>(n) +
                      static_cast<uint64_t>(j));
}

// Generate a random point inside a triangle using barycentric coordinates
__host__ __device__ Vec3 randomPointInTriangle(const Triangle& t, RandStream& rng) {
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

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                               const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

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

__device__ bool flatRayIntersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 halfExtent = (node.maxBound - node.minBound) * 0.5f;
    Vec3 center = (node.minBound + node.maxBound) * 0.5f;

    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ bool isRayBlockedFlat(const Vec3& from, const Vec3& to,
                                 const FlatNode* nodes,
                                 const int* leafTriangles,
                                 const Triangle* triangles,
                                 int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nodeIdx = stack[--sp];
        const FlatNode& node = nodes[nodeIdx];

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                int idx = leafTriangles[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int childIdx = node.child[c];
                if (childIdx >= 0 && flatRayIntersectsBox(nodes[childIdx], from, to)) {
                    if (sp < OCTREE_STACK_SIZE) stack[sp++] = childIdx;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                 const std::vector<Triangle>& triangles,
                 const Octree& octree,
                 RandStream& rng) {
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

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void warmupKernel() {}

__global__ void tauKernel(int startRow, int localRows, int n,
                          const Triangle* triangles, int* tau) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= localRows) return;

    int i = startRow + local;
    int* tauRow = tau + static_cast<size_t>(local) * static_cast<size_t>(n);

    for (int j = 0; j < n; ++j) {
        if (i == j) {
            tauRow[j] = 0;
        } else {
            val_t dist = (triangles[i].center() - triangles[j].center()).norm();
            tauRow[j] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
        }
    }
}

__global__ void formFactorKernel(int startRow, int n, size_t totalPairs,
                                 const Triangle* triangles,
                                 const FlatNode* nodes,
                                 const int* leafTriangles,
                                 float* kij) {
    size_t offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= totalPairs) return;

    size_t localRow = offset / static_cast<size_t>(n);
    int j = static_cast<int>(offset - localRow * static_cast<size_t>(n));
    int i = startRow + static_cast<int>(localRow);

    if (i == j) {
        kij[offset] = ZERO;
        return;
    }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    if (triI._normal.dot(triJ._normal) > 0.99f) {
        kij[offset] = ZERO;
        return;
    }

    RandStream rng(pairSeed(static_cast<size_t>(i), static_cast<size_t>(j),
                            static_cast<size_t>(n)));
    val_t kijValue = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlockedFlat(pI, pJ, nodes, leafTriangles, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI._normal);
        val_t cosPhiJ = cosPhi(-v, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kijValue += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[offset] = kijValue * INV_NUM_RAYS;
}

__global__ void simulationKernel(int t, int startRow, int localRows, int n,
                                 const float* kij,
                                 const int* tau,
                                 const float* areas,
                                 const float* rho,
                                 float* radB,
                                 int sourceIndex, int timeOff) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= localRows) return;

    int i = startRow + local;
    const float* kijRow = kij + static_cast<size_t>(local) * static_cast<size_t>(n);
    const int* tauRow = tau + static_cast<size_t>(local) * static_cast<size_t>(n);

    val_t sumB = ZERO;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tauRow[j];
        if (t < tauij) continue;

        val_t kijValue = kijRow[j];
        if (kijValue <= ZERO) continue;

        int srcTime = t - tauij;
        val_t radJ = radB[static_cast<size_t>(srcTime) * static_cast<size_t>(n) +
                          static_cast<size_t>(j)];
        if (radJ <= ZERO) continue;

        val_t contribution = kijValue * areas[j];
        if (contribution > ONE) contribution = ONE;
        sumB += contribution * radJ;
    }

    val_t emission = (t < timeOff && i == sourceIndex) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
        rho[i] * sumB + emission;
}

__global__ void distanceKernel(int startRow, int localRows, int n, int numTimesteps,
                               const float* radB, int sourceIndex, float* distances) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= localRows) return;

    int i = startRow + local;
    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * static_cast<size_t>(n) +
                            static_cast<size_t>(i)];
            val_t pS = radB[static_cast<size_t>(tt - t) * static_cast<size_t>(n) +
                            static_cast<size_t>(sourceIndex)];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[local] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countNonzeroKernel(const float* kij, size_t count, int* result) {
    int countLocal = 0;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (kij[idx] > EPSILON) countLocal++;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        countLocal += __shfl_down_sync(0xffffffffu, countLocal, offset);
    }

    __shared__ int warpCounts[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warpCounts[warp] = countLocal;
    __syncthreads();

    if (warp == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        countLocal = (lane < numWarps) ? warpCounts[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            countLocal += __shfl_down_sync(0xffffffffu, countLocal, offset);
        }
        if (lane == 0) atomicAdd(result, countLocal);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kijLocal;    // Local form-factor rows (rowCount x N)
    std::vector<int> tauLocal;      // Local time delays (rowCount x N)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;          // Device-friendly acceleration structure

    size_t sourceIndex = 0;

    int rank = 0;
    int numRanks = 1;
    size_t rowStart = 0;
    size_t rowCount = 0;

    size_t idxLocal(size_t localRow, size_t j) const { return localRow * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

struct DeviceBuffers {
    bool enabled = false;
    Triangle* triangles = nullptr;
    FlatNode* nodes = nullptr;
    int* leafTriangles = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* kij = nullptr;
    int* tau = nullptr;
    float* radB = nullptr;
    float* distances = nullptr;
    int* nonzeroCount = nullptr;
};

static DeviceBuffers g_dev;

static void freeDeviceBuffers() {
    if (!g_dev.enabled) return;
    cudaFree(g_dev.triangles);
    cudaFree(g_dev.nodes);
    cudaFree(g_dev.leafTriangles);
    cudaFree(g_dev.areas);
    cudaFree(g_dev.rho);
    cudaFree(g_dev.kij);
    cudaFree(g_dev.tau);
    cudaFree(g_dev.radB);
    cudaFree(g_dev.distances);
    cudaFree(g_dev.nonzeroCount);
    g_dev = DeviceBuffers{};
}

static size_t allocationSize(size_t bytes) {
    return bytes > 0 ? bytes : 1;
}

static bool initializeCuda(SimulationState& state) {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        cudaGetLastError();
        return false;
    }

    int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    size_t localCount = state.rowCount * n;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.triangles),
                          allocationSize(n * sizeof(Triangle))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.nodes),
                          allocationSize(state.flatOctree.nodes.size() * sizeof(FlatNode))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.leafTriangles),
                          allocationSize(state.flatOctree.leafTriangleIndices.size() * sizeof(int))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.areas),
                          allocationSize(n * sizeof(float))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.rho),
                          allocationSize(n * sizeof(float))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.kij),
                          allocationSize(localCount * sizeof(float))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.tau),
                          allocationSize(localCount * sizeof(int))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.radB),
                          allocationSize(t * n * sizeof(float))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.distances),
                          allocationSize(state.rowCount * sizeof(float))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&g_dev.nonzeroCount), sizeof(int)));

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(g_dev.triangles, state.triangles.data(),
                              n * sizeof(Triangle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_dev.areas, state.areas.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g_dev.rho, state.rho.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (!state.flatOctree.nodes.empty()) {
        CUDA_CHECK(cudaMemcpy(g_dev.nodes, state.flatOctree.nodes.data(),
                              state.flatOctree.nodes.size() * sizeof(FlatNode),
                              cudaMemcpyHostToDevice));
    }
    if (!state.flatOctree.leafTriangleIndices.empty()) {
        CUDA_CHECK(cudaMemcpy(g_dev.leafTriangles,
                              state.flatOctree.leafTriangleIndices.data(),
                              state.flatOctree.leafTriangleIndices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    if (t * n > 0) {
        CUDA_CHECK(cudaMemset(g_dev.radB, 0, t * n * sizeof(float)));
    }

    warmupKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    g_dev.enabled = true;
    return true;
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

    if (state.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (state.rank == 0) {
        printf("Building octree...\n");
    }
    state.octree.build(state.triangles);
    flattenOctree(state.octree, state.flatOctree);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Distribute matrix rows across MPI ranks
    size_t base = state.numTriangles / static_cast<size_t>(state.numRanks);
    size_t rem = state.numTriangles % static_cast<size_t>(state.numRanks);
    state.rowCount = base + (static_cast<size_t>(state.rank) < rem ? 1 : 0);
    state.rowStart = static_cast<size_t>(state.rank) * base +
                     std::min(static_cast<size_t>(state.rank), rem);

    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) {
        printf("Computing time delays (Tau)...\n");
    }

    size_t n = state.numTriangles;

    if (g_dev.enabled) {
        if (state.rowCount > 0 && n > 0) {
            const int blockSize = 256;
            int gridSize = static_cast<int>((state.rowCount + blockSize - 1) / blockSize);
            tauKernel<<<gridSize, blockSize>>>(
                static_cast<int>(state.rowStart),
                static_cast<int>(state.rowCount),
                static_cast<int>(n),
                g_dev.triangles,
                g_dev.tau);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        return;
    }

    state.tauLocal.assign(state.rowCount * n, 0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t localRow = 0; localRow < state.rowCount; ++localRow) {
        size_t i = state.rowStart + localRow;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            state.tauLocal[state.idxLocal(localRow, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

void computeFormFactorsCPU(SimulationState& state) {
    size_t n = state.numTriangles;

    state.kijLocal.assign(state.rowCount * n, ZERO);

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t localRow = 0; localRow < state.rowCount; ++localRow) {
        size_t i = state.rowStart + localRow;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            RandStream rng(pairSeed(i, j, n));
            state.kijLocal[state.idxLocal(localRow, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
    }
}

void computeFormFactorsGPU(SimulationState& state) {
    size_t n = state.numTriangles;
    size_t localCount = state.rowCount * n;

    if (localCount > 0) {
        CUDA_CHECK(cudaMemset(g_dev.kij, 0, localCount * sizeof(float)));
    }

    if (state.rowCount > 0 && n > 0) {
        const int blockSize = 128;
        size_t totalPairs = state.rowCount * n;
        unsigned int gridSize = static_cast<unsigned int>(
            (totalPairs + static_cast<size_t>(blockSize) - 1) /
            static_cast<size_t>(blockSize));
        formFactorKernel<<<gridSize, blockSize>>>(
            static_cast<int>(state.rowStart),
            static_cast<int>(n),
            totalPairs,
            g_dev.triangles,
            g_dev.nodes,
            g_dev.leafTriangles,
            g_dev.kij);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) {
        printf("Computing form factors (Kij)...\n");
    }

    if (g_dev.enabled) {
        computeFormFactorsGPU(state);
    } else {
        computeFormFactorsCPU(state);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

static std::vector<int> gatherCounts(size_t n, int numRanks) {
    std::vector<int> counts(static_cast<size_t>(numRanks));
    size_t base = n / static_cast<size_t>(numRanks);
    size_t rem = n % static_cast<size_t>(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[static_cast<size_t>(r)] = static_cast<int>(
            base + (static_cast<size_t>(r) < rem ? 1 : 0));
    }
    return counts;
}

static std::vector<int> gatherDisplacements(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size());
    size_t offset = 0;
    for (size_t r = 0; r < counts.size(); ++r) {
        displs[r] = static_cast<int>(offset);
        offset += static_cast<size_t>(counts[r]);
    }
    return displs;
}

void runSimulation(SimulationState& state) {
    if (state.rank == 0) {
        printf("Running wave propagation simulation...\n");
    }

    size_t n = state.numTriangles;
    int sourceIndex = static_cast<int>(state.sourceIndex);
    int timeOff = static_cast<int>(state.numTimesteps / 2);

    std::vector<int> counts = gatherCounts(n, state.numRanks);
    std::vector<int> displs = gatherDisplacements(counts);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        float* fullRow = state.radB.data() + state.idxTN(t, 0);

        if (g_dev.enabled) {
            if (state.rowCount > 0 && n > 0) {
                const int blockSize = 256;
                int gridSize = static_cast<int>((state.rowCount + blockSize - 1) / blockSize);
                simulationKernel<<<gridSize, blockSize>>>(
                    static_cast<int>(t),
                    static_cast<int>(state.rowStart),
                    static_cast<int>(state.rowCount),
                    static_cast<int>(n),
                    g_dev.kij,
                    g_dev.tau,
                    g_dev.areas,
                    g_dev.rho,
                    g_dev.radB,
                    sourceIndex,
                    timeOff);
                CUDA_CHECK(cudaGetLastError());
            }
            if (state.rowCount > 0) {
                CUDA_CHECK(cudaMemcpy(fullRow + state.rowStart,
                                      g_dev.radB + state.idxTN(t, 0) + state.rowStart,
                                      state.rowCount * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t localRow = 0; localRow < state.rowCount; ++localRow) {
                size_t i = state.rowStart + localRow;
                val_t sumB = ZERO;

                for (size_t j = 0; j < n; ++j) {
                    if (i == j) continue;
                    int tauij = state.tauLocal[state.idxLocal(localRow, j)];
                    if (static_cast<int>(t) < tauij) continue;
                    val_t kijValue = state.kijLocal[state.idxLocal(localRow, j)];
                    if (kijValue <= ZERO) continue;

                    size_t srcTime = t - static_cast<size_t>(tauij);
                    val_t radJ = state.radB[state.idxTN(srcTime, j)];
                    if (radJ <= ZERO) continue;

                    sumB += std::min(kijValue * state.areas[j], ONE) * radJ;
                }

                val_t emission = (static_cast<int>(t) < timeOff &&
                                  static_cast<int>(i) == sourceIndex) ? ONE : ZERO;
                state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + emission;
            }
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       fullRow, counts.data(), displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        if (g_dev.enabled && n > 0) {
            CUDA_CHECK(cudaMemcpy(g_dev.radB + state.idxTN(t, 0), fullRow,
                                  n * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) {
        printf("Computing distances via cross-correlation...\n");
    }

    size_t n = state.numTriangles;
    size_t numTimesteps = state.numTimesteps;
    int sourceIndex = static_cast<int>(state.sourceIndex);

    if (g_dev.enabled && numTimesteps > 0) {
        if (state.rowCount > 0 && n > 0) {
            const int blockSize = 128;
            int gridSize = static_cast<int>((state.rowCount + blockSize - 1) / blockSize);
            distanceKernel<<<gridSize, blockSize>>>(
                static_cast<int>(state.rowStart),
                static_cast<int>(state.rowCount),
                static_cast<int>(n),
                static_cast<int>(numTimesteps),
                g_dev.radB,
                sourceIndex,
                g_dev.distances);
            CUDA_CHECK(cudaGetLastError());
        }
        if (state.rowCount > 0) {
            CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowStart,
                                  g_dev.distances,
                                  state.rowCount * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t localRow = 0; localRow < state.rowCount; ++localRow) {
            size_t i = state.rowStart + localRow;
            val_t maxCorr = ZERO;
            int bestT = 0;

            for (size_t t = 0; t < numTimesteps; ++t) {
                val_t sum = ZERO;
                for (size_t tt = t; tt < numTimesteps; ++tt) {
                    val_t pB = state.radB[state.idxTN(tt, i)];
                    val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                    sum += pS * pB;
                }
                if (sum > maxCorr) {
                    maxCorr = sum;
                    bestT = static_cast<int>(t);
                }
            }

            state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
        }
    }

    std::vector<int> counts = gatherCounts(n, state.numRanks);
    std::vector<int> displs = gatherDisplacements(counts);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), counts.data(), displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

long long countNonzeroFormFactors(const SimulationState& state) {
    size_t localCount = state.rowCount * state.numTriangles;
    if (localCount == 0) return 0;

    if (g_dev.enabled) {
        CUDA_CHECK(cudaMemset(g_dev.nonzeroCount, 0, sizeof(int)));
        int blockSize = 256;
        int gridSize = static_cast<int>(
            std::min<size_t>((localCount + blockSize - 1) / blockSize, 65535));
        countNonzeroKernel<<<gridSize, blockSize>>>(g_dev.kij, localCount,
                                                    g_dev.nonzeroCount);
        CUDA_CHECK(cudaGetLastError());
        int local = 0;
        CUDA_CHECK(cudaMemcpy(&local, g_dev.nonzeroCount, sizeof(int),
                              cudaMemcpyDeviceToHost));
        return local;
    }

    long long local = 0;
    #pragma omp parallel for reduction(+:local) schedule(static)
    for (size_t idx = 0; idx < localCount; ++idx) {
        if (state.kijLocal[idx] > EPSILON) local++;
    }
    return local;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, long long nonZeroKij) {
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
    size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
           nonZeroKij, totalKij,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalKij));

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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    if (!getenv("OMP_NUM_THREADS")) {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
        int localSize = 1;
        MPI_Comm_size(nodeComm, &localSize);
        MPI_Comm_free(&nodeComm);
        int procs = omp_get_num_procs();
        int threads = std::max(1, procs / std::max(1, localSize));
        omp_set_num_threads(threads);
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.numRanks = numRanks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (!initializeCuda(state)) {
        if (rank == 0) {
            printf("WARNING: No CUDA device available; using CPU fallback\n");
        }
    }

    if (rank == 0) {
        printf("\n");
    }

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    long localPhaseDurations[3] = {preDuration, simDuration, distDuration};
    long globalPhaseDurations[3];
    MPI_Reduce(localPhaseDurations, globalPhaseDurations, 3, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    long long localNonZeroKij = 0;
    if (validate) {
        localNonZeroKij = countNonzeroFormFactors(state);
    }
    long long globalNonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    int returnCode = 0;

    if (rank == 0) {
        // Total time
        long totalTime = globalPhaseDurations[0] + globalPhaseDurations[1] + globalPhaseDurations[2];
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
            if (!validateResults(state, globalNonZeroKij)) {
                returnCode = 1;
            }
        }
    }

    freeDeviceBuffers();
    MPI_Finalize();
    return returnCode;
}

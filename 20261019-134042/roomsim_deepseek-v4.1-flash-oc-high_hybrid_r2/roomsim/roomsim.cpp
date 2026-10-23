/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of the radiosity-based room
 * impulse response simulation.
 *
 * Parallelisation strategy:
 *   - MPI splits the triangle rows (and therefore the form-factor matrix)
 *     across processes.  Each process owns a contiguous block of receiver
 *     triangles.
 *   - CUDA performs the per-pair ray tracing / octree traversal, the time
 *     delay computation, the time-stepping radiosity update and the
 *     cross-correlation used for distance estimation.
 *   - OpenMP parallelises the host-side setup, mesh/octree construction and
 *     validation work.
 *
 * The sequential random-number stream is preserved exactly: each process
 * fast-forwards a std::mt19937 seeded with 42 to its first row and then
 * generates the per-pair samples in the original order.
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

static int g_rank = 0;
static int g_size = 1;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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
// Device-side POD types and helpers
// ============================================================================

struct DVec3 {
    float x, y, z;

    __host__ __device__ DVec3() : x(0), y(0), z(0) {}
    __host__ __device__ DVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ DVec3 operator+(const DVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ DVec3 operator-(const DVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ DVec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ DVec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ DVec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ float dot(const DVec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ DVec3 cross(const DVec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
};

struct DTrian {
    DVec3 a, b, c, n;
};

struct DNode {
    DVec3 center, half;
    int child[8];
    int triStart;
    int triCount;
};

__device__ __forceinline__ DVec3 dcenter(const DTrian& t) {
    return (t.a + t.b + t.c) / 3.0f;
}

__device__ __forceinline__ float dcosPhi(const DVec3& v, const DVec3& normal) {
    float vNorm = sqrtf(v.dot(v));
    if (vNorm <= EPSILON) return ZERO;
    float d = v.dot(normal) / vNorm;
    return d > ZERO ? d : ZERO;
}

__device__ __forceinline__ float dRayTriangle(const DVec3& orig, const DVec3& dir,
                                              const DVec3& v0, const DVec3& v1, const DVec3& v2) {
    DVec3 e1 = v1 - v0;
    DVec3 e2 = v2 - v0;
    DVec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    DVec3 tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    DVec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

__device__ __forceinline__ bool dRayIntersectsBox(const DNode& node, const DVec3& p1, const DVec3& p2) {
    DVec3 d = (p2 - p1) * 0.5f;
    DVec3 c = p1 + d - node.center;
    float adx = fabsf(d.x), ady = fabsf(d.y), adz = fabsf(d.z);

    if (fabsf(c.x) > node.half.x + adx) return false;
    if (fabsf(c.y) > node.half.y + ady) return false;
    if (fabsf(c.z) > node.half.z + adz) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.half.y * adz + node.half.z * ady + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.half.z * adx + node.half.x * adz + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.half.x * ady + node.half.y * adx + EPSILON) return false;

    return true;
}

__device__ __forceinline__ bool dRayBlocked(const DVec3& from, const DVec3& to,
                                            const DNode* nodes, const int* nodeTris,
                                            const DTrian* tris, int srcTriIdx, int dstTriIdx) {
    DVec3 dir = to - from;
    float rayLen = sqrtf(dir.dot(dir));
    if (rayLen < EPSILON) return true;
    DVec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        const DNode& nd = nodes[ni];

        if (nd.triCount > 0) {
            int end = nd.triStart + nd.triCount;
            for (int p = nd.triStart; p < end; ++p) {
                int idx = nodeTris[p];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const DTrian& tr = tris[idx];
                float dist = dRayTriangle(from, dirNorm, tr.a, tr.b, tr.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 7; c >= 0; --c) {
                int ci = nd.child[c];
                if (ci >= 0 && dRayIntersectsBox(nodes[ci], from, to)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

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
        #pragma omp parallel
        {
            Vec3 lmin = minBound, lmax = maxBound;
            #pragma omp for nowait
            for (size_t ti = 0; ti < triangles.size(); ++ti) {
                const auto& tri = triangles[ti];
                for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                    lmin.x = std::min(lmin.x, v->x);
                    lmin.y = std::min(lmin.y, v->y);
                    lmin.z = std::min(lmin.z, v->z);
                    lmax.x = std::max(lmax.x, v->x);
                    lmax.y = std::max(lmax.y, v->y);
                    lmax.z = std::max(lmax.z, v->z);
                }
            }
            #pragma omp critical
            {
                minBound.x = std::min(minBound.x, lmin.x);
                minBound.y = std::min(minBound.y, lmin.y);
                minBound.z = std::min(minBound.z, lmin.z);
                maxBound.x = std::max(maxBound.x, lmax.x);
                maxBound.y = std::max(maxBound.y, lmax.y);
                maxBound.z = std::max(maxBound.z, lmax.z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        #pragma omp parallel for
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
    void discard(uint64_t n) { rng.discard(n); }
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
// Visibility Testing (Octree-accelerated, host reference version)
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major) - on device
    std::vector<int> tau;           // Time delays (N x N matrix, row-major) - on device
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    long long nonZeroKij;

    SimulationState() : numTriangles(0), numTimesteps(0), sourceIndex(0), nonZeroKij(0) {}

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// CUDA device buffers
// ============================================================================

struct GpuContext {
    DTrian* d_tris = nullptr;
    DNode* d_nodes = nullptr;
    int* d_nodeTris = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_rng = nullptr;
    float* d_dist = nullptr;
    unsigned long long* d_nzKij = nullptr;

    int numNodes = 0;
    int numNodeTris = 0;
    int N = 0;
    int T = 0;
    int rowStart = 0;
    int localRows = 0;
};

static int flattenNode(const Octree& node, std::vector<DNode>& nodes,
                       std::vector<int>& triIdx) {
    int idx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    nodes[idx].center = DVec3(node.center.x, node.center.y, node.center.z);
    nodes[idx].half = DVec3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
    nodes[idx].triStart = -1;
    nodes[idx].triCount = 0;
    for (int c = 0; c < 8; ++c) nodes[idx].child[c] = -1;

    if (!node.triangleIndices.empty()) {
        nodes[idx].triStart = static_cast<int>(triIdx.size());
        nodes[idx].triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t t : node.triangleIndices) triIdx.push_back(static_cast<int>(t));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                int ci = flattenNode(*node.children[c], nodes, triIdx);
                nodes[idx].child[c] = ci;
            }
        }
    }
    return idx;
}

static void setupDeviceBuffers(SimulationState& state, GpuContext& ctx,
                               int rowStart, int localRows) {
    ctx.N = static_cast<int>(state.numTriangles);
    ctx.T = static_cast<int>(state.numTimesteps);
    ctx.rowStart = rowStart;
    ctx.localRows = localRows;
    const int N = ctx.N;

    std::vector<DTrian> hostTris(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        hostTris[i].a = DVec3(t.a.x, t.a.y, t.a.z);
        hostTris[i].b = DVec3(t.b.x, t.b.y, t.b.z);
        hostTris[i].c = DVec3(t.c.x, t.c.y, t.c.z);
        hostTris[i].n = DVec3(t._normal.x, t._normal.y, t._normal.z);
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_tris, sizeof(DTrian) * N));
    CUDA_CHECK(cudaMemcpy(ctx.d_tris, hostTris.data(), sizeof(DTrian) * N,
                          cudaMemcpyHostToDevice));

    std::vector<DNode> hostNodes;
    std::vector<int> hostNodeTris;
    hostNodes.reserve(N * 2);
    hostNodeTris.reserve(N);
    flattenNode(state.octree, hostNodes, hostNodeTris);
    ctx.numNodes = static_cast<int>(hostNodes.size());
    ctx.numNodeTris = static_cast<int>(hostNodeTris.size());
    CUDA_CHECK(cudaMalloc(&ctx.d_nodes, sizeof(DNode) * hostNodes.size()));
    CUDA_CHECK(cudaMemcpy(ctx.d_nodes, hostNodes.data(), sizeof(DNode) * hostNodes.size(),
                          cudaMemcpyHostToDevice));
    if (!hostNodeTris.empty()) {
        CUDA_CHECK(cudaMalloc(&ctx.d_nodeTris, sizeof(int) * hostNodeTris.size()));
        CUDA_CHECK(cudaMemcpy(ctx.d_nodeTris, hostNodeTris.data(),
                              sizeof(int) * hostNodeTris.size(), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&ctx.d_areas, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(ctx.d_areas, state.areas.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_rho, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(ctx.d_rho, state.rho.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));

    size_t radElems = static_cast<size_t>(ctx.T) * N;
    CUDA_CHECK(cudaMalloc(&ctx.d_radE, sizeof(float) * radElems));
    CUDA_CHECK(cudaMemcpy(ctx.d_radE, state.radE.data(), sizeof(float) * radElems,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_radB, sizeof(float) * radElems));
    CUDA_CHECK(cudaMemset(ctx.d_radB, 0, sizeof(float) * radElems));

    if (localRows > 0) {
        size_t localElems = static_cast<size_t>(localRows) * N;
        CUDA_CHECK(cudaMalloc(&ctx.d_kij, sizeof(float) * localElems));
        CUDA_CHECK(cudaMalloc(&ctx.d_tau, sizeof(int) * localElems));
        CUDA_CHECK(cudaMalloc(&ctx.d_dist, sizeof(float) * localRows));
        CUDA_CHECK(cudaMemset(ctx.d_kij, 0, sizeof(float) * localElems));
        CUDA_CHECK(cudaMemset(ctx.d_tau, 0, sizeof(int) * localElems));
    }

    CUDA_CHECK(cudaMalloc(&ctx.d_nzKij, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.d_nzKij, 0, sizeof(unsigned long long)));
}

static void freeDeviceBuffers(GpuContext& ctx) {
    if (ctx.d_tris) cudaFree(ctx.d_tris);
    if (ctx.d_nodes) cudaFree(ctx.d_nodes);
    if (ctx.d_nodeTris) cudaFree(ctx.d_nodeTris);
    if (ctx.d_areas) cudaFree(ctx.d_areas);
    if (ctx.d_rho) cudaFree(ctx.d_rho);
    if (ctx.d_radE) cudaFree(ctx.d_radE);
    if (ctx.d_radB) cudaFree(ctx.d_radB);
    if (ctx.d_kij) cudaFree(ctx.d_kij);
    if (ctx.d_tau) cudaFree(ctx.d_tau);
    if (ctx.d_rng) cudaFree(ctx.d_rng);
    if (ctx.d_dist) cudaFree(ctx.d_dist);
    if (ctx.d_nzKij) cudaFree(ctx.d_nzKij);
    ctx = GpuContext{};
}

// ============================================================================
// CUDA kernels
// ============================================================================

// One pair of triangles is handled by 16 lanes (one lane per ray).  The lane
// results are combined in ray order by lane 0, preserving the sequential
// floating-point summation order of the reference implementation.
__global__ void kijKernel(const DTrian* __restrict__ tris,
                          const DNode* __restrict__ nodes,
                          const int* __restrict__ nodeTris,
                          const float* __restrict__ rng,
                          float* __restrict__ kij,
                          int N, int rowBase, int numRows) {
    extern __shared__ float terms[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = tid >> 4;
    int lane = tid & 15;
    int total = numRows * N;
    bool active = pair < total;

    int lr = 0, j = 0, i = 0;
    if (active) {
        lr = pair / N;
        j = pair - lr * N;
        i = rowBase + lr;
    }

    float term = 0.0f;

    if (active && i != j) {
        const DTrian& ti = tris[i];
        const DTrian& tj = tris[j];

        if (ti.n.dot(tj.n) <= 0.99f) {
            const float* rp = rng + (static_cast<size_t>(lr) * N + j) * 64u
                                    + static_cast<size_t>(lane) * 4u;

            float u = rp[0];
            float v = rp[1];
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            DVec3 pI = ti.a + (ti.b - ti.a) * u + (ti.c - ti.a) * v;

            float u2 = rp[2];
            float v2 = rp[3];
            if (u2 + v2 > 1.0f) {
                u2 = 1.0f - u2;
                v2 = 1.0f - v2;
            }
            DVec3 pJ = tj.a + (tj.b - tj.a) * u2 + (tj.c - tj.a) * v2;

            if (!dRayBlocked(pI, pJ, nodes, nodeTris, tris, i, j)) {
                DVec3 vv = pJ - pI;
                float distSqr = vv.dot(vv);
                if (distSqr >= EPSILON) {
                    float cosPhiI = dcosPhi(vv, ti.n);
                    float cosPhiJ = dcosPhi(-vv, tj.n);
                    if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                        term = (cosPhiI * cosPhiJ) / (PI * distSqr);
                    }
                }
            }
        }
    }

    terms[threadIdx.x] = term;
    __syncthreads();

    if (active && lane == 0) {
        int base = (threadIdx.x >> 4) << 4;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += terms[base + k];
        }
        kij[pair] = sum * INV_NUM_RAYS;
    }
}

__global__ void tauKernel(const DTrian* __restrict__ tris,
                          int* __restrict__ tau,
                          int N, int rowBase, int localRows) {
    int total = localRows * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {
        int lr = idx / N;
        int j = idx - lr * N;
        int i = rowBase + lr;
        int value = 0;
        if (i != j) {
            DVec3 ci = dcenter(tris[i]);
            DVec3 cj = dcenter(tris[j]);
            DVec3 d = ci - cj;
            float dist = sqrtf(d.dot(d));
            value = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
        }
        tau[idx] = value;
    }
}

__global__ void simKernel(const float* __restrict__ kij,
                          const int* __restrict__ tau,
                          const float* __restrict__ areas,
                          const float* __restrict__ rho,
                          const float* __restrict__ radE,
                          float* __restrict__ radB,
                          int N, int rowBase, int localRows,
                          int t, int sourceIndex) {
    int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= localRows) return;

    int i = rowBase + lr;
    float sumB = 0.0f;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[static_cast<size_t>(lr) * N + j];
        if (t < tauij) continue;

        float kijv = kij[static_cast<size_t>(lr) * N + j];
        if (kijv <= ZERO) continue;

        int srcTime = t - tauij;
        float radJ = radB[static_cast<size_t>(srcTime) * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijv * areas[j], ONE) * radJ;
    }

    radB[static_cast<size_t>(t) * N + i] =
        rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

__global__ void distanceKernel(const float* __restrict__ radB,
                               float* __restrict__ distances,
                               int N, int rowBase, int localRows,
                               int T, int sourceIndex) {
    int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= localRows) return;

    int i = rowBase + lr;
    float maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * N + i];
            float pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[lr] = WAVE_SPEED * static_cast<float>(bestT);
}

__global__ void countNonZeroKernel(const float* __restrict__ data, size_t n,
                                   unsigned long long* __restrict__ out) {
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned long long local = 0;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (data[i] > EPSILON) local++;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xffffffffu, local, off);
    }

    __shared__ unsigned long long warpSums[32];
    if (lane == 0) warpSums[warp] = local;
    __syncthreads();

    if (warp == 0) {
        unsigned int nwarps = (blockDim.x + 31u) >> 5;
        unsigned long long v = (lane < nwarps) ? warpSums[lane] : 0ull;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) atomicAdd(out, v);
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

    if (g_rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Matrices kij/tau live on the device; only the replicated radiosity,
    // emission and result arrays are kept on the host.
    state.kij.clear();
    state.tau.clear();
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

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

static void computeTimeDelaysGpu(const GpuContext& ctx) {
    if (ctx.localRows == 0) return;
    long long total = static_cast<long long>(ctx.localRows) * ctx.N;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    tauKernel<<<blocks, threads>>>(ctx.d_tris, ctx.d_tau, ctx.N,
                                   ctx.rowStart, ctx.localRows);
    CUDA_CHECK(cudaGetLastError());
}

static void computeFormFactorsGpu(GpuContext& ctx) {
    if (ctx.localRows == 0) return;

    RandomGenerator rng(42);
    const int N = ctx.N;

    // Fast-forward the exact sequential random stream to this rank's first row.
    uint64_t pairsBefore = static_cast<uint64_t>(ctx.rowStart) *
                           static_cast<uint64_t>(N > 0 ? N - 1 : 0);
    rng.discard(pairsBefore * 64ull);

    const size_t perRow = static_cast<size_t>(64) * N;
    const size_t budget = static_cast<size_t>(256) << 20;  // 256 MiB staging
    size_t maxRows = perRow > 0 ? budget / (perRow * sizeof(float)) : 1;
    if (maxRows < 1) maxRows = 1;
    size_t chunk = std::min<size_t>(maxRows, static_cast<size_t>(ctx.localRows));

    std::vector<float> rngHost(chunk * perRow);
    CUDA_CHECK(cudaMalloc(&ctx.d_rng, sizeof(float) * rngHost.size()));

    const int threads = 256;
    const int pairsPerBlock = threads / 16;

    for (size_t lr0 = 0; lr0 < static_cast<size_t>(ctx.localRows); lr0 += chunk) {
        size_t b = std::min(chunk, static_cast<size_t>(ctx.localRows) - lr0);

        for (size_t r = 0; r < b; ++r) {
            size_t globalI = static_cast<size_t>(ctx.rowStart) + lr0 + r;
            size_t rowBase = r * perRow;
            for (int j = 0; j < N; ++j) {
                if (static_cast<size_t>(j) == globalI) continue;
                size_t base = rowBase + static_cast<size_t>(j) * 64;
                for (int k = 0; k < 64; ++k) {
                    rngHost[base + k] = rng.rand();
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(ctx.d_rng, rngHost.data(),
                              sizeof(float) * b * perRow, cudaMemcpyHostToDevice));

        int numPairs = static_cast<int>(b) * N;
        int blocks = (numPairs + pairsPerBlock - 1) / pairsPerBlock;
        kijKernel<<<blocks, threads, sizeof(float) * threads>>>(
            ctx.d_tris, ctx.d_nodes, ctx.d_nodeTris, ctx.d_rng,
            ctx.d_kij + lr0 * N, N,
            ctx.rowStart + static_cast<int>(lr0), static_cast<int>(b));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(ctx.d_rng));
    ctx.d_rng = nullptr;

    // Count non-zero form factors for validation.
    size_t localElems = static_cast<size_t>(ctx.localRows) * N;
    int blocks = static_cast<int>((localElems + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    countNonZeroKernel<<<blocks, threads>>>(ctx.d_kij, localElems, ctx.d_nzKij);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuContext& ctx,
                   int* recvCounts, int* displs) {
    if (g_rank == 0) printf("Running wave propagation simulation...\n");

    const int N = ctx.N;
    const int T = ctx.T;
    std::vector<float> localVals(ctx.localRows);

    for (int t = 0; t < T; ++t) {
        if (ctx.localRows > 0) {
            int threads = 256;
            int blocks = (ctx.localRows + threads - 1) / threads;
            simKernel<<<blocks, threads>>>(ctx.d_kij, ctx.d_tau, ctx.d_areas,
                                           ctx.d_rho, ctx.d_radE, ctx.d_radB,
                                           N, ctx.rowStart, ctx.localRows,
                                           t, static_cast<int>(state.sourceIndex));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localVals.data(),
                                  ctx.d_radB + static_cast<size_t>(t) * N + ctx.rowStart,
                                  sizeof(float) * ctx.localRows,
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(localVals.data(), ctx.localRows, MPI_FLOAT,
                       state.radB.data() + static_cast<size_t>(t) * N,
                       recvCounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

        // Make the complete time slice available on every device for future
        // time steps.
        CUDA_CHECK(cudaMemcpy(ctx.d_radB + static_cast<size_t>(t) * N,
                              state.radB.data() + static_cast<size_t>(t) * N,
                              sizeof(float) * N, cudaMemcpyHostToDevice));

        if (g_rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuContext& ctx,
                      int* recvCounts, int* displs) {
    if (g_rank == 0) printf("Computing distances via cross-correlation...\n");

    std::vector<float> localDist(ctx.localRows);

    if (ctx.localRows > 0) {
        int threads = 128;
        int blocks = (ctx.localRows + threads - 1) / threads;
        distanceKernel<<<blocks, threads>>>(ctx.d_radB, ctx.d_dist,
                                            ctx.N, ctx.rowStart, ctx.localRows,
                                            ctx.T, static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDist.data(), ctx.d_dist,
                              sizeof(float) * ctx.localRows,
                              cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(localDist.data(), ctx.localRows, MPI_FLOAT,
                state.distances.data(), recvCounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
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
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
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
    long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
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

static void initMpiDevice(int rank) {
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        if (g_rank == 0) {
            fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(err));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    initMpiDevice(g_rank);

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (g_rank == 0) printf("\n");

    // Distribute triangle rows across ranks.
    int N = static_cast<int>(state.numTriangles);
    int base = N / g_size;
    int rem = N % g_size;
    int rowStart = g_rank * base + std::min(g_rank, rem);
    int localRows = base + (g_rank < rem ? 1 : 0);

    int* recvCounts = new int[g_size];
    int* displs = new int[g_size];
    for (int r = 0; r < g_size; ++r) {
        recvCounts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + std::min(r, rem);
    }

    GpuContext ctx;
    setupDeviceBuffers(state, ctx, rowStart, localRows);

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    if (g_rank == 0) printf("Computing time delays (Tau)...\n");
    computeTimeDelaysGpu(ctx);

    if (g_rank == 0) printf("Computing form factors (Kij)...\n");
    computeFormFactorsGpu(ctx);

    // Gather the non-zero form-factor count.
    unsigned long long localNz = 0, totalNz = 0;
    CUDA_CHECK(cudaMemcpy(&localNz, ctx.d_nzKij, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    MPI_Reduce(&localNz, &totalNz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    state.nonZeroKij = static_cast<long long>(totalNz);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (g_rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, ctx, recvCounts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (g_rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, ctx, recvCounts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (g_rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long localTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
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
                freeDeviceBuffers(ctx);
                delete[] recvCounts;
                delete[] displs;
                MPI_Finalize();
                return 1;
            }
        }
    } else if (validate) {
        // Other ranks still participate in the radiosity validation logic only
        // through the replicated data; no communication is required here.
    }

    freeDeviceBuffers(ctx);
    delete[] recvCounts;
    delete[] displs;

    MPI_Finalize();
    return 0;
}

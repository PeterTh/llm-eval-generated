/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of a simplified room impulse response
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

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#ifdef __CUDACC__
#define RS_HOST_DEVICE __host__ __device__
#else
#define RS_HOST_DEVICE
#endif

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

    RS_HOST_DEVICE constexpr Vec3() : x(0), y(0), z(0) {}
    RS_HOST_DEVICE constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    RS_HOST_DEVICE explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

// A compact, index-based representation of the host octree.  The original
// tree owns its children with unique_ptrs, which is convenient on the host but
// cannot be traversed by a CUDA kernel.  Child indices preserve the original
// child visitation order (0 through 7).
struct DeviceOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    idx_t leafOffset;
    idx_t leafCount;
};

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
    void flatten(std::vector<DeviceOctreeNode>& nodes,
                 std::vector<idx_t>& leafIndices) const {
        const size_t nodeIndex = nodes.size();
        DeviceOctreeNode flat{};
        flat.center = center;
        flat.halfExtent = halfExtent;
        flat.leafOffset = static_cast<idx_t>(leafIndices.size());
        flat.leafCount = static_cast<idx_t>(triangleIndices.size());
        for (int i = 0; i < 8; ++i) flat.children[i] = -1;
        nodes.push_back(flat);

        for (size_t idx : triangleIndices) {
            leafIndices.push_back(static_cast<idx_t>(idx));
        }

        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                nodes[nodeIndex].children[i] = static_cast<int>(nodes.size());
                children[i]->flatten(nodes, leafIndices);
            }
        }
    }

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

class CudaSimulation;

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
    std::unique_ptr<CudaSimulation> cuda;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    SimulationState() = default;
    ~SimulationState();
};

// ============================================================================
// CUDA kernels and persistent device state
// ============================================================================

__device__ __forceinline__ Vec3 deviceSub(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 deviceAdd(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 deviceScale(const Vec3& a, val_t s) {
    return Vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ val_t deviceDot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 deviceCross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ val_t deviceSquaredNorm(const Vec3& v) {
    return deviceDot(v, v);
}

__device__ __forceinline__ Vec3 deviceTriangleCenter(const Triangle& tri) {
    return deviceScale(deviceAdd(deviceAdd(tri.a, tri.b), tri.c), 1.0f / 3.0f);
}

__device__ __forceinline__ val_t deviceCosPhi(const Vec3& v, const Vec3& normal) {
    const val_t normSquared = deviceSquaredNorm(v);
    if (normSquared <= EPSILON * EPSILON) return ZERO;
    return fmaxf(ZERO, deviceDot(v, normal) / sqrtf(normSquared));
}

__device__ __forceinline__ val_t deviceRayTriangleIntersect(
    const Vec3& orig, const Vec3& dir, const Triangle& tri) {
    const Vec3 e1 = deviceSub(tri.b, tri.a);
    const Vec3 e2 = deviceSub(tri.c, tri.a);
    const Vec3 pvec = deviceCross(dir, e2);
    const val_t det = deviceDot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    const val_t invDet = 1.0f / det;
    const Vec3 tvec = deviceSub(orig, tri.a);
    const val_t u = deviceDot(tvec, pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;

    const Vec3 qvec = deviceCross(tvec, e1);
    const val_t v = deviceDot(dir, qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;

    return deviceDot(e2, qvec) * invDet;
}

__device__ __forceinline__ bool deviceRayIntersectsBox(
    const Vec3& p1, const Vec3& p2, const DeviceOctreeNode& node) {
    const Vec3 d = deviceScale(deviceSub(p2, p1), 0.5f);
    const Vec3 c = deviceSub(deviceAdd(p1, d), node.center);
    const Vec3 ad = Vec3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

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

__device__ bool deviceRayBlocked(
    const Vec3& from, const Vec3& to,
    const Triangle* triangles,
    const DeviceOctreeNode* nodes, const idx_t* leafIndices,
    idx_t srcTriIdx, idx_t dstTriIdx) {
    const Vec3 direction = deviceSub(to, from);
    const val_t rayLength = sqrtf(deviceSquaredNorm(direction));
    if (rayLength < EPSILON) return true;
    const Vec3 directionNormalized = deviceScale(direction, 1.0f / rayLength);

    constexpr int STACK_CAPACITY = 64;
    int stack[STACK_CAPACITY];
    int stackSize = 1;
    stack[0] = 0;

    while (stackSize != 0) {
        const int nodeIndex = stack[--stackSize];
        const DeviceOctreeNode& node = nodes[nodeIndex];
        if (!deviceRayIntersectsBox(from, to, node)) continue;

        if (node.leafCount != 0) {
            for (idx_t offset = 0; offset < node.leafCount; ++offset) {
                const idx_t triIndex = leafIndices[node.leafOffset + offset];
                if (triIndex == srcTriIdx || triIndex == dstTriIdx) continue;

                const val_t distance = deviceRayTriangleIntersect(
                    from, directionNormalized, triangles[triIndex]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }

        // Push in reverse so the pop order is the host octree's 0..7 order.
        for (int child = 7; child >= 0; --child) {
            const int childIndex = node.children[child];
            if (childIndex >= 0 && stackSize < STACK_CAPACITY) {
                stack[stackSize++] = childIndex;
            }
        }
    }
    return false;
}

__device__ __forceinline__ uint32_t deviceRandom(uint32_t& state) {
    // A small per-pair generator removes the serialized host RNG dependency
    // while retaining deterministic, repeatable Monte-Carlo samples.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ val_t deviceUniform(uint32_t& state) {
    return static_cast<val_t>(deviceRandom(state)) * (1.0f / 4294967296.0f);
}

__device__ Vec3 deviceRandomPoint(const Triangle& tri, uint32_t& state) {
    val_t u = deviceUniform(state);
    val_t v = deviceUniform(state);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return deviceAdd(tri.a, deviceAdd(deviceScale(deviceSub(tri.b, tri.a), u),
                                      deviceScale(deviceSub(tri.c, tri.a), v)));
}

__device__ __forceinline__ uint32_t devicePairSeed(idx_t i, idx_t j) {
    uint32_t state = 42u ^ (i + 1u) * 0x9e3779b9u ^ (j + 1u) * 0x85ebca6bu;
    state ^= state >> 16;
    state *= 0x7feb352du;
    state ^= state >> 15;
    state *= 0x846ca68bu;
    state ^= state >> 16;
    return state == 0 ? 0x6d2b79f5u : state;
}

__global__ void computeTimeDelaysKernel(
    const Triangle* triangles, int* tau, idx_t n) {
    const uint64_t linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(n) * n;
    if (linear >= total) return;

    const idx_t i = static_cast<idx_t>(linear / n);
    const idx_t j = static_cast<idx_t>(linear % n);
    if (i == j) return;

    const Vec3 delta = deviceSub(deviceTriangleCenter(triangles[i]),
                                  deviceTriangleCenter(triangles[j]));
    tau[linear] = static_cast<int>(ceilf(sqrtf(deviceSquaredNorm(delta)) * INV_WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(
    const Triangle* triangles, val_t* kij,
    const DeviceOctreeNode* nodes, const idx_t* leafIndices,
    idx_t n) {
    const idx_t j = static_cast<idx_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const idx_t i = static_cast<idx_t>(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= n || j >= n || i == j) return;

    const Triangle triI = triangles[i];
    const Triangle triJ = triangles[j];
    const size_t outputIndex = static_cast<size_t>(i) * n + j;

    if (deviceDot(triI._normal, triJ._normal) > 0.99f) {
        kij[outputIndex] = ZERO;
        return;
    }

    uint32_t randomState = devicePairSeed(i, j);
    val_t formFactor = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const Vec3 pI = deviceRandomPoint(triI, randomState);
        const Vec3 pJ = deviceRandomPoint(triJ, randomState);

        if (deviceRayBlocked(pI, pJ, triangles, nodes, leafIndices, i, j)) continue;

        const Vec3 v = deviceSub(pJ, pI);
        const val_t distanceSquared = deviceSquaredNorm(v);
        if (distanceSquared < EPSILON) continue;

        const val_t cosPhiI = deviceCosPhi(v, triI._normal);
        const Vec3 negativeV(-v.x, -v.y, -v.z);
        const val_t cosPhiJ = deviceCosPhi(negativeV, triJ._normal);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        formFactor += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
    }
    kij[outputIndex] = formFactor * INV_NUM_RAYS;
}

__global__ void propagateRadiosityKernel(
    idx_t timestep, idx_t n, const int* tau, const val_t* kij,
    const val_t* areas, const val_t* rho, const val_t* radE,
    val_t* radB) {
    const idx_t receiver = static_cast<idx_t>(blockIdx.x);
    if (receiver >= n) return;

    val_t sumB = ZERO;
    for (idx_t emitter = static_cast<idx_t>(threadIdx.x);
         emitter < n; emitter += static_cast<idx_t>(blockDim.x)) {
        if (receiver == emitter) continue;

        const size_t matrixIndex = static_cast<size_t>(receiver) * n + emitter;
        const int delay = tau[matrixIndex];
        if (timestep < static_cast<idx_t>(delay)) continue;

        const val_t formFactor = kij[matrixIndex];
        if (formFactor <= ZERO) continue;

        const idx_t sourceTime = timestep - static_cast<idx_t>(delay);
        const val_t sourceRadiosity = radB[static_cast<size_t>(sourceTime) * n + emitter];
        if (sourceRadiosity <= ZERO) continue;

        sumB += fminf(formFactor * areas[emitter], ONE) * sourceRadiosity;
    }

    __shared__ val_t warpSums[8];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        sumB += __shfl_down_sync(0xffffffffu, sumB, offset);
    }
    if (lane == 0) warpSums[warp] = sumB;
    __syncthreads();

    if (warp == 0) {
        sumB = lane < (blockDim.x >> 5) ? warpSums[lane] : ZERO;
        for (unsigned offset = 16; offset != 0; offset >>= 1) {
            sumB += __shfl_down_sync(0xffffffffu, sumB, offset);
        }
        if (threadIdx.x == 0) {
            const size_t outputIndex = static_cast<size_t>(timestep) * n + receiver;
            radB[outputIndex] = rho[receiver] * sumB + radE[outputIndex];
        }
    }
}

__global__ void computeDistancesKernel(
    const val_t* radB, val_t* distances, idx_t n, idx_t timesteps, idx_t sourceIndex) {
    const idx_t triangle = static_cast<idx_t>(blockIdx.x);
    if (triangle >= n) return;

    val_t bestCorrelation = ZERO;
    int bestLag = 0;
    for (idx_t lag = static_cast<idx_t>(threadIdx.x); lag < timesteps;
         lag += static_cast<idx_t>(blockDim.x)) {
        val_t sum = ZERO;
        for (idx_t time = lag; time < timesteps; ++time) {
            const val_t pB = radB[static_cast<size_t>(time) * n + triangle];
            const val_t pS = radB[static_cast<size_t>(time - lag) * n + sourceIndex];
            sum += pS * pB;
        }
        if (sum > bestCorrelation ||
            (sum == bestCorrelation && static_cast<int>(lag) < bestLag)) {
            bestCorrelation = sum;
            bestLag = static_cast<int>(lag);
        }
    }

    __shared__ val_t blockCorrelations[256];
    __shared__ int blockLags[256];
    blockCorrelations[threadIdx.x] = bestCorrelation;
    blockLags[threadIdx.x] = bestLag;
    __syncthreads();

    for (unsigned stride = blockDim.x >> 1; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t otherCorrelation = blockCorrelations[threadIdx.x + stride];
            const int otherLag = blockLags[threadIdx.x + stride];
            if (otherCorrelation > blockCorrelations[threadIdx.x] ||
                (otherCorrelation == blockCorrelations[threadIdx.x] &&
                 otherLag < blockLags[threadIdx.x])) {
                blockCorrelations[threadIdx.x] = otherCorrelation;
                blockLags[threadIdx.x] = otherLag;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distances[triangle] = WAVE_SPEED * static_cast<val_t>(blockLags[0]);
    }
}

void checkCuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                     file, line, expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define RS_CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

class CudaSimulation {
public:
    CudaSimulation(const SimulationState& state)
        : n_(static_cast<idx_t>(state.numTriangles)),
          timesteps_(static_cast<idx_t>(state.numTimesteps)),
          sourceIndex_(static_cast<idx_t>(state.sourceIndex)) {
        int deviceCount = 0;
        RS_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::fprintf(stderr, "CUDA initialization failed: no CUDA device is available\n");
            std::exit(EXIT_FAILURE);
        }
        RS_CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp properties{};
        RS_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::printf("CUDA device: %s (compute capability %d.%d)\n",
                    properties.name, properties.major, properties.minor);

        std::vector<DeviceOctreeNode> nodes;
        std::vector<idx_t> leafIndices;
        state.octree.flatten(nodes, leafIndices);
        if (nodes.empty()) {
            std::fprintf(stderr, "CUDA initialization failed: empty octree\n");
            std::exit(EXIT_FAILURE);
        }

        allocate(dTriangles_, state.triangles.size());
        allocate(dAreas_, state.areas.size());
        allocate(dRho_, state.rho.size());
        allocate(dKij_, state.kij.size());
        allocate(dTau_, state.tau.size());
        allocate(dRadE_, state.radE.size());
        allocate(dRadB_, state.radB.size());
        allocate(dDistances_, state.distances.size());
        allocate(dNodes_, nodes.size());
        allocate(dLeafIndices_, leafIndices.size());

        copyToDevice(dTriangles_, state.triangles);
        copyToDevice(dAreas_, state.areas);
        copyToDevice(dRho_, state.rho);
        copyToDevice(dKij_, state.kij);
        copyToDevice(dTau_, state.tau);
        copyToDevice(dRadE_, state.radE);
        copyToDevice(dRadB_, state.radB);
        copyToDevice(dNodes_, nodes);
        copyToDevice(dLeafIndices_, leafIndices);
        if (!state.distances.empty()) {
            RS_CUDA_CHECK(cudaMemset(dDistances_, 0,
                                     state.distances.size() * sizeof(val_t)));
        }
    }

    ~CudaSimulation() {
        cudaFree(dTriangles_);
        cudaFree(dAreas_);
        cudaFree(dRho_);
        cudaFree(dKij_);
        cudaFree(dTau_);
        cudaFree(dRadE_);
        cudaFree(dRadB_);
        cudaFree(dDistances_);
        cudaFree(dNodes_);
        cudaFree(dLeafIndices_);
    }

    void computeTimeDelays(SimulationState& state) {
        if (n_ == 0) return;
        const uint64_t total = static_cast<uint64_t>(n_) * n_;
        constexpr unsigned BLOCK_SIZE = 256;
        const unsigned gridSize = static_cast<unsigned>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        computeTimeDelaysKernel<<<gridSize, BLOCK_SIZE>>>(dTriangles_, dTau_, n_);
        RS_CUDA_CHECK(cudaGetLastError());
        RS_CUDA_CHECK(cudaDeviceSynchronize());
        copyToHost(state.tau, dTau_);
    }

    void computeFormFactors(SimulationState& state) {
        if (n_ == 0) return;
        const dim3 block(32, 8);
        const dim3 grid((n_ + block.x - 1) / block.x,
                        (n_ + block.y - 1) / block.y);
        computeFormFactorsKernel<<<grid, block>>>(
            dTriangles_, dKij_, dNodes_, dLeafIndices_, n_);
        RS_CUDA_CHECK(cudaGetLastError());
        RS_CUDA_CHECK(cudaDeviceSynchronize());
        copyToHost(state.kij, dKij_);
    }

    void runSimulation(SimulationState& state) {
        if (timesteps_ != 0) {
            constexpr unsigned BLOCK_SIZE = 256;
            for (idx_t timestep = 0; timestep < timesteps_; ++timestep) {
                propagateRadiosityKernel<<<n_, BLOCK_SIZE>>>(
                    timestep, n_, dTau_, dKij_, dAreas_, dRho_, dRadE_, dRadB_);
                RS_CUDA_CHECK(cudaGetLastError());
                if ((timestep + 1) % 10 == 0 || timestep + 1 == timesteps_) {
                    std::printf("  Timestep %u/%u\n", timestep + 1, timesteps_);
                }
            }
            RS_CUDA_CHECK(cudaDeviceSynchronize());
        }
        copyToHost(state.radB, dRadB_);
    }

    void computeDistances(SimulationState& state) {
        if (n_ != 0) {
            constexpr unsigned BLOCK_SIZE = 256;
            computeDistancesKernel<<<n_, BLOCK_SIZE>>>(
                dRadB_, dDistances_, n_, timesteps_, sourceIndex_);
            RS_CUDA_CHECK(cudaGetLastError());
            RS_CUDA_CHECK(cudaDeviceSynchronize());
        }
        copyToHost(state.distances, dDistances_);
    }

private:
    template<typename T>
    void allocate(T*& pointer, size_t count) {
        pointer = nullptr;
        if (count != 0) {
            RS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
        }
    }

    template<typename T>
    void copyToDevice(T* destination, const std::vector<T>& source) {
        if (!source.empty()) {
            RS_CUDA_CHECK(cudaMemcpy(destination, source.data(),
                                     source.size() * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    template<typename T>
    void copyToHost(std::vector<T>& destination, const T* source) {
        if (!destination.empty()) {
            RS_CUDA_CHECK(cudaMemcpy(destination.data(), source,
                                     destination.size() * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    idx_t n_;
    idx_t timesteps_;
    idx_t sourceIndex_;
    Triangle* dTriangles_ = nullptr;
    val_t* dAreas_ = nullptr;
    val_t* dRho_ = nullptr;
    val_t* dKij_ = nullptr;
    int* dTau_ = nullptr;
    val_t* dRadE_ = nullptr;
    val_t* dRadB_ = nullptr;
    val_t* dDistances_ = nullptr;
    DeviceOctreeNode* dNodes_ = nullptr;
    idx_t* dLeafIndices_ = nullptr;
};

SimulationState::~SimulationState() = default;

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

    // Allocate the complete device working set once.  All subsequent
    // numerical phases use these persistent buffers.
    state.cuda = std::make_unique<CudaSimulation>(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on CUDA...\n");
    state.cuda->computeFormFactors(state);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on CUDA...\n");
    state.cuda->computeTimeDelays(state);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on CUDA...\n");
    state.cuda->runSimulation(state);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on CUDA...\n");
    state.cuda->computeDistances(state);
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

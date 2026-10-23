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
#include <vector>

#include <mpi.h>
#include <omp.h>
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

HD inline val_t hd_sqrt(val_t v) { return sqrtf(v); }
HD inline val_t hd_abs(val_t v) { return v < 0.0f ? -v : v; }
HD inline val_t hd_max(val_t a, val_t b) { return a > b ? a : b; }

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
    HD val_t norm() const { return hd_sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return hd_abs(x - o.x) < EPSILON && hd_abs(y - o.y) < EPSILON && hd_abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    HD Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

// Flattened octree node used by the CUDA traversal kernels.
struct FlatOctree {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

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

    // Flatten this octree into contiguous arrays for GPU traversal.
    int flatten(std::vector<FlatOctree>& nodes, std::vector<int>& triIndices) const {
        int childIdx[8];
        for (int i = 0; i < 8; ++i) {
            childIdx[i] = children[i] ? children[i]->flatten(nodes, triIndices) : -1;
        }

        int index = static_cast<int>(nodes.size());
        FlatOctree fn;
        fn.center = center;
        fn.halfExtent = halfExtent;
        fn.triStart = static_cast<int>(triIndices.size());
        fn.triCount = static_cast<int>(triangleIndices.size());
        for (int i = 0; i < 8; ++i) fn.child[i] = childIdx[i];
        for (size_t idx : triangleIndices) triIndices.push_back(static_cast<int>(idx));
        nodes.push_back(fn);
        return index;
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
    uint64_t state;

    static HD uint64_t mix(uint64_t z) {
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
public:
    HD explicit RandomGenerator(uint32_t seed = 42) : state(mix(static_cast<uint64_t>(seed) + 1ULL)) {}
    HD val_t rand() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = mix(state);
        return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);
    }
};

// Generate a random point inside a triangle using barycentric coordinates
HD Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (hd_abs(det) < EPSILON) return 3.4028234663852886e38f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.4028234663852886e38f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.4028234663852886e38f;

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
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return hd_max(ZERO, v.dot(normal) / vNorm);
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
// CUDA Device Helpers and Kernels
// ============================================================================

HD inline bool rayIntersectsBoxFlat(const FlatOctree& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {hd_abs(d.x), hd_abs(d.y), hd_abs(d.z)};

    if (hd_abs(c.x) > node.halfExtent.x + ad.x) return false;
    if (hd_abs(c.y) > node.halfExtent.y + ad.y) return false;
    if (hd_abs(c.z) > node.halfExtent.z + ad.z) return false;

    if (hd_abs(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (hd_abs(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (hd_abs(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

HD inline bool isRayBlockedGpu(const Vec3& from, const Vec3& to,
                               const Triangle* __restrict__ triangles,
                               const FlatOctree* __restrict__ nodes,
                               const int* __restrict__ triIndices,
                               int root, int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        const FlatOctree& node = nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ci = node.child[c];
                if (ci >= 0 && sp < 128) {
                    const FlatOctree& child = nodes[ci];
                    if (rayIntersectsBoxFlat(child, from, to)) stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

__global__ void formFactorKernel(const Triangle* __restrict__ triangles,
                                 const FlatOctree* __restrict__ nodes,
                                 const int* __restrict__ triIndices,
                                 int root, val_t* __restrict__ kij,
                                 int numTriangles, int rowStart,
                                 long long localPairCount) {
    long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= localPairCount) return;

    int li = static_cast<int>(q / numTriangles);
    int j = static_cast<int>(q - static_cast<long long>(li) * numTriangles);
    int i = rowStart + li;
    if (i == j) return;

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    val_t* out = kij + static_cast<size_t>(li) * numTriangles + j;

    if (triI.normal().dot(triJ.normal()) > 0.99f) {
        *out = ZERO;
        return;
    }

    RandomGenerator rng(static_cast<uint32_t>(static_cast<uint64_t>(i) * numTriangles + j));
    val_t acc = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlockedGpu(pI, pJ, triangles, nodes, triIndices, root, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        acc += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    *out = acc * INV_NUM_RAYS;
}

__global__ void simulationStepKernel(const val_t* __restrict__ kij,
                                     const int* __restrict__ tau,
                                     const val_t* __restrict__ areas,
                                     const val_t* __restrict__ rho,
                                     val_t* __restrict__ radB,
                                     int numTriangles, int rowStart,
                                     int localRows, int t,
                                     int sourceIndex, int timeOff) {
    int li = blockIdx.x;
    if (li >= localRows) return;

    int i = rowStart + li;
    const val_t* krow = kij + static_cast<size_t>(li) * numTriangles;
    const int* trow = tau + static_cast<size_t>(li) * numTriangles;

    val_t sum = ZERO;
    for (int j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (j == i) continue;
        int tauij = trow[j];
        if (t < tauij) continue;
        val_t k = krow[j];
        if (k <= ZERO) continue;
        val_t radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= ZERO) continue;
        val_t contrib = k * areas[j];
        if (contrib > ONE) contrib = ONE;
        sum += contrib * radJ;
    }

    extern __shared__ val_t sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        val_t emission = (t < timeOff && i == sourceIndex) ? ONE : ZERO;
        radB[static_cast<size_t>(t) * numTriangles + i] = rho[i] * sdata[0] + emission;
    }
}

__global__ void distanceCorrKernel(const val_t* __restrict__ radB,
                                   val_t* __restrict__ corr,
                                   int numTriangles, int numTimesteps,
                                   int rowStart, int localRows,
                                   int sourceIndex) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= localRows || ty >= numTimesteps) return;

    int i = rowStart + tx;
    int t = ty;
    val_t sum = ZERO;
    const val_t* base = radB + i;
    const val_t* src = radB + sourceIndex;

    for (int tt = t; tt < numTimesteps; ++tt) {
        val_t pB = base[static_cast<size_t>(tt) * numTriangles];
        val_t pS = src[static_cast<size_t>(tt - t) * numTriangles];
        sum += pS * pB;
    }

    corr[static_cast<size_t>(tx) * numTimesteps + ty] = sum;
}

__global__ void distanceMaxKernel(const val_t* __restrict__ corr,
                                  val_t* __restrict__ distances,
                                  int numTimesteps,
                                  int localRows, val_t waveSpeed) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= localRows) return;

    const val_t* row = corr + static_cast<size_t>(li) * numTimesteps;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = row[t];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[li] = waveSpeed * static_cast<val_t>(bestT);
}

__global__ void countNonzeroKernel(const val_t* __restrict__ kij, long long total,
                                   int* __restrict__ result) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    int localCount = 0;
    for (; idx < total; idx += stride) {
        if (kij[idx] > EPSILON) ++localCount;
    }

    extern __shared__ int counts[];
    counts[threadIdx.x] = localCount;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) counts[threadIdx.x] += counts[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && counts[0] > 0) atomicAdd(result, counts[0]);
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

    int rank = 0;
    int worldSize = 1;
    size_t rowStart = 0;
    size_t rowEnd = 0;
    size_t localRows = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Local form-factor rows (localRows x N, row-major)
    std::vector<int> tau;           // Local time-delay rows (localRows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t nonZeroKij = 0;

    size_t idx2d(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// CUDA Runtime Management
// ============================================================================

struct GpuContext {
    bool active = false;
    int device = -1;

    Triangle* dTriangles = nullptr;
    FlatOctree* dNodes = nullptr;
    int* dTriIndices = nullptr;
    int root = 0;
    size_t numNodes = 0;
    size_t numTriIndices = 0;

    val_t* dKij = nullptr;
    int* dTau = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadB = nullptr;
    val_t* dCorr = nullptr;
    val_t* dDistances = nullptr;
    int* dCount = nullptr;
};

static bool initGpu(GpuContext& gpu, int localRank) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return false;

    int dev = localRank % count;
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    if (cudaFree(nullptr) != cudaSuccess) return false;

    gpu.active = true;
    gpu.device = dev;
    return true;
}

static bool setupGpu(GpuContext& gpu, SimulationState& state,
                     const std::vector<FlatOctree>& nodes,
                     const std::vector<int>& triIndices, int root) {
    if (!gpu.active) return false;

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t localPairs = state.localRows * N;

    gpu.root = root;
    gpu.numNodes = nodes.size();
    gpu.numTriIndices = triIndices.size();

    bool ok = true;
    if (N > 0) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dTriangles), N * sizeof(Triangle)) == cudaSuccess;
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dAreas), N * sizeof(val_t)) == cudaSuccess;
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dRho), N * sizeof(val_t)) == cudaSuccess;
    }
    if (N > 0 && T > 0) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dRadB), T * N * sizeof(val_t)) == cudaSuccess;
    }
    if (!nodes.empty()) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dNodes), nodes.size() * sizeof(FlatOctree)) == cudaSuccess;
    }
    if (!triIndices.empty()) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dTriIndices), triIndices.size() * sizeof(int)) == cudaSuccess;
    }
    if (localPairs > 0) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dKij), localPairs * sizeof(val_t)) == cudaSuccess;
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dTau), localPairs * sizeof(int)) == cudaSuccess;
    }
    if (state.localRows > 0 && T > 0) {
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dCorr), state.localRows * T * sizeof(val_t)) == cudaSuccess;
        ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dDistances), state.localRows * sizeof(val_t)) == cudaSuccess;
    }
    ok = ok && cudaMalloc(reinterpret_cast<void**>(&gpu.dCount), sizeof(int)) == cudaSuccess;

    if (!ok) return false;

    if (N > 0) {
        cudaMemcpy(gpu.dTriangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu.dAreas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu.dRho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice);
        if (T > 0) cudaMemset(gpu.dRadB, 0, T * N * sizeof(val_t));
    }
    if (!nodes.empty()) {
        cudaMemcpy(gpu.dNodes, nodes.data(), nodes.size() * sizeof(FlatOctree), cudaMemcpyHostToDevice);
    }
    if (!triIndices.empty()) {
        cudaMemcpy(gpu.dTriIndices, triIndices.data(), triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (localPairs > 0) {
        cudaMemset(gpu.dKij, 0, localPairs * sizeof(val_t));
    }

    return cudaGetLastError() == cudaSuccess;
}

static void freeGpu(GpuContext& gpu) {
    if (!gpu.active) return;
    cudaFree(gpu.dTriangles);
    cudaFree(gpu.dNodes);
    cudaFree(gpu.dTriIndices);
    cudaFree(gpu.dKij);
    cudaFree(gpu.dTau);
    cudaFree(gpu.dAreas);
    cudaFree(gpu.dRho);
    cudaFree(gpu.dRadB);
    cudaFree(gpu.dCorr);
    cudaFree(gpu.dDistances);
    cudaFree(gpu.dCount);
    gpu = GpuContext{};
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int worldSize) {
    state.rank = rank;
    state.worldSize = worldSize;

    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    const size_t N = state.numTriangles;
    state.rowStart = N * static_cast<size_t>(rank) / static_cast<size_t>(worldSize);
    state.rowEnd = N * static_cast<size_t>(rank + 1) / static_cast<size_t>(worldSize);
    state.localRows = state.rowEnd - state.rowStart;

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize local matrices and global fields
    state.kij.assign(state.localRows * N, ZERO);
    state.tau.assign(state.localRows * N, 0);
    state.radE.assign(timesteps * N, ZERO);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

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

void computeFormFactors(SimulationState& state, GpuContext& gpu) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    long long localPairs = static_cast<long long>(state.localRows) * static_cast<long long>(N);
    long long localCount = 0;

    if (gpu.active) {
        if (localPairs > 0) {
            int block = 256;
            long long blocks = (localPairs + block - 1) / block;
            formFactorKernel<<<static_cast<unsigned int>(blocks), block>>>(
                gpu.dTriangles, gpu.dNodes, gpu.dTriIndices, gpu.root,
                gpu.dKij, static_cast<int>(N), static_cast<int>(state.rowStart),
                localPairs);

            cudaMemset(gpu.dCount, 0, sizeof(int));
            unsigned int countBlocks = static_cast<unsigned int>(
                std::min<long long>(4096, blocks));
            if (countBlocks == 0) countBlocks = 1;
            countNonzeroKernel<<<countBlocks, 256, 256 * sizeof(int)>>>(
                gpu.dKij, localPairs, gpu.dCount);
            cudaMemcpy(&localCount, gpu.dCount, sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
    } else {
        #pragma omp parallel for schedule(dynamic, 1)
        for (long long li = 0; li < static_cast<long long>(state.localRows); ++li) {
            size_t i = state.rowStart + static_cast<size_t>(li);
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                RandomGenerator rng(static_cast<uint32_t>(i * N + j));
                state.kij[state.idx2d(static_cast<size_t>(li), j)] =
                    computeKij(i, j, state.triangles, state.octree, rng);
            }
        }
        for (size_t li = 0; li < state.localRows; ++li) {
            for (size_t j = 0; j < N; ++j) {
                if (state.kij[state.idx2d(li, j)] > EPSILON) ++localCount;
            }
        }
    }

    long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = static_cast<size_t>(globalCount);
}

void computeTimeDelays(SimulationState& state, GpuContext& gpu) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(state.localRows); ++li) {
        size_t i = state.rowStart + static_cast<size_t>(li);
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(static_cast<size_t>(li), j)] =
                computeTau(state.triangles[i], state.triangles[j]);
        }
    }

    if (gpu.active && state.localRows > 0) {
        cudaMemcpy(gpu.dTau, state.tau.data(),
                   state.localRows * N * sizeof(int), cudaMemcpyHostToDevice);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

static void rowPartition(const SimulationState& state, std::vector<int>& counts,
                         std::vector<int>& displs) {
    counts.resize(state.worldSize);
    displs.resize(state.worldSize);
    for (int r = 0; r < state.worldSize; ++r) {
        size_t begin = state.numTriangles * static_cast<size_t>(r) / static_cast<size_t>(state.worldSize);
        size_t end = state.numTriangles * static_cast<size_t>(r + 1) / static_cast<size_t>(state.worldSize);
        counts[r] = static_cast<int>(end - begin);
        displs[r] = static_cast<int>(begin);
    }
}

void runSimulation(SimulationState& state, GpuContext& gpu) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const int timeOff = static_cast<int>(T / 2);
    const int source = static_cast<int>(state.sourceIndex);

    std::vector<int> counts, displs;
    rowPartition(state, counts, displs);

    if (gpu.active && N > 0 && T > 0) {
        cudaMemset(gpu.dRadB, 0, T * N * sizeof(val_t));
    }

    const int block = 256;
    for (size_t t = 0; t < T; ++t) {
        if (gpu.active) {
            if (state.localRows > 0) {
                simulationStepKernel<<<static_cast<unsigned int>(state.localRows), block,
                                       block * sizeof(val_t)>>>(
                    gpu.dKij, gpu.dTau, gpu.dAreas, gpu.dRho, gpu.dRadB,
                    static_cast<int>(N), static_cast<int>(state.rowStart),
                    static_cast<int>(state.localRows), static_cast<int>(t), source, timeOff);
                cudaMemcpy(state.radB.data() + state.idxTN(t, state.rowStart),
                           gpu.dRadB + state.idxTN(t, state.rowStart),
                           state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (long long li = 0; li < static_cast<long long>(state.localRows); ++li) {
                size_t i = state.rowStart + static_cast<size_t>(li);
                val_t sumB = ZERO;
                const size_t row = state.idx2d(static_cast<size_t>(li), 0);
                for (size_t j = 0; j < N; ++j) {
                    if (i == j) continue;
                    int tauij = state.tau[row + j];
                    if (static_cast<int>(t) < tauij) continue;
                    val_t kij = state.kij[row + j];
                    if (kij <= ZERO) continue;
                    size_t srcTime = t - static_cast<size_t>(tauij);
                    val_t radJ = state.radB[state.idxTN(srcTime, j)];
                    if (radJ <= ZERO) continue;
                    sumB += std::min(kij * state.areas[j], ONE) * radJ;
                }
                val_t emission = (static_cast<int>(t) < timeOff && i == state.sourceIndex) ? ONE : ZERO;
                state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + emission;
            }
        }

        MPI_Allgatherv(MPI_IN_PLACE, counts[state.rank], MPI_FLOAT,
                       state.radB.data() + state.idxTN(t, 0),
                       counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if (gpu.active && N > 0) {
            cudaMemcpy(gpu.dRadB + state.idxTN(t, 0),
                       state.radB.data() + state.idxTN(t, 0),
                       N * sizeof(val_t), cudaMemcpyHostToDevice);
        }

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuContext& gpu) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    std::vector<int> counts, displs;
    rowPartition(state, counts, displs);

    if (gpu.active) {
        if (state.localRows > 0 && T > 0) {
            dim3 block(32, 8);
            dim3 grid(static_cast<unsigned int>((state.localRows + 31) / 32),
                      static_cast<unsigned int>((T + 7) / 8));
            distanceCorrKernel<<<grid, block>>>(
                gpu.dRadB, gpu.dCorr, static_cast<int>(N), static_cast<int>(T),
                static_cast<int>(state.rowStart), static_cast<int>(state.localRows),
                static_cast<int>(state.sourceIndex));

            int threads = 256;
            unsigned int blocks = static_cast<unsigned int>((state.localRows + threads - 1) / threads);
            distanceMaxKernel<<<blocks, threads>>>(
                gpu.dCorr, gpu.dDistances, static_cast<int>(T),
                static_cast<int>(state.localRows), WAVE_SPEED);

            cudaMemcpy(state.distances.data() + state.rowStart, gpu.dDistances,
                       state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
    } else {
        #pragma omp parallel for schedule(static)
        for (long long li = 0; li < static_cast<long long>(state.localRows); ++li) {
            size_t i = state.rowStart + static_cast<size_t>(li);
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

            state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, counts[state.rank], MPI_FLOAT,
                   state.distances.data(), counts.data(), displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = 3.4028234663852886e38f;
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
    size_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * static_cast<double>(nonZeroKij) / static_cast<double>(state.numTriangles * state.numTriangles));

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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int localRank = 0;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &localRank);

    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Comm_free(&nodeComm);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&nodeComm);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (isRoot) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", worldSize, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, worldSize);

    // Flatten the octree and initialize the per-rank CUDA device.
    std::vector<FlatOctree> flatNodes;
    std::vector<int> flatTriIndices;
    int rootNode = state.octree.flatten(flatNodes, flatTriIndices);

    GpuContext gpu;
    bool gpuReady = initGpu(gpu, localRank);
    if (gpuReady) {
        gpuReady = setupGpu(gpu, state, flatNodes, flatTriIndices, rootNode);
        if (!gpuReady) freeGpu(gpu);
    }
    if (isRoot) {
        if (gpuReady) {
            printf("CUDA device: %d (rank-local)\n", gpu.device);
        } else {
            printf("CUDA device unavailable; falling back to MPI+OpenMP execution\n");
        }
        printf("\n");
    }

    const bool useGpu = gpuReady;

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, gpu);
    computeFormFactors(state, gpu);
    if (useGpu) cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (isRoot) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, gpu);
    if (useGpu) cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (isRoot) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, gpu);
    if (useGpu) cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (isRoot) {
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

        // Memory usage (distributed form-factor/tau rows)
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

    }

    if (validate) {
        int localOk = 1;
        if (isRoot) localOk = validateResults(state) ? 1 : 0;
        int globalOk = 1;
        MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!globalOk) {
            freeGpu(gpu);
            MPI_Comm_free(&nodeComm);
            MPI_Finalize();
            return 1;
        }
    }

    freeGpu(gpu);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return 0;
}

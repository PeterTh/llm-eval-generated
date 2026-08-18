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

#define CUDA_HD __host__ __device__

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

    CUDA_HD constexpr Vec3() : x(0), y(0), z(0) {}
    CUDA_HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    CUDA_HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

// A compact, pointer-free representation used by the CUDA visibility kernel.
struct FlatOctree {
    std::vector<Vec3> centers, halfExtents;
    std::vector<int> child;       // eight consecutive child indices per node
    std::vector<int> leafBegin;
    std::vector<int> leafCount;
    std::vector<int> triangleIndices;

    int flatten(const Octree& node) {
        const int id = static_cast<int>(centers.size());
        centers.push_back(node.center);
        halfExtents.push_back(node.halfExtent);
        child.resize(child.size() + 8, -1);
        leafBegin.push_back(static_cast<int>(triangleIndices.size()));
        leafCount.push_back(static_cast<int>(node.triangleIndices.size()));
        triangleIndices.insert(triangleIndices.end(), node.triangleIndices.begin(), node.triangleIndices.end());
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) child[id * 8 + c] = flatten(*node.children[c]);
        }
        return id;
    }

    explicit FlatOctree(const Octree& root) { flatten(root); }
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
// CUDA kernels
// ============================================================================

__device__ inline Vec3 dsub(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline Vec3 dcross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ inline float ddot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ inline float dnorm(const Vec3& a) { return sqrtf(ddot(a, a)); }
__device__ inline float dabs(float x) { return fabsf(x); }

__device__ inline float pairRandom(unsigned int& seed) {
    // Stateless-per-pair streams make every (i,j) independent and reproducible.
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return (seed & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ bool dRayBox(const Vec3& p1, const Vec3& p2, const Vec3& center, const Vec3& half) {
    Vec3 d = dsub(p2, p1);
    d.x *= .5f; d.y *= .5f; d.z *= .5f;
    Vec3 c = Vec3(p1.x + d.x - center.x, p1.y + d.y - center.y, p1.z + d.z - center.z);
    Vec3 ad = Vec3(dabs(d.x), dabs(d.y), dabs(d.z));
    if (dabs(c.x) > half.x + ad.x || dabs(c.y) > half.y + ad.y || dabs(c.z) > half.z + ad.z) return false;
    if (dabs(d.y*c.z-d.z*c.y) > half.y*ad.z+half.z*ad.y+EPSILON) return false;
    if (dabs(d.z*c.x-d.x*c.z) > half.z*ad.x+half.x*ad.z+EPSILON) return false;
    if (dabs(d.x*c.y-d.y*c.x) > half.x*ad.y+half.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool dBlocked(const Vec3& from, const Vec3& to, int src, int dst,
                         const Triangle* tris, const Vec3* centers, const Vec3* halves,
                         const int* children, const int* leafBegin, const int* leafCount,
                         const int* leafTris, int nodeCount) {
    Vec3 dir = dsub(to, from);
    float rayLen = dnorm(dir);
    if (rayLen < EPSILON) return true;
    Vec3 inv = Vec3(dir.x/rayLen, dir.y/rayLen, dir.z/rayLen);
    int stack[64]; int top = 0; stack[top++] = 0;
    while (top) {
        int node = stack[--top];
        if (!dRayBox(from, to, centers[node], halves[node])) continue;
        int count = leafCount[node];
        for (int k = 0; k < count; ++k) {
            int idx = leafTris[leafBegin[node] + k];
            if (idx == src || idx == dst) continue;
            const Triangle& tri = tris[idx];
            Vec3 e1 = dsub(tri.b, tri.a), e2 = dsub(tri.c, tri.a);
            Vec3 p = dcross(inv, e2);
            float det = ddot(e1, p);
            if (dabs(det) < EPSILON) continue;
            float invDet = 1.0f / det;
            Vec3 tv = dsub(from, tri.a);
            float u = ddot(tv, p) * invDet;
            if (u < 0.0f || u > 1.0f) continue;
            Vec3 q = dcross(tv, e1);
            float v = ddot(inv, q) * invDet;
            if (v < 0.0f || u + v > 1.0f) continue;
            float dist = ddot(e2, q) * invDet;
            if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        }
        for (int c = 0; c < 8 && top < 64; ++c) {
            int child = children[node*8+c];
            if (child >= 0) stack[top++] = child;
        }
    }
    return false;
}

__global__ void formFactorKernel(const Triangle* tris, int n, const Vec3* centers,
                                 const Vec3* halves, const int* children,
                                 const int* leafBegin, const int* leafCount,
                                 const int* leafTris, int nodeCount, float* kij) {
    size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * n;
    if (pair >= total) return;
    int i = static_cast<int>(pair / n), j = static_cast<int>(pair % n);
    if (i == j || ddot(tris[i]._normal, tris[j]._normal) > 0.99f) { kij[pair] = 0.0f; return; }
    unsigned int seed = 0x9e3779b9u ^ (static_cast<unsigned int>(i) * 0x85ebca6bu)
                      ^ (static_cast<unsigned int>(j) * 0xc2b2ae35u);
    float result = 0.0f;
    const Triangle& a = tris[i]; const Triangle& b = tris[j];
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u = pairRandom(seed), v = pairRandom(seed);
        if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
        Vec3 pI = Vec3(a.a.x+(a.b.x-a.a.x)*u+(a.c.x-a.a.x)*v,
                       a.a.y+(a.b.y-a.a.y)*u+(a.c.y-a.a.y)*v,
                       a.a.z+(a.b.z-a.a.z)*u+(a.c.z-a.a.z)*v);
        u = pairRandom(seed); v = pairRandom(seed);
        if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
        Vec3 pJ = Vec3(b.a.x+(b.b.x-b.a.x)*u+(b.c.x-b.a.x)*v,
                       b.a.y+(b.b.y-b.a.y)*u+(b.c.y-b.a.y)*v,
                       b.a.z+(b.b.z-b.a.z)*u+(b.c.z-b.a.z)*v);
        if (dBlocked(pI, pJ, i, j, tris, centers, halves, children, leafBegin, leafCount, leafTris, nodeCount)) continue;
        Vec3 ray = dsub(pJ, pI); float d2 = ddot(ray, ray); if (d2 < EPSILON) continue;
        float len = sqrtf(d2);
        float ci = fmaxf(0.0f, ddot(ray, a._normal)/len);
        float cj = fmaxf(0.0f, -ddot(ray, b._normal)/len);
        if (ci > 0.0f && cj > 0.0f) result += ci*cj/(PI*d2);
    }
    kij[pair] = result * INV_NUM_RAYS;
}

__global__ void tauKernel(const Triangle* tris, int n, int* tau) {
    size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= static_cast<size_t>(n)*n) return;
    int i = static_cast<int>(pair/n), j = static_cast<int>(pair%n);
    if (i == j) { tau[pair] = 0; return; }
    Vec3 ci = Vec3((tris[i].a.x+tris[i].b.x+tris[i].c.x)/3.0f,
                   (tris[i].a.y+tris[i].b.y+tris[i].c.y)/3.0f,
                   (tris[i].a.z+tris[i].b.z+tris[i].c.z)/3.0f);
    Vec3 cj = Vec3((tris[j].a.x+tris[j].b.x+tris[j].c.x)/3.0f,
                   (tris[j].a.y+tris[j].b.y+tris[j].c.y)/3.0f,
                   (tris[j].a.z+tris[j].b.z+tris[j].c.z)/3.0f);
    tau[pair] = static_cast<int>(ceilf(dnorm(dsub(ci,cj))*INV_WAVE_SPEED));
}

__global__ void propagationKernel(int t, int n, const float* kij, const int* tau,
                                  const float* areas, const float* rho, const float* radE,
                                  float* radB) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[static_cast<size_t>(i)*n+j];
        if (t < delay) continue;
        float k = kij[static_cast<size_t>(i)*n+j];
        if (k <= 0.0f) continue;
        float source = radB[static_cast<size_t>(t-delay)*n+j];
        if (source > 0.0f) sum += fminf(k*areas[j], 1.0f)*source;
    }
    radB[static_cast<size_t>(t)*n+i] = rho[i]*sum + radE[static_cast<size_t>(t)*n+i];
}

__global__ void distanceKernel(int n, int timesteps, int source, const float* radB, float* distances) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float maxCorr = 0.0f; int best = 0;
    for (int t = 0; t < timesteps; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < timesteps; ++tt)
            sum += radB[static_cast<size_t>(tt)*n+i] * radB[static_cast<size_t>(tt-t)*n+source];
        if (sum > maxCorr) { maxCorr = sum; best = t; }
    }
    distances[i] = WAVE_SPEED * best;
}

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
static T* cudaUpload(const std::vector<T>& host) {
    T* device = nullptr;
    cudaCheck(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
    if (!host.empty()) cudaCheck(cudaMemcpy(device, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

static void cudaFreeChecked(void* ptr) { if (ptr) cudaCheck(cudaFree(ptr), "cudaFree"); }

static void uploadOctree(const FlatOctree& tree, Vec3*& centers, Vec3*& halves,
                         int*& children, int*& leafBegin, int*& leafCount, int*& leafTris) {
    centers = cudaUpload(tree.centers); halves = cudaUpload(tree.halfExtents);
    children = cudaUpload(tree.child); leafBegin = cudaUpload(tree.leafBegin);
    leafCount = cudaUpload(tree.leafCount); leafTris = cudaUpload(tree.triangleIndices);
}

static void freeOctree(Vec3* centers, Vec3* halves, int* children, int* leafBegin,
                       int* leafCount, int* leafTris) {
    cudaFreeChecked(centers); cudaFreeChecked(halves); cudaFreeChecked(children);
    cudaFreeChecked(leafBegin); cudaFreeChecked(leafCount); cudaFreeChecked(leafTris);
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
    printf("Computing form factors (Kij)...\n");
    FlatOctree tree(state.octree);
    Triangle* dTris = cudaUpload(state.triangles);
    Vec3 *dCenters, *dHalves; int *dChildren, *dLeafBegin, *dLeafCount, *dLeafTris;
    uploadOctree(tree, dCenters, dHalves, dChildren, dLeafBegin, dLeafCount, dLeafTris);
    float* dKij = cudaUpload(state.kij);
    size_t total = state.numTriangles * state.numTriangles;
    formFactorKernel<<<(total + 255) / 256, 256>>>(dTris, static_cast<int>(state.numTriangles),
        dCenters, dHalves, dChildren, dLeafBegin, dLeafCount, dLeafTris,
        static_cast<int>(tree.centers.size()), dKij);
    cudaCheck(cudaGetLastError(), "formFactorKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "formFactorKernel");
    cudaCheck(cudaMemcpy(state.kij.data(), dKij, total*sizeof(float), cudaMemcpyDeviceToHost), "Kij D2H");
    cudaFreeChecked(dKij); cudaFreeChecked(dTris);
    freeOctree(dCenters, dHalves, dChildren, dLeafBegin, dLeafCount, dLeafTris);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    Triangle* dTris = cudaUpload(state.triangles);
    int* dTau = cudaUpload(state.tau);
    size_t total = state.numTriangles * state.numTriangles;
    tauKernel<<<(total + 255) / 256, 256>>>(dTris, static_cast<int>(state.numTriangles), dTau);
    cudaCheck(cudaGetLastError(), "tauKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "tauKernel");
    cudaCheck(cudaMemcpy(state.tau.data(), dTau, total*sizeof(int), cudaMemcpyDeviceToHost), "Tau D2H");
    cudaFreeChecked(dTris); cudaFreeChecked(dTau);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    float *dKij = cudaUpload(state.kij), *dAreas = cudaUpload(state.areas);
    float *dRho = cudaUpload(state.rho), *dRadE = cudaUpload(state.radE), *dRadB = cudaUpload(state.radB);
    int* dTau = cudaUpload(state.tau);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<(state.numTriangles + 255) / 256, 256>>>(static_cast<int>(t), static_cast<int>(state.numTriangles),
            dKij, dTau, dAreas, dRho, dRadE, dRadB);
        cudaCheck(cudaGetLastError(), "propagationKernel launch");
        cudaCheck(cudaDeviceSynchronize(), "propagationKernel");

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaMemcpy(state.radB.data(), dRadB, state.radB.size()*sizeof(float), cudaMemcpyDeviceToHost), "radB D2H");
    cudaFreeChecked(dKij); cudaFreeChecked(dAreas); cudaFreeChecked(dRho); cudaFreeChecked(dRadE);
    cudaFreeChecked(dRadB); cudaFreeChecked(dTau);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    float* dRadB = cudaUpload(state.radB);
    float* dDistances = cudaUpload(state.distances);
    distanceKernel<<<(state.numTriangles + 255) / 256, 256>>>(static_cast<int>(state.numTriangles),
        static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex), dRadB, dDistances);
    cudaCheck(cudaGetLastError(), "distanceKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "distanceKernel");
    cudaCheck(cudaMemcpy(state.distances.data(), dDistances, state.distances.size()*sizeof(float), cudaMemcpyDeviceToHost), "distances D2H");
    cudaFreeChecked(dRadB); cudaFreeChecked(dDistances);
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

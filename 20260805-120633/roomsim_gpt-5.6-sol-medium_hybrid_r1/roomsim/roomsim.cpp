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

#define HD __host__ __device__

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure in %s: %s\n", operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

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

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// Pointer-free representation used by CUDA visibility traversal.
struct FlatOctreeNode {
    Vec3 center, halfExtent;
    int children[8];
    uint32_t firstTriangle;
    uint32_t triangleCount;
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

    int flatten(std::vector<FlatOctreeNode>& nodes,
                std::vector<uint32_t>& indices) const {
        const int nodeIndex = static_cast<int>(nodes.size());
        nodes.emplace_back();
        FlatOctreeNode node{};
        node.center = center;
        node.halfExtent = halfExtent;
        std::fill(std::begin(node.children), std::end(node.children), -1);
        node.firstTriangle = static_cast<uint32_t>(indices.size());
        node.triangleCount = static_cast<uint32_t>(triangleIndices.size());
        for (size_t index : triangleIndices) {
            indices.push_back(static_cast<uint32_t>(index));
        }
        nodes[nodeIndex] = node;
        for (int child = 0; child < 8; ++child) {
            if (children[child]) {
                const int flatChild = children[child]->flatten(nodes, indices);
                nodes[nodeIndex].children[child] = flatChild;
            }
        }
        return nodeIndex;
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
// CUDA kernels
// ============================================================================

__device__ uint32_t mixBits(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

__device__ val_t sample01(uint32_t i, uint32_t j, uint32_t ray, uint32_t lane) {
    uint32_t x = 42U ^ (i * 0x9e3779b9U) ^ (j * 0x85ebca6bU) ^
                 (ray * 0xc2b2ae35U) ^ (lane * 0x27d4eb2fU);
    return static_cast<val_t>(mixBits(x) >> 8) * (1.0f / 16777216.0f);
}

__device__ Vec3 sampledPoint(const Triangle& triangle, uint32_t i, uint32_t j,
                             uint32_t ray, uint32_t lane) {
    val_t u = sample01(i, j, ray, lane);
    val_t v = sample01(i, j, ray, lane + 1U);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u +
           (triangle.c - triangle.a) * v;
}

__device__ bool segmentIntersectsBox(const Vec3& p1, const Vec3& p2,
                                     const FlatOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    return fabsf(d.x * c.y - d.y * c.x) <=
           node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON;
}

__device__ val_t deviceRayTriangle(const Vec3& origin, const Vec3& direction,
                                   const Triangle& triangle) {
    Vec3 e1 = triangle.b - triangle.a;
    Vec3 e2 = triangle.c - triangle.a;
    Vec3 p = direction.cross(e2);
    val_t det = e1.dot(p);
    if (fabsf(det) < EPSILON) return 3.402823466e+38F;
    val_t invDet = 1.0f / det;
    Vec3 tv = origin - triangle.a;
    val_t u = tv.dot(p) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38F;
    Vec3 q = tv.cross(e1);
    val_t v = direction.dot(q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e+38F;
    return e2.dot(q) * invDet;
}

__device__ bool rayBlockedDevice(const Vec3& from, const Vec3& to,
                                 const Triangle* triangles,
                                 const FlatOctreeNode* nodes,
                                 const uint32_t* indices,
                                 uint32_t triangleCount, uint32_t src, uint32_t dst) {
    Vec3 delta = to - from;
    val_t rayLength = delta.norm();
    if (rayLength < EPSILON) return true;
    Vec3 direction = delta / rayLength;
    // The generated octree is shallow; the guarded stack avoids recursion and
    // falls back to a complete scan if a future mesh produces excessive fanout.
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top) {
        const FlatOctreeNode& node = nodes[stack[--top]];
        if (node.triangleCount) {
            for (uint32_t k = 0; k < node.triangleCount; ++k) {
                uint32_t index = indices[node.firstTriangle + k];
                if (index == src || index == dst) continue;
                val_t distance = deviceRayTriangle(from, direction, triangles[index]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
        } else {
            for (int child = 7; child >= 0; --child) {
                int childIndex = node.children[child];
                if (childIndex >= 0 && segmentIntersectsBox(from, to, nodes[childIndex])) {
                    if (top == 64) {
                        // Correctness-preserving rare overflow path.
                        for (uint32_t index = 0; index < triangleCount; ++index) {
                            if (index == src || index == dst) continue;
                            val_t distance = deviceRayTriangle(from, direction, triangles[index]);
                            if (distance > EPSILON && distance < rayLength - EPSILON) return true;
                        }
                        return false;
                    }
                    stack[top++] = childIndex;
                }
            }
        }
    }
    return false;
}

__global__ void precomputeKernel(const Triangle* triangles,
                                 const FlatOctreeNode* nodes,
                                 const uint32_t* indices, size_t n,
                                 size_t rowStart, size_t rowCount,
                                 val_t* kij, int* tau) {
    size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = rowCount * n;
    if (linear >= total) return;
    size_t localI = linear / n;
    size_t i = rowStart + localI;
    size_t j = linear - localI * n;
    if (i == j) {
        kij[linear] = ZERO;
        tau[linear] = 0;
        return;
    }
    Vec3 centers = triangles[i].center() - triangles[j].center();
    tau[linear] = static_cast<int>(ceilf(centers.norm() * INV_WAVE_SPEED));
    if (triangles[i].normal().dot(triangles[j].normal()) > 0.99f) {
        kij[linear] = ZERO;
        return;
    }
    val_t factor = ZERO;
    for (uint32_t ray = 0; ray < NUM_RAYS; ++ray) {
        Vec3 pi = sampledPoint(triangles[i], static_cast<uint32_t>(i),
                               static_cast<uint32_t>(j), ray, 0);
        Vec3 pj = sampledPoint(triangles[j], static_cast<uint32_t>(i),
                               static_cast<uint32_t>(j), ray, 2);
        if (rayBlockedDevice(pi, pj, triangles, nodes, indices, static_cast<uint32_t>(n),
                             static_cast<uint32_t>(i), static_cast<uint32_t>(j))) continue;
        Vec3 v = pj - pi;
        val_t distance2 = v.squaredNorm();
        if (distance2 < EPSILON) continue;
        val_t inverseLength = rsqrtf(distance2);
        val_t ci = fmaxf(ZERO, v.dot(triangles[i].normal()) * inverseLength);
        val_t cj = fmaxf(ZERO, (-v).dot(triangles[j].normal()) * inverseLength);
        if (ci > ZERO && cj > ZERO) factor += ci * cj / (PI * distance2);
    }
    kij[linear] = factor * INV_NUM_RAYS;
}

__global__ void propagationKernel(size_t n, size_t rowStart, size_t rowCount,
                                  size_t timestep, const val_t* kij,
                                  const int* tau, const val_t* areas,
                                  const val_t* rho, const val_t* allRadiosity,
                                  size_t source, size_t sourceOff, val_t* current) {
    size_t localI = blockIdx.x;
    if (localI >= rowCount) return;
    size_t i = rowStart + localI;
    val_t sum = ZERO;
    size_t row = localI * n;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        int delay = tau[row + j];
        val_t factor = kij[row + j];
        if (i == j || timestep < static_cast<size_t>(delay) || factor <= ZERO) continue;
        val_t incoming = allRadiosity[(timestep - static_cast<size_t>(delay)) * n + j];
        if (incoming > ZERO) sum += fminf(factor * areas[j], ONE) * incoming;
    }
    __shared__ val_t partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        current[localI] = rho[i] * partial[0] +
                          ((i == source && timestep < sourceOff) ? ONE : ZERO);
}

__global__ void distanceKernel(size_t n, size_t timesteps, size_t rowStart,
                               size_t rowCount, size_t source,
                               const val_t* radiosity, val_t* distances) {
    size_t localI = blockIdx.x;
    if (localI >= rowCount) return;
    size_t i = rowStart + localI;
    val_t threadBest = ZERO;
    size_t threadBestT = 0;
    for (size_t t = threadIdx.x; t < timesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt) {
            sum += radiosity[tt * n + i] * radiosity[(tt - t) * n + source];
        }
        if (sum > threadBest) {
            threadBest = sum;
            threadBestT = t;
        }
    }
    __shared__ val_t correlations[256];
    __shared__ size_t delays[256];
    correlations[threadIdx.x] = threadBest;
    delays[threadIdx.x] = threadBestT;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            val_t candidate = correlations[threadIdx.x + stride];
            size_t candidateT = delays[threadIdx.x + stride];
            if (candidate > correlations[threadIdx.x] ||
                (candidate == correlations[threadIdx.x] && candidateT < delays[threadIdx.x])) {
                correlations[threadIdx.x] = candidate;
                delays[threadIdx.x] = candidateT;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        distances[localI] = WAVE_SPEED * static_cast<val_t>(delays[0]);
}

__global__ void countFormFactorsKernel(const val_t* kij, size_t count,
                                       unsigned long long* nonzero) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int local = index < count && kij[index] > EPSILON;
    // Aggregate within a block to keep validation overhead negligible.
    __shared__ unsigned int blockCount[256];
    blockCount[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) blockCount[threadIdx.x] += blockCount[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(nonzero, static_cast<unsigned long long>(blockCount[0]));
}

__global__ void countEnergizedKernel(const val_t* radiosity, size_t n,
                                     size_t timesteps, unsigned int* energized) {
    size_t triangle = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (triangle >= n) return;
    for (size_t t = 0; t < timesteps; ++t) {
        if (radiosity[t * n + triangle] > EPSILON) {
            atomicAdd(energized, 1U);
            return;
        }
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
    std::vector<val_t> distances;   // Computed distances from source

    int rank = 0, ranks = 1;
    size_t localStart = 0, localCount = 0;
    std::vector<int> rowCounts, rowDisplacements;
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<uint32_t> flatIndices;

    Triangle* dTriangles = nullptr;
    FlatOctreeNode* dNodes = nullptr;
    uint32_t* dIndices = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dKij = nullptr;
    int* dTau = nullptr;
    val_t* dRadB = nullptr;
    val_t* dCurrent = nullptr;
    val_t* dDistances = nullptr;
    val_t* hostCurrent = nullptr;
    val_t* hostLocal = nullptr;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

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

    if (state.rank == 0)
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    state.rowCounts.resize(state.ranks);
    state.rowDisplacements.resize(state.ranks);
    const size_t base = state.numTriangles / static_cast<size_t>(state.ranks);
    const size_t remainder = state.numTriangles % static_cast<size_t>(state.ranks);
    size_t offset = 0;
    for (int rank = 0; rank < state.ranks; ++rank) {
        size_t count = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        state.rowCounts[rank] = static_cast<int>(count);
        state.rowDisplacements[rank] = static_cast<int>(offset);
        if (rank == state.rank) {
            state.localStart = offset;
            state.localCount = count;
        }
        offset += count;
    }

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.octree.flatten(state.flatNodes, state.flatIndices);

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    // Kij and Tau stay distributed in GPU memory; retaining full host copies
    // would defeat MPI memory scaling at the largest mesh levels.
    state.distances.resize(state.numTriangles, ZERO);

    // Source emission is generated directly by the propagation kernel for the
    // first half of the timesteps, avoiding another replicated T-by-N array.
}

void allocateDeviceState(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t localMatrix = state.localCount * n;
    cudaCheck(cudaMalloc(&state.dTriangles, n * sizeof(Triangle)), "triangle allocation");
    cudaCheck(cudaMalloc(&state.dNodes, state.flatNodes.size() * sizeof(FlatOctreeNode)), "octree allocation");
    cudaCheck(cudaMalloc(&state.dIndices, state.flatIndices.size() * sizeof(uint32_t)), "octree-index allocation");
    cudaCheck(cudaMalloc(&state.dAreas, n * sizeof(val_t)), "area allocation");
    cudaCheck(cudaMalloc(&state.dRho, n * sizeof(val_t)), "reflectivity allocation");
    cudaCheck(cudaMalloc(&state.dKij, localMatrix * sizeof(val_t)), "form-factor allocation");
    cudaCheck(cudaMalloc(&state.dTau, localMatrix * sizeof(int)), "delay allocation");
    cudaCheck(cudaMalloc(&state.dRadB, state.numTimesteps * n * sizeof(val_t)), "radiosity allocation");
    cudaCheck(cudaMalloc(&state.dCurrent, state.localCount * sizeof(val_t)), "current-row allocation");
    cudaCheck(cudaMalloc(&state.dDistances, state.localCount * sizeof(val_t)), "distance allocation");
    cudaCheck(cudaHostAlloc(&state.hostCurrent, n * sizeof(val_t), cudaHostAllocPortable), "MPI receive staging allocation");
    cudaCheck(cudaHostAlloc(&state.hostLocal, state.localCount * sizeof(val_t), cudaHostAllocPortable), "MPI send staging allocation");
    cudaCheck(cudaMemcpy(state.dTriangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "triangle upload");
    cudaCheck(cudaMemcpy(state.dNodes, state.flatNodes.data(), state.flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice), "octree upload");
    cudaCheck(cudaMemcpy(state.dIndices, state.flatIndices.data(), state.flatIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "octree-index upload");
    cudaCheck(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "area upload");
    cudaCheck(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "reflectivity upload");
    cudaCheck(cudaMemset(state.dRadB, 0, state.numTimesteps * n * sizeof(val_t)), "radiosity initialization");
}

void releaseDeviceState(SimulationState& state) {
    cudaFree(state.dTriangles); cudaFree(state.dNodes); cudaFree(state.dIndices);
    cudaFree(state.dAreas); cudaFree(state.dRho); cudaFree(state.dKij);
    cudaFree(state.dTau); cudaFree(state.dRadB); cudaFree(state.dCurrent);
    cudaFree(state.dDistances); cudaFreeHost(state.hostCurrent); cudaFreeHost(state.hostLocal);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    const size_t work = state.localCount * state.numTriangles;
    constexpr int blockSize = 128;
    precomputeKernel<<<static_cast<unsigned>((work + blockSize - 1) / blockSize), blockSize>>>(
        state.dTriangles, state.dNodes, state.dIndices, state.numTriangles,
        state.localStart, state.localCount, state.dKij, state.dTau);
    cudaCheck(cudaGetLastError(), "precomputation kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "precomputation kernel");
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    // Delays and form factors share one launch to reuse triangle data and avoid
    // a second traversal of the distributed N-by-N row block.
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    constexpr int blockSize = 256;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<static_cast<unsigned>(state.localCount), blockSize>>>(
            state.numTriangles, state.localStart, state.localCount, t,
            state.dKij, state.dTau, state.dAreas, state.dRho, state.dRadB,
            state.sourceIndex, state.numTimesteps / 2, state.dCurrent);
        cudaCheck(cudaGetLastError(), "propagation kernel launch");
        cudaCheck(cudaMemcpy(state.hostLocal, state.dCurrent,
                             state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "local radiosity download");
        MPI_Allgatherv(state.hostLocal, static_cast<int>(state.localCount), MPI_FLOAT,
                       state.hostCurrent, state.rowCounts.data(),
                       state.rowDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(state.dRadB + t * state.numTriangles, state.hostCurrent,
                             state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice),
                  "global radiosity upload");
        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    constexpr int blockSize = 256;
    distanceKernel<<<static_cast<unsigned>(state.localCount), blockSize>>>(
        state.numTriangles, state.numTimesteps, state.localStart, state.localCount,
        state.sourceIndex, state.dRadB, state.dDistances);
    cudaCheck(cudaGetLastError(), "distance kernel launch");
    cudaCheck(cudaMemcpy(state.hostLocal, state.dDistances,
                         state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost),
              "distance download");
    MPI_Gatherv(state.hostLocal, static_cast<int>(state.localCount), MPI_FLOAT,
                state.distances.data(), state.rowCounts.data(),
                state.rowDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    if (state.rank == 0) printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    if (state.rank == 0) {
#pragma omp parallel for reduction(&&:allNonNegative) reduction(min:minDist) reduction(max:maxDist) reduction(+:sumDist,nonZeroCount)
        for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
            val_t d = state.distances[i];
            allNonNegative = allNonNegative && d >= 0 && std::isfinite(d);
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            nonZeroCount += d > EPSILON;
        }
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    }

    // Check source distance is zero or very small
    if (state.rank == 0) {
        val_t srcDist = state.distances[state.sourceIndex];
        if (srcDist > WAVE_SPEED * 2)
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    unsigned int* deviceEnergized = nullptr;
    unsigned int receivedEnergy = 0;
    cudaCheck(cudaMalloc(&deviceEnergized, sizeof(unsigned int)), "energy counter allocation");
    cudaCheck(cudaMemset(deviceEnergized, 0, sizeof(unsigned int)), "energy counter reset");
    constexpr int validationBlockSize = 256;
    countEnergizedKernel<<<static_cast<unsigned>((state.numTriangles + validationBlockSize - 1) /
                                                 validationBlockSize), validationBlockSize>>>(
        state.dRadB, state.numTriangles, state.numTimesteps, deviceEnergized);
    cudaCheck(cudaGetLastError(), "energy validation kernel launch");
    cudaCheck(cudaMemcpy(&receivedEnergy, deviceEnergized, sizeof(unsigned int),
                         cudaMemcpyDeviceToHost), "energy counter download");
    cudaFree(deviceEnergized);

    if (state.rank == 0)
        printf("  Triangles receiving energy: %u/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        if (state.rank == 0) printf("  ERROR: No triangles received energy - simulation failed\n");
        allNonNegative = false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long* deviceNonzero = nullptr;
    unsigned long long localNonZeroUnsigned = 0;
    cudaCheck(cudaMalloc(&deviceNonzero, sizeof(unsigned long long)), "validation counter allocation");
    cudaCheck(cudaMemset(deviceNonzero, 0, sizeof(unsigned long long)), "validation counter reset");
    const size_t localFactors = state.localCount * state.numTriangles;
    countFormFactorsKernel<<<static_cast<unsigned>((localFactors + validationBlockSize - 1) /
                                                   validationBlockSize), validationBlockSize>>>(
        state.dKij, localFactors, deviceNonzero);
    cudaCheck(cudaGetLastError(), "validation kernel launch");
    cudaCheck(cudaMemcpy(&localNonZeroUnsigned, deviceNonzero, sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost), "validation counter download");
    cudaFree(deviceNonzero);
    long long localNonZeroKij = static_cast<long long>(localNonZeroUnsigned);
    long long nonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (state.rank == 0)
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (state.rank == 0 && nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        allNonNegative = false;
    }
    int valid = allNonNegative ? 1 : 0;
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (state.rank == 0 && valid) printf("  Validation: PASSED\n");
    return valid != 0;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "GPU discovery");
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "roomsim requires at least one CUDA GPU per node\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "GPU selection");
    MPI_Comm_free(&localCommunicator);

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

    if (targetTriangles <= 0 || timesteps <= 0 || reflectivity < 0.0f || reflectivity > 1.0f) {
        if (rank == 0) std::fprintf(stderr, "Triangle/timestep counts must be positive and reflectivity must be in [0,1]\n");
        MPI_Finalize();
        return 1;
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
        printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA enabled\n",
               ranks, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.ranks = ranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    if (ranks > static_cast<int>(state.numTriangles)) {
        if (rank == 0) std::fprintf(stderr, "MPI rank count must not exceed triangle count\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    allocateDeviceState(state);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::steady_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::steady_clock::now();
    long localPre = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long preDuration = 0;
    MPI_Reduce(&localPre, &preDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::steady_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::steady_clock::now();
    long localSim = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    long simDuration = 0;
    MPI_Reduce(&localSim, &simDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::steady_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::steady_clock::now();
    long localDist = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    long distDuration = 0;
    MPI_Reduce(&localDist, &distDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Distance computation time: %ld ms\n\n", distDuration);

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    if (rank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (rank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    }

    // Memory usage
    size_t maxLocalRows = (n + static_cast<size_t>(ranks) - 1) / static_cast<size_t>(ranks);
    size_t memKij = maxLocalRows * n * sizeof(val_t);
    size_t memTau = maxLocalRows * n * sizeof(int);
    size_t memRad = (t * n + n + maxLocalRows) * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    if (rank == 0) printf("  Memory usage per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    if (rank == 0) {
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n\n", hash);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            releaseDeviceState(state);
            MPI_Finalize();
            return 1;
        }
    }

    releaseDeviceState(state);
    MPI_Finalize();
    return 0;
}

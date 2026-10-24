/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of the room impulse response
 * simulation.  Work is partitioned across MPI ranks (contiguous groups of
 * receiver triangles), each rank drives an NVIDIA GPU for the two dominant
 * phases (form factors and wave propagation) and uses OpenMP for the
 * remaining bandwidth-bound host phases.  The result is identical to the
 * sequential reference output while scaling across the whole cluster.
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility
 *    and geometry (GPU, octree-accelerated, one thread per pair).
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 *    (GPU, one thread per receiver triangle per timestep).
 * 3. Distance estimation via cross-correlation of radiosity values
 *    (OpenMP, distributed across ranks).
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

#define HD __host__ __device__

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
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON &&
               fabsf(z - o.z) < EPSILON;
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
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }
};

// ----------------------------------------------------------------------------
// Flat octree representation used by CUDA kernels
// ----------------------------------------------------------------------------

struct GpuNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

static int flattenOctreeNode(const Octree& node, std::vector<GpuNode>& nodes,
                             std::vector<int>& triIndices, int& nextTri) {
    int myIndex = static_cast<int>(nodes.size());
    nodes.push_back(GpuNode{});

    Vec3 c = node.center;
    Vec3 h = node.halfExtent;
    int triStart = 0;
    int triCount = 0;

    if (!node.triangleIndices.empty()) {
        triStart = nextTri;
        triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            triIndices.push_back(static_cast<int>(idx));
        }
        nextTri += triCount;
    } else {
        for (int i = 0; i < 8; ++i) {
            int ci = -1;
            if (node.children[i]) {
                ci = flattenOctreeNode(*node.children[i], nodes, triIndices, nextTri);
            }
            nodes[myIndex].child[i] = ci;
        }
    }
    for (int i = 0; i < 8; ++i) {
        if (node.children[i] == nullptr) nodes[myIndex].child[i] = -1;
    }

    nodes[myIndex].center = c;
    nodes[myIndex].halfExtent = h;
    nodes[myIndex].triStart = triStart;
    nodes[myIndex].triCount = triCount;
    return myIndex;
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
// Random number generation (bit-exact reproduction of the reference stream)
// ============================================================================

// The exact reference stream is obtained with std::mt19937 directly; each
// rank jumps to its own offset with mt19937::discard and then advances the
// stream locally in row-major order.

// Build a sample point from two uniform barycentric samples.
HD inline Vec3 pointInTriangleUV(const Triangle& t, val_t u, val_t v) {
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

HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    const val_t BIG = 3.402823466e+38f;

    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return BIG;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return BIG;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return BIG;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (octree-accelerated, device)
// ============================================================================

HD inline bool rayIntersectsBoxDev(const Vec3& p1, const Vec3& p2, const GpuNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

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

HD inline bool isRayBlockedDev(const Vec3& from, const Vec3& to,
                               const GpuNode* nodes,
                               const int* triIndices,
                               const Triangle* triangles,
                               int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int ni = stack[--sp];
        GpuNode node = nodes[ni];
        if (!rayIntersectsBoxDev(from, to, node)) continue;

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
                if (ci >= 0 && sp < 128) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

// ----------------------------------------------------------------------------
// CUDA kernels
// ----------------------------------------------------------------------------

__global__ void computeKijKernel(const Triangle* __restrict__ triangles,
                                 int numTriangles,
                                 const GpuNode* __restrict__ nodes,
                                 const int* __restrict__ triIndices,
                                 int rowBegin,
                                 int localRowOffset,
                                 int numChunkRows,
                                 int kijStride,
                                 const val_t* __restrict__ randoms,
                                 const unsigned char* __restrict__ active,
                                 val_t* __restrict__ kij,
                                 int* __restrict__ nonZeroCount) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(numChunkRows) * numTriangles;
    if (idx >= total) return;

    int r = static_cast<int>(idx / numTriangles);
    int j = static_cast<int>(idx % numTriangles);
    int i = rowBegin + r;
    int outRow = localRowOffset + r;

    val_t value = ZERO;

    if (active[idx]) {
        const Triangle& triI = triangles[i];
        const Triangle& triJ = triangles[j];
        const val_t* rnd = randoms + static_cast<size_t>(idx) * (4 * NUM_RAYS);

        val_t kijVal = ZERO;

        for (int ray = 0; ray < NUM_RAYS; ++ray) {
            const val_t* rv = rnd + ray * 4;
            Vec3 pI = pointInTriangleUV(triI, rv[0], rv[1]);
            Vec3 pJ = pointInTriangleUV(triJ, rv[2], rv[3]);

            if (isRayBlockedDev(pI, pJ, nodes, triIndices, triangles, i, j)) continue;

            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr < EPSILON) continue;

            val_t cosPhiI = cosPhi(v, triI._normal);
            val_t cosPhiJ = cosPhi(-v, triJ._normal);

            if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

            kijVal += (cosPhiI * cosPhiJ) / (PI * distSqr);
        }

        value = kijVal * INV_NUM_RAYS;
    }

    // Store transposed (column-major over local rows) so that the propagation
    // kernel reads Kij coalesced across neighbouring receiver triangles.
    kij[static_cast<size_t>(j) * kijStride + outRow] = value;
    if (value > EPSILON) atomicAdd(nonZeroCount, 1);
}

__global__ void propagationStepKernel(const val_t* __restrict__ kij,
                                      const int* __restrict__ tau,
                                      val_t* __restrict__ radB,
                                      const val_t* __restrict__ areas,
                                      const val_t* __restrict__ rho,
                                      int numTriangles,
                                      int rowBegin,
                                      int numLocalRows,
                                      int timestep,
                                      int timeOff,
                                      int sourceIndex) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numLocalRows) return;

    int i = rowBegin + r;

    val_t sumB = ZERO;

    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tau[static_cast<size_t>(j) * numLocalRows + r];
        if (timestep < tauij) continue;

        val_t kijv = kij[static_cast<size_t>(j) * numLocalRows + r];
        if (kijv <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(timestep - tauij);
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        val_t contribution = kijv * areas[j];
        if (contribution > ONE) contribution = ONE;
        sumB += contribution * radJ;
    }

    val_t emission = ZERO;
    if (i == sourceIndex && timestep < timeOff) emission = ONE;

    radB[static_cast<size_t>(timestep) * numTriangles + i] = rho[i] * sumB + emission;
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
    std::vector<int> tauLocal;      // Time delays (localRows x N, row-major)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    int rank;
    int numRanks;
    size_t rowBegin;                // First triangle owned by this rank
    size_t rowEnd;                  // One past last triangle owned by this rank
    size_t localRows;

    std::vector<int> recvCounts;
    std::vector<int> displs;

    size_t nonZeroKij;              // Global count of positive form factors

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    size_t idxLocal(size_t r, size_t j) const { return r * numTriangles + j; }
};

// Device-side working set (one GPU per rank)
struct GpuData {
    Triangle* triangles = nullptr;
    GpuNode* nodes = nullptr;
    int* triIndices = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    int* nonZeroCount = nullptr;

    // Pinned host staging buffers for per-timestep radiosity exchange
    val_t* hostRow = nullptr;
    val_t* hostLocal = nullptr;

    // Per-chunk random stream staging (row-chunked to bound memory)
    val_t* randoms = nullptr;        // device: chunkPairs * 4 * NUM_RAYS floats
    unsigned char* active = nullptr; // device: chunkPairs bytes
    val_t* hostRandoms = nullptr;    // pinned host mirror
    unsigned char* hostActive = nullptr;
    size_t chunkRows = 0;
    size_t chunkPairs = 0;

    int numNodes = 0;
    int numIndices = 0;
    int deviceId = -1;
};

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

    if (state.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas (OpenMP: bandwidth bound host phase)
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(state.numTriangles, reflectivity);

    // Initialize radiosity / distance buffers
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Partition receiver triangles across MPI ranks (contiguous rows)
    state.rowBegin = state.numTriangles * static_cast<size_t>(state.rank) /
                     static_cast<size_t>(state.numRanks);
    state.rowEnd = state.numTriangles * static_cast<size_t>(state.rank + 1) /
                   static_cast<size_t>(state.numRanks);
    state.localRows = state.rowEnd - state.rowBegin;

    state.recvCounts.resize(state.numRanks);
    state.displs.resize(state.numRanks);
    for (int r = 0; r < state.numRanks; ++r) {
        size_t b = state.numTriangles * static_cast<size_t>(r) /
                   static_cast<size_t>(state.numRanks);
        size_t e = state.numTriangles * static_cast<size_t>(r + 1) /
                   static_cast<size_t>(state.numRanks);
        state.recvCounts[r] = static_cast<int>(e - b);
        state.displs[r] = static_cast<int>(b);
    }
    state.tauLocal.assign(state.localRows * state.numTriangles, 0);
}

// ============================================================================
// GPU setup
// ============================================================================

void initializeGpu(SimulationState& state, GpuData& gpu,
                   const std::vector<GpuNode>& hostNodes,
                   const std::vector<int>& hostTriIndices) {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int localRank = 0;
    int localSize = 1;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        state.rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    (void)localSize;

    gpu.deviceId = localRank % deviceCount;
    checkCuda(cudaSetDevice(gpu.deviceId), "cudaSetDevice");

    gpu.numNodes = static_cast<int>(hostNodes.size());
    gpu.numIndices = static_cast<int>(hostTriIndices.size());

    if (gpu.numNodes > 0) {
        checkCuda(cudaMalloc(&gpu.nodes, sizeof(GpuNode) * gpu.numNodes), "cudaMalloc nodes");
        checkCuda(cudaMemcpy(gpu.nodes, hostNodes.data(),
                             sizeof(GpuNode) * gpu.numNodes, cudaMemcpyHostToDevice),
                  "cudaMemcpy nodes");
    }
    if (gpu.numIndices > 0) {
        checkCuda(cudaMalloc(&gpu.triIndices, sizeof(int) * gpu.numIndices), "cudaMalloc triIndices");
        checkCuda(cudaMemcpy(gpu.triIndices, hostTriIndices.data(),
                             sizeof(int) * gpu.numIndices, cudaMemcpyHostToDevice),
                  "cudaMemcpy triIndices");
    }

    checkCuda(cudaMalloc(&gpu.triangles, sizeof(Triangle) * state.numTriangles),
              "cudaMalloc triangles");
    checkCuda(cudaMemcpy(gpu.triangles, state.triangles.data(),
                         sizeof(Triangle) * state.numTriangles, cudaMemcpyHostToDevice),
              "cudaMemcpy triangles");

    checkCuda(cudaMalloc(&gpu.areas, sizeof(val_t) * state.numTriangles), "cudaMalloc areas");
    checkCuda(cudaMemcpy(gpu.areas, state.areas.data(),
                         sizeof(val_t) * state.numTriangles, cudaMemcpyHostToDevice),
              "cudaMemcpy areas");

    checkCuda(cudaMalloc(&gpu.rho, sizeof(val_t) * state.numTriangles), "cudaMalloc rho");
    checkCuda(cudaMemcpy(gpu.rho, state.rho.data(),
                         sizeof(val_t) * state.numTriangles, cudaMemcpyHostToDevice),
              "cudaMemcpy rho");

    size_t localPairs = state.localRows * state.numTriangles;
    if (localPairs > 0) {
        checkCuda(cudaMalloc(&gpu.kij, sizeof(val_t) * localPairs), "cudaMalloc kij");
        checkCuda(cudaMalloc(&gpu.tau, sizeof(int) * localPairs), "cudaMalloc tau");
    }
    if (!state.radB.empty()) {
        checkCuda(cudaMalloc(&gpu.radB, sizeof(val_t) * state.radB.size()), "cudaMalloc radB");
        checkCuda(cudaMemset(gpu.radB, 0, sizeof(val_t) * state.radB.size()), "cudaMemset radB");
    }
    checkCuda(cudaMalloc(&gpu.nonZeroCount, sizeof(int)), "cudaMalloc nonZeroCount");
    checkCuda(cudaMemset(gpu.nonZeroCount, 0, sizeof(int)), "cudaMemset nonZeroCount");

    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&gpu.hostRow),
                            sizeof(val_t) * (state.numTriangles + 1),
                            cudaHostAllocDefault), "cudaHostAlloc hostRow");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&gpu.hostLocal),
                            sizeof(val_t) * (state.localRows + 1),
                            cudaHostAllocDefault), "cudaHostAlloc hostLocal");

    // Bound the per-chunk random stream to ~128 MiB.  Rows are processed in
    // chunks; only a chunk's sample buffer is ever resident at a time.
    const size_t budgetBytes = 128ull * 1024ull * 1024ull;
    const size_t bytesPerRow = state.numTriangles * 4ull * NUM_RAYS * sizeof(val_t);
    size_t chunkRows = 0;
    if (state.localRows > 0) {
        chunkRows = bytesPerRow > 0 ? budgetBytes / bytesPerRow : state.localRows;
        if (chunkRows < 1) chunkRows = 1;
        if (chunkRows > state.localRows) chunkRows = state.localRows;
    }
    gpu.chunkRows = chunkRows;
    gpu.chunkPairs = chunkRows * state.numTriangles;

    if (gpu.chunkPairs > 0) {
        size_t numRandoms = gpu.chunkPairs * 4ull * NUM_RAYS;
        checkCuda(cudaMalloc(&gpu.randoms, sizeof(val_t) * numRandoms),
                  "cudaMalloc randoms");
        checkCuda(cudaMalloc(&gpu.active, sizeof(unsigned char) * gpu.chunkPairs),
                  "cudaMalloc active");
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&gpu.hostRandoms),
                                sizeof(val_t) * numRandoms,
                                cudaHostAllocDefault), "cudaHostAlloc hostRandoms");
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&gpu.hostActive),
                                sizeof(unsigned char) * gpu.chunkPairs,
                                cudaHostAllocDefault), "cudaHostAlloc hostActive");
    }

    MPI_Comm_free(&localComm);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, GpuData& gpu) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(state.localRows); ++r) {
        size_t i = state.rowBegin + static_cast<size_t>(r);
        for (size_t j = 0; j < N; ++j) {
            state.tauLocal[static_cast<size_t>(j) * state.localRows + r] =
                (i == j) ? 0 : computeTau(state.triangles[i], state.triangles[j]);
        }
    }

    if (state.localRows > 0) {
        checkCuda(cudaMemcpy(gpu.tau, state.tauLocal.data(),
                             sizeof(int) * state.tauLocal.size(),
                             cudaMemcpyHostToDevice), "upload tau");
    }
}

void computeFormFactors(SimulationState& state, GpuData& gpu) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t localRows = state.localRows;
    const size_t perPairFloats = 4ull * NUM_RAYS;

    int localNonZero = 0;

    // --- Reproduce the exact reference random stream -----------------------
    // The reference consumes 2*NUM_RAYS uniforms for every non-culled pair in
    // row-major order.  Each rank independently computes how many draws occur
    // before its first row and jumps there, so ranks never serialize on the
    // random stream.  libstdc++'s uniform_real_distribution<float>(0,1) maps
    // one engine output to v/2^32, reproduced here exactly with a power-of-two
    // multiply.
    const val_t INV_2P32 = 1.0f / 4294967296.0f;

    uint64_t activeBefore = 0;
    if (state.rowBegin > 0) {
        const size_t rb = state.rowBegin;
        #pragma omp parallel for reduction(+:activeBefore) schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(rb); ++ii) {
            size_t i = static_cast<size_t>(ii);
            const Vec3& ni = state.triangles[i]._normal;
            uint64_t local = 0;
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                val_t d = ni.dot(state.triangles[j]._normal);
                if (!(d > 0.99f)) ++local;
            }
            activeBefore += local;
        }
    }

    std::mt19937 rng(42);
    rng.discard(static_cast<unsigned long long>(activeBefore) * perPairFloats);

    if (localRows > 0) {
        checkCuda(cudaMemset(gpu.nonZeroCount, 0, sizeof(int)), "reset nonZeroCount");

        const size_t chunkRows = gpu.chunkRows > 0 ? gpu.chunkRows : 1;
        const size_t numChunks = (localRows + chunkRows - 1) / chunkRows;
        const int threads = 256;

        // Generate one chunk of samples on the host.  This runs while the GPU
        // is still consuming the previous chunk, hiding most of the cost.
        auto generateChunk = [&](size_t c) {
            size_t r0 = c * chunkRows;
            size_t r1 = std::min(localRows, r0 + chunkRows);
            for (size_t rr = r0; rr < r1; ++rr) {
                size_t i = state.rowBegin + rr;
                const Vec3& ni = state.triangles[i]._normal;
                size_t pRow = (rr - r0) * N;
                for (size_t j = 0; j < N; ++j) {
                    size_t p = pRow + j;
                    bool isActive = false;
                    if (i != j) {
                        val_t d = ni.dot(state.triangles[j]._normal);
                        isActive = !(d > 0.99f);
                    }
                    gpu.hostActive[p] = isActive ? 1 : 0;
                    if (isActive) {
                        val_t* out = gpu.hostRandoms + p * perPairFloats;
                        for (size_t k = 0; k < perPairFloats; ++k) {
                            out[k] = static_cast<val_t>(rng()) * INV_2P32;
                        }
                    }
                }
            }
        };

        auto uploadAndRun = [&](size_t c) {
            size_t r0 = c * chunkRows;
            size_t r1 = std::min(localRows, r0 + chunkRows);
            size_t rows = r1 - r0;
            size_t pairs = rows * N;

            checkCuda(cudaMemcpy(gpu.randoms, gpu.hostRandoms,
                                 sizeof(val_t) * pairs * perPairFloats,
                                 cudaMemcpyHostToDevice), "upload randoms");
            checkCuda(cudaMemcpy(gpu.active, gpu.hostActive,
                                 sizeof(unsigned char) * pairs,
                                 cudaMemcpyHostToDevice), "upload active");

            long long blocksLL = (static_cast<long long>(pairs) + threads - 1) / threads;
            int blocks = static_cast<int>(blocksLL);

            computeKijKernel<<<blocks, threads>>>(
                gpu.triangles,
                static_cast<int>(N),
                gpu.nodes,
                gpu.triIndices,
                static_cast<int>(state.rowBegin + r0),
                static_cast<int>(r0),
                static_cast<int>(rows),
                static_cast<int>(localRows),
                gpu.randoms,
                gpu.active,
                gpu.kij,
                gpu.nonZeroCount);

            checkCuda(cudaGetLastError(), "computeKijKernel launch");
        };

        generateChunk(0);
        for (size_t c = 0; c < numChunks; ++c) {
            uploadAndRun(c);
            if (c + 1 < numChunks) generateChunk(c + 1);
        }

        checkCuda(cudaDeviceSynchronize(), "computeKijKernel");
        checkCuda(cudaMemcpy(&localNonZero, gpu.nonZeroCount, sizeof(int),
                             cudaMemcpyDeviceToHost), "read nonZeroCount");
    }

    uint64_t contrib = static_cast<uint64_t>(localNonZero);
    uint64_t total = 0;
    MPI_Allreduce(&contrib, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = static_cast<size_t>(total);

    if (state.rank == 0) {
        for (size_t done = 100; done < N; done += 100) {
            printf("  Progress: %zu/%zu triangles\n", done, N);
        }
        printf("  Progress: %zu/%zu triangles\n", N, N);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuData& gpu) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const int threads = 256;
    int blocks = state.localRows > 0
        ? static_cast<int>((state.localRows + threads - 1) / threads) : 0;

    const bool singleRank = (state.numRanks == 1);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows > 0) {
            propagationStepKernel<<<blocks, threads>>>(
                gpu.kij, gpu.tau, gpu.radB, gpu.areas, gpu.rho,
                static_cast<int>(N),
                static_cast<int>(state.rowBegin),
                static_cast<int>(state.localRows),
                static_cast<int>(t),
                static_cast<int>(state.numTimesteps / 2),
                static_cast<int>(state.sourceIndex));
            checkCuda(cudaGetLastError(), "propagationStepKernel launch");

            if (!singleRank) {
                checkCuda(cudaMemcpy(gpu.hostLocal,
                                     gpu.radB + t * N + state.rowBegin,
                                     sizeof(val_t) * state.localRows,
                                     cudaMemcpyDeviceToHost), "download radB row slice");
            }
        }

        if (!singleRank) {
            MPI_Allgatherv(gpu.hostLocal, static_cast<int>(state.localRows), MPI_FLOAT,
                           gpu.hostRow,
                           state.recvCounts.data(), state.displs.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);

            checkCuda(cudaMemcpy(gpu.radB + t * N, gpu.hostRow,
                                 sizeof(val_t) * N, cudaMemcpyHostToDevice),
                      "upload radB row");
        }

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    // Bring the complete radiosity history back to the host for the
    // cross-correlation stage (a single bulk transfer).
    if (!state.radB.empty()) {
        checkCuda(cudaMemcpy(state.radB.data(), gpu.radB,
                             sizeof(val_t) * state.radB.size(),
                             cudaMemcpyDeviceToHost), "download radB history");
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t T = state.numTimesteps;
    const size_t src = state.sourceIndex;

    std::vector<val_t> localDistances(state.localRows > 0 ? state.localRows : 1, ZERO);

    #pragma omp parallel for schedule(static)
    for (long long rr = 0; rr < static_cast<long long>(state.localRows); ++rr) {
        size_t r = static_cast<size_t>(rr);
        size_t i = state.rowBegin + r;

        val_t maxCorr = ZERO;
        int bestT = 0;

        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, src)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDistances[r] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(localDistances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                   state.distances.data(), state.recvCounts.data(), state.displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
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
    int nonZeroKij = static_cast<int>(state.nonZeroKij);
    size_t totalPairs = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, totalPairs,
           100.0f * nonZeroKij / static_cast<val_t>(totalPairs));

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

    SimulationState state;
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.numRanks);

    // Avoid oversubscribing OpenMP threads when several ranks share a node.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        int localSize = 1;
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                            state.rank, MPI_INFO_NULL, &localComm);
        MPI_Comm_size(localComm, &localSize);
        MPI_Comm_free(&localComm);
        int procs = omp_get_num_procs();
        int threads = procs / (localSize > 0 ? localSize : 1);
        if (threads < 1) threads = 1;
        omp_set_num_threads(threads);
    }

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
            if (state.rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (state.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (state.rank == 0) {
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
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    // Flatten octree for the GPU
    std::vector<GpuNode> hostNodes;
    std::vector<int> hostTriIndices;
    int nextTri = 0;
    if (!state.triangles.empty()) {
        flattenOctreeNode(state.octree, hostNodes, hostTriIndices, nextTri);
    }

    GpuData gpu;
    initializeGpu(state, gpu, hostNodes, hostTriIndices);

    if (state.rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, gpu);
    computeFormFactors(state, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (state.rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (state.rank == 0) {
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

    int validationStatus = 1;

    if (state.rank == 0) {
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
            validationStatus = validateResults(state) ? 1 : 0;
        }
    }

    MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Release device resources
    if (gpu.kij) cudaFree(gpu.kij);
    if (gpu.tau) cudaFree(gpu.tau);
    if (gpu.radB) cudaFree(gpu.radB);
    if (gpu.areas) cudaFree(gpu.areas);
    if (gpu.rho) cudaFree(gpu.rho);
    if (gpu.triangles) cudaFree(gpu.triangles);
    if (gpu.nodes) cudaFree(gpu.nodes);
    if (gpu.triIndices) cudaFree(gpu.triIndices);
    if (gpu.nonZeroCount) cudaFree(gpu.nonZeroCount);
    if (gpu.hostRow) cudaFreeHost(gpu.hostRow);
    if (gpu.hostLocal) cudaFreeHost(gpu.hostLocal);
    if (gpu.randoms) cudaFree(gpu.randoms);
    if (gpu.active) cudaFree(gpu.active);
    if (gpu.hostRandoms) cudaFreeHost(gpu.hostRandoms);
    if (gpu.hostActive) cudaFreeHost(gpu.hostActive);

    MPI_Finalize();
    return validationStatus ? 0 : 1;
}

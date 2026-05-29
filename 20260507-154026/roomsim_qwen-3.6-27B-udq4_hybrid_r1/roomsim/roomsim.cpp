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
 * 
 * Parallelization strategy:
 * - MPI: Distribute triangle rows across ranks for N*N computations
 * - CUDA: GPU kernels for form factors, time delays, simulation, distances
 * - OpenMP: Parallelize host-side initialization, validation, and hash
 */

#include <mpi.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

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
// Flat Octree for GPU Access
// ============================================================================

// Flat (linear) octree node stored in a contiguous array for GPU access
struct FlatOctreeNode {
    float minBound[3];
    float maxBound[3];
    float halfExtent[3];
    float center[3];
    int triStart;       // Index into flatTriangleIndices array (inclusive)
    int triCount;       // Number of triangle indices in this leaf
    int childOffset;    // Index of first child in nodes array (-1 if leaf)
    int isLeaf;         // 1 if leaf, 0 if internal
};

struct FlatOctree {
    std::vector<FlatOctreeNode> nodes;
    std::vector<int> flatTriangleIndices;

    void buildFromOctree(const Octree& octree) {
        nodes.clear();
        flatTriangleIndices.clear();
        // Reserve space - at most one node per original octree node
        nodes.reserve(256);
        flatTriangleIndices.reserve(octree.allTriangles ? octree.allTriangles->size() : 0);

        int rootOffset = 0;
        buildFlatNode(octree, rootOffset);
    }

private:
    int buildFlatNode(const Octree& src, int& offset) {
        FlatOctreeNode node;
        node.minBound[0] = src.minBound.x;
        node.minBound[1] = src.minBound.y;
        node.minBound[2] = src.minBound.z;
        node.maxBound[0] = src.maxBound.x;
        node.maxBound[1] = src.maxBound.y;
        node.maxBound[2] = src.maxBound.z;
        node.halfExtent[0] = src.halfExtent.x;
        node.halfExtent[1] = src.halfExtent.y;
        node.halfExtent[2] = src.halfExtent.z;
        node.center[0] = src.center.x;
        node.center[1] = src.center.y;
        node.center[2] = src.center.z;

        if (!src.triangleIndices.empty()) {
            node.isLeaf = 1;
            node.triStart = static_cast<int>(flatTriangleIndices.size());
            node.triCount = static_cast<int>(src.triangleIndices.size());
            node.childOffset = -1;
            for (size_t idx : src.triangleIndices) {
                flatTriangleIndices.push_back(static_cast<int>(idx));
            }
        } else {
            node.isLeaf = 0;
            node.triStart = 0;
            node.triCount = 0;
            int childBase = offset;
            for (int i = 0; i < 8; ++i) {
                if (src.children[i]) {
                    offset++;
                    buildFlatNode(*src.children[i], offset);
                } else {
                    // Empty child - create a dummy leaf
                    offset++;
                    FlatOctreeNode dummy;
                    dummy.isLeaf = 1;
                    dummy.triStart = 0;
                    dummy.triCount = 0;
                    dummy.childOffset = -1;
                    dummy.minBound[0] = dummy.minBound[1] = dummy.minBound[2] = 0;
                    dummy.maxBound[0] = dummy.maxBound[1] = dummy.maxBound[2] = 0;
                    dummy.halfExtent[0] = dummy.halfExtent[1] = dummy.halfExtent[2] = 0;
                    dummy.center[0] = dummy.center[1] = dummy.center[2] = 0;
                    nodes.push_back(dummy);
                }
            }
            node.childOffset = childBase;
        }
        nodes.push_back(node);
        return static_cast<int>(nodes.size()) - 1;
    }
};

// ============================================================================
// CUDA Device Code
// ============================================================================

#define CUDA_EPSILON 1e-6f
#define CUDA_PI 3.14159265358979323846f
#define CUDA_WAVE_SPEED 0.5f
#define CUDA_INV_WAVE_SPEED 2.0f
#define CUDA_NUM_RAYS 16
#define CUDA_INV_NUM_RAYS (1.0f / 16.0f)
#define CUDA_MAX_OCTREE_DEPTH 12

// Simple LCG for GPU random numbers (per-thread state)
// Uses same parameters as original sequential RNG to ensure reproducibility
__device__ unsigned int lcg_state;

__device__ unsigned int lcg_next() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

__device__ float lcg_rand() {
    return static_cast<float>(lcg_next()) / 4294967296.0f;
}

// Sequential RNG matching original for pre-seeding
class SequentialRNG {
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
public:
    explicit SequentialRNG(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    float rand() { return dist(rng); }
};

// Device Vec3
struct dVec3 {
    float x, y, z;
    __host__ __device__ dVec3() : x(0), y(0), z(0) {}
    __host__ __device__ dVec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ dVec3 operator+(const dVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ dVec3 operator-(const dVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ dVec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ dVec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ dVec3 operator-() const { return {-x, -y, -z}; }
    __host__ __device__ float dot(const dVec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ dVec3 cross(const dVec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    __host__ __device__ float squaredNorm() const { return x * x + y * y + z * z; }
    __device__ float norm() const { return sqrtf(squaredNorm()); }
    __device__ dVec3 normalized() const {
        float n = norm();
        return n > CUDA_EPSILON ? *this / n : dVec3();
    }
};

// Device Triangle
struct dTriangle {
    dVec3 a, b, c, _normal;
};

// Device FlatOctreeNode
struct dFlatOctreeNode {
    float minBound[3];
    float maxBound[3];
    float halfExtent[3];
    float center[3];
    int triStart;
    int triCount;
    int childOffset;
    int isLeaf;
};

// Device: ray-box intersection
__device__ bool rayIntersectsBox(const dVec3& p1, const dVec3& p2, const dFlatOctreeNode& node) {
    dVec3 d = (p2 - p1) * 0.5f;
    dVec3 c = p1 + d - dVec3(node.center[0], node.center[1], node.center[2]);
    dVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent[0] + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent[1] + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent[2] + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent[1] * ad.z + node.halfExtent[2] * ad.y + CUDA_EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent[2] * ad.x + node.halfExtent[0] * ad.z + CUDA_EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent[0] * ad.y + node.halfExtent[1] * ad.x + CUDA_EPSILON) return false;

    return true;
}

// Device: ray-triangle intersection (Möller-Trumbore)
__device__ float rayTriangleIntersect(const dVec3& orig, const dVec3& dir,
                                       const dVec3& v0, const dVec3& v1, const dVec3& v2) {
    dVec3 e1 = v1 - v0;
    dVec3 e2 = v2 - v0;
    dVec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < CUDA_EPSILON) return 1e30f;

    float invDet = 1.0f / det;
    dVec3 tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;

    dVec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    return e2.dot(qvec) * invDet;
}

// Device: check if ray is blocked by octree traversal
__device__ bool isRayBlockedGPU(const dVec3& from, const dVec3& to,
                                 const dFlatOctreeNode* nodes, const int* flatTriIndices,
                                 const dTriangle* tris,
                                 int srcTriIdx, int dstTriIdx) {
    dVec3 dir = to - from;
    float rayLen = dir.norm();
    if (rayLen < CUDA_EPSILON) return true;
    dVec3 dirNorm = dir / rayLen;

    // Stack-based octree traversal
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int idx = stack[--sp];
        const dFlatOctreeNode& node = nodes[idx];

        if (!rayIntersectsBox(from, to, node)) continue;

        if (node.isLeaf) {
            for (int k = 0; k < node.triCount; ++k) {
                int triIdx = flatTriIndices[node.triStart + k];
                if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;

                float dist = rayTriangleIntersect(from, dirNorm,
                    tris[triIdx].a, tris[triIdx].b, tris[triIdx].c);
                if (dist > CUDA_EPSILON && dist < rayLen - CUDA_EPSILON) {
                    return true;
                }
            }
        } else {
            if (sp < 31) {
                for (int c = 0; c < 8; ++c) {
                    stack[sp++] = node.childOffset + c;
                }
            }
        }
    }
    return false;
}

// Device: random point in triangle
__device__ dVec3 randomPointInTriangleGPU(const dTriangle& t) {
    float u = lcg_rand();
    float v = lcg_rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    dVec3 ab = t.b - t.a;
    dVec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Device: cosPhi
__device__ float cosPhiGPU(const dVec3& v, const dVec3& normal) {
    float vNorm = v.norm();
    if (vNorm <= CUDA_EPSILON) return 0.0f;
    return fmaxf(0.0f, v.dot(normal) / vNorm);
}

// CUDA Kernel: Compute form factors for assigned rows
__global__ void computeKijKernel(const dTriangle* __restrict__ tris,
                                  const dFlatOctreeNode* __restrict__ octreeNodes,
                                  const int* __restrict__ octreeTriIndices,
                                  float* __restrict__ kij,
                                  size_t numTriangles,
                                  size_t rowStart,
                                  size_t rowEnd,
                                  unsigned int* rngStates) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = (rowEnd - rowStart) * numTriangles;
    if (idx >= total) return;

    size_t i = rowStart + idx / numTriangles;
    size_t j = idx % numTriangles;
    if (i == j) {
        kij[i * numTriangles + j] = 0.0f;
        return;
    }

    const dTriangle& triI = tris[i];
    const dTriangle& triJ = tris[j];

    // Cull triangles facing the same direction
    float ndot = triI._normal.x * triJ._normal.x + triI._normal.y * triJ._normal.y + triI._normal.z * triJ._normal.z;
    if (ndot > 0.99f) {
        kij[i * numTriangles + j] = 0.0f;
        return;
    }

    // Set up per-thread RNG state
    lcg_state = rngStates[idx];

    float kij_val = 0.0f;

    for (int r = 0; r < CUDA_NUM_RAYS; ++r) {
        dVec3 pI = randomPointInTriangleGPU(triI);
        dVec3 pJ = randomPointInTriangleGPU(triJ);

        if (isRayBlockedGPU(pI, pJ, octreeNodes, octreeTriIndices, tris, static_cast<int>(i), static_cast<int>(j))) continue;

        dVec3 v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < CUDA_EPSILON) continue;

        float cosPhiI = cosPhiGPU(v, triI._normal);
        float cosPhiJ = cosPhiGPU(dVec3(-v.x, -v.y, -v.z), triJ._normal);

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij_val += (cosPhiI * cosPhiJ) / (CUDA_PI * distSqr);
    }

    kij[i * numTriangles + j] = kij_val * CUDA_INV_NUM_RAYS;
}

// CUDA Kernel: Compute time delays
__global__ void computeTauKernel(const dTriangle* __restrict__ tris,
                                  int* __restrict__ tau,
                                  size_t numTriangles,
                                  size_t rowStart,
                                  size_t rowEnd) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = (rowEnd - rowStart) * numTriangles;
    if (idx >= total) return;

    size_t i = rowStart + idx / numTriangles;
    size_t j = idx % numTriangles;
    if (i == j) {
        tau[i * numTriangles + j] = 0;
        return;
    }

    dTriangle triI = tris[i];
    dTriangle triJ = tris[j];
    dVec3 ci = (triI.a + triI.b + triI.c) / 3.0f;
    dVec3 cj = (triJ.a + triJ.b + triJ.c) / 3.0f;
    float dist = (ci - cj).norm();
    tau[i * numTriangles + j] = static_cast<int>(ceilf(dist * CUDA_INV_WAVE_SPEED));
}

// CUDA Kernel: Simulation step for one timestep
// Each thread computes radB[t][i] for one triangle
__global__ void simulationKernel(float* __restrict__ kij,
                                  int* __restrict__ tau,
                                  float* __restrict__ areas,
                                  float* __restrict__ rho,
                                  float* __restrict__ radE,
                                  float* __restrict__ radB,
                                  size_t numTriangles,
                                  size_t numTimesteps,
                                  size_t t,
                                  size_t rowStart,
                                  size_t rowEnd) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t localI = rowStart + idx;
    if (localI >= rowEnd) return;

    float sumB = 0.0f;

    for (size_t j = 0; j < numTriangles; ++j) {
        if (localI == j) continue;

        int tauij = tau[localI * numTriangles + j];
        if (static_cast<int>(t) < tauij) continue;

        float k = kij[localI * numTriangles + j];
        if (k <= 0.0f) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        float radJ = radB[srcTime * numTriangles + j];
        if (radJ <= 0.0f) continue;

        sumB += fminf(k * areas[j], 1.0f) * radJ;
    }

    radB[t * numTriangles + localI] = rho[localI] * sumB + radE[t * numTriangles + localI];
}

// CUDA Kernel: Distance computation via cross-correlation
__global__ void distanceKernel(const float* __restrict__ radB,
                                float* __restrict__ distances,
                                size_t numTriangles,
                                size_t numTimesteps,
                                size_t sourceIndex,
                                size_t rowStart,
                                size_t rowEnd) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t localI = rowStart + idx;
    if (localI >= rowEnd) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (size_t t = 0; t < numTimesteps; ++t) {
        float sum = 0.0f;
        for (size_t tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[tt * numTriangles + localI];
            float pS = radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[localI] = CUDA_WAVE_SPEED * static_cast<float>(bestT);
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

    Octree octree;                  // Spatial acceleration structure (host)
    FlatOctree flatOctree;          // Flat octree for GPU access

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

    // Build flat octree for GPU
    state.flatOctree.buildFromOctree(state.octree);

    // Initialize areas with OpenMP
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
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
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Computing form factors (Kij)...\n");

    size_t N = state.numTriangles;

    // Distribute rows across MPI ranks
    size_t totalRows = N;
    size_t rowsPerRank = totalRows / static_cast<size_t>(numRanks);
    size_t remainder = totalRows % static_cast<size_t>(numRanks);
    size_t rowStart = static_cast<size_t>(rank) * rowsPerRank + std::min(static_cast<size_t>(rank), remainder);
    size_t rowEnd = (static_cast<size_t>(rank) + 1) * rowsPerRank + std::min(static_cast<size_t>(rank) + 1, remainder);

    size_t localRows = rowEnd - rowStart;
    if (localRows == 0) return;

    // Allocate device memory
    dTriangle* d_tris = nullptr;
    dFlatOctreeNode* d_octreeNodes = nullptr;
    int* d_octreeTriIndices = nullptr;
    float* d_kij = nullptr;
    unsigned int* d_rngStates = nullptr;

    // Upload triangles
    size_t numOctreeNodes = state.flatOctree.nodes.size();
    size_t numOctreeTriIndices = state.flatOctree.flatTriangleIndices.size();

    cudaMalloc((void**)&d_tris, N * sizeof(dTriangle));
    cudaMalloc((void**)&d_octreeNodes, numOctreeNodes * sizeof(dFlatOctreeNode));
    cudaMalloc((void**)&d_octreeTriIndices, numOctreeTriIndices * sizeof(int));
    cudaMalloc((void**)&d_kij, N * N * sizeof(float));
    cudaMalloc((void**)&d_rngStates, localRows * N * sizeof(unsigned int));

    // Prepare triangle data for GPU
    std::vector<dTriangle> h_tris(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        h_tris[i].a = dVec3(state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z);
        h_tris[i].b = dVec3(state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z);
        h_tris[i].c = dVec3(state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z);
        h_tris[i]._normal = dVec3(state.triangles[i]._normal.x, state.triangles[i]._normal.y, state.triangles[i]._normal.z);
    }
    cudaMemcpy(d_tris, h_tris.data(), N * sizeof(dTriangle), cudaMemcpyHostToDevice);

    // Prepare octree data for GPU
    std::vector<dFlatOctreeNode> h_octree(numOctreeNodes);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOctreeNodes; ++i) {
        h_octree[i].minBound[0] = state.flatOctree.nodes[i].minBound[0];
        h_octree[i].minBound[1] = state.flatOctree.nodes[i].minBound[1];
        h_octree[i].minBound[2] = state.flatOctree.nodes[i].minBound[2];
        h_octree[i].maxBound[0] = state.flatOctree.nodes[i].maxBound[0];
        h_octree[i].maxBound[1] = state.flatOctree.nodes[i].maxBound[1];
        h_octree[i].maxBound[2] = state.flatOctree.nodes[i].maxBound[2];
        h_octree[i].halfExtent[0] = state.flatOctree.nodes[i].halfExtent[0];
        h_octree[i].halfExtent[1] = state.flatOctree.nodes[i].halfExtent[1];
        h_octree[i].halfExtent[2] = state.flatOctree.nodes[i].halfExtent[2];
        h_octree[i].center[0] = state.flatOctree.nodes[i].center[0];
        h_octree[i].center[1] = state.flatOctree.nodes[i].center[1];
        h_octree[i].center[2] = state.flatOctree.nodes[i].center[2];
        h_octree[i].triStart = state.flatOctree.nodes[i].triStart;
        h_octree[i].triCount = state.flatOctree.nodes[i].triCount;
        h_octree[i].childOffset = state.flatOctree.nodes[i].childOffset;
        h_octree[i].isLeaf = state.flatOctree.nodes[i].isLeaf;
    }
    cudaMemcpy(d_octreeNodes, h_octree.data(), numOctreeNodes * sizeof(dFlatOctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_octreeTriIndices, state.flatOctree.flatTriangleIndices.data(),
               numOctreeTriIndices * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize RNG states with deterministic seeds based on global (i,j) indices
    std::vector<unsigned int> h_rngStates(localRows * N);
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t globalI = rowStart + i;
            h_rngStates[i * N + j] = static_cast<unsigned int>((globalI * N + j + 42) * 2654435761u);
        }
    }
    cudaMemcpy(d_rngStates, h_rngStates.data(), localRows * N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Set device pointer for triangles (used by isRayBlockedGPU)
    cudaFuncSetAttribute(computeKijKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    // Use cudaFuncSetCacheConfig for better performance
    cudaFuncSetCacheConfig(computeKijKernel, cudaFuncCachePreferL1);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (int)((localRows * N + blockSize - 1) / blockSize);
    // Limit blocks to avoid launch failure
    int maxBlocks = 65535;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;

    computeKijKernel<<<numBlocks, blockSize>>>(
        d_tris, d_octreeNodes, d_octreeTriIndices, d_kij,
        N, rowStart, rowEnd, d_rngStates);
    cudaDeviceSynchronize();

    // Download results
    cudaMemcpy(state.kij.data(), d_kij, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_tris);
    cudaFree(d_octreeNodes);
    cudaFree(d_octreeTriIndices);
    cudaFree(d_kij);
    cudaFree(d_rngStates);

    // Gather results from all ranks (each rank computed its rows, but all need full matrix)
    // Use MPI_Allgatherv to distribute row chunks
    {
        std::vector<val_t> localResult(localRows * N);
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t i = rowStart; i < rowEnd; ++i) {
            for (size_t j = 0; j < N; ++j) {
                localResult[(i - rowStart) * N + j] = state.kij[i * N + j];
            }
        }

        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            size_t rStart = static_cast<size_t>(r) * rowsPerRank + std::min(static_cast<size_t>(r), remainder);
            size_t rEnd = (static_cast<size_t>(r) + 1) * rowsPerRank + std::min(static_cast<size_t>(r) + 1, remainder);
            recvCounts[r] = static_cast<int>((rEnd - rStart) * N);
            displs[r] = static_cast<int>(rStart * N);
        }

        MPI_Allgatherv(localResult.data(), static_cast<int>(localRows * N), MPI_FLOAT,
                       state.kij.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
    }

    if (rank == 0) printf("  Form factors computed on %d ranks\n", numRanks);
}

void computeTimeDelays(SimulationState& state) {
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;

    // Distribute rows across MPI ranks
    size_t totalRows = N;
    size_t rowsPerRank = totalRows / static_cast<size_t>(numRanks);
    size_t remainder = totalRows % static_cast<size_t>(numRanks);
    size_t rowStart = static_cast<size_t>(rank) * rowsPerRank + std::min(static_cast<size_t>(rank), remainder);
    size_t rowEnd = (static_cast<size_t>(rank) + 1) * rowsPerRank + std::min(static_cast<size_t>(rank) + 1, remainder);

    size_t localRows = rowEnd - rowStart;
    if (localRows == 0) return;

    // Allocate device memory
    dTriangle* d_tris = nullptr;
    int* d_tau = nullptr;

    cudaMalloc((void**)&d_tris, N * sizeof(dTriangle));
    cudaMalloc((void**)&d_tau, N * N * sizeof(int));

    // Prepare and upload triangles
    std::vector<dTriangle> h_tris(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        h_tris[i].a = dVec3(state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z);
        h_tris[i].b = dVec3(state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z);
        h_tris[i].c = dVec3(state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z);
        h_tris[i]._normal = dVec3(state.triangles[i]._normal.x, state.triangles[i]._normal.y, state.triangles[i]._normal.z);
    }
    cudaMemcpy(d_tris, h_tris.data(), N * sizeof(dTriangle), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (int)((localRows * N + blockSize - 1) / blockSize);
    int maxBlocks = 65535;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;

    computeTauKernel<<<numBlocks, blockSize>>>(
        d_tris, d_tau, N, rowStart, rowEnd);
    cudaDeviceSynchronize();

    // Download results
    cudaMemcpy(state.tau.data(), d_tau, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_tris);
    cudaFree(d_tau);

    // Gather results from all ranks
    {
        std::vector<int> localResult(localRows * N);
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t i = rowStart; i < rowEnd; ++i) {
            for (size_t j = 0; j < N; ++j) {
                localResult[(i - rowStart) * N + j] = state.tau[i * N + j];
            }
        }

        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            size_t rStart = static_cast<size_t>(r) * rowsPerRank + std::min(static_cast<size_t>(r), remainder);
            size_t rEnd = (static_cast<size_t>(r) + 1) * rowsPerRank + std::min(static_cast<size_t>(r) + 1, remainder);
            recvCounts[r] = static_cast<int>((rEnd - rStart) * N);
            displs[r] = static_cast<int>(rStart * N);
        }

        MPI_Allgatherv(localResult.data(), static_cast<int>(localRows * N), MPI_INT,
                       state.tau.data(), recvCounts.data(), displs.data(), MPI_INT,
                       MPI_COMM_WORLD);
    }

    if (rank == 0) printf("  Time delays computed on %d ranks\n", numRanks);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Running wave propagation simulation...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Distribute triangles across MPI ranks
    size_t rowsPerRank = N / static_cast<size_t>(numRanks);
    size_t remainder = N % static_cast<size_t>(numRanks);
    size_t rowStart = static_cast<size_t>(rank) * rowsPerRank + std::min(static_cast<size_t>(rank), remainder);
    size_t rowEnd = (static_cast<size_t>(rank) + 1) * rowsPerRank + std::min(static_cast<size_t>(rank) + 1, remainder);
    size_t localRows = rowEnd - rowStart;

    // Allocate device memory for simulation
    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;

    cudaMalloc((void**)&d_kij, N * N * sizeof(float));
    cudaMalloc((void**)&d_tau, N * N * sizeof(int));
    cudaMalloc((void**)&d_areas, N * sizeof(float));
    cudaMalloc((void**)&d_rho, N * sizeof(float));
    cudaMalloc((void**)&d_radE, T * N * sizeof(float));
    cudaMalloc((void**)&d_radB, T * N * sizeof(float));

    cudaMemcpy(d_kij, state.kij.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, state.tau.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_radB, 0, T * N * sizeof(float));

    // Launch simulation timesteps
    int blockSize = 256;
    int numBlocks = static_cast<int>((localRows + blockSize - 1) / blockSize);

    for (size_t t = 0; t < T; ++t) {
        // Launch GPU kernel for this timestep
        simulationKernel<<<numBlocks, blockSize>>>(
            d_kij, d_tau, d_areas, d_rho, d_radE, d_radB,
            N, T, t, rowStart, rowEnd);
        cudaDeviceSynchronize();

        // Download radB for this timestep from GPU to host (local rows only)
        cudaMemcpy(state.radB.data() + t * N + rowStart,
                   d_radB + t * N + rowStart,
                   localRows * sizeof(float), cudaMemcpyDeviceToHost);

        // Broadcast radB for this timestep to all ranks so next timestep has full data
        MPI_Bcast(state.radB.data() + t * N, static_cast<int>(N), MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Final download of all radB data
    cudaMemcpy(state.radB.data(), d_radB, T * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_kij);
    cudaFree(d_tau);
    cudaFree(d_areas);
    cudaFree(d_rho);
    cudaFree(d_radE);
    cudaFree(d_radB);

    if (rank == 0) printf("  Simulation complete on %d ranks\n", numRanks);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Computing distances via cross-correlation...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Distribute triangles across MPI ranks
    size_t rowsPerRank = N / static_cast<size_t>(numRanks);
    size_t remainder = N % static_cast<size_t>(numRanks);
    size_t rowStart = static_cast<size_t>(rank) * rowsPerRank + std::min(static_cast<size_t>(rank), remainder);
    size_t rowEnd = (static_cast<size_t>(rank) + 1) * rowsPerRank + std::min(static_cast<size_t>(rank) + 1, remainder);
    size_t localRows = rowEnd - rowStart;

    // Allocate device memory
    float* d_radB = nullptr;
    float* d_distances = nullptr;

    cudaMalloc((void**)&d_radB, T * N * sizeof(float));
    cudaMalloc((void**)&d_distances, N * sizeof(float));

    cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (int)((localRows + blockSize - 1) / blockSize);

    distanceKernel<<<numBlocks, blockSize>>>(
        d_radB, d_distances, N, T, state.sourceIndex, rowStart, rowEnd);
    cudaDeviceSynchronize();

    // Download results
    cudaMemcpy(state.distances.data(), d_distances, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_radB);
    cudaFree(d_distances);

    // Gather results from all ranks
    {
        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            size_t rStart = static_cast<size_t>(r) * rowsPerRank + std::min(static_cast<size_t>(r), remainder);
            size_t rEnd = (static_cast<size_t>(r) + 1) * rowsPerRank + std::min(static_cast<size_t>(r) + 1, remainder);
            recvCounts[r] = static_cast<int>(rEnd - rStart);
            displs[r] = static_cast<int>(rStart);
        }

        MPI_Allgatherv(state.distances.data() + rowStart, static_cast<int>(localRows), MPI_FLOAT,
                       state.distances.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
    }

    if (rank == 0) printf("  Distance computation complete on %d ranks\n", numRanks);
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

    #pragma omp parallel for reduction(min: minDist) reduction(max: maxDist) \
        reduction(+: sumDist, nonZeroCount) schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
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
    #pragma omp parallel for reduction(+: receivedEnergy) schedule(static)
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
    #pragma omp parallel for reduction(+: nonZeroKij) schedule(static)
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
    #pragma omp parallel for reduction(^: hash) schedule(static)
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, but only rank 0 prints)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n", numRanks);
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

    if (rank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;

    if (rank == 0) {
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

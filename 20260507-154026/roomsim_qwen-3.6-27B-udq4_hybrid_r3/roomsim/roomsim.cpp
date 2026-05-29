/**
 * Room Response Simulation Benchmark - Hybrid MPI + CUDA + OpenMP
 *
 * MPI:   distributes work across nodes/ranks
 * CUDA:  GPU-accelerated N×N kernels (form factors, time delays)
 * OpenMP: within-node threading (simulation, distance computation)
 */

#include <mpi.h>

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
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Host Vector and Triangle Types
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
// Triangle-Box Overlap Test (host-side, for octree construction)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };
    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;
    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;
    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0), p1 = axis.dot(v1), p2 = axis.dot(v2);
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
// Octree (host-side, for construction)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;
    Octree() : allTriangles(nullptr) {}

    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const;

    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const;

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;
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
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;
        buildNode(allIndices, minBound, maxBound);
    }
private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin; maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices; return;
        }
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];
        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
                if (triangleBoxOverlap(childCenter, childHalfSize, tri))
                    childIndices[i].push_back(idx);
            }
        }
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true; break;
            }
        }
        if (!canSplit) { triangleIndices = indices; return; }
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center, childMax = center;
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
};

// ============================================================================
// GPU Octree: Flattened representation
// ============================================================================

struct __align__(4) Vec3f {
    float x, y, z;
    __device__ Vec3f operator+(const Vec3f& o) const { return {x+o.x, y+o.y, z+o.z}; }
    __device__ Vec3f operator-(const Vec3f& o) const { return {x-o.x, y-o.y, z-o.z}; }
    __device__ Vec3f operator*(float s) const { return {x*s, y*s, z*s}; }
    __device__ Vec3f operator/(float s) const { return {x/s, y/s, z/s}; }
    __device__ Vec3f operator-() const { return {-x, -y, -z}; }
    __device__ float dot(const Vec3f& o) const { return x*o.x + y*o.y + z*o.z; }
    __device__ Vec3f cross(const Vec3f& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    __device__ float squaredNorm() const { return x*x + y*y + z*z; }
    __device__ float norm() const { return sqrtf(squaredNorm()); }
    __device__ Vec3f normalized() const {
        float n = norm();
        return n > 1e-6f ? *this / n : Vec3f{0,0,0};
    }
};

struct __align__(4) TriangleGPU {
    Vec3f a, b, c;
};

struct OctreeNodeGPU {
    Vec3f center;
    Vec3f halfExtent;
    int childOffsets[8];
    int triStart;
    int triCount;
};

void flattenOctree(const Octree& octree,
                   std::vector<OctreeNodeGPU>& nodes,
                   std::vector<uint32_t>& triIndices) {
    std::map<const Octree*, int> ptrToIdx;
    std::vector<const Octree*> queue;
    queue.push_back(&octree);
    int ci = 0;
    while (!queue.empty()) {
        const Octree* node = queue.back(); queue.pop_back();
        ptrToIdx[node] = ci++;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) queue.push_back(node->children[i].get());
        }
    }
    nodes.resize(ci);
    // Second BFS to fill nodes in same order
    std::vector<const Octree*> queue2;
    queue2.push_back(&octree);
    int ni = 0;
    while (!queue2.empty()) {
        const Octree* node = queue2.back(); queue2.pop_back();
        OctreeNodeGPU& gn = nodes[ni];
        gn.center = {node->center.x, node->center.y, node->center.z};
        gn.halfExtent = {node->halfExtent.x, node->halfExtent.y, node->halfExtent.z};
        gn.triStart = static_cast<int>(triIndices.size());
        gn.triCount = static_cast<int>(node->triangleIndices.size());
        for (int i = 0; i < 8; ++i) gn.childOffsets[i] = -1;
        for (const auto& ti : node->triangleIndices)
            triIndices.push_back(static_cast<uint32_t>(ti));
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                gn.childOffsets[i] = ptrToIdx[node->children[i].get()];
                queue2.push_back(node->children[i].get());
            }
        }
        ni++;
    }
}

// ============================================================================
// CUDA Device Functions
// ============================================================================

__device__ float rayTriangleIntersectD(const Vec3f& orig, const Vec3f& dir,
                                        const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f e1 = v1 - v0;
    Vec3f e2 = v2 - v0;
    Vec3f pvec = dir.cross(e2);
    float det = e1.dot(pvec);
    if (fabsf(det) < 1e-6f) return 1e30f;
    float invDet = 1.0f / det;
    Vec3f tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;
    Vec3f qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;
    return e2.dot(qvec) * invDet;
}

__device__ bool deviceTriangleBoxOverlap(const Vec3f& boxCenter, const Vec3f& boxHalfSize,
                                          const TriangleGPU& tri) {
    Vec3f v0 = tri.a - boxCenter;
    Vec3f v1 = tri.b - boxCenter;
    Vec3f v2 = tri.c - boxCenter;
    Vec3f e0 = v1 - v0;
    Vec3f e1 = v2 - v1;
    Vec3f e2 = v0 - v2;
    // AABB axes
    float mn = v0.x, mx = v0.x;
    if (v1.x < mn) mn = v1.x; if (v1.x > mx) mx = v1.x;
    if (v2.x < mn) mn = v2.x; if (v2.x > mx) mx = v2.x;
    if (mn > boxHalfSize.x || mx < -boxHalfSize.x) return false;
    mn = v0.y; mx = v0.y;
    if (v1.y < mn) mn = v1.y; if (v1.y > mx) mx = v1.y;
    if (v2.y < mn) mn = v2.y; if (v2.y > mx) mx = v2.y;
    if (mn > boxHalfSize.y || mx < -boxHalfSize.y) return false;
    mn = v0.z; mx = v0.z;
    if (v1.z < mn) mn = v1.z; if (v1.z > mx) mx = v1.z;
    if (v2.z < mn) mn = v2.z; if (v2.z > mx) mx = v2.z;
    if (mn > boxHalfSize.z || mx < -boxHalfSize.z) return false;
    Vec3f triNormal = e0.cross(e1);
    float d = triNormal.dot(v0);
    float r = boxHalfSize.x * fabsf(triNormal.x) +
              boxHalfSize.y * fabsf(triNormal.y) +
              boxHalfSize.z * fabsf(triNormal.z);
    if (fabsf(d) > r) return false;
    Vec3f axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3f edges[3] = {e0, e1, e2};
    for (int a = 0; a < 3; a++) {
        for (int e = 0; e < 3; e++) {
            Vec3f crossAxis = axes[a].cross(edges[e]);
            if (crossAxis.squaredNorm() > 1e-6f) {
                float r2 = boxHalfSize.x * fabsf(crossAxis.x) +
                           boxHalfSize.y * fabsf(crossAxis.y) +
                           boxHalfSize.z * fabsf(crossAxis.z);
                float ap0 = crossAxis.dot(v0), ap1 = crossAxis.dot(v1), ap2 = crossAxis.dot(v2);
                float mn2 = ap0, mx2 = ap0;
                if (ap1 < mn2) mn2 = ap1; if (ap1 > mx2) mx2 = ap1;
                if (ap2 < mn2) mn2 = ap2; if (ap2 > mx2) mx2 = ap2;
                if (mn2 > r2 || mx2 < -r2) return false;
            }
        }
    }
    return true;
}

__device__ bool isRayBlockedD(const Vec3f& from, const Vec3f& to,
                               const OctreeNodeGPU* nodes, const uint32_t* triIndices,
                               const TriangleGPU* triangles,
                               int rootIdx,
                               uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3f dir = to - from;
    float rayLen = dir.norm();
    if (rayLen < 1e-6f) return true;
    Vec3f dirNorm = dir / rayLen;
    int stack[64];
    int sp = 0;
    stack[sp++] = rootIdx;
    while (sp > 0) {
        int nodeIdx = stack[--sp];
        const OctreeNodeGPU& node = nodes[nodeIdx];
        Vec3f d = (to - from) * 0.5f;
        Vec3f c = from + d - node.center;
        Vec3f ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
        if (fabsf(c.x) > node.halfExtent.x + ad.x) continue;
        if (fabsf(c.y) > node.halfExtent.y + ad.y) continue;
        if (fabsf(c.z) > node.halfExtent.z + ad.z) continue;
        if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + 1e-6f) continue;
        if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + 1e-6f) continue;
        if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + 1e-6f) continue;
        if (node.triCount > 0) {
            for (int ti = 0; ti < node.triCount; ti++) {
                uint32_t idx = triIndices[node.triStart + ti];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const TriangleGPU& tri = triangles[idx];
                float dist = rayTriangleIntersectD(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > 1e-6f && dist < rayLen - 1e-6f) return true;
            }
        } else {
            for (int i = 0; i < 8; i++) {
                if (node.childOffsets[i] >= 0) {
                    if (sp < 64) stack[sp++] = node.childOffsets[i];
                }
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernel: Compute Form Factors (Kij) with offset
// ============================================================================

__global__ void computeKijKernelOffset(
    const TriangleGPU* d_triangles,
    const OctreeNodeGPU* d_octreeNodes,
    const uint32_t* d_triIndices,
    const float* d_rngValues,
    float* d_kij,
    int N,
    int rootIdx,
    int startOffset,
    int count)
{
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= count) return;

    int globalIdx = startOffset + localIdx;
    int i = globalIdx / N;
    int j = globalIdx % N;
    if (i == j) { d_kij[localIdx] = 0.0f; return; }

    const TriangleGPU& triI = d_triangles[i];
    const TriangleGPU& triJ = d_triangles[j];

    Vec3f normalI = (triI.b - triI.a).cross(triI.c - triI.a).normalized();
    Vec3f normalJ = (triJ.b - triJ.a).cross(triJ.c - triJ.a).normalized();
    if (normalI.dot(normalJ) > 0.99f) { d_kij[localIdx] = 0.0f; return; }

    // Compute RNG offset for this (i,j) pair
    // In sequential code: for i, for j (skip i==j), 2*NUM_RAYS values each
    int rngOffset = (i * (N - 1) + (j < i ? j : j - 1)) * (2 * NUM_RAYS);

    float kij = 0.0f;

    for (int r = 0; r < NUM_RAYS; r++) {
        float u1 = d_rngValues[rngOffset + r * 2];
        float v1 = d_rngValues[rngOffset + r * 2 + 1];
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        Vec3f pI = triI.a + (triI.b - triI.a) * u1 + (triI.c - triI.a) * v1;

        float u2 = d_rngValues[rngOffset + (r + NUM_RAYS) * 2];
        float v2 = d_rngValues[rngOffset + (r + NUM_RAYS) * 2 + 1];
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        Vec3f pJ = triJ.a + (triJ.b - triJ.a) * u2 + (triJ.c - triJ.a) * v2;

        if (isRayBlockedD(pI, pJ, d_octreeNodes, d_triIndices, d_triangles, rootIdx,
                          static_cast<uint32_t>(i), static_cast<uint32_t>(j)))
            continue;

        Vec3f v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < 1e-6f) continue;

        float vNorm = v.norm();
        float cosPhiI = fmaxf(0.0f, v.dot(normalI) / vNorm);
        float cosPhiJ = fmaxf(0.0f, (-v).dot(normalJ) / vNorm);

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    d_kij[localIdx] = kij * INV_NUM_RAYS;
}

// ============================================================================
// CUDA Kernel: Compute Time Delays (Tau) with offset
// ============================================================================

__global__ void computeTauKernelOffset(
    const TriangleGPU* d_triangles,
    int* d_tau,
    int N,
    int startOffset,
    int count)
{
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= count) return;

    int globalIdx = startOffset + localIdx;
    int i = globalIdx / N;
    int j = globalIdx % N;
    if (i == j) { d_tau[localIdx] = 0; return; }

    Vec3f centerI = (d_triangles[i].a + d_triangles[i].b + d_triangles[i].c) / 3.0f;
    Vec3f centerJ = (d_triangles[j].a + d_triangles[j].b + d_triangles[j].c) / 3.0f;
    Vec3f diff = centerI - centerJ;
    float dist = diff.norm();
    d_tau[localIdx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;
    IcosphereMesh(int subdivisions, val_t radius) {
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
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };
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
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
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

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;
    Octree octree;
    size_t sourceIndex;
    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Pre-generate RNG values matching sequential code order
// ============================================================================

void pregenerateRNG(std::vector<float>& rngValues, size_t N) {
    RandomGenerator rng(42);
    size_t totalRNG = N * (N - 1) * 2 * NUM_RAYS;
    rngValues.resize(totalRNG);
    size_t offset = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            for (int r = 0; r < NUM_RAYS; r++) {
                rngValues[offset++] = rng.rand();
                rngValues[offset++] = rng.rand();
            }
        }
    }
}

// ============================================================================
// Host-side computeKij (for reference, used in octree construction path)
// ============================================================================

Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
}

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
        if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        return false;
    });
}

val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];
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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// Octree ray traversal (needed for host-side reference)
template<typename Func>
bool Octree::applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
    if (!triangleIndices.empty()) {
        for (size_t idx : triangleIndices) {
            if (func(idx, (*allTriangles)[idx])) return true;
        }
        return false;
    }
    for (int i = 0; i < 8; ++i) {
        if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
            if (children[i]->applyToTris(p1, p2, std::forward<Func>(func))) return true;
        }
    }
    return false;
}

bool Octree::rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
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

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.octree.build(state.triangles);
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i)
        state.areas[i] = state.triangles[i].area();
    state.rho.resize(state.numTriangles, reflectivity);
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
    size_t timeOn = 0, timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t)
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
}

// ============================================================================
// Compute Form Factors with MPI + CUDA
// ============================================================================

void computeFormFactors(SimulationState& state, int rank, int numRanks) {
    size_t N = state.numTriangles;
    size_t totalPairs = N * N;

    // Pre-generate RNG values (same order as sequential code)
    std::vector<float> rngValues;
    pregenerateRNG(rngValues, N);

    // Flatten octree for GPU
    std::vector<OctreeNodeGPU> octreeNodes;
    std::vector<uint32_t> triIndices;
    flattenOctree(state.octree, octreeNodes, triIndices);

    // Build GPU triangles
    std::vector<TriangleGPU> gpuTriangles(N);
    for (size_t i = 0; i < N; ++i) {
        gpuTriangles[i] = {
            {state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z},
            {state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z},
            {state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z}
        };
    }

    // Distribute pairs across MPI ranks
    size_t pairsPerRank = totalPairs / numRanks;
    size_t remainder = totalPairs % numRanks;
    size_t myStart = rank * pairsPerRank + std::min((size_t)rank, remainder);
    size_t myCount = pairsPerRank + (rank < remainder ? 1 : 0);

    if (rank == 0) printf("Computing form factors (Kij) with CUDA+MPI...\n");

    if (myCount == 0) {
        std::vector<int> recvcounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvcounts[r] = pairsPerRank + (r < remainder ? 1 : 0);
            int disp = 0;
            for (int rr = 0; rr < r; ++rr)
                disp += pairsPerRank + (rr < remainder ? 1 : 0);
            displs[r] = disp;
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                       state.kij.data(), recvcounts.data(), displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        return;
    }

    // Allocate device memory
    TriangleGPU* d_triangles = nullptr;
    OctreeNodeGPU* d_octreeNodes = nullptr;
    uint32_t* d_triIndices = nullptr;
    float* d_rngValues = nullptr;
    float* d_kijLocal = nullptr;

    cudaMalloc(&d_triangles, N * sizeof(TriangleGPU));
    cudaMalloc(&d_octreeNodes, octreeNodes.size() * sizeof(OctreeNodeGPU));
    cudaMalloc(&d_triIndices, triIndices.size() * sizeof(uint32_t));
    cudaMalloc(&d_rngValues, rngValues.size() * sizeof(float));
    cudaMalloc(&d_kijLocal, myCount * sizeof(float));

    cudaMemcpy(d_triangles, gpuTriangles.data(), N * sizeof(TriangleGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_octreeNodes, octreeNodes.data(), octreeNodes.size() * sizeof(OctreeNodeGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triIndices, triIndices.data(), triIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rngValues, rngValues.data(), rngValues.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (int)((myCount + blockSize - 1) / blockSize);
    computeKijKernelOffset<<<numBlocks, blockSize>>>(
        d_triangles, d_octreeNodes, d_triIndices, d_rngValues,
        d_kijLocal, (int)N, 0, (int)myStart, (int)myCount);

    // Copy results back
    std::vector<float> localResults(myCount);
    cudaMemcpy(localResults.data(), d_kijLocal, myCount * sizeof(float), cudaMemcpyDeviceToHost);

    // Scatter results into global array at correct positions
    for (size_t li = 0; li < myCount; ++li) {
        size_t gi = myStart + li;
        int i = (int)(gi / N);
        int j = (int)(gi % N);
        state.kij[i * N + j] = localResults[li];
    }

    // Broadcast kij to all ranks
    MPI_Bcast(state.kij.data(), totalPairs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Cleanup GPU memory
    cudaFree(d_triangles);
    cudaFree(d_octreeNodes);
    cudaFree(d_triIndices);
    cudaFree(d_rngValues);
    cudaFree(d_kijLocal);
}

// ============================================================================
// Compute Time Delays with MPI + CUDA
// ============================================================================

void computeTimeDelays(SimulationState& state, int rank, int numRanks) {
    size_t N = state.numTriangles;
    size_t totalPairs = N * N;

    if (rank == 0) printf("Computing time delays (Tau) with CUDA+MPI...\n");

    // Build GPU triangles
    std::vector<TriangleGPU> gpuTriangles(N);
    for (size_t i = 0; i < N; ++i) {
        gpuTriangles[i] = {
            {state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z},
            {state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z},
            {state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z}
        };
    }

    // Distribute pairs across MPI ranks
    size_t pairsPerRank = totalPairs / numRanks;
    size_t remainder = totalPairs % numRanks;
    size_t myStart = rank * pairsPerRank + std::min((size_t)rank, remainder);
    size_t myCount = pairsPerRank + (rank < remainder ? 1 : 0);

    if (myCount == 0) {
        MPI_Bcast(state.tau.data(), totalPairs, MPI_INT, 0, MPI_COMM_WORLD);
        return;
    }

    TriangleGPU* d_triangles = nullptr;
    int* d_tauLocal = nullptr;
    cudaMalloc(&d_triangles, N * sizeof(TriangleGPU));
    cudaMalloc(&d_tauLocal, myCount * sizeof(int));
    cudaMemcpy(d_triangles, gpuTriangles.data(), N * sizeof(TriangleGPU), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (int)((myCount + blockSize - 1) / blockSize);
    computeTauKernelOffset<<<numBlocks, blockSize>>>(
        d_triangles, d_tauLocal, (int)N, (int)myStart, (int)myCount);

    std::vector<int> localResults(myCount);
    cudaMemcpy(localResults.data(), d_tauLocal, myCount * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t li = 0; li < myCount; ++li) {
        size_t gi = myStart + li;
        int i = (int)(gi / N);
        int j = (int)(gi % N);
        state.tau[i * N + j] = localResults[li];
    }

    MPI_Bcast(state.tau.data(), totalPairs, MPI_INT, 0, MPI_COMM_WORLD);

    cudaFree(d_triangles);
    cudaFree(d_tauLocal);
}

// ============================================================================
// Simulation Phase with OpenMP + MPI
// ============================================================================

void runSimulation(SimulationState& state) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Running wave propagation simulation with OpenMP+MPI...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Distribute triangles across ranks
    size_t trisPerRank = N / numRanks;
    size_t remainder = N % numRanks;
    size_t myStart = rank * trisPerRank + std::min((size_t)rank, remainder);
    size_t myCount = trisPerRank + (rank < remainder ? 1 : 0);

    // Pre-compute recvcounts/displs for radB gather
    std::vector<int> radB_recvcounts(numRanks);
    std::vector<int> radB_displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        radB_recvcounts[r] = trisPerRank + (r < remainder ? 1 : 0);
        int disp = 0;
        for (int rr = 0; rr < r; ++rr)
            disp += trisPerRank + (rr < remainder ? 1 : 0);
        radB_displs[r] = disp;
    }

    for (size_t t = 0; t < T; ++t) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (ssize_t mi = 0; mi < (ssize_t)myCount; ++mi) {
            size_t i = myStart + mi;
            val_t sumB = ZERO;
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                int tauij = state.tau[state.idx2d(i, j)];
                if (static_cast<int>(t) < tauij) continue;
                val_t kij = state.kij[state.idx2d(i, j)];
                if (kij <= ZERO) continue;
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radB values across all ranks
        MPI_Allgatherv(state.radB.data() + state.idxTN(t, myStart), (int)myCount, MPI_FLOAT,
                       state.radB.data() + state.idxTN(t, 0),
                       radB_recvcounts.data(), radB_displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            if (rank == 0) printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation with OpenMP + MPI
// ============================================================================

void computeDistances(SimulationState& state) {
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) printf("Computing distances via cross-correlation with OpenMP+MPI...\n");

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    size_t trisPerRank = N / numRanks;
    size_t remainder = N % numRanks;
    size_t myStart = rank * trisPerRank + std::min((size_t)rank, remainder);
    size_t myCount = trisPerRank + (rank < remainder ? 1 : 0);

    #pragma omp parallel for schedule(dynamic, 16)
    for (ssize_t mi = 0; mi < (ssize_t)myCount; ++mi) {
        size_t i = myStart + mi;
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

    // Gather all distances
    std::vector<int> recvcounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        recvcounts[r] = trisPerRank + (r < remainder ? 1 : 0);
        int disp = 0;
        for (int rr = 0; rr < r; ++rr)
            disp += trisPerRank + (rr < remainder ? 1 : 0);
        displs[r] = disp;
    }

    MPI_Gatherv(state.distances.data() + myStart, (int)myCount, MPI_FLOAT,
                state.distances.data(), recvcounts.data(), displs.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) { allNonNegative = false; printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d); }
        if (!std::isfinite(d)) { printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d); return false; }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }
    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) { receivedEnergy++; break; }
        }
    }
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
    if (receivedEnergy == 0) { printf("  ERROR: No triangles received energy - simulation failed\n"); return false; }
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i)
        if (state.kij[i] > EPSILON) nonZeroKij++;
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
    if (nonZeroKij == 0) { printf("  ERROR: All form factors are zero - visibility computation failed\n"); return false; }
    if (!allNonNegative) return false;
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
// Helper: subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) targetTriangles = atoi(argv[++i]);
            else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) timesteps = atoi(argv[++i]);
            else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sourceIdx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reflectivity = static_cast<val_t>(atof(argv[++i]));
            else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-o") == 0) printResults = true;
            else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); MPI_Finalize(); return 0; }
            else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); MPI_Finalize(); return 1; }
        }
    }

    MPI_Bcast(&targetTriangles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&timesteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sourceIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflectivity, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
        printf("\n");
    }

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();
    computeTimeDelays(state, rank, numRanks);
    computeFormFactors(state, rank, numRanks);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) { printf("Precomputation time: %ld ms\n", preDuration); printf("\n"); }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulation(state);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) { printf("Simulation time: %ld ms\n", simDuration); printf("\n"); }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistances(state);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) { printf("Distance computation time: %ld ms\n", distDuration); printf("\n"); }

    if (rank == 0) {
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);
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
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
        if (validate) {
            if (!validateResults(state)) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}

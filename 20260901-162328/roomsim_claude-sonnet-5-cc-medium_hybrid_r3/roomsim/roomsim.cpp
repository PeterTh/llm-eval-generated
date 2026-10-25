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
 *  - MPI partitions the receiver triangles (rows of the NxN matrices) into
 *    contiguous blocks across ranks. Every rank owns one GPU.
 *  - CUDA kernels perform all O(N^2) and O(N^2*T) heavy numerical work
 *    (form factors, time delays, wave propagation, cross-correlation) for the
 *    rows owned by each rank.
 *  - OpenMP parallelizes host-side preparation work: octree construction,
 *    geometry flattening for the GPU, and reduction-heavy post-processing
 *    (hash/validation).
 *  - Because the wave-propagation recurrence couples all triangles at each
 *    timestep, ranks exchange their freshly computed radiosity rows via
 *    MPI_Allgatherv once per timestep so every rank's GPU has the data it
 *    needs for the next step.
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
#include <omp.h>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err__));                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types (host-side geometry construction)
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
// Octree for Spatial Acceleration (built redundantly on every rank, in
// parallel via OpenMP tasks since the mesh is deterministic and cheap to
// reproduce everywhere -- this avoids an MPI broadcast of the whole tree).
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;
constexpr size_t OCTREE_TASK_THRESHOLD = 256;  // min triangles to spawn an OpenMP task

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
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

#pragma omp parallel
        {
#pragma omp single
            { buildNode(allIndices, minBound, maxBound); }
        }
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

        // Subdivide into 8 children. Each child's candidate index list can be
        // computed independently (read-only scan over `indices`), so compute
        // all 8 buckets in parallel.
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 8; ++i) {
            Vec3 childCenter = center;
            childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
            childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
            childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

            for (size_t idx : indices) {
                const Triangle& tri = (*allTriangles)[idx];
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

        // Build children, spawning OpenMP tasks for large subtrees so sibling
        // subtrees are constructed concurrently.
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
                Octree* childPtr = children[i].get();
                const std::vector<size_t>& childIdx = childIndices[i];

                if (childIdx.size() >= OCTREE_TASK_THRESHOLD) {
#pragma omp task firstprivate(childPtr) shared(childIdx, childMin, childMax)
                    { childPtr->buildNode(childIdx, childMin, childMax); }
                } else {
                    childPtr->buildNode(childIdx, childMin, childMax);
                }
            }
        }
#pragma omp taskwait
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
};

// ============================================================================
// Flattened Octree (device-friendly representation)
// ============================================================================

struct FlatOctreeNode {
    float cx, cy, cz;
    float hx, hy, hz;
    int children[8];
    int isLeaf;
    int triStart;
    int triCount;
};

// Recursively flattens the (host-built) Octree into index-based arrays that
// can be uploaded to the GPU and traversed iteratively in CUDA kernels.
int flattenOctree(const Octree& node, std::vector<FlatOctreeNode>& nodes, std::vector<int>& triIdx) {
    int myIndex = static_cast<int>(nodes.size());
    nodes.emplace_back();

    FlatOctreeNode entry{};
    entry.cx = node.center.x; entry.cy = node.center.y; entry.cz = node.center.z;
    entry.hx = node.halfExtent.x; entry.hy = node.halfExtent.y; entry.hz = node.halfExtent.z;
    for (int c = 0; c < 8; ++c) entry.children[c] = -1;

    if (!node.triangleIndices.empty()) {
        entry.isLeaf = 1;
        entry.triStart = static_cast<int>(triIdx.size());
        entry.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) triIdx.push_back(static_cast<int>(idx));
    } else {
        entry.isLeaf = 0;
        entry.triStart = -1;
        entry.triCount = 0;
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                entry.children[c] = flattenOctree(*node.children[c], nodes, triIdx);
            }
        }
    }

    nodes[myIndex] = entry;
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
        triangles.resize(faces.size());
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            // Reverse winding to make normals point inward
            const auto& face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// GPU geometry representation
// ============================================================================

struct GPUTriangle {
    float3 a, b, c, n;
};

__host__ __device__ __forceinline__ float3 f3sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 f3add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3 f3scale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ __forceinline__ float f3dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __forceinline__ float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ __forceinline__ float f3normsq(float3 a) { return f3dot(a, a); }

// Splitmix64-style mixing used to derive an independent, deterministic RNG
// seed for every (i, j) triangle pair so that form-factor sampling is fully
// parallel (and reproducible) regardless of how the pairs are distributed
// across MPI ranks / GPU threads.
__host__ __device__ __forceinline__ uint64_t pairSeed(uint64_t base, uint32_t i, uint32_t j) {
    uint64_t x = base ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL) ^
                 (static_cast<uint64_t>(j) * 0xBF58476D1CE4E5B9ULL + 0x94D049BB133111EBULL);
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// ============================================================================
// Device-side ray/octree traversal
// ============================================================================

__device__ __forceinline__ bool rayIntersectsBoxDevice(const FlatOctreeNode& node, float3 p1, float3 p2) {
    float3 d = f3scale(f3sub(p2, p1), 0.5f);
    float3 c = f3sub(f3add(p1, d), make_float3(node.cx, node.cy, node.cz));
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > node.hx + ad.x) return false;
    if (fabsf(c.y) > node.hy + ad.y) return false;
    if (fabsf(c.z) > node.hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.hy * ad.z + node.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.hz * ad.x + node.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.hx * ad.y + node.hy * ad.x + EPSILON) return false;

    return true;
}

__device__ __forceinline__ float rayTriangleIntersectDevice(float3 orig, float3 dir,
                                                              float3 v0, float3 v1, float3 v2) {
    float3 e1 = f3sub(v1, v0);
    float3 e2 = f3sub(v2, v0);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);

    if (fabsf(det) < EPSILON) return 3.4e38f;

    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.4e38f;

    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.4e38f;

    return f3dot(e2, qvec) * invDet;
}

#define OCTREE_STACK_SIZE 64

__device__ bool isRayBlockedDevice(float3 from, float3 to,
                                    const FlatOctreeNode* __restrict__ nodes,
                                    const int* __restrict__ triIdx,
                                    const GPUTriangle* __restrict__ tris,
                                    int srcIdx, int dstIdx) {
    float3 dir = f3sub(to, from);
    float rayLen = sqrtf(f3normsq(dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = f3scale(dir, 1.0f / rayLen);

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root, matches host code: root's own box is never tested

    while (sp > 0) {
        int ni = stack[--sp];
        const FlatOctreeNode& node = nodes[ni];

        if (node.isLeaf) {
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIdx[node.triStart + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const GPUTriangle& tri = tris[idx];
                float dist = rayTriangleIntersectDevice(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int childIdx = node.children[c];
                if (childIdx >= 0 && rayIntersectsBoxDevice(nodes[childIdx], from, to)) {
                    stack[sp++] = childIdx;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ float cosPhiDevice(float3 v, float3 normal) {
    float vNorm = sqrtf(f3normsq(v));
    if (vNorm <= EPSILON) return 0.0f;
    return fmaxf(0.0f, f3dot(v, normal) / vNorm);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Time delay (Tau) for a block of receiver rows [rowStart, rowStart+localRows)
__global__ void tauKernel(const float3* __restrict__ centers, int N, size_t rowStart, size_t localRows,
                           int* __restrict__ tauLocal) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = localRows * static_cast<size_t>(N);
    if (tid >= total) return;

    size_t li = tid / N;
    int j = static_cast<int>(tid % N);
    size_t i = rowStart + li;

    if (static_cast<int>(i) == j) { tauLocal[tid] = 0; return; }

    float3 d = f3sub(centers[i], centers[j]);
    float dist = sqrtf(f3normsq(d));
    tauLocal[tid] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Form factor (Kij) for a block of receiver rows [rowStart, rowStart+localRows)
__global__ void kijKernel(const GPUTriangle* __restrict__ tris, int N,
                           const FlatOctreeNode* __restrict__ nodes, const int* __restrict__ triIdx,
                           size_t rowStart, size_t localRows, float* __restrict__ kijLocal,
                           uint64_t baseSeed) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = localRows * static_cast<size_t>(N);
    if (tid >= total) return;

    size_t li = tid / N;
    int j = static_cast<int>(tid % N);
    int i = static_cast<int>(rowStart + li);

    if (i == j) { kijLocal[tid] = 0.0f; return; }

    GPUTriangle triI = tris[i];
    GPUTriangle triJ = tris[j];

    if (f3dot(triI.n, triJ.n) > 0.99f) { kijLocal[tid] = 0.0f; return; }

    curandStatePhilox4_32_10_t rng;
    curand_init(pairSeed(baseSeed, static_cast<uint32_t>(i), static_cast<uint32_t>(j)), 0, 0, &rng);

    float3 abI = f3sub(triI.b, triI.a);
    float3 acI = f3sub(triI.c, triI.a);
    float3 abJ = f3sub(triJ.b, triJ.a);
    float3 acJ = f3sub(triJ.c, triJ.a);

    float kij = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u1 = curand_uniform(&rng);
        float v1 = curand_uniform(&rng);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        float3 pI = f3add(triI.a, f3add(f3scale(abI, u1), f3scale(acI, v1)));

        float u2 = curand_uniform(&rng);
        float v2 = curand_uniform(&rng);
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        float3 pJ = f3add(triJ.a, f3add(f3scale(abJ, u2), f3scale(acJ, v2)));

        if (isRayBlockedDevice(pI, pJ, nodes, triIdx, tris, i, j)) continue;

        float3 v = f3sub(pJ, pI);
        float distSqr = f3normsq(v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = cosPhiDevice(v, triI.n);
        float cosPhiJ = cosPhiDevice(f3scale(v, -1.0f), triJ.n);
        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijLocal[tid] = kij * INV_NUM_RAYS;
}

// One wave-propagation timestep for local receiver rows.
__global__ void simStepKernel(const float* __restrict__ kijLocal, const int* __restrict__ tauLocal,
                               const float* __restrict__ areas, const float* __restrict__ rho,
                               const float* __restrict__ radBFull, const float* __restrict__ radE,
                               int N, size_t rowStart, size_t localRows, size_t t,
                               float* __restrict__ newRow) {
    size_t li = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= localRows) return;
    size_t i = rowStart + li;

    float sumB = 0.0f;
    const float* kijRow = kijLocal + li * static_cast<size_t>(N);
    const int* tauRow = tauLocal + li * static_cast<size_t>(N);

    for (int j = 0; j < N; ++j) {
        if (static_cast<size_t>(j) == i) continue;

        int tauij = tauRow[j];
        if (static_cast<long long>(t) < tauij) continue;

        float kij = kijRow[j];
        if (kij <= 0.0f) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        float radJ = radBFull[srcTime * static_cast<size_t>(N) + j];
        if (radJ <= 0.0f) continue;

        sumB += fminf(kij * areas[j], 1.0f) * radJ;
    }

    newRow[li] = rho[i] * sumB + radE[t * static_cast<size_t>(N) + i];
}

// Cross-correlation distance estimation for local rows.
__global__ void distKernel(const float* __restrict__ radBFull, int N, int T,
                            size_t rowStart, size_t localRows, int sourceIndex,
                            float* __restrict__ distOut) {
    size_t li = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (li >= localRows) return;
    size_t i = rowStart + li;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < T; ++tt) {
            float pB = radBFull[static_cast<size_t>(tt) * N + i];
            float pS = radBFull[static_cast<size_t>(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
    }

    distOut[li] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// MPI row decomposition
// ============================================================================

struct RowRange {
    size_t start;
    size_t count;
};

RowRange computeRowRange(int rank, int size, size_t N) {
    size_t base = N / static_cast<size_t>(size);
    size_t rem = N % static_cast<size_t>(size);
    RowRange rr;
    rr.start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    rr.count = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    return rr;
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle (replicated on every rank)
    std::vector<val_t> rho;         // Reflectivity (replicated on every rank)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, replicated)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (full, valid on rank 0)

    Octree octree;                  // Spatial acceleration structure (built redundantly per-rank)

    size_t sourceIndex;

    // MPI decomposition
    int rank = 0, worldSize = 1;
    size_t rowStart = 0, localRows = 0;

    // Host mirrors of locally-owned chunks of the NxN matrices (row-block, local rows x N)
    std::vector<val_t> kijLocalHost;  // only populated on demand (validation)

    // Device-resident data
    GPUTriangle* d_tris = nullptr;
    FlatOctreeNode* d_nodes = nullptr;
    int* d_triIdx = nullptr;
    float3* d_centers = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_kijLocal = nullptr;
    int* d_tauLocal = nullptr;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int worldSize) {
    state.rank = rank;
    state.worldSize = worldSize;

    // Generate mesh redundantly on every rank: the procedure is fully
    // deterministic, so reproducing it locally is cheaper than broadcasting it.
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration (OpenMP-parallel, redundant per rank)
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Row decomposition for this rank
    RowRange rr = computeRowRange(rank, worldSize, state.numTriangles);
    state.rowStart = rr.start;
    state.localRows = rr.count;

    // Initialize areas / reflectivity (replicated, OpenMP-parallel)
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }
    state.rho.assign(state.numTriangles, reflectivity);

    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // --- Upload geometry to the GPU owned by this rank ---
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available; this build requires a GPU.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    size_t N = state.numTriangles;

    // Flatten triangles and centers for the GPU
    std::vector<GPUTriangle> gpuTris(N);
    std::vector<float3> centers(N);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        gpuTris[i].a = make_float3(t.a.x, t.a.y, t.a.z);
        gpuTris[i].b = make_float3(t.b.x, t.b.y, t.b.z);
        gpuTris[i].c = make_float3(t.c.x, t.c.y, t.c.z);
        Vec3 n = t.normal();
        gpuTris[i].n = make_float3(n.x, n.y, n.z);
        Vec3 ctr = t.center();
        centers[i] = make_float3(ctr.x, ctr.y, ctr.z);
    }

    // Flatten octree
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<int> flatTriIdx;
    flatNodes.reserve(2 * N);
    flatTriIdx.reserve(N);
    flattenOctree(state.octree, flatNodes, flatTriIdx);

    CUDA_CHECK(cudaMalloc(&state.d_tris, N * sizeof(GPUTriangle)));
    CUDA_CHECK(cudaMemcpy(state.d_tris, gpuTris.data(), N * sizeof(GPUTriangle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_centers, N * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(state.d_centers, centers.data(), N * sizeof(float3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_nodes, flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, flatNodes.data(), flatNodes.size() * sizeof(FlatOctreeNode),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_triIdx, flatTriIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.d_triIdx, flatTriIdx.data(), flatTriIdx.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radE, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), timesteps * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&state.d_kijLocal, state.localRows * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tauLocal, state.localRows * N * sizeof(int)));
}

void freeDeviceResources(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodes);
    cudaFree(state.d_triIdx);
    cudaFree(state.d_centers);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_kijLocal);
    cudaFree(state.d_tauLocal);
}

// ============================================================================
// Precomputation Phase (GPU, row-distributed across MPI ranks)
// ============================================================================

constexpr int CUDA_BLOCK_SIZE = 256;

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;
    size_t total = state.localRows * N;
    if (total > 0) {
        size_t blocks = (total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        tauKernel<<<blocks, CUDA_BLOCK_SIZE>>>(state.d_centers, static_cast<int>(N), state.rowStart,
                                                state.localRows, state.d_tauLocal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    size_t N = state.numTriangles;
    size_t total = state.localRows * N;
    if (total > 0) {
        size_t blocks = (total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        kijKernel<<<blocks, CUDA_BLOCK_SIZE>>>(state.d_tris, static_cast<int>(N), state.d_nodes,
                                                state.d_triIdx, state.rowStart, state.localRows,
                                                state.d_kijLocal, 42ULL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (state.rank == 0) printf("  Progress: %zu/%zu triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation) -- MPI + CUDA
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    int N = static_cast<int>(state.numTriangles);
    int size = state.worldSize;

    // Every rank can derive the decomposition analytically, so no extra
    // communication is needed to build gather counts/displacements.
    std::vector<int> recvCounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        RowRange rr = computeRowRange(r, size, state.numTriangles);
        recvCounts[r] = static_cast<int>(rr.count);
        displs[r] = static_cast<int>(rr.start);
    }

    float* d_newRow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_newRow, state.localRows * sizeof(float)));
    std::vector<float> newRowHost(state.localRows);
    std::vector<float> fullRowHost(state.numTriangles);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows > 0) {
            size_t blocks = (state.localRows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            simStepKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
                state.d_kijLocal, state.d_tauLocal, state.d_areas, state.d_rho, state.d_radB,
                state.d_radE, N, state.rowStart, state.localRows, t, d_newRow);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(newRowHost.data(), d_newRow, state.localRows * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(state.localRows > 0 ? newRowHost.data() : nullptr, static_cast<int>(state.localRows),
                        MPI_FLOAT, fullRowHost.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                        MPI_COMM_WORLD);

        std::memcpy(&state.radB[state.idxTN(t, 0)], fullRowHost.data(), state.numTriangles * sizeof(float));
        CUDA_CHECK(cudaMemcpy(state.d_radB + t * state.numTriangles, fullRowHost.data(),
                              state.numTriangles * sizeof(float), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            if (state.rank == 0) printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    cudaFree(d_newRow);
}

// ============================================================================
// Distance Computation (Cross-Correlation) -- MPI + CUDA
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int size = state.worldSize;

    std::vector<int> recvCounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        RowRange rr = computeRowRange(r, size, state.numTriangles);
        recvCounts[r] = static_cast<int>(rr.count);
        displs[r] = static_cast<int>(rr.start);
    }

    std::vector<float> distLocalHost(std::max<size_t>(state.localRows, 1));

    if (state.localRows > 0) {
        float* d_dist = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dist, state.localRows * sizeof(float)));

        size_t blocks = (state.localRows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        distKernel<<<blocks, CUDA_BLOCK_SIZE>>>(state.d_radB, N, T, state.rowStart, state.localRows,
                                                 static_cast<int>(state.sourceIndex), d_dist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(distLocalHost.data(), d_dist, state.localRows * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_dist);
    }

    MPI_Gatherv(state.localRows > 0 ? distLocalHost.data() : nullptr, static_cast<int>(state.localRows),
                MPI_FLOAT, state.distances.data(), recvCounts.data(), displs.data(), MPI_FLOAT, 0,
                MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    bool verbose = (state.rank == 0);
    if (verbose) printf("\nValidation:\n");

    // Check that distances are non-negative. Distances are only gathered to
    // rank 0 by computeDistances(), so this check only runs there; other
    // ranks still participate in the collective calls further below.
    bool allNonNegative = true;
    bool nonFinite = false;
    if (verbose) {
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
                nonFinite = true;
                break;
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) nonZeroCount++;
        }

        if (!nonFinite) {
            printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
            printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
            printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

            // Check source distance is zero or very small
            val_t srcDist = state.distances[state.sourceIndex];
            if (srcDist > WAVE_SPEED * 2) {
                printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
            }
        }
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
#pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    if (verbose) printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        if (verbose) printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries). Gather the full
    // matrix from every rank's locally-owned rows only for this diagnostic.
    size_t N = state.numTriangles;
    int worldSize = state.worldSize;
    std::vector<int> recvCounts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        RowRange rr = computeRowRange(r, worldSize, N);
        recvCounts[r] = static_cast<int>(rr.count * N);
        displs[r] = static_cast<int>(rr.start * N);
    }

    state.kijLocalHost.resize(std::max<size_t>(state.localRows * N, 1));
    if (state.localRows > 0) {
        CUDA_CHECK(cudaMemcpy(state.kijLocalHost.data(), state.d_kijLocal, state.localRows * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<float> fullKij(N * N);
    MPI_Gatherv(state.localRows > 0 ? state.kijLocalHost.data() : nullptr,
                static_cast<int>(state.localRows * N), MPI_FLOAT, fullKij.data(), recvCounts.data(),
                displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    long long nonZeroKij = 0;
#pragma omp parallel for schedule(static) reduction(+:nonZeroKij)
    for (size_t i = 0; i < N * N; ++i) {
        if (fullKij[i] > EPSILON) nonZeroKij++;
    }
    if (verbose) {
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
               nonZeroKij, N * N, 100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(N * N));
    }

    if (nonZeroKij == 0) {
        if (verbose) printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (nonFinite || !allNonNegative) {
        return false;
    }

    if (verbose) printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
#pragma omp parallel for schedule(static) reduction(^:hash)
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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize, omp_get_max_threads());
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
                         static_cast<size_t>(sourceIdx), reflectivity, rank, worldSize);

    if (rank == 0) printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) { printf("Precomputation time: %ld ms\n", preDuration); printf("\n"); }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) { printf("Simulation time: %ld ms\n", simDuration); printf("\n"); }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) { printf("Distance computation time: %ld ms\n", distDuration); printf("\n"); }

    int exitCode = 0;

    if (rank == 0) {
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

        // Memory usage (aggregate, across all ranks)
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t) * static_cast<size_t>(worldSize);
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
    }

    // Validation
    if (validate) {
        bool ok = validateResults(state);
        if (rank == 0 && !ok) exitCode = 1;
    }

    freeDeviceResources(state);

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}

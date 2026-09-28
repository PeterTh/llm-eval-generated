/**
 * Room Response Simulation Benchmark
 *
 * Hybrid-parallel implementation (MPI + OpenMP + CUDA) of room impulse response
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
 *  - MPI: the triangle set is row-decomposed across ranks; each rank owns a
 *    contiguous block of rows of the Kij/Tau matrices and drives its own GPU
 *    (round-robin over the GPUs visible on its node). Per-timestep radiosity
 *    is exchanged with MPI_Allgatherv.
 *  - CUDA: the three dominant computations run as GPU kernels: the Monte-Carlo
 *    form-factor estimation (octree-accelerated visibility ray tracing over
 *    all owned (i,j) pairs), the per-timestep radiosity propagation (one block
 *    per owned row, reduction over all source triangles), and the distance
 *    cross-correlation. The octree is flattened into linear arrays for
 *    stack-based traversal on the GPU.
 *  - OpenMP: CPU-side O(N^2) time-delay (Tau) computation and host-side
 *    initialization/validation loops.
 *
 * Note on random numbers: the original sequential code consumed a single
 * mt19937 stream across all triangle pairs, which cannot be reproduced in
 * parallel. The Monte-Carlo form-factor sampling instead uses a deterministic
 * counter-based generator (splitmix64) seeded per (pair, ray), which yields
 * statistically equivalent uniform samples and results that are independent
 * of the number of ranks/threads.
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types (host-side geometry setup)
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
// Octree for Spatial Acceleration (built on host, flattened for the GPU)
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
};

// ----------------------------------------------------------------------------
// Flattened octree representation for stack-based traversal in CUDA kernels.
// A node is a leaf iff triCount > 0 (matching the pointer-based traversal,
// which treats any node with stored triangle indices as a leaf).
// ----------------------------------------------------------------------------

struct FlatOctreeNode {
    float cx, cy, cz;       // node center
    float hx, hy, hz;       // node half extent
    int child[8];           // child node indices, -1 if absent
    int triStart, triCount; // range into the flat triangle-index list
};

static int flattenOctree(const Octree& node,
                         std::vector<FlatOctreeNode>& nodes,
                         std::vector<int>& triIndices) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    FlatOctreeNode fn;
    fn.cx = node.center.x;  fn.cy = node.center.y;  fn.cz = node.center.z;
    fn.hx = node.halfExtent.x; fn.hy = node.halfExtent.y; fn.hz = node.halfExtent.z;
    fn.triStart = static_cast<int>(triIndices.size());
    fn.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t t : node.triangleIndices) triIndices.push_back(static_cast<int>(t));

    for (int c = 0; c < 8; ++c) {
        fn.child[c] = node.children[c]
            ? flattenOctree(*node.children[c], nodes, triIndices) : -1;
    }
    nodes[id] = fn;
    return id;
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
// Device Math Helpers
// ============================================================================

__device__ __forceinline__ float3 f3add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 f3sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 f3scale(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 f3neg(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float f3norm(float3 a) { return sqrtf(f3dot(a, a)); }

// ============================================================================
// Random Number Generation (counter-based, deterministic per pair/ray)
// ============================================================================

// splitmix64 step -> uniform float in [0, 1)
__device__ __forceinline__ float rngUniform(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= (z >> 31);
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ float3 randomPointInTriangleDev(float3 a, float3 b, float3 c,
                                                           float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = f3sub(b, a);
    float3 ac = f3sub(c, a);
    return f3add(a, f3add(f3scale(ab, u), f3scale(ac, v)));
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm, device)
// ============================================================================

__device__ __forceinline__ float rayTriangleIntersectDev(float3 orig, float3 dir,
                                                         float3 v0, float3 v1, float3 v2) {
    constexpr float MAXV = 3.402823466e+38f;
    float3 e1 = f3sub(v1, v0);
    float3 e2 = f3sub(v2, v0);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);

    if (fabsf(det) < EPSILON) return MAXV;

    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return MAXV;

    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return MAXV;

    return f3dot(e2, qvec) * invDet;
}

// ============================================================================
// Visibility Testing (flattened-octree traversal, device)
// ============================================================================

// Check if a ray segment intersects a node's bounding box
__device__ __forceinline__ bool rayIntersectsBoxDev(const FlatOctreeNode& n,
                                                    float3 p1, float3 p2) {
    float3 d = f3scale(f3sub(p2, p1), 0.5f);
    float3 c = f3sub(f3add(p1, d), make_float3(n.cx, n.cy, n.cz));
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > n.hx + ad.x) return false;
    if (fabsf(c.y) > n.hy + ad.y) return false;
    if (fabsf(c.z) > n.hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.hy * ad.z + n.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.hz * ad.x + n.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.hx * ad.y + n.hy * ad.x + EPSILON) return false;

    return true;
}

constexpr int OCTREE_STACK_SIZE = 64;

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlockedDev(float3 from, float3 to,
                                const FlatOctreeNode* __restrict__ nodes,
                                const int* __restrict__ nodeTriIndices,
                                const float3* __restrict__ triA,
                                const float3* __restrict__ triB,
                                const float3* __restrict__ triC,
                                int srcTriIdx, int dstTriIdx) {
    float3 dir = f3sub(to, from);
    float rayLen = f3norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = f3scale(dir, 1.0f / rayLen);

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root is visited without a box test (as in applyToTris)

    while (sp > 0) {
        const FlatOctreeNode n = nodes[stack[--sp]];

        if (n.triCount > 0) {
            // Leaf: test contained triangles
            for (int k = 0; k < n.triCount; ++k) {
                int idx = nodeTriIndices[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                float dist = rayTriangleIntersectDev(from, dirNorm,
                                                     triA[idx], triB[idx], triC[idx]);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Internal node: descend into children overlapping the ray
            for (int c = 0; c < 8; ++c) {
                int ch = n.child[c];
                if (ch >= 0 && rayIntersectsBoxDev(nodes[ch], from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - CUDA kernel over all owned (i, j) pairs
// ============================================================================

__device__ __forceinline__ float cosPhiDev(float3 v, float3 normal) {
    float vNorm = f3norm(v);
    if (vNorm <= EPSILON) return 0.0f;
    return fmaxf(0.0f, f3dot(v, normal) / vNorm);
}

__global__ void formFactorKernel(int numTriangles, int rowStart, int rowsLocal,
                                 const float3* __restrict__ triA,
                                 const float3* __restrict__ triB,
                                 const float3* __restrict__ triC,
                                 const float3* __restrict__ triN,
                                 const FlatOctreeNode* __restrict__ nodes,
                                 const int* __restrict__ nodeTriIndices,
                                 float* __restrict__ kijLocal) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(rowsLocal) * numTriangles;
    if (idx >= total) return;

    int iLocal = static_cast<int>(idx / numTriangles);
    int j = static_cast<int>(idx % numTriangles);
    int i = rowStart + iLocal;

    float result = 0.0f;
    // Cull the diagonal and triangles facing the same direction
    if (i != j && f3dot(triN[i], triN[j]) <= 0.99f) {
        float3 aI = triA[i], bI = triB[i], cI = triC[i], nI = triN[i];
        float3 aJ = triA[j], bJ = triB[j], cJ = triC[j], nJ = triN[j];

        // Deterministic per-pair RNG stream (independent of parallel layout)
        uint64_t rngState = (static_cast<uint64_t>(i) * numTriangles + j)
                            * 0x2545F4914F6CDD1DULL + 42ULL;

        float kij = 0.0f;
        for (int r = 0; r < NUM_RAYS; ++r) {
            float u1 = rngUniform(rngState);
            float v1 = rngUniform(rngState);
            float u2 = rngUniform(rngState);
            float v2 = rngUniform(rngState);
            float3 pI = randomPointInTriangleDev(aI, bI, cI, u1, v1);
            float3 pJ = randomPointInTriangleDev(aJ, bJ, cJ, u2, v2);

            if (isRayBlockedDev(pI, pJ, nodes, nodeTriIndices, triA, triB, triC, i, j)) continue;

            float3 v = f3sub(pJ, pI);
            float distSqr = f3dot(v, v);
            if (distSqr < EPSILON) continue;

            float cosPhiI = cosPhiDev(v, nI);
            float cosPhiJ = cosPhiDev(f3neg(v), nJ);

            if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

            kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
        }
        result = kij * INV_NUM_RAYS;
    }

    kijLocal[idx] = result;
}

// Count non-zero form factors (for validation) without a host-side copy
__global__ void countNonZeroKernel(const float* __restrict__ data, long long n,
                                   unsigned long long* __restrict__ count) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (long long k = idx; k < n; k += stride) {
        if (data[k] > EPSILON) ++local;
    }
    if (local) atomicAdd(count, local);
}

// ============================================================================
// Simulation Phase (Wave Propagation) - one block per owned row per timestep
// ============================================================================

constexpr int SIM_BLOCK_SIZE = 256;

__global__ void simulationStepKernel(int numTriangles, int rowStart, int t,
                                     const float* __restrict__ kijLocal,
                                     const int* __restrict__ tauLocal,
                                     const float* __restrict__ areas,
                                     const float* __restrict__ rho,
                                     const float* __restrict__ radE,
                                     float* __restrict__ radB) {
    int iLocal = blockIdx.x;
    int i = rowStart + iLocal;
    const float* kRow = kijLocal + static_cast<size_t>(iLocal) * numTriangles;
    const int* tauRow = tauLocal + static_cast<size_t>(iLocal) * numTriangles;

    float sumB = 0.0f;
    for (int j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tauRow[j];
        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float kij = kRow[j];
        if (kij <= 0.0f) continue;

        // Get radiosity from source triangle at time when emission occurred
        float radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= 0.0f) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    __shared__ float sdata[SIM_BLOCK_SIZE];
    sdata[threadIdx.x] = sumB;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        size_t out = static_cast<size_t>(t) * numTriangles + i;
        // Update radiosity: reflection + emission
        radB[out] = rho[i] * sdata[0] + radE[out];
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation) - one thread per owned triangle
// ============================================================================

__global__ void distanceKernel(int numTriangles, int numTimesteps,
                               int rowStart, int rowsLocal,
                               const float* __restrict__ radB,
                               const float* __restrict__ srcSeries,
                               float* __restrict__ distLocal) {
    int iLocal = blockIdx.x * blockDim.x + threadIdx.x;
    if (iLocal >= rowsLocal) return;
    int i = rowStart + iLocal;

    float maxCorr = 0.0f;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = 0; t < numTimesteps; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            float pS = srcSeries[tt - t];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distLocal[iLocal] = WAVE_SPEED * static_cast<float>(bestT);
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
    std::vector<int> tauLocal;      // Time delays (owned rows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI row decomposition of the triangle set
    int mpiRank = 0, mpiSize = 1;
    int rowStart = 0, rowsLocal = 0;
    std::vector<int> allCounts, allDispls;  // Allgatherv layout per rank

    // Device-resident data (per rank, on its assigned GPU)
    float3* dTriA = nullptr;
    float3* dTriB = nullptr;
    float3* dTriC = nullptr;
    float3* dTriN = nullptr;
    FlatOctreeNode* dNodes = nullptr;
    int* dNodeTris = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dKijLocal = nullptr;
    int* dTauLocal = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (identically on every rank; the mesh is deterministic)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Row decomposition across MPI ranks
    int n = static_cast<int>(state.numTriangles);
    state.allCounts.resize(state.mpiSize);
    state.allDispls.resize(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        int base = n / state.mpiSize;
        int rem = n % state.mpiSize;
        state.allCounts[r] = base + (r < rem ? 1 : 0);
        state.allDispls[r] = r * base + std::min(r, rem);
    }
    state.rowStart = state.allDispls[state.mpiRank];
    state.rowsLocal = state.allCounts[state.mpiRank];

    // Build octree for spatial acceleration
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas and reflectivity
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (Kij/Tau hold only the rows owned by this rank)
    state.tauLocal.resize(static_cast<size_t>(state.rowsLocal) * state.numTriangles, 0);
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

// Select this rank's GPU (round-robin over the GPUs of its node) and upload
// all static simulation data.
void initializeDevice(SimulationState& state) {
    // Determine node-local rank for GPU assignment
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.mpiRank,
                        MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    size_t localMat = static_cast<size_t>(state.rowsLocal) * n;

    // Triangle geometry as structure-of-arrays
    std::vector<float3> hA(n), hB(n), hC(n), hN(n);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const Triangle& tri = state.triangles[i];
        hA[i] = make_float3(tri.a.x, tri.a.y, tri.a.z);
        hB[i] = make_float3(tri.b.x, tri.b.y, tri.b.z);
        hC[i] = make_float3(tri.c.x, tri.c.y, tri.c.z);
        Vec3 nm = tri.normal();
        hN[i] = make_float3(nm.x, nm.y, nm.z);
    }

    // Flatten the octree for GPU traversal
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<int> flatTris;
    flattenOctree(state.octree, flatNodes, flatTris);

    CUDA_CHECK(cudaMalloc(&state.dTriA, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&state.dTriB, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&state.dTriC, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&state.dTriN, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&state.dNodeTris, std::max<size_t>(1, flatTris.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dKijLocal, std::max<size_t>(1, localMat) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTauLocal, std::max<size_t>(1, localMat) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, t * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, t * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.dTriA, hA.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriB, hB.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriC, hC.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriN, hN.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, flatNodes.data(),
                          flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    if (!flatTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dNodeTris, flatTris.data(),
                              flatTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), t * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, t * n * sizeof(float)));
}

void freeDevice(SimulationState& state) {
    cudaFree(state.dTriA);
    cudaFree(state.dTriB);
    cudaFree(state.dTriC);
    cudaFree(state.dTriN);
    cudaFree(state.dNodes);
    cudaFree(state.dNodeTris);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dKijLocal);
    cudaFree(state.dTauLocal);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) on GPU...\n");

    long long total = static_cast<long long>(state.rowsLocal) * state.numTriangles;
    if (total > 0) {
        int blockSize = 128;
        long long numBlocks = (total + blockSize - 1) / blockSize;
        formFactorKernel<<<static_cast<unsigned>(numBlocks), blockSize>>>(
            static_cast<int>(state.numTriangles), state.rowStart, state.rowsLocal,
            state.dTriA, state.dTriB, state.dTriC, state.dTriN,
            state.dNodes, state.dNodeTris, state.dKijLocal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) with OpenMP...\n");

    size_t n = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (long long iLocal = 0; iLocal < state.rowsLocal; ++iLocal) {
        size_t i = static_cast<size_t>(state.rowStart + iLocal);
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            state.tauLocal[static_cast<size_t>(iLocal) * n + j] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }

    // Upload the owned Tau block for the GPU propagation kernels
    size_t localMat = static_cast<size_t>(state.rowsLocal) * n;
    if (localMat > 0) {
        CUDA_CHECK(cudaMemcpy(state.dTauLocal, state.tauLocal.data(),
                              localMat * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    int n = static_cast<int>(state.numTriangles);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Each rank computes radB[t][i] for its owned rows on the GPU. The
        // dependency radB[t - tau][j] with tau >= 1 keeps timesteps sequential.
        if (state.rowsLocal > 0) {
            simulationStepKernel<<<state.rowsLocal, SIM_BLOCK_SIZE>>>(
                n, state.rowStart, static_cast<int>(t),
                state.dKijLocal, state.dTauLocal, state.dAreas, state.dRho,
                state.dRadE, state.dRadB);
            CUDA_CHECK(cudaGetLastError());

            // Fetch the owned slice of this timestep's radiosity
            CUDA_CHECK(cudaMemcpy(state.radB.data() + state.idxTN(t, state.rowStart),
                                  state.dRadB + state.idxTN(t, state.rowStart),
                                  state.rowsLocal * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Exchange this timestep's radiosity across all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       state.radB.data() + state.idxTN(t, 0),
                       state.allCounts.data(), state.allDispls.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Push the assembled full timestep back to the device
        CUDA_CHECK(cudaMemcpy(state.dRadB + state.idxTN(t, 0),
                              state.radB.data() + state.idxTN(t, 0),
                              n * sizeof(float), cudaMemcpyHostToDevice));

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    size_t T = state.numTimesteps;

    // Radiosity time series of the source triangle (full radB is on every rank)
    std::vector<float> srcSeries(T);
    #pragma omp parallel for schedule(static)
    for (long long tt = 0; tt < static_cast<long long>(T); ++tt) {
        srcSeries[tt] = state.radB[state.idxTN(tt, state.sourceIndex)];
    }

    float* dSrc = nullptr;
    float* dDist = nullptr;
    CUDA_CHECK(cudaMalloc(&dSrc, T * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDist, std::max(1, state.rowsLocal) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dSrc, srcSeries.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    if (state.rowsLocal > 0) {
        int blockSize = 128;
        int numBlocks = (state.rowsLocal + blockSize - 1) / blockSize;
        distanceKernel<<<numBlocks, blockSize>>>(
            static_cast<int>(state.numTriangles), static_cast<int>(T),
            state.rowStart, state.rowsLocal, state.dRadB, dSrc, dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowStart, dDist,
                              state.rowsLocal * sizeof(float), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dSrc));
    CUDA_CHECK(cudaFree(dDist));

    // Assemble the full distance vector on every rank
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(),
                   state.allCounts.data(), state.allDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
    // The non-zero Kij count is reduced across ranks by the caller; everything
    // else (distances, radB) is fully replicated, so rank 0 validates alone.
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
    long long receivedEnergy = 0;
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %lld/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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

// Count this rank's non-zero form factors on the GPU
unsigned long long countLocalNonZeroKij(const SimulationState& state) {
    long long total = static_cast<long long>(state.rowsLocal) * state.numTriangles;
    if (total == 0) return 0;

    unsigned long long* dCount = nullptr;
    CUDA_CHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));

    int blockSize = 256;
    int numBlocks = static_cast<int>(std::min<long long>(4096, (total + blockSize - 1) / blockSize));
    countNonZeroKernel<<<numBlocks, blockSize>>>(state.dKijLocal, total, dCount);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long count = 0;
    CUDA_CHECK(cudaMemcpy(&count, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCount));
    return count;
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

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI rank(s) x %d OpenMP thread(s) + CUDA\n",
               mpiSize, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    initializeDevice(state);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long localDurations[3] = {preDuration, simDuration, distDuration};
    long maxDurations[3] = {0, 0, 0};
    MPI_Reduce(localDurations, maxDurations, 3, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    long totalTime = maxDurations[0] + maxDurations[1] + maxDurations[2];

    if (mpiRank == 0) {
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
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        unsigned long long localNonZero = countLocalNonZeroKij(state);
        unsigned long long nonZeroKij = 0;
        MPI_Allreduce(&localNonZero, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);

        int ok = 1;
        if (mpiRank == 0) {
            ok = validateResults(state, nonZeroKij) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) exitCode = 1;
    }

    freeDevice(state);
    MPI_Finalize();
    return exitCode;
}

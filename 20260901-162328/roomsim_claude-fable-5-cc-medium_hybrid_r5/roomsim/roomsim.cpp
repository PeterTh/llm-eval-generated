/**
 * Room Response Simulation Benchmark (Hybrid MPI + OpenMP + CUDA)
 *
 * This is a hybrid-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *  - MPI:    the N x N triangle-pair matrix is partitioned by rows across ranks;
 *            each rank drives one GPU (rank % deviceCount). Radiosity rows are
 *            exchanged with MPI_Allgatherv after every timestep.
 *  - CUDA:   form-factor Monte Carlo ray tracing (octree-accelerated), the
 *            per-timestep wave propagation update, and the cross-correlation
 *            distance estimation run as GPU kernels.
 *  - OpenMP: host-side O(N^2) work (time-delay matrix, validation scans,
 *            form-factor statistics) is threaded on the CPU.
 *
 * The Monte Carlo sampling uses a deterministic counter-based hash RNG seeded
 * per (i, j, ray) pair, so results are identical regardless of the number of
 * ranks/threads used.
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
constexpr uint64_t RNG_SEED = 42;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types (host)
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

// ============================================================================
// Flattened GPU data structures
// ============================================================================

struct DevTri {
    float3 a, b, c, n;
};

struct DevNode {
    float3 center, halfExtent;
    int children[8];   // -1 if absent
    int triStart;      // offset into flat leaf triangle index array
    int triCount;      // > 0 => leaf node
};

// Flatten the octree into contiguous arrays for stack-based GPU traversal.
static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<int>& leafTris) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.push_back(DevNode{});
    DevNode dn;
    dn.center = make_float3(node.center.x, node.center.y, node.center.z);
    dn.halfExtent = make_float3(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z);
    for (int i = 0; i < 8; ++i) dn.children[i] = -1;
    dn.triStart = static_cast<int>(leafTris.size());
    dn.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) leafTris.push_back(static_cast<int>(idx));
    if (dn.triCount == 0) {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                dn.children[i] = flattenOctree(*node.children[i], nodes, leafTris);
            }
        }
    }
    nodes[myIdx] = dn;
    return myIdx;
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
// Device math helpers
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
// Deterministic counter-based RNG (same value for a given (pair, counter)
// regardless of execution order / decomposition)
// ============================================================================

__device__ __forceinline__ float rnd01(uint64_t base, uint32_t ctr) {
    uint64_t s = (base + RNG_SEED) * 0x9E3779B97F4A7C15ULL +
                 (static_cast<uint64_t>(ctr) + 1ULL) * 0xBF58476D1CE4E5B9ULL;
    s ^= s >> 33; s *= 0xFF51AFD7ED558CCDULL;
    s ^= s >> 33; s *= 0xC4CEB9FE1A85EC53ULL;
    s ^= s >> 33;
    // 24 high-quality bits -> uniform float in [0, 1)
    return static_cast<float>(s >> 40) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ float3 randomPointInTriangle(const DevTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = f3sub(t.b, t.a);
    float3 ac = f3sub(t.c, t.a);
    return f3add(t.a, f3add(f3scale(ab, u), f3scale(ac, v)));
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm, device)
// ============================================================================

__device__ __forceinline__ float rayTriangleIntersect(float3 orig, float3 dir,
                                                      float3 v0, float3 v1, float3 v2) {
    float3 e1 = f3sub(v1, v0);
    float3 e2 = f3sub(v2, v0);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);

    const float FLT_MAX_V = 3.402823466e+38f;
    if (fabsf(det) < EPSILON) return FLT_MAX_V;

    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX_V;

    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_V;

    return f3dot(e2, qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, device)
// ============================================================================

// Check if a ray segment intersects a node's bounding box
__device__ __forceinline__ bool rayIntersectsBox(float3 p1, float3 p2, const DevNode& nd) {
    float3 d = f3scale(f3sub(p2, p1), 0.5f);
    float3 c = f3sub(f3add(p1, d), nd.center);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    float3 h = nd.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal with an explicit stack.
__device__ bool isRayBlocked(float3 from, float3 to,
                             const DevNode* nodes, const int* leafTris,
                             const DevTri* tris,
                             int srcTriIdx, int dstTriIdx) {
    float3 dir = f3sub(to, from);
    float rayLen = f3norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = f3scale(dir, 1.0f / rayLen);

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;  // root (its box is never tested, matching the original)

    while (sp > 0) {
        const DevNode nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            // Leaf: test triangles directly
            for (int k = 0; k < nd.triCount; ++k) {
                int idx = leafTris[nd.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const DevTri tri = tris[idx];
                float dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;  // Ray is blocked
            }
        } else {
            // Internal node: descend into children whose boxes the ray hits
            for (int i = 0; i < 8; ++i) {
                int ci = nd.children[i];
                if (ci >= 0 && rayIntersectsBox(from, to, nodes[ci])) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - GPU kernel, one thread per (i, j) pair
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ float cosPhi(float3 v, float3 normal) {
    float vNorm = f3norm(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, f3dot(v, normal) / vNorm);
}

__global__ void kijKernel(const DevTri* __restrict__ tris, int numTriangles,
                          const DevNode* __restrict__ nodes,
                          const int* __restrict__ leafTris,
                          int rowStart, long long numLocalPairs,
                          float* __restrict__ kijOut) {
    long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= numLocalPairs) return;

    int i = rowStart + static_cast<int>(p / numTriangles);
    int j = static_cast<int>(p % numTriangles);

    float kij = ZERO;
    if (i != j) {
        const DevTri triI = tris[i];
        const DevTri triJ = tris[j];

        // Cull triangles facing the same direction
        if (f3dot(triI.n, triJ.n) <= 0.99f) {
            uint64_t base = static_cast<uint64_t>(i) * numTriangles + j;

            for (int r = 0; r < NUM_RAYS; ++r) {
                float3 pI = randomPointInTriangle(triI, rnd01(base, r * 4 + 0), rnd01(base, r * 4 + 1));
                float3 pJ = randomPointInTriangle(triJ, rnd01(base, r * 4 + 2), rnd01(base, r * 4 + 3));

                if (isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) continue;

                float3 v = f3sub(pJ, pI);
                float distSqr = f3dot(v, v);
                if (distSqr < EPSILON) continue;

                float cosPhiI = cosPhi(v, triI.n);
                float cosPhiJ = cosPhi(f3neg(v), triJ.n);

                if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
            kij *= INV_NUM_RAYS;
        }
    }
    kijOut[p] = kij;
}

// ============================================================================
// Simulation Phase (Wave Propagation) - GPU kernel, one thread per local row
// ============================================================================

__global__ void simStepKernel(const float* __restrict__ kijLocal,
                              const int* __restrict__ tauLocal,
                              const float* __restrict__ areas,
                              const float* __restrict__ rho,
                              float* __restrict__ radB,
                              int t, int rowStart, int numLocalRows,
                              int numTriangles, int srcIdx, int timeOff) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= numLocalRows) return;
    int i = rowStart + li;

    const float* kijRow = kijLocal + static_cast<size_t>(li) * numTriangles;
    const int* tauRow = tauLocal + static_cast<size_t>(li) * numTriangles;

    float sumB = ZERO;
    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tauRow[j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float kij = kijRow[j];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        float radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission (source is on for t < timeOff)
    float radE = (i == srcIdx && t < timeOff) ? 1.0f : ZERO;
    radB[static_cast<size_t>(t) * numTriangles + i] = rho[i] * sumB + radE;
}

// ============================================================================
// Distance Computation (Cross-Correlation) - GPU kernel, one thread per row
// ============================================================================

__global__ void distKernel(const float* __restrict__ radB,
                           float* __restrict__ distances,
                           int rowStart, int numLocalRows,
                           int numTriangles, int numTimesteps, int srcIdx) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= numLocalRows) return;
    int i = rowStart + li;

    float maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = 0; t < numTimesteps; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            float pS = radB[static_cast<size_t>(tt - t) * numTriangles + srcIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[li] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Tau (time delay) Computation (host)
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
    std::vector<int> tau;           // Time delays (local rows x N, row-major)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, fully replicated)
    std::vector<val_t> distances;   // Computed distances from source (full, all ranks)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t timeOff;                 // Source emission stops at this timestep

    // MPI decomposition (contiguous row blocks of the pair matrix)
    int mpiRank = 0, mpiSize = 1;
    size_t rowStart = 0, rowEnd = 0, numLocalRows = 0;
    std::vector<int> rowCounts, rowDispls;  // per-rank row counts/offsets

    // Device buffers
    DevTri* d_tris = nullptr;
    DevNode* d_nodes = nullptr;
    int* d_leafTris = nullptr;
    float* d_kij = nullptr;       // local rows x N
    int* d_tau = nullptr;         // local rows x N
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radB = nullptr;      // T x N (full)
    float* d_distances = nullptr; // local rows

    size_t idx2dLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = (state.mpiRank == 0);

    // Generate mesh (deterministic, replicated on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.timeOff = timesteps / 2;  // Source emission active for first half of timesteps

    if (root) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);

    const size_t n = state.numTriangles;

    // MPI row decomposition of the N x N pair matrix
    state.rowStart = (n * static_cast<size_t>(state.mpiRank)) / state.mpiSize;
    state.rowEnd = (n * static_cast<size_t>(state.mpiRank + 1)) / state.mpiSize;
    state.numLocalRows = state.rowEnd - state.rowStart;
    state.rowCounts.resize(state.mpiSize);
    state.rowDispls.resize(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t rs = (n * static_cast<size_t>(r)) / state.mpiSize;
        size_t re = (n * static_cast<size_t>(r + 1)) / state.mpiSize;
        state.rowCounts[r] = static_cast<int>(re - rs);
        state.rowDispls[r] = static_cast<int>(rs);
    }

    // Initialize areas (OpenMP)
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    // Host matrices: tau only for local rows, radB fully replicated
    state.tau.resize(state.numLocalRows * n, 0);
    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // -------- Upload static data to this rank's GPU --------
    std::vector<DevTri> hostTris(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Triangle& tri = state.triangles[i];
        hostTris[i].a = make_float3(tri.a.x, tri.a.y, tri.a.z);
        hostTris[i].b = make_float3(tri.b.x, tri.b.y, tri.b.z);
        hostTris[i].c = make_float3(tri.c.x, tri.c.y, tri.c.z);
        Vec3 nrm = tri.normal();
        hostTris[i].n = make_float3(nrm.x, nrm.y, nrm.z);
    }

    std::vector<DevNode> hostNodes;
    std::vector<int> hostLeafTris;
    flattenOctree(state.octree, hostNodes, hostLeafTris);

    CUDA_CHECK(cudaMalloc(&state.d_tris, n * sizeof(DevTri)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, hostNodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&state.d_leafTris, std::max<size_t>(1, hostLeafTris.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, state.numLocalRows * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, state.numLocalRows * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, std::max<size_t>(1, state.numLocalRows) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.d_tris, hostTris.data(), n * sizeof(DevTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, hostNodes.data(), hostNodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    if (!hostLeafTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_leafTris, hostLeafTris.data(), hostLeafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * n * sizeof(float)));
}

void freeDeviceState(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodes);
    cudaFree(state.d_leafTris);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) on GPU...\n");

    const long long numLocalPairs =
        static_cast<long long>(state.numLocalRows) * static_cast<long long>(state.numTriangles);
    if (numLocalPairs > 0) {
        const int block = 128;
        const long long grid = (numLocalPairs + block - 1) / block;
        kijKernel<<<static_cast<unsigned>(grid), block>>>(
            state.d_tris, static_cast<int>(state.numTriangles),
            state.d_nodes, state.d_leafTris,
            static_cast<int>(state.rowStart), numLocalPairs, state.d_kij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (state.mpiRank == 0) {
        printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) with OpenMP...\n");

    const size_t n = state.numTriangles;
    // Each rank computes only its own rows of the tau matrix (OpenMP-threaded)
    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < state.numLocalRows; ++li) {
        size_t i = state.rowStart + li;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            state.tau[state.idx2dLocal(li, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }

    if (state.numLocalRows > 0) {
        CUDA_CHECK(cudaMemcpy(state.d_tau, state.tau.data(),
                              state.numLocalRows * n * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const int block = 128;
    const int grid = static_cast<int>((state.numLocalRows + block - 1) / block);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // radB[t][i] only depends on radB[t - tau][j] with tau >= 1, so all
        // rows within a timestep are independent: each rank updates its rows.
        if (state.numLocalRows > 0) {
            simStepKernel<<<grid, block>>>(
                state.d_kij, state.d_tau, state.d_areas, state.d_rho, state.d_radB,
                static_cast<int>(t), static_cast<int>(state.rowStart),
                static_cast<int>(state.numLocalRows), static_cast<int>(n),
                static_cast<int>(state.sourceIndex), static_cast<int>(state.timeOff));
            CUDA_CHECK(cudaGetLastError());

            // Bring this rank's slice of the new radiosity row to the host
            CUDA_CHECK(cudaMemcpy(state.radB.data() + state.idxTN(t, state.rowStart),
                                  state.d_radB + state.idxTN(t, state.rowStart),
                                  state.numLocalRows * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Exchange the completed timestep row among all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       state.radB.data() + state.idxTN(t, 0),
                       state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        // Push the fully assembled row back to the GPU for future timesteps
        CUDA_CHECK(cudaMemcpy(state.d_radB + state.idxTN(t, 0),
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
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation on GPU...\n");

    if (state.numLocalRows > 0) {
        const int block = 128;
        const int grid = static_cast<int>((state.numLocalRows + block - 1) / block);
        distKernel<<<grid, block>>>(
            state.d_radB, state.d_distances,
            static_cast<int>(state.rowStart), static_cast<int>(state.numLocalRows),
            static_cast<int>(state.numTriangles), static_cast<int>(state.numTimesteps),
            static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowStart, state.d_distances,
                              state.numLocalRows * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Assemble the full distance vector on every rank
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(),
                   state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Form factor statistics (for validation; distributed + OpenMP reduction)
// ============================================================================

long long countNonZeroKij(const SimulationState& state) {
    const size_t localSize = state.numLocalRows * state.numTriangles;
    std::vector<float> localKij(localSize);
    if (localSize > 0) {
        CUDA_CHECK(cudaMemcpy(localKij.data(), state.d_kij,
                              localSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    long long localCount = 0;
    #pragma omp parallel for schedule(static) reduction(+:localCount)
    for (size_t i = 0; i < localSize; ++i) {
        if (localKij[i] > EPSILON) localCount++;
    }

    long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return globalCount;
}

// ============================================================================
// Validation (rank 0; radB and distances are fully replicated)
// ============================================================================

bool validateResults(const SimulationState& state, long long nonZeroKij) {
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
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    const bool root = (mpiRank == 0);

    // Bind each rank to a GPU (round-robin over the devices visible per node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (root) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(mpiRank % deviceCount));

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks x %d OpenMP threads, %d CUDA device(s)/node\n",
               mpiSize, omp_get_max_threads(), deviceCount);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (root) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;

    if (root) {
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

        // Memory usage (per rank: local slabs of Kij/Tau plus replicated radiosity)
        size_t memKij = state.numLocalRows * n * sizeof(val_t);
        size_t memTau = state.numLocalRows * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (printResults && root) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        long long nonZeroKij = countNonZeroKij(state);
        int ok = 1;
        if (root) {
            ok = validateResults(state, nonZeroKij) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) exitCode = 1;
    }

    freeDeviceState(state);
    MPI_Finalize();
    return exitCode;
}

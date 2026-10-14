/**
 * Room Response Simulation Benchmark
 *
 * This is a simplified implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization (hybrid MPI + OpenMP + CUDA):
 *  - MPI: receiver triangles (rows of Kij and Tau) are block-partitioned across
 *    ranks. Each timestep's radiosity is all-gathered so every rank holds the
 *    full history needed by the wave propagation.
 *  - CUDA: every rank drives one GPU. Visibility ray casting (the dominant cost),
 *    time delays, wave propagation and cross-correlation run as kernels.
 *  - OpenMP: host-side back-face pre-screening of the triangle pairs.
 *
 * The ray samples come from one random stream (seed 42) consumed in the same
 * (i, j) order as the sequential code, so the results match it.
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

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Only the root rank writes to stdout
static int g_rank = 0;
#define ROOT_PRINTF(...)                                                       \
    do {                                                                       \
        if (g_rank == 0) printf(__VA_ARGS__);                                  \
    } while (0)

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
constexpr val_t NO_HIT = 3.40282347e+38f; // FLT_MAX: "no intersection" distance
constexpr size_t DRAWS_PER_PAIR = NUM_RAYS * 4;  // per ray: pI (u, v), pJ (u, v)
constexpr size_t PAIRS_PER_BATCH = size_t(1) << 19;  // pairs staged per GPU launch
constexpr size_t CULL_CHUNK_ROWS = 64;    // receiver rows pre-screened per OpenMP pass
constexpr size_t MAX_OCTREE_DEPTH = 32;   // bounds the device traversal stack
constexpr int OCTREE_STACK_SIZE = 256;    // >= 7 * MAX_OCTREE_DEPTH + 1

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
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
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

// Octree node in the flat layout used on the GPU (node 0 is the root)
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];   // index of child node, or -1
    uint32_t triBegin;  // leaf: first entry in the triangle index list
    uint32_t triCount;  // leaf: number of triangles (0 for inner nodes)
};

// Device view of the scene; passed to kernels by value
struct SceneView {
    const Triangle* tris;
    const DevNode* nodes;
    const idx_t* triIdx;
    const val_t* areas;
    const val_t* rho;
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

    // Flatten the tree (pre-order) into arrays the GPU traversal can read.
    // A node is a leaf exactly when triCount > 0, matching !triangleIndices.empty().
    void flatten(std::vector<DevNode>& nodes, std::vector<idx_t>& triList,
                 size_t depth, size_t& maxDepth) const {
        const size_t self = nodes.size();
        nodes.emplace_back();
        maxDepth = std::max(maxDepth, depth);

        DevNode node{};
        node.center = center;
        node.halfExtent = halfExtent;
        node.triBegin = static_cast<uint32_t>(triList.size());
        node.triCount = static_cast<uint32_t>(triangleIndices.size());
        for (size_t idx : triangleIndices) triList.push_back(static_cast<idx_t>(idx));
        for (int i = 0; i < 8; ++i) {
            node.child[i] = -1;
            if (children[i]) {
                node.child[i] = static_cast<int32_t>(nodes.size());
                children[i]->flatten(nodes, triList, depth + 1, maxDepth);
            }
        }
        nodes[self] = node;
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
    // Advance the stream past n variates. uniform_real_distribution<float> on
    // mt19937 consumes exactly one engine output per variate, so this skips
    // exactly what n calls to rand() would consume.
    void skip(size_t n) { rng.discard(n); }
};

// ============================================================================
// Device Geometry: Ray-Triangle Intersection and Visibility
// ============================================================================

HD inline val_t dMin(val_t a, val_t b) { return (b < a) ? b : a; }

__device__ inline val_t dRayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                              const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

// Check if the segment p1-p2 intersects a node's bounding box
__device__ inline bool dRayIntersectsBox(const DevNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - nd.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = nd.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
// (octree-accelerated; the answer does not depend on traversal order)
__device__ bool dIsRayBlocked(const Vec3& from, const Vec3& to, const SceneView& sv,
                              idx_t srcTriIdx, idx_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root
    while (sp > 0) {
        const DevNode& nd = sv.nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                idx_t idx = sv.triIdx[nd.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = sv.tris[idx];
                val_t dist = dRayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = nd.child[c];
                if (ch >= 0 && dRayIntersectsBox(sv.nodes[ch], from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

__device__ inline val_t dCosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;
}

// Point in triangle from two uniform samples (same mapping as the sequential code)
__device__ inline Vec3 dPointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Form factor Kij between receiver i and emitter j, using the 4*NUM_RAYS
// uniform samples prepared for this pair in draws
__device__ val_t dComputeKij(const SceneView& sv, idx_t i, idx_t j, const val_t* draws) {
    const Triangle& triI = sv.tris[i];
    const Triangle& triJ = sv.tris[j];

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* d = draws + 4 * r;
        Vec3 pI = dPointInTriangle(triI, d[0], d[1]);
        Vec3 pJ = dPointInTriangle(triJ, d[2], d[3]);

        if (dIsRayBlocked(pI, pJ, sv, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = dCosPhi(v, triI.normal());
        val_t cosPhiJ = dCosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// Time delay in timesteps between two triangles
__device__ inline int dComputeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Device Kernels
// ============================================================================

// Time delays for the local receiver rows: tau[il][j], row il = global row i0 + il
__global__ void computeTauKernel(SceneView sv, size_t i0, size_t nLocal, size_t n, int* tau) {
    size_t g = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (g >= nLocal * n) return;
    size_t i = i0 + g / n;
    size_t j = g % n;
    tau[g] = (i == j) ? 0 : dComputeTau(sv.tris[i], sv.tris[j]);
}

// Form factors for a batch of staged pairs; kij holds the local rows
__global__ void formFactorKernel(SceneView sv, const val_t* draws, const uint32_t* pairI,
                                 const uint32_t* pairJ, size_t count, size_t i0, size_t n,
                                 val_t* kij) {
    size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (p >= count) return;
    size_t il = pairI[p];
    size_t j = pairJ[p];
    kij[il * n + j] = dComputeKij(sv, static_cast<idx_t>(i0 + il), static_cast<idx_t>(j),
                                  draws + DRAWS_PER_PAIR * p);
}

// One wave-propagation timestep for the local receiver rows
__global__ void simulateStepKernel(SceneView sv, size_t t, size_t i0, size_t nLocal, size_t n,
                                   size_t timeOff, size_t src, const int* tau, const val_t* kij,
                                   val_t* radB) {
    size_t il = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (il >= nLocal) return;
    size_t i = i0 + il;

    val_t sumB = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tau[il * n + j];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t k = kij[il * n + j];
        if (k <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += dMin(k * sv.areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission (source emits in the first half)
    val_t radE = (t < timeOff && i == src) ? ONE : ZERO;
    radB[t * n + i] = sv.rho[i] * sumB + radE;
}

// Cross-correlation distance for the local receiver rows
__global__ void distanceKernel(size_t i0, size_t nLocal, size_t numTimesteps, size_t n,
                               size_t src, const val_t* radB, val_t* dist) {
    size_t il = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (il >= nLocal) return;
    size_t i = i0 + il;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[tt * n + i];
            val_t pS = radB[(tt - t) * n + src];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    dist[il] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countNonZeroKernel(const val_t* kij, size_t count, unsigned long long* out) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx < count && kij[idx] > EPSILON) atomicAdd(out, 1ULL);
}

// ============================================================================
// Device Memory Helpers
// ============================================================================

template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    if (count > 0) CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
T* toDevice(const std::vector<T>& host) {
    T* dev = deviceAlloc<T>(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev;
}

template <typename T>
void deviceFree(T* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

template <typename T>
T* pinnedAlloc(size_t count) {
    T* ptr = nullptr;
    if (count > 0) CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
}

inline unsigned int gridFor(size_t count, unsigned int block) {
    return static_cast<unsigned int>((count + block - 1) / block);
}

constexpr unsigned int BLOCK = 128;

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<DevNode> devNodes;  // Flattened octree (for the GPU)
    std::vector<idx_t> devTriIdx;   // Flattened octree triangle lists

    Octree octree;                  // Spatial acceleration structure (host)
};

// Block partition of triangle rows: rank r owns [rowBegin(r), rowBegin(r + 1))
size_t rowBegin(size_t n, int ranks, int r) {
    return n * static_cast<size_t>(r) / static_cast<size_t>(ranks);
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

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    state.octree.build(state.triangles);

    size_t maxDepth = 0;
    state.devNodes.clear();
    state.devTriIdx.clear();
    state.octree.flatten(state.devNodes, state.devTriIdx, 0, maxDepth);
    if (maxDepth >= MAX_OCTREE_DEPTH) {
        fprintf(stderr, "Octree depth %zu exceeds the traversal stack bound\n", maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(const SceneView& sv, size_t i0, size_t nLocal, size_t n, int* dTau) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");
    if (nLocal == 0) return;

    computeTauKernel<<<gridFor(nLocal * n, BLOCK), BLOCK>>>(sv, i0, nLocal, n, dTau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Two-slot staging so host-side ray sampling overlaps with GPU ray casting
struct FormFactorStage {
    val_t* draws = nullptr;     // pinned: DRAWS_PER_PAIR samples per staged pair
    uint32_t* pairI = nullptr;  // pinned: local receiver row
    uint32_t* pairJ = nullptr;  // pinned: emitter triangle
    val_t* dDraws = nullptr;
    uint32_t* dPairI = nullptr;
    uint32_t* dPairJ = nullptr;
    cudaStream_t stream = nullptr;
    size_t count = 0;
};

void launchFormFactors(FormFactorStage& s, const SceneView& sv, size_t i0, size_t n, val_t* dKij) {
    if (s.count == 0) return;
    CUDA_CHECK(cudaMemcpyAsync(s.dDraws, s.draws, sizeof(val_t) * DRAWS_PER_PAIR * s.count,
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.dPairI, s.pairI, sizeof(uint32_t) * s.count,
                               cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.dPairJ, s.pairJ, sizeof(uint32_t) * s.count,
                               cudaMemcpyHostToDevice, s.stream));
    formFactorKernel<<<gridFor(s.count, BLOCK), BLOCK, 0, s.stream>>>(
        sv, s.dDraws, s.dPairI, s.dPairJ, s.count, i0, n, dKij);
    CUDA_CHECK(cudaGetLastError());
}

// Computes the form factors of local rows [i0, i1) on this rank's GPU.
// Pairs are enumerated in the sequential order (i, then j) and consume the
// random stream exactly as the sequential code does; culled pairs consume none.
void computeFormFactors(const SimulationState& state, const SceneView& sv, size_t i0, size_t i1,
                        val_t* dKij) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const size_t nLocal = i1 - i0;
    RandomGenerator rng(42);

    FormFactorStage stage[2];
    for (auto& s : stage) {
        s.draws = pinnedAlloc<val_t>(DRAWS_PER_PAIR * PAIRS_PER_BATCH);
        s.pairI = pinnedAlloc<uint32_t>(PAIRS_PER_BATCH);
        s.pairJ = pinnedAlloc<uint32_t>(PAIRS_PER_BATCH);
        s.dDraws = deviceAlloc<val_t>(DRAWS_PER_PAIR * PAIRS_PER_BATCH);
        s.dPairI = deviceAlloc<uint32_t>(PAIRS_PER_BATCH);
        s.dPairJ = deviceAlloc<uint32_t>(PAIRS_PER_BATCH);
        CUDA_CHECK(cudaStreamCreate(&s.stream));
    }

    std::vector<std::vector<uint32_t>> visible(CULL_CHUNK_ROWS);
    size_t cur = 0;

    for (size_t a = 0; a < i1; a += CULL_CHUNK_ROWS) {
        const size_t b = std::min(a + CULL_CHUNK_ROWS, i1);

        // Back-face pre-screen of each receiver row in the chunk (OpenMP over rows)
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = a; i < b; ++i) {
            std::vector<uint32_t>& list = visible[i - a];
            list.clear();
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                // Cull triangles facing the same direction
                if (state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                list.push_back(static_cast<uint32_t>(j));
            }
        }

        // Sequential: consume the random stream in pair order and stage the pairs
        for (size_t i = a; i < b; ++i) {
            const std::vector<uint32_t>& list = visible[i - a];
            if (i < i0) {
                rng.skip(DRAWS_PER_PAIR * list.size());  // belongs to another rank
                continue;
            }

            for (uint32_t j : list) {
                if (stage[cur].count == PAIRS_PER_BATCH) {
                    launchFormFactors(stage[cur], sv, i0, n, dKij);
                    cur ^= 1;
                    CUDA_CHECK(cudaStreamSynchronize(stage[cur].stream));
                    stage[cur].count = 0;
                }

                FormFactorStage& s = stage[cur];
                val_t* out = s.draws + DRAWS_PER_PAIR * s.count;
                for (size_t k = 0; k < DRAWS_PER_PAIR; ++k) out[k] = rng.rand();
                s.pairI[s.count] = static_cast<uint32_t>(i - i0);
                s.pairJ[s.count] = j;
                ++s.count;
            }

            if (g_rank == 0 && ((i - i0 + 1) % 100 == 0 || i + 1 == i1)) {
                printf("  Progress: %zu/%zu triangles\n", i - i0 + 1, nLocal);
            }
        }
    }

    launchFormFactors(stage[cur], sv, i0, n, dKij);
    for (auto& s : stage) CUDA_CHECK(cudaStreamSynchronize(s.stream));

    for (auto& s : stage) {
        CUDA_CHECK(cudaStreamDestroy(s.stream));
        deviceFree(s.dPairJ);
        deviceFree(s.dPairI);
        deviceFree(s.dDraws);
        CUDA_CHECK(cudaFreeHost(s.pairJ));
        CUDA_CHECK(cudaFreeHost(s.pairI));
        CUDA_CHECK(cudaFreeHost(s.draws));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Advances the simulation one timestep at a time. Each rank computes its own rows,
// then all ranks exchange rows so the full radiosity history is available to all.
void runSimulation(const SimulationState& state, const SceneView& sv, size_t i0, size_t nLocal,
                   const int* dTau, const val_t* dKij, val_t* dRadB, std::vector<val_t>& radB,
                   const std::vector<int>& counts, const std::vector<int>& displs) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t timeOff = T / 2;
    const size_t src = state.sourceIndex;

    for (size_t t = 0; t < T; ++t) {
        if (nLocal > 0) {
            simulateStepKernel<<<gridFor(nLocal, BLOCK), BLOCK>>>(
                sv, t, i0, nLocal, n, timeOff, src, dTau, dKij, dRadB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(radB.data() + t * n + i0, dRadB + t * n + i0,
                                  nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        // Share this timestep's row across ranks, then refresh the device copy
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, radB.data() + t * n, counts.data(),
                       displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dRadB + t * n, radB.data() + t * n, n * sizeof(val_t),
                              cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// Computes the distances of the local rows and gathers all of them on the root
void computeDistances(const SimulationState& state, size_t i0, size_t nLocal, const val_t* dRadB,
                      val_t* dDist, const std::vector<int>& counts, const std::vector<int>& displs,
                      std::vector<val_t>& distances) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    std::vector<val_t> local(nLocal);
    if (nLocal > 0) {
        distanceKernel<<<gridFor(nLocal, BLOCK), BLOCK>>>(
            i0, nLocal, state.numTimesteps, state.numTriangles, state.sourceIndex, dRadB, dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), dDist, nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(local.data(), static_cast<int>(nLocal), MPI_FLOAT, distances.data(), counts.data(),
                displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const std::vector<val_t>& distances, const std::vector<val_t>& radB,
                     size_t n, size_t numTimesteps, size_t sourceIndex, size_t nonZeroKij) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < n; ++i) {
        val_t d = distances[i];
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);

    // Check source distance is zero or very small
    val_t srcDist = distances[sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t t = 0; t < numTimesteps; ++t) {
            if (radB[t * n + i] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, n);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
           nonZeroKij, n * n,
           100.0f * nonZeroKij / static_cast<val_t>(n * n));

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

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&distances[i]);
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
    ROOT_PRINTF("Usage: %s [options]\n", progName);
    ROOT_PRINTF("Options:\n");
    ROOT_PRINTF("  -n <num>     Target number of triangles (default: 320)\n");
    ROOT_PRINTF("               Actual count will be rounded to nearest icosphere level:\n");
    ROOT_PRINTF("               20, 80, 320, 1280, 5120, 20480\n");
    ROOT_PRINTF("  -t <num>     Number of timesteps (default: 50)\n");
    ROOT_PRINTF("  -s <num>     Source triangle index (default: 0)\n");
    ROOT_PRINTF("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    ROOT_PRINTF("  -v           Enable validation\n");
    ROOT_PRINTF("  -o           Print results for external validation\n");
    ROOT_PRINTF("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    g_rank = rank;

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
            MPI_Finalize();
            return 0;
        } else {
            ROOT_PRINTF("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank (ranks sharing a node take distinct devices)
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("Parallelization: %d MPI ranks, %d GPU(s) visible per node\n", ranks, deviceCount);
    ROOT_PRINTF("\n");

    // Initialize (every rank builds the same scene)
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t i0 = rowBegin(n, ranks, rank);
    const size_t i1 = rowBegin(n, ranks, rank + 1);
    const size_t nLocal = i1 - i0;

    std::vector<int> counts(ranks);
    std::vector<int> displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        displs[r] = static_cast<int>(rowBegin(n, ranks, r));
        counts[r] = static_cast<int>(rowBegin(n, ranks, r + 1) - rowBegin(n, ranks, r));
    }

    // Scene on the device
    Triangle* dTris = toDevice(state.triangles);
    DevNode* dNodes = toDevice(state.devNodes);
    idx_t* dTriIdx = toDevice(state.devTriIdx);
    val_t* dAreas = toDevice(state.areas);
    val_t* dRho = toDevice(state.rho);
    SceneView sv{dTris, dNodes, dTriIdx, dAreas, dRho};

    // Per-rank matrices: local rows of Tau and Kij, full radiosity history
    int* dTau = deviceAlloc<int>(nLocal * n);
    val_t* dKij = deviceAlloc<val_t>(nLocal * n);
    val_t* dRadB = deviceAlloc<val_t>(T * n);
    val_t* dDist = deviceAlloc<val_t>(nLocal);
    if (nLocal * n > 0) CUDA_CHECK(cudaMemset(dKij, 0, nLocal * n * sizeof(val_t)));
    if (T * n > 0) CUDA_CHECK(cudaMemset(dRadB, 0, T * n * sizeof(val_t)));
    std::vector<val_t> radB(T * n, ZERO);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(sv, i0, nLocal, n, dTau);
    computeFormFactors(state, sv, i0, i1, dKij);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, sv, i0, nLocal, dTau, dKij, dRadB, radB, counts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    std::vector<val_t> distances(rank == 0 ? n : 0);
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, i0, nLocal, dRadB, dDist, counts, displs, distances);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    MPI_Allreduce(MPI_IN_PLACE, &totalTime, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * T);
    double distOps = static_cast<double>(n * T * T);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", T);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * T * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = 0;
    if (rank == 0) hash = computeHash(distances);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");

    // Print results for external validation
    if (printResults && rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(distances.begin(), distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int ok = 1;
    if (validate) {
        unsigned long long nonZeroLocal = 0;
        unsigned long long* dCount = deviceAlloc<unsigned long long>(1);
        CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
        if (nLocal * n > 0) {
            countNonZeroKernel<<<gridFor(nLocal * n, BLOCK), BLOCK>>>(dKij, nLocal * n, dCount);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(&nonZeroLocal, dCount, sizeof(nonZeroLocal), cudaMemcpyDeviceToHost));
        deviceFree(dCount);

        unsigned long long nonZeroKij = 0;
        MPI_Reduce(&nonZeroLocal, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            ok = validateResults(distances, radB, n, T, state.sourceIndex,
                                 static_cast<size_t>(nonZeroKij)) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    deviceFree(dRho);
    deviceFree(dAreas);
    deviceFree(dTriIdx);
    deviceFree(dNodes);
    deviceFree(dTris);
    deviceFree(dDist);
    deviceFree(dRadB);
    deviceFree(dKij);
    deviceFree(dTau);

    MPI_Finalize();
    return ok ? 0 : 1;
}

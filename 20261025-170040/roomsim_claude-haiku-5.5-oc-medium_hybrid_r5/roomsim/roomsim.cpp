/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy:
 *  - MPI: receiver rows i are distributed cyclically over ranks (i = rank + P*li).
 *    Each rank owns the Kij/Tau columns for its rows. After every simulation
 *    timestep the newly computed radiosity values are exchanged with an allgather,
 *    so every rank keeps the full radiosity history needed for the delayed gathers.
 *  - CUDA: the form-factor (visibility ray casting over a flattened octree), time-delay
 *    and wave-propagation kernels run on the GPU of each rank. The Kij/Tau data stays
 *    resident on the device.
 *  - OpenMP: host-side loops (culling census and cross-correlation) run multi-threaded.
 *
 * Results are bit-identical to the sequential algorithm: the random stream used for
 * ray sampling is reproduced exactly (each rank skips ahead over other ranks' rows),
 * and all floating-point reductions keep the original summation order.
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
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

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ============================================================================
// Parallel runtime helpers
// ============================================================================

static int gRank = 0;        // global MPI rank
static int gSize = 1;        // number of MPI ranks
static int gLocalRank = 0;   // rank within the shared-memory node
static int gLocalSize = 1;   // ranks on this node

#define ROOT_PRINTF(...)                 \
    do {                                 \
        if (gRank == 0) printf(__VA_ARGS__); \
    } while (0)

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

static_assert(sizeof(val_t) == sizeof(float), "val_t must be single precision");

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// Iterative octree traversal stack capacity (device). Checked against the tree depth.
constexpr int TRAVERSAL_STACK = 96;
constexpr int BLOCK_SIZE = 128;

// Branch-free min/max with the same semantics as std::min/std::max for floats.
HD inline val_t hmin(val_t a, val_t b) { return b < a ? b : a; }
HD inline val_t hmax(val_t a, val_t b) { return a < b ? b : a; }

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
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host only)
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
// Octree for Spatial Acceleration (built on host, flattened for the device)
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

// Flattened octree node: POD layout suitable for device memory.
struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];   // index into node array, -1 if absent
    uint32_t triBegin;  // leaf: first entry in leafTris
    uint32_t triCount;  // leaf: number of triangles (0 for internal nodes)
};

struct FlatOctree {
    std::vector<FlatNode> nodes;      // node 0 is the root
    std::vector<uint32_t> leafTris;   // concatenated leaf triangle lists
    int maxDepth = 0;
};

static int32_t flattenNode(const Octree& node, FlatOctree& out, int depth) {
    const int32_t self = static_cast<int32_t>(out.nodes.size());
    out.nodes.emplace_back();
    out.maxDepth = std::max(out.maxDepth, depth);

    FlatNode fn{};
    fn.center = node.center;
    fn.halfExtent = node.halfExtent;
    for (auto& c : fn.child) c = -1;
    fn.triBegin = 0;
    fn.triCount = 0;

    if (!node.triangleIndices.empty()) {
        fn.triBegin = static_cast<uint32_t>(out.leafTris.size());
        fn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            out.leafTris.push_back(static_cast<uint32_t>(idx));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                fn.child[i] = flattenNode(*node.children[i], out, depth + 1);
            }
        }
    }

    out.nodes[self] = fn;
    return self;
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

    // Advance the stream by 'draws' rand() calls. For float output with mt19937
    // each rand() consumes exactly one engine output, so this equals calling rand()
    // 'draws' times without producing the values.
    void skip(uint64_t draws) { rng.discard(draws); }
};

// ============================================================================
// Geometry on device and host (shared code paths guarantee identical results)
// ============================================================================

// Point in triangle from two uniform samples (same arithmetic as the original sampler)
HD inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-AABB slab/separating-axis test; identical to the original Octree::rayIntersectsBox
HD inline bool rayIntersectsBox(const Vec3& center, const Vec3& halfExtent,
                                const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Moller-Trumbore ray/triangle intersection; returns FLT_MAX when there is no hit
HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// Check if the segment from->to is blocked by any triangle other than src/dst.
// Traverses the flattened octree with an explicit stack (order-independent result).
HD inline bool isRayBlocked(const Vec3& from, const Vec3& to,
                            const FlatNode* nodes, const uint32_t* leafTris,
                            const Triangle* tris, uint32_t srcIdx, uint32_t dstIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root (no box test on the root, as in the original)

    while (sp > 0) {
        const FlatNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                const uint32_t idx = leafTris[nd.triBegin + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int32_t ci = nd.child[c];
                if (ci >= 0 && rayIntersectsBox(nodes[ci].center, nodes[ci].halfExtent, from, to)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// Cosine between vector and triangle normal, clamped at zero
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return hmax(ZERO, v.dot(normal) / vNorm);
}

// Culling test: triangles facing the same direction exchange no energy
HD inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Form factor Kij for receiver i and emitter j given the 64 uniform samples
// consumed in the order (uI, vI, uJ, vJ) per ray.
HD inline val_t computeKij(uint32_t i, uint32_t j, const Triangle* tris,
                           const FlatNode* nodes, const uint32_t* leafTris,
                           const float* uu) {
    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];

    if (isCulled(triI, triJ)) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const float* q = uu + r * 4;
        Vec3 pI = pointInTriangle(triI, q[0], q[1]);
        Vec3 pJ = pointInTriangle(triJ, q[2], q[3]);

        if (isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) continue;

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

// Time delay in simulation steps between two triangle centers
HD inline int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Tau for the owned receiver rows, stored as tauT[j * L + li] (coalesced in li).
__global__ void timeDelayKernel(const Triangle* tris, int* tauT,
                                uint32_t N, uint32_t L, uint32_t P, uint32_t rank) {
    // N * L < 2^32 for every supported problem size, so 32-bit indexing is exact.
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N * L) return;
    const uint32_t j = p / L;
    const uint32_t li = p % L;
    const uint32_t i = rank + P * li;
    tauT[p] = (i == j) ? 0 : computeTau(tris[i], tris[j]);
}

// Form factors for one round of owned rows. Thread p maps to (j, slot s) with
// p = j * K + s; the uniform samples for that pair start at uni[p * 64].
__global__ void formFactorKernel(const Triangle* tris, const FlatNode* nodes,
                                 const uint32_t* leafTris, const float* uni,
                                 float* kijT, uint32_t N, uint32_t L, uint32_t P,
                                 uint32_t rank, uint32_t K, uint32_t windowStart) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= static_cast<size_t>(N) * K) return;
    const uint32_t j = static_cast<uint32_t>(p / K);
    const uint32_t s = static_cast<uint32_t>(p % K);
    const uint32_t i = windowStart + rank + P * s;
    if (i >= N || i == j) return;

    const uint32_t li = (i - rank) / P;
    const float* uu = uni + p * 64;
    kijT[static_cast<size_t>(j) * L + li] =
        computeKij(i, j, tris, nodes, leafTris, uu);
}

// One wave-propagation timestep for the owned receivers. Each thread sums over j
// in ascending order exactly as the sequential algorithm does.
__global__ void simulateKernel(const float* kijT, const int* tauT, const float* areas,
                               const float* rho, const float* radB, float* outLocal,
                               uint32_t t, uint32_t N, uint32_t L, uint32_t P,
                               uint32_t rank, uint32_t timeOff, uint32_t source) {
    const uint32_t li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= L) return;
    const uint32_t i = rank + P * li;

    val_t sumB = ZERO;
    for (uint32_t j = 0; j < N; ++j) {
        if (j == i) continue;

        const size_t off = static_cast<size_t>(j) * L + li;
        const int tauij = tauT[off];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        const val_t kij = kijT[off];
        if (kij <= ZERO) continue;

        // Radiosity emitted by j at the time the wave left it
        const size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        const val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += hmin(kij * areas[j], ONE) * radJ;
    }

    const val_t radE = (t < timeOff && i == source) ? ONE : ZERO;
    outLocal[li] = rho[i] * sumB + radE;
}

// Count entries above EPSILON (validation only)
__global__ void countNonZeroKernel(const float* kijT, size_t n, unsigned long long* count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n && kijT[idx] > EPSILON) {
        atomicAdd(count, 1ULL);
    }
}

// ============================================================================
// Simulation State
// ============================================================================

// Number of receiver rows owned by 'rank' in the cyclic distribution i = rank + P*li
static size_t ownedCount(size_t N, size_t rank, size_t P) {
    return rank < N ? (N - rank - 1) / P + 1 : 0;
}

struct DeviceData {
    Triangle* tris = nullptr;
    FlatNode* nodes = nullptr;
    uint32_t* leafTris = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* kijT = nullptr;      // [j * L + li]
    int* tauT = nullptr;        // [j * L + li]
    float* radB = nullptr;      // [t * N + i], full history
    float* outLocal = nullptr;  // [li] values of the current timestep
    float* uni[2] = {nullptr, nullptr};        // round sample buffers (device)
    float* hostUni[2] = {nullptr, nullptr};    // round sample buffers (pinned host)
    cudaStream_t streams[2] = {nullptr, nullptr};
    unsigned long long* counter = nullptr;
};

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    size_t numOwned = 0;      // receiver rows owned by this rank
    size_t numOwnedMax = 0;   // ceil(N / P): padded per-rank slot count

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Radiosity history (T x N), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source (N), replicated

    FlatOctree octree;              // Spatial acceleration structure
    DeviceData dev;
};

// Gather one value per owned row from every rank into a full N-length array.
// 'local' must hold numOwnedMax entries (padding unused).
static void allgatherOwnedRows(const std::vector<val_t>& local, size_t N, val_t* full) {
    const size_t P = static_cast<size_t>(gSize);
    const size_t Lmax = (N + P - 1) / P;
    std::vector<val_t> recv(P * Lmax);
    MPI_Allgather(local.data(), static_cast<int>(Lmax), MPI_FLOAT,
                  recv.data(), static_cast<int>(Lmax), MPI_FLOAT, MPI_COMM_WORLD);
    for (size_t r = 0; r < P; ++r) {
        const size_t cnt = ownedCount(N, r, P);
        for (size_t li = 0; li < cnt; ++li) {
            full[r + P * li] = recv[r * Lmax + li];
        }
    }
}

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
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

    const size_t N = state.numTriangles;
    const size_t P = static_cast<size_t>(gSize);
    state.numOwned = ownedCount(N, static_cast<size_t>(gRank), P);
    state.numOwnedMax = (N + P - 1) / P;

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration and flatten it for the device
    ROOT_PRINTF("Building octree...\n");
    {
        Octree octree;
        octree.build(state.triangles);
        flattenNode(octree, state.octree, 0);
    }
    if (7 * state.octree.maxDepth + 1 > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep (%d levels) for traversal stack\n", state.octree.maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Radiosity history and distances
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);
}

// Select the GPU, allocate device data and upload the static problem description.
void setupDevice(SimulationState& state) {
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "roomsim requires a CUDA device, none found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(gLocalRank % numDevices));

    const size_t N = state.numTriangles;
    const size_t L = state.numOwned;
    const size_t T = state.numTimesteps;
    DeviceData& d = state.dev;

    d.tris = deviceAlloc<Triangle>(N);
    d.nodes = deviceAlloc<FlatNode>(state.octree.nodes.size());
    d.leafTris = deviceAlloc<uint32_t>(state.octree.leafTris.size());
    d.areas = deviceAlloc<float>(N);
    d.rho = deviceAlloc<float>(N);
    d.kijT = deviceAlloc<float>(N * L);
    d.tauT = deviceAlloc<int>(N * L);
    d.radB = deviceAlloc<float>(T * N);
    d.outLocal = deviceAlloc<float>(state.numOwnedMax);
    d.counter = deviceAlloc<unsigned long long>(1);

    CUDA_CHECK(cudaMemcpy(d.tris, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.nodes, state.octree.nodes.data(),
                          state.octree.nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.leafTris, state.octree.leafTris.data(),
                          state.octree.leafTris.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.kijT, 0, std::max<size_t>(N * L, 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.radB, 0, std::max<size_t>(T * N, 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.counter, 0, sizeof(unsigned long long)));

    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaStreamCreate(&d.streams[k]));
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;
    const size_t L = state.numOwned;
    if (N * L > 0) {
        const size_t total = N * L;
        const unsigned blocks = static_cast<unsigned>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        timeDelayKernel<<<blocks, BLOCK_SIZE>>>(state.dev.tris, state.dev.tauT,
                                                 static_cast<uint32_t>(N), static_cast<uint32_t>(L),
                                                 static_cast<uint32_t>(gSize), static_cast<uint32_t>(gRank));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void computeFormFactors(SimulationState& state) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t L = state.numOwned;
    const size_t P = static_cast<size_t>(gSize);
    const size_t R = static_cast<size_t>(gRank);
    DeviceData& d = state.dev;

    // The sampling stream of the sequential algorithm advances 64 draws for every
    // non-culled pair (i, j), i != j. Count those per receiver so that each rank can
    // skip over rows it does not own without generating their samples.
    std::vector<uint64_t> partners(N);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        uint64_t c = 0;
        const Triangle& ti = state.triangles[i];
        for (size_t j = 0; j < N; ++j) {
            if (j != static_cast<size_t>(i) && !isCulled(ti, state.triangles[j])) ++c;
        }
        partners[i] = c;
    }

    // Rows are processed in rounds. A round covers a window of P*K consecutive
    // global rows; rank r handles the K rows congruent to r modulo P in that window.
    const size_t K = std::min<size_t>(std::max<size_t>(1, (size_t{1} << 20) / N), (N + P - 1) / P);
    const size_t window = K * P;
    const size_t numRounds = (N + window - 1) / window;
    const size_t bufFloats = N * K * 64;

    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaMallocHost(&d.hostUni[k], bufFloats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.uni[k], bufFloats * sizeof(float)));
    }

    // The sequential sampling stream is walked once, in row order. Owned rows record
    // the engine state at their start; other rows are skipped (cheap engine discard).
    // Each owned row can then generate its samples independently from its snapshot.
    std::vector<RandomGenerator> rowRng;
    rowRng.reserve(L);
    RandomGenerator stream(42);
    size_t nextRow = 0;
    auto advanceStreamTo = [&](size_t rowEnd) {
        for (; nextRow < rowEnd; ++nextRow) {
            if (nextRow % P == R) rowRng.push_back(stream);
            stream.skip(64ULL * partners[nextRow]);
        }
    };

    // Fill the host sample buffer for round g. The stream is advanced to the end of the
    // round's window first, so this serial skip overlaps with the previous round's GPU work.
    // Layout: buf[(j * K + s) * 64 + r * 4 + q] with q = (uI, vI, uJ, vJ).
    auto generateRound = [&](size_t g, float* buf) {
        advanceStreamTo(std::min(N, (g + 1) * window));
        const size_t firstLi = g * K;
        const size_t lastLi = std::min(L, firstLi + K);
        #pragma omp parallel for schedule(dynamic, 1)
        for (long li = static_cast<long>(firstLi); li < static_cast<long>(lastLi); ++li) {
            const size_t i = R + P * static_cast<size_t>(li);
            const size_t s = static_cast<size_t>(li) - firstLi;
            RandomGenerator gen = rowRng[li];
            const Triangle& ti = state.triangles[i];
            for (size_t j = 0; j < N; ++j) {
                if (j == i || isCulled(ti, state.triangles[j])) continue;
                float* dst = buf + (j * K + s) * 64;
                for (int q = 0; q < 64; ++q) dst[q] = gen.rand();
            }
        }
    };

    // Two-stage pipeline with double-buffered samples. While round g runs on
    // streams[k], round g+1 is generated on the host and uploaded on streams[k^1].
    size_t doneBefore = 0;
    generateRound(0, d.hostUni[0]);
    CUDA_CHECK(cudaMemcpyAsync(d.uni[0], d.hostUni[0], bufFloats * sizeof(float),
                               cudaMemcpyHostToDevice, d.streams[0]));
    for (size_t g = 0; g < numRounds; ++g) {
        const int k = static_cast<int>(g & 1);
        const size_t W = g * window;
        const size_t done = std::min(N, W + window);

        // Round g's samples were uploaded on streams[k] before its kernel (stream order).
        const size_t total = N * K;
        const unsigned blocks = static_cast<unsigned>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        formFactorKernel<<<blocks, BLOCK_SIZE, 0, d.streams[k]>>>(
            d.tris, d.nodes, d.leafTris, d.uni[k], d.kijT,
            static_cast<uint32_t>(N), static_cast<uint32_t>(L), static_cast<uint32_t>(P),
            static_cast<uint32_t>(R), static_cast<uint32_t>(K), static_cast<uint32_t>(W));
        CUDA_CHECK(cudaGetLastError());

        // Prepare round g+1 while round g computes: the other stream's previous work
        // (round g-1) must be finished before its host and device buffers are reused.
        if (g + 1 < numRounds) {
            CUDA_CHECK(cudaStreamSynchronize(d.streams[k ^ 1]));
            generateRound(g + 1, d.hostUni[k ^ 1]);
            CUDA_CHECK(cudaMemcpyAsync(d.uni[k ^ 1], d.hostUni[k ^ 1], bufFloats * sizeof(float),
                                       cudaMemcpyHostToDevice, d.streams[k ^ 1]));
        }

        CUDA_CHECK(cudaStreamSynchronize(d.streams[k]));
        MPI_Barrier(MPI_COMM_WORLD);

        // Progress reporting in the same form as the sequential code
        size_t mark = 100 * (doneBefore / 100 + 1);
        for (; mark <= done; mark += 100) {
            ROOT_PRINTF("  Progress: %zu/%zu triangles\n", mark, N);
        }
        if (done == N && N % 100 != 0) {
            ROOT_PRINTF("  Progress: %zu/%zu triangles\n", N, N);
        }
        doneBefore = done;
    }

    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaFreeHost(d.hostUni[k]));
        CUDA_CHECK(cudaFree(d.uni[k]));
        d.hostUni[k] = nullptr;
        d.uni[k] = nullptr;
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t L = state.numOwned;
    const size_t P = static_cast<size_t>(gSize);
    const size_t R = static_cast<size_t>(gRank);
    DeviceData& d = state.dev;

    std::vector<val_t> owned(state.numOwnedMax, ZERO);
    const uint32_t timeOff = static_cast<uint32_t>(T / 2);  // source emits for the first half

    for (size_t t = 0; t < T; ++t) {
        if (L > 0) {
            const unsigned blocks = static_cast<unsigned>((L + BLOCK_SIZE - 1) / BLOCK_SIZE);
            simulateKernel<<<blocks, BLOCK_SIZE>>>(
                d.kijT, d.tauT, d.areas, d.rho, d.radB, d.outLocal,
                static_cast<uint32_t>(t), static_cast<uint32_t>(N), static_cast<uint32_t>(L),
                static_cast<uint32_t>(P), static_cast<uint32_t>(R), timeOff,
                static_cast<uint32_t>(state.sourceIndex));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(owned.data(), d.outLocal, L * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Share this step's radiosity with all ranks
        allgatherOwnedRows(owned, N, state.radB.data() + t * N);

        // Update the device-resident history row for the next steps' delayed reads
        CUDA_CHECK(cudaMemcpy(d.radB + t * N, state.radB.data() + t * N,
                              N * sizeof(float), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// Count non-zero Kij entries (device-side reduction) across all ranks.
// Only used for validation; the matrix itself stays on the GPUs.
unsigned long long countNonZeroKij(SimulationState& state) {
    const size_t n = state.numTriangles * state.numOwned;
    DeviceData& d = state.dev;
    CUDA_CHECK(cudaMemset(d.counter, 0, sizeof(unsigned long long)));
    if (n > 0) {
        const unsigned blocks = static_cast<unsigned>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        countNonZeroKernel<<<blocks, BLOCK_SIZE>>>(d.kijT, n, d.counter);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long local = 0;
    CUDA_CHECK(cudaMemcpy(&local, d.counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long total = 0;
    MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t L = state.numOwned;
    const size_t P = static_cast<size_t>(gSize);
    const size_t R = static_cast<size_t>(gRank);
    const size_t src = state.sourceIndex;
    const val_t* radB = state.radB.data();

    std::vector<val_t> owned(state.numOwnedMax, ZERO);

    #pragma omp parallel for schedule(dynamic, 4)
    for (long li = 0; li < static_cast<long>(L); ++li) {
        const size_t i = R + P * static_cast<size_t>(li);
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = radB[tt * N + i];
                val_t pS = radB[(tt - t) * N + src];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        owned[li] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    allgatherOwnedRows(owned, N, state.distances.data());
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKijTotal) {
    ROOT_PRINTF("\nValidation:\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < N; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            ROOT_PRINTF("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            ROOT_PRINTF("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    ROOT_PRINTF("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    ROOT_PRINTF("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
    ROOT_PRINTF("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        ROOT_PRINTF("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = 0; t < T; ++t) {
            if (state.radB[t * N + i] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    ROOT_PRINTF("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

    if (receivedEnergy == 0) {
        ROOT_PRINTF("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const int nonZeroKij = static_cast<int>(nonZeroKijTotal);
    ROOT_PRINTF("  Non-zero form factors: %d/%zu (%.2f%%)\n",
                nonZeroKij, N * N,
                100.0f * nonZeroKij / static_cast<val_t>(N * N));

    if (nonZeroKij == 0) {
        ROOT_PRINTF("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    ROOT_PRINTF("  Validation: PASSED\n");
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

static int runBenchmark(int argc, char** argv) {
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
            ROOT_PRINTF("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    setupDevice(state);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", t);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");

    // Print results for external validation
    if (printResults && gRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (all ranks evaluate the same replicated data, so the result agrees)
    bool ok = true;
    if (validate) {
        const unsigned long long nonZeroKij = countNonZeroKij(state);
        ok = validateResults(state, nonZeroKij);
    }

    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, gRank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &gLocalRank);
    MPI_Comm_size(localComm, &gLocalSize);
    MPI_Comm_free(&localComm);

    // Share the node's cores between its ranks unless the user chose a thread count
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / gLocalSize));
    }

    const int rc = runBenchmark(argc, argv);

    MPI_Finalize();
    return rc;
}

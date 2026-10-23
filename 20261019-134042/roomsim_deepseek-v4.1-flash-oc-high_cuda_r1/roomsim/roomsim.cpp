/**
 * Room Response Simulation Benchmark (CUDA)
 *
 * This is a CUDA-parallelized implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * All expensive phases (form factors, time delays, wave propagation and
 * cross-correlation) run on the GPU.  The random stream used by the original
 * sequential implementation is reproduced exactly: the CPU advances an
 * MT19937 stream to obtain a checkpoint state for every contiguous chunk of
 * (i,j) pairs, and each GPU thread replays that chunk in the original order.
 * Consequently the numerical results are bit-for-bit identical to the
 * sequential reference (given identical floating point code generation).
 */

#include <cuda_runtime.h>

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

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

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
constexpr val_t FAR_DISTANCE = 3.402823466e+38f;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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
};

// Flattened, GPU-friendly octree node.
// Children that do not exist are stored as -1.  Triangle indices for a node
// are a contiguous slice of the global octree triangle index array.
struct DevOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    uint32_t triStart;
    uint32_t triCount;
};

static int flattenOctree(const Octree& node,
                         std::vector<DevOctreeNode>& nodes,
                         std::vector<uint32_t>& triIndices) {
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    DevOctreeNode nd;
    nd.center = node.center;
    nd.halfExtent = node.halfExtent;
    nd.triStart = static_cast<uint32_t>(triIndices.size());
    nd.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    for (int c = 0; c < 8; ++c) nd.child[c] = -1;

    for (size_t idx : node.triangleIndices) {
        triIndices.push_back(static_cast<uint32_t>(idx));
    }
    nodes[nodeIdx] = nd;

    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) {
            int childIdx = flattenOctree(*node.children[c], nodes, triIndices);
            nodes[nodeIdx].child[c] = childIdx;
        }
    }
    return nodeIdx;
}

// ============================================================================
// Random Number Generation (MT19937, matching std::mt19937/libstdc++)
// ============================================================================

// Host-side engine.  Besides normal draws it supports fast skipping of an
// arbitrary number of outputs, which is used to build per-chunk checkpoints.
struct MT19937Host {
    uint32_t mt[624];
    int mti;

    void seed(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < 624; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        mti = 624;
    }

    void generate() {
        for (int i = 0; i < 624; ++i) {
            int i1 = (i + 1 < 624) ? (i + 1) : 0;
            int i397 = (i + 397 < 624) ? (i + 397) : (i + 397 - 624);
            uint32_t y = (mt[i] & 0x80000000u) | (mt[i1] & 0x7fffffffu);
            mt[i] = mt[i397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        mti = 0;
    }

    uint32_t next() {
        if (mti >= 624) generate();
        uint32_t y = mt[mti++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    void skip(uint64_t n) {
        while (n > 0) {
            if (mti >= 624) generate();
            uint64_t avail = static_cast<uint64_t>(624 - mti);
            if (n < avail) {
                mti += static_cast<int>(n);
                return;
            }
            n -= avail;
            mti = 624;
        }
    }
};

// Device-side engine.  The state lives in local memory (one copy per thread).
struct MT19937Dev {
    uint32_t mt[624];
    int mti;

    HD void generate() {
        for (int i = 0; i < 624; ++i) {
            int i1 = (i + 1 < 624) ? (i + 1) : 0;
            int i397 = (i + 397 < 624) ? (i + 397) : (i + 397 - 624);
            uint32_t y = (mt[i] & 0x80000000u) | (mt[i1] & 0x7fffffffu);
            mt[i] = mt[i397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        mti = 0;
    }

    HD uint32_t next() {
        if (mti >= 624) generate();
        uint32_t y = mt[mti++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    // Equivalent to std::uniform_real_distribution<float>(0, 1) fed by
    // std::mt19937 (libstdc++ generate_canonical with one 32-bit draw).
    HD val_t rand() {
        val_t sum = static_cast<val_t>(next());
        return sum / 4294967296.0f;
    }
};

// Generate a random point inside a triangle using barycentric coordinates
HD Vec3 randomPointInTriangle(const Triangle& t, MT19937Dev& rng) {
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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                              const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FAR_DISTANCE;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FAR_DISTANCE;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FAR_DISTANCE;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, device)
// ============================================================================

HD bool rayIntersectsBox(const Vec3& p1, const Vec3& p2,
                         const Vec3& center, const Vec3& halfExtent) {
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

HD bool isRayBlocked(const Vec3& from, const Vec3& to,
                     const DevOctreeNode* __restrict__ nodes,
                     const uint32_t* __restrict__ triIndices,
                     const Triangle* __restrict__ triangles,
                     uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const DevOctreeNode& node = nodes[stack[--sp]];

        for (uint32_t k = 0; k < node.triCount; ++k) {
            uint32_t idx = triIndices[node.triStart + k];
            if (idx == srcTriIdx || idx == dstTriIdx) continue;

            const Triangle& tri = triangles[idx];
            val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < rayLen - EPSILON) {
                return true;
            }
        }

        for (int c = 7; c >= 0; --c) {
            int child = node.child[c];
            if (child >= 0) {
                const DevOctreeNode& cn = nodes[child];
                if (rayIntersectsBox(from, to, cn.center, cn.halfExtent)) {
                    if (sp < 64) stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij, device)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
HD val_t computeKij(uint32_t idxI, uint32_t idxJ,
                    const Triangle* __restrict__ triangles,
                    const DevOctreeNode* __restrict__ nodes,
                    const uint32_t* __restrict__ triIndices,
                    MT19937Dev& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, nodes, triIndices, triangles, idxI, idxJ)) continue;

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

constexpr uint32_t TARGET_CHUNKS = 1u << 17;
constexpr uint32_t MIN_CHUNK_PAIRS = 8;

__global__ void formFactorsKernel(const Triangle* __restrict__ triangles,
                                  const DevOctreeNode* __restrict__ nodes,
                                  const uint32_t* __restrict__ triIndices,
                                  val_t* __restrict__ kij,
                                  uint32_t numTriangles,
                                  uint64_t totalPairs,
                                  uint32_t chunkPairs,
                                  const uint32_t* __restrict__ cpState,
                                  const int* __restrict__ cpMti,
                                  const uint8_t* __restrict__ cullMask,
                                  uint32_t chunkOffset,
                                  uint32_t chunkCount) {
    uint32_t localChunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (localChunk >= chunkCount) return;
    uint32_t chunk = chunkOffset + localChunk;

    uint64_t start = static_cast<uint64_t>(chunk) * chunkPairs;
    if (start >= totalPairs) return;
    uint64_t end = start + chunkPairs;
    if (end > totalPairs) end = totalPairs;

    MT19937Dev rng;
    for (int k = 0; k < 624; ++k) {
        rng.mt[k] = cpState[static_cast<size_t>(chunk) * 624 + k];
    }
    rng.mti = cpMti[chunk];

    for (uint64_t p = start; p < end; ++p) {
        uint32_t i = static_cast<uint32_t>(p / numTriangles);
        uint32_t j = static_cast<uint32_t>(p - static_cast<uint64_t>(i) * numTriangles);

        // Culling decisions are made once on the host so that the replayed
        // random stream is guaranteed to match the host checkpoints.
        if (cullMask[p]) continue;

        kij[static_cast<size_t>(i) * numTriangles + j] =
            computeKij(i, j, triangles, nodes, triIndices, rng);
    }
}

__global__ void countNonZeroKijKernel(const val_t* __restrict__ kij,
                                      uint64_t count,
                                      int* __restrict__ result) {
    uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    int local = 0;
    for (; idx < count; idx += stride) {
        if (kij[idx] > EPSILON) ++local;
    }
    if (local != 0) atomicAdd(result, local);
}

// ============================================================================
// Tau (time delay) Computation (device)
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void tauKernel(const Triangle* __restrict__ triangles,
                          int* __restrict__ tau,
                          uint32_t numTriangles,
                          uint64_t totalPairs) {
    uint64_t p = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (; p < totalPairs; p += stride) {
        uint32_t i = static_cast<uint32_t>(p / numTriangles);
        uint32_t j = static_cast<uint32_t>(p - static_cast<uint64_t>(i) * numTriangles);
        if (i == j) continue;
        tau[p] = computeTau(triangles[i], triangles[j]);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;

    // Flattened octree (host mirrors of the device arrays)
    std::vector<DevOctreeNode> octreeNodes;
    std::vector<uint32_t> octreeTriIndices;

    // Device buffers
    Triangle* d_triangles = nullptr;
    DevOctreeNode* d_nodes = nullptr;
    uint32_t* d_triIndices = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

    // Host copy of the final distances
    std::vector<val_t> distances;

    size_t sourceIndex = 0;

    // Validation summaries produced on the device
    int nonZeroKij = 0;
    int receivedEnergy = 0;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
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
    Octree octree;
    octree.build(state.triangles);

    // Flatten the octree for the GPU
    state.octreeNodes.clear();
    state.octreeTriIndices.clear();
    flattenOctree(octree, state.octreeNodes, state.octreeTriIndices);

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Upload immutable scene data
    CUDA_CHECK(cudaMalloc(&state.d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(),
                          N * sizeof(Triangle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_nodes, state.octreeNodes.size() * sizeof(DevOctreeNode)));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, state.octreeNodes.data(),
                          state.octreeNodes.size() * sizeof(DevOctreeNode),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_triIndices, state.octreeTriIndices.size() * sizeof(uint32_t)));
    if (!state.octreeTriIndices.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_triIndices, state.octreeTriIndices.data(),
                              state.octreeTriIndices.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }

    // Areas and reflectivity
    std::vector<val_t> areas(N);
    for (size_t i = 0; i < N; ++i) {
        areas[i] = state.triangles[i].area();
    }
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, areas.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    std::vector<val_t> rho(N, reflectivity);
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, rho.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    // Matrices
    CUDA_CHECK(cudaMalloc(&state.d_kij, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_kij, 0, N * N * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&state.d_tau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_tau, 0, N * N * sizeof(int)));

    // Radiosity buffers
    std::vector<val_t> radE(T * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = T / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[t * N + state.sourceIndex] = 1.0f;
    }
    const size_t radCount = (T * N != 0) ? (T * N) : 1;
    CUDA_CHECK(cudaMalloc(&state.d_radE, radCount * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_radE, radE.data(), T * N * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radB, radCount * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * N * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint64_t totalPairs = static_cast<uint64_t>(N) * N;
    if (totalPairs == 0) return;

    int threads = 256;
    uint64_t blocks64 = (totalPairs + threads - 1) / threads;
    uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(blocks64, 0x7fffffffULL));
    tauKernel<<<blocks, threads>>>(state.d_triangles, state.d_tau, N, totalPairs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint64_t totalPairs = static_cast<uint64_t>(N) * N;
    if (totalPairs == 0) return;

    // Build per-chunk RNG checkpoints and culling decisions on the host.  The
    // random stream is advanced exactly as the sequential reference would.
    uint32_t chunkPairs = static_cast<uint32_t>((totalPairs + TARGET_CHUNKS - 1) / TARGET_CHUNKS);
    if (chunkPairs < MIN_CHUNK_PAIRS) chunkPairs = MIN_CHUNK_PAIRS;
    uint32_t numChunks = static_cast<uint32_t>((totalPairs + chunkPairs - 1) / chunkPairs);

    std::vector<uint32_t> cpState(static_cast<size_t>(624) * numChunks);
    std::vector<int> cpMti(numChunks);
    std::vector<uint8_t> cullMask(totalPairs);

    MT19937Host rng;
    rng.seed(42);

    uint64_t p = 0;
    for (uint32_t c = 0; c < numChunks; ++c) {
        for (int k = 0; k < 624; ++k) {
            cpState[static_cast<size_t>(c) * 624 + k] = rng.mt[k];
        }
        cpMti[c] = rng.mti;

        uint64_t end = std::min(p + chunkPairs, totalPairs);
        uint64_t draws = 0;
        for (; p < end; ++p) {
            uint32_t i = static_cast<uint32_t>(p / N);
            uint32_t j = static_cast<uint32_t>(p - static_cast<uint64_t>(i) * N);
            uint8_t cull = 0;
            if (i == j) {
                cull = 1;
            } else if (state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) {
                cull = 1;
            }
            cullMask[p] = cull;
            if (!cull) draws += static_cast<uint64_t>(2 * NUM_RAYS);
        }
        if (draws != 0) rng.skip(draws);
    }

    uint32_t* d_cpState = nullptr;
    int* d_cpMti = nullptr;
    uint8_t* d_cullMask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cpState, cpState.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_cpState, cpState.data(), cpState.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cpMti, cpMti.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cpMti, cpMti.data(), cpMti.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cullMask, cullMask.size() * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_cullMask, cullMask.data(), cullMask.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    std::vector<uint8_t>().swap(cullMask);

    const int threads = 256;
    uint32_t blocks = (numChunks + threads - 1) / threads;
    formFactorsKernel<<<blocks, threads>>>(state.d_triangles, state.d_nodes,
                                           state.d_triIndices, state.d_kij,
                                           N, totalPairs, chunkPairs,
                                           d_cpState, d_cpMti, d_cullMask,
                                           0, numChunks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_cpState));
    CUDA_CHECK(cudaFree(d_cpMti));
    CUDA_CHECK(cudaFree(d_cullMask));

    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void countNonZeroKij(SimulationState& state) {
    const uint64_t count = static_cast<uint64_t>(state.numTriangles) * state.numTriangles;
    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int threads = 256;
    uint64_t blocks64 = (count + threads - 1) / threads;
    uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(blocks64, 65535ULL));
    if (blocks == 0) blocks = 1;
    countNonZeroKijKernel<<<blocks, threads>>>(state.d_kij, count, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

__global__ void simStepKernel(const val_t* __restrict__ kij,
                              const int* __restrict__ tau,
                              const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* __restrict__ radB,
                              uint32_t numTriangles,
                              uint32_t t) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    const size_t row = static_cast<size_t>(i) * numTriangles;
    const val_t* __restrict__ krow = kij + row;
    const int* __restrict__ trow = tau + row;

    val_t sumB = ZERO;
    for (uint32_t j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = trow[j];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kijv = krow[j];
        if (kijv <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijv * areas[j], ONE) * radJ;
    }

    radB[static_cast<size_t>(t) * numTriangles + i] =
        rho[i] * sumB + radE[static_cast<size_t>(t) * numTriangles + i];
}

__global__ void countEnergyKernel(const val_t* __restrict__ radB,
                                  uint32_t numTriangles,
                                  uint32_t numTimesteps,
                                  int* __restrict__ result) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    for (uint32_t t = 0; t < numTimesteps; ++t) {
        if (radB[static_cast<size_t>(t) * numTriangles + i] > EPSILON) {
            atomicAdd(result, 1);
            break;
        }
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);

    int threads = 256;
    uint32_t blocks = (N + threads - 1) / threads;

    for (uint32_t t = 0; t < T; ++t) {
        simStepKernel<<<blocks, threads>>>(state.d_kij, state.d_tau,
                                           state.d_areas, state.d_rho,
                                           state.d_radE, state.d_radB, N, t);
        CUDA_CHECK(cudaGetLastError());
        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %u/%u\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    int* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    countEnergyKernel<<<blocks, threads>>>(state.d_radB, N, T, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&state.receivedEnergy, d_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distancesKernel(const val_t* __restrict__ radB,
                                val_t* __restrict__ distances,
                                uint32_t numTriangles,
                                uint32_t numTimesteps,
                                uint32_t sourceIndex) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (uint32_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (uint32_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);

    int threads = 256;
    uint32_t blocks = (N + threads - 1) / threads;
    distancesKernel<<<blocks, threads>>>(state.d_radB, state.d_distances,
                                         N, T, static_cast<uint32_t>(state.sourceIndex));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    state.distances.resize(N);
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances,
                          N * sizeof(val_t), cudaMemcpyDeviceToHost));
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
    printf("  Triangles receiving energy: %d/%zu\n", state.receivedEnergy, state.numTriangles);

    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           state.nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * state.nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (state.nonZeroKij == 0) {
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
    countNonZeroKij(state);

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

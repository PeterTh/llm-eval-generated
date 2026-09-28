/**
 * Room Response Simulation Benchmark -- CUDA implementation
 *
 * This is a CUDA-parallel implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (all phases run on the GPU):
 *
 *  - Kij: one thread per triangle pair, tracing NUM_RAYS shadow rays through a
 *    flattened octree with an explicit traversal stack.  The Monte-Carlo sample
 *    stream of the sequential reference is a *single* std::mt19937 sequence
 *    consumed in row-major pair order, so it is reproduced exactly: pairs that
 *    are culled consume nothing, every other pair consumes exactly 4*NUM_RAYS
 *    variates.  A prefix sum over the cull predicate gives each pair its
 *    absolute position in the stream, and the Mersenne Twister itself is
 *    advanced on the GPU by a single-block generator kernel that emits a
 *    512-word wavefront per barrier (see the generator section for how the
 *    recurrence's short lag is eliminated).  Generation runs on a second,
 *    higher-priority stream against several random-number buffers, so it
 *    overlaps with the ray tracing of earlier batches.
 *
 *  - Wave propagation: one warp per receiver triangle, lanes evaluate 32
 *    emitters at a time (coalesced Kij/Tau reads) and the per-emitter terms are
 *    then accumulated through warp shuffles in ascending emitter order, which
 *    preserves the summation order (and hence the exact floating-point result)
 *    of the sequential loop.
 *
 *  - Cross-correlation: one block per triangle, threads own individual lag
 *    values, radiosity traces are staged in shared memory, and the arg-max is
 *    combined with a lowest-lag-wins block reduction to match the strict
 *    "greater than" comparison of the sequential code.
 *
 * Every phase keeps the reference's summation order, so results are typically
 * bit-identical to the sequential build.  What cannot be reproduced exactly is
 * the last-bit rounding of the ray-tracing geometry: gcc contracts the dot and
 * cross products into FMAs, and it picks a different operand for each inlined
 * copy, so a few form factors land one ulp away.  Distances are quantised lags,
 * so this is normally invisible; only when very many timesteps make the
 * cross-correlation peak nearly tie does it move one or two lags by one step.
 * (The reference is equally sensitive: it changes results between -O2 and
 * -O3 -march=native for the same reason.)
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

// Number of random variates consumed per triangle pair that is not culled
constexpr int RNG_PER_PAIR = 4 * NUM_RAYS;

// ============================================================================
// CUDA Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t _err = (call);                                                   \
        if (_err != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(_err));                                       \
            exit(EXIT_FAILURE);                                                      \
        }                                                                           \
    } while (0)

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
// Device-side geometry helpers
// ============================================================================

__device__ __forceinline__ float3 mkf3(const float4& v) { return make_float3(v.x, v.y, v.z); }
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float norm3(const float3& a) { return sqrtf(dot3(a, a)); }

// Flattened octree node.  Interior nodes have triCount == 0 and reference their
// children by index; leaves reference a slice of the shared triangle-index list.
// Keeping the whole node in one record beat splitting the bounds out into their
// own array: the tree is small enough to stay cached, so locality between a
// node's bounds and its child pointers matters more than the bytes fetched.
struct GNode {
    float4 center;      // w unused
    float4 halfExtent;  // w unused
    int child[8];       // -1 when absent
    int triStart;
    int triCount;
    int pad[2];
};

// Ray-invariant part of the segment/box test, hoisted out of the traversal
struct SegTest {
    float3 d;    // half the segment vector
    float3 mid;  // segment midpoint, i.e. p1 + d
    float3 ad;   // |d|
};

// Maximum explicit-stack depth of the GPU octree traversal
constexpr int TRAVERSAL_STACK_SIZE = 64;

// Mirrors Octree::rayIntersectsBox
__device__ __forceinline__ bool rayIntersectsBox(const SegTest& st, const GNode& nd) {
    const float3 he = mkf3(nd.halfExtent);
    const float3 c = st.mid - mkf3(nd.center);
    const float3 d = st.d;
    const float3 ad = st.ad;

    if (fabsf(c.x) > he.x + ad.x) return false;
    if (fabsf(c.y) > he.y + ad.y) return false;
    if (fabsf(c.z) > he.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Ray-Triangle Intersection (Moeller-Trumbore algorithm)
__device__ __forceinline__ float rayTriangleIntersect(const float3& orig, const float3& dir,
                                                      const float3& v0, const float3& v1,
                                                      const float3& v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);

    if (fabsf(det) < EPSILON) return __int_as_float(0x7f7fffff);  // FLT_MAX

    float invDet = 1.0f / det;
    float3 tvec = orig - v0;
    float u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return __int_as_float(0x7f7fffff);

    float3 qvec = cross3(tvec, e1);
    float v = dot3(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return __int_as_float(0x7f7fffff);

    return dot3(e2, qvec) * invDet;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Octree-accelerated, using an explicit depth-first stack.  Leaves are consumed
// as soon as they pass the box test, so only interior nodes ever get pushed.
__device__ bool isRayBlocked(const float3& from, const float3& to,
                             int srcTriIdx, int dstTriIdx,
                             const GNode* __restrict__ nodes, const int* __restrict__ nodeTris,
                             const float4* __restrict__ triA, const float4* __restrict__ triB,
                             const float4* __restrict__ triC) {
    const float3 dir = to - from;
    const float rayLen = norm3(dir);
    if (rayLen < EPSILON) return true;
    const float3 dirNorm = dir / rayLen;
    const float limit = rayLen - EPSILON;

    // Ray-invariant terms of the box test, evaluated once instead of per node
    SegTest st;
    st.d = dir * 0.5f;
    st.mid = from + st.d;
    st.ad = make_float3(fabsf(st.d.x), fabsf(st.d.y), fabsf(st.d.z));

    int stack[TRAVERSAL_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // the root's own bounding box is never tested

    while (sp > 0) {
        const int ni = stack[--sp];
        const GNode& nd = nodes[ni];

        if (nd.triCount > 0) {
            for (int k = 0; k < nd.triCount; ++k) {
                const int idx = nodeTris[nd.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const float dist = rayTriangleIntersect(from, dirNorm, mkf3(triA[idx]),
                                                        mkf3(triB[idx]), mkf3(triC[idx]));
                if (dist > EPSILON && dist < limit) return true;
            }
        } else {
            for (int ci = 0; ci < 8; ++ci) {
                const int c = nd.child[ci];
                if (c >= 0 && rayIntersectsBox(st, nodes[c])) stack[sp++] = c;
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ float cosPhi(const float3& v, const float3& normal) {
    float vNorm = norm3(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, dot3(v, normal) / vNorm);
}

// The reference culls pairs whose normals point in (nearly) the same direction;
// culled pairs consume no random variates.
__device__ __forceinline__ bool pairNeedsSampling(int i, int j, const float4& nI, const float4& nJ) {
    if (i == j) return false;
    float d = nI.x * nJ.x + nI.y * nJ.y + nI.z * nJ.z;
    return !(d > 0.99f);
}

// ============================================================================
// Mersenne Twister (std::mt19937) generation on the GPU
// ============================================================================
//
// The sequential reference draws all Monte-Carlo samples from one std::mt19937
// stream, so the stream has to be reproduced verbatim.  Viewed as one flat
// sequence of state words w[] (the tempered w[624], w[625], ... are exactly the
// successive draws), MT19937 is the linear recurrence
//     w[k] = w[k-227] ^ T(w[k-624], w[k-623]).
// The 227-word minimum lag already allows 227 words to be produced in
// parallel, but substituting the recurrence into its own w[k-227] term twice
// eliminates the short lag and yields
//     w[k] = w[k-681] ^ T(w[k-1078], w[k-1077])
//                     ^ T(w[k-851],  w[k-850])
//                     ^ T(w[k-624],  w[k-623]),
// whose minimum lag is 623.  That lets one block emit a 512-word wavefront per
// barrier -- roughly three times fewer barriers per word than the textbook
// block refill -- which matters because this generator is a single sequential
// chain and therefore latency bound.

constexpr int MT_N = 624;
constexpr uint32_t MT_A = 0x9908b0dfu;
constexpr int MT_PROLOGUE = 1078;      // words produced with the base recurrence
constexpr int MT_RING = 2048;          // >= MT_PROLOGUE + MT_WAVE, power of two
constexpr int MT_RING_MASK = MT_RING - 1;
constexpr int MT_WAVE = 512;           // words per wavefront (must be <= 623)
constexpr int MT_GEN_THREADS = 512;
constexpr int MT_PER_THREAD = MT_WAVE / MT_GEN_THREADS;

__device__ __forceinline__ uint32_t mtTwist(uint32_t u, uint32_t v) {
    uint32_t y = (u & 0x80000000u) | (v & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? MT_A : 0u);
}

// Tempering plus the float conversion performed by
// std::uniform_real_distribution<float> via std::generate_canonical:
// float(word) / 2^32, clamped below 1.
__device__ __forceinline__ float mtTemperToFloat(uint32_t z) {
    z ^= z >> 11;
    z ^= (z << 7) & 0x9d2c5680u;
    z ^= (z << 15) & 0xefc60000u;
    z ^= z >> 18;
    float f = static_cast<float>(z) / 4294967296.0f;
    if (f >= 1.0f) f = __int_as_float(0x3f7fffff);  // nextafter(1, 0)
    return f;
}

// Emit `count` variates of the persistent stream into `out`.
// Launched as a single block; `ring` (the trailing MT_RING state words) and
// `counters` (words produced / words already emitted) carry the stream across
// calls.  A call may run past `count` to finish a wavefront; the surplus stays
// pending for the next call.
__global__ void mtGenerateKernel(uint32_t* __restrict__ ring,
                                 unsigned long long* __restrict__ counters,
                                 float* __restrict__ out, unsigned long long count) {
    __shared__ uint32_t sh[MT_RING];
    const int tid = threadIdx.x;

    for (int k = tid; k < MT_RING; k += MT_GEN_THREADS) sh[k] = ring[k];
    unsigned long long produced = counters[0];
    const unsigned long long base = counters[1];
    __syncthreads();

    // Words produced by an earlier call but not yet handed out
    const unsigned long long pending = min(produced - base, count);
    for (unsigned long long q = tid; q < pending; q += MT_GEN_THREADS) {
        out[q] = mtTemperToFloat(sh[(base + q) & MT_RING_MASK]);
    }

    // Produce (and immediately temper) whole wavefronts.  Every word read has a
    // lag of at least 623, so it lies outside the range being written.
    while (produced - base < count) {
        const unsigned long long K = produced;
        #pragma unroll
        for (int s = 0; s < MT_PER_THREAD; ++s) {
            const unsigned long long k = K + s * MT_GEN_THREADS + tid;
            const uint32_t v = sh[(k - 681) & MT_RING_MASK] ^
                               mtTwist(sh[(k - 1078) & MT_RING_MASK], sh[(k - 1077) & MT_RING_MASK]) ^
                               mtTwist(sh[(k - 851) & MT_RING_MASK], sh[(k - 850) & MT_RING_MASK]) ^
                               mtTwist(sh[(k - 624) & MT_RING_MASK], sh[(k - 623) & MT_RING_MASK]);
            sh[k & MT_RING_MASK] = v;
            if (k - base < count) out[k - base] = mtTemperToFloat(v);
        }
        __syncthreads();
        produced = K + MT_WAVE;
    }

    for (int k = tid; k < MT_RING; k += MT_GEN_THREADS) ring[k] = sh[k];
    if (tid == 0) {
        counters[0] = produced;
        counters[1] = base + count;
    }
}

// ============================================================================
// Kernels: cull counting and per-row prefix sums of the random-stream offsets
// ============================================================================

constexpr int SCAN_THREADS = 256;

// One block per row: number of pairs in the row that consume random variates
__global__ void rowSampleCountKernel(int n, const float4* __restrict__ triN,
                                     unsigned int* __restrict__ rowCount) {
    __shared__ unsigned int sh[SCAN_THREADS];
    const int i = blockIdx.x;
    const float4 nI = triN[i];

    unsigned int c = 0;
    for (int j = threadIdx.x; j < n; j += SCAN_THREADS) {
        if (pairNeedsSampling(i, j, nI, triN[j])) ++c;
    }
    sh[threadIdx.x] = c;
    __syncthreads();
    for (int off = SCAN_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) rowCount[i] = sh[0];
}

// One block per row of the current batch: batch-local offset (in variates) of
// each pair's slice of the random stream.
__global__ void rngOffsetKernel(int n, int rowBegin, const float4* __restrict__ triN,
                                const unsigned long long* __restrict__ rowStart,
                                unsigned int* __restrict__ rngOffset) {
    __shared__ unsigned int sh[SCAN_THREADS];
    __shared__ unsigned int shRun;

    const int row = blockIdx.x;
    const int i = rowBegin + row;
    const float4 nI = triN[i];
    unsigned int* out = rngOffset + static_cast<size_t>(row) * n;

    if (threadIdx.x == 0) shRun = static_cast<unsigned int>(rowStart[i] - rowStart[rowBegin]);
    __syncthreads();

    for (int base = 0; base < n; base += SCAN_THREADS) {
        const int j = base + threadIdx.x;
        unsigned int p = (j < n && pairNeedsSampling(i, j, nI, triN[j])) ? 1u : 0u;
        sh[threadIdx.x] = p;
        __syncthreads();
        for (int off = 1; off < SCAN_THREADS; off <<= 1) {
            unsigned int v = (threadIdx.x >= off) ? sh[threadIdx.x - off] : 0u;
            __syncthreads();
            sh[threadIdx.x] += v;
            __syncthreads();
        }
        if (j < n) out[j] = RNG_PER_PAIR * (shRun + sh[threadIdx.x] - p);
        __syncthreads();
        if (threadIdx.x == SCAN_THREADS - 1) shRun += sh[SCAN_THREADS - 1];
        __syncthreads();
    }
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// One thread per triangle pair, tracing its NUM_RAYS shadow rays in sequence.
// Consecutive threads share the receiver triangle i, so neighbouring lanes
// enter the octree in the same place, which is where the traversal spends most
// of its time.  (Splitting the rays of a pair across lanes instead was measured
// to be considerably slower: the rays are long chords across the room and
// diverge almost immediately.)
constexpr int KIJ_THREADS = 128;

__global__ void __launch_bounds__(KIJ_THREADS) kijKernel(int n, int rowBegin,
                          const float* __restrict__ rng,
                          const unsigned int* __restrict__ rngOffset,
                          float* __restrict__ kij,
                          const float4* __restrict__ triA, const float4* __restrict__ triB,
                          const float4* __restrict__ triC, const float4* __restrict__ triN,
                          const GNode* __restrict__ nodes, const int* __restrict__ nodeTris) {
    const int j = blockIdx.x * KIJ_THREADS + threadIdx.x;
    if (j >= n) return;
    const int row = blockIdx.y;
    const int i = rowBegin + row;

    float result = ZERO;
    const float4 nI4 = triN[i];
    const float4 nJ4 = triN[j];

    if (pairNeedsSampling(i, j, nI4, nJ4)) {
        const float3 nI = mkf3(nI4);
        const float3 nJ = mkf3(nJ4);
        const float3 aI = mkf3(triA[i]);
        const float3 abI = mkf3(triB[i]) - aI;
        const float3 acI = mkf3(triC[i]) - aI;
        const float3 aJ = mkf3(triA[j]);
        const float3 abJ = mkf3(triB[j]) - aJ;
        const float3 acJ = mkf3(triC[j]) - aJ;

        // Each pair owns RNG_PER_PAIR consecutive variates: (u, v) for the
        // receiver sample followed by (u, v) for the emitter sample, per ray.
        const float4* samples =
            reinterpret_cast<const float4*>(rng + rngOffset[static_cast<size_t>(row) * n + j]);

        float sum = ZERO;
        for (int r = 0; r < NUM_RAYS; ++r) {
            const float4 s = samples[r];
            float uI = s.x, vI = s.y, uJ = s.z, vJ = s.w;
            if (uI + vI > 1.0f) { uI = 1.0f - uI; vI = 1.0f - vI; }
            if (uJ + vJ > 1.0f) { uJ = 1.0f - uJ; vJ = 1.0f - vJ; }

            const float3 pI = aI + abI * uI + acI * vI;
            const float3 pJ = aJ + abJ * uJ + acJ * vJ;

            if (isRayBlocked(pI, pJ, i, j, nodes, nodeTris, triA, triB, triC)) continue;

            const float3 v = pJ - pI;
            const float distSqr = dot3(v, v);
            if (distSqr < EPSILON) continue;

            const float cosPhiI = cosPhi(v, nI);
            const float cosPhiJ = cosPhi(-v, nJ);
            if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

            sum += (cosPhiI * cosPhiJ) / (PI * distSqr);
        }
        result = sum * INV_NUM_RAYS;
    }

    kij[static_cast<size_t>(i) * n + j] = result;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__global__ void tauKernel(int n, const float4* __restrict__ triCenter, int* __restrict__ tau) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int i = blockIdx.y;

    int result = 0;
    if (i != j) {
        const float3 d = mkf3(triCenter[i]) - mkf3(triCenter[j]);
        result = static_cast<int>(ceilf(norm3(d) * INV_WAVE_SPEED));
    }
    tau[static_cast<size_t>(i) * n + j] = result;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

constexpr int SIM_WARPS_PER_BLOCK = 4;

// One warp per receiver triangle.  Lanes evaluate 32 emitters at a time and the
// terms are folded into the accumulator in ascending emitter order so that the
// floating-point summation matches the sequential loop exactly (adding an exact
// +0.0f for emitters the reference skips is a no-op).
__global__ void simKernel(int n, int t,
                          const float* __restrict__ kij, const int* __restrict__ tau,
                          const float* __restrict__ areas, const float* __restrict__ rho,
                          const float* __restrict__ radE, float* __restrict__ radB) {
    const int warpInBlock = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * SIM_WARPS_PER_BLOCK + warpInBlock;
    if (i >= n) return;

    const size_t rowBase = static_cast<size_t>(i) * n;
    float sumB = ZERO;

    for (int base = 0; base < n; base += 32) {
        const int j = base + lane;
        float term = ZERO;

        if (j < n && j != i) {
            const int tauij = tau[rowBase + j];
            if (t >= tauij) {
                const float k = kij[rowBase + j];
                if (k > ZERO) {
                    const float radJ = radB[static_cast<size_t>(t - tauij) * n + j];
                    if (radJ > ZERO) term = fminf(k * areas[j], ONE) * radJ;
                }
            }
        }

        #pragma unroll
        for (int l = 0; l < 32; ++l) sumB += __shfl_sync(0xffffffffu, term, l);
    }

    if (lane == 0) {
        radB[static_cast<size_t>(t) * n + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * n + i];
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

constexpr int DIST_THREADS = 256;

// One block per triangle.  `useShared` stages both radiosity traces in shared
// memory when they fit; otherwise they are read from global memory.
template <bool useShared>
__global__ void distanceKernel(int n, int numTimesteps, int sourceIndex,
                               const float* __restrict__ radB, float* __restrict__ distances) {
    extern __shared__ float shBuf[];
    const int i = blockIdx.x;

    const float* recv;
    const float* src;
    if (useShared) {
        float* shRecv = shBuf;
        float* shSrc = shBuf + numTimesteps;
        for (int k = threadIdx.x; k < numTimesteps; k += DIST_THREADS) {
            shRecv[k] = radB[static_cast<size_t>(k) * n + i];
            shSrc[k] = radB[static_cast<size_t>(k) * n + sourceIndex];
        }
        __syncthreads();
        recv = shRecv;
        src = shSrc;
    } else {
        recv = radB + i;
        src = radB + sourceIndex;
    }
    const int stride = useShared ? 1 : n;

    // Discrete cross-correlation to find time delay
    float bestVal = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < numTimesteps; t += DIST_THREADS) {
        float sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            sum += src[static_cast<size_t>(tt - t) * stride] * recv[static_cast<size_t>(tt) * stride];
        }
        // Strict '>' keeps the earliest lag on ties, as in the reference
        if (sum > bestVal) { bestVal = sum; bestT = t; }
    }

    // Block-wide arg-max; ties resolve to the lowest lag
    __shared__ float shVal[DIST_THREADS];
    __shared__ int shIdx[DIST_THREADS];
    shVal[threadIdx.x] = bestVal;
    shIdx[threadIdx.x] = bestVal > ZERO ? bestT : 0;
    __syncthreads();
    for (int off = DIST_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            const float ov = shVal[threadIdx.x + off];
            const int oi = shIdx[threadIdx.x + off];
            if (ov > shVal[threadIdx.x] || (ov == shVal[threadIdx.x] && oi < shIdx[threadIdx.x])) {
                shVal[threadIdx.x] = ov;
                shIdx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) distances[i] = WAVE_SPEED * static_cast<float>(shIdx[0]);
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // ---- device-side mirrors ----
    float4* d_triA = nullptr;
    float4* d_triB = nullptr;
    float4* d_triC = nullptr;
    float4* d_triN = nullptr;
    float4* d_triCenter = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_kij = nullptr;
    int* d_tau = nullptr;
    float* d_radE = nullptr;
    float* d_radB = nullptr;
    float* d_distances = nullptr;
    GNode* d_nodes = nullptr;
    int* d_nodeTris = nullptr;

    // Random-stream plumbing
    uint32_t* d_mtRing = nullptr;
    unsigned long long* d_mtCounters = nullptr;
    float* d_rng[3] = {nullptr, nullptr, nullptr};
    unsigned int* d_rngOffset[3] = {nullptr, nullptr, nullptr};
    unsigned long long* d_rowStart = nullptr;
    std::vector<unsigned long long> rowStart;   // prefix sum of sampled pairs per row
    std::vector<int> batchRowBegin;             // batch boundaries (rows)

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Octree flattening
// ============================================================================

static int flattenOctree(const Octree* node, std::vector<GNode>& nodes,
                         std::vector<int>& nodeTris, int depth, int& maxDepth) {
    const int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    if (depth > maxDepth) maxDepth = depth;

    GNode g{};
    g.center = make_float4(node->center.x, node->center.y, node->center.z, 0.0f);
    g.halfExtent = make_float4(node->halfExtent.x, node->halfExtent.y, node->halfExtent.z, 0.0f);
    for (int k = 0; k < 8; ++k) g.child[k] = -1;
    g.triStart = 0;
    g.triCount = 0;

    if (!node->triangleIndices.empty()) {
        g.triStart = static_cast<int>(nodeTris.size());
        g.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) nodeTris.push_back(static_cast<int>(idx));
    } else {
        for (int k = 0; k < 8; ++k) {
            if (node->children[k]) {
                g.child[k] = flattenOctree(node->children[k].get(), nodes, nodeTris, depth + 1,
                                           maxDepth);
            }
        }
    }

    nodes[id] = g;
    return id;
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);
    state.distances.resize(state.numTriangles, ZERO);

    const size_t n = state.numTriangles;
    const size_t T = timesteps;

    // ---- upload geometry ----
    std::vector<float4> hA(n), hB(n), hC(n), hN(n), hCenter(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& tri = state.triangles[i];
        hA[i] = make_float4(tri.a.x, tri.a.y, tri.a.z, 0.0f);
        hB[i] = make_float4(tri.b.x, tri.b.y, tri.b.z, 0.0f);
        hC[i] = make_float4(tri.c.x, tri.c.y, tri.c.z, 0.0f);
        const Vec3 nrm = tri.normal();
        hN[i] = make_float4(nrm.x, nrm.y, nrm.z, 0.0f);
        const Vec3 ctr = tri.center();
        hCenter[i] = make_float4(ctr.x, ctr.y, ctr.z, 0.0f);
    }

    auto uploadF4 = [&](float4** dst, const std::vector<float4>& src) {
        CUDA_CHECK(cudaMalloc(dst, src.size() * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(*dst, src.data(), src.size() * sizeof(float4), cudaMemcpyHostToDevice));
    };
    uploadF4(&state.d_triA, hA);
    uploadF4(&state.d_triB, hB);
    uploadF4(&state.d_triC, hC);
    uploadF4(&state.d_triN, hN);
    uploadF4(&state.d_triCenter, hCenter);

    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- flatten and upload the octree ----
    std::vector<GNode> nodes;
    std::vector<int> nodeTris;
    int maxDepth = 0;
    flattenOctree(&state.octree, nodes, nodeTris, 0, maxDepth);
    if (1 + 7 * maxDepth > TRAVERSAL_STACK_SIZE) {
        fprintf(stderr, "Octree too deep (%d) for the traversal stack\n", maxDepth);
        exit(EXIT_FAILURE);
    }
    printf("Octree: %zu nodes, %zu triangle references, depth %d\n",
           nodes.size(), nodeTris.size(), maxDepth);

    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(GNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_nodeTris, nodeTris.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.d_nodeTris, nodeTris.data(), nodeTris.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ---- matrices ----
    CUDA_CHECK(cudaMalloc(&state.d_kij, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, T * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, T * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * n * sizeof(float)));

    // Set source emission (active for first half of timesteps)
    std::vector<val_t> radE(T * n, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
    CUDA_CHECK(cudaMemcpy(state.d_radE, radE.data(), T * n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Mersenne Twister ring, seeded exactly like std::mt19937(42) ----
    // The lag-623 recurrence used on the device only holds for k >= 1078, so
    // the seeded state plus the first 454 words are laid down here with the
    // base recurrence.  Draw i of the reference stream is tempered w[624 + i].
    std::vector<uint32_t> ring(MT_RING, 0u);
    ring[0] = 42u;
    for (int k = 1; k < MT_N; ++k) {
        ring[k] = static_cast<uint32_t>(1812433253u * (ring[k - 1] ^ (ring[k - 1] >> 30)) +
                                        static_cast<uint32_t>(k));
    }
    for (int k = MT_N; k < MT_PROLOGUE; ++k) {
        const uint32_t y = (ring[k - 624] & 0x80000000u) | (ring[k - 623] & 0x7fffffffu);
        ring[k] = ring[k - 227] ^ (y >> 1) ^ ((y & 1u) ? MT_A : 0u);
    }
    CUDA_CHECK(cudaMalloc(&state.d_mtRing, MT_RING * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(state.d_mtRing, ring.data(), MT_RING * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    const unsigned long long counters[2] = {MT_PROLOGUE, MT_N};  // produced, emitted
    CUDA_CHECK(cudaMalloc(&state.d_mtCounters, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(state.d_mtCounters, counters, 2 * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
}

void releaseSimulation(SimulationState& state) {
    for (void* p : {(void*)state.d_triA, (void*)state.d_triB, (void*)state.d_triC,
                    (void*)state.d_triN, (void*)state.d_triCenter, (void*)state.d_areas,
                    (void*)state.d_rho, (void*)state.d_kij, (void*)state.d_tau,
                    (void*)state.d_radE, (void*)state.d_radB, (void*)state.d_distances,
                    (void*)state.d_nodes, (void*)state.d_nodeTris, (void*)state.d_mtRing,
                    (void*)state.d_mtCounters, (void*)state.d_rowStart,
                    (void*)state.d_rng[0], (void*)state.d_rng[1], (void*)state.d_rng[2],
                    (void*)state.d_rngOffset[0], (void*)state.d_rngOffset[1],
                    (void*)state.d_rngOffset[2]}) {
        if (p) cudaFree(p);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Random variates buffered per batch (32 MB per buffer).  Several buffers are
// kept in flight so that stream generation, which is one long dependency chain
// confined to a single SM, overlaps with the ray tracing of earlier batches.
constexpr unsigned long long RNG_BUDGET_HINT = 8ull * 1024 * 1024;
constexpr int RNG_BUFFERS = 3;

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);

    // How many pairs in each row need random variates (i.e. survive culling)?
    unsigned int* d_rowCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowCount, n * sizeof(unsigned int)));
    rowSampleCountKernel<<<n, SCAN_THREADS>>>(n, state.d_triN, d_rowCount);
    std::vector<unsigned int> rowCount(n);
    CUDA_CHECK(cudaMemcpy(rowCount.data(), d_rowCount, n * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rowCount));

    state.rowStart.assign(n + 1, 0ull);
    for (int i = 0; i < n; ++i) state.rowStart[i + 1] = state.rowStart[i] + rowCount[i];
    CUDA_CHECK(cudaMalloc(&state.d_rowStart, (n + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(state.d_rowStart, state.rowStart.data(),
                          (n + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Batches are whole rows, so a buffer has to hold at least the widest row
    const unsigned long long budget =
        std::max(RNG_BUDGET_HINT, RNG_PER_PAIR * static_cast<unsigned long long>(n));

    // Split the rows into batches whose random variates fit into one buffer
    state.batchRowBegin.clear();
    state.batchRowBegin.push_back(0);
    int maxBatchRows = 1;
    {
        int r0 = 0;
        for (int i = 0; i < n; ++i) {
            const unsigned long long words =
                RNG_PER_PAIR * (state.rowStart[i + 1] - state.rowStart[r0]);
            if (words > budget && i > r0) {
                state.batchRowBegin.push_back(i);
                maxBatchRows = std::max(maxBatchRows, i - r0);
                r0 = i;
            }
        }
        maxBatchRows = std::max(maxBatchRows, n - r0);
        state.batchRowBegin.push_back(n);
    }
    const int numBatches = static_cast<int>(state.batchRowBegin.size()) - 1;

    for (int b = 0; b < RNG_BUFFERS; ++b) {
        CUDA_CHECK(cudaMalloc(&state.d_rng[b], budget * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_rngOffset[b],
                              static_cast<size_t>(maxBatchRows) * n * sizeof(unsigned int)));
    }

    // The generator is a single block and must not be starved by the thousands
    // of ray-tracing blocks it runs alongside, so it gets the higher priority.
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    cudaStream_t genStream, computeStream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&genStream, cudaStreamNonBlocking, prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&computeStream, cudaStreamNonBlocking, prioLow));
    cudaEvent_t genDone[RNG_BUFFERS], kijDone[RNG_BUFFERS];
    for (int b = 0; b < RNG_BUFFERS; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&genDone[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&kijDone[b], cudaEventDisableTiming));
    }
    bool bufInUse[RNG_BUFFERS] = {};

    for (int b = 0; b < numBatches; ++b) {
        const int r0 = state.batchRowBegin[b];
        const int r1 = state.batchRowBegin[b + 1];
        const int rows = r1 - r0;
        const int buf = b % RNG_BUFFERS;
        const unsigned long long words =
            RNG_PER_PAIR * (state.rowStart[r1] - state.rowStart[r0]);

        // Refill this buffer only once its previous consumer is finished
        if (bufInUse[buf]) CUDA_CHECK(cudaStreamWaitEvent(genStream, kijDone[buf], 0));
        if (words > 0) {
            mtGenerateKernel<<<1, MT_GEN_THREADS, 0, genStream>>>(
                state.d_mtRing, state.d_mtCounters, state.d_rng[buf], words);
        }
        CUDA_CHECK(cudaEventRecord(genDone[buf], genStream));

        rngOffsetKernel<<<rows, SCAN_THREADS, 0, computeStream>>>(
            n, r0, state.d_triN, state.d_rowStart, state.d_rngOffset[buf]);

        CUDA_CHECK(cudaStreamWaitEvent(computeStream, genDone[buf], 0));
        dim3 grid((n + KIJ_THREADS - 1) / KIJ_THREADS, rows);
        kijKernel<<<grid, KIJ_THREADS, 0, computeStream>>>(
            n, r0, state.d_rng[buf], state.d_rngOffset[buf], state.d_kij,
            state.d_triA, state.d_triB, state.d_triC, state.d_triN,
            state.d_nodes, state.d_nodeTris);
        CUDA_CHECK(cudaEventRecord(kijDone[buf], computeStream));
        bufInUse[buf] = true;

        // Mirror the progress reporting of the sequential reference.  Reported
        // on submission rather than on completion: waiting here would drain the
        // pipeline that keeps generation and ray tracing overlapped.
        for (int i = r0; i < r1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %d/%d triangles\n", i + 1, n);
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(genStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaGetLastError());

    for (int b = 0; b < RNG_BUFFERS; ++b) {
        CUDA_CHECK(cudaEventDestroy(genDone[b]));
        CUDA_CHECK(cudaEventDestroy(kijDone[b]));
        cudaFree(state.d_rng[b]);
        state.d_rng[b] = nullptr;
        cudaFree(state.d_rngOffset[b]);
        state.d_rngOffset[b] = nullptr;
    }
    CUDA_CHECK(cudaStreamDestroy(genStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    dim3 grid((n + 255) / 256, n);
    tauKernel<<<grid, 256>>>(n, state.d_triCenter, state.d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int blocks = (n + SIM_WARPS_PER_BLOCK - 1) / SIM_WARPS_PER_BLOCK;

    for (int t = 0; t < T; ++t) {
        simKernel<<<blocks, SIM_WARPS_PER_BLOCK * 32>>>(
            n, t, state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaGetLastError());
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    const size_t traceBytes = 2 * static_cast<size_t>(T) * sizeof(float);
    if (traceBytes <= 40 * 1024) {
        distanceKernel<true><<<n, DIST_THREADS, traceBytes>>>(
            n, T, static_cast<int>(state.sourceIndex), state.d_radB, state.d_distances);
    } else {
        distanceKernel<false><<<n, DIST_THREADS>>>(
            n, T, static_cast<int>(state.sourceIndex), state.d_radB, state.d_distances);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    printf("\nValidation:\n");

    // The validation inspects the full Kij and radiosity matrices, so pull them
    // back from the device on demand.
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    state.kij.resize(n * n);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij, n * n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    state.radB.resize(T * n);
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, T * n * sizeof(float),
                          cudaMemcpyDeviceToHost));

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
    int nonZeroKij = 0;
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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    printf("CUDA device: %s (%d SMs, compute %d.%d)\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);
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
            releaseSimulation(state);
            return 1;
        }
    }

    releaseSimulation(state);
    return 0;
}

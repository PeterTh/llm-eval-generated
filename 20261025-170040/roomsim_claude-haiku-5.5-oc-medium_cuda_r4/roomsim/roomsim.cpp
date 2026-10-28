/**
 * Room Response Simulation Benchmark (CUDA)
 * 
 * This is a CUDA implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * GPU parallelization:
 *  - Time delays (Tau): one thread per (i, j) triangle pair.
 *  - Form factors (Kij): the host generates the random sample stream in exactly
 *    the same order as the sequential algorithm (one shared mt19937 stream,
 *    64 draws per non-culled pair) and streams it in chunks to the device, where
 *    one thread evaluates one pair. Visibility uses the octree flattened into
 *    device arrays and traversed with an explicit stack.
 *  - Wave propagation: timesteps are sequential (tau >= 1 for distinct triangles),
 *    but within a timestep one thread per receiving triangle i performs the sum
 *    over j in the original order, so results are identical to the sequential code.
 *  - Cross-correlation: one thread per (triangle, lag) pair, followed by a
 *    per-triangle argmax kernel.
 */

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
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Handling
// ============================================================================

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err_ = (expr);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
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

// Random samples consumed per non-culled triangle pair (2 points x 2 coords per ray)
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;
// Number of triangle pairs staged per form-factor chunk
constexpr size_t PAIRS_PER_CHUNK = size_t(1) << 18;
// Octree traversal stack capacities; a traversal of a tree of depth D needs at most
// 8 * (D + 1) + 1 entries. The smallest capacity that fits the tree is chosen on the host.
constexpr int OCTREE_STACK_SMALL = 64;
constexpr int OCTREE_STACK_LARGE = 192;

// ============================================================================
// Vector and Triangle Types (host side)
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
// Device-side POD types
// ============================================================================

// Plain float3-like struct with the same arithmetic as Vec3
struct Vf {
    float x, y, z;
};

// Triangle vertices and normal as seen by the device
struct DTri {
    Vf a, b, c;
    Vf n;
};

// Flattened octree node
struct DNode {
    Vf center;
    Vf half;
    int32_t child[8];   // -1 if absent
    uint32_t triBegin;  // leaf: offset into triangle list
    uint32_t triCount;  // leaf: number of triangles
    uint32_t leaf;      // 1 if leaf node
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
// Octree for Spatial Acceleration (built on host, traversed on device)
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
// Random Number Generation
// ============================================================================

// Mersenne Twister MT19937 (bit-identical to std::mt19937 for the same seed).
// Produces the raw word stream; each word is mapped to a uniform float in [0, 1)
// on the device by uniformFromRaw(), which reproduces
// std::uniform_real_distribution<float>(0, 1) over mt19937 exactly.
// The state update is written as independent loops so the compiler can vectorize it.
class RandomGenerator {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER_MASK = 0x80000000u;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffu;

    uint32_t mt[N];
    int pos;  // next word of mt[] to output; N forces a twist

    static uint32_t temper(uint32_t y) {
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    void twist() {
        // kk in [0, N-M): reads mt[kk+M], which is still the previous block
#pragma GCC ivdep
        for (int kk = 0; kk < N - M; ++kk) {
            uint32_t y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ (-(y & 1u) & MATRIX_A);
        }
        // kk in [N-M, N-1): reads mt[kk+M-N], already updated (distance M-N >= vector width)
#pragma GCC ivdep
        for (int kk = N - M; kk < N - 1; ++kk) {
            uint32_t y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ (-(y & 1u) & MATRIX_A);
        }
        uint32_t y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ (-(y & 1u) & MATRIX_A);
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) : pos(N) {
        // Same initialization as std::mt19937
        mt[0] = seed;
        for (int i = 1; i < N; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }

    // Fill out[0..count) with the next `count` words of the stream
    void fill(uint32_t* out, size_t count) {
        while (count > 0) {
            if (pos == N) {
                twist();
                pos = 0;
            }
            const size_t take = std::min(count, static_cast<size_t>(N - pos));
            const uint32_t* src = mt + pos;
            for (size_t k = 0; k < take; ++k) {
                out[k] = temper(src[k]);
            }
            out += take;
            count -= take;
            pos += static_cast<int>(take);
        }
    }
};

// ============================================================================
// Device Geometry Helpers
// ============================================================================

__device__ __forceinline__ Vf dSub(Vf a, Vf b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ Vf dAdd(Vf a, Vf b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ Vf dMulS(Vf a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ Vf dDivS(Vf a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__device__ __forceinline__ Vf dNeg(Vf a) { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ float dDot(Vf a, Vf b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ Vf dCross(Vf a, Vf b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ float dSqNorm(Vf a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
__device__ __forceinline__ float dNorm(Vf a) { return sqrtf(dSqNorm(a)); }

// Random point in triangle from two uniform samples (same arithmetic as host version)
__device__ __forceinline__ Vf dRandomPointInTriangle(const DTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vf ab = dSub(t.b, t.a);
    Vf ac = dSub(t.c, t.a);
    return dAdd(dAdd(t.a, dMulS(ab, u)), dMulS(ac, v));
}

// Equivalent to std::uniform_real_distribution<float>(0, 1) applied to one mt19937 word:
// float(raw) * 2^-32, clamped to the largest float below 1.
__device__ __forceinline__ float uniformFromRaw(uint32_t raw) {
    float u = static_cast<float>(raw) * 2.3283064365386963e-10f;  // 2^-32
    return (u >= 1.0f) ? 0.99999994f : u;
}

// Möller-Trumbore ray-triangle intersection; returns FLT_MAX on miss
__device__ __forceinline__ float dRayTriangleIntersect(Vf orig, Vf dir, Vf v0, Vf v1, Vf v2) {
    Vf e1 = dSub(v1, v0);
    Vf e2 = dSub(v2, v0);
    Vf pvec = dCross(dir, e2);
    float det = dDot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    Vf tvec = dSub(orig, v0);
    float u = dDot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vf qvec = dCross(tvec, e1);
    float v = dDot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return dDot(e2, qvec) * invDet;
}

// Ray vs. node bounding box (same separating-axis test as the host octree)
__device__ __forceinline__ bool dRayIntersectsBox(const DNode& nd, Vf p1, Vf p2) {
    Vf d = dMulS(dSub(p2, p1), 0.5f);
    Vf c = dSub(dAdd(p1, d), nd.center);
    Vf ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > nd.half.x + ad.x) return false;
    if (fabsf(c.y) > nd.half.y + ad.y) return false;
    if (fabsf(c.z) > nd.half.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > nd.half.y * ad.z + nd.half.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.half.z * ad.x + nd.half.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.half.x * ad.y + nd.half.y * ad.x + EPSILON) return false;

    return true;
}

// Is the segment from->to blocked by any triangle other than src/dst?
template <int STACK_CAP>
__device__ bool dIsRayBlocked(Vf from, Vf to, const DTri* tris, const DNode* nodes,
                              const uint32_t* triList, uint32_t srcIdx, uint32_t dstIdx) {
    Vf dir = dSub(to, from);
    float rayLen = dNorm(dir);
    if (rayLen < EPSILON) return true;
    Vf dirNorm = dDivS(dir, rayLen);

    int32_t stack[STACK_CAP];
    int sp = 0;
    stack[sp++] = 0;  // root node

    while (sp > 0) {
        const DNode& nd = nodes[stack[--sp]];
        if (nd.leaf) {
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                uint32_t idx = triList[nd.triBegin + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const DTri& tri = tris[idx];
                float dist = dRayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int32_t ch = nd.child[c];
                if (ch >= 0 && dRayIntersectsBox(nodes[ch], from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Kernels
// ============================================================================

// Time delay for every ordered pair; stored transposed: tauT[j*n + i] = tau(i, j)
__global__ void timeDelayKernel(const DTri* __restrict__ tris, int* __restrict__ tauT, uint32_t n) {
    uint64_t k = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total = static_cast<uint64_t>(n) * n;
    if (k >= total) return;

    uint32_t i = static_cast<uint32_t>(k / n);
    uint32_t j = static_cast<uint32_t>(k - static_cast<uint64_t>(i) * n);

    int tau = 0;
    if (i != j) {
        const DTri& ti = tris[i];
        const DTri& tj = tris[j];
        Vf ci = dDivS(dAdd(dAdd(ti.a, ti.b), ti.c), 3.0f);
        Vf cj = dDivS(dAdd(dAdd(tj.a, tj.b), tj.c), 3.0f);
        float dist = dNorm(dSub(ci, cj));
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[static_cast<uint64_t>(j) * n + i] = tau;
}

// Form factor for a batch of non-culled pairs. Thread k handles pair pairs[k]
// using DRAWS_PER_PAIR raw mt19937 words at draws[k * DRAWS_PER_PAIR ...].
// Writes kijT[j*n + i] = Kij(i, j).
template <int STACK_CAP>
__global__ void formFactorKernel(const DTri* __restrict__ tris, const DNode* __restrict__ nodes,
                                 const uint32_t* __restrict__ triList,
                                 const uint2* __restrict__ pairs, const uint32_t* __restrict__ draws,
                                 uint32_t count, uint32_t n, float* __restrict__ kijT) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;

    const uint32_t i = pairs[k].x;
    const uint32_t j = pairs[k].y;
    const DTri& triI = tris[i];
    const DTri& triJ = tris[j];
    const uint32_t* raw = draws + static_cast<size_t>(k) * DRAWS_PER_PAIR;

    float kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vf pI = dRandomPointInTriangle(triI, uniformFromRaw(raw[r * 4 + 0]), uniformFromRaw(raw[r * 4 + 1]));
        Vf pJ = dRandomPointInTriangle(triJ, uniformFromRaw(raw[r * 4 + 2]), uniformFromRaw(raw[r * 4 + 3]));

        if (dIsRayBlocked<STACK_CAP>(pI, pJ, tris, nodes, triList, i, j)) continue;

        Vf v = dSub(pJ, pI);
        float distSqr = dSqNorm(v);
        if (distSqr < EPSILON) continue;

        // cosPhi(v, triI.normal) and cosPhi(-v, triJ.normal)
        float vNormI = dNorm(v);
        float cosPhiI = ZERO;
        if (vNormI > EPSILON) {
            float x = dDot(v, triI.n) / vNormI;
            cosPhiI = (ZERO < x) ? x : ZERO;
        }
        Vf negV = dNeg(v);
        float vNormJ = dNorm(negV);
        float cosPhiJ = ZERO;
        if (vNormJ > EPSILON) {
            float x = dDot(negV, triJ.n) / vNormJ;
            cosPhiJ = (ZERO < x) ? x : ZERO;
        }

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijT[static_cast<size_t>(j) * n + i] = kij * INV_NUM_RAYS;
}

// Wave propagation for one timestep. Thread i computes radB[t][i] with the
// sum over j performed in the same order as the sequential algorithm.
__global__ void simulationStepKernel(const int* __restrict__ tauT, const float* __restrict__ kijT,
                                     const float* __restrict__ areas, const float* __restrict__ rho,
                                     const float* __restrict__ radE, float* __restrict__ radB,
                                     uint32_t n, int t) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sumB = ZERO;
    for (uint32_t j = 0; j < n; ++j) {
        if (i == j) continue;

        const size_t off = static_cast<size_t>(j) * n + i;
        int tauij = tauT[off];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float kij = kijT[off];
        if (kij <= ZERO) continue;

        // Radiosity from source triangle at time when emission occurred
        size_t srcTime = static_cast<size_t>(t - tauij);
        float radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    const size_t off = static_cast<size_t>(t) * n + i;
    radB[off] = rho[i] * sumB + radE[off];
}

// Cross-correlation sums: corr[i*T + t] = sum_{tt>=t} radB[tt][src] * radB[tt-t][...]
__global__ void correlationKernel(const float* __restrict__ radB, float* __restrict__ corr,
                                  uint32_t n, uint32_t T, uint32_t src) {
    uint64_t k = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total = static_cast<uint64_t>(n) * T;
    if (k >= total) return;

    uint32_t i = static_cast<uint32_t>(k / T);
    uint32_t t = static_cast<uint32_t>(k - static_cast<uint64_t>(i) * T);

    float sum = ZERO;
    for (uint32_t tt = t; tt < T; ++tt) {
        float pB = radB[static_cast<size_t>(tt) * n + i];
        float pS = radB[static_cast<size_t>(tt - t) * n + src];
        sum += pS * pB;
    }
    corr[k] = sum;
}

// Per-triangle argmax over lags (first maximum wins, as in the sequential code)
__global__ void argmaxKernel(const float* __restrict__ corr, float* __restrict__ distances,
                             uint32_t n, uint32_t T) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float maxCorr = ZERO;
    int bestT = 0;
    for (uint32_t t = 0; t < T; ++t) {
        float sum = corr[static_cast<size_t>(i) * T + t];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
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
    std::vector<val_t> kij;         // Form factors (N x N, row-major; host copy for validation)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix; host copy for validation)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host build)

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Device-resident simulation data and staging buffers
struct DeviceState {
    DTri* tris = nullptr;
    DNode* nodes = nullptr;
    uint32_t* triList = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* radE = nullptr;
    float* radB = nullptr;
    float* kijT = nullptr;   // transposed form factors: kijT[j*n + i] = Kij(i, j)
    int* tauT = nullptr;     // transposed time delays:   tauT[j*n + i] = Tau(i, j)
    float* corr = nullptr;
    float* distances = nullptr;
    int stackCap = 0;        // octree traversal stack capacity used by the form-factor kernel

    // Double-buffered form-factor staging (pinned host, device)
    uint32_t* hostDraws[2] = {nullptr, nullptr};
    uint2* hostPairs[2] = {nullptr, nullptr};
    uint32_t* devDraws[2] = {nullptr, nullptr};
    uint2* devPairs[2] = {nullptr, nullptr};
    cudaStream_t streams[2] = {nullptr, nullptr};
};

// ============================================================================
// Initialization
// ============================================================================

// Flatten the host octree into device-friendly arrays (depth-first, root = 0)
int flattenOctree(const Octree& node, std::vector<DNode>& nodes, std::vector<uint32_t>& triList,
                  int depth, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    int id = static_cast<int>(nodes.size());
    nodes.push_back(DNode{});

    DNode d{};
    d.center = {node.center.x, node.center.y, node.center.z};
    d.half = {node.halfExtent.x, node.halfExtent.y, node.halfExtent.z};
    for (int c = 0; c < 8; ++c) d.child[c] = -1;

    if (!node.triangleIndices.empty()) {
        d.leaf = 1;
        d.triBegin = static_cast<uint32_t>(triList.size());
        for (size_t idx : node.triangleIndices) triList.push_back(static_cast<uint32_t>(idx));
        d.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                d.child[c] = flattenOctree(*node.children[c], nodes, triList, depth + 1, maxDepth);
            }
        }
    }
    nodes[id] = d;
    return id;
}

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

    // Initialize emission / radiosity buffers
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.clear();
    state.kij.clear();
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// Allocate device buffers and upload the static simulation data
void setupDevice(const SimulationState& state, DeviceState& dev) {
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Triangles
    std::vector<DTri> hTris(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        hTris[i].a = {t.a.x, t.a.y, t.a.z};
        hTris[i].b = {t.b.x, t.b.y, t.b.z};
        hTris[i].c = {t.c.x, t.c.y, t.c.z};
        Vec3 nrm = t.normal();
        hTris[i].n = {nrm.x, nrm.y, nrm.z};
    }
    CUDA_CHECK(cudaMalloc(&dev.tris, n * sizeof(DTri)));
    CUDA_CHECK(cudaMemcpy(dev.tris, hTris.data(), n * sizeof(DTri), cudaMemcpyHostToDevice));

    // Octree
    std::vector<DNode> hNodes;
    std::vector<uint32_t> hTriList;
    int maxDepth = 0;
    flattenOctree(state.octree, hNodes, hTriList, 0, maxDepth);
    const int needed = 8 * (maxDepth + 1) + 1;
    if (needed > OCTREE_STACK_LARGE) {
        fprintf(stderr, "Octree depth %d exceeds traversal stack capacity\n", maxDepth);
        std::exit(EXIT_FAILURE);
    }
    dev.stackCap = (needed <= OCTREE_STACK_SMALL) ? OCTREE_STACK_SMALL : OCTREE_STACK_LARGE;
    CUDA_CHECK(cudaMalloc(&dev.nodes, hNodes.size() * sizeof(DNode)));
    CUDA_CHECK(cudaMemcpy(dev.nodes, hNodes.data(), hNodes.size() * sizeof(DNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev.triList, hTriList.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dev.triList, hTriList.data(), hTriList.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Per-triangle data
    CUDA_CHECK(cudaMalloc(&dev.areas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev.areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev.rho, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev.rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dev.distances, n * sizeof(float)));

    // Matrices
    CUDA_CHECK(cudaMalloc(&dev.kijT, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.kijT, 0, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.tauT, n * n * sizeof(int)));

    // Radiosity
    const size_t radBytes = std::max<size_t>(T * n, 1) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dev.radE, radBytes));
    CUDA_CHECK(cudaMalloc(&dev.radB, radBytes));
    CUDA_CHECK(cudaMemset(dev.radB, 0, radBytes));
    if (T * n > 0) {
        CUDA_CHECK(cudaMemcpy(dev.radE, state.radE.data(), T * n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Correlation buffer
    CUDA_CHECK(cudaMalloc(&dev.corr, std::max<size_t>(n * T, 1) * sizeof(float)));

    // Form-factor staging
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&dev.hostDraws[b], PAIRS_PER_CHUNK * DRAWS_PER_PAIR * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&dev.hostPairs[b], PAIRS_PER_CHUNK * sizeof(uint2)));
        CUDA_CHECK(cudaMalloc(&dev.devDraws[b], PAIRS_PER_CHUNK * DRAWS_PER_PAIR * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dev.devPairs[b], PAIRS_PER_CHUNK * sizeof(uint2)));
        CUDA_CHECK(cudaStreamCreate(&dev.streams[b]));
    }
}

void destroyDevice(DeviceState& dev) {
    cudaFree(dev.tris);
    cudaFree(dev.nodes);
    cudaFree(dev.triList);
    cudaFree(dev.areas);
    cudaFree(dev.rho);
    cudaFree(dev.radE);
    cudaFree(dev.radB);
    cudaFree(dev.kijT);
    cudaFree(dev.tauT);
    cudaFree(dev.corr);
    cudaFree(dev.distances);
    for (int b = 0; b < 2; ++b) {
        cudaFreeHost(dev.hostDraws[b]);
        cudaFreeHost(dev.hostPairs[b]);
        cudaFree(dev.devDraws[b]);
        cudaFree(dev.devPairs[b]);
        cudaStreamDestroy(dev.streams[b]);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Same culling rule as the sequential algorithm: triangles facing the same way
// contribute nothing and consume no random samples.
bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

void computeTimeDelays(const SimulationState& state, DeviceState& dev) {
    printf("Computing time delays (Tau)...\n");

    const uint64_t total = static_cast<uint64_t>(state.numTriangles) * state.numTriangles;
    if (total == 0) return;
    const unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    timeDelayKernel<<<grid, block>>>(dev.tris, dev.tauT, static_cast<uint32_t>(state.numTriangles));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(const SimulationState& state, DeviceState& dev) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    if (n == 0) return;

    // A single RNG stream, consumed in the same order as the sequential code:
    // row-major over (i, j), skipping i == j and culled pairs.
    RandomGenerator rng(42);

    size_t i = 0;
    size_t j = 0;
    int chunk = 0;

    while (i < n) {
        const int b = chunk & 1;
        ++chunk;

        // Wait until the previous kernel using this staging buffer has finished
        CUDA_CHECK(cudaStreamSynchronize(dev.streams[b]));

        uint32_t* hDraws = dev.hostDraws[b];
        uint2* hPairs = dev.hostPairs[b];
        size_t count = 0;

        while (count < PAIRS_PER_CHUNK && i < n) {
            if (i != j && !isCulled(state.triangles[i], state.triangles[j])) {
                hPairs[count] = make_uint2(static_cast<unsigned int>(i), static_cast<unsigned int>(j));
                rng.fill(hDraws + count * DRAWS_PER_PAIR, DRAWS_PER_PAIR);
                ++count;
            }

            ++j;
            if (j == n) {
                j = 0;
                if ((i + 1) % 100 == 0 || i + 1 == n) {
                    printf("  Progress: %zu/%zu triangles\n", i + 1, n);
                }
                ++i;
            }
        }

        if (count == 0) continue;

        cudaStream_t s = dev.streams[b];
        CUDA_CHECK(cudaMemcpyAsync(dev.devPairs[b], hPairs, count * sizeof(uint2),
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dev.devDraws[b], hDraws, count * DRAWS_PER_PAIR * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, s));

        const unsigned int block = 128;
        const unsigned int grid = static_cast<unsigned int>((count + block - 1) / block);
        if (dev.stackCap == OCTREE_STACK_SMALL) {
            formFactorKernel<OCTREE_STACK_SMALL><<<grid, block, 0, s>>>(
                dev.tris, dev.nodes, dev.triList, dev.devPairs[b], dev.devDraws[b],
                static_cast<uint32_t>(count), static_cast<uint32_t>(n), dev.kijT);
        } else {
            formFactorKernel<OCTREE_STACK_LARGE><<<grid, block, 0, s>>>(
                dev.tris, dev.nodes, dev.triList, dev.devPairs[b], dev.devDraws[b],
                static_cast<uint32_t>(count), static_cast<uint32_t>(n), dev.kijT);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(dev.streams[b]));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(const SimulationState& state, DeviceState& dev) {
    printf("Running wave propagation simulation...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const size_t T = state.numTimesteps;
    if (n == 0) return;

    const unsigned int block = 128;
    const unsigned int grid = (n + block - 1) / block;

    for (size_t t = 0; t < T; ++t) {
        simulationStepKernel<<<grid, block>>>(dev.tauT, dev.kijT, dev.areas, dev.rho, dev.radE,
                                              dev.radB, n, static_cast<int>(t));
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(const SimulationState& state, DeviceState& dev, std::vector<val_t>& outDistances) {
    printf("Computing distances via cross-correlation...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);
    if (n > 0 && T > 0) {
        const uint64_t total = static_cast<uint64_t>(n) * T;
        const unsigned int block = 128;
        const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        correlationKernel<<<grid, block>>>(dev.radB, dev.corr, n, T,
                                           static_cast<uint32_t>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
    }
    if (n > 0) {
        const unsigned int block = 128;
        const unsigned int grid = (n + block - 1) / block;
        argmaxKernel<<<grid, block>>>(dev.corr, dev.distances, n, T);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    outDistances.assign(state.numTriangles, ZERO);
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(outDistances.data(), dev.distances, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// Copy full matrices back to the host (only needed for validation)
void downloadForValidation(SimulationState& state, const DeviceState& dev) {
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // kijT is transposed; store as row-major kij[i*n + j] = Kij(i, j)
    std::vector<float> kijT(n * n);
    if (n * n > 0) {
        CUDA_CHECK(cudaMemcpy(kijT.data(), dev.kijT, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    }
    state.kij.assign(n * n, ZERO);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            state.kij[i * n + j] = kijT[j * n + i];
        }
    }

    state.radB.assign(T * n, ZERO);
    if (T * n > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), dev.radB, T * n * sizeof(float), cudaMemcpyDeviceToHost));
    }
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
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    DeviceState dev;
    setupDevice(state, dev);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, dev);
    computeFormFactors(state, dev);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, dev);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, dev, state.distances);

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
        downloadForValidation(state, dev);
        if (!validateResults(state)) {
            destroyDevice(dev);
            return 1;
        }
    }

    destroyDevice(dev);
    return 0;
}

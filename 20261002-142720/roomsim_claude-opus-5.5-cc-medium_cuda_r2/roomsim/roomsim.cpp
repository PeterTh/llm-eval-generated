/**
 * Room Response Simulation Benchmark (CUDA implementation)
 *
 * Room impulse response simulation using radiosity-based wave propagation.
 * It models how sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * GPU parallelization notes:
 *  - The Monte-Carlo form factors consume one sequential mt19937 stream (seed 42) in
 *    (i, j) order, 64 numbers per non-culled pair. To reproduce that stream exactly,
 *    non-culled pairs are compacted on the GPU (giving each pair its stream offset),
 *    the host advances the raw Mersenne-Twister state and ships periodic state
 *    snapshots, and the GPU regenerates the tempered numbers from the snapshots.
 *    Host snapshot generation is pipelined with GPU ray tracing.
 *  - Every floating-point sum keeps the order of the sequential algorithm, and the
 *    device code is compiled without FMA contraction, so results are deterministic.
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

#include <atomic>
#include <thread>

#include <cuda_runtime.h>
#if defined(__PCLMUL__)
#include <wmmintrin.h>
#endif

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

#define HD __host__ __device__ __forceinline__

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
constexpr val_t FLOAT_MAX = 3.402823466e+38f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr HD Vec3() : x(0), y(0), z(0) {}
    constexpr HD Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr HD Vec3(val_t v) : x(v), y(v), z(v) {}

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
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
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
// Octree for Spatial Acceleration (built on the host, flattened for the GPU)
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

// Flattened octree node: c = (center.xyz, first), h = (halfExtent.xyz, count).
// Leaves (count > 0) reference `count` entries of the leaf triangle array starting at
// `first`; inner nodes (count < 0) have -count children stored contiguously at `first`.
struct FlatOctree {
    std::vector<float4> nodeC, nodeH;
    std::vector<float4> leafTris;  // per entry: v0 (w = index bits), e1, e2
    int maxDepth = 0;
};

inline float intBitsToFloat(int v) {
    float f;
    memcpy(&f, &v, sizeof(f));
    return f;
}

void flattenOctree(const Octree& root, const std::vector<Triangle>& tris, FlatOctree& out) {
    std::vector<const Octree*> nodes{&root};
    std::vector<int> depth{0};
    out.nodeC.resize(1);
    out.nodeH.resize(1);
    for (size_t n = 0; n < nodes.size(); ++n) {
        const Octree* node = nodes[n];
        int first, count;
        if (!node->triangleIndices.empty()) {
            first = static_cast<int>(out.leafTris.size() / 3);
            count = static_cast<int>(node->triangleIndices.size());
            for (size_t idx : node->triangleIndices) {
                const Triangle& t = tris[idx];
                Vec3 e1 = t.b - t.a;
                Vec3 e2 = t.c - t.a;
                out.leafTris.push_back(make_float4(t.a.x, t.a.y, t.a.z,
                                                   intBitsToFloat(static_cast<int>(idx))));
                out.leafTris.push_back(make_float4(e1.x, e1.y, e1.z, 0.0f));
                out.leafTris.push_back(make_float4(e2.x, e2.y, e2.z, 0.0f));
            }
        } else {
            first = static_cast<int>(nodes.size());
            count = 0;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    nodes.push_back(node->children[i].get());
                    depth.push_back(depth[n] + 1);
                    out.maxDepth = std::max(out.maxDepth, depth[n] + 1);
                    ++count;
                }
            }
            out.nodeC.resize(nodes.size());
            out.nodeH.resize(nodes.size());
            count = -count;
        }
        out.nodeC[n] = make_float4(node->center.x, node->center.y, node->center.z,
                                   intBitsToFloat(first));
        out.nodeH[n] = make_float4(node->halfExtent.x, node->halfExtent.y, node->halfExtent.z,
                                   intBitsToFloat(count));
    }
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
// Random Number Generation: std::mt19937 (seed 42) + uniform_real_distribution<float>
// ============================================================================
//
// The stream is split into "twist blocks" of 624 outputs. Block k consists of the
// tempered words of the generator state after k+1 twists. The host advances the raw
// state and records a snapshot every RNG_JOB_BLOCKS blocks; the device regenerates
// RNG_JOB_BLOCKS blocks from each snapshot.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr uint32_t MT_MATRIX = 0x9908b0dfu;
constexpr int RNG_JOB_BLOCKS = 32;         // twist blocks generated per snapshot
constexpr size_t BATCH_BLOCKS = 8192;      // twist blocks per batch (multiple of 4)
constexpr int VALUES_PER_PAIR = 4 * NUM_RAYS;
static_assert((BATCH_BLOCKS * MT_N) % VALUES_PER_PAIR == 0, "batch must hold whole pairs");
static_assert(BATCH_BLOCKS % RNG_JOB_BLOCKS == 0, "batch must hold whole jobs");
constexpr size_t BATCH_PAIRS = BATCH_BLOCKS * MT_N / VALUES_PER_PAIR;
constexpr size_t BATCH_JOBS = BATCH_BLOCKS / RNG_JOB_BLOCKS;

HD uint32_t mtMag(uint32_t y) { return (y >> 1) ^ ((y & 1u) ? MT_MATRIX : 0u); }

class HostMersenneState {
public:
    uint32_t mt[MT_N];

    explicit HostMersenneState(uint32_t seed) {
        mt[0] = seed;
        for (int i = 1; i < MT_N; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }

    void twist() {
        int k = 0;
        for (; k < MT_N - MT_M; ++k) {
            uint32_t y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
            mt[k] = mt[k + MT_M] ^ mtMag(y);
        }
        for (; k < MT_N - 1; ++k) {
            uint32_t y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
            mt[k] = mt[k + (MT_M - MT_N)] ^ mtMag(y);
        }
        uint32_t y = (mt[MT_N - 1] & MT_UPPER) | (mt[0] & MT_LOWER);
        mt[MT_N - 1] = mt[MT_M - 1] ^ mtMag(y);
    }
};

// ----------------------------------------------------------------------------
// MT19937 jump-ahead (lets several host threads produce the snapshot stream).
//
// One word step of the generator is a linear map A over GF(2). Its characteristic
// polynomial phi (degree 19937) is recovered with Berlekamp-Massey from the output
// bits; jumping J steps applies g(A) with g = x^J mod phi by Horner's scheme, using
// "step" and "xor of two windows" as the primitive operations. Windows reached after
// >= 31 steps lie in ker phi(A), where g(A) = A^J exactly.
// ----------------------------------------------------------------------------

namespace gf2 {

constexpr int DEG = 19937;
constexpr int WORDS = (DEG + 64) / 64;  // 312 words hold degrees 0..19967
using Poly = std::vector<uint64_t>;

inline void clmul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
#if defined(__PCLMUL__)
    __m128i r = _mm_clmulepi64_si128(_mm_cvtsi64_si128(static_cast<long long>(a)),
                                     _mm_cvtsi64_si128(static_cast<long long>(b)), 0);
    lo = static_cast<uint64_t>(_mm_cvtsi128_si64(r));
    hi = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)));
#else
    lo = 0;
    hi = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1u) {
            lo ^= a << i;
            if (i) hi ^= a >> (64 - i);
        }
    }
#endif
}

inline bool bit(const Poly& p, size_t k) { return (p[k >> 6] >> (k & 63)) & 1u; }

// out = a * b (out has a.size() + b.size() words)
inline void mul(const Poly& a, const Poly& b, Poly& out) {
    out.assign(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) {
        const uint64_t ai = a[i];
        if (!ai) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            uint64_t lo, hi;
            clmul64(ai, b[j], lo, hi);
            out[i + j] ^= lo;
            out[i + j + 1] ^= hi;
        }
    }
}

inline void square(const Poly& a, Poly& out) {
    out.assign(2 * a.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) clmul64(a[i], a[i], out[2 * i], out[2 * i + 1]);
}

// floor(p / x^DEG), WORDS words
inline Poly shiftDown(const Poly& p) {
    Poly r(WORDS, 0);
    const int ws = DEG / 64, bs = DEG % 64;
    for (int i = 0; i < WORDS; ++i) {
        const size_t k = static_cast<size_t>(i + ws);
        uint64_t lo = k < p.size() ? p[k] : 0;
        uint64_t hi = k + 1 < p.size() ? p[k + 1] : 0;
        r[i] = (lo >> bs) | (bs ? (hi << (64 - bs)) : 0);
    }
    return r;
}

inline void truncate(Poly& p) {
    p.resize(WORDS);
    p[DEG / 64] &= (uint64_t(1) << (DEG % 64)) - 1;
}

struct Field {
    Poly phi;  // characteristic polynomial (degree DEG)
    Poly mu;   // floor(x^(2 DEG) / phi), for Barrett reduction

    // p (degree < 2 DEG) mod phi
    Poly reduce(const Poly& p) const {
        Poly q, t;
        mul(shiftDown(p), mu, q);
        mul(shiftDown(q), phi, t);
        Poly r(WORDS);
        for (int i = 0; i < WORDS; ++i) r[i] = p[i] ^ t[i];
        truncate(r);
        return r;
    }

    // x^e mod phi
    Poly powx(uint64_t e) const {
        Poly r(WORDS, 0);
        r[0] = 1;
        Poly sq;
        int topBit = 63;
        while (topBit > 0 && !((e >> topBit) & 1u)) --topBit;
        for (int b = topBit; b >= 0; --b) {
            square(r, sq);
            r = reduce(sq);
            if ((e >> b) & 1u) {  // r *= x
                uint64_t carry = 0;
                for (int i = 0; i < WORDS; ++i) {
                    uint64_t nc = r[i] >> 63;
                    r[i] = (r[i] << 1) | carry;
                    carry = nc;
                }
                if (bit(r, DEG)) {
                    for (int i = 0; i < WORDS; ++i) r[i] ^= phi[i];
                }
            }
        }
        return r;
    }
};

// XOR src << shift into dst
inline void xorShifted(Poly& dst, const Poly& src, size_t srcBits, size_t shift) {
    const size_t ws = shift / 64, bs = shift % 64;
    const size_t srcWords = (srcBits + 63) / 64;
    for (size_t i = 0; i < srcWords; ++i) {
        const uint64_t v = src[i];
        if (!v) continue;
        if (i + ws < dst.size()) dst[i + ws] ^= v << bs;
        if (bs && i + ws + 1 < dst.size()) dst[i + ws + 1] ^= v >> (64 - bs);
    }
}

// Computes the MT19937 characteristic polynomial. Returns false on failure.
inline bool buildField(Field& f) {
    // Bit sequence: lowest bit of the raw state words after the first twist
    const size_t M = 2 * static_cast<size_t>(DEG);
    const size_t seqWords = (M + 63) / 64 + 2;
    Poly rev(seqWords, 0);  // rev bit (M - 1 - k) = s_k
    {
        HostMersenneState st(42);
        size_t k = 0;
        while (k < M) {
            st.twist();
            for (int i = 0; i < MT_N && k < M; ++i, ++k) {
                if (st.mt[i] & 1u) {
                    const size_t r = M - 1 - k;
                    rev[r >> 6] |= uint64_t(1) << (r & 63);
                }
            }
        }
    }

    // Berlekamp-Massey over GF(2)
    const size_t polyWords = (M + 63) / 64 + 2;
    Poly C(polyWords, 0), B(polyWords, 0), T;
    C[0] = 1;
    B[0] = 1;
    size_t L = 0, m = 1, degB = 0;
    for (size_t n = 0; n < M; ++n) {
        // d = sum_{i=0..L} C_i s_{n-i} = parity(C & (rev >> (M - 1 - n)))
        const size_t off = M - 1 - n;
        const size_t ws = off / 64, bs = off % 64;
        uint64_t acc = 0;
        const size_t cw = L / 64 + 1;
        for (size_t i = 0; i < cw; ++i) {
            uint64_t lo = rev[i + ws];
            uint64_t hi = (i + ws + 1 < rev.size()) ? rev[i + ws + 1] : 0;
            uint64_t r = (lo >> bs) | (bs ? (hi << (64 - bs)) : 0);
            uint64_t c = C[i];
            if (i == cw - 1 && (L % 64) != 63) c &= (uint64_t(2) << (L % 64)) - 1;
            acc ^= c & r;
        }
        if (!(__builtin_popcountll(acc) & 1)) {
            ++m;
        } else if (2 * L <= n) {
            T = C;
            xorShifted(C, B, degB + 1, m);
            degB = L;
            L = n + 1 - L;
            B = T;
            m = 1;
        } else {
            xorShifted(C, B, degB + 1, m);
            ++m;
        }
    }
    if (L != static_cast<size_t>(DEG)) return false;

    // phi(x) = x^L C(1/x)
    f.phi.assign(WORDS, 0);
    for (size_t i = 0; i <= L; ++i) {
        if (bit(C, i)) {
            const size_t k = L - i;
            f.phi[k >> 6] |= uint64_t(1) << (k & 63);
        }
    }
    if (!bit(f.phi, 0) || !bit(f.phi, DEG)) return false;

    // mu = floor(x^(2 DEG) / phi) by long division
    Poly rem(2 * WORDS + 1, 0);
    rem[(2 * DEG) / 64] |= uint64_t(1) << ((2 * DEG) % 64);
    f.mu.assign(WORDS + 1, 0);
    for (size_t k = 2 * DEG + 1; k-- > static_cast<size_t>(DEG);) {
        if (bit(rem, k)) {
            xorShifted(rem, f.phi, DEG + 1, k - DEG);
            const size_t q = k - DEG;
            f.mu[q >> 6] |= uint64_t(1) << (q & 63);
        }
    }
    return true;
}

// Window of 624 consecutive sequence words, used for Horner evaluation
struct MtWindow {
    uint32_t w[MT_N];
    int pos = 0;

    void step() {
        const int p1 = pos + 1 == MT_N ? 0 : pos + 1;
        const int pm = pos + MT_M >= MT_N ? pos + MT_M - MT_N : pos + MT_M;
        const uint32_t y = (w[pos] & MT_UPPER) | (w[p1] & MT_LOWER);
        w[pos] = w[pm] ^ mtMag(y);
        pos = p1;
    }

    void addAligned(const uint32_t* x) {
        const int n1 = MT_N - pos;
        for (int k = 0; k < n1; ++k) w[pos + k] ^= x[k];
        for (int k = n1; k < MT_N; ++k) w[k - n1] ^= x[k];
    }
};

// dst = window J steps after src, where g = x^J mod phi (src must be a reachable state)
inline void jump(const uint32_t* src, const Poly& g, uint32_t* dst) {
    MtWindow S;
    memset(S.w, 0, sizeof(S.w));
    int top = -1;
    for (int k = DEG - 1; k >= 0; --k) {
        if (bit(g, static_cast<size_t>(k))) {
            top = k;
            break;
        }
    }
    for (int k = top; k >= 0; --k) {
        S.step();
        if (bit(g, static_cast<size_t>(k))) S.addAligned(src);
    }
    for (int k = 0; k < MT_N; ++k) dst[k] = S.w[(S.pos + k) % MT_N];
}

}  // namespace gf2

// Tempering + generate_canonical<float, 24> (exactly as libstdc++ does it)
__device__ __forceinline__ float mtToFloat(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    float r = __fmul_rn(__uint2float_rn(y), 2.3283064365386963e-10f);  // / 2^32 (exact)
    return r >= 1.0f ? 0.99999994f : r;
}

// One block per snapshot; MT_N threads; each block writes RNG_JOB_BLOCKS * MT_N values.
__global__ void __launch_bounds__(MT_N)
generateRandomKernel(const uint32_t* __restrict__ snapshots, float* __restrict__ out, int numJobs) {
    __shared__ uint32_t mt[MT_N];
    const int job = blockIdx.x;
    if (job >= numJobs) return;
    const int k = threadIdx.x;
    mt[k] = snapshots[static_cast<size_t>(job) * MT_N + k];
    float* dst = out + static_cast<size_t>(job) * RNG_JOB_BLOCKS * MT_N;
    __syncthreads();

    for (int b = 0; b < RNG_JOB_BLOCKS; ++b) {
        dst[b * MT_N + k] = mtToFloat(mt[k]);
        if (b + 1 == RNG_JOB_BLOCKS) break;

        // Parallel twist in three dependency phases
        uint32_t y = 0;
        if (k < MT_N - 1) y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
        __syncthreads();
        if (k < MT_N - MT_M) mt[k] = mt[k + MT_M] ^ mtMag(y);
        __syncthreads();
        if (k >= MT_N - MT_M && k < 2 * (MT_N - MT_M))
            mt[k] = mt[k + (MT_M - MT_N)] ^ mtMag(y);
        __syncthreads();
        if (k >= 2 * (MT_N - MT_M) && k < MT_N - 1)
            mt[k] = mt[k + (MT_M - MT_N)] ^ mtMag(y);
        if (k == MT_N - 1) {
            y = (mt[MT_N - 1] & MT_UPPER) | (mt[0] & MT_LOWER);
            mt[MT_N - 1] = mt[MT_M - 1] ^ mtMag(y);
        }
        __syncthreads();
    }
}

// ============================================================================
// Device geometry helpers
// ============================================================================

__device__ __forceinline__ Vec3 f3(const float4& v) { return Vec3(v.x, v.y, v.z); }

// Segment p1-p2 vs. node bounding box, identical to the sequential test. The per-ray
// terms d = (p2 - p1) / 2, |d| and the midpoint p1 + d are precomputed; all six
// separating-axis conditions are evaluated without branches.
struct SegmentBoxRay {
    Vec3 d, ad, mid;
    __device__ __forceinline__ void init(const Vec3& p1, const Vec3& p2) {
        d = (p2 - p1) * 0.5f;
        ad = Vec3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
        mid = p1 + d;
    }
};

__device__ __forceinline__ bool rayIntersectsBox(const SegmentBoxRay& ray,
                                                 const Vec3& center, const Vec3& halfExtent) {
    const Vec3& d = ray.d;
    const Vec3& ad = ray.ad;
    const Vec3 c = ray.mid - center;

    bool out = fabsf(c.x) > halfExtent.x + ad.x;
    out |= fabsf(c.y) > halfExtent.y + ad.y;
    out |= fabsf(c.z) > halfExtent.z + ad.z;
    out |= fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON;
    out |= fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON;
    out |= fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON;
    return !out;
}

// Möller-Trumbore with precomputed edges e1 = v1 - v0, e2 = v2 - v0
__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                      const Vec3& v0, const Vec3& e1, const Vec3& e2) {
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLOAT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLOAT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLOAT_MAX;

    return e2.dot(qvec) * invDet;
}

constexpr int OCTREE_STACK = 128;

struct StackEntry {
    int node;
    unsigned mask;  // lanes whose ray reached this node
};

// Warp-cooperative ("packet") visibility test. Every lane carries one ray; the warp
// walks the octree together, each node holding the mask of lanes whose ray passed
// all ancestor box tests. A ray therefore tests exactly the triangles the sequential
// per-ray traversal tests, and the any-hit result is identical. Must be called by
// all 32 lanes; `trace` selects lanes with a ray. Returns true if the ray is blocked.
__device__ bool packetRayBlocked(bool trace, const Vec3& from, const Vec3& to,
                                 const float4* __restrict__ nodeC, const float4* __restrict__ nodeH,
                                 const float4* __restrict__ leafTris, int srcTriIdx, int dstTriIdx,
                                 StackEntry* __restrict__ stack) {
    const int lane = threadIdx.x & 31;
    bool blocked = false;
    Vec3 dirNorm;
    val_t maxDist = ZERO;
    SegmentBoxRay seg;
    seg.init(from, to);
    if (trace) {
        Vec3 dir = to - from;
        val_t rayLen = dir.norm();
        if (rayLen < EPSILON) {
            blocked = true;
            trace = false;
        } else {
            dirNorm = dir / rayLen;
            maxDist = rayLen - EPSILON;
        }
    }

    const unsigned live = __ballot_sync(0xffffffffu, trace);
    if (live == 0) return blocked;
    unsigned done = 0;  // lanes whose ray has been found blocked
    int sp = 0;
    if (lane == 0) stack[0] = {0, live};
    sp = 1;
    __syncwarp();
    while (sp > 0) {
        --sp;
        const StackEntry e = stack[sp];
        __syncwarp();
        unsigned m = e.mask & ~done;
        if (m == 0) continue;
        const float4 c = __ldg(&nodeC[e.node]);
        const float4 h = __ldg(&nodeH[e.node]);
        // The root box is not tested by the sequential traversal
        if (e.node != 0) {
            const bool pass = ((m >> lane) & 1u) && rayIntersectsBox(seg, f3(c), f3(h));
            m = __ballot_sync(0xffffffffu, pass);
            if (m == 0) continue;
        }
        const int first = __float_as_int(c.w);
        const int count = __float_as_int(h.w);
        if (count > 0) {
            const bool mine = (m >> lane) & 1u;
            for (int k = first; k < first + count; ++k) {
                const float4 a = __ldg(&leafTris[3 * k]);
                const int idx = __float_as_int(a.w);
                if (mine && !blocked && idx != srcTriIdx && idx != dstTriIdx) {
                    const float4 e1 = __ldg(&leafTris[3 * k + 1]);
                    const float4 e2 = __ldg(&leafTris[3 * k + 2]);
                    val_t dist = rayTriangleIntersect(from, dirNorm, f3(a), f3(e1), f3(e2));
                    if (dist > EPSILON && dist < maxDist) blocked = true;
                }
            }
            done |= __ballot_sync(0xffffffffu, blocked);
            if ((live & ~done) == 0) break;
        } else {
            // Push children in reverse so they are visited in the original order
            const int n = -count;
            if (lane < n) stack[sp + lane] = {first + n - 1 - lane, m};
            sp += n;
            __syncwarp();
        }
    }
    return blocked;
}

__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

__device__ __forceinline__ Vec3 pointInTriangle(const Vec3& a, const Vec3& b, const Vec3& c,
                                                val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return a + ab * u + ac * v;
}

// ============================================================================
// Precomputation kernels
// ============================================================================

struct DevTriangles {
    const float4* a;
    const float4* b;
    const float4* c;
    const float4* n;  // normal (w = area)
};

__device__ __forceinline__ bool pairCulled(const DevTriangles& t, int i, int j) {
    return f3(__ldg(&t.n[i])).dot(f3(__ldg(&t.n[j]))) > 0.99f;
}

// tau[i * ld + j] = ceil(|c_i - c_j| / WAVE_SPEED), diagonal 0. Triangle centers lie
// inside the room sphere of radius 10, so tau <= ceil(20 / WAVE_SPEED) = 40 fits 8 bits.
using tau_t = uint8_t;

__global__ void tauKernel(DevTriangles t, tau_t* __restrict__ tau, int N, int ld) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    const Vec3 cj = (f3(t.a[j]) + f3(t.b[j]) + f3(t.c[j])) / 3.0f;
    for (int i = blockIdx.y; i < N; i += gridDim.y) {
        int tauij = 0;
        if (i != j) {
            Vec3 ci = (f3(t.a[i]) + f3(t.b[i]) + f3(t.c[i])) / 3.0f;
            val_t dist = (ci - cj).norm();
            tauij = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
        }
        tau[static_cast<size_t>(i) * ld + j] = static_cast<tau_t>(tauij);
    }
}

constexpr int ROW_THREADS = 256;

// Number of non-culled pairs (i, j), j != i, per row i
__global__ void __launch_bounds__(ROW_THREADS)
countPairsKernel(DevTriangles t, unsigned long long* __restrict__ rowCount, int N) {
    __shared__ int warpSums[ROW_THREADS / 32];
    const int i = blockIdx.x;
    int cnt = 0;
    for (int j = threadIdx.x; j < N; j += ROW_THREADS) {
        if (j != i && !pairCulled(t, i, j)) ++cnt;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long s = 0;
        for (int w = 0; w < ROW_THREADS / 32; ++w) s += warpSums[w];
        rowCount[i] = s;
    }
}

// Ordered compaction of non-culled pairs, packed as (i << 16) | j
__global__ void __launch_bounds__(ROW_THREADS)
fillPairsKernel(DevTriangles t, const unsigned long long* __restrict__ rowOffset,
                uint32_t* __restrict__ pairs, int N) {
    __shared__ int warpSums[ROW_THREADS / 32];
    const int i = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned long long base = rowOffset[i];
    for (int j0 = 0; j0 < N; j0 += ROW_THREADS) {
        const int j = j0 + threadIdx.x;
        const bool keep = j < N && j != i && !pairCulled(t, i, j);
        const unsigned ballot = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) warpSums[warp] = __popc(ballot);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < ROW_THREADS / 32; ++w) {
            if (w < warp) before += warpSums[w];
            total += warpSums[w];
        }
        if (keep) {
            const int rank = before + __popc(ballot & ((1u << lane) - 1u));
            pairs[base + rank] = (static_cast<uint32_t>(i) << 16) | static_cast<uint32_t>(j);
        }
        base += total;
        __syncthreads();
    }
}

constexpr int KIJ_THREADS = 128;

// One thread per (pair, ray); 16 consecutive lanes form one pair, each warp traces
// the 32 rays of two consecutive pairs as one packet.
// Writes the form factors in pair order: kijPairs[p] = Kij.
__global__ void __launch_bounds__(KIJ_THREADS, 10)
formFactorKernel(const uint32_t* __restrict__ pairs, size_t numPairs,
                 const float* __restrict__ rnd, DevTriangles t,
                 const float4* __restrict__ nodeC, const float4* __restrict__ nodeH,
                 const float4* __restrict__ leafTris, float* __restrict__ kijPairs) {
    __shared__ StackEntry stacks[KIJ_THREADS / 32][OCTREE_STACK];
    const size_t gt = static_cast<size_t>(blockIdx.x) * KIJ_THREADS + threadIdx.x;
    const size_t p = gt / NUM_RAYS;
    const int r = static_cast<int>(gt % NUM_RAYS);
    const bool active = p < numPairs;

    val_t contrib = ZERO;
    int i = 0, j = 0;
    Vec3 pI, pJ;
    if (active) {
        const uint32_t packed = pairs[p];
        i = static_cast<int>(packed >> 16);
        j = static_cast<int>(packed & 0xffffu);
        const float4 rv = *reinterpret_cast<const float4*>(rnd + p * VALUES_PER_PAIR + 4 * r);
        pI = pointInTriangle(f3(__ldg(&t.a[i])), f3(__ldg(&t.b[i])), f3(__ldg(&t.c[i])), rv.x, rv.y);
        pJ = pointInTriangle(f3(__ldg(&t.a[j])), f3(__ldg(&t.b[j])), f3(__ldg(&t.c[j])), rv.z, rv.w);
    }

    const bool blocked = packetRayBlocked(active, pI, pJ, nodeC, nodeH, leafTris, i, j,
                                          stacks[threadIdx.x >> 5]);

    if (active && !blocked) {
        const Vec3 nI = f3(__ldg(&t.n[i]));
        const Vec3 nJ = f3(__ldg(&t.n[j]));
        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr >= EPSILON) {
            val_t cosPhiI = cosPhi(v, nI);
            val_t cosPhiJ = cosPhi(-v, nJ);
            if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }
    }

    // Sum the rays of a pair in ray order (skipped rays contribute +0)
    const int lane = threadIdx.x & 31;
    const int groupBase = lane & ~(NUM_RAYS - 1);
    val_t kij = ZERO;
#pragma unroll
    for (int k = 0; k < NUM_RAYS; ++k) {
        kij += __shfl_sync(0xffffffffu, contrib, groupBase + k);
    }
    if (active && r == 0) {
        kijPairs[p] = kij * INV_NUM_RAYS;
    }
}

// Scatters the pair-ordered form factors into the propagation weight matrix
// w[i*ld+j] = min(Kij * area_j, 1) (left 0 where Kij <= 0, which the simulation skips)
// and counts form factors > EPSILON.
__global__ void weightKernel(const uint32_t* __restrict__ pairs, const float* __restrict__ kijPairs,
                             size_t numPairs, DevTriangles t, float* __restrict__ w, int ld,
                             unsigned long long* __restrict__ nonZeroCount) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool nz = false;
    if (p < numPairs) {
        const uint32_t packed = pairs[p];
        const int i = static_cast<int>(packed >> 16);
        const int j = static_cast<int>(packed & 0xffffu);
        const val_t kij = kijPairs[p];
        nz = kij > EPSILON;
        if (kij > ZERO) {
            const val_t kw = kij * t.n[j].w;
            w[static_cast<size_t>(i) * ld + j] = ONE < kw ? ONE : kw;  // std::min(kw, ONE)
        }
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, nz);
    if ((threadIdx.x & 31) == 0 && ballot) {
        atomicAdd(nonZeroCount, static_cast<unsigned long long>(__popc(ballot)));
    }
}

// ============================================================================
// Simulation and distance kernels
// ============================================================================

constexpr int SIM_ROWS = 32;     // receiving triangles per block
constexpr int SIM_CHUNK = 128;   // emitters per tile (4 per lane)
constexpr int SIM_WARPS = 8;     // warp 0 sums, the others compute contributions
constexpr int SIM_THREADS = 32 * SIM_WARPS;
constexpr int SIM_STRIDE = SIM_CHUNK + 4;  // padded tile row (keeps float4 alignment)

// One simulation timestep. Loader warps evaluate the contributions w_ij * radB[t - tau_ij][j]
// of a 32 x 128 tile (one row per warp iteration, 4 emitters per lane) into shared
// memory; warp 0 (one lane per receiving triangle i) accumulates them in sequential j
// order. Tiles are double buffered so loading and summation overlap. Skipped terms are
// stored as +0, which leaves the non-negative sums unchanged. Rows of w / tau are
// padded to a multiple of 8 entries with w = 0.
__global__ void __launch_bounds__(SIM_THREADS)
simulationStepKernel(const float* __restrict__ w, const tau_t* __restrict__ tau,
                     float* __restrict__ radB, int N, int ld, int t, int sourceIndex,
                     int timeOff, val_t rho) {
    __shared__ __align__(16) float tile[2][SIM_ROWS][SIM_STRIDE];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i0 = blockIdx.x * SIM_ROWS;
    const float* radNow = radB + static_cast<size_t>(t) * N;
    const int numChunks = (N + SIM_CHUNK - 1) / SIM_CHUNK;
    val_t sumB = ZERO;

    for (int c = 0; c <= numChunks; ++c) {
        if (warp == 0) {
            if (c > 0) {
                const float4* row = reinterpret_cast<const float4*>(tile[(c - 1) & 1][lane]);
                const int cnt4 = (min(SIM_CHUNK, N - (c - 1) * SIM_CHUNK) + 3) / 4;
                for (int k = 0; k < cnt4; ++k) {
                    const float4 v = row[k];
                    sumB += v.x;
                    sumB += v.y;
                    sumB += v.z;
                    sumB += v.w;
                }
            }
        } else if (c < numChunks) {
            const int j = c * SIM_CHUNK + 4 * lane;  // first of this lane's 4 emitters
#pragma unroll 2
            for (int r = warp - 1; r < SIM_ROWS; r += SIM_WARPS - 1) {
                const int i = i0 + r;
                float4 out = make_float4(ZERO, ZERO, ZERO, ZERO);
                if (i < N && j < ld) {
                    const size_t k = static_cast<size_t>(i) * ld + j;
                    const float4 wv = __ldg(reinterpret_cast<const float4*>(w + k));
                    const unsigned tv = __ldg(reinterpret_cast<const unsigned*>(tau + k));
                    float* o = &out.x;
                    const float wq[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int tauij = static_cast<int>((tv >> (8 * q)) & 0xffu);
                        // w == 0 covers Kij <= 0, the diagonal and the row padding
                        if (wq[q] != ZERO && t >= tauij && j + q != i) {
                            const val_t radJ = __ldg(radNow - static_cast<size_t>(tauij) * N + j + q);
                            if (radJ > ZERO) o[q] = wq[q] * radJ;
                        }
                    }
                }
                *reinterpret_cast<float4*>(&tile[c & 1][r][4 * lane]) = out;
            }
        }
        __syncthreads();
    }

    const int i = i0 + lane;
    if (warp == 0 && i < N) {
        const val_t radE = (i == sourceIndex && t < timeOff) ? ONE : ZERO;
        radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE;
    }
}

// corr[t * N + i] = sum_{tt >= t} radB[tt - t][src] * radB[tt][i]
__global__ void correlationKernel(const float* __restrict__ radB, float* __restrict__ corr,
                                  int N, int T, int sourceIndex) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int t = blockIdx.y; t < T; t += gridDim.y) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * N + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        corr[static_cast<size_t>(t) * N + i] = sum;
    }
}

__global__ void distanceKernel(const float* __restrict__ corr, float* __restrict__ distances,
                               int N, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        val_t s = corr[static_cast<size_t>(t) * N + i];
        if (s > maxCorr) {
            maxCorr = s;
            bestT = t;
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Per-GPU resources for the form factor computation
struct DeviceContext {
    int device = 0;
    cudaStream_t compute = nullptr;   // RNG regeneration + ray tracing
    cudaStream_t copy = nullptr;      // snapshot uploads
    float4 *a = nullptr, *b = nullptr, *c = nullptr, *n = nullptr;
    float4 *nodeC = nullptr, *nodeH = nullptr, *leafTris = nullptr;
    uint32_t* dSnap[2] = {nullptr, nullptr};
    cudaEvent_t copyDone[2], genDone[2];
    float* dRnd = nullptr;
    uint32_t* pairs = nullptr;                // non-culled pairs, packed (i << 16) | j
    float* kijPairs = nullptr;                // form factors in pair order
    unsigned long long* rowOffset = nullptr;  // first pair index of each row

    DevTriangles triangles() const { return {a, b, c, n}; }
};

constexpr int MAX_RNG_WORKERS = 32;          // host threads producing RNG snapshots
constexpr int RNG_GROUP_BATCHES = 4;         // consecutive batches per worker between jumps
constexpr size_t RNG_PREFIX_GROUPS = 2;      // groups produced sequentially while jump-ahead is set up
constexpr size_t PARALLEL_RNG_MIN_BATCHES = 8;  // below this, one sequential worker
constexpr int SNAPSHOT_SLOTS = 2 * MAX_RNG_WORKERS;  // pinned host snapshot buffers

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, host copy)
    std::vector<val_t> distances;   // Computed distances from source
    val_t reflectivity;             // Reflectivity of every triangle (0.0 to 1.0)
    unsigned long long nonZeroKij = 0;

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;

    size_t sourceIndex;

    // Characteristic polynomial of std::mt19937 for jump-ahead. It is a constant of the
    // generator (independent of the input), so like a precomputed skip-ahead table it is
    // built in the background during initialization.
    gf2::Field rngField;
    std::thread rngFieldThread;

    // GPUs; devices[0] also holds the simulation data
    std::vector<DeviceContext> devices;
    uint32_t* hSnapPool = nullptr;  // SNAPSHOT_SLOTS pinned snapshot buffers
    std::vector<std::vector<cudaEvent_t>> slotEvents;  // [slot][device]
    size_t ld = 0;                  // padded row length of dW / dTau
    float* dW = nullptr;            // Propagation weights min(Kij * area_j, 1) (N x ld)
    tau_t* dTau = nullptr;          // Time delays (N x ld)
    float* dRadB = nullptr;         // Reflected radiosity (T x N)
    float* dDist = nullptr;

    cudaStream_t stream() const { return devices[0].compute; }

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

template <typename T>
T* deviceUpload(const std::vector<T>& h) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    state.rngFieldThread = std::thread([&state] {
        if (!gf2::buildField(state.rngField)) {
            fprintf(stderr, "MT19937 jump-ahead setup failed\n");
            exit(1);
        }
    });

    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    if (state.numTriangles > 65536) {
        fprintf(stderr, "Too many triangles for this implementation (max 65536)\n");
        exit(1);
    }

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctree(state.octree, state.triangles, state.flatOctree);
    if (1 + 7 * state.flatOctree.maxDepth > OCTREE_STACK) {
        fprintf(stderr, "Octree too deep for traversal stack\n");
        exit(1);
    }

    // Initialize areas
    const size_t N = state.numTriangles;
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.radB.resize(timesteps * N, ZERO);
    state.distances.resize(N, ZERO);

    std::vector<float4> ha(N), hb(N), hc(N), hn(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        ha[i] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        hb[i] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        hc[i] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        hn[i] = make_float4(t._normal.x, t._normal.y, t._normal.z, state.areas[i]);
    }

    // Device setup: geometry is replicated on every GPU
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    state.devices.resize(deviceCount);
    for (int d = 0; d < deviceCount; ++d) {
        DeviceContext& ctx = state.devices[d];
        ctx.device = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));  // create context outside of timed regions
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.copy, cudaStreamNonBlocking));
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.copyDone[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.genDone[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaMalloc(&ctx.dSnap[k], BATCH_JOBS * MT_N * sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMalloc(&ctx.dRnd, BATCH_BLOCKS * MT_N * sizeof(float)));
        ctx.a = deviceUpload(ha);
        ctx.b = deviceUpload(hb);
        ctx.c = deviceUpload(hc);
        ctx.n = deviceUpload(hn);
        ctx.nodeC = deviceUpload(state.flatOctree.nodeC);
        ctx.nodeH = deviceUpload(state.flatOctree.nodeH);
        ctx.leafTris = deviceUpload(state.flatOctree.leafTris);
        const size_t maxPairs = std::max<size_t>(1, N * (N - 1));
        CUDA_CHECK(cudaMalloc(&ctx.pairs, maxPairs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ctx.kijPairs, maxPairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.rowOffset, N * sizeof(unsigned long long)));
        if (d > 0) {
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, d, 0));
            if (canAccess) cudaDeviceEnablePeerAccess(0, 0);
            cudaGetLastError();
        }
    }

    CUDA_CHECK(cudaMallocHost(&state.hSnapPool, SNAPSHOT_SLOTS * BATCH_JOBS * MT_N * sizeof(uint32_t)));
    state.slotEvents.assign(SNAPSHOT_SLOTS, std::vector<cudaEvent_t>(deviceCount));
    for (int d = 0; d < deviceCount; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        for (int k = 0; k < SNAPSHOT_SLOTS; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&state.slotEvents[k][d], cudaEventDisableTiming));
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    const size_t radSize = std::max<size_t>(1, timesteps * N);
    state.ld = (N + 7) & ~static_cast<size_t>(7);
    CUDA_CHECK(cudaMalloc(&state.dW, N * state.ld * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTau, N * state.ld * sizeof(tau_t)));
    CUDA_CHECK(cudaMemset(state.dTau, 0, N * state.ld * sizeof(tau_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, radSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dDist, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dW, 0, N * state.ld * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, radSize * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void releaseSimulation(SimulationState& state) {
    if (state.rngFieldThread.joinable()) state.rngFieldThread.join();
    for (DeviceContext& ctx : state.devices) {
        cudaSetDevice(ctx.device);
        cudaStreamSynchronize(ctx.compute);
        cudaStreamSynchronize(ctx.copy);
        for (void* p : {static_cast<void*>(ctx.a), static_cast<void*>(ctx.b),
                        static_cast<void*>(ctx.c), static_cast<void*>(ctx.n),
                        static_cast<void*>(ctx.nodeC), static_cast<void*>(ctx.nodeH),
                        static_cast<void*>(ctx.leafTris), static_cast<void*>(ctx.dRnd),
                        static_cast<void*>(ctx.dSnap[0]), static_cast<void*>(ctx.dSnap[1]),
                        static_cast<void*>(ctx.pairs), static_cast<void*>(ctx.kijPairs),
                        static_cast<void*>(ctx.rowOffset)}) {
            cudaFree(p);
        }
        for (auto& ev : state.slotEvents) cudaEventDestroy(ev[ctx.device]);
        for (int k = 0; k < 2; ++k) {
            cudaEventDestroy(ctx.copyDone[k]);
            cudaEventDestroy(ctx.genDone[k]);
        }
        cudaStreamDestroy(ctx.compute);
        cudaStreamDestroy(ctx.copy);
    }
    cudaFreeHost(state.hSnapPool);
    cudaSetDevice(0);
    for (void* p : {static_cast<void*>(state.dW), static_cast<void*>(state.dTau),
                    static_cast<void*>(state.dRadB), static_cast<void*>(state.dDist)}) {
        cudaFree(p);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);
    DeviceContext& dev0 = state.devices[0];

    // 1. Enumerate non-culled pairs in sequential (i, j) order; pair p consumes
    //    random numbers [64 p, 64 p + 64) of the mt19937 stream.
    CUDA_CHECK(cudaSetDevice(0));
    countPairsKernel<<<N, ROW_THREADS, 0, dev0.compute>>>(dev0.triangles(), dev0.rowOffset, N);
    std::vector<unsigned long long> rowOffset(N);
    CUDA_CHECK(cudaMemcpyAsync(rowOffset.data(), dev0.rowOffset, N * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, dev0.compute));
    CUDA_CHECK(cudaStreamSynchronize(dev0.compute));
    unsigned long long numPairs = 0;
    for (int i = 0; i < N; ++i) {
        unsigned long long c = rowOffset[i];
        rowOffset[i] = numPairs;
        numPairs += c;
    }

    const size_t numBatches = (numPairs + BATCH_PAIRS - 1) / BATCH_PAIRS;
    const int numDevices = static_cast<int>(
        std::max<size_t>(1, std::min<size_t>(state.devices.size(), numBatches)));

    // Pair lists (needed by every GPU). GPUs other than 0 send their form factors
    // into device 0's kijPairs buffer.
    for (int d = 0; d < numDevices; ++d) {
        DeviceContext& ctx = state.devices[d];
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaMemcpyAsync(ctx.rowOffset, rowOffset.data(), N * sizeof(unsigned long long),
                                   cudaMemcpyHostToDevice, ctx.compute));
        fillPairsKernel<<<N, ROW_THREADS, 0, ctx.compute>>>(ctx.triangles(), ctx.rowOffset, ctx.pairs, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Batched random number regeneration + ray tracing (round-robin over GPUs).
    //    Host worker threads produce the generator snapshots of each batch: worker c
    //    owns batch groups c, c + W, c + 2W, ... and uses jump-ahead to skip the
    //    groups of the other workers. The main thread uploads batches in order.
    const size_t numGroups = (numBatches + RNG_GROUP_BATCHES - 1) / RNG_GROUP_BATCHES;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int numWorkers = static_cast<int>(std::min<size_t>({static_cast<size_t>(MAX_RNG_WORKERS),
                                                        static_cast<size_t>(hw), numGroups}));
    if (numBatches < PARALLEL_RNG_MIN_BATCHES) numWorkers = 1;
    if (state.rngFieldThread.joinable()) state.rngFieldThread.join();
    const gf2::Field& field = state.rngField;

    // Host snapshot slots; slot k is always used by device k % numDevices
    const int numSlots = (SNAPSHOT_SLOTS / numDevices) * numDevices;
    auto hostSlot = [&](int k) { return state.hSnapPool + static_cast<size_t>(k) * BATCH_JOBS * MT_N; };
    auto slotEvent = [&](int k) { return state.slotEvents[k][state.devices[k % numDevices].device]; };
    std::unique_ptr<std::atomic<long long>[]> slotReady(new std::atomic<long long>[numSlots]);
    std::unique_ptr<std::atomic<long long>[]> slotIssued(new std::atomic<long long>[numSlots]);
    for (int k = 0; k < numSlots; ++k) {
        slotReady[k].store(-1);
        slotIssued[k].store(-1);
    }

    auto jobsOfBatch = [&](size_t b) {
        const size_t pairsInBatch = std::min<size_t>(BATCH_PAIRS, numPairs - b * BATCH_PAIRS);
        const size_t blocksNeeded = (pairsInBatch * VALUES_PER_PAIR + MT_N - 1) / MT_N;
        return static_cast<int>((blocksNeeded + RNG_JOB_BLOCKS - 1) / RNG_JOB_BLOCKS);
    };

    // Worker 0 produces the first groups sequentially (no jump-ahead needed, so the
    // GPUs get work immediately); workers 1..W-1 take the remaining groups round-robin.
    const size_t prefixGroups = numWorkers > 1 ? std::min<size_t>(numGroups, RNG_PREFIX_GROUPS) : numGroups;
    const int jumpers = numWorkers - 1;

    auto produceGroup = [&](HostMersenneState& gen, size_t g) {
        const size_t bEnd = std::min(numBatches, (g + 1) * RNG_GROUP_BATCHES);
        for (size_t b = g * RNG_GROUP_BATCHES; b < bEnd; ++b) {
            const int slot = static_cast<int>(b % numSlots);
            if (b >= static_cast<size_t>(numSlots)) {
                const long long prev = static_cast<long long>(b) - numSlots;
                long long v;
                while ((v = slotIssued[slot].load(std::memory_order_acquire)) != prev) {
                    slotIssued[slot].wait(v);
                }
                cudaEventSynchronize(slotEvent(slot));
            }
            uint32_t* snap = hostSlot(slot);
            const int jobs = jobsOfBatch(b);
            // Only the final batch can be partial; all others advance BATCH_BLOCKS
            for (int m = 0; m < jobs; ++m) {
                memcpy(snap + static_cast<size_t>(m) * MT_N, gen.mt, sizeof(gen.mt));
                for (int k = 0; k < RNG_JOB_BLOCKS; ++k) gen.twist();
            }
            slotReady[slot].store(static_cast<long long>(b), std::memory_order_release);
            slotReady[slot].notify_all();
        }
    };

    auto worker = [&](int c) {
        HostMersenneState gen(42);
        gen.twist();  // state whose tempered words are stream block 0
        if (c == 0) {
            for (size_t g = 0; g < prefixGroups; ++g) produceGroup(gen, g);
            return;
        }
        const size_t firstGroup = prefixGroups + (c - 1);
        if (firstGroup >= numGroups) return;
        constexpr uint64_t GROUP_WORDS = static_cast<uint64_t>(RNG_GROUP_BATCHES) * BATCH_BLOCKS * MT_N;
        gf2::jump(gen.mt, field.powx(GROUP_WORDS * firstGroup), gen.mt);
        gf2::Poly skip;
        for (size_t g = firstGroup; g < numGroups; g += jumpers) {
            if (g != firstGroup && jumpers > 1) {
                if (skip.empty()) skip = field.powx(GROUP_WORDS * (jumpers - 1));
                gf2::jump(gen.mt, skip, gen.mt);
            }
            produceGroup(gen, g);
        }
    };

    std::vector<std::thread> workers;
    for (int c = 0; c < numWorkers; ++c) workers.emplace_back(worker, c);

    for (size_t b = 0; b < numBatches; ++b) {
        const size_t firstPair = b * BATCH_PAIRS;
        const size_t pairsInBatch = std::min<size_t>(BATCH_PAIRS, numPairs - firstPair);
        const int jobs = jobsOfBatch(b);

        const int d = static_cast<int>(b % numDevices);
        const int slot = static_cast<int>((b / numDevices) & 1);
        const int hs = static_cast<int>(b % numSlots);
        DeviceContext& ctx = state.devices[d];

        long long v;
        while ((v = slotReady[hs].load(std::memory_order_acquire)) != static_cast<long long>(b)) {
            slotReady[hs].wait(v);
        }

        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.copy, ctx.genDone[slot], 0));
        CUDA_CHECK(cudaMemcpyAsync(ctx.dSnap[slot], hostSlot(hs),
                                   static_cast<size_t>(jobs) * MT_N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, ctx.copy));
        CUDA_CHECK(cudaEventRecord(ctx.copyDone[slot], ctx.copy));
        CUDA_CHECK(cudaEventRecord(slotEvent(hs), ctx.copy));
        slotIssued[hs].store(static_cast<long long>(b), std::memory_order_release);
        slotIssued[hs].notify_all();

        CUDA_CHECK(cudaStreamWaitEvent(ctx.compute, ctx.copyDone[slot], 0));
        generateRandomKernel<<<jobs, MT_N, 0, ctx.compute>>>(ctx.dSnap[slot], ctx.dRnd, jobs);
        CUDA_CHECK(cudaEventRecord(ctx.genDone[slot], ctx.compute));
        const size_t threads = pairsInBatch * NUM_RAYS;
        const unsigned grid = static_cast<unsigned>((threads + KIJ_THREADS - 1) / KIJ_THREADS);
        formFactorKernel<<<grid, KIJ_THREADS, 0, ctx.compute>>>(
            ctx.pairs + firstPair, pairsInBatch, ctx.dRnd, ctx.triangles(),
            ctx.nodeC, ctx.nodeH, ctx.leafTris, ctx.kijPairs + firstPair);
        CUDA_CHECK(cudaGetLastError());
        if (d != 0) {
            CUDA_CHECK(cudaMemcpyPeerAsync(dev0.kijPairs + firstPair, dev0.device,
                                           ctx.kijPairs + firstPair, ctx.device,
                                           pairsInBatch * sizeof(float), ctx.compute));
        }
    }
    for (auto& w : workers) w.join();

    for (int d = 0; d < numDevices; ++d) {
        CUDA_CHECK(cudaSetDevice(state.devices[d].device));
        CUDA_CHECK(cudaStreamSynchronize(state.devices[d].compute));
        CUDA_CHECK(cudaStreamSynchronize(state.devices[d].copy));
    }

    // 3. Scatter form factors into the propagation weight matrix (and count non-zero Kij)
    CUDA_CHECK(cudaSetDevice(0));
    unsigned long long* dNonZero = nullptr;
    CUDA_CHECK(cudaMalloc(&dNonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(dNonZero, 0, sizeof(unsigned long long), dev0.compute));
    if (numPairs > 0) {
        weightKernel<<<static_cast<unsigned>((numPairs + 255) / 256), 256, 0, dev0.compute>>>(
            dev0.pairs, dev0.kijPairs, numPairs, dev0.triangles(), state.dW,
            static_cast<int>(state.ld), dNonZero);
    }
    CUDA_CHECK(cudaMemcpyAsync(&state.nonZeroKij, dNonZero, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, dev0.compute));
    CUDA_CHECK(cudaStreamSynchronize(dev0.compute));
    CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < N; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            printf("  Progress: %d/%d triangles\n", i + 1, N);
        }
    }

    cudaFree(dNonZero);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);
    CUDA_CHECK(cudaSetDevice(0));
    tauKernel<<<dim3((N + 255) / 256, std::min(N, 65535)), 256, 0, state.stream()>>>(
        state.devices[0].triangles(), state.dTau, N, static_cast<int>(state.ld));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = T / 2;  // source emits during [0, T/2)
    const unsigned grid = static_cast<unsigned>((N + SIM_ROWS - 1) / SIM_ROWS);
    CUDA_CHECK(cudaSetDevice(0));

    for (int t = 0; t < T; ++t) {
        simulationStepKernel<<<grid, SIM_THREADS, 0, state.stream()>>>(
            state.dW, state.dTau, state.dRadB, N, static_cast<int>(state.ld), t,
            static_cast<int>(state.sourceIndex),
            timeOff, state.reflectivity);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(state.stream()));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    if (T > 0) {
        float* dCorr = nullptr;
        CUDA_CHECK(cudaMalloc(&dCorr, static_cast<size_t>(T) * N * sizeof(float)));
        correlationKernel<<<dim3((N + 127) / 128, std::min(T, 65535)), 128, 0, state.stream()>>>(
            state.dRadB, dCorr, N, T, static_cast<int>(state.sourceIndex));
        distanceKernel<<<(N + 127) / 128, 128, 0, state.stream()>>>(dCorr, state.dDist, N, T);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(state.radB.data(), state.dRadB,
                                   static_cast<size_t>(T) * N * sizeof(float),
                                   cudaMemcpyDeviceToHost, state.stream()));
        CUDA_CHECK(cudaMemcpyAsync(state.distances.data(), state.dDist, N * sizeof(float),
                                   cudaMemcpyDeviceToHost, state.stream()));
        CUDA_CHECK(cudaStreamSynchronize(state.stream()));
        cudaFree(dCorr);
    } else {
        std::fill(state.distances.begin(), state.distances.end(), ZERO);
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
    int nonZeroKij = static_cast<int>(state.nonZeroKij);  // counted on the GPU
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
    bool valid = !validate || validateResults(state);
    releaseSimulation(state);
    if (!valid) {
        return 1;
    }

    return 0;
}

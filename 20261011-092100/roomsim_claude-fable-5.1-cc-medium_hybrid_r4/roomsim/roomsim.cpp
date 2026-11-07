/**
 * Room Response Simulation Benchmark — hybrid MPI + OpenMP + CUDA implementation
 *
 * This is a parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy
 * ------------------------
 *  * MPI     : the rows of the N x N form-factor / delay matrices are distributed
 *              block-wise across ranks.  Every rank owns one GPU (selected by the
 *              node-local rank).  Radiosity rows are all-gathered once per timestep
 *              and the final distances are gathered on rank 0.
 *  * CUDA    : form factors (one lane per visibility ray, 16 lanes per triangle
 *              pair, octree traversal with an explicit stack), generation of the
 *              random stream, the wave propagation (one thread per receiving
 *              triangle, sequential over emitters so that the floating-point
 *              summation order is identical to the sequential code) and the
 *              cross-correlation (one block per triangle).
 *  * OpenMP  : host-side preprocessing (visibility culling, time delays) and the
 *              generator states needed to reproduce the exact std::mt19937 random
 *              stream of the sequential code in parallel.  Those states are
 *              obtained by Mersenne-Twister jump-ahead (characteristic polynomial
 *              from Berlekamp-Massey, x^k mod p(x) by repeated squaring), so every
 *              GPU block can start at an arbitrary stream offset without replaying
 *              the sequence.
 *
 * All floating-point operations are performed in the same order and with the
 * same fused multiply-add pattern as the GCC-compiled sequential reference, so
 * the results are bit-identical.
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <climits>
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

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t cuda_err__ = (call);                                              \
        if (cuda_err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(cuda_err__), \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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

// Random numbers consumed per (non-culled) triangle pair: NUM_RAYS rays x 2 points x 2 coords
constexpr int RANDS_PER_PAIR = NUM_RAYS * 4;
// Marker for pairs that consume no random numbers (i == j or culled by normal test)
constexpr uint32_t PAIR_SKIPPED = 0xFFFFFFFFu;
// Maximum number of random floats per GPU tile (512 MiB of float data)
constexpr uint64_t MAX_TILE_RANDS = 128ull * 1024 * 1024;
// The random stream is generated on the GPU in chunks of 2^RAND_CHUNK_LOG numbers,
// each chunk starting from a generator state obtained by jump-ahead on the host
constexpr int RAND_CHUNK_LOG = 22;
constexpr uint64_t RAND_CHUNK = 1ull << RAND_CHUNK_LOG;

// std::min / std::max semantics (first argument returned on ties / NaN)
HD inline val_t vmin(val_t a, val_t b) { return (b < a) ? b : a; }
HD inline val_t vmax(val_t a, val_t b) { return (a < b) ? b : a; }

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

    Triangle() = default;
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

// Flattened (GPU friendly) representation of the octree.  Nodes are stored in
// pre-order; leaves reference a contiguous range of the leafTris array.
struct FlatOctree {
    std::vector<float4> center;      // xyz = node center
    std::vector<float4> halfExtent;  // xyz = node half extent
    std::vector<int> child;          // 8 entries per node, -1 if absent
    std::vector<int2> leaf;          // x = first index in leafTris, y = count (0 for inner nodes)
    std::vector<int> leafTris;

    int addNode(const Octree& node) {
        int id = static_cast<int>(center.size());
        center.push_back(make_float4(node.center.x, node.center.y, node.center.z, 0.0f));
        halfExtent.push_back(make_float4(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z, 0.0f));
        child.insert(child.end(), 8, -1);
        int2 lf = make_int2(static_cast<int>(leafTris.size()), static_cast<int>(node.triangleIndices.size()));
        for (size_t idx : node.triangleIndices) leafTris.push_back(static_cast<int>(idx));
        leaf.push_back(lf);
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                int cid = addNode(*node.children[c]);
                child[static_cast<size_t>(id) * 8 + c] = cid;
            }
        }
        return id;
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
// Random Number Generation: std::mt19937 compatible generator with jump-ahead
// ============================================================================
//
// The sequential reference implementation draws all random numbers from a single
// std::mt19937(42) through std::uniform_real_distribution<float>(0, 1).  To keep
// the semantics identical while generating in parallel, this module reproduces
// the exact stream (libstdc++ seeding, generation, tempering and float
// conversion) and provides an O(1)-per-number jump to arbitrary offsets via the
// linear-algebraic structure of the Mersenne Twister:
//
//   * The state transition "advance by one word" A is linear over GF(2) and its
//     characteristic polynomial p(x) (degree 19937) is recovered with the
//     Berlekamp-Massey algorithm from the output sequence.
//   * Jumping by J words is the evaluation of g(x) = x^J mod p(x) at A applied to
//     the state (Horner scheme).  x^(2^b) mod p are tabulated by squaring, so a
//     jump costs at most one Horner evaluation per set bit of J.

namespace mt {

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0dfu;
constexpr uint32_t UPPER_MASK = 0x80000000u;
constexpr uint32_t LOWER_MASK = 0x7fffffffu;
constexpr int DEG = 19937;  // degree of the characteristic polynomial

struct State {
    uint32_t mt[MT_N];
    int p;
};

inline void seed(State& s, uint32_t sd) {
    s.mt[0] = sd;
    for (int i = 1; i < MT_N; ++i) {
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    s.p = MT_N;
}

// Advance the full block of MT_N words (standard MT19937 block regeneration)
inline void genRand(uint32_t* mt) {
    int k = 0;
    for (; k < MT_N - MT_M; ++k) {
        uint32_t y = (mt[k] & UPPER_MASK) | (mt[k + 1] & LOWER_MASK);
        mt[k] = mt[k + MT_M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }
    for (; k < MT_N - 1; ++k) {
        uint32_t y = (mt[k] & UPPER_MASK) | (mt[k + 1] & LOWER_MASK);
        mt[k] = mt[k + (MT_M - MT_N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }
    uint32_t y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
}

inline uint32_t temper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

inline uint32_t next(State& s) {
    if (s.p >= MT_N) {
        genRand(s.mt);
        s.p = 0;
    }
    return temper(s.mt[s.p++]);
}

// std::uniform_real_distribution<float>(0,1) applied to a 32-bit engine output
// (libstdc++ generate_canonical<float, 24>)
inline float toFloat(uint32_t y) {
    float r = static_cast<float>(y) / 4294967296.0f;
    if (r >= 1.0f) r = nextafterf(1.0f, 0.0f);
    return r;
}

inline float nextFloat(State& s) { return toFloat(next(s)); }

// ---------------- GF(2) polynomial arithmetic (bit i = coefficient of x^i) ----------------

using Poly = std::vector<uint64_t>;

inline bool polyBit(const Poly& a, size_t i) { return (a[i >> 6] >> (i & 63)) & 1u; }

// dst ^= src << shift
inline void xorShifted(Poly& dst, const Poly& src, size_t shift) {
    size_t ws = shift >> 6;
    unsigned bs = static_cast<unsigned>(shift & 63);
    size_t n = src.size();
    if (bs == 0) {
        for (size_t w = 0; w < n; ++w) dst[w + ws] ^= src[w];
    } else {
        for (size_t w = 0; w < n; ++w) {
            dst[w + ws] ^= src[w] << bs;
            dst[w + ws + 1] ^= src[w] >> (64 - bs);
        }
    }
}

struct JumpTable {
    Poly charPoly;                 // p(x), DEG + 1 bits
    std::vector<Poly> pow2;        // pow2[b] = x^(2^b) mod p(x), DEG bits each
    std::vector<std::vector<int>> pow2Bits;  // indices of the set bits of pow2[b]
    uint32_t base[MT_N];           // window (x_1, ..., x_624) of the seeded generator
    int bits = 0;
};

// Reduce a polynomial of degree < 2*DEG modulo p (p has degree DEG, top bit set)
inline void reduceMod(Poly& r, const Poly& p) {
    for (size_t d = 2 * DEG - 2; d >= static_cast<size_t>(DEG); --d) {
        if (polyBit(r, d)) xorShifted(r, p, d - DEG);
    }
}

inline Poly squareMod(const Poly& a, const Poly& p) {
    const size_t words = (DEG + 63) / 64;
    Poly r(2 * words + 2, 0);
    for (size_t w = 0; w < words; ++w) {
        uint64_t v = a[w];
        for (int half = 0; half < 2; ++half) {
            uint64_t x = (half == 0) ? (v & 0xFFFFFFFFull) : (v >> 32);
            x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
            x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
            x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
            x = (x | (x << 2)) & 0x3333333333333333ull;
            x = (x | (x << 1)) & 0x5555555555555555ull;
            r[2 * w + half] ^= x;
        }
    }
    reduceMod(r, p);
    r.resize(words);
    return r;
}

// Berlekamp-Massey over GF(2): returns the characteristic polynomial of the
// minimal linear recurrence of the bit sequence s (nb bits).
inline Poly berlekampMassey(const std::vector<uint64_t>& s, size_t nb, int& degreeOut) {
    const size_t words = nb / 64 + 2;
    // Reversed sequence: R bit k = s bit (nb - 1 - k); s_{n-i} == R bit (nb-1-n+i)
    std::vector<uint64_t> R(words + 1, 0);
    for (size_t k = 0; k < nb; ++k) {
        if ((s[k >> 6] >> (k & 63)) & 1u) {
            size_t rk = nb - 1 - k;
            R[rk >> 6] |= 1ull << (rk & 63);
        }
    }
    Poly C(words, 0), B(words, 0), T(words, 0);
    C[0] = 1;
    B[0] = 1;
    int L = 0;
    size_t m = 1;
    for (size_t n = 0; n < nb; ++n) {
        // discrepancy d = sum_{i=0..L} C_i s_{n-i}
        size_t base = nb - 1 - n;
        size_t bw = base >> 6;
        unsigned bs = static_cast<unsigned>(base & 63);
        size_t nw = static_cast<size_t>(L) / 64 + 1;
        uint64_t acc = 0;
        for (size_t w = 0; w < nw; ++w) {
            uint64_t win = R[bw + w] >> bs;
            if (bs) win |= R[bw + w + 1] << (64 - bs);
            acc ^= win & C[w];
        }
        int d = __builtin_parityll(acc);
        if (d) {
            T = C;
            xorShifted(C, B, m);
            if (2 * static_cast<size_t>(L) <= n) {
                L = static_cast<int>(n + 1) - L;
                B = T;
                m = 1;
            } else {
                ++m;
            }
        } else {
            ++m;
        }
    }
    degreeOut = L;
    // characteristic polynomial p_i = C_{L-i}
    Poly p(static_cast<size_t>(L) / 64 + 1, 0);
    for (int i = 0; i <= L; ++i) {
        if (polyBit(C, static_cast<size_t>(L - i))) p[i >> 6] |= 1ull << (i & 63);
    }
    return p;
}

// Build the jump table able to reach word offsets below 2^bits
inline void buildJumpTable(JumpTable& jt, uint32_t sd, int bits) {
    // Window (x_1 ... x_624): the seeded state advanced by one word
    State s;
    seed(s, sd);
    {
        uint32_t y = (s.mt[0] & UPPER_MASK) | (s.mt[1] & LOWER_MASK);
        uint32_t x624 = s.mt[MT_M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
        for (int k = 0; k < MT_N - 1; ++k) jt.base[k] = s.mt[k + 1];
        jt.base[MT_N - 1] = x624;
    }

    // Output bit sequence (bit 0 of the raw state words x_624, x_625, ...)
    const size_t nb = 2 * DEG + 256;
    std::vector<uint64_t> bitsSeq(nb / 64 + 2, 0);
    {
        State g;
        seed(g, sd);
        size_t k = 0;
        while (k < nb) {
            genRand(g.mt);
            for (int w = 0; w < MT_N && k < nb; ++w, ++k) {
                if (g.mt[w] & 1u) bitsSeq[k >> 6] |= 1ull << (k & 63);
            }
        }
    }
    int degree = 0;
    jt.charPoly = berlekampMassey(bitsSeq, nb, degree);
    if (degree != DEG) {
        fprintf(stderr, "MT19937 characteristic polynomial has unexpected degree %d\n", degree);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Sanity check: the polynomial must annihilate the sequence
    for (size_t k = 0; k < 64; ++k) {
        int acc = 0;
        for (int i = 0; i <= DEG; ++i) {
            if (polyBit(jt.charPoly, static_cast<size_t>(i)) &&
                ((bitsSeq[(k + i) >> 6] >> ((k + i) & 63)) & 1u)) acc ^= 1;
        }
        if (acc) {
            fprintf(stderr, "MT19937 characteristic polynomial verification failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    const size_t words = (DEG + 63) / 64;
    jt.bits = bits;
    jt.pow2.resize(static_cast<size_t>(bits));
    Poly x(words, 0);
    x[0] = 2;  // x^1
    jt.pow2[0] = x;
    for (int b = 1; b < bits; ++b) jt.pow2[b] = squareMod(jt.pow2[b - 1], jt.charPoly);
    jt.pow2Bits.resize(static_cast<size_t>(bits));
    for (int b = 0; b < bits; ++b) {
        for (int i = 0; i < DEG; ++i) {
            if (polyBit(jt.pow2[b], static_cast<size_t>(i))) jt.pow2Bits[b].push_back(i);
        }
    }
}

// r <- g(A) v, where A advances the 624-word window by one word.
// Since A^i v is the window (x_i, ..., x_{i+623}) of the sequence generated from
// v, the result is the convolution r[k] = XOR_{g_i = 1} x_{i+k}: the sequence is
// extended once and the set bits of g select which shifted windows to combine.
inline void applyJumpPoly(const std::vector<int>& gBits, const uint32_t* v, uint32_t* out) {
    constexpr int SEQ_LEN = DEG + MT_N + 64;
    alignas(32) uint32_t x[SEQ_LEN];
    for (int k = 0; k < MT_N; ++k) x[k] = v[k];
    for (int i = MT_N; i < SEQ_LEN; ++i) {
        uint32_t y = (x[i - MT_N] & UPPER_MASK) | (x[i - MT_N + 1] & LOWER_MASK);
        x[i] = x[i - MT_N + MT_M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }
    constexpr int BLOCK = 64;
    constexpr int R_LEN = ((MT_N + BLOCK - 1) / BLOCK) * BLOCK;
    alignas(32) uint32_t r[R_LEN];
    const int* bits = gBits.data();
    const int nbits = static_cast<int>(gBits.size());
    for (int kb = 0; kb < MT_N; kb += BLOCK) {
#if defined(__AVX2__) && defined(__GNUC__)
        // 8 x 256-bit accumulators (GCC vector extensions: no intrinsics header needed)
        typedef uint32_t v8u __attribute__((vector_size(32)));
        v8u a0 = {}, a1 = {}, a2 = {}, a3 = {}, a4 = {}, a5 = {}, a6 = {}, a7 = {};
        for (int n = 0; n < nbits; ++n) {
            const uint32_t* p = x + bits[n] + kb;
            v8u t0, t1, t2, t3, t4, t5, t6, t7;
            memcpy(&t0, p, 32); memcpy(&t1, p + 8, 32); memcpy(&t2, p + 16, 32); memcpy(&t3, p + 24, 32);
            memcpy(&t4, p + 32, 32); memcpy(&t5, p + 40, 32); memcpy(&t6, p + 48, 32); memcpy(&t7, p + 56, 32);
            a0 ^= t0; a1 ^= t1; a2 ^= t2; a3 ^= t3; a4 ^= t4; a5 ^= t5; a6 ^= t6; a7 ^= t7;
        }
        memcpy(r + kb, &a0, 32); memcpy(r + kb + 8, &a1, 32); memcpy(r + kb + 16, &a2, 32); memcpy(r + kb + 24, &a3, 32);
        memcpy(r + kb + 32, &a4, 32); memcpy(r + kb + 40, &a5, 32); memcpy(r + kb + 48, &a6, 32); memcpy(r + kb + 56, &a7, 32);
#else
        uint32_t acc[BLOCK];
        for (int k = 0; k < BLOCK; ++k) acc[k] = 0;
        for (int n = 0; n < nbits; ++n) {
            const uint32_t* p = x + bits[n] + kb;
            for (int k = 0; k < BLOCK; ++k) acc[k] ^= p[k];
        }
        for (int k = 0; k < BLOCK; ++k) r[kb + k] = acc[k];
#endif
    }
    for (int k = 0; k < MT_N; ++k) out[k] = r[k];
}

// Position the generator so that the next output is output number K of
// std::mt19937(seed) (i.e. equivalent to seeding followed by discard(K)).
inline void jumpTo(const JumpTable& jt, uint64_t K, State& s) {
    uint64_t target = K + static_cast<uint64_t>(MT_N - 1);  // window index 624+K reached from window 1
    uint32_t cur[MT_N], nxt[MT_N];
    for (int k = 0; k < MT_N; ++k) cur[k] = jt.base[k];
    for (int b = 0; b < jt.bits; ++b) {
        if ((target >> b) & 1ull) {
            applyJumpPoly(jt.pow2Bits[b], cur, nxt);
            for (int k = 0; k < MT_N; ++k) cur[k] = nxt[k];
        }
    }
    if (target >> jt.bits) {
        fprintf(stderr, "MT19937 jump target exceeds jump table range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int k = 0; k < MT_N; ++k) s.mt[k] = cur[k];
    s.p = 0;
}

// Verify the generator and the jump machinery against the standard library
inline void selfTest(const JumpTable& jt) {
    const uint64_t offsets[3] = {0, 100003, 1u << 20};
    for (uint64_t K : offsets) {
        std::mt19937 ref(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        ref.discard(K);
        State s;
        jumpTo(jt, K, s);
        for (int n = 0; n < 4000; ++n) {
            float a = dist(ref);
            float b = nextFloat(s);
            if (a != b) {
                fprintf(stderr, "MT19937 self test failed at offset %llu + %d\n",
                        static_cast<unsigned long long>(K), n);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
}

}  // namespace mt

// ----------------------------------------------------------------------------
// Device floating-point helpers
//
// The sequential reference is compiled by GCC with FMA contraction enabled.  To
// reproduce its results bit-for-bit the device code spells out the same fusion
// pattern explicitly (GCC fuses the first product of a sum/difference and rounds
// the second one, chaining left to right), so that no implicit contraction by
// the CUDA compiler can change the rounding.
// ----------------------------------------------------------------------------

__device__ __forceinline__ val_t dotDev(const Vec3& a, const Vec3& b) {
    return fmaf(a.z, b.z, fmaf(a.x, b.x, __fmul_rn(a.y, b.y)));
}

__device__ __forceinline__ val_t squaredNormDev(const Vec3& v) {
    return fmaf(v.z, v.z, fmaf(v.x, v.x, __fmul_rn(v.y, v.y)));
}

__device__ __forceinline__ Vec3 crossDev(const Vec3& a, const Vec3& b) {
    return {fmaf(a.y, b.z, -__fmul_rn(a.z, b.y)),
            fmaf(a.z, b.x, -__fmul_rn(a.x, b.z)),
            fmaf(a.x, b.y, -__fmul_rn(a.y, b.x))};
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ Vec3 randomPointInTriangleDev(const Vec3& a, const Vec3& b, const Vec3& c,
                                                         val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return {fmaf(ac.x, v, fmaf(ab.x, u, a.x)),
            fmaf(ac.y, v, fmaf(ab.y, u, a.y)),
            fmaf(ac.z, v, fmaf(ab.z, u, a.z))};
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ __forceinline__ val_t rayTriangleIntersectDev(const Vec3& orig, const Vec3& dir,
                                                         const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = crossDev(dir, e2);
    val_t det = dotDev(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = __fdiv_rn(1.0f, det);
    Vec3 tvec = orig - v0;
    val_t u = __fmul_rn(dotDev(tvec, pvec), invDet);
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = crossDev(tvec, e1);
    val_t v = __fmul_rn(dotDev(dir, qvec), invDet);
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return __fmul_rn(dotDev(e2, qvec), invDet);
}

// ============================================================================
// Form Factor helpers
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = sqrtf(squaredNormDev(v));
    if (vNorm <= EPSILON) return ZERO;
    return vmax(ZERO, __fdiv_rn(dotDev(v, normal), vNorm));
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

inline int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// GPU kernels
// ============================================================================

struct GpuGeometry {
    const float4* tris;        // 4 float4 per triangle: a, b, c, normal
    const float4* nodeCenter;
    const float4* nodeHalf;
    const int* nodeChild;      // 8 per node
    const int2* nodeLeaf;      // x = first leaf triangle, y = count
    const int* leafTris;
};

__device__ __forceinline__ Vec3 toVec3(const float4& f) { return Vec3(f.x, f.y, f.z); }

// Check if a ray (segment p1-p2) intersects a node's bounding box
__device__ __forceinline__ bool rayIntersectsBoxDev(const float4& c4, const float4& h4,
                                                    const Vec3& p1, const Vec3& p2) {
    Vec3 center = toVec3(c4);
    Vec3 halfExtent = toVec3(h4);
    Vec3 diff = p2 - p1;
    Vec3 d = {__fmul_rn(diff.x, 0.5f), __fmul_rn(diff.y, 0.5f), __fmul_rn(diff.z, 0.5f)};
    Vec3 c = {__fadd_rn(p1.x, d.x) - center.x, __fadd_rn(p1.y, d.y) - center.y, __fadd_rn(p1.z, d.z) - center.z};
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(fmaf(d.y, c.z, -__fmul_rn(d.z, c.y))) >
        fmaf(halfExtent.y, ad.z, __fmul_rn(halfExtent.z, ad.y)) + EPSILON) return false;
    if (fabsf(fmaf(d.z, c.x, -__fmul_rn(d.x, c.z))) >
        fmaf(halfExtent.z, ad.x, __fmul_rn(halfExtent.x, ad.z)) + EPSILON) return false;
    if (fabsf(fmaf(d.x, c.y, -__fmul_rn(d.y, c.x))) >
        fmaf(halfExtent.x, ad.y, __fmul_rn(halfExtent.y, ad.x)) + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
// (octree traversal with an explicit stack; same set of candidate triangles as
// the recursive traversal of the sequential code)
__device__ bool isRayBlockedDev(const GpuGeometry& g, const Vec3& from, const Vec3& to,
                                const Vec3& dirNorm, val_t rayLen, int srcTriIdx, int dstTriIdx) {
    const val_t maxDist = rayLen - EPSILON;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int n = stack[--sp];
        int2 lf = __ldg(&g.nodeLeaf[n]);
        if (lf.y > 0) {
            for (int k = 0; k < lf.y; ++k) {
                int idx = __ldg(&g.leafTris[lf.x + k]);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const float4* t = g.tris + 4 * idx;
                val_t dist = rayTriangleIntersectDev(from, dirNorm, toVec3(__ldg(t)), toVec3(__ldg(t + 1)),
                                                  toVec3(__ldg(t + 2)));
                if (dist > EPSILON && dist < maxDist) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = __ldg(&g.nodeChild[n * 8 + c]);
                if (ch >= 0 && rayIntersectsBoxDev(__ldg(&g.nodeCenter[ch]), __ldg(&g.nodeHalf[ch]), from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

// Per-pair offsets into a tile's random stream: one block per row of the tile,
// exclusive scan of the visibility mask (pairs that consume random numbers).
__global__ void __launch_bounds__(1024)
pairOffsetsKernel(int N, int wordsPerRow, int rowStart, int tileRow0, unsigned long long tileRand0,
                  const uint32_t* __restrict__ pairMask, const unsigned long long* __restrict__ rowBase,
                  uint32_t* __restrict__ offsets) {
    __shared__ unsigned warpSums[32];
    const int il = tileRow0 + blockIdx.x;
    const int i = rowStart + il;
    const uint32_t* mask = pairMask + static_cast<size_t>(i) * wordsPerRow;
    uint32_t* out = offsets + static_cast<size_t>(blockIdx.x) * N;
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = blockDim.x >> 5;
    unsigned long long running = rowBase[i] - tileRand0;
    for (int j0 = 0; j0 < N; j0 += blockDim.x) {
        const int j = j0 + static_cast<int>(threadIdx.x);
        const unsigned v = (j < N) ? ((mask[j >> 5] >> (j & 31)) & 1u) : 0u;
        unsigned incl = v;
        for (unsigned o = 1; o < 32; o <<= 1) {
            unsigned n = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += n;
        }
        if (lane == 31) warpSums[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            unsigned ws = (lane < nwarps) ? warpSums[lane] : 0u;
            for (unsigned o = 1; o < 32; o <<= 1) {
                unsigned n = __shfl_up_sync(0xffffffffu, ws, o);
                if (lane >= o) ws += n;
            }
            warpSums[lane] = ws;
        }
        __syncthreads();
        const unsigned ex = ((warp > 0) ? warpSums[warp - 1] : 0u) + incl - v;
        const unsigned total = warpSums[nwarps - 1];
        if (j < N) {
            out[j] = v ? static_cast<uint32_t>(running + static_cast<unsigned long long>(ex) * RANDS_PER_PAIR)
                       : PAIR_SKIPPED;
        }
        running += static_cast<unsigned long long>(total) * RANDS_PER_PAIR;
        __syncthreads();
    }
}

// Mersenne-Twister stream generation on the GPU.  One block per chunk of
// RAND_CHUNK numbers; the block regenerates the 624-word state cooperatively
// (three dependency phases, read-before-write) and emits the tempered outputs
// converted exactly like std::uniform_real_distribution<float>(0, 1).
constexpr int MT_GEN_THREADS = 640;  // >= 624, multiple of the warp size

__device__ __forceinline__ float mtToFloatDev(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    float r = __fdiv_rn(__uint2float_rn(y), 4294967296.0f);
    if (r >= 1.0f) r = 0.99999994f;  // nextafter(1, 0)
    return r;
}

__device__ __forceinline__ void mtPhase(uint32_t* mt, int k, int lo, int hi) {
    uint32_t a = 0, b = 0, c = 0;
    const bool active = (k >= lo && k < hi);
    if (active) {
        int k1 = k + 1; if (k1 >= mt::MT_N) k1 -= mt::MT_N;
        int kM = k + mt::MT_M; if (kM >= mt::MT_N) kM -= mt::MT_N;
        a = mt[k]; b = mt[k1]; c = mt[kM];
    }
    __syncthreads();
    if (active) {
        uint32_t y = (a & mt::UPPER_MASK) | (b & mt::LOWER_MASK);
        mt[k] = c ^ (y >> 1) ^ ((y & 1u) ? mt::MATRIX_A : 0u);
    }
    __syncthreads();
}

__global__ void __launch_bounds__(MT_GEN_THREADS)
mtGenerateKernel(const uint32_t* __restrict__ chunkStates, unsigned long long firstChunkStart,
                 unsigned long long tileStart, unsigned long long tileEnd, float* __restrict__ out) {
    __shared__ uint32_t mt[mt::MT_N];
    const int k = threadIdx.x;
    const uint32_t* st = chunkStates + static_cast<size_t>(blockIdx.x) * mt::MT_N;
    if (k < mt::MT_N) mt[k] = st[k];
    __syncthreads();

    const unsigned long long chunkStart = firstChunkStart + static_cast<unsigned long long>(blockIdx.x) * RAND_CHUNK;
    const unsigned long long chunkEnd = chunkStart + RAND_CHUNK;
    const unsigned long long stop = (chunkEnd < tileEnd) ? chunkEnd : tileEnd;
    for (unsigned long long g0 = chunkStart; g0 < stop; g0 += mt::MT_N) {
        if (k < mt::MT_N) {
            unsigned long long g = g0 + static_cast<unsigned long long>(k);
            if (g >= tileStart && g < stop) out[g - tileStart] = mtToFloatDev(mt[k]);
        }
        if (g0 + mt::MT_N >= stop) break;
        mtPhase(mt, k, 0, mt::MT_N - mt::MT_M);
        mtPhase(mt, k, mt::MT_N - mt::MT_M, 2 * (mt::MT_N - mt::MT_M));
        mtPhase(mt, k, 2 * (mt::MT_N - mt::MT_M), mt::MT_N);
    }
}

// Form factor kernel: 16 consecutive lanes handle the 16 rays of one (i, j) pair.
// The per-ray contributions are summed in ray order to reproduce the sequential
// floating-point result exactly.
__global__ void __launch_bounds__(256)
formFactorKernel(int N, int rowStart, int tileRow0, long long numPairs,
                 const uint32_t* __restrict__ offsets, const float* __restrict__ rands,
                 GpuGeometry g, float* __restrict__ kij) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long pair = tid >> 4;
    const int r = static_cast<int>(tid & 15);
    const bool valid = pair < numPairs;

    int il = 0, j = 0;
    uint32_t off = PAIR_SKIPPED;
    if (valid) {
        il = tileRow0 + static_cast<int>(pair / N);
        j = static_cast<int>(pair - static_cast<long long>(pair / N) * N);
        off = offsets[pair];
    }

    val_t contrib = ZERO;
    if (off != PAIR_SKIPPED) {
        const int i = rowStart + il;
        const float4* ti = g.tris + 4 * i;
        const float4* tj = g.tris + 4 * j;
        const Vec3 aI = toVec3(__ldg(ti)), bI = toVec3(__ldg(ti + 1)), cI = toVec3(__ldg(ti + 2));
        const Vec3 nI = toVec3(__ldg(ti + 3));
        const Vec3 aJ = toVec3(__ldg(tj)), bJ = toVec3(__ldg(tj + 1)), cJ = toVec3(__ldg(tj + 2));
        const Vec3 nJ = toVec3(__ldg(tj + 3));

        const float4 rs = __ldg(reinterpret_cast<const float4*>(rands + off + 4 * r));
        const Vec3 pI = randomPointInTriangleDev(aI, bI, cI, rs.x, rs.y);
        const Vec3 pJ = randomPointInTriangleDev(aJ, bJ, cJ, rs.z, rs.w);

        // isRayBlocked
        Vec3 dir = pJ - pI;
        val_t rayLen = sqrtf(squaredNormDev(dir));
        bool blocked = true;
        if (!(rayLen < EPSILON)) {
            Vec3 dirNorm = {__fdiv_rn(dir.x, rayLen), __fdiv_rn(dir.y, rayLen), __fdiv_rn(dir.z, rayLen)};
            blocked = isRayBlockedDev(g, pI, pJ, dirNorm, rayLen, i, j);
        }

        if (!blocked) {
            Vec3 v = pJ - pI;
            val_t distSqr = squaredNormDev(v);
            if (!(distSqr < EPSILON)) {
                val_t cosPhiI = cosPhiDev(v, nI);
                val_t cosPhiJ = cosPhiDev(-v, nJ);
                if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                    contrib = __fdiv_rn(__fmul_rn(cosPhiI, cosPhiJ), __fmul_rn(PI, distSqr));
                }
            }
        }
    }

    // Sequential (in ray order) sum over the 16 lanes of this pair
    const int half = threadIdx.x & 16;
    val_t sum = ZERO;
#pragma unroll
    for (int rr = 0; rr < NUM_RAYS; ++rr) {
        val_t c = __shfl_sync(0xffffffffu, contrib, half + rr);
        sum = __fadd_rn(sum, c);
    }

    if (valid && r == 0) {
        kij[static_cast<size_t>(il) * N + j] = (off == PAIR_SKIPPED) ? ZERO : __fmul_rn(sum, INV_NUM_RAYS);
    }
}

// Pack form factor weights and delays into a transposed [j][i_local] layout:
// x = min(kij * area_j, 1) as float bits, y = tau (INT_MAX for pairs that never contribute)
__global__ void packWeightsKernel(int N, int L, int rowStart, const float* __restrict__ kij,
                                  const int* __restrict__ tau, const float* __restrict__ areas,
                                  int2* __restrict__ wtauT) {
    __shared__ int2 tile[32][33];
    const int j0 = blockIdx.x * 32;
    const int i0 = blockIdx.y * 32;
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        int il = i0 + k;
        int j = j0 + threadIdx.x;
        int2 o = make_int2(0, INT_MAX);
        if (il < L && j < N) {
            size_t idx = static_cast<size_t>(il) * N + j;
            float k_ij = kij[idx];
            if (j != rowStart + il && !(k_ij <= ZERO)) {
                o.x = __float_as_int(vmin(__fmul_rn(k_ij, areas[j]), ONE));
                o.y = tau[idx];
            }
        }
        tile[k][threadIdx.x] = o;
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        int j = j0 + k;
        int il = i0 + threadIdx.x;
        if (il < L && j < N) wtauT[static_cast<size_t>(j) * L + il] = tile[threadIdx.x][k];
    }
}

// One simulation timestep: one thread per local receiving triangle, sequential
// accumulation over all emitters (identical summation order to the reference).
__global__ void simulationStepKernel(int N, int L, int rowStart, int t, int timeOff, int sourceIndex,
                                     const float* __restrict__ rho, const int2* __restrict__ wtauT,
                                     float* __restrict__ radB) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= L) return;
    const int i = rowStart + il;
    const int2* col = wtauT + il;

    val_t sumB = ZERO;
#pragma unroll 8
    for (int j = 0; j < N; ++j) {
        int2 wt = __ldg(col + static_cast<size_t>(j) * L);
        if (t < wt.y) continue;  // wave not yet arrived (or pair never contributes)
        // rows < t of radB are complete before this kernel runs; row t is only written
        val_t radJ = __ldg(radB + static_cast<size_t>(t - wt.y) * N + j);
        if (radJ <= ZERO) continue;
        sumB = __fadd_rn(sumB, __fmul_rn(__int_as_float(wt.x), radJ));
    }
    const val_t radE = (i == sourceIndex && t < timeOff) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * N + i] = fmaf(rho[i], sumB, radE);
}

struct CorrCandidate {
    val_t corr;
    int t;
};

__device__ __forceinline__ CorrCandidate betterCandidate(const CorrCandidate& a, const CorrCandidate& b) {
    if (a.corr > b.corr) return a;
    if (b.corr > a.corr) return b;
    return (a.t <= b.t) ? a : b;
}

// Cross-correlation distance estimation: one block per local triangle.
template <bool STAGED>
__global__ void __launch_bounds__(256)
distanceKernel(int N, int T, int rowStart, int sourceIndex, const float* __restrict__ radB,
               float* __restrict__ distances) {
    extern __shared__ float shmem[];
    __shared__ CorrCandidate red[256];

    const int il = blockIdx.x;
    const int i = rowStart + il;
    const float* colB;
    const float* colS;
    int strideB, strideS;
    if (STAGED) {
        float* sB = shmem;
        float* sS = shmem + T;
        for (int tt = threadIdx.x; tt < T; tt += blockDim.x) {
            sB[tt] = radB[static_cast<size_t>(tt) * N + i];
            sS[tt] = radB[static_cast<size_t>(tt) * N + sourceIndex];
        }
        __syncthreads();
        colB = sB; colS = sS; strideB = 1; strideS = 1;
    } else {
        colB = radB + i; colS = radB + sourceIndex; strideB = N; strideS = N;
    }

    CorrCandidate best{ZERO, 0};
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = colB[static_cast<size_t>(tt) * strideB];
            val_t pS = colS[static_cast<size_t>(tt - t) * strideS];
            sum = __fadd_rn(sum, __fmul_rn(pS, pB));
        }
        if (sum > best.corr) {
            best.corr = sum;
            best.t = t;
        }
    }
    red[threadIdx.x] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = betterCandidate(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[il] = WAVE_SPEED * static_cast<val_t>(red[0].t);
    }
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
    std::vector<val_t> kij;         // Form factors (local rows x N, row-major)
    std::vector<int> tau;           // Time delays (local rows x N, row-major)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (complete on rank 0)

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;          // GPU representation of the octree

    size_t sourceIndex;

    // MPI decomposition (rows of the N x N matrices)
    int rank = 0, size = 1;
    size_t rowStart = 0;            // first global row owned by this rank
    size_t localRows = 0;           // number of rows owned by this rank
    std::vector<int> rowCounts;     // rows per rank
    std::vector<int> rowDispls;     // first row per rank

    // Device data
    float4* d_tris = nullptr;
    float4* d_nodeCenter = nullptr;
    float4* d_nodeHalf = nullptr;
    int* d_nodeChild = nullptr;
    int2* d_nodeLeaf = nullptr;
    int* d_leafTris = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_kij = nullptr;         // local rows x N
    int* d_tau = nullptr;           // local rows x N
    int2* d_wtauT = nullptr;        // N x local rows (transposed)
    float* d_radB = nullptr;        // T x N
    float* d_dist = nullptr;        // local rows

    size_t idx2d(size_t il, size_t j) const { return il * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
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

    if (state.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.flatOctree.addNode(state.octree);

    // Row decomposition over MPI ranks
    const size_t N = state.numTriangles;
    state.rowCounts.resize(state.size);
    state.rowDispls.resize(state.size);
    for (int r = 0; r < state.size; ++r) {
        size_t base = N / state.size, rem = N % state.size;
        size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        size_t start = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
        state.rowCounts[r] = static_cast<int>(cnt);
        state.rowDispls[r] = static_cast<int>(start);
    }
    state.rowStart = static_cast<size_t>(state.rowDispls[state.rank]);
    state.localRows = static_cast<size_t>(state.rowCounts[state.rank]);
    const size_t L = state.localRows;

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    // Initialize matrices (local rows only) and vectors
    state.kij.assign(L * N, ZERO);
    state.tau.assign(L * N, 0);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Upload geometry to the GPU
    std::vector<float4> tris(4 * N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        tris[4 * i + 0] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        tris[4 * i + 1] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        tris[4 * i + 2] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        tris[4 * i + 3] = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
    }
    const FlatOctree& fo = state.flatOctree;
    CUDA_CHECK(cudaMalloc(&state.d_tris, tris.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeCenter, fo.center.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeHalf, fo.halfExtent.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeChild, fo.child.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeLeaf, fo.leaf.size() * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&state.d_leafTris, std::max<size_t>(1, fo.leafTris.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_tris, tris.data(), tris.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeCenter, fo.center.data(), fo.center.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeHalf, fo.halfExtent.data(), fo.halfExtent.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeChild, fo.child.data(), fo.child.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeLeaf, fo.leaf.data(), fo.leaf.size() * sizeof(int2), cudaMemcpyHostToDevice));
    if (!fo.leafTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_leafTris, fo.leafTris.data(), fo.leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_kij, std::max<size_t>(1, L * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, std::max<size_t>(1, L * N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_wtauT, std::max<size_t>(1, L * N) * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_dist, std::max<size_t>(1, L) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_kij, 0, std::max<size_t>(1, L * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, std::max<size_t>(1, timesteps * N) * sizeof(float)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t L = state.localRows;
    const int R0 = static_cast<int>(state.rowStart);

    // Visibility culling mask (same test as the sequential code) for all rows:
    // determines how many random numbers each row of the global stream consumes.
    std::vector<uint64_t> rowBase(N + 1, 0);
    const size_t wordsPerRow = (N + 31) / 32;
    std::vector<uint32_t> pairMask(N * wordsPerRow, 0u);
    {
        std::vector<uint32_t> cnt(N, 0);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            const Triangle& triI = state.triangles[i];
            uint32_t* mask = pairMask.data() + i * wordsPerRow;
            uint32_t c = 0;
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                if (triI.normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                mask[j >> 5] |= 1u << (j & 31);
                ++c;
            }
            cnt[i] = c;
        }
        for (size_t i = 0; i < N; ++i) rowBase[i + 1] = rowBase[i] + static_cast<uint64_t>(cnt[i]) * RANDS_PER_PAIR;
    }
    const uint64_t totalRands = rowBase[N];

    // Jump table for the random stream (shared by all threads of this rank)
    mt::JumpTable jt;
    {
        uint64_t maxTarget = totalRands + mt::MT_N + 1;
        int bits = 24;  // at least 2^24 so that the self test offsets are reachable
        while ((maxTarget >> bits) != 0) ++bits;
        mt::buildJumpTable(jt, 42, bits);
        if (state.rank == 0) mt::selfTest(jt);
    }

    if (L == 0) return;

    // Generator states at the start of every chunk of this rank's stream segment
    const uint64_t rankBase = rowBase[R0];
    const uint64_t rankEnd = rowBase[R0 + L];
    const uint64_t numChunks = (rankEnd - rankBase + RAND_CHUNK - 1) / RAND_CHUNK;
    uint32_t* d_chunkStates = nullptr;
    {
        std::vector<uint32_t> chunkStates(std::max<uint64_t>(1, numChunks) * mt::MT_N);
        mt::State s0;
        mt::jumpTo(jt, rankBase, s0);
        // Thread groups own contiguous chunk ranges: the first state of a group is
        // reached by a logarithmic number of jumps, the others by chaining one
        // RAND_CHUNK jump per chunk.
        const uint64_t numGroups = std::min<uint64_t>(numChunks, static_cast<uint64_t>(omp_get_max_threads()));
#pragma omp parallel for schedule(dynamic, 1)
        for (uint64_t grp = 0; grp < numGroups; ++grp) {
            const uint64_t c0 = numChunks * grp / numGroups;
            const uint64_t c1 = numChunks * (grp + 1) / numGroups;
            uint32_t cur[mt::MT_N], nxt[mt::MT_N];
            for (int k = 0; k < mt::MT_N; ++k) cur[k] = s0.mt[k];
            for (int b = 0; (c0 >> b) != 0; ++b) {
                if ((c0 >> b) & 1ull) {
                    mt::applyJumpPoly(jt.pow2Bits[RAND_CHUNK_LOG + b], cur, nxt);
                    for (int k = 0; k < mt::MT_N; ++k) cur[k] = nxt[k];
                }
            }
            for (uint64_t c = c0; c < c1; ++c) {
                uint32_t* dst = chunkStates.data() + c * mt::MT_N;
                for (int k = 0; k < mt::MT_N; ++k) dst[k] = cur[k];
                if (c + 1 < c1) {
                    mt::applyJumpPoly(jt.pow2Bits[RAND_CHUNK_LOG], cur, nxt);
                    for (int k = 0; k < mt::MT_N; ++k) cur[k] = nxt[k];
                }
            }
        }
        CUDA_CHECK(cudaMalloc(&d_chunkStates, chunkStates.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_chunkStates, chunkStates.data(), chunkStates.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }

    // Tiles of consecutive local rows whose random stream fits into one device buffer
    struct Tile { int row0, row1; uint64_t rand0, rand1; };
    std::vector<Tile> tiles;
    {
        uint64_t localRands = rankEnd - rankBase;
        uint64_t maxRow = 0;
        for (size_t il = 0; il < L; ++il) maxRow = std::max(maxRow, rowBase[R0 + il + 1] - rowBase[R0 + il]);
        uint64_t budget = std::max(std::min(MAX_TILE_RANDS, localRands), std::max<uint64_t>(maxRow, 4));
        int row0 = 0;
        while (row0 < static_cast<int>(L)) {
            int row1 = row0;
            while (row1 < static_cast<int>(L) && rowBase[R0 + row1 + 1] - rowBase[R0 + row0] <= budget) ++row1;
            if (row1 == row0) row1 = row0 + 1;
            tiles.push_back({row0, row1, rowBase[R0 + row0], rowBase[R0 + row1]});
            row0 = row1;
        }
    }
    size_t maxTileRows = 0;
    uint64_t maxTileRands = 4;
    for (const Tile& t : tiles) {
        maxTileRows = std::max(maxTileRows, static_cast<size_t>(t.row1 - t.row0));
        maxTileRands = std::max(maxTileRands, t.rand1 - t.rand0);
    }

    // Visibility mask and row stream bases on the device (used by the offsets kernel)
    uint32_t* d_pairMask = nullptr;
    unsigned long long* d_rowBase = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pairMask, pairMask.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rowBase, rowBase.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_pairMask, pairMask.data(), pairMask.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowBase, rowBase.data(), rowBase.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Double-buffered device buffers and streams: random numbers of tile k+1 are
    // generated while the form factors of tile k are computed
    float* d_rands[2];
    uint32_t* d_off[2];
    cudaStream_t streams[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_rands[b], maxTileRands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_off[b], maxTileRows * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
    }

    GpuGeometry geom;
    geom.tris = state.d_tris;
    geom.nodeCenter = state.d_nodeCenter;
    geom.nodeHalf = state.d_nodeHalf;
    geom.nodeChild = state.d_nodeChild;
    geom.nodeLeaf = state.d_nodeLeaf;
    geom.leafTris = state.d_leafTris;

    size_t nextProgress = 100;
    for (size_t k = 0; k < tiles.size(); ++k) {
        const Tile& tile = tiles[k];
        const int b = static_cast<int>(k & 1);
        const int tileRows = tile.row1 - tile.row0;
        const long long numPairs = static_cast<long long>(tileRows) * static_cast<long long>(N);

        // Per-pair offsets into the tile's random stream
        pairOffsetsKernel<<<static_cast<unsigned>(tileRows), 1024, 0, streams[b]>>>(
            static_cast<int>(N), static_cast<int>(wordsPerRow), R0, tile.row0, tile.rand0,
            d_pairMask, d_rowBase, d_off[b]);
        CUDA_CHECK(cudaGetLastError());

        // Random numbers of the tile, generated on the GPU chunk by chunk
        if (tile.rand1 > tile.rand0) {
            const uint64_t cFirst = (tile.rand0 - rankBase) >> RAND_CHUNK_LOG;
            const uint64_t cLast = (tile.rand1 - 1 - rankBase) >> RAND_CHUNK_LOG;
            mtGenerateKernel<<<static_cast<unsigned>(cLast - cFirst + 1), MT_GEN_THREADS, 0, streams[b]>>>(
                d_chunkStates + cFirst * mt::MT_N, rankBase + (cFirst << RAND_CHUNK_LOG),
                tile.rand0, tile.rand1, d_rands[b]);
            CUDA_CHECK(cudaGetLastError());
        }

        const long long threads = numPairs * NUM_RAYS;
        const unsigned blocks = static_cast<unsigned>((threads + 255) / 256);
        formFactorKernel<<<blocks, 256, 0, streams[b]>>>(static_cast<int>(N), R0, tile.row0, numPairs,
                                                        d_off[b], d_rands[b], geom, state.d_kij);
        CUDA_CHECK(cudaGetLastError());

        if (state.rank == 0) {
            // Progress report (scaled to the global triangle count)
            size_t doneGlobal = (k + 1 == tiles.size()) ? N : static_cast<size_t>(tile.row1) * N / L;
            while (nextProgress <= doneGlobal && nextProgress <= N) {
                printf("  Progress: %zu/%zu triangles\n", nextProgress, N);
                nextProgress += 100;
            }
            if (k + 1 == tiles.size() && (N % 100) != 0) printf("  Progress: %zu/%zu triangles\n", N, N);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij, L * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFree(d_rands[b]));
        CUDA_CHECK(cudaFree(d_off[b]));
    }
    CUDA_CHECK(cudaFree(d_pairMask));
    CUDA_CHECK(cudaFree(d_rowBase));
    CUDA_CHECK(cudaFree(d_chunkStates));
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    const size_t L = state.localRows;
    std::vector<Vec3> centers(N);
    for (size_t i = 0; i < N; ++i) centers[i] = state.triangles[i].center();
#pragma omp parallel for schedule(static)
    for (size_t il = 0; il < L; ++il) {
        const size_t i = state.rowStart + il;
        const Vec3 ci = centers[i];
        int* row = state.tau.data() + state.idx2d(il, 0);
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            row[j] = computeTau(ci, centers[j]);
        }
    }
    if (L > 0) {
        CUDA_CHECK(cudaMemcpy(state.d_tau, state.tau.data(), L * N * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t L = state.localRows;
    const size_t T = state.numTimesteps;
    const int Ni = static_cast<int>(N), Li = static_cast<int>(L);
    const int R0 = static_cast<int>(state.rowStart);

    // Pack weights and delays into the transposed layout used by the step kernel
    if (L > 0) {
        dim3 block(32, 8);
        dim3 grid(static_cast<unsigned>((N + 31) / 32), static_cast<unsigned>((L + 31) / 32));
        packWeightsKernel<<<grid, block>>>(Ni, Li, R0, state.d_kij, state.d_tau, state.d_areas, state.d_wtauT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(state.d_kij));
    state.d_kij = nullptr;
    CUDA_CHECK(cudaFree(state.d_tau));
    state.d_tau = nullptr;

    // Source emission active for the first half of the timesteps
    const int timeOff = static_cast<int>(T / 2);
    const int src = static_cast<int>(state.sourceIndex);

    for (size_t t = 0; t < T; ++t) {
        if (L > 0) {
            const unsigned blocks = static_cast<unsigned>((L + 63) / 64);
            simulationStepKernel<<<blocks, 64>>>(Ni, Li, R0, static_cast<int>(t), timeOff, src,
                                                  state.d_rho, state.d_wtauT, state.d_radB);
            CUDA_CHECK(cudaGetLastError());
        }
        if (state.size > 1) {
            // Exchange the radiosity of timestep t between ranks
            float* row = state.radB.data() + t * N;
            if (L > 0) {
                CUDA_CHECK(cudaMemcpy(row + state.rowStart, state.d_radB + t * N + state.rowStart,
                                      L * sizeof(float), cudaMemcpyDeviceToHost));
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, row, state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(state.d_radB + t * N, row, N * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (state.size == 1) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, T * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t L = state.localRows;
    const size_t T = state.numTimesteps;
    const int src = static_cast<int>(state.sourceIndex);

    std::vector<float> local(std::max<size_t>(1, L));
    if (L > 0) {
        const size_t shBytes = 2 * T * sizeof(float);
        if (shBytes <= 40 * 1024) {
            distanceKernel<true><<<static_cast<unsigned>(L), 256, shBytes>>>(
                static_cast<int>(N), static_cast<int>(T), static_cast<int>(state.rowStart), src,
                state.d_radB, state.d_dist);
        } else {
            distanceKernel<false><<<static_cast<unsigned>(L), 256, 0>>>(
                static_cast<int>(N), static_cast<int>(T), static_cast<int>(state.rowStart), src,
                state.d_radB, state.d_dist);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), state.d_dist, L * sizeof(float), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(local.data(), static_cast<int>(L), MPI_FLOAT, state.distances.data(),
                state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t L = state.localRows;

    // Kij statistics are distributed: reduce the local counts first
    long long nonZeroKijLocal = 0;
#pragma omp parallel for reduction(+ : nonZeroKijLocal) schedule(static)
    for (size_t i = 0; i < L * N; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKijLocal++;
    }
    long long nonZeroKij = 0;
    MPI_Reduce(&nonZeroKijLocal, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (state.rank == 0) {
        printf("\nValidation:\n");

        // Check that distances are non-negative
        bool allNonNegative = true;
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;
        bool finite = true;

        for (size_t i = 0; i < N; ++i) {
            val_t d = state.distances[i];
            if (d < 0) {
                allNonNegative = false;
                printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
            }
            if (!std::isfinite(d)) {
                printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
                finite = false;
                break;
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) nonZeroCount++;
        }

        if (!finite) {
            ok = 0;
        } else {
            printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
            printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
            printf("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

            // Check source distance is zero or very small
            val_t srcDist = state.distances[state.sourceIndex];
            if (srcDist > WAVE_SPEED * 2) {
                printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
            }

            // Check radiosity propagation (some triangles should have received energy)
            int receivedEnergy = 0;
#pragma omp parallel for reduction(+ : receivedEnergy) schedule(static)
            for (size_t i = 0; i < N; ++i) {
                for (size_t t = 0; t < state.numTimesteps; ++t) {
                    if (state.radB[state.idxTN(t, i)] > EPSILON) {
                        receivedEnergy++;
                        break;
                    }
                }
            }

            printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

            if (receivedEnergy == 0) {
                printf("  ERROR: No triangles received energy - simulation failed\n");
                ok = 0;
            } else {
                // Check Kij matrix (should have some non-zero entries)
                printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
                       nonZeroKij, N * N,
                       100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(N * N));

                if (nonZeroKij == 0) {
                    printf("  ERROR: All form factors are zero - visibility computation failed\n");
                    ok = 0;
                } else if (!allNonNegative) {
                    ok = 0;
                } else {
                    printf("  Validation: PASSED\n");
                }
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
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

static void releaseDevice(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodeCenter);
    cudaFree(state.d_nodeHalf);
    cudaFree(state.d_nodeChild);
    cudaFree(state.d_nodeLeaf);
    cudaFree(state.d_leafTris);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_wtauT);
    cudaFree(state.d_radB);
    cudaFree(state.d_dist);
}

static double wallTime() { return MPI_Wtime(); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int parseResult = -1;
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
            parseResult = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseResult = 1;
            break;
        }
    }
    if (parseResult >= 0) {
        MPI_Finalize();
        return parseResult;
    }

    // Select the GPU: one device per node-local rank
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        // Restrict the process to a single GPU before the driver initializes:
        // enumerating all GPUs of a node costs noticeable startup time per rank.
        if (getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
            int nodeDevices = 0;
            char path[64];
            for (;; ++nodeDevices) {
                snprintf(path, sizeof(path), "/dev/nvidia%d", nodeDevices);
                if (access(path, F_OK) != 0) break;
            }
            if (nodeDevices > 0) {
                char dev[16];
                snprintf(dev, sizeof(dev), "%d", localRank % nodeDevices);
                setenv("CUDA_VISIBLE_DEVICES", dev, 1);
            }
        }
        int devCount = 0;
        cudaError_t err = cudaGetDeviceCount(&devCount);
        if (err != cudaSuccess || devCount == 0) {
            fprintf(stderr, "No CUDA device available (%s)\n", cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
        CUDA_CHECK(cudaFree(0));
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.size = size;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = wallTime();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endPre = wallTime();
    long preDuration = static_cast<long>((endPre - startPre) * 1000.0);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = wallTime();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endSim = wallTime();
    long simDuration = static_cast<long>((endSim - startSim) * 1000.0);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = wallTime();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endDist = wallTime();
    long distDuration = static_cast<long>((endDist - startDist) * 1000.0);

    if (rank == 0) {
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
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (!validateResults(state)) {
            exitCode = 1;
        }
    }

    releaseDevice(state);
    MPI_Finalize();
    return exitCode;
}

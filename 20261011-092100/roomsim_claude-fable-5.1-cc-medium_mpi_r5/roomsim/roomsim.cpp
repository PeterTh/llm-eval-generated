/**
 * Room Response Simulation Benchmark (MPI distributed-memory version)
 * 
 * This is an MPI-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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

#include <mpi.h>
#include <sstream>

#include "../common/results_output.hpp"

// ============================================================================
// MPI process context
// ============================================================================
//
// Parallelization strategy
// ------------------------
// * The mesh and octree are small and deterministic: every rank builds them
//   redundantly, so no geometry has to be communicated.
// * The N x N form-factor (Kij) and time-delay (Tau) matrices are distributed
//   by rows. Each rank owns a contiguous block of triangles (rows) and never
//   stores the full matrices, so memory scales with 1/P.
// * The row blocks are balanced by the number of non-culled pairs per row,
//   which is exactly proportional to the form-factor work of that row.
// * The serial program draws all form-factor Monte-Carlo samples from one
//   mt19937 stream in row-major order. A non-culled pair consumes exactly
//   4*NUM_RAYS draws, a culled pair consumes none. Each rank therefore
//   positions its own generator at the exact stream offset where its row
//   block begins (fast O(log n) Mersenne-Twister jump-ahead for large
//   offsets, discard() for small ones). The results are bit-identical to the
//   serial implementation.
// * Wave propagation is advanced one timestep at a time: each rank computes
//   its rows of radB(t,:) and the row is MPI_Allgatherv'ed so that every rank
//   holds the full radiosity history (T x N, small).
// * Distances are computed for local rows and allgathered.

static int g_rank = 0;
static int g_size = 1;

#define ROOT_PRINTF(...)                        \
    do {                                        \
        if (g_rank == 0) printf(__VA_ARGS__);   \
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

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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

// ----------------------------------------------------------------------------
// Mersenne Twister (mt19937) jump-ahead.
//
// Advances the generator by n outputs in O(log n) polynomial operations plus a
// fixed O(19937 * 624) Horner evaluation, instead of O(n) for discard().
// The MT state transition M is linear over GF(2); for the 19937-bit effective
// state its minimal polynomial p(x) is primitive of degree 19937. We obtain
// p(x) at runtime with Berlekamp-Massey on an output bit sequence, compute
// g(x) = x^(n-1) mod p(x) by square-and-multiply, evaluate g(M)*state with a
// Horner scheme on the 624-word window, and then apply one final real step
// (this last step makes the 31 "dead" bits of the first word exact as well).
// Every polynomial is bit-packed in 64-bit words.
// ----------------------------------------------------------------------------
namespace mtjump {

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0dfU;
constexpr uint32_t UPPER_MASK = 0x80000000U;
constexpr uint32_t LOWER_MASK = 0x7fffffffU;
constexpr int DEG = 19937;                 // degree of the minimal polynomial
constexpr int PW = (DEG + 1 + 63) / 64;    // words holding a degree-DEG polynomial
constexpr int NSEQ = 2 * DEG;              // sequence length needed by Berlekamp-Massey

using Poly = std::vector<uint64_t>;

inline uint32_t stepWord(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t y = (a & UPPER_MASK) | (b & LOWER_MASK);
    return c ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
}

inline void seedWindow(uint32_t seed, uint32_t* x) {
    x[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        x[i] = 1812433253U * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

inline bool getBit(const Poly& p, size_t b) { return (p[b >> 6] >> (b & 63)) & 1u; }
inline void setBit(Poly& p, size_t b) { p[b >> 6] |= (uint64_t{1} << (b & 63)); }

// dst ^= (src[0..nsrc) << shift); dst must be large enough.
inline void xorShifted(Poly& dst, const Poly& src, size_t nsrc, size_t shift) {
    size_t ws = shift >> 6;
    unsigned bs = static_cast<unsigned>(shift & 63);
    if (bs == 0) {
        for (size_t i = 0; i < nsrc; ++i) dst[i + ws] ^= src[i];
    } else {
        for (size_t i = 0; i < nsrc; ++i) {
            dst[i + ws] ^= src[i] << bs;
            dst[i + ws + 1] ^= src[i] >> (64 - bs);
        }
    }
}

// Extract nwords words starting at bit offset 'off' of src (src must be padded).
inline void extractBits(const Poly& src, size_t off, uint64_t* out, size_t nwords) {
    size_t ws = off >> 6;
    unsigned bs = static_cast<unsigned>(off & 63);
    if (bs == 0) {
        for (size_t i = 0; i < nwords; ++i) out[i] = src[ws + i];
    } else {
        for (size_t i = 0; i < nwords; ++i) {
            out[i] = (src[ws + i] >> bs) | (src[ws + i + 1] << (64 - bs));
        }
    }
}

// Number of source words that can be xor-shifted by 'shift' bits into a buffer of 'size' words
inline size_t shiftWords(size_t size, size_t shift) {
    size_t ws = (shift >> 6) + 2;
    return size > ws ? size - ws : 0;
}

// Compute the minimal polynomial p(x) (degree DEG) of the MT recurrence.
// Returns false if the computed degree is unexpected.
inline bool computeMinimalPolynomial(Poly& p) {
    // Output bit sequence s_j = bit 0 of x_{624+j}, stored bit-reversed so that
    // the Berlekamp-Massey discrepancy becomes an aligned AND + parity.
    uint32_t w[MT_N];
    seedWindow(5489U, w);
    const size_t rwords = static_cast<size_t>(NSEQ + 63) / 64;
    Poly R(rwords + 2 * PW + 2, 0);   // padded for unaligned extraction
    int k = 0;
    for (int j = 0; j < NSEQ; ++j) {
        uint32_t nw = stepWord(w[k], w[(k + 1) % MT_N], w[(k + MT_M) % MT_N]);
        w[k] = nw;
        k = (k + 1) % MT_N;
        if (nw & 1u) setBit(R, static_cast<size_t>(NSEQ - 1 - j));
    }

    Poly C(2 * PW + 2, 0), B(2 * PW + 2, 0), T(2 * PW + 2, 0);
    std::vector<uint64_t> tmp(2 * PW + 2, 0);
    C[0] = 1; B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t i = 0; i < static_cast<size_t>(NSEQ); ++i) {
        // d = sum_{k=0..L} c_k s_{i-k} = parity(C & R[(NSEQ-1-i) ... ])
        size_t nw = (L >> 6) + 1;
        extractBits(R, static_cast<size_t>(NSEQ - 1) - i, tmp.data(), nw);
        uint64_t acc = 0;
        for (size_t q = 0; q < nw; ++q) acc ^= C[q] & tmp[q];
        int d = __builtin_parityll(acc);
        if (d == 0) {
            ++m;
        } else if (2 * L <= i) {
            T = C;
            xorShifted(C, B, shiftWords(B.size(), m), m);
            L = i + 1 - L;
            B = T;
            m = 1;
        } else {
            xorShifted(C, B, shiftWords(B.size(), m), m);
            ++m;
        }
    }
    if (L != static_cast<size_t>(DEG)) return false;

    // Reciprocal polynomial: p_k = c_{L-k}
    p.assign(PW, 0);
    for (size_t b = 0; b <= L; ++b) {
        if (getBit(C, L - b)) setBit(p, b);
    }
    return true;
}

// r = r^2 mod p
inline uint64_t spreadBits32(uint32_t v) {
    // Interleave zeros: bit i of v -> bit 2i of result
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

inline void squareMod(Poly& r, const Poly& p, Poly& tmp) {
    std::fill(tmp.begin(), tmp.end(), 0);
    // Bit spreading: bit i -> bit 2i (squaring over GF(2)[x])
    for (int i = 0; i < PW; ++i) {
        tmp[2 * i]     = spreadBits32(static_cast<uint32_t>(r[i]));
        tmp[2 * i + 1] = spreadBits32(static_cast<uint32_t>(r[i] >> 32));
    }
    // Reduce from degree 2*(DEG-1) down to DEG
    for (int b = 2 * (DEG - 1); b >= DEG; --b) {
        if (getBit(tmp, static_cast<size_t>(b))) {
            xorShifted(tmp, p, PW, static_cast<size_t>(b - DEG));
        }
    }
    std::copy(tmp.begin(), tmp.begin() + PW, r.begin());
}

// r = r*x mod p
inline void mulXMod(Poly& r, const Poly& p) {
    uint64_t carry = 0;
    for (int i = 0; i < PW; ++i) {
        uint64_t nc = r[i] >> 63;
        r[i] = (r[i] << 1) | carry;
        carry = nc;
    }
    if (getBit(r, DEG)) {
        for (int i = 0; i < PW; ++i) r[i] ^= p[i];
    }
}

// g = x^e mod p
inline void powXMod(uint64_t e, const Poly& p, Poly& g) {
    g.assign(PW, 0);
    g[0] = 1;
    Poly tmp(2 * PW + 2, 0);
    if (e == 0) return;
    int top = 63 - __builtin_clzll(e);
    for (int bit = top; bit >= 0; --bit) {
        squareMod(g, p, tmp);
        if ((e >> bit) & 1u) mulXMod(g, p);
    }
}

// Produce the 624-word window after advancing the seeded generator by n outputs.
inline bool jumpWindow(uint32_t seed, uint64_t n, uint32_t* out) {
    static Poly p;
    static bool pOk = false;
    static bool pTried = false;
    if (!pTried) {
        pTried = true;
        pOk = computeMinimalPolynomial(p);
    }
    if (!pOk) return false;
    if (n == 0) {
        seedWindow(seed, out);
        return true;
    }

    Poly g;
    powXMod(n - 1, p, g);

    uint32_t w[MT_N];
    seedWindow(seed, w);
    uint32_t acc[MT_N];
    std::memset(acc, 0, sizeof(acc));
    int k = 0;  // acc logical position i lives at acc[(k + i) % MT_N]
    for (int j = DEG - 1; j >= 0; --j) {
        // acc = M * acc
        acc[k] = stepWord(acc[k], acc[(k + 1) % MT_N], acc[(k + MT_M) % MT_N]);
        k = (k + 1) % MT_N;
        if (getBit(g, static_cast<size_t>(j))) {
            for (int i = 0; i < MT_N; ++i) acc[(k + i) % MT_N] ^= w[i];
        }
    }
    // Final real step: acc = M^n * w exactly (including dead bits)
    acc[k] = stepWord(acc[k], acc[(k + 1) % MT_N], acc[(k + MT_M) % MT_N]);
    k = (k + 1) % MT_N;
    for (int i = 0; i < MT_N; ++i) out[i] = acc[(k + i) % MT_N];
    return true;
}

// Set 'rng' to the state of std::mt19937(seed) advanced by n outputs.
inline bool jump(std::mt19937& rng, uint32_t seed, uint64_t n) {
    uint32_t win[MT_N];
    if (!jumpWindow(seed, n, win)) return false;
    // Textual state representation: x_{i-n} ... x_{i-1} followed by the index.
    std::ostringstream os;
    for (int i = 0; i < MT_N; ++i) os << win[i] << ' ';
    os << MT_N;
    std::istringstream is(os.str());
    is >> rng;
    return static_cast<bool>(is);
}

}  // namespace mtjump

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
    uint32_t seed_;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f), seed_(seed) {}
    val_t rand() { return dist(rng); }

    // Advance the stream so that the next rand() returns what the (n+1)-th
    // rand() of a freshly seeded generator would return. Must be called
    // before any rand() call. Equivalent to rng.discard(n).
    void skip(uint64_t n) {
        constexpr uint64_t JUMP_THRESHOLD = 200000000ULL;  // discard() is cheaper below this
        if (n < JUMP_THRESHOLD) {
            rng.discard(n);
            return;
        }
        std::mt19937 jumped;
        if (mtjump::jump(jumped, seed_, n)) {
            rng = jumped;
        } else {
            rng.discard(n);
        }
    }
};

// Number of rand() draws consumed by computeKij() for a non-culled pair
constexpr uint64_t RANDS_PER_PAIR = 4ULL * NUM_RAYS;

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

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
    std::vector<val_t> kij;         // Form factors (local rows x N, row-major)
    std::vector<int> tau;           // Time delays (local rows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Row (triangle) distribution across MPI ranks
    std::vector<size_t> rowStarts;  // size g_size + 1; rank r owns [rowStarts[r], rowStarts[r+1])
    std::vector<int> rowCounts;     // per-rank row counts (for Allgatherv)
    std::vector<int> rowDispls;     // per-rank row displacements (for Allgatherv)
    size_t rowBegin;                // first local row
    size_t rowEnd;                  // one past last local row
    uint64_t rngOffset;             // RNG draws consumed before the first local row

    size_t numLocalRows() const { return rowEnd - rowBegin; }
    // Index into the local row block (i is a global row index owned by this rank)
    size_t idx2dLocal(size_t i, size_t j) const { return (i - rowBegin) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

// Split rows into contiguous per-rank blocks so that the cumulative weight
// (number of non-culled pairs, i.e. form-factor work) is balanced.
static void partitionRows(const std::vector<uint64_t>& prefix, size_t n, std::vector<size_t>& starts) {
    starts.assign(static_cast<size_t>(g_size) + 1, 0);
    starts[g_size] = n;
    const uint64_t total = prefix[n];
    for (int r = 1; r < g_size; ++r) {
        size_t s;
        if (total == 0) {
            s = (n * static_cast<size_t>(r)) / static_cast<size_t>(g_size);
        } else {
            // Target: r/P of the total weight. Pick the row boundary closest to it.
            const double target = static_cast<double>(total) * static_cast<double>(r) / g_size;
            auto it = std::lower_bound(prefix.begin(), prefix.end(), static_cast<uint64_t>(target));
            s = static_cast<size_t>(it - prefix.begin());
            if (s > 0 && s <= n) {
                double above = static_cast<double>(prefix[s]) - target;
                double below = target - static_cast<double>(prefix[s - 1]);
                if (below < above) s -= 1;
            }
        }
        s = std::min(s, n);
        s = std::max(s, starts[r - 1]);
        starts[r] = s;
    }
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (deterministic; replicated on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    state.octree.build(state.triangles);

    const size_t n = state.numTriangles;

    // Initialize areas
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    // ------------------------------------------------------------------
    // Row distribution. The weight of row i is the number of pairs (i, j)
    // that are not culled by the normal test in computeKij(); this is both
    // the form-factor work of the row and (times RANDS_PER_PAIR) the number
    // of RNG draws it consumes in the serial program.
    // ------------------------------------------------------------------
    std::vector<uint64_t> rowWeight(n, 0);
    {
        // Compute weights in a plain block split, then share them.
        const size_t b0 = (n * static_cast<size_t>(g_rank)) / static_cast<size_t>(g_size);
        const size_t b1 = (n * static_cast<size_t>(g_rank + 1)) / static_cast<size_t>(g_size);
        std::vector<uint64_t> local(b1 - b0, 0);
        for (size_t i = b0; i < b1; ++i) {
            const Vec3 ni = state.triangles[i].normal();
            uint64_t cnt = 0;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                if (ni.dot(state.triangles[j].normal()) > 0.99f) continue;
                ++cnt;
            }
            local[i - b0] = cnt;
        }
        std::vector<int> counts(g_size), displs(g_size);
        for (int r = 0; r < g_size; ++r) {
            size_t s0 = (n * static_cast<size_t>(r)) / static_cast<size_t>(g_size);
            size_t s1 = (n * static_cast<size_t>(r + 1)) / static_cast<size_t>(g_size);
            counts[r] = static_cast<int>(s1 - s0);
            displs[r] = static_cast<int>(s0);
        }
        MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_UINT64_T,
                       rowWeight.data(), counts.data(), displs.data(), MPI_UINT64_T,
                       MPI_COMM_WORLD);
    }
    std::vector<uint64_t> prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + rowWeight[i];

    partitionRows(prefix, n, state.rowStarts);
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        state.rowCounts[r] = static_cast<int>(state.rowStarts[r + 1] - state.rowStarts[r]);
        state.rowDispls[r] = static_cast<int>(state.rowStarts[r]);
    }
    state.rowBegin = state.rowStarts[g_rank];
    state.rowEnd = state.rowStarts[g_rank + 1];
    state.rngOffset = RANDS_PER_PAIR * prefix[state.rowBegin];

    // Initialize matrices (local row blocks only for the N x N matrices)
    const size_t localRows = state.numLocalRows();
    state.kij.assign(localRows * n, ZERO);
    state.tau.assign(localRows * n, 0);
    state.radE.assign(timesteps * n, ZERO);
    state.radB.assign(timesteps * n, ZERO);
    state.distances.assign(n, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);
    rng.skip(state.rngOffset);  // position at the serial stream offset of our first row

    const size_t n = state.numTriangles;
    const size_t localRows = state.numLocalRows();
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        val_t* row = &state.kij[state.idx2dLocal(i, 0)];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            row[j] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        const size_t done = i + 1 - state.rowBegin;
        if (done % 100 == 0 || done == localRows) {
            ROOT_PRINTF("  Progress: %zu/%zu triangles (rank 0 block)\n", done, localRows);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        int* row = &state.tau[state.idx2dLocal(i, 0)];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            row[j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const int localRows = static_cast<int>(state.numLocalRows());

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        val_t* radBt = &state.radB[state.idxTN(t, 0)];

        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            const int* tauRow = &state.tau[state.idx2dLocal(i, 0)];
            const val_t* kijRow = &state.kij[state.idx2dLocal(i, 0)];
            val_t sumB = ZERO;

            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                int tauij = tauRow[j];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = kijRow[j];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            radBt[i] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radiosity row with all ranks (tau >= 1 for
        // every pair, so timestep t only depends on earlier timesteps).
        MPI_Allgatherv(localRows > 0 ? MPI_IN_PLACE : radBt, localRows, MPI_FLOAT,
                       radBt, state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    const int localRows = static_cast<int>(state.numLocalRows());

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    // Replicate the full distance vector on every rank
    MPI_Allgatherv(localRows > 0 ? MPI_IN_PLACE : state.distances.data(), localRows, MPI_FLOAT,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

// Collective: every rank must call it; the result is identical on all ranks.
bool validateResults(const SimulationState& state) {
    ROOT_PRINTF("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    bool allFinite = true;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            ROOT_PRINTF("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            ROOT_PRINTF("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            allFinite = false;
            break;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    // Count non-zero local form factors (global count via reduction)
    long long localNonZeroKij = 0;
    for (size_t i = 0; i < state.kij.size(); ++i) {
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (!allFinite) return false;

    ROOT_PRINTF("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    ROOT_PRINTF("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    ROOT_PRINTF("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        ROOT_PRINTF("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    ROOT_PRINTF("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        ROOT_PRINTF("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const size_t totalPairs = state.numTriangles * state.numTriangles;
    ROOT_PRINTF("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
                nonZeroKij, totalPairs,
                100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalPairs));

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

static double phaseTimerStart() {
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime();
}

static long phaseTimerStopMs(double start) {
    MPI_Barrier(MPI_COMM_WORLD);
    return static_cast<long>((MPI_Wtime() - start) * 1000.0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank)
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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("MPI ranks: %d\n", g_size);
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    // Precomputation
    double startPre = phaseTimerStart();

    computeTimeDelays(state);
    computeFormFactors(state);

    long preDuration = phaseTimerStopMs(startPre);

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    double startSim = phaseTimerStart();

    runSimulation(state);

    long simDuration = phaseTimerStopMs(startSim);

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    double startDist = phaseTimerStart();

    computeDistances(state);

    long distDuration = phaseTimerStopMs(startDist);

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

    // Memory usage (aggregate over all ranks for the distributed matrices)
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
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (collective)
    int rc = 0;
    if (validate) {
        if (!validateResults(state)) {
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}

/**
 * Room Response Simulation Benchmark (OpenMP shared-memory parallel version)
 * 
 * This is a simplified implementation of room impulse response
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
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>
#include <sched.h>

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
    // Per-ray quantities shared by every box test along a traversal.
    struct RayBox {
        Vec3 mid, d, ad;
        RayBox(const Vec3& p1, const Vec3& p2) {
            d = (p2 - p1) * 0.5f;
            mid = p1 + d;
            ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};
        }
    };

    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const RayBox& rb) const {
        const Vec3& d = rb.d;
        const Vec3& ad = rb.ad;
        Vec3 c = rb.mid - center;

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
        RayBox rb(p1, p2);
        return applyToTrisImpl(rb, func);
    }

private:
    template<typename Func>
    bool applyToTrisImpl(const RayBox& rb, Func& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            const Triangle* tris = allTriangles->data();
            for (size_t idx : triangleIndices) {
                if (func(idx, tris[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            const Octree* ch = children[i].get();
            if (ch && ch->rayIntersectsBox(rb)) {
                if (ch->applyToTrisImpl(rb, func)) return true;
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

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    // Skip ahead in the underlying engine stream (uniform_real_distribution is
    // stateless, so this is equivalent to drawing and dropping n values).
    void discard(unsigned long long n) { rng.discard(n); }
    std::mt19937& engine() { return rng; }
};

// ============================================================================
// Exact skip-ahead for std::mt19937
// ============================================================================
//
// The form factor phase consumes one sequential mt19937 stream. Each thread
// needs the engine state at its own offset in that stream. std::mt19937::
// discard is O(n); for large offsets we instead jump in O(log n) polynomial
// applications over GF(2) (Haramoto et al., "Efficient Jump Ahead for
// F2-Linear Random Number Generators"): the state transition A satisfies
// phi(A) = 0 for the characteristic polynomial phi (degree 19937), so
// A^d = (x^d mod phi)(A). The engine state is moved in and out of the
// std::mt19937 object through its stream operators (libstdc++ layout:
// 624 words followed by the index). Everything here is verified against
// std::mt19937::discard at runtime before being used; on any mismatch the
// code falls back to the library discard, so results never depend on it.
namespace mtjump {

constexpr int N = 624;
constexpr int M = 397;
constexpr uint32_t UM = 0x80000000u;
constexpr uint32_t LM = 0x7fffffffu;
constexpr uint32_t MA = 0x9908b0dfu;
constexpr int DEG = 19937;                 // degree of the characteristic polynomial
constexpr int PW = DEG / 64 + 1;           // 64-bit words per polynomial (312)

using Poly = std::vector<uint64_t>;

// One full regeneration of the 624-word state array (same recurrence as
// libstdc++'s _M_gen_rand, written as three loops so it vectorizes).
inline void regen(uint32_t* __restrict x) {
    for (int i = 0; i < N - M; ++i) {
        uint32_t y = (x[i] & UM) | (x[i + 1] & LM);
        x[i] = x[i + M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MA);
    }
    for (int i = N - M; i < N - 1; ++i) {
        uint32_t y = (x[i] & UM) | (x[i + 1] & LM);
        x[i] = x[i + M - N] ^ (y >> 1) ^ ((0u - (y & 1u)) & MA);
    }
    uint32_t y = (x[N - 1] & UM) | (x[0] & LM);
    x[N - 1] = x[M - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MA);
}

struct State {
    uint32_t x[N];
    unsigned long long p;   // index of the next word to output (0..N)
};

inline bool extract(const std::mt19937& e, State& s) {
    std::stringstream ss;
    ss << e;
    for (int i = 0; i < N; ++i) {
        unsigned long long v;
        if (!(ss >> v) || v > 0xffffffffULL) return false;
        s.x[i] = static_cast<uint32_t>(v);
    }
    if (!(ss >> s.p) || s.p > static_cast<unsigned long long>(N)) return false;
    std::string extra;
    if (ss >> extra) return false;   // unexpected layout
    return true;
}

inline bool inject(std::mt19937& e, const State& s) {
    std::stringstream ss;
    for (int i = 0; i < N; ++i) ss << s.x[i] << ' ';
    ss << s.p;
    ss >> e;
    return !ss.fail();
}

inline bool getBit(const Poly& a, size_t i) { return (a[i >> 6] >> (i & 63)) & 1u; }
inline void setBit(Poly& a, size_t i) { a[i >> 6] |= uint64_t(1) << (i & 63); }

// a ^= b << s   (a must be large enough)
inline void xorShifted(Poly& a, const Poly& b, size_t s) {
    size_t q = s >> 6, r = s & 63;
    size_t nb = b.size();
    if (r == 0) {
        for (size_t i = 0; i < nb && q + i < a.size(); ++i) a[q + i] ^= b[i];
    } else {
        uint64_t carry = 0;
        for (size_t i = 0; i < nb && q + i < a.size(); ++i) {
            a[q + i] ^= (b[i] << r) | carry;
            carry = b[i] >> (64 - r);
        }
        if (q + nb < a.size()) a[q + nb] ^= carry;
    }
}

struct JumpTable {
    Poly phi;                       // characteristic polynomial, degree DEG
    std::vector<Poly> phiShift;     // phi << r for r = 0..63 (PW + 1 words each)
    std::vector<Poly> pow2;         // x^(2^j) mod phi, PW words each
    bool ok = false;
};

// Minimal polynomial of the MT19937 word sequence via Berlekamp-Massey on
// bit 0 of X_1, X_2, ... (any nonzero linear functional of the state works
// because the characteristic polynomial is irreducible).
inline bool characteristicPolynomial(Poly& phi) {
    const size_t NB = 2 * static_cast<size_t>(DEG);
    State s;
    if (!extract(std::mt19937(42), s)) return false;
    // Sequence words X_0 .. X_{NB}; bit k of the sequence is X_{k+1} & 1.
    std::vector<uint8_t> seq(NB);
    {
        size_t k = 0;
        size_t idx = 1;   // skip X_0
        uint32_t x[N];
        std::memcpy(x, s.x, sizeof(x));
        while (k < NB) {
            for (; idx < static_cast<size_t>(N) && k < NB; ++idx) seq[k++] = x[idx] & 1u;
            regen(x);
            idx = 0;
        }
    }
    // Reversed, bit-packed sequence: R bit j = seq[NB-1-j]; padded by 64 bits.
    const size_t RW = NB / 64 + 3;
    Poly R(RW, 0);
    for (size_t j = 0; j < NB; ++j) if (seq[NB - 1 - j]) setBit(R, j);

    const size_t CW = NB / 64 + 3;
    Poly C(CW, 0), B(CW, 0), T;
    setBit(C, 0);
    setBit(B, 0);
    size_t L = 0, m = 1;
    for (size_t n = 0; n < NB; ++n) {
        // d = sum_{i=0..L} C_i * s_{n-i} = sum_i C_i * R[(NB-1-n) + i]
        size_t off = NB - 1 - n;
        size_t q = off >> 6, r = off & 63;
        uint64_t acc = 0;
        size_t words = L / 64 + 1;
        for (size_t w = 0; w < words; ++w) {
            uint64_t v = R[q + w] >> r;
            if (r) v |= R[q + w + 1] << (64 - r);
            acc ^= C[w] & v;
        }
        bool d = __builtin_parityll(acc);
        if (d) {
            if (2 * L <= n) {
                T = C;
                xorShifted(C, B, m);
                L = n + 1 - L;
                B = T;
                m = 1;
            } else {
                xorShifted(C, B, m);
                ++m;
            }
        } else {
            ++m;
        }
    }
    if (L != static_cast<size_t>(DEG)) return false;
    // phi(x) = x^L C(1/x): reverse the coefficients of C over degrees 0..L.
    phi.assign(PW, 0);
    for (size_t i = 0; i <= L; ++i) if (getBit(C, i)) setBit(phi, L - i);
    return getBit(phi, DEG);
}

// u (2*PW words, degree < 2*DEG) -> u mod phi (PW words).
inline void reduce(Poly& u, const JumpTable& jt) {
    // For p < 2*DEG, q + PW + 1 <= 2*PW, so the shifted phi always fits in u.
    uint64_t* __restrict ub = u.data();
    for (size_t p = 2 * static_cast<size_t>(DEG) - 1; p >= static_cast<size_t>(DEG); --p) {
        if ((ub[p >> 6] >> (p & 63)) & 1u) {
            size_t s = p - DEG;
            size_t q = s >> 6, r = s & 63;
            const uint64_t* __restrict ps = jt.phiShift[r].data();
            uint64_t* __restrict dst = ub + q;
            for (int i = 0; i < PW + 1; ++i) dst[i] ^= ps[i];
        }
    }
    u.resize(PW);
}

inline bool buildTable(JumpTable& jt, int maxBits) {
    if (!characteristicPolynomial(jt.phi)) return false;
    jt.phiShift.assign(64, Poly(PW + 1, 0));
    for (size_t r = 0; r < 64; ++r) xorShifted(jt.phiShift[r], jt.phi, r);
    // Bit-spread table for squaring (byte -> 16 bits).
    uint16_t spread[256];
    for (int b = 0; b < 256; ++b) {
        uint16_t v = 0;
        for (int k = 0; k < 8; ++k) if (b & (1 << k)) v |= uint16_t(1) << (2 * k);
        spread[b] = v;
    }
    jt.pow2.assign(static_cast<size_t>(maxBits), Poly());
    Poly cur(PW, 0);
    setBit(cur, 1);   // x
    for (int j = 0; j < maxBits; ++j) {
        jt.pow2[j] = cur;
        // cur = cur^2 mod phi
        Poly sq(2 * PW, 0);
        for (size_t w = 0; w < static_cast<size_t>(PW); ++w) {
            uint64_t v = cur[w];
            uint64_t lo = 0, hi = 0;
            for (int b = 0; b < 4; ++b) lo |= uint64_t(spread[(v >> (8 * b)) & 0xff]) << (16 * b);
            for (int b = 4; b < 8; ++b) hi |= uint64_t(spread[(v >> (8 * b)) & 0xff]) << (16 * (b - 4));
            sq[2 * w] = lo;
            sq[2 * w + 1] = hi;
        }
        reduce(sq, jt);
        cur = sq;
    }
    jt.ok = true;
    return true;
}

// x <- g(A) x for the window x (624 words), g of degree <= DEG-1, via Horner
// on a sliding window buffer. The low 31 bits of word 0 are "dead" state
// (never used by the recurrence) and may end up wrong; callers always finish
// with a full regen(), which only reads the top bit of word 0.
inline void applyPoly(const Poly& g, uint32_t* x, std::vector<uint32_t>& buf) {
    int deg = DEG - 1;
    while (deg > 0 && !getBit(g, static_cast<size_t>(deg))) --deg;
    buf.assign(static_cast<size_t>(N + deg + 1), 0u);
    uint32_t* b = buf.data();
    size_t t = 0;
    for (int i = deg; i >= 0; --i) {
        uint32_t y = (b[t] & UM) | (b[t + 1] & LM);
        b[t + N] = b[t + M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MA);
        ++t;
        if (getBit(g, static_cast<size_t>(i))) {
            uint32_t* __restrict w = b + t;
            for (int j = 0; j < N; ++j) w[j] ^= x[j];
        }
    }
    std::memcpy(x, b + t, sizeof(uint32_t) * N);
}

// Advance engine e by d outputs. Uses the jump table for the bulk of the
// skip when it is available and the skip is long; otherwise regenerates
// block by block. Returns false (engine untouched) if the state layout is
// not understood.
inline bool skip(std::mt19937& e, unsigned long long d, const JumpTable* jt,
                 unsigned long long minJumpWords, std::vector<uint32_t>& scratch) {
    State s;
    if (!extract(e, s)) return false;
    unsigned long long q = s.p + d;
    unsigned long long blocks = q / N;
    if (blocks == 0) {
        s.p = q;
    } else {
        unsigned long long words = (blocks - 1) * N;
        if (jt && jt->ok && words >= minJumpWords) {
            for (size_t j = 0; j < jt->pow2.size() && (words >> j) != 0; ++j) {
                if ((words >> j) & 1ULL) applyPoly(jt->pow2[j], s.x, scratch);
            }
        } else {
            for (unsigned long long b = 0; b + 1 < blocks; ++b) regen(s.x);
        }
        regen(s.x);
        s.p = q % N;
    }
    return inject(e, s);
}

// Verify skip() against std::mt19937::discard for a few offsets.
inline bool selfTest(const JumpTable* jt, unsigned long long minJumpWords) {
    const unsigned long long offsets[] = {0ULL, 1ULL, 623ULL, 624ULL, 625ULL, 1247ULL,
                                          1248ULL, 3ULL * 624 + 7, 20000ULL, 100003ULL};
    const unsigned long long starts[] = {0ULL, 300ULL, 624ULL};
    // With a jump table only the long offsets exercise anything new.
    const unsigned long long jumpOffsets[] = {3ULL * 624 + 7, 100003ULL, (1ULL << 20) + 13};
    std::vector<uint32_t> scratch;
    const unsigned long long* offs = jt ? jumpOffsets : offsets;
    const size_t nOffs = jt ? sizeof(jumpOffsets) / sizeof(jumpOffsets[0]) : sizeof(offsets) / sizeof(offsets[0]);
    for (unsigned long long st : starts) {
        for (size_t oi = 0; oi < nOffs; ++oi) {
            const unsigned long long d = offs[oi];
            std::mt19937 a(42), b(42);
            a.discard(st);
            b.discard(st);
            if (!skip(a, d, jt, minJumpWords, scratch)) return false;
            b.discard(d);
            for (int i = 0; i < 1500; ++i) {
                if (a() != b()) return false;
            }
        }
    }
    return true;
}

}  // namespace mtjump

// Allocator that leaves trivially-constructible elements uninitialized on
// resize(n), so large matrices are first-touched by the parallel loops that
// fill them (NUMA-friendly) instead of being zero-filled serially.
template <typename T>
struct NoInitAllocator {
    using value_type = T;
    NoInitAllocator() = default;
    template <typename U> NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p); }
    template <typename U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename A> void construct(U* p, const A& a) { ::new (static_cast<void*>(p)) U(a); }
    bool operator==(const NoInitAllocator&) const { return true; }
    bool operator!=(const NoInitAllocator&) const { return false; }
};
template <typename T>
using NoInitVector = std::vector<T, NoInitAllocator<T>>;

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

// Cull triangles facing the same direction. Kept out-of-line so that the
// random-stream accounting pass and the form factor pass evaluate exactly the
// same code (and therefore agree bit-for-bit on which pairs are culled).
__attribute__((noinline)) bool sameFacing(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Random numbers consumed by computeKij for a pair that is not culled:
// NUM_RAYS rays x 2 points x 2 uniform draws.
constexpr unsigned long long DRAWS_PER_PAIR = 4ULL * NUM_RAYS;

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (sameFacing(triI, triJ)) return ZERO;

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
    NoInitVector<val_t> kij;        // Form factors (N x N matrix, row-major)
    NoInitVector<int> tau;          // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    NoInitVector<val_t> radB;       // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    int tauMin = 1;                 // Smallest off-diagonal time delay (>= 1)

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices. kij, tau and radB are left uninitialized here and
    // are fully written (including diagonals) by the parallel phases below,
    // so their pages are first-touched by the threads that use them.
    state.kij.resize(state.numTriangles * state.numTriangles);
    state.tau.resize(state.numTriangles * state.numTriangles);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles);
    state.distances.resize(state.numTriangles, ZERO);

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

// Form factors are Monte Carlo estimates drawn from ONE sequential random
// stream (mt19937 seeded with 42) consumed in row-major pair order. To keep the
// results bit-identical to the sequential version, we first count how many
// draws each row consumes (a pair consumes DRAWS_PER_PAIR draws unless it is
// culled, in which case it consumes none), prefix-sum those counts, and give
// each thread a contiguous block of rows together with a private generator
// skipped ahead to the block's offset in the stream. Skipping costs ~0.3 ns
// per draw versus ~10 us per pair of real work, so the blocks are sized so
// that (skip cost + compute cost) is balanced across threads: early blocks
// (cheap skips) get more rows than late ones.
void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const std::vector<Triangle>& tris = state.triangles;

    // --- Pass 1: random draws consumed per row and stream offsets ----------
    std::vector<unsigned long long> rowPairs(N);   // non-culled pairs in row i
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        unsigned long long cnt = 0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (!sameFacing(tris[i], tris[j])) ++cnt;
        }
        rowPairs[i] = cnt;
    }

    // Prefix over rows of non-culled pairs: pairPrefix[i] = pairs in rows < i.
    std::vector<unsigned long long> pairPrefix(N + 1);
    pairPrefix[0] = 0;
    for (size_t i = 0; i < N; ++i) pairPrefix[i + 1] = pairPrefix[i] + rowPairs[i];
    const unsigned long long totalPairs = pairPrefix[N];
    const unsigned long long totalDraws = totalPairs * DRAWS_PER_PAIR;

    // --- Pass 2: skip machinery and cost calibration --------------------
    // Three ways to move a generator to an offset, all bit-exact:
    //   (1) std::mt19937::discard            (always available)
    //   (2) mtjump::skip block regeneration   (~2x faster than (1))
    //   (3) mtjump::skip polynomial jump      (O(log n), ~10 ms per thread,
    //                                          plus a one-time table build)
    // (2)/(3) are self-tested against (1) and disabled on any mismatch; they
    // are only enabled when the stream is long enough for them to pay off.
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    double skipNsPerDraw = 0.35;        // std::mt19937::discard, typical
    bool fastSkip = false;
    if (double(totalDraws) * skipNsPerDraw > 10.0 * 1e6) {
        fastSkip = mtjump::selfTest(nullptr, ~0ULL);
        std::vector<uint32_t> scratch;
        std::mt19937 probe(42);
        auto t0 = Clock::now();
        if (fastSkip) mtjump::skip(probe, 1ULL << 20, nullptr, ~0ULL, scratch);
        else probe.discard(1ULL << 20);
        auto t1 = Clock::now();
        skipNsPerDraw = ms(t0, t1) * 1e6 / double(1ULL << 20);
    }

    // Polynomial jump: fixed cost is ~45 ms (table + self-test) plus ~10 ms
    // per thread; the expected saving is about half of the longest
    // regeneration-based skip (threads skip concurrently, which also slows
    // the regeneration itself down).
    mtjump::JumpTable jumpTable;
    double jumpNs = 0.0;                // per-thread cost of a long jump
    bool useJump = false;
    if (fastSkip && double(totalDraws) * skipNsPerDraw > 200.0 * 1e6) {
        int bits = 1;
        while ((totalDraws >> bits) != 0) ++bits;
        if (mtjump::buildTable(jumpTable, bits + 1)) {
            std::vector<uint32_t> scratch;
            std::mt19937 probe(42);
            auto t0 = Clock::now();
            bool ok = mtjump::skip(probe, totalDraws, &jumpTable, 0, scratch);
            auto t1 = Clock::now();
            jumpNs = ms(t0, t1) * 1e6;
            useJump = ok && mtjump::selfTest(&jumpTable, 0);
        }
    }
    // Draws below this length are skipped by regeneration instead of jumping.
    const unsigned long long minJumpWords =
        useJump ? static_cast<unsigned long long>(jumpNs / std::max(skipNsPerDraw, 1e-3)) : ~0ULL;

    // Time a few real pairs with a scratch generator (results discarded) to
    // relate skip cost to compute cost. Only needed when skips are not free.
    double pairNs = 15000.0;
    if (double(totalDraws) * skipNsPerDraw > 2.0 * 1e6) {
        RandomGenerator probe(42);
        volatile val_t sink = ZERO;
        size_t samples = 0;
        auto t0 = Clock::now();
        for (size_t i = 0; i < N && samples < 32; i += std::max<size_t>(1, N / 8)) {
            for (size_t j = 0; j < N && samples < 32; j += std::max<size_t>(1, N / 8)) {
                if (i == j || sameFacing(tris[i], tris[j])) continue;
                sink = sink + computeKij(i, j, tris, state.octree, probe);
                ++samples;
            }
        }
        auto t1 = Clock::now();
        if (samples > 0) pairNs = ms(t0, t1) * 1e6 / double(samples);
        (void)sink;
    }
    // Cost (in pair-equivalents) of positioning a generator at stream offset o.
    auto skipCost = [&](unsigned long long o) {
        double c = double(o) * skipNsPerDraw;
        if (useJump) c = std::min(c, jumpNs);
        return c / std::max(pairNs, 1.0);
    };

    // --- Pass 3: balanced contiguous pair ranges --------------------------
    // Work is partitioned at pair granularity in row-major order. Thread k
    // handles the non-culled pairs with ordinals [pairBegin[k], pairBegin[k+1])
    // (plus the culled/diagonal positions in between, which are free).
    // Always the full team (see runSimulation); threads whose range turns
    // out empty simply idle.
    const int P = std::max(omp_get_max_threads(), 1);
    std::vector<unsigned long long> pairBegin(P + 1, totalPairs);
    {
        auto assign = [&](double budget, std::vector<unsigned long long>* out) -> bool {
            unsigned long long a = 0;
            for (int k = 0; k < P; ++k) {
                if (out) (*out)[k] = a;
                double room = budget - skipCost(a * DRAWS_PER_PAIR);
                unsigned long long take = 1;
                if (room > 1.0) take = static_cast<unsigned long long>(std::min(room, double(totalPairs)));
                a = std::min(totalPairs, a + take);
            }
            if (out) (*out)[P] = totalPairs;
            return a >= totalPairs;
        };
        double lo = 0.0, hi = double(totalPairs) + skipCost(totalDraws) + 1.0;
        for (int it = 0; it < 64; ++it) {
            double mid = 0.5 * (lo + hi);
            if (assign(mid, nullptr)) hi = mid; else lo = mid;
        }
        assign(hi, &pairBegin);
    }

    // Row-major position (i*N + j) of the non-culled pair with ordinal `ord`.
    auto positionOf = [&](unsigned long long ord) -> size_t {
        if (ord >= totalPairs) return N * N;
        size_t i = static_cast<size_t>(std::upper_bound(pairPrefix.begin(), pairPrefix.end(), ord) - pairPrefix.begin()) - 1;
        unsigned long long remaining = ord - pairPrefix[i];
        for (size_t j = 0; j < N; ++j) {
            if (i == j || sameFacing(tris[i], tris[j])) continue;
            if (remaining == 0) return i * N + j;
            --remaining;
        }
        return (i + 1) * N;   // unreachable for consistent counts
    };

    // --- Pass 4: compute --------------------------------------------------
    size_t progress = 0;   // rows completed (atomic updates)
    #pragma omp parallel
    {
        const int k = omp_get_thread_num();
        const unsigned long long a = pairBegin[std::min(k, P)];
        const unsigned long long b = pairBegin[std::min(k + 1, P)];
        const size_t posBegin = (k == 0) ? 0 : positionOf(a);
        const size_t posEnd = (k >= P - 1) ? N * N : positionOf(b);
        if (posBegin < posEnd) {
            RandomGenerator rng(42);
            const unsigned long long offset = a * DRAWS_PER_PAIR;
            bool skipped = false;
            if (fastSkip) {
                std::vector<uint32_t> scratch;
                skipped = mtjump::skip(rng.engine(), offset,
                                       useJump ? &jumpTable : nullptr, minJumpWords, scratch);
            }
            if (!skipped) {
                rng = RandomGenerator(42);
                rng.discard(offset);
            }
            val_t* kijAll = state.kij.data();
            for (size_t p = posBegin; p < posEnd; ++p) {
                const size_t i = p / N;
                const size_t j = p - i * N;
                kijAll[p] = (i == j) ? ZERO : computeKij(i, j, tris, state.octree, rng);
                if (j + 1 == N) {
                    size_t done;
                    #pragma omp atomic capture
                    done = ++progress;
                    if (done % 100 == 0 || done == N) {
                        printf("  Progress: %zu/%zu triangles\n", done, N);
                    }
                }
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    int tauMin = std::numeric_limits<int>::max();
    #pragma omp parallel for schedule(static) reduction(min:tauMin)
    for (size_t i = 0; i < N; ++i) {
        int* tauRow = &state.tau[state.idx2d(i, 0)];
        for (size_t j = 0; j < N; ++j) {
            if (i == j) { tauRow[j] = 0; continue; }
            int t = computeTau(state.triangles[i], state.triangles[j]);
            tauRow[j] = t;
            tauMin = std::min(tauMin, t);
        }
    }
    // Distinct centers give tau >= 1; clamp defensively so that the
    // simulation's timestep blocking below never assumes more than is true.
    state.tauMin = std::max(1, tauMin == std::numeric_limits<int>::max() ? 1 : tauMin);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const val_t* areas = state.areas.data();
    const val_t* rho = state.rho.data();
    const val_t* radE = state.radE.data();
    val_t* radB = state.radB.data();
    const val_t* kijAll = state.kij.data();
    const int* tauAll = state.tau.data();

    // Timestep t only reads radB rows <= t - tauMin, so any block of tauMin
    // consecutive timesteps is independent of itself and can be computed with
    // a single barrier per block. Rows are statically partitioned so that
    // each thread keeps revisiting its own rows of kij/tau.
    //
    // Every parallel region in this program uses the full default team:
    // libgomp takes a very long time to re-form a full team after a region
    // that ran with fewer threads.
    const size_t block = static_cast<size_t>(std::max(1, state.tauMin));

    #pragma omp parallel
    {
        for (size_t t0 = 0; t0 < T; t0 += block) {
            const size_t t1 = std::min(T, t0 + block);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                const val_t* kijRow = kijAll + i * N;
                const int* tauRow = tauAll + i * N;

                for (size_t t = t0; t < t1; ++t) {
                    val_t sumB = ZERO;

                    for (size_t j = 0; j < N; ++j) {
                        if (i == j) continue;

                        int tauij = tauRow[j];

                        // Skip if wave hasn't yet propagated from j to i
                        if (static_cast<int>(t) < tauij) continue;

                        val_t kij = kijRow[j];
                        if (kij <= ZERO) continue;

                        // Get radiosity from source triangle at time when emission occurred
                        size_t srcTime = t - static_cast<size_t>(tauij);
                        val_t radJ = radB[srcTime * N + j];
                        if (radJ <= ZERO) continue;

                        // Accumulate contribution: form factor * area * source radiosity
                        sumB += std::min(kij * areas[j], ONE) * radJ;
                    }

                    // Update radiosity: reflection + emission
                    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
                }
            }

            #pragma omp master
            for (size_t t = t0; t < t1; ++t) {
                if ((t + 1) % 10 == 0 || t + 1 == T) {
                    printf("  Timestep %zu/%zu\n", t + 1, T);
                }
            }
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
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
    int nonZeroKij = 0;
    #pragma omp parallel for schedule(static) reduction(+:nonZeroKij)
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

// Bind OpenMP worker threads to distinct CPUs of the process affinity mask
// when the runtime was not asked to bind them. Unbound threads woken for a
// new parallel region can be queued behind another of our (spinning) threads
// and lose a scheduler tick; with most hardware threads in use this happens
// at nearly every region start and dominates small problem sizes. Binding
// is skipped when fewer than half the CPUs are used (plenty of idle CPUs, and
// several instances may share the machine) or when oversubscribed.
void bindThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    // An explicit user choice (even "false") is respected.
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    }
    const int nthreads = omp_get_max_threads();
    if (nthreads > static_cast<int>(cpus.size()) || 2 * nthreads <= static_cast<int>(cpus.size())) return;
    #pragma omp parallel
    {
        int k = omp_get_thread_num();
        if (k < static_cast<int>(cpus.size())) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpus[k], &one);
            sched_setaffinity(0, sizeof(one), &one);
        }
    }
}

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

    bindThreads();

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
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}

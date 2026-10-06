/**
 * Room Response Simulation Benchmark
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
#include <mpi.h>
#include <random>
#include <vector>

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

    // Work estimate for a ray query: number of node box tests plus triangle tests
    // performed by applyToTris (assuming the ray is not blocked)
    size_t traversalCost(const Vec3& p1, const Vec3& p2) const {
        if (!triangleIndices.empty()) return triangleIndices.size();
        size_t cost = 0;
        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                ++cost;
                if (children[i]->rayIntersectsBox(p1, p2)) cost += children[i]->traversalCost(p1, p2);
            }
        }
        return cost;
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

// MT19937 engine producing exactly the same stream as std::mt19937, but stored
// as a circular buffer of the 624 most recent words so that its state can be
// manipulated as a vector over GF(2) (needed for jump-ahead).
class MT19937 {
public:
    using result_type = std::uint_fast32_t;
    static constexpr int N = 624;
    static constexpr int M = 397;

    uint32_t x[N];
    int p = 0;  // position of the oldest word

    MT19937() { std::memset(x, 0, sizeof(x)); }
    explicit MT19937(uint32_t seed) {
        x[0] = seed;
        for (int i = 1; i < N; ++i) {
            x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        p = 0;
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xffffffffu; }

    // Advance the linear recurrence by one word (no tempering)
    uint32_t step() {
        int p1 = (p + 1 == N) ? 0 : p + 1;
        int pm = (p + M >= N) ? p + M - N : p + M;
        uint32_t y = (x[p] & 0x80000000u) | (x[p1] & 0x7fffffffu);
        uint32_t v = x[pm] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        x[p] = v;
        p = p1;
        return v;
    }

    result_type operator()() {
        uint32_t y = step();
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    void discard(uint64_t n) { while (n--) step(); }

    // XOR the logical state (oldest word first) into acc
    void xorStateInto(uint32_t* acc) const {
        const int head = N - p;
        for (int i = 0; i < head; ++i) acc[i] ^= x[p + i];
        for (int i = 0; i < p; ++i) acc[head + i] ^= x[i];
    }
};

// ----------------------------------------------------------------------------
// MT19937 jump-ahead via the characteristic polynomial over GF(2).
// The state transition A is linear; for a jump of J steps we compute
// g(x) = x^J mod phi(x) and evaluate g(A) * state.  (The 31 "dead" low bits of
// the oldest word form the nilpotent part of A and never influence output.)
// ----------------------------------------------------------------------------
namespace mtjump {

constexpr int DEG = 19937;
constexpr int PW = DEG / 64 + 1;  // words for a polynomial of degree <= DEG

using Poly = std::vector<uint64_t>;

inline uint64_t getBits64(const Poly& a, size_t bitpos) {
    size_t q = bitpos >> 6, r = bitpos & 63;
    uint64_t lo = a[q] >> r;
    return r ? (lo | (a[q + 1] << (64 - r))) : lo;
}

// dst ^= src << shift   (src: nsrc words)
inline void shiftXor(Poly& dst, const Poly& src, size_t nsrc, size_t shift) {
    size_t q = shift >> 6, r = shift & 63;
    if (r == 0) {
        for (size_t k = 0; k < nsrc && k + q < dst.size(); ++k) dst[k + q] ^= src[k];
    } else {
        for (size_t k = 0; k < nsrc && k + q < dst.size(); ++k) {
            dst[k + q] ^= src[k] << r;
            if (k + q + 1 < dst.size()) dst[k + q + 1] ^= src[k] >> (64 - r);
        }
    }
}

// Characteristic polynomial phi of MT19937 via Berlekamp-Massey on one output bit
Poly characteristicPolynomial() {
    const size_t NS = 2 * DEG + 128;
    const size_t W = NS / 64 + 4;
    MT19937 e(5489u);
    Poly rs(W, 0);  // sequence stored reversed: rs[NS-1-n] = s_n
    for (size_t n = 0; n < NS; ++n) {
        uint64_t bit = e() & 1u;
        size_t pos = NS - 1 - n;
        rs[pos >> 6] |= bit << (pos & 63);
    }

    Poly C(W, 0), B(W, 0), T;
    C[0] = B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t n = 0; n < NS; ++n) {
        size_t o = NS - 1 - n;
        size_t nw = L / 64 + 1;
        uint64_t acc = 0;
        for (size_t w = 0; w < nw; ++w) acc ^= C[w] & getBits64(rs, o + 64 * w);
        if ((__builtin_popcountll(acc) & 1) == 0) {
            ++m;
            continue;
        }
        if (2 * L <= n) {
            T = C;
            shiftXor(C, B, W, m);
            L = n + 1 - L;
            B.swap(T);
            m = 1;
        } else {
            shiftXor(C, B, W, m);
            ++m;
        }
    }
    if (L != static_cast<size_t>(DEG)) {
        fprintf(stderr, "MT19937 jump: unexpected linear complexity %zu\n", L);
        std::abort();
    }
    // phi_k = C_{L-k}
    Poly phi(PW, 0);
    for (size_t k = 0; k <= L; ++k) {
        size_t c = L - k;
        if ((C[c >> 6] >> (c & 63)) & 1u) phi[k >> 6] |= uint64_t(1) << (k & 63);
    }
    return phi;
}

inline uint64_t spread32(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;
    return x;
}

// Compute x^J mod phi
Poly xPowMod(uint64_t J, const Poly& phi) {
    // Pre-shifted copies of phi: phiS[r] = phi << r (PW+1 words)
    std::vector<Poly> phiS(64, Poly(PW + 1, 0));
    for (int r = 0; r < 64; ++r) shiftXor(phiS[r], phi, PW, r);

    Poly g(PW, 0), s(2 * PW + 2, 0);
    g[0] = 1;
    int top = 63;
    while (top >= 0 && !((J >> top) & 1u)) --top;
    for (int b = top; b >= 0; --b) {
        // g = g^2 mod phi
        std::fill(s.begin(), s.end(), 0);
        for (int w = 0; w < PW; ++w) {
            s[2 * w] = spread32(static_cast<uint32_t>(g[w]));
            s[2 * w + 1] = spread32(static_cast<uint32_t>(g[w] >> 32));
        }
        for (long d = 2L * (DEG - 1); d >= DEG; --d) {
            if ((s[d >> 6] >> (d & 63)) & 1u) {
                size_t sh = static_cast<size_t>(d - DEG);
                size_t q = sh >> 6;
                const Poly& ps = phiS[sh & 63];
                for (int k = 0; k <= PW; ++k) s[q + k] ^= ps[k];
            }
        }
        for (int w = 0; w < PW; ++w) g[w] = s[w];
        // g = g * x mod phi
        if ((J >> b) & 1u) {
            uint64_t carry = 0;
            for (int w = 0; w < PW; ++w) {
                uint64_t nc = g[w] >> 63;
                g[w] = (g[w] << 1) | carry;
                carry = nc;
            }
            if ((g[DEG >> 6] >> (DEG & 63)) & 1u) {
                for (int w = 0; w < PW; ++w) g[w] ^= phi[w];
            }
        }
    }
    return g;
}

// Return an engine equal to `e` advanced by J steps
MT19937 jump(const MT19937& e, uint64_t J, const Poly& phi) {
    if (J < static_cast<uint64_t>(4 * DEG)) {
        MT19937 r = e;
        r.discard(J);
        return r;
    }
    Poly g = xPowMod(J, phi);
    MT19937 cur = e;
    MT19937 res;
    for (int k = 0; k < DEG; ++k) {
        if ((g[k >> 6] >> (k & 63)) & 1u) cur.xorStateInto(res.x);
        cur.step();
    }
    res.p = 0;
    return res;
}

}  // namespace mtjump

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    explicit RandomGenerator(const MT19937& engine) : rng(engine), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Each form factor evaluation that is not culled consumes exactly this many draws
constexpr uint64_t DRAWS_PER_KIJ = 4 * NUM_RAYS;

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

// Pairs of triangles facing the same direction are culled (no RNG draws).
// Kept out-of-line so that every caller makes the bit-identical decision.
__attribute__((noinline)) bool isCulledPair(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (isCulledPair(triI, triJ)) return ZERO;

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
// MPI helpers
// ============================================================================

static int g_rank = 0;
static int g_nprocs = 1;

// Only the root rank produces console output
#define rprintf(...) do { if (g_rank == 0) printf(__VA_ARGS__); } while (0)

// Even block partition of n items over p parts
static inline size_t blockStart(size_t n, int p, int r) {
    return (n / p) * r + std::min(static_cast<size_t>(r), n % p);
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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (root only)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Row distribution of the Kij/Tau matrices (rows [rowStart[r], rowStart[r+1]) on rank r)
    std::vector<size_t> rowStart;
    size_t rowBegin = 0, rowEnd = 0;

    // Local rows of the Kij/Tau matrices in compressed form: only entries with kij > 0
    // are stored (in increasing j order), with the precomputed weight min(kij*area_j, 1).
    std::vector<size_t> rowPtr;
    std::vector<uint32_t> colIdx;
    std::vector<int> colTau;
    std::vector<val_t> colW;
    uint64_t localNonZeroKij = 0;   // entries with kij > EPSILON

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (deterministic, replicated on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    if (g_rank == 0) state.distances.resize(state.numTriangles, ZERO);

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

// Determine the row distribution and the position of each rank's first row in the
// sequential random number stream. Every non-culled pair consumes a fixed number of
// draws, so the stream offset of row i is DRAWS_PER_KIJ * (non-culled pairs before i).
// Rows are distributed in contiguous blocks with balanced estimated cost, where the
// cost of a row is estimated by tracing one ray through the octree for a strided
// sample of its non-culled pairs.
uint64_t partitionRows(SimulationState& state) {
    const size_t n = state.numTriangles;
    const int P = g_nprocs;
    constexpr size_t SAMPLE_STRIDE = 8;
    constexpr uint64_t PAIR_BASE_COST = 4;  // RNG + geometry work per evaluated pair

    std::vector<Vec3> centers(n);
    for (size_t i = 0; i < n; ++i) centers[i] = state.triangles[i].center();

    // Count non-culled pairs and estimate cost per row (distributed, then all-gathered)
    size_t cb = blockStart(n, P, g_rank), ce = blockStart(n, P, g_rank + 1);
    std::vector<uint64_t> info(2 * n, 0);  // [count, cost] per row
    for (size_t i = cb; i < ce; ++i) {
        const Triangle& triI = state.triangles[i];
        uint64_t c = 0, sampled = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (isCulledPair(triI, state.triangles[j])) continue;
            ++c;
            if ((j + i) % SAMPLE_STRIDE == 0) {
                sampled += PAIR_BASE_COST + state.octree.traversalCost(centers[i], centers[j]);
            }
        }
        info[2 * i] = c;
        info[2 * i + 1] = sampled * SAMPLE_STRIDE + c * PAIR_BASE_COST / 4 + n;
    }
    std::vector<int> rc(P), rd(P);
    for (int r = 0; r < P; ++r) {
        rd[r] = static_cast<int>(2 * blockStart(n, P, r));
        rc[r] = static_cast<int>(2 * (blockStart(n, P, r + 1) - blockStart(n, P, r)));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, info.data(), rc.data(), rd.data(),
                   MPI_UINT64_T, MPI_COMM_WORLD);

    std::vector<uint64_t> pairPrefix(n + 1, 0), prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        pairPrefix[i + 1] = pairPrefix[i] + info[2 * i];
        prefix[i + 1] = prefix[i] + info[2 * i + 1];
    }
    const uint64_t total = prefix[n];

    state.rowStart.assign(P + 1, n);
    state.rowStart[0] = 0;
    for (int r = 1; r < P; ++r) {
        uint64_t target = static_cast<uint64_t>(
            (static_cast<long double>(total) * r) / P);
        size_t s = static_cast<size_t>(
            std::lower_bound(prefix.begin(), prefix.end(), target) - prefix.begin());
        s = std::min(s, n);
        state.rowStart[r] = std::max(s, state.rowStart[r - 1]);
    }
    state.rowStart[P] = n;
    state.rowBegin = state.rowStart[g_rank];
    state.rowEnd = state.rowStart[g_rank + 1];

    // Non-culled pairs preceding this rank's first row
    return pairPrefix[state.rowBegin] * DRAWS_PER_KIJ;
}

// Computes the local rows of Kij and Tau, storing them in compressed form.
void computeFormFactors(SimulationState& state, uint64_t drawOffset) {
    rprintf("Computing form factors (Kij)...\n");

    // Position the random stream exactly where the sequential code would be
    MT19937 engine(42);
    if (drawOffset > 0) {
        mtjump::Poly phi = mtjump::characteristicPolynomial();
        engine = mtjump::jump(engine, drawOffset, phi);
    }
    RandomGenerator rng(engine);

    const size_t n = state.numTriangles;
    const size_t localRows = state.rowEnd - state.rowBegin;
    state.rowPtr.assign(localRows + 1, 0);
    state.colIdx.clear();
    state.colTau.clear();
    state.colW.clear();
    state.localNonZeroKij = 0;

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
            if (kij > EPSILON) state.localNonZeroKij++;
            if (kij <= ZERO) continue;  // never contributes to the simulation
            state.colIdx.push_back(static_cast<uint32_t>(j));
            state.colTau.push_back(computeTau(triI, state.triangles[j]));
            state.colW.push_back(std::min(kij * state.areas[j], ONE));
        }
        state.rowPtr[i - state.rowBegin + 1] = state.colIdx.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t i = 0; i < n; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == n) {
            rprintf("  Progress: %zu/%zu triangles\n", i + 1, n);
        }
    }
}

void computeTimeDelays(SimulationState& /*state*/) {
    // Time delays are computed on the fly for the local rows (and only for
    // pairs that can contribute) during computeFormFactors.
    rprintf("Computing time delays (Tau)...\n");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const int P = g_nprocs;
    std::vector<int> rc(P), rd(P);
    for (int r = 0; r < P; ++r) {
        rd[r] = static_cast<int>(state.rowStart[r]);
        rc[r] = static_cast<int>(state.rowStart[r + 1] - state.rowStart[r]);
    }

    const uint32_t* colIdx = state.colIdx.data();
    const int* colTau = state.colTau.data();
    const val_t* colW = state.colW.data();
    const val_t* radB = state.radB.data();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);
        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            val_t sumB = ZERO;
            const size_t e0 = state.rowPtr[i - state.rowBegin];
            const size_t e1 = state.rowPtr[i - state.rowBegin + 1];

            for (size_t e = e0; e < e1; ++e) {
                int tauij = colTau[e];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = radB[srcTime * n + colIdx[e]];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += colW[e] * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radiosity with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.radB.data() + t * n,
                       rc.data(), rd.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            rprintf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    rprintf("Computing distances via cross-correlation...\n");

    const size_t n = state.numTriangles;
    const int P = g_nprocs;
    size_t db = blockStart(n, P, g_rank), de = blockStart(n, P, g_rank + 1);
    std::vector<val_t> localDist(de - db);

    for (size_t i = db; i < de; ++i) {
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

        localDist[i - db] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    std::vector<int> rc(P), rd(P);
    for (int r = 0; r < P; ++r) {
        rd[r] = static_cast<int>(blockStart(n, P, r));
        rc[r] = static_cast<int>(blockStart(n, P, r + 1) - blockStart(n, P, r));
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_FLOAT,
                g_rank == 0 ? state.distances.data() : nullptr, rc.data(), rd.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, uint64_t nonZeroKijTotal) {
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
    int nonZeroKij = static_cast<int>(nonZeroKijTotal);
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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on all ranks)
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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            rprintf("Unknown option: %s\n", argv[i]);
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rprintf("Room Response Simulation Benchmark\n");
    rprintf("===================================\n");
    rprintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rprintf("Timesteps: %d\n", timesteps);
    rprintf("Source triangle: %d\n", sourceIdx);
    rprintf("Reflectivity: %.2f\n", reflectivity);
    rprintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rprintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rprintf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    uint64_t drawOffset = partitionRows(state);
    computeTimeDelays(state);
    computeFormFactors(state, drawOffset);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rprintf("Distance computation time: %ld ms\n", distDuration);
    rprintf("\n");

    uint64_t nonZeroKij = 0;
    MPI_Reduce(&state.localNonZeroKij, &nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    long localTotalTime = preDuration + simDuration + distDuration;
    long totalTime = 0;
    MPI_Reduce(&localTotalTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (g_rank == 0) {
        // Total time
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
            if (!validateResults(state, nonZeroKij)) {
                exitCode = 1;
            }
        }
        fflush(stdout);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}

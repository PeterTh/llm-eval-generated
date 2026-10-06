/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, MPI-parallel implementation of room impulse response
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

// MT19937 engine producing exactly the same output stream as std::mt19937,
// extended with an O(log n) jump-ahead so that each MPI rank can start at an
// arbitrary position of the global sequential random stream.
class MT19937 {
public:
    using result_type = uint32_t;
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr int MEXP = 19937;   // degree of characteristic polynomial

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xffffffffu; }

    explicit MT19937(uint32_t seed = 5489u) { this->seed(seed); }

    void seed(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < N; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        idx = N;
    }

    result_type operator()() {
        if (idx >= N) twist();
        uint32_t y = mt[idx++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // Position the engine (seeded with `s`) so that the next output is output #pos.
    void seekFromSeed(uint32_t s, uint64_t pos);

private:
    uint32_t mt[N];
    int idx;

    static uint32_t next(uint32_t x0, uint32_t x1, uint32_t xm) {
        uint32_t y = (x0 & 0x80000000u) | (x1 & 0x7fffffffu);
        return xm ^ (y >> 1) ^ ((x1 & 1u) ? 0x9908b0dfu : 0u);
    }

    void twist() {
        for (int k = 0; k < N; ++k)
            mt[k] = next(mt[k], mt[(k + 1) % N], mt[(k + M) % N]);
        idx = 0;
    }
};

namespace gf2 {

using Poly = std::vector<uint64_t>;

inline bool getBit(const Poly& p, size_t i) { return (p[i >> 6] >> (i & 63)) & 1u; }

// dst ^= src << shift, caller guarantees dst has srcWords + 1 words past shift/64
inline void xorShiftedFast(uint64_t* __restrict dst, const uint64_t* __restrict src,
                           size_t srcWords, size_t shift) {
    uint64_t* d = dst + (shift >> 6);
    const unsigned bs = shift & 63;
    if (bs == 0) {
        for (size_t w = 0; w < srcWords; ++w) d[w] ^= src[w];
        return;
    }
    const unsigned rs = 64 - bs;
    d[0] ^= src[0] << bs;
    for (size_t w = 1; w < srcWords; ++w) d[w] ^= (src[w] << bs) | (src[w - 1] >> rs);
    d[srcWords] ^= src[srcWords - 1] >> rs;
}

// Characteristic polynomial of MT19937 via Berlekamp-Massey over GF(2).
// Returns p with p[k] = coefficient of x^k, degree MEXP.
inline Poly mtCharPoly() {
    const size_t n = MT19937::MEXP;
    const size_t len = 2 * n + 64;
    const size_t nw = len / 64 + 4;

    // rev bit (len-1-k) = s_k
    Poly rev(nw + 2, 0);
    MT19937 g(4357u);
    for (size_t k = 0; k < len; ++k) {
        if (g() >> 31) {
            size_t q = len - 1 - k;
            rev[q >> 6] |= uint64_t(1) << (q & 63);
        }
    }
    // shifted[b][w] = 64 bits of rev starting at bit 64*w + b (aligned access)
    const size_t sw = nw + 1;
    std::vector<uint64_t> shifted(64 * sw, 0);
    for (size_t b = 0; b < 64; ++b)
        for (size_t w = 0; w < sw; ++w)
            shifted[b * sw + w] = b ? ((rev[w] >> b) | (rev[w + 1] << (64 - b))) : rev[w];

    // Padded so that B << m always fits without bounds checks
    Poly C(2 * nw + 8, 0), B(2 * nw + 8, 0), T;
    C[0] = 1; B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t Nn = 0; Nn < len; ++Nn) {
        // d = sum_{i=0..L} c_i s_{Nn-i}; s_{Nn-i} = rev bit (len-1-Nn+i)
        size_t base = len - 1 - Nn;
        uint64_t acc = 0;
        size_t words = L / 64 + 1;
        const uint64_t* sv = &shifted[(base & 63) * sw + (base >> 6)];
        for (size_t w = 0; w < words; ++w) acc ^= C[w] & sv[w];
        if (!__builtin_parityll(acc)) { ++m; continue; }
        size_t bw = std::min(nw, (Nn + 1) / 64 + 2);
        if (2 * L <= Nn) {
            T = C;
            xorShiftedFast(C.data(), B.data(), bw, m);
            L = Nn + 1 - L;
            B.swap(T);
            m = 1;
        } else {
            xorShiftedFast(C.data(), B.data(), bw, m);
            ++m;
        }
    }
    if (L != n) {
        fprintf(stderr, "MT19937 characteristic polynomial degree mismatch (%zu)\n", L);
        abort();
    }
    Poly p(n / 64 + 1, 0);
    for (size_t k = 0; k <= n; ++k)
        if (getBit(C, n - k)) p[k >> 6] |= uint64_t(1) << (k & 63);
    return p;
}

// Interleave zeros: bit i of x (< 32) -> bit 2i
inline uint64_t spreadBits(uint64_t x) {
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;
    return x;
}

// x^e mod p, deg p = n
inline Poly xPowMod(uint64_t e, const Poly& p, size_t n) {
    const size_t pw = p.size();
    Poly r(pw, 0), sq(2 * pw + 1, 0);
    r[0] = 1;
    if (e == 0) return r;
    int top = 63 - __builtin_clzll(e);
    for (int bit = top; bit >= 0; --bit) {
        // r = r^2 mod p
        std::fill(sq.begin(), sq.end(), 0);
        for (size_t w = 0; w < pw; ++w) {
            uint64_t v = r[w];
            if (!v) continue;
            uint64_t lo = spreadBits(v & 0xffffffffu);
            uint64_t hi = spreadBits(v >> 32);
            sq[2 * w] = lo;
            sq[2 * w + 1] = hi;
        }
        for (size_t i = 2 * n; i >= n; --i) {
            if (getBit(sq, i)) xorShiftedFast(sq.data(), p.data(), pw, i - n);
            if (i == n) break;
        }
        std::copy(sq.begin(), sq.begin() + pw, r.begin());
        if ((e >> bit) & 1u) {
            // r = r * x mod p
            uint64_t carry = 0;
            for (size_t w = 0; w < pw; ++w) {
                uint64_t nc = r[w] >> 63;
                r[w] = (r[w] << 1) | carry;
                carry = nc;
            }
            if (getBit(r, n)) for (size_t w = 0; w < pw; ++w) r[w] ^= p[w];
        }
    }
    return r;
}

} // namespace gf2

inline void MT19937::seekFromSeed(uint32_t s, uint64_t pos) {
    seed(s);
    if (pos == 0) return;
    // Word sequence x_k: seeded array is x_0..x_623, output #d = temper(x_{624+d}).
    // Loading window W_s = (x_s..x_{s+623}) with idx = N yields output #s next.
    // W_pos = q(T) W_1 with q = x^(pos-1) mod charpoly (valid since all indices >= 1).
    static const gf2::Poly charPoly = gf2::mtCharPoly();
    gf2::Poly q = gf2::xPowMod(pos - 1, charPoly, MEXP);

    uint32_t base[N];
    for (int k = 0; k < N - 1; ++k) base[k] = mt[k + 1];
    base[N - 1] = next(mt[0], mt[1], mt[M]);

    uint32_t r[N];
    std::memset(r, 0, sizeof(r));
    int start = 0;  // logical k -> r[(start + k) % N]
    for (int i = MEXP; i >= 0; --i) {
        // r = T(r)
        uint32_t nv = next(r[start], r[(start + 1) % N], r[(start + M) % N]);
        r[start] = nv;
        start = (start + 1 == N) ? 0 : start + 1;
        if (gf2::getBit(q, static_cast<size_t>(i))) {
            int first = N - start;
            for (int k = 0; k < first; ++k) r[start + k] ^= base[k];
            for (int k = first; k < N; ++k) r[k - first] ^= base[k];
        }
    }
    for (int k = 0; k < N; ++k) mt[k] = r[(start + k) % N];
    idx = N;
}

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    // Start at position `pos` (in draws) of the stream seeded with `seed`
    RandomGenerator(uint32_t seed, uint64_t pos) : rng(seed), dist(0.0f, 1.0f) {
        rng.seekFromSeed(seed, pos);
    }
    val_t rand() { return dist(rng); }
};

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

// Culling test (kept out-of-line so every caller evaluates it identically;
// the random-stream partitioning relies on the exact same decisions)
__attribute__((noinline))
bool facingSameDirection(const Triangle& triI, const Triangle& triJ) {
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
    if (facingSameDirection(triI, triJ)) return ZERO;

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

// Sparse non-zero form factor entry of a receiver row
struct RowEntry {
    uint32_t j;     // emitter index
    int tau;        // time delay
    val_t w;        // min(kij * area_j, 1)
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors of local rows (rowCount x N, row-major)
    std::vector<int> tau;           // Time delays of local rows (rowCount x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source

    // Compressed rows used by the simulation (local rows only)
    std::vector<size_t> rowPtr;
    std::vector<RowEntry> entries;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI decomposition: rows [rowBegin, rowEnd) are owned by this rank
    int rank = 0, nprocs = 1;
    std::vector<int> rowStarts;     // nprocs + 1 entries
    size_t rowBegin = 0, rowEnd = 0;
    uint64_t nonZeroKij = 0;        // Global count of kij > EPSILON

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    size_t localRows() const { return rowEnd - rowBegin; }
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

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Row decomposition
    const size_t n = state.numTriangles;
    state.rowStarts.resize(state.nprocs + 1);
    for (int r = 0; r <= state.nprocs; ++r)
        state.rowStarts[r] = static_cast<int>(n * static_cast<size_t>(r) / state.nprocs);
    state.rowBegin = state.rowStarts[state.rank];
    state.rowEnd = state.rowStarts[state.rank + 1];

    // Initialize matrices
    state.kij.assign(state.localRows() * n, ZERO);
    state.tau.assign(state.localRows() * n, 0);
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
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const int P = state.nprocs;
    const auto& tris = state.triangles;

    // The sequential code draws 4*NUM_RAYS random numbers for every
    // non-culled pair (i != j) in row-major order from a single stream.
    // Count non-culled pairs per row (distributed), then split the flattened
    // pair space so each rank gets an equal share of real work and jumps the
    // random stream directly to its starting position.
    std::vector<int> rowWork(n, 0);
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        int c = 0;
        for (size_t j = 0; j < n; ++j)
            if (j != i && !facingSameDirection(tris[i], tris[j])) ++c;
        rowWork[i] = c;
    }
    {
        std::vector<int> counts(P), displs(P);
        for (int r = 0; r < P; ++r) {
            counts[r] = state.rowStarts[r + 1] - state.rowStarts[r];
            displs[r] = state.rowStarts[r];
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rowWork.data(),
                       counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
    }
    std::vector<uint64_t> cum(n + 1, 0);
    for (size_t i = 0; i < n; ++i) cum[i + 1] = cum[i] + static_cast<uint64_t>(rowWork[i]);
    const uint64_t totalWork = cum[n];

    // Flattened index k = i*n + j with exactly w non-culled pairs before it
    auto findPos = [&](uint64_t w) -> size_t {
        if (w >= totalWork) return n * n;
        size_t i = static_cast<size_t>(std::upper_bound(cum.begin(), cum.end(), w) - cum.begin()) - 1;
        uint64_t c = cum[i];
        for (size_t j = 0; j < n; ++j) {
            if (c == w) return i * n + j;
            if (j != i && !facingSameDirection(tris[i], tris[j])) ++c;
        }
        return (i + 1) * n;  // not reached
    };
    const uint64_t wBegin = totalWork * static_cast<uint64_t>(state.rank) / P;
    const uint64_t wEnd = totalWork * static_cast<uint64_t>(state.rank + 1) / P;
    const size_t kBegin = (state.rank == 0) ? 0 : findPos(wBegin);
    const size_t kEnd = (state.rank == P - 1) ? n * n : findPos(wEnd);

    RandomGenerator rng(42, wBegin * static_cast<uint64_t>(4 * NUM_RAYS));
    std::vector<val_t> part(kEnd - kBegin, ZERO);
    for (size_t k = kBegin; k < kEnd; ++k) {
        size_t i = k / n, j = k % n;
        if (i == j) continue;
        part[k - kBegin] = computeKij(i, j, tris, state.octree, rng);
    }

    // Redistribute flattened pair ranges to row owners
    std::vector<int> scounts(P, 0), sdispls(P, 0), rcounts(P, 0), rdispls(P, 0);
    const size_t myRowLo = state.rowBegin * n, myRowHi = state.rowEnd * n;
    std::vector<size_t> kStarts(P + 1);
    {
        // Gather every rank's pair range
        std::vector<unsigned long long> kb(P);
        unsigned long long mine = kBegin;
        MPI_Allgather(&mine, 1, MPI_UNSIGNED_LONG_LONG, kb.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
        for (int r = 0; r < P; ++r) kStarts[r] = kb[r];
        kStarts[P] = n * n;
    }
    for (int r = 0; r < P; ++r) {
        size_t lo = static_cast<size_t>(state.rowStarts[r]) * n;
        size_t hi = static_cast<size_t>(state.rowStarts[r + 1]) * n;
        size_t a = std::max(lo, kBegin), b = std::min(hi, kEnd);
        if (a < b) { scounts[r] = static_cast<int>(b - a); sdispls[r] = static_cast<int>(a - kBegin); }
        size_t c = std::max(myRowLo, kStarts[r]), d = std::min(myRowHi, kStarts[r + 1]);
        if (c < d) { rcounts[r] = static_cast<int>(d - c); rdispls[r] = static_cast<int>(c - myRowLo); }
    }
    MPI_Alltoallv(part.data(), scounts.data(), sdispls.data(), MPI_FLOAT,
                  state.kij.data(), rcounts.data(), rdispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

    // Count non-zero form factors (for validation)
    unsigned long long nz = 0;
    for (val_t v : state.kij) if (v > EPSILON) ++nz;
    unsigned long long nzAll = 0;
    MPI_Allreduce(&nz, &nzAll, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = nzAll;

    if (state.rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    const size_t n = state.numTriangles;

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        int* row = &state.tau[(i - state.rowBegin) * n];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            row[j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// Build compressed rows of contributing (kij > 0) pairs for local receivers
void buildSparseRows(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t rows = state.localRows();
    state.rowPtr.assign(rows + 1, 0);
    state.entries.clear();
    for (size_t li = 0; li < rows; ++li) {
        size_t i = state.rowBegin + li;
        const val_t* krow = &state.kij[li * n];
        const int* trow = &state.tau[li * n];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            val_t kij = krow[j];
            if (kij <= ZERO) continue;
            state.entries.push_back({static_cast<uint32_t>(j), trow[j],
                                     std::min(kij * state.areas[j], ONE)});
        }
        state.rowPtr[li + 1] = state.entries.size();
    }
    // Dense local matrices are no longer needed
    std::vector<val_t>().swap(state.kij);
    std::vector<int>().swap(state.tau);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Sum of contributions to one receiver at timestep t. The product and the
// accumulation are rounded separately (no FMA contraction), exactly like the
// reference loop where min(kij * area, 1) * radJ is added to the sum.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("fp-contract=off")))
#endif
val_t accumulateRow(const RowEntry* begin, const RowEntry* end,
                    const val_t* radB, size_t n, size_t t) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int ti = static_cast<int>(t);
    val_t sumB = ZERO;
    for (const RowEntry* en = begin; en != end; ++en) {
        // Skip if wave hasn't yet propagated from j to i
        if (ti < en->tau) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(en->tau);
        val_t radJ = radB[srcTime * n + en->j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        val_t contrib = en->w * radJ;
        sumB += contrib;
    }
    return sumB;
}

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    const int P = state.nprocs;

    std::vector<int> counts(P), displs(P);
    for (int r = 0; r < P; ++r) {
        counts[r] = state.rowStarts[r + 1] - state.rowStarts[r];
        displs[r] = state.rowStarts[r];
    }

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        val_t* out = &state.radB[t * n];
        for (size_t li = 0; li < state.localRows(); ++li) {
            size_t i = state.rowBegin + li;
            val_t sumB = accumulateRow(&state.entries[0] + state.rowPtr[li],
                                       &state.entries[0] + state.rowPtr[li + 1],
                                       state.radB.data(), n, t);

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, out,
                       counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Contiguous time series of the source and local receivers
    std::vector<val_t> src(T), series(T);
    for (size_t t = 0; t < T; ++t) src[t] = state.radB[state.idxTN(t, state.sourceIndex)];

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t t = 0; t < T; ++t) series[t] = state.radB[state.idxTN(t, i)];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = series[tt];
                val_t pS = src[tt - t];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    std::vector<int> counts(state.nprocs), displs(state.nprocs);
    for (int r = 0; r < state.nprocs; ++r) {
        counts[r] = state.rowStarts[r + 1] - state.rowStarts[r];
        displs[r] = state.rowStarts[r];
    }
    if (state.rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                    counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(state.distances.data() + state.rowBegin, counts[state.rank], MPI_FLOAT,
                    nullptr, nullptr, nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    (void)n;
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
    int nonZeroKij = static_cast<int>(state.nonZeroKij);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);
    // Only rank 0 produces console output
    if (!root && !freopen("/dev/null", "w", stdout)) return 1;

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
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
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
    state.rank = rank;
    state.nprocs = nprocs;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);
    buildSparseRows(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);

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

    // Validation (rank 0 holds all gathered results)
    int status = 0;
    if (validate && root) {
        if (!validateResults(state)) {
            status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return status;
}

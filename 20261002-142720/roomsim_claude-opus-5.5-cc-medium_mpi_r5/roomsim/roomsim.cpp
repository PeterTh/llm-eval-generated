/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
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

// MT19937 engine producing exactly the same output sequence as std::mt19937,
// stored as a circular buffer so that it supports cheap O(1)-per-output stepping
// and GF(2) jump-ahead. Jump-ahead lets every MPI rank start at its exact offset
// in the (originally sequential) random stream without generating all prior values.
class MT19937 {
public:
    using result_type = uint32_t;
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr int MEXP = 19937;
    static constexpr result_type min() { return 0u; }
    static constexpr result_type max() { return 0xffffffffu; }

    explicit MT19937(uint32_t seed = 5489u) {
        mt[0] = seed;
        for (int k = 1; k < N; ++k) mt[k] = 1812433253u * (mt[k - 1] ^ (mt[k - 1] >> 30)) + static_cast<uint32_t>(k);
        idx = 0;
    }

    // Advance the state by one output without tempering
    void step() {
        int i1 = idx + 1 == N ? 0 : idx + 1;
        int iM = idx + M >= N ? idx + M - N : idx + M;
        uint32_t y = (mt[idx] & 0x80000000u) | (mt[i1] & 0x7fffffffu);
        mt[idx] = mt[iM] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }

    result_type operator()() {
        step();
        uint32_t y = mt[idx];
        idx = idx + 1 == N ? 0 : idx + 1;
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    void discard(uint64_t n) {
        for (uint64_t k = 0; k < n; ++k) {
            step();
            idx = idx + 1 == N ? 0 : idx + 1;
        }
    }

    // Advance the generator by n outputs (equivalent to discard(n))
    void advance(uint64_t n) {
        if (n < (uint64_t{1} << 20)) { discard(n); return; }
        jump(n);
    }

private:
    uint32_t mt[N];
    int idx;

    using Poly = std::vector<uint64_t>;
    static constexpr int PW = (MEXP + 64) / 64 + 1;   // words for a reduced polynomial (+slack)
    static constexpr int BW = 2 * PW + 2;             // words for an unreduced product

    static bool getBit(const Poly& p, size_t b) { return (p[b >> 6] >> (b & 63)) & 1u; }

    // Minimal polynomial of the MT19937 transition, via Berlekamp-Massey over GF(2)
    static const Poly& charPoly() {
        static const Poly phi = [] {
            const size_t NS = 2 * MEXP;
            const size_t W = NS / 64 + 2;
            // Reversed output-bit sequence: R bit (NS-1-k) = s_k
            Poly R(W + 1, 0);
            MT19937 g(5489u);
            for (size_t k = 0; k < NS; ++k) {
                if (g() & 1u) { size_t b = NS - 1 - k; R[b >> 6] |= uint64_t{1} << (b & 63); }
            }
            // Rsh[s] = R shifted right by s bits, so every window read is word-aligned
            std::vector<Poly> Rsh(64, Poly(W + 1, 0));
            for (size_t sh = 0; sh < 64; ++sh) {
                for (size_t q = 0; q < W; ++q) {
                    uint64_t v = R[q] >> sh;
                    if (sh) v |= R[q + 1] << (64 - sh);
                    Rsh[sh][q] = v;
                }
            }
            Poly C(W, 0), B(W, 0), T;
            C[0] = B[0] = 1;
            size_t Lc = 0, m = 1;
            auto xorShifted = [&](Poly& dst, const Poly& src, size_t shift, size_t srcWords) {
                size_t wo = shift >> 6, s = shift & 63;
                for (size_t w = 0; w < srcWords && w + wo < dst.size(); ++w) {
                    dst[w + wo] ^= src[w] << s;
                    if (s && w + wo + 1 < dst.size()) dst[w + wo + 1] ^= src[w] >> (64 - s);
                }
            };
            for (size_t n = 0; n < NS; ++n) {
                size_t off = NS - 1 - n;
                size_t words = n / 64 + 1;
                const uint64_t* rw = Rsh[off & 63].data() + (off >> 6);
                uint64_t acc = 0;
                for (size_t w = 0; w <= Lc / 64; ++w) acc ^= C[w] & rw[w];
                if (!(__builtin_popcountll(acc) & 1)) { ++m; continue; }
                if (2 * Lc <= n) {
                    T = C;
                    xorShifted(C, B, m, words);
                    Lc = n + 1 - Lc;
                    B = std::move(T);
                    m = 1;
                } else {
                    xorShifted(C, B, m, words);
                    ++m;
                }
            }
            // phi(t) = t^Lc * C(1/t)
            Poly p(PW, 0);
            for (size_t i = 0; i <= Lc; ++i) {
                if (getBit(C, i)) { size_t b = Lc - i; p[b >> 6] |= uint64_t{1} << (b & 63); }
            }
            return p;
        }();
        return phi;
    }

    static uint64_t spread32(uint32_t x) {
        uint64_t v = x;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
        v = (v | (v << 2)) & 0x3333333333333333ull;
        v = (v | (v << 1)) & 0x5555555555555555ull;
        return v;
    }

    // Compute t^n mod phi(t)
    static Poly powMod(uint64_t n) {
        const Poly& phi = charPoly();
        // phi shifted by s bits, s = 0..63, for aligned reduction
        std::vector<Poly> phiSh(64, Poly(PW + 1, 0));
        for (int s = 0; s < 64; ++s) {
            for (int w = 0; w < PW; ++w) {
                phiSh[s][w] ^= phi[w] << s;
                if (s) phiSh[s][w + 1] ^= phi[w] >> (64 - s);
            }
        }
        Poly r(PW, 0), buf(BW, 0);
        uint64_t e = 0;
        int msb = 63 - __builtin_clzll(n);
        for (int bit = msb; bit >= 0; --bit) {
            uint64_t b = (n >> bit) & 1u;
            uint64_t en = 2 * e + b;
            if (en < static_cast<uint64_t>(MEXP)) {
                std::fill(r.begin(), r.end(), 0);
                r[en >> 6] = uint64_t{1} << (en & 63);
            } else {
                // square
                std::fill(buf.begin(), buf.end(), 0);
                for (int w = 0; w < PW; ++w) {
                    buf[2 * w] = spread32(static_cast<uint32_t>(r[w]));
                    buf[2 * w + 1] = spread32(static_cast<uint32_t>(r[w] >> 32));
                }
                // multiply by t
                if (b) {
                    for (int w = BW - 1; w > 0; --w) buf[w] = (buf[w] << 1) | (buf[w - 1] >> 63);
                    buf[0] <<= 1;
                }
                // reduce
                for (size_t d = 2 * MEXP; d >= static_cast<size_t>(MEXP); --d) {
                    if (!getBit(buf, d)) continue;
                    size_t sh = d - MEXP;
                    size_t wo = sh >> 6;
                    const Poly& ps = phiSh[sh & 63];
                    for (int w = 0; w <= PW; ++w) buf[wo + w] ^= ps[w];
                }
                std::copy(buf.begin(), buf.begin() + PW, r.begin());
            }
            e = en;
        }
        return r;
    }

    void jump(uint64_t n) {
        Poly g = powMod(n);
        uint32_t acc[N] = {};
        MT19937 cur = *this;
        for (int i = 0; i < MEXP; ++i) {
            if (getBit(g, static_cast<size_t>(i))) {
                int first = N - cur.idx;
                for (int k = 0; k < first; ++k) acc[k] ^= cur.mt[cur.idx + k];
                for (int k = first; k < N; ++k) acc[k] ^= cur.mt[k - first];
            }
            cur.step();
            cur.idx = cur.idx + 1 == N ? 0 : cur.idx + 1;
        }
        std::memcpy(mt, acc, sizeof(mt));
        idx = 0;
    }
};

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    // Skip n engine outputs (each rand() consumes exactly one 32-bit output)
    void advance(uint64_t n) { rng.advance(n); }
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
// MPI helpers
// ============================================================================

static int g_rank = 0;
static int g_nprocs = 1;

// printf only on rank 0
template<typename... Args>
void log0(const char* fmt, Args... args) {
    if (g_rank == 0) printf(fmt, args...);
}
void log0(const char* s) {
    if (g_rank == 0) fputs(s, stdout);
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated on all ranks)
    std::vector<val_t> distances;   // Computed distances from source (complete on rank 0)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Row decomposition: this rank owns receiver triangles [rowBegin, rowEnd)
    size_t rowBegin = 0, rowEnd = 0;
    std::vector<int> rowCounts, rowDispls;   // per-rank row block sizes/offsets

    // Local rows of the Kij / Tau matrices in sparse form (only entries with Kij > 0,
    // stored in increasing j order)
    std::vector<size_t> rowPtr;
    std::vector<idx_t> colIdx;
    std::vector<int> tauVal;
    std::vector<val_t> weight;      // min(Kij * area_j, 1)
    long long localNonZeroKij = 0;  // count of local Kij > EPSILON

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

    log0("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    log0("Building octree...\n");
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

// Whether computeKij culls the pair (i, j) without consuming random numbers
inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Partition rows into contiguous, work-balanced blocks. Returns, for every row,
// the number of (i, j) pairs that consume random numbers.
std::vector<long long> partitionRows(SimulationState& state) {
    const size_t n = state.numTriangles;
    const int P = g_nprocs;

    // Count non-culled pairs per row, in parallel over an even row split
    std::vector<int> evenCounts(P), evenDispls(P);
    for (int r = 0; r < P; ++r) {
        size_t b = n * r / P, e = n * (r + 1) / P;
        evenDispls[r] = static_cast<int>(b);
        evenCounts[r] = static_cast<int>(e - b);
    }
    std::vector<long long> rowWork(n, 0);
    const size_t evenBegin = evenDispls[g_rank];
    const size_t evenEnd = evenBegin + evenCounts[g_rank];
    for (size_t i = evenBegin; i < evenEnd; ++i) {
        const Triangle& triI = state.triangles[i];
        long long c = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!isCulled(triI, state.triangles[j])) ++c;
        }
        rowWork[i] = c;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rowWork.data(), evenCounts.data(),
                   evenDispls.data(), MPI_LONG_LONG, MPI_COMM_WORLD);

    // Balanced contiguous partition by cumulative work
    std::vector<double> prefix(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + static_cast<double>(rowWork[i] + 1);
    const double total = prefix[n];
    std::vector<size_t> bounds(P + 1, 0);
    bounds[P] = n;
    for (int r = 1; r < P; ++r) {
        double target = total * r / P;
        size_t k = static_cast<size_t>(std::lower_bound(prefix.begin(), prefix.end(), target) - prefix.begin());
        if (k > 0 && (target - prefix[k - 1]) < (prefix[k] - target)) --k;
        bounds[r] = std::max(bounds[r - 1], std::min(k, n));
    }
    state.rowCounts.resize(P);
    state.rowDispls.resize(P);
    for (int r = 0; r < P; ++r) {
        state.rowDispls[r] = static_cast<int>(bounds[r]);
        state.rowCounts[r] = static_cast<int>(bounds[r + 1] - bounds[r]);
    }
    state.rowBegin = bounds[g_rank];
    state.rowEnd = bounds[g_rank + 1];
    return rowWork;
}

// Computes the local rows of Tau and Kij. The random stream is consumed in exactly
// the same global (i, j) order as the sequential code: every rank jumps its generator
// ahead to the offset of its first row.
void computeFormFactorsAndDelays(SimulationState& state) {
    log0("Computing time delays (Tau)...\n");
    log0("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    std::vector<long long> rowWork = partitionRows(state);

    uint64_t pairsBefore = 0;
    for (size_t i = 0; i < state.rowBegin; ++i) pairsBefore += static_cast<uint64_t>(rowWork[i]);

    RandomGenerator rng(42);
    rng.advance(pairsBefore * 4 * NUM_RAYS);  // 2 points x 2 coordinates per ray

    const size_t localRows = state.rowEnd - state.rowBegin;
    state.rowPtr.assign(localRows + 1, 0);
    state.colIdx.clear();
    state.tauVal.clear();
    state.weight.clear();
    state.localNonZeroKij = 0;

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
            if (kij > EPSILON) ++state.localNonZeroKij;
            if (kij <= ZERO) continue;
            state.colIdx.push_back(static_cast<idx_t>(j));
            state.tauVal.push_back(computeTau(state.triangles[i], state.triangles[j]));
            state.weight.push_back(std::min(kij * state.areas[j], ONE));
        }
        state.rowPtr[i - state.rowBegin + 1] = state.colIdx.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    log0("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    log0("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const size_t* rowPtr = state.rowPtr.data();
    const idx_t* colIdx = state.colIdx.data();
    const int* tauVal = state.tauVal.data();
    const val_t* weight = state.weight.data();
    val_t* radB = state.radB.data();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);
        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            val_t sumB = ZERO;
            const size_t li = i - state.rowBegin;

            for (size_t k = rowPtr[li]; k < rowPtr[li + 1]; ++k) {
                int tauij = tauVal[k];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = radB[srcTime * n + colIdx[k]];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += weight[k] * radJ;
            }

            // Update radiosity: reflection + emission
            radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, radB + t * n,
                       state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            log0("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    log0("Computing distances via cross-correlation...\n");

    const size_t T = state.numTimesteps;
    std::vector<val_t> src(T), col(T);
    for (size_t t = 0; t < T; ++t) src[t] = state.radB[state.idxTN(t, state.sourceIndex)];

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t t = 0; t < T; ++t) col[t] = state.radB[state.idxTN(t, i)];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = col[tt];
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

    // Collect all distances on rank 0
    if (g_rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
                    state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(state.distances.data() + state.rowBegin, static_cast<int>(state.rowEnd - state.rowBegin),
                    MPI_FLOAT, nullptr, nullptr, nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, long long nonZeroKijTotal) {
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

int run(int argc, char** argv) {
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
            if (g_rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            log0("Unknown option: %s\n", argv[i]);
            if (g_rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    log0("Room Response Simulation Benchmark\n");
    log0("===================================\n");
    log0("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    log0("Timesteps: %d\n", timesteps);
    log0("Source triangle: %d\n", sourceIdx);
    log0("Reflectivity: %.2f\n", reflectivity);
    log0("Validation: %s\n", validate ? "enabled" : "disabled");
    log0("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    log0("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactorsAndDelays(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    log0("Precomputation time: %ld ms\n", preDuration);
    log0("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    log0("Simulation time: %ld ms\n", simDuration);
    log0("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    log0("Distance computation time: %ld ms\n", distDuration);
    log0("\n");

    long long nonZeroKij = 0;
    MPI_Reduce(&state.localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Everything below (reporting, hash, validation) happens on rank 0 only
    if (g_rank != 0) return 0;

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
        if (!validateResults(state, nonZeroKij)) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    int rc = run(argc, argv);

    MPI_Finalize();
    return rc;
}

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
// but stepping one word at a time so that the state can be jumped ahead in
// O(polylog) time. This lets every MPI rank start at its exact offset in the
// single sequential random stream used by the original code.
class MT19937 {
public:
    using result_type = uint32_t;
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr int MEXP = 19937;

    uint32_t x[N];
    int idx = 0;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return 0xffffffffu; }

    explicit MT19937(uint32_t seed = 5489u) {
        x[0] = seed;
        for (int i = 1; i < N; ++i) {
            x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        idx = 0;
    }

    // Advance the (linear) state by one word, returning the new raw word
    inline uint32_t step() {
        int i1 = idx + 1; if (i1 == N) i1 = 0;
        int im = idx + M; if (im >= N) im -= N;
        uint32_t y = (x[idx] & 0x80000000u) | (x[i1] & 0x7fffffffu);
        uint32_t v = x[im] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        x[idx] = v;
        idx = i1;
        return v;
    }

    inline result_type operator()() {
        uint32_t y = step();
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    void discard(uint64_t n) {
        for (uint64_t k = 0; k < n; ++k) step();
    }

    // Jump ahead by n steps (equivalent to discard(n))
    void jump(uint64_t n) {
        if (n < (uint64_t(1) << 22)) { discard(n); return; }
        const Poly& p = charPoly();
        Poly g = xPowMod(n, p);

        MT19937 acc(*this);
        std::memset(acc.x, 0, sizeof(acc.x));
        acc.idx = idx;
        for (int i = MEXP - 1; i >= 0; --i) {
            acc.step();
            if ((g[i >> 6] >> (i & 63)) & 1u) {
                int a = acc.idx, b = idx;
                for (int k = 0; k < N; ++k) {
                    acc.x[a] ^= x[b];
                    if (++a == N) a = 0;
                    if (++b == N) b = 0;
                }
            }
        }
        std::memcpy(x, acc.x, sizeof(x));
        idx = acc.idx;
    }

private:
    using Poly = std::vector<uint64_t>;
    static constexpr int PW = (MEXP + 63) / 64;   // words for polys of degree < MEXP

    // Characteristic polynomial of the MT19937 recurrence (degree MEXP),
    // obtained by Berlekamp-Massey on a bit sequence of the generator.
    static const Poly& charPoly() {
        static Poly poly = [] {
            const int NS = 2 * MEXP + 64;
            const int W = NS / 64 + 4;
            // R holds the sequence reversed: bit (NS-1-k) = s_k
            Poly R(W + 2, 0);
            MT19937 g(1u);
            for (int k = 0; k < NS; ++k) {
                if (g.step() & 1u) {
                    int pos = NS - 1 - k;
                    R[pos >> 6] |= uint64_t(1) << (pos & 63);
                }
            }
            auto bits64 = [&](int pos) -> uint64_t {
                int w = pos >> 6, s = pos & 63;
                uint64_t v = R[w] >> s;
                if (s) v |= R[w + 1] << (64 - s);
                return v;
            };
            Poly C(W, 0), B(W, 0), T;
            C[0] = 1; B[0] = 1;
            int L = 0, m = 1;
            auto xorShifted = [&](Poly& dst, const Poly& src, int sh) {
                int ws = sh >> 6, bs = sh & 63;
                for (int w = W - 1 - ws; w >= 0; --w) {
                    if (!src[w]) continue;
                    dst[w + ws] ^= src[w] << bs;
                    if (bs && w + ws + 1 < W) dst[w + ws + 1] ^= src[w] >> (64 - bs);
                }
            };
            for (int n = 0; n < NS; ++n) {
                int o = NS - 1 - n;
                int lw = L >> 6;
                uint64_t acc = 0;
                for (int w = 0; w <= lw; ++w) {
                    uint64_t cw = C[w];
                    if (w == lw) {
                        int rem = (L & 63) + 1;
                        if (rem < 64) cw &= (uint64_t(1) << rem) - 1;
                    }
                    acc ^= cw & bits64(o + 64 * w);
                }
                int d = __builtin_popcountll(acc) & 1;
                if (!d) { ++m; continue; }
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
            }
            // p(x) = x^L C(1/x); expect L == MEXP
            Poly p(PW + 1, 0);
            for (int k = 0; k <= L; ++k) {
                int ci = L - k;
                if ((C[ci >> 6] >> (ci & 63)) & 1u) p[k >> 6] |= uint64_t(1) << (k & 63);
            }
            if (L != MEXP) {
                fprintf(stderr, "MT19937 jump: unexpected polynomial degree %d\n", L);
                std::abort();
            }
            return p;
        }();
        return poly;
    }

    // Compute x^n mod p over GF(2)
    static Poly xPowMod(uint64_t n, const Poly& p) {
        // Precompute p shifted by 0..63 bits (only bits below MEXP needed for reduction)
        const int SW = PW + 2;
        std::vector<Poly> psh(64, Poly(SW, 0));
        for (int s = 0; s < 64; ++s) {
            for (int w = 0; w <= PW; ++w) {
                psh[s][w] ^= p[w] << s;
                if (s && w + 1 < SW) psh[s][w + 1] ^= p[w] >> (64 - s);
            }
        }
        Poly r(PW, 0), sq(2 * PW + 2, 0);
        r[0] = 1;
        auto reduce = [&](Poly& a, int topBit) {
            for (int k = topBit; k >= MEXP; --k) {
                if (!((a[k >> 6] >> (k & 63)) & 1u)) continue;
                int sh = k - MEXP;
                int ws = sh >> 6;
                const Poly& ps = psh[sh & 63];
                for (int w = 0; w < SW && w + ws < static_cast<int>(a.size()); ++w) a[w + ws] ^= ps[w];
            }
        };
        auto spread = [](uint32_t v) -> uint64_t {
            uint64_t z = v;
            z = (z | (z << 16)) & 0x0000FFFF0000FFFFull;
            z = (z | (z << 8)) & 0x00FF00FF00FF00FFull;
            z = (z | (z << 4)) & 0x0F0F0F0F0F0F0F0Full;
            z = (z | (z << 2)) & 0x3333333333333333ull;
            z = (z | (z << 1)) & 0x5555555555555555ull;
            return z;
        };
        int top = 63 - __builtin_clzll(n);
        for (int b = top; b >= 0; --b) {
            // square
            std::fill(sq.begin(), sq.end(), 0);
            for (int w = 0; w < PW; ++w) {
                sq[2 * w] = spread(static_cast<uint32_t>(r[w]));
                sq[2 * w + 1] = spread(static_cast<uint32_t>(r[w] >> 32));
            }
            reduce(sq, 2 * MEXP - 2);
            for (int w = 0; w < PW; ++w) r[w] = sq[w];
            // multiply by x
            if ((n >> b) & 1u) {
                std::fill(sq.begin(), sq.end(), 0);
                for (int w = 0; w < PW; ++w) {
                    sq[w] |= r[w] << 1;
                    sq[w + 1] |= r[w] >> 63;
                }
                reduce(sq, MEXP);
                for (int w = 0; w < PW; ++w) r[w] = sq[w];
            }
        }
        return r;
    }
};

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void skip(uint64_t numDraws) { rng.jump(numDraws); }
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
    std::vector<val_t> distances;   // Computed distances from source (gathered on rank 0)

    // MPI decomposition: each rank owns a contiguous block of receiver rows i.
    int rank = 0;
    int nprocs = 1;
    std::vector<int> rowStart;      // nprocs + 1 entries
    size_t rowBegin = 0, rowEnd = 0;

    // Local rows of the form factor / time delay matrices, stored sparsely
    // (only entries with Kij > 0, which are the only ones that contribute).
    std::vector<size_t> rowPtr;     // (rowEnd - rowBegin) + 1 entries
    std::vector<idx_t> colIdx;      // emitter triangle j
    std::vector<int> colTau;        // Tau(i, j)
    std::vector<val_t> colWeight;   // min(Kij * area_j, 1)
    long long nonZeroKij = 0;       // global count of Kij > EPSILON

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Uniform block partition of n items over p parts
static void blockPartition(size_t n, int p, std::vector<int>& start) {
    start.resize(p + 1);
    for (int r = 0; r <= p; ++r) start[r] = static_cast<int>((n * static_cast<size_t>(r)) / p);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.rank == 0;

    // Generate mesh (replicated on every rank; cheap compared to Kij)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
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

// Number of random draws consumed by computeKij for a non-culled pair
constexpr uint64_t DRAWS_PER_PAIR = 4 * NUM_RAYS;

static inline bool pairCulled(const Triangle& a, const Triangle& b) {
    return a.normal().dot(b.normal()) > 0.99f;
}

// Decide the row decomposition. The cost of a row is dominated by the
// number of non-culled pairs (each traces NUM_RAYS visibility rays), so rows
// are split into contiguous blocks of approximately equal cost. Also returns
// the number of non-culled pairs per row, needed to locate each rank's
// starting position in the sequential random stream.
static void partitionRows(SimulationState& state, std::vector<int>& pairsPerRow) {
    const size_t n = state.numTriangles;
    const int P = state.nprocs;

    std::vector<int> ub;
    blockPartition(n, P, ub);
    std::vector<int> localCounts(ub[state.rank + 1] - ub[state.rank]);
    for (int i = ub[state.rank]; i < ub[state.rank + 1]; ++i) {
        int cnt = 0;
        const Triangle& ti = state.triangles[i];
        for (size_t j = 0; j < n; ++j) {
            if (static_cast<size_t>(i) == j) continue;
            if (!pairCulled(ti, state.triangles[j])) ++cnt;
        }
        localCounts[i - ub[state.rank]] = cnt;
    }

    std::vector<int> recvCounts(P), displs(P);
    for (int r = 0; r < P; ++r) {
        recvCounts[r] = ub[r + 1] - ub[r];
        displs[r] = ub[r];
    }
    pairsPerRow.assign(n, 0);
    MPI_Allgatherv(localCounts.data(), static_cast<int>(localCounts.size()), MPI_INT,
                   pairsPerRow.data(), recvCounts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

    // Cost model: rays dominate; add a small per-row term for the j loop
    std::vector<double> prefix(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + static_cast<double>(pairsPerRow[i]) * NUM_RAYS +
                        0.05 * static_cast<double>(n);
    }
    const double total = prefix[n];
    state.rowStart.assign(P + 1, 0);
    state.rowStart[P] = static_cast<int>(n);
    size_t i = 0;
    for (int r = 1; r < P; ++r) {
        double target = total * r / P;
        while (i < n && prefix[i] < target) ++i;
        state.rowStart[r] = static_cast<int>(i);
    }
    state.rowBegin = state.rowStart[state.rank];
    state.rowEnd = state.rowStart[state.rank + 1];
}

void computeTimeDelays(SimulationState& state) {
    // Tau is computed alongside the form factors for each rank's own rows,
    // and only stored for the (i, j) pairs that can carry energy (Kij > 0).
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    std::vector<int> pairsPerRow;
    partitionRows(state, pairsPerRow);

    // Position this rank's generator exactly where the sequential code's
    // generator would be when starting row rowBegin.
    uint64_t drawOffset = 0;
    for (size_t i = 0; i < state.rowBegin; ++i) {
        drawOffset += static_cast<uint64_t>(pairsPerRow[i]) * DRAWS_PER_PAIR;
    }
    RandomGenerator rng(42);
    rng.skip(drawOffset);

    const size_t n = state.numTriangles;
    const size_t numLocal = state.rowEnd - state.rowBegin;
    state.rowPtr.assign(numLocal + 1, 0);
    state.colIdx.clear();
    state.colTau.clear();
    state.colWeight.clear();
    long long localNonZero = 0;

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
            if (kij > EPSILON) ++localNonZero;
            if (kij > ZERO) {
                state.colIdx.push_back(static_cast<idx_t>(j));
                state.colTau.push_back(computeTau(triI, state.triangles[j]));
                state.colWeight.push_back(std::min(kij * state.areas[j], ONE));
            }
        }
        state.rowPtr[i - state.rowBegin + 1] = state.colIdx.size();
    }

    MPI_Allreduce(&localNonZero, &state.nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (state.rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const int P = state.nprocs;
    std::vector<int> recvCounts(P), displs(P);
    for (int r = 0; r < P; ++r) {
        recvCounts[r] = state.rowStart[r + 1] - state.rowStart[r];
        displs[r] = state.rowStart[r];
    }

    const idx_t* colIdx = state.colIdx.data();
    const int* colTau = state.colTau.data();
    const val_t* colWeight = state.colWeight.data();
    const val_t* radB = state.radB.data();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);
        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            val_t sumB = ZERO;

            const size_t kBeg = state.rowPtr[i - state.rowBegin];
            const size_t kEnd = state.rowPtr[i - state.rowBegin + 1];
            for (size_t k = kBeg; k < kEnd; ++k) {
                int tauij = colTau[k];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = radB[srcTime * n + colIdx[k]];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += colWeight[k] * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radiosities with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &state.radB[state.idxTN(t, 0)],
                       recvCounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

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
    const int P = state.nprocs;
    std::vector<int> part;
    blockPartition(n, P, part);
    const size_t begin = part[state.rank], end = part[state.rank + 1];

    std::vector<val_t> srcSeries(T), series(T);
    for (size_t t = 0; t < T; ++t) srcSeries[t] = state.radB[state.idxTN(t, state.sourceIndex)];

    std::vector<val_t> localDist(end - begin);
    for (size_t i = begin; i < end; ++i) {
        for (size_t t = 0; t < T; ++t) series[t] = state.radB[state.idxTN(t, i)];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                sum += srcSeries[tt - t] * series[tt];
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDist[i - begin] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    std::vector<int> recvCounts(P), displs(P);
    for (int r = 0; r < P; ++r) {
        recvCounts[r] = part[r + 1] - part[r];
        displs[r] = part[r];
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_FLOAT,
                state.distances.data(), recvCounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
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

static int finish(int code) {
    MPI_Finalize();
    return code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = rank == 0;

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
            if (root) printUsage(argv[0]);
            return finish(0);
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nprocs = nprocs;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    long localTotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTotalTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!root) return finish(0);

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

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
        if (!validateResults(state)) {
            return finish(1);
        }
    }

    return finish(0);
}

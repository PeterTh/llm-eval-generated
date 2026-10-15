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
#include <mpi.h>
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

// MT19937 (bit-identical to std::mt19937 + std::uniform_real_distribution<float>)
// with O(polynomial) jump-ahead so each MPI rank can start at its exact position
// in the original single sequential random stream.
class RandomGenerator {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr int DEG = 19937;
    uint32_t mt[N];
    int p;

    uint32_t step() {
        int p1 = (p + 1 == N) ? 0 : p + 1;
        int pm = (p + M >= N) ? p + M - N : p + M;
        uint32_t y = (mt[p] & 0x80000000u) | (mt[p1] & 0x7fffffffu);
        uint32_t v = mt[pm] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        mt[p] = v;
        p = p1;
        return v;
    }

    static uint32_t temper(uint32_t y) {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    // Characteristic polynomial of the generator (found via Berlekamp-Massey)
    struct CharPoly {
        bool valid = false;
        std::vector<std::vector<uint64_t>> shifted;  // shifted[b] = P << b
    };

    static CharPoly& charPoly() {
        static CharPoly cp;
        static bool built = false;
        if (built) return cp;
        built = true;

        const int nbits = 2 * DEG + 128;
        RandomGenerator g(5489);
        std::vector<uint8_t> s(nbits);
        for (int i = 0; i < nbits; ++i) s[i] = static_cast<uint8_t>(g.step() >> 31);

        const int W = (DEG + 64) / 64 + 2;  // words for C(x), B(x)
        std::vector<uint64_t> C(W, 0), B(W, 0), T(W), rs(W, 0);
        C[0] = 1;
        B[0] = 1;
        int L = 0, m = -1;
        for (int n = 0; n < nbits; ++n) {
            // rs bit i = s[n-i]
            for (int i = W - 1; i > 0; --i) rs[i] = (rs[i] << 1) | (rs[i - 1] >> 63);
            rs[0] = (rs[0] << 1) | s[n];
            uint64_t acc = 0;
            for (int i = 0; i < W; ++i) acc ^= C[i] & rs[i];
            if (__builtin_parityll(acc)) {
                T = C;
                int sh = n - m;
                int ws = sh >> 6, bs = sh & 63;
                for (int i = W - 1; i >= ws; --i) {
                    uint64_t v = B[i - ws] << bs;
                    if (bs && i - ws > 0) v |= B[i - ws - 1] >> (64 - bs);
                    C[i] ^= v;
                }
                if (2 * L <= n) {
                    L = n + 1 - L;
                    B = T;
                    m = n;
                }
            }
        }
        if (L != DEG) return cp;  // unexpected; fall back to discarding

        std::vector<uint64_t> P(W, 0);
        for (int i = 0; i <= L; ++i)
            if ((C[i >> 6] >> (i & 63)) & 1) {
                int k = L - i;
                P[k >> 6] |= 1ULL << (k & 63);
            }
        cp.shifted.assign(64, std::vector<uint64_t>(W + 1, 0));
        for (int b = 0; b < 64; ++b) {
            for (int i = 0; i < W; ++i) {
                cp.shifted[b][i] |= P[i] << b;
                if (b) cp.shifted[b][i + 1] |= P[i] >> (64 - b);
            }
        }
        cp.valid = true;
        return cp;
    }

    static uint64_t spread32(uint64_t x) {
        x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
        x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
        x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
        x = (x | (x << 2)) & 0x3333333333333333ULL;
        x = (x | (x << 1)) & 0x5555555555555555ULL;
        return x;
    }

    // g = x^n mod P
    static std::vector<uint64_t> powX(const CharPoly& cp, uint64_t n) {
        const int GW = (DEG + 63) / 64;          // words of a reduced polynomial
        const int RW = 2 * GW + 2;               // words of an unreduced square
        std::vector<uint64_t> g(GW + 1, 0), r(RW);
        g[0] = 1;
        const std::vector<uint64_t>& P0 = cp.shifted[0];
        for (int bit = 63; bit >= 0; --bit) {
            if (!(n >> bit)) continue;  // skip leading zeros (g == 1)
            // square
            std::fill(r.begin(), r.end(), 0);
            for (int i = 0; i < GW; ++i) {
                r[2 * i] = spread32(g[i] & 0xffffffffULL);
                r[2 * i + 1] = spread32(g[i] >> 32);
            }
            // reduce
            for (int pos = 2 * DEG - 2; pos >= DEG; --pos) {
                if ((r[pos >> 6] >> (pos & 63)) & 1) {
                    int sh = pos - DEG;
                    const std::vector<uint64_t>& ps = cp.shifted[sh & 63];
                    uint64_t* dst = r.data() + (sh >> 6);
                    int len = ((DEG + (sh & 63)) >> 6) + 1;
                    for (int i = 0; i < len; ++i) dst[i] ^= ps[i];
                }
            }
            for (int i = 0; i < GW + 1; ++i) g[i] = r[i];
            if ((n >> bit) & 1ULL) {  // multiply by x
                uint64_t carry = 0;
                for (int i = 0; i < GW + 1; ++i) {
                    uint64_t nc = g[i] >> 63;
                    g[i] = (g[i] << 1) | carry;
                    carry = nc;
                }
                if ((g[DEG >> 6] >> (DEG & 63)) & 1)
                    for (int i = 0; i < GW + 1; ++i) g[i] ^= P0[i];
            }
        }
        return g;
    }

    void jump(const std::vector<uint64_t>& g) {
        uint32_t acc[N] = {0};
        for (int k = 0; k < DEG; ++k) {
            if ((g[k >> 6] >> (k & 63)) & 1) {
                int idx = p;
                for (int i = 0; i < N; ++i) {
                    acc[i] ^= mt[idx];
                    if (++idx == N) idx = 0;
                }
            }
            step();
        }
        std::memcpy(mt, acc, sizeof(mt));
        p = 0;
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) : p(0) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }

    val_t rand() {
        val_t r = static_cast<val_t>(temper(step())) / 4294967296.0f;
        if (r >= 1.0f) r = std::nextafter(1.0f, 0.0f);
        return r;
    }

    // Advance the stream by n draws
    void skip(uint64_t n) {
        if (n < 4000000ULL) {
            for (uint64_t i = 0; i < n; ++i) step();
            return;
        }
        CharPoly& cp = charPoly();
        if (!cp.valid) {
            for (uint64_t i = 0; i < n; ++i) step();
            return;
        }
        jump(powX(cp, n));
    }
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

// Triangles facing the same direction are culled (no random numbers consumed)
inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

constexpr uint64_t DRAWS_PER_PAIR = 4 * NUM_RAYS;  // 2 points x 2 uniform draws per ray

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (isCulled(triI, triJ)) return ZERO;

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
// Work distribution
// ============================================================================

struct Range {
    size_t begin, end;
};

// Rows are split into 2*P chunks; rank r owns chunk r and chunk 2P-1-r (snake),
// both in ascending order so the original random stream is consumed in order.
std::array<Range, 2> rankRanges(size_t n, int P, int r) {
    size_t nch = 2 * static_cast<size_t>(P);
    auto chunk = [&](size_t c) { return Range{n * c / nch, n * (c + 1) / nch}; };
    return {chunk(static_cast<size_t>(r)), chunk(nch - 1 - static_cast<size_t>(r))};
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    int rank = 0, size = 1;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)

    // Locally owned rows, sparse: only entries with Kij > 0, ascending j
    std::vector<size_t> localRows;  // global row index of each local row
    std::vector<size_t> rowStart;   // CSR offsets (localRows.size()+1)
    std::vector<idx_t> colJ;
    std::vector<int> colTau;
    std::vector<val_t> colW;        // min(Kij * area[j], 1)
    uint64_t localNonZeroKij = 0;   // entries with Kij > EPSILON

    // Allgatherv layout
    std::vector<int> counts, displs;
    std::vector<idx_t> order;       // gathered position -> global row

    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex = 0;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Gather each rank's local values (one per owned row) into natural row order
void gatherRows(const SimulationState& state, const val_t* local, val_t* dest,
                std::vector<val_t>& tmp) {
    MPI_Allgatherv(local, static_cast<int>(state.localRows.size()), MPI_FLOAT,
                   tmp.data(), state.counts.data(), state.displs.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
    for (size_t k = 0; k < state.numTriangles; ++k) dest[state.order[k]] = tmp[k];
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.size);

    // Generate mesh (replicated on every rank)
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

    // Row ownership and gather layout
    for (int r = 0; r < state.size; ++r) {
        size_t cnt = 0;
        for (const Range& rg : rankRanges(state.numTriangles, state.size, r)) {
            for (size_t i = rg.begin; i < rg.end; ++i) {
                state.order.push_back(static_cast<idx_t>(i));
                if (r == state.rank) state.localRows.push_back(i);
            }
            cnt += rg.end - rg.begin;
        }
        state.counts.push_back(static_cast<int>(cnt));
    }
    state.displs.assign(state.size, 0);
    for (int r = 1; r < state.size; ++r) state.displs[r] = state.displs[r - 1] + state.counts[r - 1];

    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, val_t reflectivity) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\nComputing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const int P = state.size;

    state.rho.assign(n, reflectivity);

    // Number of random draws consumed by each row in the original sequential stream
    std::vector<uint64_t> rowDraws(n, 0);
    {
        size_t b = n * static_cast<size_t>(state.rank) / P;
        size_t e = n * (static_cast<size_t>(state.rank) + 1) / P;
        for (size_t i = b; i < e; ++i) {
            uint64_t c = 0;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                if (!isCulled(state.triangles[i], state.triangles[j])) ++c;
            }
            rowDraws[i] = c * DRAWS_PER_PAIR;
        }
        MPI_Allreduce(MPI_IN_PLACE, rowDraws.data(), static_cast<int>(n), MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    RandomGenerator rng(42);
    uint64_t pos = 0;  // draws consumed so far by rng
    uint64_t prefix = 0;
    size_t nextRow = 0;

    state.rowStart.push_back(0);
    for (const Range& rg : rankRanges(n, P, state.rank)) {
        for (; nextRow < rg.begin; ++nextRow) prefix += rowDraws[nextRow];
        if (prefix > pos) {
            rng.skip(prefix - pos);
            pos = prefix;
        }
        for (size_t i = rg.begin; i < rg.end; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                val_t k = computeKij(i, j, state.triangles, state.octree, rng);
                if (k > EPSILON) ++state.localNonZeroKij;
                if (k <= ZERO) continue;
                state.colJ.push_back(static_cast<idx_t>(j));
                state.colTau.push_back(computeTau(state.triangles[i], state.triangles[j]));
                state.colW.push_back(std::min(k * state.areas[j], ONE));
            }
            state.rowStart.push_back(state.colJ.size());
            pos += rowDraws[i];
            prefix += rowDraws[i];
            nextRow = i + 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (state.rank == 0) printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const size_t timeOff = state.numTimesteps / 2;
    const size_t nLocal = state.localRows.size();
    std::vector<val_t> local(nLocal), tmp(n);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        for (size_t r = 0; r < nLocal; ++r) {
            const size_t i = state.localRows[r];
            val_t sumB = ZERO;

            const size_t e = state.rowStart[r + 1];
            for (size_t k = state.rowStart[r]; k < e; ++k) {
                int tauij = state.colTau[k];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[srcTime * n + state.colJ[k]];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += state.colW[k] * radJ;
            }

            // Update radiosity: reflection + emission
            val_t emit = (i == state.sourceIndex && t < timeOff) ? 1.0f : ZERO;
            local[r] = state.rho[i] * sumB + emit;
        }

        gatherRows(state, local.data(), &state.radB[t * n], tmp);

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
    const size_t nLocal = state.localRows.size();

    // Contiguous copy of the source triangle's time series
    std::vector<val_t> src(T), col(T);
    for (size_t t = 0; t < T; ++t) src[t] = state.radB[t * n + state.sourceIndex];

    std::vector<val_t> local(nLocal), tmp(n);
    for (size_t r = 0; r < nLocal; ++r) {
        const size_t i = state.localRows[r];
        for (size_t t = 0; t < T; ++t) col[t] = state.radB[t * n + i];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < T; ++tt) {
                sum += src[tt - t] * col[tt];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        local[r] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    gatherRows(state, local.data(), state.distances.data(), tmp);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, uint64_t nonZeroKij) {
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
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           static_cast<int>(nonZeroKij), state.numTriangles * state.numTriangles,
           100.0f * static_cast<int>(nonZeroKij) / static_cast<val_t>(state.numTriangles * state.numTriangles));

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
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool root = worldRank == 0;

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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
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
        printf("MPI ranks: %d\n", [] { int s; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx));

    if (root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state, reflectivity);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (root) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    uint64_t nonZeroKij = 0;
    MPI_Reduce(&state.localNonZeroKij, &nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    long localDurations[3] = {preDuration, simDuration, distDuration};
    long maxDurations[3];
    MPI_Reduce(localDurations, maxDurations, 3, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (root) {
        // Total time
        long totalTime = maxDurations[0] + maxDurations[1] + maxDurations[2];
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

        // Memory usage (equivalent sequential footprint)
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
            if (!validateResults(state, nonZeroKij)) rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}

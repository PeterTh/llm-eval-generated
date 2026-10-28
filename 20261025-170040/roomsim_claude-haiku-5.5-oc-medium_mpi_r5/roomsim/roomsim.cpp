/**
 * Room Response Simulation Benchmark (MPI, distributed memory)
 * 
 * This is a simplified implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization (MPI):
 *  - Triangles are partitioned into contiguous blocks of "receiver" rows i.
 *    Each rank owns the Kij and Tau rows of its block and computes them.
 *  - Form factors consume one global pseudo-random stream in (i, j) order.
 *    Each rank computes the number of random draws consumed by the rows before
 *    its block (MPI_Exscan) and advances its generator to that exact position
 *    (exact jump-ahead for std::mt19937), so the results are identical to the
 *    sequential program for any number of ranks.
 *  - Wave propagation is advanced one timestep at a time. Each rank updates the
 *    radiosity of its own receivers and the full radiosity vector of the step is
 *    then exchanged with MPI_Allgatherv. Tau >= 1 for distinct triangles, so a
 *    step only reads data from earlier steps.
 *  - Distances are computed per owned triangle and gathered on all ranks.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
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

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

static_assert(sizeof(val_t) == sizeof(float), "MPI transfers assume val_t is float");

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// Random draws consumed per (non-culled) form factor pair: NUM_RAYS rays, each
// drawing two barycentric samples for each of the two triangles, one 32-bit
// engine output per draw.
constexpr uint64_t RNG_DRAWS_PER_PAIR = static_cast<uint64_t>(NUM_RAYS) * 4;

// ============================================================================
// MPI Context
// ============================================================================

static int g_rank = 0;  // rank of this process in MPI_COMM_WORLD

// printf restricted to rank 0 so that the output is produced once
__attribute__((format(printf, 1, 2)))
static void rprintf(const char* fmt, ...) {
    if (g_rank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

// Wall-clock duration of a collective phase in ms (maximum over ranks)
static long maxPhaseMs(double startSec, double endSec) {
    double d = endSec - startSec;
    MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return static_cast<long>(d * 1000.0);
}

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

// Exact jump-ahead for std::mt19937.
//
// The MT19937 state recurrence is linear over GF(2), so the engine state after n
// outputs is obtained from the characteristic polynomial phi of the recurrence:
//   x^n mod phi = sum_i p_i x^i   =>   W_n = sum_i p_i W_i
// where W_i is the 624-word window starting at output i. phi is recovered from the
// output sequence with Berlekamp-Massey (linear complexity 19937 for MT19937).
// The result is bit-identical to calling discard(n).
namespace mtjump {

constexpr int DEG = 19937;                  // degree of the MT19937 recurrence
constexpr int MT_N = 624, MT_M = 397;
constexpr size_t PW = (DEG + 63) / 64;      // words in a polynomial of degree < DEG
constexpr uint32_t UPPER_MASK = 0x80000000u, LOWER_MASK = 0x7fffffffu, MATRIX_A = 0x9908b0dfu;

inline uint32_t twist(uint32_t a, uint32_t b) {
    uint32_t y = (a & UPPER_MASK) | (b & LOWER_MASK);
    return (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
}

// Berlekamp-Massey over GF(2). Returns the connection polynomial C (C[0] = 1)
// and sets L to the linear complexity of s.
inline std::vector<uint8_t> berlekampMassey(const std::vector<uint8_t>& s, int& L) {
    const size_t n = s.size();
    std::vector<uint8_t> C(n + 1, 0), B(n + 1, 0), T;
    C[0] = 1;
    B[0] = 1;
    L = 0;
    int LB = 0;  // B[j] == 0 for j > LB
    size_t m = 1;
    for (size_t i = 0; i < n; ++i) {
        uint8_t d = s[i];
        for (int j = 1; j <= L; ++j) d ^= C[j] & s[i - j];
        if (!d) { ++m; continue; }
        const bool grow = 2 * L <= static_cast<long>(i);
        if (grow) T.assign(C.begin(), C.begin() + L + 1);
        for (int j = 0; j <= LB && j + m <= n; ++j) C[j + m] ^= B[j];
        if (grow) {
            const int newL = static_cast<int>(i + 1) - L;
            std::fill(B.begin(), B.begin() + LB + 1, 0);
            std::copy(T.begin(), T.end(), B.begin());
            LB = L;
            L = newL;
            m = 1;
        } else {
            ++m;
        }
    }
    C.resize(L + 1);
    return C;
}

using Poly = std::vector<uint64_t>;

// dst ^= src << shift (dst must be large enough)
inline void xorShift(uint64_t* dst, const uint64_t* src, size_t srcWords, size_t shift) {
    const size_t ws = shift >> 6;
    const unsigned bs = shift & 63;
    for (size_t w = 0; w < srcWords; ++w) {
        const uint64_t v = src[w];
        if (!v) continue;
        dst[w + ws] ^= v << bs;
        if (bs) dst[w + ws + 1] ^= v >> (64 - bs);
    }
}

// Arithmetic modulo phi, where x^DEG == R (mod phi)
struct JumpPolyRing {
    Poly R;  // R has bits < DEG

    // Reduce t (at least 2*PW+2 words) modulo phi; result has bits < DEG
    void reduce(Poly& t) const {
        for (long d = static_cast<long>(t.size()) * 64 - 1; d >= DEG; --d) {
            if ((t[d >> 6] >> (d & 63)) & 1ull) {
                t[d >> 6] ^= 1ull << (d & 63);
                xorShift(t.data(), R.data(), PW, static_cast<size_t>(d - DEG));
            }
        }
    }

    Poly square(const Poly& a) const {
        Poly t(2 * PW + 2, 0);
        for (size_t w = 0; w < PW; ++w) {
            uint64_t v = a[w];
            while (v) {
                const size_t i = w * 64 + static_cast<size_t>(std::countr_zero(v));
                v &= v - 1;
                t[(2 * i) >> 6] |= 1ull << ((2 * i) & 63);
            }
        }
        reduce(t);
        t.resize(PW);
        return t;
    }

    // a = a * x mod phi
    void mulX(Poly& a) const {
        const bool carry = (a[(DEG - 1) >> 6] >> ((DEG - 1) & 63)) & 1ull;
        for (size_t w = PW - 1; w > 0; --w) a[w] = (a[w] << 1) | (a[w - 1] >> 63);
        a[0] <<= 1;
        const unsigned topBit = (DEG - 1) & 63;
        a[PW - 1] &= ~0ull >> (63 - topBit);  // clear bit DEG and above
        if (carry)
            for (size_t w = 0; w < PW; ++w) a[w] ^= R[w];
    }

    // x^k mod phi
    Poly xpow(uint64_t k) const {
        Poly r(PW, 0);
        r[0] = 1;
        if (k == 0) return r;
        mulX(r);  // handles the leading bit of k
        for (int b = std::bit_width(k) - 2; b >= 0; --b) {
            r = square(r);
            if ((k >> b) & 1ull) mulX(r);
        }
        return r;
    }
};

// Returns the engine state equal to std::mt19937(seed) after n outputs.
// Returns false if the recurrence analysis does not yield degree DEG.
inline bool engineAt(uint32_t seed, uint64_t n, std::mt19937& out) {
    // Bit 0 of the tempered output stream of the seeded engine
    const int numBits = 2 * DEG + 128;
    std::mt19937 gen(seed);
    std::vector<uint8_t> bits(numBits);
    for (int i = 0; i < numBits; ++i) bits[i] = static_cast<uint8_t>(gen() & 1u);

    int L = 0;
    const std::vector<uint8_t> C = berlekampMassey(bits, L);
    if (L != DEG) return false;

    // phi(x) = x^L + sum_{j<L} C[L-j] x^j, so x^L == sum_{j<L} C[L-j] x^j
    JumpPolyRing ring;
    ring.R.assign(PW, 0);
    for (int j = 0; j < L; ++j)
        if (C[L - j]) ring.R[j >> 6] |= 1ull << (j & 63);

    // Raw state words: after 624 outputs the engine holds w_0..w_623 (untempered)
    std::mt19937 base(seed);
    for (int i = 0; i < MT_N; ++i) base();
    std::ostringstream os;
    os << base;
    std::istringstream is(os.str());
    std::vector<uint32_t> w(DEG + MT_N);
    for (int i = 0; i < MT_N; ++i) is >> w[i];
    for (size_t m = MT_N; m < w.size(); ++m)
        w[m] = w[m - MT_N + MT_M] ^ twist(w[m - MT_N], w[m - MT_N + 1]);

    // Window at output n: W_n = sum_i p_i W_i with p = x^n mod phi
    const Poly p = ring.xpow(n);
    std::array<uint32_t, MT_N> window{};
    for (int i = 0; i < DEG; ++i) {
        if ((p[i >> 6] >> (i & 63)) & 1ull) {
            for (int j = 0; j < MT_N; ++j) window[j] ^= w[i + j];
        }
    }

    // Load the window as the engine's state with index 0
    std::ostringstream ss;
    for (int j = 0; j < MT_N; ++j) ss << window[j] << ' ';
    ss << 0;
    std::istringstream si(ss.str());
    si >> out;
    return true;
}

}  // namespace mtjump

class RandomGenerator {
    // Below this many outputs, discarding is cheaper than the jump-ahead analysis
    static constexpr uint64_t JUMP_THRESHOLD = 1ull << 27;

    uint32_t seed_;
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
    bool fresh_ = true;  // no outputs drawn yet (the jump is defined from the seed)
public:
    explicit RandomGenerator(uint32_t seed = 42) : seed_(seed), rng(seed), dist(0.0f, 1.0f) {}

    val_t rand() {
        fresh_ = false;
        return dist(rng);
    }

    // Advance the stream by n outputs; equivalent to n calls to rand()'s engine draws
    void skip(uint64_t n) {
        std::mt19937 jumped;
        if (fresh_ && n >= JUMP_THRESHOLD && mtjump::engineAt(seed_, n, jumped)) {
            rng = jumped;
        } else {
            rng.discard(n);
        }
        fresh_ = false;
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

// Triangles facing the same direction exchange no energy; such pairs are
// skipped and consume no random numbers.
inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

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
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    int rank = 0;
    int nranks = 1;
    // Rows [rowBegin, rowEnd) of the Kij/Tau matrices are owned by this rank.
    // Per-rank row counts and displacements for allgather operations.
    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    size_t rowBegin = 0;
    size_t rowEnd = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors, local rows (rowEnd-rowBegin) x N, row-major
    std::vector<int> tau;           // Time delays, local rows (rowEnd-rowBegin) x N, row-major
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t numLocalRows() const { return rowEnd - rowBegin; }
    // Local (row-owned) index of the matrix entry (i, j), i must be owned
    size_t idxLocal(size_t i, size_t j) const { return (i - rowBegin) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.nranks);

    // Generate mesh (identical on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    const size_t N = state.numTriangles;
    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration (identical on every rank)
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    // Block row partition of the receiver index space
    state.rowCounts.resize(state.nranks);
    state.rowDispls.resize(state.nranks);
    for (int r = 0; r < state.nranks; ++r) {
        const size_t b = N * static_cast<size_t>(r) / static_cast<size_t>(state.nranks);
        const size_t e = N * static_cast<size_t>(r + 1) / static_cast<size_t>(state.nranks);
        state.rowDispls[r] = static_cast<int>(b);
        state.rowCounts[r] = static_cast<int>(e - b);
    }
    state.rowBegin = static_cast<size_t>(state.rowDispls[state.rank]);
    state.rowEnd = state.rowBegin + static_cast<size_t>(state.rowCounts[state.rank]);

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    // Initialize matrices: form factors and time delays for owned rows only
    state.kij.assign(state.numLocalRows() * N, ZERO);
    state.tau.assign(state.numLocalRows() * N, 0);
    state.radE.assign(timesteps * N, ZERO);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

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
    rprintf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const size_t localRows = state.numLocalRows();

    // Number of random draws consumed by the rows owned by lower ranks: the
    // sequential program draws from one stream over all (i, j) pairs in order.
    uint64_t localPairs = 0;
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j && !isCulled(state.triangles[i], state.triangles[j])) ++localPairs;
        }
    }
    uint64_t pairsBefore = 0;
    MPI_Exscan(&localPairs, &pairsBefore, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (state.rank == 0) pairsBefore = 0;  // result of MPI_Exscan is undefined on rank 0

    RandomGenerator rng(42);
    rng.skip(pairsBefore * RNG_DRAWS_PER_PAIR);

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.kij[state.idxLocal(i, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
        const size_t done = i - state.rowBegin + 1;
        if (done % 100 == 0 || done == localRows) {
            rprintf("  Progress (rank 0): %zu/%zu triangles\n", done, localRows);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.tau[state.idxLocal(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// Smallest time delay between distinct triangles (over all ranks)
int globalMinTau(const SimulationState& state) {
    const size_t N = state.numTriangles;
    int localMin = INT_MAX;
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j) localMin = std::min(localMin, state.tau[state.idxLocal(i, j)]);
        }
    }
    int globalMin = INT_MAX;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalMin;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Owned receivers: all sources j of step t-tau have been exchanged in earlier steps
        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            val_t sumB = ZERO;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;

                int tauij = state.tau[state.idxLocal(i, j)];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[state.idxLocal(i, j)];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Exchange this step's radiosity (each rank contributes its owned receivers)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       &state.radB[state.idxTN(t, 0)], state.rowCounts.data(),
                       state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

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

    // Share all distances with every rank
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), state.rowCounts.data(),
                   state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

// nonZeroKij: number of form factors above EPSILON over all ranks
bool validateResults(const SimulationState& state, long long nonZeroKij) {
    rprintf("\nValidation:\n");

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
            rprintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rprintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rprintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rprintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rprintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rprintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    rprintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rprintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const size_t numEntries = state.numTriangles * state.numTriangles;
    rprintf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
            nonZeroKij, numEntries,
            100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(numEntries));

    if (nonZeroKij == 0) {
        rprintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rprintf("  Validation: PASSED\n");
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
    rprintf("Usage: %s [options]\n", progName);
    rprintf("Options:\n");
    rprintf("  -n <num>     Target number of triangles (default: 320)\n");
    rprintf("               Actual count will be rounded to nearest icosphere level:\n");
    rprintf("               20, 80, 320, 1280, 5120, 20480\n");
    rprintf("  -t <num>     Number of timesteps (default: 50)\n");
    rprintf("  -s <num>     Source triangle index (default: 0)\n");
    rprintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rprintf("  -v           Enable validation\n");
    rprintf("  -o           Print results for external validation\n");
    rprintf("  -h           Show this help message\n");
}

static int runBenchmark(int argc, char** argv) {
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
            rprintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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
    double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state);

    double endPre = MPI_Wtime();
    long preDuration = maxPhaseMs(startPre, endPre);

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // The parallel wave propagation requires every pair of distinct triangles to
    // have a delay of at least one timestep (true for any icosphere used here).
    const int minTau = globalMinTau(state);
    if (minTau < 1) {
        rprintf("ERROR: zero time delay between distinct triangles; cannot propagate in parallel\n");
        return 1;
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = MPI_Wtime();

    runSimulation(state);

    double endSim = MPI_Wtime();
    long simDuration = maxPhaseMs(startSim, endSim);

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = MPI_Wtime();

    computeDistances(state);

    double endDist = MPI_Wtime();
    long distDuration = maxPhaseMs(startDist, endDist);

    rprintf("Distance computation time: %ld ms\n", distDuration);
    rprintf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    rprintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    rprintf("\nPerformance:\n");
    rprintf("  Triangles: %zu\n", n);
    rprintf("  Timesteps: %zu\n", t);
    rprintf("  Form factor computations: %.2e\n", kijOps);
    rprintf("  Simulation operations: %.2e\n", simOps);
    rprintf("  Distance computations: %.2e\n", distOps);
    rprintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage (total over all ranks, as for the sequential program)
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    rprintf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    rprintf("  Result hash: %016lX\n", hash);
    rprintf("\n");

    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        if (g_rank == 0) print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        // Form factors are distributed; count the non-zero entries over all ranks
        long long localNonZero = 0;
        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (state.kij[state.idxLocal(i, j)] > EPSILON) localNonZero++;
            }
        }
        long long nonZeroKij = 0;
        MPI_Allreduce(&localNonZero, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (!validateResults(state, nonZeroKij)) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

    const int rc = runBenchmark(argc, argv);

    MPI_Finalize();
    return rc;
}

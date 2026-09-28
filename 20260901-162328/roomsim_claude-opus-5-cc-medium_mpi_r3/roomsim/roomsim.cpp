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
 *
 * Parallelization (distributed memory, MPI):
 *   - The mesh and the octree are replicated on every rank (they are needed for
 *     ray casting everywhere and are cheap compared to the O(N^2) kernels).
 *   - The Kij and Tau matrices are distributed by rows: rank p owns a contiguous
 *     block of receiver triangles. Neither matrix is ever gathered, so the
 *     dominant memory cost scales as O(N^2 / P).
 *   - The wave propagation is distributed over the same row blocks; a single
 *     Allgatherv of N floats per timestep republishes the new radiosity slice.
 *   - The cross-correlation is distributed over the same row blocks followed by
 *     one Allgatherv of the distances.
 *
 * Bit-exactness: the sequential code draws form-factor sample points from one
 * global std::mt19937 stream, so the random numbers a triangle pair consumes
 * depend on all preceding pairs. Each rank therefore fast-forwards its own
 * generator to the exact stream position its first row starts at, using a GF(2)
 * jump-ahead of the Mersenne twister (see class RandomGenerator). All floating
 * point reductions keep the sequential operand order, so results are identical
 * to the original program for any number of ranks.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// MPI helpers
// ============================================================================

static int mpiRank = 0;
static int mpiSize = 1;

// printf that only produces output on rank 0
__attribute__((format(printf, 1, 2)))
static void rprintf(const char* fmt, ...) {
    if (mpiRank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

// Contiguous block decomposition of n items over `size` ranks
static void blockRange(size_t n, int rank, int size, size_t& begin, size_t& count) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    begin = r * base + std::min(r, rem);
    count = base + (r < rem ? 1 : 0);
}

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

// Bit-exact replacement for std::mt19937 + std::uniform_real_distribution<float>(0,1)
// that additionally supports O(state^2) jump-ahead, so that a rank can position
// itself at an arbitrary offset of the global random stream without having to
// generate all the preceding numbers.
//
// The jump uses the classic polynomial method: the state transition f of MT19937
// is linear over GF(2) with characteristic polynomial phi of degree 19937, hence
// f^n = g(f) with g(x) = x^n mod phi(x), which is evaluated on the state by
// Horner's scheme. phi is recovered once with Berlekamp-Massey from an output
// bit sequence (MT19937 has maximal period, so phi is primitive and equals the
// minimal polynomial of any non-trivial output bit).
class RandomGenerator {
    static constexpr int NN = 624;
    static constexpr int MM = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER_MASK = 0x80000000u;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffu;
    static constexpr int DEG = 19937;                  // degree of phi
    static constexpr int PW = (DEG + 1 + 63) / 64;     // 64-bit words holding phi

    uint32_t mt[NN];
    int pos;

    static uint32_t temper(uint32_t y) {
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // --- GF(2) bit-vector helpers (little endian bit order inside each word) ---
    static bool getbit(const uint64_t* a, size_t i) { return (a[i >> 6] >> (i & 63)) & 1ull; }
    static void setbit(uint64_t* a, size_t i) { a[i >> 6] |= 1ull << (i & 63); }

    // dst ^= src << shift, src spanning nw words (dst must have nw + shift/64 + 1)
    static void xorShifted(uint64_t* dst, const uint64_t* src, size_t shift, int nw) {
        const size_t ws = shift >> 6;
        const unsigned bs = shift & 63;
        if (bs == 0) {
            for (int i = 0; i < nw; ++i) dst[ws + i] ^= src[i];
        } else {
            uint64_t carry = 0;
            for (int i = 0; i < nw; ++i) {
                const uint64_t v = src[i];
                dst[ws + i] ^= (v << bs) | carry;
                carry = v >> (64 - bs);
            }
            dst[ws + nw] ^= carry;
        }
    }

    // Minimal polynomial of the bit sequence given in reversed order (srev[M-1-k] = s_k)
    static void berlekampMassey(const uint64_t* srev, size_t M, uint64_t* phi) {
        const int CW = PW + 2;
        std::vector<uint64_t> c(CW, 0), b(CW, 0), t(CW, 0);
        c[0] = 1;
        b[0] = 1;
        int L = 0;
        size_t m = 1;
        for (size_t n = 0; n < M; ++n) {
            // discrepancy = XOR_{i=0..L} c_i * s_{n-i}, and s_{n-i} = srev[M-1-n+i]
            const size_t off = M - 1 - n;
            const size_t ws = off >> 6;
            const unsigned bs = off & 63;
            const int nw = (L >> 6) + 1;
            uint64_t acc = 0;
            if (bs == 0) {
                for (int i = 0; i < nw; ++i) acc ^= c[i] & srev[ws + i];
            } else {
                for (int i = 0; i < nw; ++i) {
                    const uint64_t v = (srev[ws + i] >> bs) | (srev[ws + i + 1] << (64 - bs));
                    acc ^= c[i] & v;
                }
            }
            const int nwShift = CW - 1 - static_cast<int>(m >> 6);
            if ((__builtin_popcountll(acc) & 1) == 0) {
                ++m;
            } else if (2 * L <= static_cast<int>(n)) {
                t = c;
                xorShifted(c.data(), b.data(), m, nwShift);
                L = static_cast<int>(n) + 1 - L;
                b = t;
                m = 1;
            } else {
                xorShifted(c.data(), b.data(), m, nwShift);
                ++m;
            }
        }
        // s_n = sum_{i=1..L} c_i s_{n-i}  =>  phi(x) = x^L + sum_i c_i x^{L-i}
        std::memset(phi, 0, PW * sizeof(uint64_t));
        for (int i = 0; i <= L; ++i)
            if (getbit(c.data(), i)) setbit(phi, static_cast<size_t>(L - i));
    }

    static const uint64_t* characteristicPolynomial() {
        static uint64_t phi[PW];
        static bool ready = false;
        if (!ready) {
            const size_t M = 2 * DEG;
            std::vector<uint64_t> srev((M + 127) / 64 + 2, 0);
            RandomGenerator g(1u);
            for (size_t i = 0; i < M; ++i)
                if (g.nextRaw() & 1u) setbit(srev.data(), M - 1 - i);
            berlekampMassey(srev.data(), M, phi);
            ready = true;
        }
        return phi;
    }

    // out = a*a (GF(2) squaring is a plain bit spread), a of degree < DEG
    static void polySquare(const uint64_t* a, uint64_t* out) {
        std::memset(out, 0, (2 * PW + 1) * sizeof(uint64_t));
        for (int i = 0; i < PW; ++i) {
            uint64_t v = a[i];
            if (!v) continue;
            for (int half = 0; half < 2; ++half) {
                uint64_t h = (v >> (32 * half)) & 0xffffffffull;
                uint64_t r = 0;
                while (h) {
                    r |= 1ull << (2 * __builtin_ctzll(h));
                    h &= h - 1;
                }
                out[2 * i + half] ^= r;
            }
        }
    }

    static void polyReduce(uint64_t* p, const uint64_t* phi, int topbit) {
        for (int j = topbit; j >= DEG; --j)
            if (getbit(p, static_cast<size_t>(j))) xorShifted(p, phi, static_cast<size_t>(j - DEG), PW);
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) {
        mt[0] = seed;
        for (int i = 1; i < NN; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        pos = 0;
    }

    // One step of the MT19937 recurrence (identical output to std::mt19937)
    uint32_t nextRaw() {
        int p1 = pos + 1;
        if (p1 >= NN) p1 -= NN;
        int pm = pos + MM;
        if (pm >= NN) pm -= NN;
        const uint32_t y = (mt[pos] & UPPER_MASK) | (mt[p1] & LOWER_MASK);
        const uint32_t x = mt[pm] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
        mt[pos] = x;
        pos = p1;
        return temper(x);
    }

    // Identical to std::uniform_real_distribution<float>(0,1)(std::mt19937),
    // which is std::generate_canonical<float, 24>: x / 2^32 clamped below 1.
    val_t rand() {
        const val_t v = static_cast<val_t>(nextRaw()) * (ONE / 4294967296.0f);
        return v < ONE ? v : 0.99999994f;  // nextafter(1.0f, 0.0f)
    }

    // Advance the stream by n numbers in O(DEG^2 / 64 + DEG * NN) time.
    void discard(uint64_t n) {
        if (n == 0) return;
        const uint64_t* phi = characteristicPolynomial();

        // g(x) = x^n mod phi(x), MSB-first square-and-multiply-by-x
        std::vector<uint64_t> g(2 * PW + 1, 0), tmp(2 * PW + 1, 0);
        g[0] = 1;
        for (int k = 63 - __builtin_clzll(n); k >= 0; --k) {
            polySquare(g.data(), tmp.data());
            polyReduce(tmp.data(), phi, 2 * DEG - 2);
            std::memcpy(g.data(), tmp.data(), PW * sizeof(uint64_t));
            std::memset(g.data() + PW, 0, (PW + 1) * sizeof(uint64_t));
            if ((n >> k) & 1ull) {
                uint64_t carry = 0;
                for (int i = 0; i < PW; ++i) {
                    const uint64_t v = g[i];
                    g[i] = (v << 1) | carry;
                    carry = v >> 63;
                }
                if (getbit(g.data(), DEG)) {
                    for (int i = 0; i < PW; ++i) g[i] ^= phi[i];
                    g[DEG >> 6] &= ~(1ull << (DEG & 63));
                }
            }
        }

        // Horner: h = 0; for k = DEG..0: h = f(h) ^ (g_k ? state : 0)
        uint32_t h[NN];
        std::memset(h, 0, sizeof(h));
        int hpos = 0;
        for (int k = DEG; k >= 0; --k) {
            int p1 = hpos + 1;
            if (p1 >= NN) p1 -= NN;
            int pm = hpos + MM;
            if (pm >= NN) pm -= NN;
            const uint32_t y = (h[hpos] & UPPER_MASK) | (h[p1] & LOWER_MASK);
            h[hpos] = h[pm] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            hpos = p1;
            if (getbit(g.data(), static_cast<size_t>(k))) {
                // XOR the current state in, matching the logical (oldest first) order
                const int off = hpos - pos;
                for (int m = 0; m < NN; ++m) {
                    int hi = m + off;
                    if (hi >= NN) hi -= NN;
                    if (hi < 0) hi += NN;
                    h[hi] ^= mt[m];
                }
            }
        }
        std::memcpy(mt, h, sizeof(mt));
        pos = hpos;
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

// One emitter contribution of a locally owned receiver row, precomputed once so
// that the timestep loop is a flat scan over the row's non-zero form factors.
struct Contribution {
    int32_t offset;   // j - tau * N, so that radB[t * N + offset] == radB[(t - tau) * N + j]
    int32_t tau;      // propagation delay in timesteps
    val_t weight;     // kij * area_j (clamped in the accumulation, as in the
                      // sequential code, so that the arithmetic is identical)
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    // Rows [rowBegin, rowBegin + rowCount) of the N x N matrices are owned locally
    size_t rowBegin = 0;
    size_t rowCount = 0;
    std::vector<int> rowCounts;     // per-rank row counts / displacements (for Allgatherv)
    std::vector<int> rowDispls;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (rowCount x N, row-major)
    std::vector<int> tau;           // Time delays (rowCount x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, replicated)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    // Compressed per-row emitter lists used by the wave propagation
    std::vector<Contribution> contrib;
    std::vector<size_t> contribStart;  // rowCount + 1 offsets into contrib

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t nonZeroKij = 0;          // global count, collected during precomputation

    // Local (i is a rank-local row index) and global indexing helpers
    size_t idxLocal(size_t iLocal, size_t j) const { return iLocal * numTriangles + j; }
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

    // Distribute the receiver triangles (rows of Kij/Tau) over the ranks
    blockRange(state.numTriangles, mpiRank, mpiSize, state.rowBegin, state.rowCount);
    state.rowCounts.resize(mpiSize);
    state.rowDispls.resize(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        size_t b, c;
        blockRange(state.numTriangles, r, mpiSize, b, c);
        state.rowDispls[r] = static_cast<int>(b);
        state.rowCounts[r] = static_cast<int>(c);
    }

    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration (replicated: every rank casts rays)
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (Kij/Tau only hold the locally owned rows)
    state.kij.resize(state.rowCount * state.numTriangles, ZERO);
    state.tau.resize(state.rowCount * state.numTriangles, 0);
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

// Number of random draws computeKij() consumes for a pair: culled pairs consume
// none, all others consume exactly two points per ray, two draws per point.
constexpr uint64_t DRAWS_PER_PAIR = 4 * NUM_RAYS;

void computeFormFactors(SimulationState& state) {
    rprintf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;

    // The sequential program consumes one global random stream in row-major pair
    // order. Count the draws the locally owned rows will consume, obtain the
    // number of draws consumed by all preceding rows with an exclusive scan, and
    // jump the generator there. The count only depends on the (cheap) culling
    // test performed at the top of computeKij().
    uint64_t localDraws = 0;
    for (size_t i = state.rowBegin; i < state.rowBegin + state.rowCount; ++i) {
        const Vec3 ni = state.triangles[i].normal();
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (ni.dot(state.triangles[j].normal()) > 0.99f) continue;
            localDraws += DRAWS_PER_PAIR;
        }
    }
    uint64_t precedingDraws = 0;
    MPI_Exscan(&localDraws, &precedingDraws, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (mpiRank == 0) precedingDraws = 0;

    RandomGenerator rng(42);
    rng.discard(precedingDraws);

    size_t localNonZero = 0;
    for (size_t li = 0; li < state.rowCount; ++li) {
        const size_t i = state.rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const val_t k = computeKij(i, j, state.triangles, state.octree, rng);
            state.kij[state.idxLocal(li, j)] = k;
            if (k > EPSILON) ++localNonZero;
        }
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            rprintf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }

    uint64_t local = localNonZero, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = static_cast<size_t>(global);
}

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");

    for (size_t li = 0; li < state.rowCount; ++li) {
        const size_t i = state.rowBegin + li;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idxLocal(li, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Compress the locally owned rows into flat lists of the emitters that actually
// contribute (non-zero form factor), keeping the sequential j order so that the
// floating point accumulation is unchanged. Kij/Tau are released afterwards.
static void buildContributions(SimulationState& state) {
    const size_t N = state.numTriangles;

    state.contribStart.assign(state.rowCount + 1, 0);
    size_t total = 0;
    for (size_t li = 0; li < state.rowCount; ++li) {
        const size_t i = state.rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            if (i != j && state.kij[state.idxLocal(li, j)] > ZERO) ++total;
        }
        state.contribStart[li + 1] = total;
    }

    state.contrib.resize(total);
    size_t pos = 0;
    for (size_t li = 0; li < state.rowCount; ++li) {
        const size_t i = state.rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const val_t kij = state.kij[state.idxLocal(li, j)];
            if (kij <= ZERO) continue;
            const int tauij = state.tau[state.idxLocal(li, j)];
            Contribution& c = state.contrib[pos++];
            c.tau = tauij;
            c.offset = static_cast<int32_t>(j) - tauij * static_cast<int32_t>(N);
            c.weight = kij * state.areas[j];
        }
    }

    std::vector<val_t>().swap(state.kij);
    std::vector<int>().swap(state.tau);
}

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    buildContributions(state);

    const size_t N = state.numTriangles;
    val_t* __restrict radB = state.radB.data();
    const Contribution* __restrict contrib = state.contrib.data();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);
        const size_t rowBase = t * N;
        const ptrdiff_t rowBaseS = static_cast<ptrdiff_t>(rowBase);

        for (size_t li = 0; li < state.rowCount; ++li) {
            const size_t i = state.rowBegin + li;
            val_t sumB = ZERO;

            const size_t end = state.contribStart[li + 1];
            for (size_t k = state.contribStart[li]; k < end; ++k) {
                const Contribution c = contrib[k];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < c.tau) continue;

                // Radiosity of the emitter at the time the wave left it
                const val_t radJ = radB[rowBaseS + c.offset];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(c.weight, ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            radB[rowBase + i] = state.rho[i] * sumB + state.radE[rowBase + i];
        }

        // Publish this timestep's slice: every rank needs the full row for the
        // delayed lookups of all later timesteps.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       radB + rowBase, state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

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

    const size_t T = state.numTimesteps;

    for (size_t li = 0; li < state.rowCount; ++li) {
        const size_t i = state.rowBegin + li;

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
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

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
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

    // Check Kij matrix (should have some non-zero entries); the count was
    // reduced across all ranks while the distributed rows were still available.
    const long long nonZeroKij = static_cast<long long>(state.nonZeroKij);
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
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

// Wall clock of the slowest rank for a phase, in milliseconds
static long phaseTime(double start) {
    double local = MPI_Wtime() - start;
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return static_cast<long>(global * 1000.0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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
            rprintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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
    rprintf("MPI ranks: %d\n", mpiSize);
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

    long preDuration = phaseTime(startPre);

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    double startSim = MPI_Wtime();

    runSimulation(state);

    long simDuration = phaseTime(startSim);

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    double startDist = MPI_Wtime();

    computeDistances(state);

    long distDuration = phaseTime(startDist);

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

    // Memory usage (aggregated over all ranks; the N x N matrices are distributed)
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
    if (printResults && mpiRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (all replicated data, evaluated on the root and broadcast)
    int status = 0;
    if (validate) {
        if (mpiRank == 0 && !validateResults(state)) status = 1;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return status;
}

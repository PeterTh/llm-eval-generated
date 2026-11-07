/**
 * Room Response Simulation Benchmark
 * 
 * This is an MPI-parallel (distributed memory) implementation of room impulse
 * response simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (MPI):
 *  - The mesh and octree are replicated on every rank (cheap, deterministic).
 *  - The N x N form factor (Kij) and time delay (Tau) matrices are distributed
 *    by contiguous row blocks; each rank only stores and computes its own rows.
 *    The row partition is balanced by the number of non-culled pairs per row.
 *  - The sequential code draws all random numbers from one mt19937 stream. To
 *    keep bit-identical results, each rank fast-forwards (discards) the stream
 *    to the position the sequential code would have reached at its first row.
 *  - Wave propagation computes radB[t, i] for local rows i and all-gathers the
 *    row of radiosities at every timestep (N floats).
 *  - Distances are computed for local rows and all-gathered.
 *  - Output is printed by rank 0 only.
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
// Mersenne Twister Jump-Ahead
// ============================================================================
//
// The sequential program consumes one global std::mt19937 stream. To reproduce
// it exactly, each MPI rank must position its generator where the sequential
// program would be at the rank's first row. mt19937::discard() is linear in the
// distance, which would eventually dominate at large rank counts. Instead we
// use the polynomial jump method (Haramoto, Matsumoto, Nishimura, Panneton,
// L'Ecuyer 2008): with phi the characteristic polynomial of the state
// transition A, the state after J steps is g(A) * S where g = x^J mod phi.
// phi is recovered at run time with Berlekamp-Massey, so no large constant
// tables are needed.

namespace mtjump {

constexpr size_t MT_N = 624;
constexpr size_t MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr size_t MT_DEG = 19937;                 // degree of the characteristic polynomial
constexpr size_t PW = (MT_DEG + 64) / 64;        // words for a polynomial of degree <= MT_DEG (312)

using Poly = std::vector<uint64_t>;              // bit i == coefficient of x^i

inline uint32_t twist(uint32_t x0, uint32_t x1, uint32_t xm) {
    uint32_t y = (x0 & MT_UPPER) | (x1 & MT_LOWER);
    return xm ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
}

inline bool getBit(const Poly& p, size_t i) { return (p[i >> 6] >> (i & 63)) & 1u; }
inline void setBit(Poly& p, size_t i) { p[i >> 6] |= uint64_t(1) << (i & 63); }

// dst ^= src << k   (src has nsrc words; dst must have room)
inline void shiftXor(Poly& dst, const Poly& src, size_t nsrc, size_t k) {
    const size_t ws = k >> 6;
    const unsigned bs = static_cast<unsigned>(k & 63);
    if (bs == 0) {
        for (size_t i = 0; i < nsrc; ++i) dst[i + ws] ^= src[i];
    } else {
        for (size_t i = 0; i < nsrc; ++i) {
            dst[i + ws] ^= src[i] << bs;
            dst[i + ws + 1] ^= src[i] >> (64 - bs);
        }
    }
}

// Reduce p (any size) modulo phi (degree MT_DEG); result has degree < MT_DEG.
inline void reduceMod(Poly& p, const Poly& phi) {
    for (size_t bit = p.size() * 64; bit-- > MT_DEG;) {
        if (getBit(p, bit)) shiftXor(p, phi, PW, bit - MT_DEG);
    }
}

// Spread the 32 bits of v into the even bit positions of a 64-bit word.
inline uint64_t spread32(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

// a^2 mod phi (squaring over GF(2) just spreads the bits)
inline Poly squareMod(const Poly& a, const Poly& phi) {
    Poly r(2 * PW + 1, 0);
    for (size_t w = 0; w < PW; ++w) {
        r[2 * w] = spread32(static_cast<uint32_t>(a[w]));
        r[2 * w + 1] = spread32(static_cast<uint32_t>(a[w] >> 32));
    }
    reduceMod(r, phi);
    r.resize(PW);
    return r;
}

// a*x mod phi
inline void mulXMod(Poly& a, const Poly& phi) {
    Poly r(PW + 1, 0);
    shiftXor(r, a, PW, 1);
    reduceMod(r, phi);
    r.resize(PW);
    a.swap(r);
}

// x^J mod phi by square-and-multiply
inline Poly powXMod(unsigned long long J, const Poly& phi) {
    Poly r(PW, 0);
    setBit(r, 0);  // r = 1
    for (int bit = 63; bit >= 0; --bit) {
        r = squareMod(r, phi);
        if ((J >> bit) & 1ULL) mulXMod(r, phi);
    }
    return r;
}

// Characteristic polynomial of the MT19937 recurrence via Berlekamp-Massey on
// the top bit of the raw recurrence (a non-degenerate linear functional of the
// state; since the polynomial is primitive, the minimal polynomial of any such
// sequence is the full characteristic polynomial).
inline const Poly& characteristicPolynomial() {
    static Poly phi;
    if (!phi.empty()) return phi;

    // Generate the bit sequence
    const size_t nbits = 2 * MT_DEG + 128;
    std::vector<uint32_t> x(MT_N);
    {
        std::mt19937 seeder(0x5EEDu);
        for (auto& v : x) v = static_cast<uint32_t>(seeder());
    }
    size_t pos = 0;
    const size_t nwordsSeq = nbits / 64 + 2;
    Poly rev(nwordsSeq + 2, 0);  // reversed sequence: rev bit t == s[nbits-1-t]
    for (size_t k = 0; k < nbits; ++k) {
        uint32_t nx = twist(x[pos], x[(pos + 1) % MT_N], x[(pos + MT_M) % MT_N]);
        x[pos] = nx;
        pos = (pos + 1) % MT_N;
        if (nx & MT_UPPER) setBit(rev, nbits - 1 - k);
    }

    // Berlekamp-Massey over GF(2) (bit-packed)
    const size_t pwords = nbits / 64 + 2;
    Poly C(pwords, 0), B(pwords, 0), T(pwords, 0);
    setBit(C, 0);
    setBit(B, 0);
    size_t L = 0, m = 1;
    for (size_t i = 0; i < nbits; ++i) {
        // discrepancy d = sum_{j=0..L} C_j * s[i-j] = sum_j C_j * rev[(nbits-1-i)+j]
        const size_t off = nbits - 1 - i;
        const size_t ws = off >> 6;
        const unsigned bs = static_cast<unsigned>(off & 63);
        const size_t nw = (L >> 6) + 1;
        uint64_t acc = 0;
        for (size_t w = 0; w < nw; ++w) {
            uint64_t win = rev[ws + w] >> bs;
            if (bs) win |= rev[ws + w + 1] << (64 - bs);
            acc ^= C[w] & win;
        }
        const bool d = __builtin_parityll(acc);
        if (!d) {
            ++m;
            continue;
        }
        T = C;
        shiftXor(C, B, pwords - (m >> 6) - 1, m);
        if (2 * L <= i) {
            L = i + 1 - L;
            B.swap(T);
            m = 1;
        } else {
            ++m;
        }
    }

    if (L != MT_DEG) {
        fprintf(stderr, "MT jump: unexpected linear complexity %zu\n", L);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // phi(x) = x^L * C(1/x)  (reciprocal of the connection polynomial)
    phi.assign(PW, 0);
    for (size_t j = 0; j <= L; ++j) {
        if (getBit(C, j)) setBit(phi, L - j);
    }
    return phi;
}

// Advance a std::mt19937 by J steps in O(log J) polynomial operations.
inline void jump(std::mt19937& eng, unsigned long long J) {
    if (J == 0) return;

    // Extract the engine state (libstdc++ streams the 624 words then the index)
    std::vector<uint32_t> mx(MT_N);
    size_t mp = 0;
    {
        std::stringstream ss;
        ss << eng;
        unsigned long long v;
        for (size_t i = 0; i < MT_N; ++i) { ss >> v; mx[i] = static_cast<uint32_t>(v); }
        ss >> v;
        mp = static_cast<size_t>(v);
    }

    // The 624 stored words always form a valid canonical window
    // S = (x_q, ..., x_{q+623}) whose next raw output is x_{q+624}. With block
    // index p < 624 the engine still has 624 - p pending outputs from this
    // window, i.e. it is 624 - p steps *behind* the canonical state. Account for
    // that by shortening the jump (falling back to discard for tiny jumps).
    const size_t behind = MT_N - std::min(mp, MT_N);
    if (J <= behind) {
        eng.discard(J);
        return;
    }
    J -= behind;
    const std::vector<uint32_t>& S = mx;

    // g = x^J mod phi
    const Poly& phi = characteristicPolynomial();
    Poly g = powXMod(J, phi);

    // Horner: R = sum_k g_k A^k S, with R kept in a circular buffer.
    std::vector<uint32_t> R(MT_N, 0);
    size_t base = 0;  // canonical element t lives at R[(base + t) % MT_N]
    for (size_t k = MT_DEG; k-- > 0;) {
        // R = A R : append x_624 computed from the window, drop x_0
        const size_t i1 = (base + 1) % MT_N;
        const size_t im = (base + MT_M) % MT_N;
        R[base] = twist(R[base], R[i1], R[im]);
        base = i1;
        if (getBit(g, k)) {
            for (size_t t = 0; t < MT_N; ++t) R[(base + t) % MT_N] ^= S[t];
        }
    }

    // Load back: canonical window with index 624 (block exhausted)
    {
        std::stringstream ss;
        for (size_t t = 0; t < MT_N; ++t) ss << R[(base + t) % MT_N] << ' ';
        ss << MT_N;
        ss >> eng;
    }
}

// Verify the jump implementation against discard() on this platform.
inline void selfTest() {
    static bool done = false;
    if (done) return;
    done = true;
    const unsigned long long jumps[2] = {1000003ULL, 777ULL};
    for (int c = 0; c < 2; ++c) {
        std::mt19937 a(12345u), b(12345u);
        for (int i = 0; i < 10 * c; ++i) { (void)a(); (void)b(); }  // second case: mid-block start
        a.discard(jumps[c]);
        jump(b, jumps[c]);
        for (int i = 0; i < 2000; ++i) {
            if (a() != b()) {
                fprintf(stderr, "MT jump self-test FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
}

}  // namespace mtjump

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    // Fast-forward the underlying engine by n engine invocations. Short
    // distances use discard() (linear, tiny constant); long distances use the
    // O(log n) polynomial jump, which is verified against discard() first.
    void skip(unsigned long long n) {
        constexpr unsigned long long JUMP_THRESHOLD = 50000000ULL;
        if (n < JUMP_THRESHOLD) {
            rng.discard(n);
        } else {
            mtjump::selfTest();
            mtjump::jump(rng, n);
        }
    }

    // Number of engine invocations consumed by a single rand() call. This is
    // determined empirically (rather than assumed) so that skipping ahead in the
    // stream reproduces exactly the sequential program's random sequence.
    static unsigned enginePullsPerRand() {
        for (unsigned k = 1; k <= 8; ++k) {
            RandomGenerator a, b;
            a.rand();
            b.rng.discard(k);
            if (a.rng == b.rng) return k;
        }
        fprintf(stderr, "Unable to determine RNG engine pulls per sample\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 0;
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

// Pairs of triangles facing (nearly) the same direction are culled: their form
// factor is zero and, importantly, no random numbers are consumed for them.
// This predicate is evaluated in exactly one place so that the random stream
// bookkeeping and the actual computation can never disagree.
inline bool isCulledPair(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Fill mask[j] = 1 if pair (i, j) requires a form factor computation (and thus
// consumes random numbers), 0 otherwise. Returns the number of such pairs.
size_t computeActiveMask(size_t idxI, const std::vector<Triangle>& triangles,
                         std::vector<uint8_t>& mask) {
    const size_t n = triangles.size();
    const Triangle& triI = triangles[idxI];
    size_t count = 0;
    for (size_t j = 0; j < n; ++j) {
        const bool active = (j != idxI) && !isCulledPair(triI, triangles[j]);
        mask[j] = static_cast<uint8_t>(active);
        count += active;
    }
    return count;
}

// Number of rand() calls consumed by one (non-culled) computeKij invocation.
constexpr unsigned long long RANDS_PER_KIJ = 4ULL * NUM_RAYS;

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter).
// Must only be called for pairs that are not culled (see isCulledPair).
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

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
// MPI Helpers
// ============================================================================

static int g_rank = 0;
static int g_size = 1;

// printf that only produces output on rank 0
#define rprintf(...)                         \
    do {                                     \
        if (g_rank == 0) printf(__VA_ARGS__); \
    } while (0)

static double wallTime() { return MPI_Wtime(); }

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)

    // Row-block distribution of the N x N matrices: this rank owns rows
    // [rowBegin, rowEnd) of Kij and Tau, stored as (numLocalRows x N).
    size_t rowBegin;
    size_t rowEnd;
    size_t numLocalRows;
    std::vector<int> rowCounts;     // Rows owned by each rank
    std::vector<int> rowDispls;     // First row owned by each rank
    unsigned long long randOffset;  // rand() calls consumed before our first row

    std::vector<val_t> kij;         // Form factors (local rows x N, row-major)
    std::vector<int> tau;           // Time delays (local rows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, replicated)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Index into the local row block (i is a global row index)
    size_t idx2dLocal(size_t i, size_t j) const { return (i - rowBegin) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Row Partitioning
// ============================================================================

// Partition the rows among ranks so that each rank gets (approximately) the
// same number of active (non-culled) pairs, which is proportional to both the
// form factor work and the number of random numbers consumed. Also computes
// the random stream offset at which this rank's first row begins.
void partitionRows(SimulationState& state) {
    const size_t n = state.numTriangles;
    const int P = g_size;

    // Step 1: each rank counts the active pairs for a naive block of rows.
    std::vector<int> naiveCounts(P), naiveDispls(P);
    for (int r = 0; r < P; ++r) {
        size_t b = (n * static_cast<size_t>(r)) / P;
        size_t e = (n * static_cast<size_t>(r + 1)) / P;
        naiveDispls[r] = static_cast<int>(b);
        naiveCounts[r] = static_cast<int>(e - b);
    }

    std::vector<int> rowActive(n, 0);
    {
        std::vector<uint8_t> mask(n);
        const size_t b = static_cast<size_t>(naiveDispls[g_rank]);
        const size_t e = b + static_cast<size_t>(naiveCounts[g_rank]);
        for (size_t i = b; i < e; ++i) {
            rowActive[i] = static_cast<int>(computeActiveMask(i, state.triangles, mask));
        }
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   rowActive.data(), naiveCounts.data(), naiveDispls.data(), MPI_INT,
                   MPI_COMM_WORLD);

    // Step 2: prefix sums of active pairs (identical on all ranks).
    std::vector<unsigned long long> prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + static_cast<unsigned long long>(rowActive[i]);
    const unsigned long long total = prefix[n];

    // Step 3: choose contiguous row ranges with balanced active-pair counts.
    state.rowCounts.assign(P, 0);
    state.rowDispls.assign(P, 0);
    size_t row = 0;
    for (int r = 0; r < P; ++r) {
        state.rowDispls[r] = static_cast<int>(row);
        if (r == P - 1) {
            state.rowCounts[r] = static_cast<int>(n - row);
            row = n;
            continue;
        }
        // Target cumulative work at the end of rank r
        const unsigned long long target = (total * static_cast<unsigned long long>(r + 1)) / P;
        size_t end = row;
        // Advance while adding the next row keeps us at or below target, or
        // while the row contributes nothing (so we never fall behind).
        while (end < n && prefix[end + 1] <= target) ++end;
        // Make sure ranks get at least one row when enough rows exist
        if (end == row && row < n && (n - row) > static_cast<size_t>(P - r)) end = row + 1;
        // Leave enough rows for the remaining ranks if possible
        const size_t remainingRanks = static_cast<size_t>(P - r - 1);
        if (n > remainingRanks && end > n - remainingRanks) end = n - remainingRanks;
        if (end < row) end = row;
        state.rowCounts[r] = static_cast<int>(end - row);
        row = end;
    }

    state.rowBegin = static_cast<size_t>(state.rowDispls[g_rank]);
    state.numLocalRows = static_cast<size_t>(state.rowCounts[g_rank]);
    state.rowEnd = state.rowBegin + state.numLocalRows;
    state.randOffset = prefix[state.rowBegin] * RANDS_PER_KIJ;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (replicated, deterministic)
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

    // Distribute matrix rows among ranks
    partitionRows(state);
    rprintf("Distributed %zu rows over %d MPI ranks\n", state.numTriangles, g_size);

    // Initialize matrices (local row blocks for the N x N ones)
    state.kij.assign(state.numLocalRows * state.numTriangles, ZERO);
    state.tau.assign(state.numLocalRows * state.numTriangles, 0);
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

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
    const double localStart = wallTime();
    RandomGenerator rng(42);

    // Fast-forward the random stream to where the sequential program would be
    // when starting this rank's first row.
    const unsigned long long pulls = RandomGenerator::enginePullsPerRand();
    rng.skip(state.randOffset * pulls);

    const size_t n = state.numTriangles;
    std::vector<uint8_t> mask(n);

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        computeActiveMask(i, state.triangles, mask);
        val_t* kijRow = &state.kij[state.idx2dLocal(i, 0)];
        for (size_t j = 0; j < n; ++j) {
            if (!mask[j]) continue;  // i == j or culled: Kij stays zero, no RNG use
            kijRow[j] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        if (g_rank == 0) {
            const size_t done = i - state.rowBegin + 1;
            if (done % 100 == 0 || done == state.numLocalRows) {
                printf("  Progress: %zu/%zu triangles (rank 0)\n", done, state.numLocalRows);
            }
        }
    }

    // Load balance diagnostic: min/max per-rank time for this phase
    const double localTime = wallTime() - localStart;
    double minTime = 0.0, maxTime = 0.0;
    MPI_Reduce(&localTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    rprintf("  Per-rank form factor time: min %.0f ms, max %.0f ms\n", minTime * 1000.0, maxTime * 1000.0);
}

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        int* tauRow = &state.tau[state.idx2dLocal(i, 0)];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            tauRow[j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const val_t* areas = state.areas.data();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        const int ti = static_cast<int>(t);
        val_t* radBt = &state.radB[state.idxTN(t, 0)];

        for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
            val_t sumB = ZERO;
            const int* tauRow = &state.tau[state.idx2dLocal(i, 0)];
            const val_t* kijRow = &state.kij[state.idx2dLocal(i, 0)];

            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                int tauij = tauRow[j];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                val_t kij = kijRow[j];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            radBt[i] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radiosities with all ranks (N floats). Tau is
        // >= 1 for all i != j, so no rank needs another rank's values from the
        // current timestep before this exchange.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       radBt, state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);

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

    // Contiguous copies of the time series (columns of radB) for cache locality
    std::vector<val_t> pS(T), pB(T);
    for (size_t tt = 0; tt < T; ++tt) pS[tt] = state.radB[state.idxTN(tt, state.sourceIndex)];

    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t tt = 0; tt < T; ++tt) pB[tt] = state.radB[state.idxTN(tt, i)];

        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                sum += pS[tt - t] * pB[tt];
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    // Gather all distances on every rank
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

// Collective: all ranks must call this; the result is identical on all ranks.
bool validateResults(const SimulationState& state) {
    rprintf("\nValidation:\n");

    // Count non-zero local form factors and reduce (collective, done first so
    // that early returns below cannot desynchronize the ranks).
    long long localNonZeroKij = 0;
    for (size_t i = 0; i < state.kij.size(); ++i) {
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

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
    const size_t totalPairs = state.numTriangles * state.numTriangles;
    rprintf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
            nonZeroKij, totalPairs,
            100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalPairs));

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
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

// Barrier + wall clock, used to time phases consistently across ranks
static double syncTime() {
    MPI_Barrier(MPI_COMM_WORLD);
    return wallTime();
}

static long toMs(double seconds) {
    return static_cast<long>(seconds * 1000.0 + 0.5);
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

    // Parse command line arguments (identical on all ranks)
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
    rprintf("MPI ranks: %d\n", g_size);
    rprintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rprintf("\n");

    // Precomputation
    double startPre = syncTime();

    computeTimeDelays(state);
    computeFormFactors(state);

    double endPre = syncTime();
    long preDuration = toMs(endPre - startPre);

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    double startSim = syncTime();

    runSimulation(state);

    double endSim = syncTime();
    long simDuration = toMs(endSim - startSim);

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    double startDist = syncTime();

    computeDistances(state);

    double endDist = syncTime();
    long distDuration = toMs(endDist - startDist);

    rprintf("Distance computation time: %ld ms\n", distDuration);
    rprintf("\n");

    // Total time
    const double localPhaseTimes[3] = {endPre - startPre, endSim - startSim, endDist - startDist};
    double globalPhaseTimes[3] = {};
    MPI_Reduce(localPhaseTimes, globalPhaseTimes, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long totalTime = 0;
    if (g_rank == 0) {
        totalTime = toMs(globalPhaseTimes[0] + globalPhaseTimes[1] + globalPhaseTimes[2]);
    }
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

    // Memory usage (aggregate over all ranks; N x N matrices are distributed)
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
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (collective; same outcome on all ranks)
    int exitCode = 0;
    if (validate) {
        if (!validateResults(state)) {
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}

/**
 * Room Response Simulation Benchmark
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
 *
 * Parallelization (hybrid MPI + OpenMP + CUDA):
 *  - MPI: the rows i (receiver triangles) of the Kij/Tau matrices are split
 *    across ranks; radiosity rows are all-gathered after each timestep.
 *  - CUDA: form factors (ray casting through a flattened octree), wave
 *    propagation and cross-correlation run on one GPU per rank.
 *  - OpenMP: host-side setup (time delays, culling masks, RNG jump-ahead).
 *  The random number stream of the sequential version is reproduced exactly
 *  using Mersenne Twister jump-ahead, so every pair sees the same samples.
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
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
// Error Checking Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Mersenne Twister (std::mt19937) Jump-Ahead
// ============================================================================
//
// The sequential code draws every random number from a single std::mt19937
// stream (seed 42), consuming 4 * NUM_RAYS numbers for each (i, j) pair that is
// not culled, in row-major order. To reproduce exactly the same numbers in
// parallel, each worker starts at its own offset in that stream. The start
// states are obtained with the polynomial jump-ahead method over GF(2):
//   1. the characteristic polynomial p(x) of the MT transition is recovered
//      with Berlekamp-Massey from 2 * 19937 output bits,
//   2. x^J mod p(x) is computed by square-and-multiply,
//   3. the jumped state is sum_k g_k A^k S, evaluated with Horner's rule.

namespace mtjump {

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MEXP = 19937;
constexpr uint32_t MATRIX_A = 0x9908b0dfU;
constexpr uint32_t UPPER_MASK = 0x80000000U;
constexpr uint32_t LOWER_MASK = 0x7fffffffU;
constexpr int PW = (MEXP + 1 + 63) / 64;   // words of a polynomial of degree <= MEXP
constexpr int SW = 2 * PW + 2;             // words of a product (with slack)

using Poly = std::vector<uint64_t>;
using MTState = std::array<uint32_t, MT_N>;  // window (x_k, ..., x_{k+623})

inline int getBit(const uint64_t* a, size_t k) { return (a[k >> 6] >> (k & 63)) & 1; }
inline void flipBit(uint64_t* a, size_t k) { a[k >> 6] ^= (uint64_t(1) << (k & 63)); }

// 64 bits of a starting at bit offset o (a must be padded by one word)
inline uint64_t get64(const uint64_t* a, size_t o) {
    size_t w = o >> 6, b = o & 63;
    return b ? ((a[w] >> b) | (a[w + 1] << (64 - b))) : a[w];
}

// dst ^= src << shift  (src has n words)
inline void xorShifted(uint64_t* __restrict dst, const uint64_t* __restrict src, int n, size_t shift) {
    size_t ws = shift >> 6, bs = shift & 63;
    uint64_t* d = dst + ws;
    if (bs == 0) {
        for (int i = 0; i < n; ++i) d[i] ^= src[i];
    } else {
        d[0] ^= src[0] << bs;
        for (int i = 1; i < n; ++i) d[i] ^= (src[i] << bs) | (src[i - 1] >> (64 - bs));
        d[n] ^= src[n - 1] >> (64 - bs);
    }
}

// Initial std::mt19937 state window for a given seed
MTState seedState(uint32_t seed) {
    MTState s;
    s[0] = seed;
    for (int i = 1; i < MT_N; ++i)
        s[i] = 1812433253U * (s[i - 1] ^ (s[i - 1] >> 30)) + static_cast<uint32_t>(i);
    return s;
}

class Jumper {
public:
    Poly p;                           // characteristic polynomial, degree MEXP
    std::vector<uint64_t> redTable;   // 256 x PW reduction table for 8-bit chunks

    explicit Jumper(uint32_t seed) {
        computeCharPoly(seed);
        buildTable();
    }

    // x^e mod p
    Poly powX(uint64_t e) const {
        Poly r(PW, 0);
        r[0] = 1;
        bool started = false;
        for (int b = 63; b >= 0; --b) {
            if (started) r = sqrMod(r);
            if ((e >> b) & 1) {
                r = mulX(r);
                started = true;
            }
        }
        return r;
    }

    Poly sqrMod(const Poly& a) const {
        std::vector<uint64_t> buf(SW + 1, 0);
        for (int i = 0; i < PW; ++i) {
            buf[2 * i] = spread(static_cast<uint32_t>(a[i]));
            buf[2 * i + 1] = spread(static_cast<uint32_t>(a[i] >> 32));
        }
        return reduce(buf, 2 * MEXP - 2);
    }

    Poly mulX(const Poly& a) const {
        Poly r(PW, 0);
        uint64_t carry = 0;
        for (int i = 0; i < PW; ++i) {
            r[i] = (a[i] << 1) | carry;
            carry = a[i] >> 63;
        }
        if (getBit(r.data(), MEXP))
            for (int i = 0; i < PW; ++i) r[i] ^= p[i];
        return r;
    }

private:
    static uint64_t spread(uint32_t x) {
        uint64_t v = x;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v << 2)) & 0x3333333333333333ULL;
        v = (v | (v << 1)) & 0x5555555555555555ULL;
        return v;
    }

    // Reduce buf (degree <= topDeg) modulo p
    Poly reduce(std::vector<uint64_t>& buf, int topDeg) const {
        int lo = topDeg - 7;
        for (; lo >= MEXP; lo -= 8) {
            uint32_t b = static_cast<uint32_t>(get64(buf.data(), lo) & 0xFF);
            if (b) xorShifted(buf.data(), &redTable[size_t(b) * PW], PW, lo - MEXP);
        }
        for (int d = lo + 7; d >= MEXP; --d)
            if (getBit(buf.data(), d)) xorShifted(buf.data(), p.data(), PW, d - MEXP);
        return Poly(buf.begin(), buf.begin() + PW);
    }

    void buildTable() {
        redTable.assign(size_t(256) * PW, 0);
        for (int b = 0; b < 256; ++b) {
            std::vector<uint64_t> acc(PW + 2, 0);
            for (int k = 7; k >= 0; --k) {
                if (getBit(acc.data(), MEXP + k) != ((b >> k) & 1))
                    xorShifted(acc.data(), p.data(), PW, k);
            }
            std::copy(acc.begin(), acc.begin() + PW, redTable.begin() + size_t(b) * PW);
        }
    }

    void computeCharPoly(uint32_t seed) {
        const int N = 2 * MEXP;
        const int RW = N / 64 + 4;
        std::vector<uint64_t> rev(RW + 2, 0);  // rev bit k = s_{N-1-k}
        std::mt19937 rng(seed);
        for (int k = 0; k < N; ++k)
            if (rng() & 1U) flipBit(rev.data(), N - 1 - k);

        const int CW = N / 64 + 4;
        std::vector<uint64_t> C(CW + 2, 0), B(CW + 2, 0), T;
        C[0] = 1;
        B[0] = 1;
        int L = 0, m = 1;
        for (int n = 0; n < N; ++n) {
            size_t o = static_cast<size_t>(N - 1 - n);
            uint64_t acc = 0;
            int nw = L / 64 + 1;
            for (int w = 0; w < nw; ++w) acc ^= C[w] & get64(rev.data(), o + 64 * size_t(w));
            int d = __builtin_parityll(acc);
            if (!d) {
                ++m;
                continue;
            }
            int bw = std::min(CW - m / 64 - 1, L / 64 + 2);
            if (2 * L <= n) {
                T = C;
                xorShifted(C.data(), B.data(), bw, m);
                L = n + 1 - L;
                B = T;
                m = 1;
            } else {
                xorShifted(C.data(), B.data(), bw, m);
                ++m;
            }
        }
        if (L != MEXP) {
            fprintf(stderr, "MT jump: unexpected linear complexity %d\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        p.assign(PW, 0);
        for (int k = 0; k <= MEXP; ++k)
            if (getBit(C.data(), MEXP - k)) flipBit(p.data(), k);
    }
};

}  // namespace mtjump

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Device Data Structures
// ============================================================================

struct DevTri {
    float ax, ay, az;       // vertex a
    float e1x, e1y, e1z;    // b - a
    float e2x, e2y, e2z;    // c - a
    float nx, ny, nz;       // normal
};

// Octree node in depth-first preorder: a = (center, halfExtent.x),
// b = (halfExtent.y, halfExtent.z, skip, unused) where skip is the index of the
// next node after this node's subtree (stackless traversal).
struct alignas(16) DevNode {
    float4 a, b;
};

// Leaf triangle copy: (a, e1.x), (e1.y, e1.z, e2.x, e2.y), (e2.z, index, -, -)
struct alignas(16) DevLeafTri {
    float4 p0, p1, p2;
};

constexpr int FF_THREADS = 256;                       // threads per form-factor block
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;          // 64 random numbers per pair
constexpr int FF_PAIRS = 16;                          // pairs per batch
constexpr int FF_RAYS = FF_PAIRS * NUM_RAYS;          // rays per batch
constexpr int FF_RBUF = FF_PAIRS * RANDS_PER_PAIR + mtjump::MT_N;
static_assert(FF_THREADS >= mtjump::MT_N - mtjump::MT_M, "cooperative MT twist needs >= 227 threads");
static_assert(FF_THREADS % 32 == 0 && FF_PAIRS <= FF_THREADS, "invalid form factor block layout");

// ============================================================================
// Device Geometry Kernels
// ============================================================================

__device__ __forceinline__ float rayTriangleIntersectDev(
    float ox, float oy, float oz, float dx, float dy, float dz,
    float ax, float ay, float az, float e1x, float e1y, float e1z,
    float e2x, float e2y, float e2z) {
    const float FMAX = 3.40282347e+38f;
    // pvec = dir x e2
    float px = dy * e2z - dz * e2y;
    float py = dz * e2x - dx * e2z;
    float pz = dx * e2y - dy * e2x;
    float det = e1x * px + e1y * py + e1z * pz;
    if (fabsf(det) < EPSILON) return FMAX;
    float invDet = 1.0f / det;
    float tx = ox - ax, ty = oy - ay, tz = oz - az;
    float u = (tx * px + ty * py + tz * pz) * invDet;
    if (u < 0.0f || u > 1.0f) return FMAX;
    // qvec = tvec x e1
    float qx = ty * e1z - tz * e1y;
    float qy = tz * e1x - tx * e1z;
    float qz = tx * e1y - ty * e1x;
    float v = (dx * qx + dy * qy + dz * qz) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FMAX;
    return (e2x * qx + e2y * qy + e2z * qz) * invDet;
}

// Octree-accelerated occlusion test. Visits exactly the leaves reached by the
// recursive host traversal (a child is entered iff the segment overlaps its
// box); the result is an "any hit" and therefore order-independent.
__device__ bool isRayBlockedDev(float fx, float fy, float fz, float tx, float ty, float tz,
                                const DevNode* __restrict__ nodes,
                                const int2* __restrict__ nodeTris,
                                const DevLeafTri* __restrict__ leafGeo,
                                int numNodes, int srcIdx, int dstIdx) {
    float dx = tx - fx, dy = ty - fy, dz = tz - fz;
    float rayLen = sqrtf(dx * dx + dy * dy + dz * dz);
    if (rayLen < EPSILON) return true;
    float ndx = dx / rayLen, ndy = dy / rayLen, ndz = dz / rayLen;
    float maxDist = rayLen - EPSILON;

    // Segment constants for the box tests
    float hdx = dx * 0.5f, hdy = dy * 0.5f, hdz = dz * 0.5f;
    float mx = fx + hdx, my = fy + hdy, mz = fz + hdz;
    float adx = fabsf(hdx), ady = fabsf(hdy), adz = fabsf(hdz);

    auto testLeaf = [&](int2 info) -> bool {
        for (int k = info.x; k < info.x + info.y; ++k) {
            const float4 p0 = __ldg(&leafGeo[k].p0);
            const float4 p1 = __ldg(&leafGeo[k].p1);
            const float4 p2 = __ldg(&leafGeo[k].p2);
            int idx = __float_as_int(p2.y);
            if (idx == srcIdx || idx == dstIdx) continue;
            float dist = rayTriangleIntersectDev(fx, fy, fz, ndx, ndy, ndz,
                                                 p0.x, p0.y, p0.z, p0.w, p1.x, p1.y,
                                                 p1.z, p1.w, p2.x);
            if (dist > EPSILON && dist < maxDist) return true;
        }
        return false;
    };

    const int2 rootInfo = __ldg(&nodeTris[0]);
    if (rootInfo.y > 0) return testLeaf(rootInfo);

    int i = 1;
    while (i < numNodes) {
        const float4 a = __ldg(&nodes[i].a);
        const float4 b = __ldg(&nodes[i].b);
        const int skip = __float_as_int(b.z);
        float cx = mx - a.x, cy = my - a.y, cz = mz - a.z;
        bool overlap = !(fabsf(cx) > a.w + adx) && !(fabsf(cy) > b.x + ady) &&
                       !(fabsf(cz) > b.y + adz) &&
                       !(fabsf(hdy * cz - hdz * cy) > b.x * adz + b.y * ady + EPSILON) &&
                       !(fabsf(hdz * cx - hdx * cz) > b.y * adx + a.w * adz + EPSILON) &&
                       !(fabsf(hdx * cy - hdy * cx) > a.w * ady + b.x * adx + EPSILON);
        if (!overlap) {
            i = skip;
            continue;
        }
        const int2 info = __ldg(&nodeTris[i]);
        if (info.y > 0) {
            if (testLeaf(info)) return true;
            i = skip;
        } else {
            ++i;
        }
    }
    return false;
}

__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0, 1) on top of std::mt19937 (libstdc++)
__device__ __forceinline__ float mtToFloat(uint32_t y) {
    float r = __uint2float_rn(y) * 2.3283064365386963e-10f;  // / 2^32 (exact)
    return r >= 1.0f ? 0.99999994f : r;
}

// MT jump-ahead on the GPU (one block per jump). With x_0, x_1, ... the raw
// MT sequence continuing the source window (x_0 .. x_623), the window advanced
// by J is  w[m] = XOR_{k : g_k = 1} x_{k+m}  where g = x^J mod p(x).
constexpr int JUMP_THREADS = 640;
constexpr int JUMP_SEQ = mtjump::MEXP + mtjump::MT_N - 1;
constexpr size_t JUMP_SMEM = JUMP_SEQ * sizeof(uint32_t) + mtjump::PW * sizeof(uint64_t);

__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(const uint64_t* __restrict__ g, const uint32_t* __restrict__ src, size_t srcStride,
             uint32_t* __restrict__ dst, size_t dstStride) {
    using namespace mtjump;
    extern __shared__ uint64_t jumpSmem[];
    uint64_t* gs = jumpSmem;
    uint32_t* x = reinterpret_cast<uint32_t*>(jumpSmem + PW);
    const int tid = threadIdx.x;
    src += blockIdx.x * srcStride;
    dst += blockIdx.x * dstStride;
    for (int k = tid; k < PW; k += JUMP_THREADS) gs[k] = g[k];
    for (int k = tid; k < MT_N; k += JUMP_THREADS) x[k] = src[k];
    __syncthreads();
    // x_{k+624} = x_{k+397} ^ twist(x_k, x_{k+1}): 227 independent words per step
    for (int base = MT_N; base < JUMP_SEQ; base += MT_N - MT_M) {
        int idx = base + tid;
        if (tid < MT_N - MT_M && idx < JUMP_SEQ) {
            uint32_t y = (x[idx - MT_N] & UPPER_MASK) | (x[idx - MT_N + 1] & LOWER_MASK);
            x[idx] = x[idx - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        __syncthreads();
    }
    if (tid < MT_N) {
        uint32_t acc = 0;
        for (int w = 0; w < PW; ++w) {
            uint64_t bits = gs[w];
            while (bits) {
                int k = w * 64 + __ffsll(static_cast<long long>(bits)) - 1;
                bits &= bits - 1;
                acc ^= x[k + tid];
            }
        }
        dst[tid] = acc;
    }
}

// Form factor kernel: each block owns a contiguous range of non-culled pairs
// (in the original row-major order) and the matching contiguous slice of the
// random number stream. Within a batch, warps fetch rays dynamically.
__global__ void __launch_bounds__(FF_THREADS)
formFactorKernel(const uint32_t* __restrict__ states,
                 const int* __restrict__ startRow, const int* __restrict__ startCol,
                 const long long* __restrict__ pairCount,
                 const uint32_t* __restrict__ active, int wordsPerRow,
                 const DevTri* __restrict__ tris, const DevNode* __restrict__ nodes,
                 const int2* __restrict__ nodeTris, const DevLeafTri* __restrict__ leafGeo,
                 int numNodes,
                 int n, int rowBegin, int rows, float* __restrict__ kT) {
    using namespace mtjump;
    __shared__ uint32_t mt[2][MT_N];
    __shared__ float rbuf[FF_RBUF];
    __shared__ float contrib[FF_RAYS];
    __shared__ int pairRow[FF_PAIRS], pairCol[FF_PAIRS];
    __shared__ int curRow, curCol, have, nextRay;

    const int tid = threadIdx.x;
    const int s = blockIdx.x;
    for (int k = tid; k < MT_N; k += FF_THREADS) mt[0][k] = states[size_t(s) * MT_N + k];
    if (tid == 0) {
        curRow = startRow[s];
        curCol = startCol[s];
        have = 0;
        nextRay = 0;
    }
    int cur = 0;
    long long remaining = pairCount[s];
    __syncthreads();

    while (remaining > 0) {
        const int np = remaining < FF_PAIRS ? static_cast<int>(remaining) : FF_PAIRS;
        const int need = np * RANDS_PER_PAIR;

        // Warp 0: find the next np active pairs
        if (tid < 32) {
            const int lane = tid;
            int li = curRow, j = curCol, filled = 0;
            while (filled < np) {
                if (j >= n) { ++li; j = 0; }
                int jj = j + lane;
                bool v = jj < n && ((active[size_t(li) * wordsPerRow + (jj >> 5)] >> (jj & 31)) & 1U);
                unsigned b = __ballot_sync(0xffffffffU, v);
                int cnt = __popc(b);
                int take = min(cnt, np - filled);
                int rk = __popc(b & ((1U << lane) - 1U));
                if (v && rk < take) {
                    pairRow[filled + rk] = li;
                    pairCol[filled + rk] = jj;
                }
                filled += take;
                if (take == cnt) {
                    j += 32;
                } else {
                    unsigned nb = __ballot_sync(0xffffffffU, v && rk == take);
                    j += __ffs(nb) - 1;
                }
            }
            if (lane == 0) { curRow = li; curCol = j; }
        }

        // Generate random numbers until the batch is covered
        while (have < need) {
            const uint32_t* o = mt[cur];
            uint32_t* w = mt[cur ^ 1];
            if (tid < MT_N - MT_M) {
                int k = tid;
                uint32_t y = (o[k] & UPPER_MASK) | (o[k + 1] & LOWER_MASK);
                w[k] = o[k + MT_M] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
            }
            __syncthreads();
            if (tid < MT_N - MT_M) {
                int k = tid + (MT_N - MT_M);
                uint32_t y = (o[k] & UPPER_MASK) | (o[k + 1] & LOWER_MASK);
                w[k] = w[k - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
            }
            __syncthreads();
            {
                int k = tid + 2 * (MT_N - MT_M);
                if (k < MT_N - 1) {
                    uint32_t y = (o[k] & UPPER_MASK) | (o[k + 1] & LOWER_MASK);
                    w[k] = w[k - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
                } else if (k == MT_N - 1) {
                    uint32_t y = (o[k] & UPPER_MASK) | (w[0] & LOWER_MASK);
                    w[k] = w[MT_M - 1] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
                }
            }
            __syncthreads();
            const int h = have;
            for (int k = tid; k < MT_N; k += FF_THREADS) rbuf[h + k] = mtToFloat(mtTemper(w[k]));
            cur ^= 1;
            __syncthreads();
            if (tid == 0) have = h + MT_N;
            __syncthreads();
        }
        __syncthreads();

        // Rays are handed out to warps dynamically to balance traversal costs
        {
            const int lane = tid & 31;
            const int numRays = np * NUM_RAYS;
            while (true) {
                int base = 0;
                if (lane == 0) base = atomicAdd(&nextRay, 32);
                base = __shfl_sync(0xffffffffU, base, 0);
                if (base >= numRays) break;
                const int ray = base + lane;
                if (ray < numRays) {
                    const int p = ray / NUM_RAYS;
                    const int r = ray % NUM_RAYS;
                    float c = 0.0f;
                    const int li = pairRow[p];
                    const int i = rowBegin + li;
                    const int j = pairCol[p];
                    const DevTri tI = tris[i];
                    const DevTri tJ = tris[j];
                    const float* rr = &rbuf[p * RANDS_PER_PAIR + r * 4];
                    float u = rr[0], v = rr[1];
                    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                    float pIx = tI.ax + tI.e1x * u + tI.e2x * v;
                    float pIy = tI.ay + tI.e1y * u + tI.e2y * v;
                    float pIz = tI.az + tI.e1z * u + tI.e2z * v;
                    u = rr[2]; v = rr[3];
                    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                    float pJx = tJ.ax + tJ.e1x * u + tJ.e2x * v;
                    float pJy = tJ.ay + tJ.e1y * u + tJ.e2y * v;
                    float pJz = tJ.az + tJ.e1z * u + tJ.e2z * v;

                    if (!isRayBlockedDev(pIx, pIy, pIz, pJx, pJy, pJz, nodes, nodeTris, leafGeo, numNodes, i, j)) {
                        float vx = pJx - pIx, vy = pJy - pIy, vz = pJz - pIz;
                        float distSqr = vx * vx + vy * vy + vz * vz;
                        if (!(distSqr < EPSILON)) {
                            float vNorm = sqrtf(vx * vx + vy * vy + vz * vz);
                            float cosI = 0.0f, cosJ = 0.0f;
                            if (vNorm > EPSILON) {
                                cosI = fmaxf(0.0f, (vx * tI.nx + vy * tI.ny + vz * tI.nz) / vNorm);
                                cosJ = fmaxf(0.0f, (-vx * tJ.nx + -vy * tJ.ny + -vz * tJ.nz) / vNorm);
                            }
                            if (cosI > 0.0f && cosJ > 0.0f) c = (cosI * cosJ) / (PI * distSqr);
                        }
                    }
                    contrib[ray] = c;
                }
            }
        }
        __syncthreads();

        if (tid < np) {
            float kij = 0.0f;
            for (int r = 0; r < NUM_RAYS; ++r) kij += contrib[tid * NUM_RAYS + r];
            const int j = pairCol[tid];
            kT[(size_t(j >> 2) * rows + pairRow[tid]) * 4 + (j & 3)] = kij * INV_NUM_RAYS;
        }

        // Drop consumed random numbers
        {
            const int left = have - need;
            float tmp[(mtjump::MT_N + FF_THREADS - 1) / FF_THREADS];
            int cntv = 0;
            for (int k = tid; k < left; k += FF_THREADS) tmp[cntv++] = rbuf[need + k];
            __syncthreads();
            cntv = 0;
            for (int k = tid; k < left; k += FF_THREADS) rbuf[k] = tmp[cntv++];
            if (tid == 0) {
                have = left;
                nextRay = 0;
            }
        }
        remaining -= np;
        __syncthreads();
    }
}

// Per-rank (n x rows) matrices are stored interleaved in groups of 4 columns
// j, so that each thread (row) issues coalesced 16-byte loads:
//   element (li, j) -> ((j / 4) * rows + li) * 4 + j % 4

// Converts form factors into propagation weights min(Kij * A_j, 1) (0 if Kij <= 0)
// and counts the non-zero form factors.
__global__ void weightKernel(float* __restrict__ kT, const float* __restrict__ areas,
                             size_t total, int n, int rows, unsigned long long* __restrict__ nonZero) {
    unsigned long long cnt = 0;
    for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
         idx += size_t(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(idx / (size_t(rows) * 4)) * 4 + static_cast<int>(idx & 3);
        float kij = kT[idx];
        if (kij > EPSILON) ++cnt;
        float w = 0.0f;
        if (j < n && !(kij <= 0.0f)) {
            float m = kij * areas[j];
            w = (ONE < m) ? ONE : m;
        }
        kT[idx] = w;
    }
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_down_sync(0xffffffffU, cnt, off);
    if ((threadIdx.x & 31) == 0 && cnt) atomicAdd(nonZero, cnt);
}

// One timestep of wave propagation for the rows owned by this rank. The sum
// over j is accumulated in the original sequential order.
__global__ void simStepKernel(int t, int n, int rowBegin, int rows,
                              const float4* __restrict__ w4, const uchar4* __restrict__ tau4,
                              const float* __restrict__ radB, const float* __restrict__ radE,
                              const float* __restrict__ rho, float* __restrict__ out) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= rows) return;
    int i = rowBegin + li;
    const int jBlocks = (n + 3) / 4;
    float sumB = 0.0f;
    auto accumulate = [&](int j, int tauij, float w) {
        if (t < tauij) return;
        if (w == 0.0f) return;
        float radJ = __ldg(&radB[size_t(t - tauij) * n + j]);
        if (radJ <= 0.0f) return;
        sumB += w * radJ;
    };
    #pragma unroll 4
    for (int jb = 0; jb < jBlocks; ++jb) {
        const size_t e = size_t(jb) * rows + li;
        const uchar4 tq = __ldg(&tau4[e]);
        const float4 w = __ldg(&w4[e]);
        const int j = jb * 4;
        accumulate(j, tq.x, w.x);
        accumulate(j + 1, tq.y, w.y);
        accumulate(j + 2, tq.z, w.z);
        accumulate(j + 3, tq.w, w.w);
    }
    out[li] = rho[i] * sumB + radE[size_t(t) * n + i];
}

__global__ void distanceKernel(int n, int T, int rowBegin, int rows, int src,
                               const float* __restrict__ radB, float* __restrict__ dist) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= rows) return;
    int i = rowBegin + li;
    float maxCorr = 0.0f;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[size_t(tt) * n + i];
            float pS = radB[size_t(tt - t) * n + src];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    dist[li] = WAVE_SPEED * static_cast<float>(bestT);
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
    std::vector<val_t> distances;   // Computed distances from source (rank 0)
    unsigned long long nonZeroKij = 0;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Parallel decomposition: this rank owns rows [rowBegin, rowBegin + rows)
    int rank = 0, nranks = 1;
    int rowBegin = 0, rows = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    DevTri* dTris = nullptr;
    DevNode* dNodes = nullptr;
    int2* dNodeTris = nullptr;
    DevLeafTri* dLeafGeo = nullptr;
    int numNodes = 0;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;
    float* dW = nullptr;             // kij -> weights (interleaved n x rows)
    uint8_t* dTau = nullptr;         // time delays (interleaved n x rows)
    float* dRowOut = nullptr;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

static float intBits(int v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

// Flattens the octree in depth-first preorder with skip indices
static void flattenOctree(const Octree& node, const std::vector<DevTri>& tris,
                          std::vector<DevNode>& nodes, std::vector<int2>& nodeTris,
                          std::vector<DevLeafTri>& leafGeo) {
    const size_t id = nodes.size();
    nodes.emplace_back();
    nodeTris.push_back(make_int2(static_cast<int>(leafGeo.size()),
                                 static_cast<int>(node.triangleIndices.size())));
    if (!node.triangleIndices.empty()) {
        for (size_t idx : node.triangleIndices) {
            const DevTri& t = tris[idx];
            leafGeo.push_back({make_float4(t.ax, t.ay, t.az, t.e1x),
                               make_float4(t.e1y, t.e1z, t.e2x, t.e2y),
                               make_float4(t.e2z, intBits(static_cast<int>(idx)), 0.0f, 0.0f)});
        }
    } else {
        for (int c = 0; c < 8; ++c)
            if (node.children[c]) flattenOctree(*node.children[c], tris, nodes, nodeTris, leafGeo);
    }
    nodes[id].a = make_float4(node.center.x, node.center.y, node.center.z, node.halfExtent.x);
    nodes[id].b = make_float4(node.halfExtent.y, node.halfExtent.z,
                              intBits(static_cast<int>(nodes.size())), 0.0f);
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.rank == 0;
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    const size_t n = state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", n);

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    state.radE.resize(timesteps * n, ZERO);
    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Row decomposition across ranks
    state.rowCounts.resize(state.nranks);
    state.rowDispls.resize(state.nranks);
    for (int r = 0, off = 0; r < state.nranks; ++r) {
        int cnt = static_cast<int>(n / state.nranks + (static_cast<size_t>(r) < n % state.nranks ? 1 : 0));
        state.rowCounts[r] = cnt;
        state.rowDispls[r] = off;
        off += cnt;
    }
    state.rowBegin = state.rowDispls[state.rank];
    state.rows = state.rowCounts[state.rank];

    // Device copies of the geometry
    std::vector<DevTri> dtris(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 e1 = t.b - t.a, e2 = t.c - t.a, nn = t.normal();
        dtris[i] = {t.a.x, t.a.y, t.a.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, nn.x, nn.y, nn.z};
    }
    std::vector<DevNode> nodes;
    std::vector<int2> nodeTris;
    std::vector<DevLeafTri> leafGeo;
    flattenOctree(state.octree, dtris, nodes, nodeTris, leafGeo);
    if (leafGeo.empty()) leafGeo.push_back({});
    state.numNodes = static_cast<int>(nodes.size());

    CUDA_CHECK(cudaMalloc(&state.dTris, n * sizeof(DevTri)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&state.dNodeTris, nodeTris.size() * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&state.dLeafGeo, leafGeo.size() * sizeof(DevLeafTri)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, timesteps * n * sizeof(float) + sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * n * sizeof(float) + sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRowOut, (state.rows + 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.dTris, dtris.data(), n * sizeof(DevTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, nodes.data(), nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodeTris, nodeTris.data(), nodeTris.size() * sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dLeafGeo, leafGeo.data(), leafGeo.size() * sizeof(DevLeafTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), timesteps * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * n * sizeof(float)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================
//
// The form factor kernels are launched asynchronously first; the host then
// computes the time delays while the GPU traces rays.

// Number of OpenMP threads worth waking up for a loop of the given size
[[maybe_unused]] static int ompThreads(size_t work) {
    const size_t perThread = 1 << 16;
    return static_cast<int>(std::clamp<size_t>(work / perThread, 1, static_cast<size_t>(omp_get_max_threads())));
}

static size_t localMatrixElems(const SimulationState& state) {
    return size_t((state.numTriangles + 3) / 4) * 4 * state.rows;
}

struct FormFactorLaunch {
    uint32_t* dStates = nullptr;
    uint32_t* dActive = nullptr;
    int* dStartRow = nullptr;
    int* dStartCol = nullptr;
    long long* dPairCount = nullptr;
    unsigned long long* dNonZero = nullptr;
};

FormFactorLaunch launchFormFactors(SimulationState& state) {
    FormFactorLaunch L;
    const int n = static_cast<int>(state.numTriangles);
    const int rows = state.rows;
    const int rowBegin = state.rowBegin;
    const int wordsPerRow = (n + 31) / 32;

    // Characteristic polynomial of MT19937 (independent of the work split)
    std::unique_ptr<mtjump::Jumper> jumper;
    std::thread jumperThread([&jumper] { jumper = std::make_unique<mtjump::Jumper>(42); });

    // Pairs that consume random numbers: i != j and not culled (same test as host code)
    const size_t activeWords = size_t(std::max(rows, 1)) * wordsPerRow;
    std::unique_ptr<uint32_t[]> active(new uint32_t[activeWords]);
    std::vector<long long> rowPairs(rows + 1, 0);
    #pragma omp parallel for schedule(dynamic, 4) num_threads(ompThreads(size_t(rows) * n))
    for (int li = 0; li < rows; ++li) {
        const int i = rowBegin + li;
        const Triangle& triI = state.triangles[i];
        uint32_t* row = &active[size_t(li) * wordsPerRow];
        std::fill(row, row + wordsPerRow, 0U);
        long long cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const Triangle& triJ = state.triangles[j];
            if (triI.normal().dot(triJ.normal()) > 0.99f) continue;
            row[j >> 5] |= 1U << (j & 31);
            ++cnt;
        }
        rowPairs[li + 1] = cnt;
    }
    for (int li = 0; li < rows; ++li) rowPairs[li + 1] += rowPairs[li];
    const long long localPairs = rowPairs[rows];
    long long globalStart = 0;
    MPI_Exscan(&localPairs, &globalStart, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (state.rank == 0) globalStart = 0;

    // Split this rank's pairs into substreams (one per CUDA block)
    int numSM = 1;
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev));
    }
    const long long maxStreams = static_cast<long long>(numSM) * 24;
    const long long minPairsPerStream = 32;
    long long S = std::min(maxStreams, (localPairs + minPairsPerStream - 1) / minPairsPerStream);
    S = std::max<long long>(S, 1);
    const long long Q = (localPairs + S - 1) / S;
    if (localPairs > 0) S = (localPairs + Q - 1) / Q;

    std::vector<int> startRow(S), startCol(S);
    std::vector<long long> pairCount(S);
    #pragma omp parallel for schedule(static) num_threads(ompThreads(size_t(S) * wordsPerRow))
    for (long long s = 0; s < S; ++s) {
        long long q = s * Q;
        pairCount[s] = std::min(Q, localPairs - q);
        int li = static_cast<int>(std::upper_bound(rowPairs.begin(), rowPairs.end(), q) - rowPairs.begin()) - 1;
        li = std::min(std::max(li, 0), std::max(rows - 1, 0));
        long long k = q - rowPairs[li];
        int j = 0;
        if (rows > 0) {
            const uint32_t* row = &active[size_t(li) * wordsPerRow];
            int w = 0;
            while (w < wordsPerRow && __builtin_popcount(row[w]) <= k) k -= __builtin_popcount(row[w++]);
            j = w * 32;
            if (w < wordsPerRow) {
                uint32_t bits = row[w];
                for (; k > 0 || !(bits & 1U); bits >>= 1, ++j)
                    if (bits & 1U) --k;
            }
        }
        startRow[s] = li;
        startCol[s] = j;
    }
    jumperThread.join();

    // Start states of all substreams via MT jump-ahead: the jump polynomials
    // are computed on the host, the jumps themselves on the GPU along a binary
    // tree (state[s] = jump(state[s - lowbit(s)], lowbit(s) * stride)).
    CUDA_CHECK(cudaMalloc(&L.dStates, size_t(S) * mtjump::MT_N * sizeof(uint32_t)));
    if (localPairs > 0) {
        const uint64_t baseOffset = static_cast<uint64_t>(globalStart) * RANDS_PER_PAIR;
        const uint64_t stride = static_cast<uint64_t>(Q) * RANDS_PER_PAIR;
        int levels = 0;
        while ((1LL << levels) < S) ++levels;
        std::vector<mtjump::Poly> polys(levels + 1);  // [0] base, [1 + k] level k
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            polys[0] = jumper->powX(baseOffset);
            #pragma omp section
            {
                if (levels > 0) {
                    polys[1] = jumper->powX(stride);
                    for (int k = 1; k < levels; ++k) polys[1 + k] = jumper->sqrMod(polys[k]);
                }
            }
        }
        std::vector<uint64_t> flat(size_t(levels + 1) * mtjump::PW);
        for (int k = 0; k <= levels; ++k)
            std::copy(polys[k].begin(), polys[k].end(), flat.begin() + size_t(k) * mtjump::PW);
        mtjump::MTState seed = mtjump::seedState(42);

        uint64_t* dPolys = nullptr;
        uint32_t* dSeed = nullptr;
        CUDA_CHECK(cudaMalloc(&dPolys, flat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dSeed, mtjump::MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(dPolys, flat.data(), flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dSeed, seed.data(), mtjump::MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFuncSetAttribute(mtJumpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(JUMP_SMEM)));
        mtJumpKernel<<<1, JUMP_THREADS, JUMP_SMEM>>>(dPolys, dSeed, 0, L.dStates, 0);
        CUDA_CHECK(cudaGetLastError());
        for (int k = levels - 1; k >= 0; --k) {
            const long long step = 1LL << k;
            const long long count = (S - step + 2 * step - 1) / (2 * step);
            const size_t stateStride = size_t(2 * step) * mtjump::MT_N;
            mtJumpKernel<<<static_cast<unsigned>(count), JUMP_THREADS, JUMP_SMEM>>>(
                dPolys + size_t(1 + k) * mtjump::PW, L.dStates, stateStride,
                L.dStates + size_t(step) * mtjump::MT_N, stateStride);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(dPolys));  // implicitly ordered after the kernels
        CUDA_CHECK(cudaFree(dSeed));
    }

    // Form factors (asynchronous)
    const size_t localElems = localMatrixElems(state);
    CUDA_CHECK(cudaMalloc(&state.dW, localElems * sizeof(float) + sizeof(float4)));
    CUDA_CHECK(cudaMemset(state.dW, 0, localElems * sizeof(float) + sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&L.dNonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(L.dNonZero, 0, sizeof(unsigned long long)));

    if (localPairs > 0) {
        CUDA_CHECK(cudaMalloc(&L.dActive, activeWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&L.dStartRow, S * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&L.dStartCol, S * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&L.dPairCount, S * sizeof(long long)));
        CUDA_CHECK(cudaMemcpy(L.dActive, active.get(), activeWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(L.dStartRow, startRow.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(L.dStartCol, startCol.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(L.dPairCount, pairCount.data(), S * sizeof(long long), cudaMemcpyHostToDevice));

        formFactorKernel<<<static_cast<unsigned>(S), FF_THREADS>>>(
            L.dStates, L.dStartRow, L.dStartCol, L.dPairCount, L.dActive, wordsPerRow,
            state.dTris, state.dNodes, state.dNodeTris, state.dLeafGeo, state.numNodes,
            n, rowBegin, rows, state.dW);
        CUDA_CHECK(cudaGetLastError());
    }
    if (localElems > 0) {
        weightKernel<<<numSM * 8, 256>>>(state.dW, state.dAreas, localElems, n, rows, L.dNonZero);
        CUDA_CHECK(cudaGetLastError());
    }
    return L;
}

void finishFormFactors(SimulationState& state, FormFactorLaunch& L) {
    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpy(&localNonZero, L.dNonZero, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(L.dNonZero));
    CUDA_CHECK(cudaFree(L.dStates));
    CUDA_CHECK(cudaFree(L.dActive));
    CUDA_CHECK(cudaFree(L.dStartRow));
    CUDA_CHECK(cudaFree(L.dStartCol));
    CUDA_CHECK(cudaFree(L.dPairCount));
    MPI_Allreduce(&localNonZero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (state.rank == 0) {
        for (size_t i = 0; i < state.numTriangles; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    const int n = static_cast<int>(state.numTriangles);
    const int rows = state.rows;
    const int rowBegin = state.rowBegin;
    const int jBlocks = (n + 3) / 4;
    const size_t localElems = localMatrixElems(state);

    // Stored as bytes (delays are bounded by the room diameter) in the
    // interleaved column layout used by the simulation kernel
    std::unique_ptr<uint8_t[]> tau(new uint8_t[localElems + 4]);
    int overflow = 0;
    #pragma omp parallel for schedule(static) reduction(| : overflow) num_threads(ompThreads(localElems))
    for (int jb = 0; jb < jBlocks; ++jb) {
        uint8_t* blk = &tau[size_t(jb) * rows * 4];
        for (int li = 0; li < rows; ++li) {
            const int i = rowBegin + li;
            for (int q = 0; q < 4; ++q) {
                const int j = jb * 4 + q;
                int t = 0;
                if (j < n && i != j) {
                    t = computeTau(state.triangles[i], state.triangles[j]);
                    if (t < 0 || t > 255) overflow = 1;
                }
                blk[size_t(li) * 4 + q] = static_cast<uint8_t>(t);
            }
        }
    }
    if (overflow) {
        fprintf(stderr, "Time delay exceeds storage range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaMalloc(&state.dTau, localElems + 4));
    CUDA_CHECK(cudaMemcpy(state.dTau, tau.get(), localElems + 4, cudaMemcpyHostToDevice));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = state.rank == 0;
    if (root) printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int rows = state.rows;
    const int threads = 128;
    const int blocks = std::max(1, (rows + threads - 1) / threads);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        float* rowT = &state.radB[t * n];
        if (rows > 0) {
            simStepKernel<<<blocks, threads>>>(static_cast<int>(t), n, state.rowBegin, rows,
                                               reinterpret_cast<const float4*>(state.dW),
                                               reinterpret_cast<const uchar4*>(state.dTau),
                                               state.dRadB, state.dRadE,
                                               state.dRho, state.dRowOut);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(rowT + state.rowBegin, state.dRowOut, rows * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        if (state.nranks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rowT, state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        }
        CUDA_CHECK(cudaMemcpy(state.dRadB + t * n, rowT, n * sizeof(float), cudaMemcpyHostToDevice));

        if (root && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    const int rows = state.rows;
    std::vector<float> local(std::max(rows, 1));
    if (rows > 0) {
        float* dDist = nullptr;
        CUDA_CHECK(cudaMalloc(&dDist, rows * sizeof(float)));
        distanceKernel<<<(rows + 127) / 128, 128>>>(
            static_cast<int>(state.numTriangles), static_cast<int>(state.numTimesteps),
            state.rowBegin, rows, static_cast<int>(state.sourceIndex), state.dRadB, dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), dDist, rows * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dDist));
    }
    MPI_Gatherv(local.data(), rows, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
                state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void releaseDevice(SimulationState& state) {
    cudaFree(state.dTris);
    cudaFree(state.dNodes);
    cudaFree(state.dNodeTris);
    cudaFree(state.dLeafGeo);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dW);
    cudaFree(state.dTau);
    cudaFree(state.dRowOut);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = rank == 0;

    // One GPU per rank (round-robin over the devices of a node), and the
    // node's cores shared among its ranks unless OMP_NUM_THREADS is given.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }
    // Start the OpenMP thread pool outside of the timed regions
    #pragma omp parallel
    {
    }

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
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nranks = nranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    if (root) {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");
    }
    FormFactorLaunch ffLaunch = launchFormFactors(state);  // GPU, asynchronous
    computeTimeDelays(state);                              // host, overlapped
    finishFormFactors(state, ffLaunch);
    CUDA_CHECK(cudaDeviceSynchronize());
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

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    releaseDevice(state);

    int exitCode = 0;
    if (root) {
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
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}

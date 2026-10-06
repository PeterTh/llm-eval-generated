/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *  - MPI: the rows i (receiving triangles) of the N x N Kij/Tau matrices are
 *    block-distributed over ranks. Each simulation timestep exchanges the newly
 *    computed radiosity row with MPI_Allgatherv.
 *  - CUDA: form factors (ray casting through a flattened octree), the wave
 *    propagation and the cross-correlation run on one GPU per rank.
 *  - OpenMP: host-side precomputation (time delays, mesh attributes).
 *
 * The form factors of the sequential code consume one std::mt19937 stream
 * (seed 42) in row-major pair order. To reproduce that stream exactly in
 * parallel, the MT19937 characteristic polynomial is derived (Berlekamp-Massey)
 * and GPU blocks jump directly to their position in the stream (GF(2)
 * polynomial jump-ahead), so every pair sees the very same random numbers.
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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define HD __host__ __device__

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
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;  // Random numbers consumed per (non-culled) pair

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

// Flattened octree node used on the GPU
struct alignas(16) FlatNode {
    Vec3 center, halfExtent;
    int triStart;   // Offset into the leaf triangle list
    int triCount;   // > 0 for leaves
    int child[8];   // -1 if absent
};

// Leaf triangle record for ray tests: v0, e1 = v1 - v0, e2 = v2 - v0 and the triangle index
struct alignas(16) LeafTri {
    float4 a;   // v0.x v0.y v0.z e1.x
    float4 b;   // e1.y e1.z e2.x e2.y
    float4 c;   // e2.z idx  -    -
};

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

    // Flatten the tree into arrays (pre-order); returns the node index and
    // tracks the maximum depth.
    int flatten(std::vector<FlatNode>& nodes, std::vector<int>& leafTris,
                int depth, int& maxDepth) const {
        maxDepth = std::max(maxDepth, depth);
        int self = static_cast<int>(nodes.size());
        nodes.emplace_back();
        FlatNode fn;
        fn.center = center;
        fn.halfExtent = halfExtent;
        fn.triStart = static_cast<int>(leafTris.size());
        fn.triCount = static_cast<int>(triangleIndices.size());
        for (size_t idx : triangleIndices) leafTris.push_back(static_cast<int>(idx));
        for (int i = 0; i < 8; ++i) {
            fn.child[i] = (triangleIndices.empty() && children[i])
                              ? children[i]->flatten(nodes, leafTris, depth + 1, maxDepth)
                              : -1;
        }
        nodes[self] = fn;
        return self;
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
// Random Number Generation: MT19937 jump-ahead
// ============================================================================
//
// The sequential code draws from std::mt19937(42) through
// std::uniform_real_distribution<float>(0, 1), which consumes exactly one
// 32-bit output per draw. Raw MT words satisfy x[k+624] = f(x[k], x[k+1], x[k+397]);
// draw number m is temper(x[624 + m]). A "state window" at position n is
// x[n .. n+623]; the window at n = m is the state right before draw m.
//
// Jumping by J steps: with q(x) = x^J mod P(x) (P = characteristic polynomial,
// degree 19937), window(n + J)[w] = XOR_{i : q_i = 1} x[n + i + w].

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DEG = 19937;
constexpr int MT_POLY_W64 = 312;       // 64-bit words for a polynomial of degree < 19968
constexpr int MT_POLY_W32 = 2 * MT_POLY_W64;
constexpr uint32_t MT_SEED = 42;

HD inline uint32_t mtTwistWord(uint32_t xk, uint32_t xk1, uint32_t xkm) {
    uint32_t y = (xk & 0x80000000u) | (xk1 & 0x7fffffffu);
    return xkm ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

class MtJumpTables {
public:
    std::vector<uint64_t> charPoly;   // P(x), MT_DEG + 1 coefficients
    std::vector<uint32_t> powPolys;   // x^(2^b) mod P for b = 0..numPolys-1 (MT_POLY_W32 words each)
    int numPolys = 0;

    static std::vector<uint32_t> seedWindow() {
        std::vector<uint32_t> mt(MT_N);
        mt[0] = MT_SEED;
        for (uint32_t i = 1; i < MT_N; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
        return mt;
    }

    void build(int bitsNeeded) {
        computeCharPoly();
        buildReductionTable();
        numPolys = std::max(bitsNeeded, 1);
        powPolys.assign(static_cast<size_t>(numPolys) * MT_POLY_W32, 0);
        std::vector<uint64_t> cur(MT_POLY_W64, 0);
        for (int b = 0; b < numPolys; ++b) {
            if (b < 63 && (1ULL << b) < static_cast<uint64_t>(MT_DEG)) {
                std::fill(cur.begin(), cur.end(), 0);
                uint64_t e = 1ULL << b;
                cur[e >> 6] = 1ULL << (e & 63);
            } else {
                cur = squareMod(cur);
            }
            for (int w = 0; w < MT_POLY_W64; ++w) {
                powPolys[static_cast<size_t>(b) * MT_POLY_W32 + 2 * w] = static_cast<uint32_t>(cur[w]);
                powPolys[static_cast<size_t>(b) * MT_POLY_W32 + 2 * w + 1] = static_cast<uint32_t>(cur[w] >> 32);
            }
        }
    }

private:
    std::vector<uint64_t> redTable;   // 256 multiples of P with distinct top bytes

    static inline bool getBit(const std::vector<uint64_t>& v, size_t i) {
        return (v[i >> 6] >> (i & 63)) & 1ULL;
    }

    static inline uint64_t getWord(const std::vector<uint64_t>& v, size_t pos) {
        size_t wi = pos >> 6, sh = pos & 63;
        uint64_t lo = v[wi] >> sh;
        uint64_t hi = sh ? (v[wi + 1] << (64 - sh)) : 0;
        return lo | hi;
    }

    // dst ^= src << shift (src has nSrc words)
    static inline void xorShifted(uint64_t* dst, const uint64_t* src, int nSrc, size_t shift) {
        size_t ws = shift >> 6, bs = shift & 63;
        if (bs == 0) {
            for (int w = 0; w < nSrc; ++w) dst[w + ws] ^= src[w];
        } else {
            for (int w = 0; w < nSrc; ++w) {
                dst[w + ws] ^= src[w] << bs;
                dst[w + ws + 1] ^= src[w] >> (64 - bs);
            }
        }
    }

    // Berlekamp-Massey over GF(2) on one output bit of mt19937(42)
    void computeCharPoly() {
        const int M = 2 * MT_DEG + 64;
        std::mt19937 gen(MT_SEED);
        const int RW = (M + MT_DEG + 256) / 64 + 2;
        std::vector<uint64_t> R(RW, 0);          // R bit k = s[M-1-k]
        for (int n = 0; n < M; ++n) {
            uint32_t bit = static_cast<uint32_t>(gen() >> 31);
            size_t k = static_cast<size_t>(M - 1 - n);
            if (bit) R[k >> 6] |= 1ULL << (k & 63);
        }
        // Rs[sh][w] = R bits [64w + sh, 64w + sh + 64): aligned windows for the discrepancy
        std::vector<uint64_t> Rs(64 * static_cast<size_t>(RW), 0);
        for (int sh = 0; sh < 64; ++sh)
            for (int w = 0; w + 1 < RW; ++w)
                Rs[static_cast<size_t>(sh) * RW + w] = getWord(R, 64 * static_cast<size_t>(w) + sh);
        const int CW = 2 * (M / 64) + 8;
        std::vector<uint64_t> C(CW, 0), B(CW, 0), T(CW, 0);
        C[0] = 1; B[0] = 1;
        int L = 0, m = 1;
        for (int n = 0; n < M; ++n) {
            size_t off = static_cast<size_t>(M - 1 - n);
            int nw = L / 64 + 1;
            uint64_t acc = 0;
            const uint64_t* win = Rs.data() + (off & 63) * RW + (off >> 6);
            for (int w = 0; w < nw; ++w) acc ^= C[w] & win[w];
            int d = __builtin_popcountll(acc) & 1;
            if (d == 0) {
                ++m;
            } else if (2 * L <= n) {
                T = C;
                xorShifted(C.data(), B.data(), std::min(L / 64 + 2, CW - (m >> 6) - 2), static_cast<size_t>(m));
                L = n + 1 - L;
                B = T;
                m = 1;
            } else {
                xorShifted(C.data(), B.data(), std::min(L / 64 + 2, CW - (m >> 6) - 2), static_cast<size_t>(m));
                ++m;
            }
        }
        if (L != MT_DEG) {
            fprintf(stderr, "MT19937 characteristic polynomial: unexpected degree %d\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        charPoly.assign(MT_POLY_W64, 0);
        for (int k = 0; k <= MT_DEG; ++k) {
            if (getBit(C, static_cast<size_t>(MT_DEG - k))) charPoly[k >> 6] |= 1ULL << (k & 63);
        }
    }

    // Table indexed by the 8 coefficients [MT_DEG, MT_DEG+7] of a multiple m(x)*P(x), deg m < 8
    void buildReductionTable() {
        redTable.assign(256 * MT_POLY_W64, 0);
        for (int mm = 0; mm < 256; ++mm) {
            std::vector<uint64_t> prod(MT_POLY_W64 + 1, 0);
            for (int b = 0; b < 8; ++b)
                if (mm & (1 << b)) xorShifted(prod.data(), charPoly.data(), MT_POLY_W64, b);
            int h = static_cast<int>(getWord(prod, MT_DEG) & 0xff);
            std::copy(prod.begin(), prod.begin() + MT_POLY_W64, redTable.begin() + h * MT_POLY_W64);
        }
    }

    std::vector<uint64_t> squareMod(const std::vector<uint64_t>& a) const {
        std::vector<uint64_t> t(2 * MT_POLY_W64 + 4, 0);
        auto spread = [](uint32_t v) {
            uint64_t x = v;
            x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
            x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
            x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
            x = (x | (x << 2)) & 0x3333333333333333ULL;
            x = (x | (x << 1)) & 0x5555555555555555ULL;
            return x;
        };
        for (int w = 0; w < MT_POLY_W64; ++w) {
            t[2 * w] = spread(static_cast<uint32_t>(a[w]));
            t[2 * w + 1] = spread(static_cast<uint32_t>(a[w] >> 32));
        }
        long top = 2L * (MT_DEG - 1);
        for (; top - 7 >= MT_DEG; top -= 8) {
            size_t p = static_cast<size_t>(top - 7);
            int h = static_cast<int>(getWord(t, p) & 0xff);
            if (h) xorShifted(t.data(), redTable.data() + h * MT_POLY_W64, MT_POLY_W64, p - MT_DEG);
        }
        for (; top >= MT_DEG; --top) {
            if (getBit(t, static_cast<size_t>(top)))
                xorShifted(t.data(), charPoly.data(), MT_POLY_W64, static_cast<size_t>(top - MT_DEG));
        }
        return std::vector<uint64_t>(t.begin(), t.begin() + MT_POLY_W64);
    }
};

// ============================================================================
// Device: MT19937 helpers
// ============================================================================

// In-place twist of a 624-word state in shared memory (requires blockDim.x >= 227)
__device__ void blockTwist(uint32_t* mt) {
    const int tid = threadIdx.x;
    uint32_t v = 0;
    if (tid < 227) v = mtTwistWord(mt[tid], mt[tid + 1], mt[tid + MT_M]);
    __syncthreads();
    if (tid < 227) mt[tid] = v;
    __syncthreads();
    if (tid < 227) { int k = tid + 227; v = mtTwistWord(mt[k], mt[k + 1], mt[k - 227]); }
    __syncthreads();
    if (tid < 227) mt[tid + 227] = v;
    __syncthreads();
    if (tid < 170) { int k = tid + 454; v = mtTwistWord(mt[k], mt[(k + 1) % MT_N], mt[k - 227]); }
    __syncthreads();
    if (tid < 170) mt[tid + 454] = v;
    __syncthreads();
}

// Same mapping as std::generate_canonical<float, 24>(mt19937) in libstdc++
__device__ __forceinline__ float mtToUniform(uint32_t y) {
    float r = __uint2float_rn(y) * 2.3283064365386962890625e-10f;  // / 2^32 (exact)
    return r >= 1.0f ? 0x1.fffffep-1f : r;
}

constexpr int JUMP_THREADS = 640;
constexpr int JUMP_SPAN = 2 * MT_N;   // bits of the jump polynomial processed per step (= 39 words)

// st (shared) <- state advanced by J steps, where poly = x^J mod P.
// buf: 4 * MT_N shared words holding x[n + base .. n + base + 4*MT_N).
__device__ void blockApplyJump(uint32_t* st, const uint32_t* __restrict__ poly, uint32_t* buf, uint32_t* q) {
    const int tid = threadIdx.x;
    for (int k = tid; k < MT_POLY_W32; k += blockDim.x) q[k] = poly[k];
    if (tid < MT_N) { buf[tid] = st[tid]; buf[MT_N + tid] = st[tid]; }
    __syncthreads();
    for (int r = 1; r < 4; ++r) {
        blockTwist(buf + r * MT_N);
        if (r < 3 && tid < MT_N) buf[(r + 1) * MT_N + tid] = buf[r * MT_N + tid];
        __syncthreads();
    }
    uint32_t acc = 0;
    const int numWords = (MT_DEG + 31) / 32;
    for (int base = 0; base < MT_DEG; base += JUMP_SPAN) {
        if (tid < MT_N) {
            const int w0 = base / 32;
            const int w1 = min(w0 + JUMP_SPAN / 32, numWords);
            for (int wi = w0; wi < w1; ++wi) {
                uint32_t bits = q[wi];
                const uint32_t* src = buf + (wi * 32 - base) + tid;
                while (bits) {
                    int bit = __ffs(bits) - 1;
                    bits &= bits - 1;
                    acc ^= src[bit];
                }
            }
        }
        __syncthreads();
        if (base + JUMP_SPAN >= MT_DEG) break;
        // Slide the window by two rounds and generate the next two
        if (tid < MT_N) {
            buf[tid] = buf[2 * MT_N + tid];
            buf[MT_N + tid] = buf[3 * MT_N + tid];
            buf[2 * MT_N + tid] = buf[3 * MT_N + tid];
        }
        __syncthreads();
        blockTwist(buf + 2 * MT_N);
        if (tid < MT_N) buf[3 * MT_N + tid] = buf[2 * MT_N + tid];
        __syncthreads();
        blockTwist(buf + 3 * MT_N);
    }
    if (tid < MT_N) st[tid] = acc;
    __syncthreads();
}

// out <- window(src advanced by `offset` steps), using x^(2^b) mod P for every set bit b
__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(const uint32_t* __restrict__ src, unsigned long long offset,
             const uint32_t* __restrict__ polys, uint32_t* __restrict__ out) {
    __shared__ uint32_t st[MT_N];
    __shared__ uint32_t buf[4 * MT_N];
    __shared__ uint32_t q[MT_POLY_W32];
    const int tid = threadIdx.x;
    if (tid < MT_N) st[tid] = src[tid];
    __syncthreads();
    for (int b = 0; b < 64; ++b) {
        if ((offset >> b) & 1ULL) blockApplyJump(st, polys + static_cast<size_t>(b) * MT_POLY_W32, buf, q);
    }
    if (tid < MT_N) out[tid] = st[tid];
}

// Binary-tree construction of block states: state c = (2k+1) * 2^level is obtained from
// state c - 2^level (already available) by a jump of 2^level strides.
__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpTreeKernel(uint32_t* __restrict__ states, int level, long long numChunks,
                 const uint32_t* __restrict__ poly) {
    __shared__ uint32_t st[MT_N];
    __shared__ uint32_t buf[4 * MT_N];
    __shared__ uint32_t q[MT_POLY_W32];
    const int tid = threadIdx.x;
    const long long c = (2LL * blockIdx.x + 1) << level;
    if (c >= numChunks) return;
    const uint32_t* src = states + static_cast<size_t>(c - (1LL << level)) * MT_N;
    if (tid < MT_N) st[tid] = src[tid];
    __syncthreads();
    blockApplyJump(st, poly, buf, q);
    if (tid < MT_N) states[static_cast<size_t>(c) * MT_N + tid] = st[tid];
}

// ============================================================================
// Device: geometry
// ============================================================================

__device__ __forceinline__ Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Möller-Trumbore ray-triangle intersection (edges precomputed: e1 = v1 - v0, e2 = v2 - v0).
// Returns whether the hit distance lies in (EPSILON, maxDist); evaluated without branches,
// equivalent to the early-exit formulation (a miss yields the maximal float distance).
__device__ __forceinline__ bool rayTriangleBlocks(const Vec3& orig, const Vec3& dir,
                                                  const Vec3& v0, const Vec3& e1, const Vec3& e2,
                                                  val_t maxDist) {
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    val_t dist = e2.dot(qvec) * invDet;
    const bool miss = (fabsf(det) < EPSILON) | (u < 0.0f) | (u > 1.0f) | (v < 0.0f) | (u + v > 1.0f);
    return !miss & (dist > EPSILON) & (dist < maxDist);
}

// Segment p1 -> p2 against a node box; d = (p2 - p1) * 0.5, mid = p1 + d, ad = |d|
__device__ __forceinline__ bool rayIntersectsBox(const float4 nb0, const float4 nb1,
                                                 const Vec3& d, const Vec3& mid, const Vec3& ad) {
    const Vec3 center = {nb0.x, nb0.y, nb0.z};
    const Vec3 h = {nb0.w, nb1.x, nb1.y};
    Vec3 c = mid - center;

    // Separating axis tests, evaluated without branches (same result as early exits)
    const bool sep = (fabsf(c.x) > h.x + ad.x) |
                     (fabsf(c.y) > h.y + ad.y) |
                     (fabsf(c.z) > h.z + ad.z) |
                     (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) |
                     (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) |
                     (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON);
    return !sep;
}

constexpr int OCT_STACK = 64;
constexpr int NODE_CODE_BITS = 11;   // packed node header: start (20 bits) | signed code (11 bits)

// Warp-cooperative visibility test: each lane owns one ray (from -> to, excluding triangles
// src and dst). The warp traverses the octree with a shared stack whose entries carry the
// mask of lanes whose own traversal reaches that node, so every ray tests exactly the
// leaves it would reach on its own. Returns whether this lane's ray is blocked.
// Must be called by all 32 lanes of the warp.
__device__ bool warpIsRayBlocked(bool active, const Vec3& from, const Vec3& to, int srcTriIdx, int dstTriIdx,
                                 const int rootHdr, const float4* __restrict__ childBoxes,
                                 const LeafTri* __restrict__ leafTris, int* stackNode, unsigned* stackMask) {
    const int lane = threadIdx.x & 31;
    const unsigned laneBit = 1u << lane;

    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    const bool degenerate = active && rayLen < EPSILON;
    Vec3 dirNorm = dir / rayLen;
    const val_t maxDist = rayLen - EPSILON;

    // Box test terms are identical for every node
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    unsigned blocked = __ballot_sync(0xffffffffu, degenerate);
    const unsigned live = __ballot_sync(0xffffffffu, active && !degenerate);
    int sp = 0;
    if (live) {
        if (lane == 0) { stackNode[0] = rootHdr; stackMask[0] = live; }
        sp = 1;
    }
    __syncwarp();
    while (sp > 0) {
        --sp;
        const int hdr = stackNode[sp];
        unsigned mask = stackMask[sp] & ~blocked;
        __syncwarp();
        if (!mask) continue;
        const int start = hdr >> NODE_CODE_BITS;
        const int code = (hdr << (32 - NODE_CODE_BITS)) >> (32 - NODE_CODE_BITS);  // sign-extended
        if (code < 0) {
            for (int k = start; k < start - code; ++k) {
                const float4* tp = reinterpret_cast<const float4*>(leafTris + k);
                const float4 ta = __ldg(tp), tb = __ldg(tp + 1), tc = __ldg(tp + 2);
                const int idx = __float_as_int(tc.y);
                bool hit = false;
                if ((mask & laneBit) && idx != srcTriIdx && idx != dstTriIdx) {
                    hit = rayTriangleBlocks(from, dirNorm, Vec3(ta.x, ta.y, ta.z), Vec3(ta.w, tb.x, tb.y),
                                            Vec3(tb.z, tb.w, tc.x), maxDist);  // Ray is blocked
                }
                blocked |= __ballot_sync(0xffffffffu, hit);
                mask &= ~blocked;
                if (!mask) break;
            }
        } else {
            for (int c = start + code - 1; c >= start; --c) {
                const float4 cb0 = __ldg(&childBoxes[2 * c]);
                const float4 cb1 = __ldg(&childBoxes[2 * c + 1]);
                const bool in = (mask & laneBit) && rayIntersectsBox(cb0, cb1, d, mid, ad);
                const unsigned m = __ballot_sync(0xffffffffu, in);
                if (m) {
                    if (lane == 0) { stackNode[sp] = __float_as_int(cb1.z); stackMask[sp] = m; }
                    ++sp;
                }
            }
            __syncwarp();
        }
    }
    return (blocked & laneBit) != 0;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;
}

// Contribution of a single unblocked ray to Kij (0 if the sequential code would `continue`)
__device__ __forceinline__ val_t rayContribution(const Vec3& pI, const Vec3& pJ, const Vec3& nI, const Vec3& nJ) {
    Vec3 v = pJ - pI;
    val_t distSqr = v.squaredNorm();
    if (distSqr < EPSILON) return ZERO;

    val_t cosPhiI = cosPhi(v, nI);
    val_t cosPhiJ = cosPhi(-v, nJ);

    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) return ZERO;

    return (cosPhiI * cosPhiJ) / (PI * distSqr);
}

// Triangles facing the same direction are culled (no random numbers consumed)
__device__ __forceinline__ bool pairActive(const Vec3* __restrict__ normals, int i, int j, int n) {
    return j < n && j != i && !(normals[i].dot(normals[j]) > 0.99f);
}

// ============================================================================
// Device: block scan helper
// ============================================================================

// Exclusive prefix count of `flag` across the block; returns the prefix, total in `total`.
__device__ __forceinline__ int blockScan(bool flag, int* warpSums, int& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nWarps = (blockDim.x + 31) >> 5;
    unsigned bal = __ballot_sync(0xffffffffu, flag);
    int pre = __popc(bal & ((1u << lane) - 1u));
    if (lane == 0) warpSums[warp] = __popc(bal);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int w = 0; w < nWarps; ++w) { int v = warpSums[w]; warpSums[w] = s; s += v; }
        warpSums[32] = s;
    }
    __syncthreads();
    pre += warpSums[warp];
    total = warpSums[32];
    __syncthreads();
    return pre;
}

// ============================================================================
// Device: form factor kernels
// ============================================================================

__global__ void countActivePairsKernel(const Vec3* __restrict__ normals, int n, int* __restrict__ rowCount) {
    __shared__ int warpSums[33];
    const int i = blockIdx.x;
    int cnt = 0;
    for (int j0 = 0; j0 < n; j0 += blockDim.x) {
        int tot;
        blockScan(pairActive(normals, i, j0 + threadIdx.x, n), warpSums, tot);
        cnt += tot;
    }
    if (threadIdx.x == 0) rowCount[i] = cnt;
}

// In-place twist of a 624-word state in shared memory by one warp
__device__ __forceinline__ void warpTwist(uint32_t* mt) {
    const int lane = threadIdx.x & 31;
    uint32_t v[8];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int k = lane + 32 * m;
        if (k < 227) v[m] = mtTwistWord(mt[k], mt[k + 1], mt[k + MT_M]);
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int k = lane + 32 * m;
        if (k < 227) mt[k] = v[m];
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int k = 227 + lane + 32 * m;
        if (k < 454) v[m] = mtTwistWord(mt[k], mt[k + 1], mt[k - 227]);
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int k = 227 + lane + 32 * m;
        if (k < 454) mt[k] = v[m];
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        int k = 454 + lane + 32 * m;
        if (k < MT_N) v[m] = mtTwistWord(mt[k], mt[(k + 1) % MT_N], mt[k - 227]);
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        int k = 454 + lane + 32 * m;
        if (k < MT_N) mt[k] = v[m];
    }
    __syncwarp();
}

// Local Kij (weights) and Tau matrices use a tiled layout: rows are grouped by 32 (one warp
// in the simulation) and each row of a tile holds 16 consecutive columns contiguously.
constexpr int TILE_ROWS = 32;
constexpr int TILE_COLS = 16;
constexpr int TILE_SIZE = TILE_ROWS * TILE_COLS;

struct TileLayout {
    int numColTiles;   // ceil(N / TILE_COLS)

    HD size_t index(int li, int j) const {
        return (static_cast<size_t>(li / TILE_ROWS) * numColTiles + j / TILE_COLS) * TILE_SIZE +
               (li % TILE_ROWS) * TILE_COLS + j % TILE_COLS;
    }
    HD void decode(size_t e, int& li, int& j) const {
        const size_t tile = e / TILE_SIZE;
        const int r = static_cast<int>(e % TILE_SIZE);
        li = static_cast<int>(tile / numColTiles) * TILE_ROWS + r / TILE_COLS;
        j = static_cast<int>(tile % numColTiles) * TILE_COLS + r % TILE_COLS;
    }
    size_t elements(int nLocal) const {
        return static_cast<size_t>((nLocal + TILE_ROWS - 1) / TILE_ROWS) * numColTiles * TILE_SIZE;
    }
};

constexpr int FF_THREADS = 256;
constexpr int FF_WARPS = FF_THREADS / 32;
constexpr long long FF_TARGET_CHUNKS = 32768;   // independent warp chunks per rank
constexpr int PAIRS_PER_STEP = 32 / NUM_RAYS;   // pairs traced together by one warp
static_assert(PAIRS_PER_STEP == 2, "formFactorKernel assumes two pairs (2 x 16 rays) per warp step");

// Each warp handles one chunk of `pairsPerChunk` consecutive non-culled pairs (in the
// sequential row-major order), starting from the matching MT19937 state.
__global__ void __launch_bounds__(FF_THREADS, 4)
formFactorKernel(int n, int rowBegin, TileLayout layout, long long numChunks,
                 long long pairBegin, long long pairEnd, long long pairsPerChunk,
                 const long long* __restrict__ rowOffset, const uint32_t* __restrict__ blockStates,
                 const Vec3* __restrict__ normals, const Triangle* __restrict__ tris,
                 const int rootHdr, const float4* __restrict__ childBoxes,
                 const LeafTri* __restrict__ leafTris, val_t* __restrict__ kijT) {
    __shared__ uint32_t sMt[FF_WARPS][MT_N];
    __shared__ int sStackNode[FF_WARPS][OCT_STACK];
    __shared__ unsigned sStackMask[FF_WARPS][OCT_STACK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const long long chunk = static_cast<long long>(blockIdx.x) * FF_WARPS + warp;
    if (chunk >= numChunks) return;
    uint32_t* mt = sMt[warp];

    const long long pStart = pairBegin + chunk * pairsPerChunk;
    const long long pEnd = min(pStart + pairsPerChunk, pairEnd);

    // Block state from the jump tree, then advance sequentially to this warp's first draw
    for (int k = lane; k < MT_N; k += 32) mt[k] = blockStates[static_cast<size_t>(blockIdx.x) * MT_N + k];
    __syncwarp();
    int pos = MT_N;   // MT_N: the next draw requires a twist
    {
        const long long skip = static_cast<long long>(warp) * pairsPerChunk * DRAWS_PER_PAIR;
        const long long rounds = skip / MT_N;
        for (long long r = 0; r < rounds; ++r) warpTwist(mt);
        const int rest = static_cast<int>(skip - rounds * MT_N);
        if (rest > 0) { warpTwist(mt); pos = rest; }
    }

    // Largest row ci with rowOffset[ci] <= pStart
    int ci = 0;
    if (lane == 0) {
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (rowOffset[mid] <= pStart) lo = mid; else hi = mid - 1;
        }
        ci = lo;
    }
    ci = __shfl_sync(0xffffffffu, ci, 0);

    // Locate the first pair: the k-th active column of row ci
    long long k = pStart - rowOffset[ci];
    int cj = 0;
    unsigned bal = 0;   // remaining active columns of the current window [winBase, winBase + 32)
    int winBase = 0;
    while (true) {
        bal = __ballot_sync(0xffffffffu, pairActive(normals, ci, cj + lane, n));
        int cnt = __popc(bal);
        if (k < cnt) {
            for (long long s = 0; s < k; ++s) bal &= bal - 1;
            winBase = cj;
            cj += 32;
            break;
        }
        k -= cnt;
        cj += 32;
    }
    __syncwarp();

    long long remaining = pEnd - pStart;
    while (remaining > 0) {
        const int nPairs = static_cast<int>(min(static_cast<long long>(PAIRS_PER_STEP), remaining));
        int pi[PAIRS_PER_STEP], pj[PAIRS_PER_STEP];
        #pragma unroll
        for (int h = 0; h < PAIRS_PER_STEP; ++h) {
            pi[h] = 0; pj[h] = 0;
            if (h < nPairs) {
                while (bal == 0) {
                    if (cj >= n) { ++ci; cj = 0; }
                    bal = __ballot_sync(0xffffffffu, pairActive(normals, ci, cj + lane, n));
                    winBase = cj;
                    cj += 32;
                }
                pi[h] = ci;
                pj[h] = winBase + __ffs(bal) - 1;
                bal &= bal - 1;
            }
        }

        // Lane l traces ray (l % 16) of pair (l / 16), using draws 4l .. 4l+3 of this step
        const int need = nPairs * DRAWS_PER_PAIR;
        val_t rnd[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int idx = pos + 4 * lane + q;
            rnd[q] = (idx < MT_N && 4 * lane + q < need) ? mtToUniform(mtTemper(mt[idx])) : ZERO;
        }
        if (pos + need > MT_N) {
            warpTwist(mt);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                int idx = pos + 4 * lane + q - MT_N;
                if (idx >= 0 && 4 * lane + q < need) rnd[q] = mtToUniform(mtTemper(mt[idx]));
            }
            pos = pos + need - MT_N;
        } else {
            pos += need;
        }

        const int h = lane / NUM_RAYS;
        const bool valid = h < nPairs;
        const int i = h ? pi[PAIRS_PER_STEP - 1] : pi[0];
        const int j = h ? pj[PAIRS_PER_STEP - 1] : pj[0];
        const Triangle triI = tris[i];
        const Triangle triJ = tris[j];
        const Vec3 pI = randomPointInTriangle(triI, rnd[0], rnd[1]);
        const Vec3 pJ = randomPointInTriangle(triJ, rnd[2], rnd[3]);
        const bool blocked = warpIsRayBlocked(valid, pI, pJ, i, j, rootHdr, childBoxes, leafTris,
                                              sStackNode[warp], sStackMask[warp]);
        const val_t term = (valid && !blocked) ? rayContribution(pI, pJ, triI.normal(), triJ.normal()) : ZERO;

        // Accumulate rays in sequential order
        val_t kij = ZERO;
        #pragma unroll
        for (int r = 0; r < NUM_RAYS; ++r) kij += __shfl_sync(0xffffffffu, term, (lane & ~(NUM_RAYS - 1)) + r);
        if (valid && (lane % NUM_RAYS) == 0)
            kijT[layout.index(i - rowBegin, j)] = kij * INV_NUM_RAYS;

        remaining -= nPairs;
    }
}

// Count Kij > EPSILON, then replace Kij by the propagation weight min(Kij * A_j, 1)
// (0 where Kij <= 0, which the simulation skips).
__global__ void finalizeWeightsKernel(val_t* __restrict__ kijT, const val_t* __restrict__ areas,
                                      TileLayout layout, size_t total, unsigned long long* __restrict__ nonZero) {
    unsigned long long cnt = 0;
    for (size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<size_t>(gridDim.x) * blockDim.x) {
        val_t kij = kijT[e];
        if (kij > EPSILON) ++cnt;
        val_t w = ZERO;
        if (kij > ZERO) {
            int li, j;
            layout.decode(e, li, j);
            val_t p = kij * areas[j];
            w = (ONE < p) ? ONE : p;
        }
        kijT[e] = w;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    if ((threadIdx.x & 31) == 0 && cnt) atomicAdd(nonZero, cnt);
}

// ============================================================================
// Device: simulation and distances
// ============================================================================

constexpr int SIM_THREADS = 64;

// One thread per local row; columns are accumulated in the sequential order j = 0 .. N-1.
__global__ void __launch_bounds__(SIM_THREADS)
simulationStepKernel(int t, int n, int rowBegin, int nLocal, int numColTiles,
                     const val_t* __restrict__ weight, const uint8_t* __restrict__ tau,
                     val_t* __restrict__ radB, val_t rho, int sourceIdx, int timeOff) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;
    const int i = rowBegin + li;
    const size_t base = static_cast<size_t>(li / TILE_ROWS) * numColTiles * TILE_SIZE + (li % TILE_ROWS) * TILE_COLS;
    const val_t* radPrev = radB;
    val_t sumB = ZERO;
    for (int jb = 0; jb < numColTiles; ++jb) {
        const size_t off = base + static_cast<size_t>(jb) * TILE_SIZE;
        const uint4 tv = __ldg(reinterpret_cast<const uint4*>(tau + off));
        const float4* wp = reinterpret_cast<const float4*>(weight + off);
        const float4 w0 = __ldg(wp), w1 = __ldg(wp + 1), w2 = __ldg(wp + 2), w3 = __ldg(wp + 3);
        const val_t w[TILE_COLS] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                                    w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w};
        const uint32_t tw[4] = {tv.x, tv.y, tv.z, tv.w};
        val_t rad[TILE_COLS];
        #pragma unroll
        for (int jj = 0; jj < TILE_COLS; ++jj) {
            const int tauij = (tw[jj / 4] >> (8 * (jj % 4))) & 0xff;
            const int j = jb * TILE_COLS + jj;
            // Skip if wave hasn't yet propagated from j to i, or no coupling
            rad[jj] = (t >= tauij && w[jj] > ZERO)
                          ? __ldg(&radPrev[static_cast<size_t>(t - tauij) * n + j]) : ZERO;
        }
        #pragma unroll
        for (int jj = 0; jj < TILE_COLS; ++jj) {
            if (rad[jj] > ZERO) sumB += w[jj] * rad[jj];
        }
    }
    val_t radE = (i == sourceIdx && t < timeOff) ? 1.0f : ZERO;
    radB[static_cast<size_t>(t) * n + i] = rho * sumB + radE;
}

__global__ void distancesKernel(int n, int rowBegin, int nLocal, int numT, int sourceIdx,
                                const val_t* __restrict__ radB, val_t* __restrict__ distances) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;
    const int i = rowBegin + li;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numT; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < numT; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD inline int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void timeDelaysKernel(const Triangle* __restrict__ tris, int n, int rowBegin, int nLocal,
                                 TileLayout layout, size_t total, uint8_t* __restrict__ tauT,
                                 int* __restrict__ maxTau) {
    int localMax = 0;
    for (size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<size_t>(gridDim.x) * blockDim.x) {
        int li, j;
        layout.decode(e, li, j);
        const int i = rowBegin + li;
        int tau = 0;   // padding entries have zero weight and are never used
        if (li < nLocal && j < n && i != j) tau = computeTau(tris[i], tris[j]);
        localMax = max(localMax, tau);
        tauT[e] = static_cast<uint8_t>(tau);
    }
    for (int o = 16; o > 0; o >>= 1) localMax = max(localMax, __shfl_down_sync(0xffffffffu, localMax, o));
    if ((threadIdx.x & 31) == 0) atomicMax(maxTau, localMax);
}


// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;             // Reflectivity rho (0.0 to 1.0), identical for all triangles
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), filled on rank 0 for validation
    std::vector<val_t> distances;   // Computed distances from source (all triangles, rank 0)
    unsigned long long nonZeroKij = 0;

    Octree octree;                  // Spatial acceleration structure
    MtJumpTables mtTables;          // MT19937 jump-ahead polynomials

    size_t sourceIndex;

    // MPI decomposition: this rank owns rows [rowBegin, rowBegin + numLocal)
    int rank = 0, numRanks = 1;
    int rowBegin = 0, numLocal = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    Triangle* dTris = nullptr;
    Vec3* dNormals = nullptr;
    int rootHdr;                   // Octree root header, see childBoxes
    float4* dChildBoxes = nullptr; // Per child slot: (center.xyz, half.x), (half.y, half.z, header, -) where
                                   // header = start << NODE_CODE_BITS | code (code signed):
                                   // code < 0: leaf with -code triangles from `start` in the leaf list,
                                   // code > 0: inner node with `code` child slots from `start`
    LeafTri* dLeafTris = nullptr;
    val_t* dAreas = nullptr;
    TileLayout layout;             // Layout of the local matrices
    size_t localElems = 0;         // Elements of a local matrix (including padding)
    val_t* dKijT = nullptr;        // Local Kij (then weights), tiled: layout.index(i - rowBegin, j)
    uint8_t* dTauT = nullptr;      // Local Tau, tiled like dKijT
    val_t* dRadB = nullptr;        // Full radiosity history (T x N)
    val_t* dDist = nullptr;        // Local distances
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.rank == 0;
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    const size_t n = state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", n);

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Row decomposition
    state.rowCounts.resize(state.numRanks);
    state.rowDispls.resize(state.numRanks);
    for (int r = 0; r < state.numRanks; ++r) {
        size_t b = n * r / state.numRanks, e = n * (r + 1) / state.numRanks;
        state.rowDispls[r] = static_cast<int>(b);
        state.rowCounts[r] = static_cast<int>(e - b);
    }
    state.rowBegin = state.rowDispls[state.rank];
    state.numLocal = state.rowCounts[state.rank];

    // Device copies of the geometry
    std::vector<FlatNode> nodes;
    std::vector<int> leafTris;
    int maxDepth = 0;
    state.octree.flatten(nodes, leafTris, 0, maxDepth);
    if (7 * maxDepth + 1 > OCT_STACK) {
        fprintf(stderr, "Octree too deep (%d) for traversal stack\n", maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Child slots are laid out per inner node; a node's header is stored in its parent's slot
    std::vector<int> firstSlot(nodes.size(), 0), numSlots(nodes.size(), 0);
    int slotCount = 0;
    for (size_t k = 0; k < nodes.size(); ++k) {
        firstSlot[k] = slotCount;
        for (int c = 0; c < 8; ++c) numSlots[k] += nodes[k].child[c] >= 0;
        slotCount += numSlots[k];
    }
    auto nodeCode = [&](int k) {
        int start = nodes[k].triCount > 0 ? nodes[k].triStart : firstSlot[k];
        int code = nodes[k].triCount > 0 ? -nodes[k].triCount : numSlots[k];
        if (start >= (1 << (31 - NODE_CODE_BITS)) || code < -(1 << (NODE_CODE_BITS - 1)) ||
            code >= (1 << (NODE_CODE_BITS - 1))) {
            fprintf(stderr, "Octree too large for node header encoding\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return static_cast<int>((static_cast<uint32_t>(start) << NODE_CODE_BITS) |
                                (static_cast<uint32_t>(code) & ((1u << NODE_CODE_BITS) - 1)));
    };
    std::vector<float4> childBoxes;
    childBoxes.reserve(2 * static_cast<size_t>(slotCount));
    for (size_t k = 0; k < nodes.size(); ++k) {
        for (int c = 0; c < 8; ++c) {
            if (nodes[k].child[c] < 0) continue;
            const FlatNode& ch = nodes[nodes[k].child[c]];
            int hdr = nodeCode(nodes[k].child[c]);
            float hdrBits;
            std::memcpy(&hdrBits, &hdr, sizeof(float));
            childBoxes.push_back(make_float4(ch.center.x, ch.center.y, ch.center.z, ch.halfExtent.x));
            childBoxes.push_back(make_float4(ch.halfExtent.y, ch.halfExtent.z, hdrBits, 0.0f));
        }
    }
    state.rootHdr = nodeCode(0);
    if (childBoxes.empty()) childBoxes.resize(2);
    std::vector<LeafTri> leafTriData(leafTris.size());
    for (size_t k = 0; k < leafTris.size(); ++k) {
        const Triangle& tri = state.triangles[leafTris[k]];
        Vec3 e1 = tri.b - tri.a;
        Vec3 e2 = tri.c - tri.a;
        int idx = leafTris[k];
        float idxBits;
        std::memcpy(&idxBits, &idx, sizeof(float));
        leafTriData[k].a = make_float4(tri.a.x, tri.a.y, tri.a.z, e1.x);
        leafTriData[k].b = make_float4(e1.y, e1.z, e2.x, e2.y);
        leafTriData[k].c = make_float4(e2.z, idxBits, 0.0f, 0.0f);
    }
    std::vector<Vec3> normals(n);
    for (size_t i = 0; i < n; ++i) normals[i] = state.triangles[i].normal();

    CUDA_CHECK(cudaMalloc(&state.dTris, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNormals, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&state.dChildBoxes, childBoxes.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dLeafTris, std::max<size_t>(leafTris.size(), 1) * sizeof(LeafTri)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNormals, normals.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dChildBoxes, childBoxes.data(), childBoxes.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dLeafTris, leafTriData.data(), leafTriData.size() * sizeof(LeafTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));

    // Initialize matrices
    state.layout.numColTiles = static_cast<int>((n + TILE_COLS - 1) / TILE_COLS);
    state.localElems = state.layout.elements(state.numLocal);
    const size_t localElems = std::max<size_t>(state.localElems, 1);
    CUDA_CHECK(cudaMalloc(&state.dKijT, localElems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTauT, localElems * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(state.dKijT, 0, localElems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(timesteps * n, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(timesteps * n, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDist, std::max(state.numLocal, 1) * sizeof(val_t)));
    state.distances.resize(n, ZERO);

    // Source emission (active for first half of timesteps) is applied inside the
    // simulation kernel: radE[t][source] = 1 for t < timesteps / 2.
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const bool root = state.rank == 0;
    if (root) printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);

    // Number of non-culled pairs per row -> position of every pair in the random stream
    int* dRowCount = nullptr;
    long long* dRowOffset = nullptr;
    CUDA_CHECK(cudaMalloc(&dRowCount, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dRowOffset, (n + 1) * sizeof(long long)));
    countActivePairsKernel<<<n, 256>>>(state.dNormals, n, dRowCount);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> rowCount(n);
    CUDA_CHECK(cudaMemcpy(rowCount.data(), dRowCount, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<long long> rowOffset(n + 1, 0);
    for (int i = 0; i < n; ++i) rowOffset[i + 1] = rowOffset[i] + rowCount[i];
    CUDA_CHECK(cudaMemcpy(dRowOffset, rowOffset.data(), (n + 1) * sizeof(long long), cudaMemcpyHostToDevice));

    const long long pairBegin = rowOffset[state.rowBegin];
    const long long pairEnd = rowOffset[state.rowBegin + state.numLocal];
    const long long localPairs = pairEnd - pairBegin;
    const long long totalPairs = rowOffset[n];

    if (localPairs > 0) {
        // Chunking: each warp handles a power-of-two number of consecutive pairs; the RNG state of
        // every block (FF_WARPS chunks) comes from the jump tree, warps then advance sequentially
        long long pairsPerChunk = 256;
        while (localPairs / pairsPerChunk > FF_TARGET_CHUNKS) pairsPerChunk *= 2;
        const long long numChunks = (localPairs + pairsPerChunk - 1) / pairsPerChunk;
        const long long numBlocks = (numChunks + FF_WARPS - 1) / FF_WARPS;

        const MtJumpTables& tables = state.mtTables;   // built during computeTimeDelays
        if (static_cast<unsigned long long>(totalPairs) * DRAWS_PER_PAIR >= (1ULL << tables.numPolys) &&
            tables.numPolys < 64) {
            fprintf(stderr, "Insufficient MT19937 jump tables\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<uint32_t> seed = MtJumpTables::seedWindow();
        uint32_t *dPolys = nullptr, *dSeed = nullptr, *dBlockStates = nullptr;
        CUDA_CHECK(cudaMalloc(&dPolys, tables.powPolys.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dSeed, MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dBlockStates, static_cast<size_t>(numBlocks) * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(dPolys, tables.powPolys.data(), tables.powPolys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dSeed, seed.data(), MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // State at this rank's first draw (block 0), then all block starts via a jump tree
        mtJumpKernel<<<1, JUMP_THREADS>>>(dSeed, static_cast<unsigned long long>(pairBegin) * DRAWS_PER_PAIR,
                                          dPolys, dBlockStates);
        CUDA_CHECK(cudaGetLastError());
        int blockLog = 0;
        while ((1LL << blockLog) < pairsPerChunk * FF_WARPS * DRAWS_PER_PAIR) ++blockLog;
        int topLevel = 0;
        while ((2LL << topLevel) < numBlocks) ++topLevel;
        for (int level = topLevel; level >= 0 && numBlocks > 1; --level) {
            long long blocks = (numBlocks - (1LL << level) + (2LL << level) - 1) / (2LL << level);
            if (blocks <= 0) continue;
            mtJumpTreeKernel<<<static_cast<unsigned>(blocks), JUMP_THREADS>>>(
                dBlockStates, level, numBlocks, dPolys + static_cast<size_t>(blockLog + level) * MT_POLY_W32);
            CUDA_CHECK(cudaGetLastError());
        }

        formFactorKernel<<<static_cast<unsigned>(numBlocks), FF_THREADS>>>(
            n, state.rowBegin, state.layout, numChunks, pairBegin, pairEnd, pairsPerChunk,
            dRowOffset, dBlockStates, state.dNormals, state.dTris, state.rootHdr, state.dChildBoxes, state.dLeafTris,
            state.dKijT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(dPolys));
        CUDA_CHECK(cudaFree(dSeed));
        CUDA_CHECK(cudaFree(dBlockStates));
    }

    // Count non-zero form factors and convert Kij into propagation weights
    unsigned long long* dNonZero = nullptr;
    CUDA_CHECK(cudaMalloc(&dNonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dNonZero, 0, sizeof(unsigned long long)));
    if (state.numLocal > 0) {
        finalizeWeightsKernel<<<1024, 256>>>(state.dKijT, state.dAreas, state.layout, state.localElems, dNonZero);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpy(&localNonZero, dNonZero, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&localNonZero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dNonZero));
    CUDA_CHECK(cudaFree(dRowCount));
    CUDA_CHECK(cudaFree(dRowOffset));

    if (root) {
        for (size_t i = 0; i < state.numTriangles; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    const size_t n = state.numTriangles;
    const size_t nLocal = static_cast<size_t>(state.numLocal);

    // Tau on the GPU (asynchronous) ...
    int* dMaxTau = nullptr;
    CUDA_CHECK(cudaMalloc(&dMaxTau, sizeof(int)));
    CUDA_CHECK(cudaMemset(dMaxTau, 0, sizeof(int)));
    if (nLocal > 0) {
        timeDelaysKernel<<<2048, 256>>>(state.dTris, static_cast<int>(n), state.rowBegin,
                                        static_cast<int>(nLocal), state.layout, state.localElems,
                                        state.dTauT, dMaxTau);
        CUDA_CHECK(cudaGetLastError());
    }

    // ... while the host builds the MT19937 jump polynomials x^(2^b) mod P covering every
    // possible stream offset (64 draws per pair, at most N * (N - 1) pairs)
    unsigned long long maxOffset = static_cast<unsigned long long>(n) * (n - 1) * DRAWS_PER_PAIR;
    int bits = 1;
    while (bits < 64 && (maxOffset >> bits) != 0) ++bits;
    state.mtTables.build(bits);

    int maxTau = 0;
    CUDA_CHECK(cudaMemcpy(&maxTau, dMaxTau, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMaxTau));
    if (maxTau > 255) {
        fprintf(stderr, "Time delay %d exceeds storage range\n", maxTau);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = state.rank == 0;
    if (root) printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int nLocal = state.numLocal;
    const int timeOff = static_cast<int>(state.numTimesteps / 2);
    std::vector<val_t> hostRow(state.numRanks > 1 ? n : 0);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        val_t* dRow = state.dRadB + t * n;
        if (nLocal > 0) {
            simulationStepKernel<<<(nLocal + SIM_THREADS - 1) / SIM_THREADS, SIM_THREADS>>>(
                static_cast<int>(t), n, state.rowBegin, nLocal, state.layout.numColTiles,
                state.dKijT, state.dTauT, state.dRadB,
                state.reflectivity, static_cast<int>(state.sourceIndex), timeOff);
            CUDA_CHECK(cudaGetLastError());
        }
        if (state.numRanks > 1) {
            if (nLocal > 0)
                CUDA_CHECK(cudaMemcpy(hostRow.data() + state.rowBegin, dRow + state.rowBegin,
                                      nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostRow.data(), state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(dRow, hostRow.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        if (root && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    const int nLocal = state.numLocal;
    std::vector<val_t> local(std::max(nLocal, 1));
    if (nLocal > 0) {
        distancesKernel<<<(nLocal + 127) / 128, 128>>>(
            static_cast<int>(state.numTriangles), state.rowBegin, nLocal,
            static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex),
            state.dRadB, state.dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), state.dDist, nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(local.data(), nLocal, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
                state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    #pragma omp parallel for schedule(static) reduction(+ : receivedEnergy)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[t * state.numTriangles + i] > EPSILON) {
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

static double wallMs(MPI_Comm comm) {
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    return MPI_Wtime() * 1000.0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = rank == 0;

    // One GPU per rank (round-robin over the node's devices)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0, localSize = 1;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
        MPI_Comm_free(&nodeComm);
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
        // Share the node's cores among the ranks placed on it
        if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
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
    state.numRanks = numRanks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");

    // Precomputation
    double startPre = wallMs(MPI_COMM_WORLD);

    computeTimeDelays(state);
    computeFormFactors(state);

    double endPre = wallMs(MPI_COMM_WORLD);
    double localPreDuration = endPre - startPre;
    double globalPreDuration = 0.0;
    MPI_Reduce(&localPreDuration, &globalPreDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long preDuration = static_cast<long>(globalPreDuration);

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    double startSim = wallMs(MPI_COMM_WORLD);

    runSimulation(state);

    double endSim = wallMs(MPI_COMM_WORLD);
    double localSimDuration = endSim - startSim;
    double globalSimDuration = 0.0;
    MPI_Reduce(&localSimDuration, &globalSimDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long simDuration = static_cast<long>(globalSimDuration);

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    double startDist = wallMs(MPI_COMM_WORLD);

    computeDistances(state);

    double endDist = wallMs(MPI_COMM_WORLD);
    double localDistDuration = endDist - startDist;
    double globalDistDuration = 0.0;
    MPI_Reduce(&localDistDuration, &globalDistDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long distDuration = static_cast<long>(globalDistDuration);

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
            state.radB.resize(t * n);
            if (!state.radB.empty())
                CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, t * n * sizeof(val_t),
                                      cudaMemcpyDeviceToHost));
            if (!validateResults(state)) {
                exitCode = 1;
            }
        }
        fflush(stdout);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cudaFree(state.dTris);
    cudaFree(state.dNormals);
    cudaFree(state.dChildBoxes);
    cudaFree(state.dLeafTris);
    cudaFree(state.dAreas);
    cudaFree(state.dKijT);
    cudaFree(state.dTauT);
    cudaFree(state.dRadB);
    cudaFree(state.dDist);

    MPI_Finalize();
    return exitCode;
}

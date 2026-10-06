/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI + OpenMP + CUDA implementation of room impulse response
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
 *  - Rows (receiver triangles i) are block-distributed over MPI ranks; each rank
 *    drives one GPU (node-local rank -> device).
 *  - Form factors are Monte Carlo estimates drawn from one sequential mt19937
 *    stream (seed 42). To reproduce that exact stream in parallel, every row's
 *    stream offset is derived from per-row pair counts, and the generator state
 *    at that offset is obtained on the host (OpenMP threads) via GF(2)
 *    polynomial jump-ahead. The GPU then expands each row's stream and traces
 *    the visibility rays through a flattened octree.
 *  - The wave propagation is computed on the GPU per timestep for the local rows,
 *    with an MPI_Allgatherv of each new radiosity timestep.
 *  - Cross-correlation distances are computed with OpenMP on the local rows.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <cfloat>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

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
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;  // 2 random points x 2 numbers per ray

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

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    HD Triangle() {}
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
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
// Random Number Generation: reproducible, parallel mt19937 stream
// ============================================================================
//
// The reference algorithm draws all random numbers from one std::mt19937(42)
// with std::uniform_real_distribution<float>(0,1), consuming DRAWS_PER_PAIR
// numbers per non-culled pair (i, j) in row-major order. Each rank reconstructs
// the generator state at the beginning of each of its rows: OpenMP threads
// either discard (short distances) or jump ahead with x^J mod phi(x), where
// phi is the characteristic polynomial of the MT19937 transition (obtained via
// Berlekamp-Massey). The GPU then expands each row's stream.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr int MT_DEG = 19937;
constexpr int MT_PW = (MT_DEG + 63) / 64;   // words of a polynomial with degree < MT_DEG
constexpr uint32_t MT_SEED = 42;
// Below this many draws, a plain sequential discard is cheaper than a jump
constexpr uint64_t MT_JUMP_THRESHOLD = uint64_t(1) << 28;

HD inline uint32_t mtTwistWord(uint32_t x0, uint32_t x1, uint32_t xm) {
    uint32_t y = (x0 & MT_UPPER) | (x1 & MT_LOWER);
    return xm ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
}

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0,1) on top of std::mt19937 (libstdc++
// generate_canonical<float, 24> with a single 32-bit draw)
HD inline float mtToUniform(uint32_t y) {
    float r = static_cast<float>(y) / 4294967296.0f;
    if (r >= 1.0f) r = 0x1.fffffep-1f;
    return r;
}

static void mtTwist(uint32_t* mt) {
    int k = 0;
    for (; k < MT_N - MT_M; ++k) mt[k] = mtTwistWord(mt[k], mt[k + 1], mt[k + MT_M]);
    for (; k < MT_N - 1; ++k) mt[k] = mtTwistWord(mt[k], mt[k + 1], mt[k + MT_M - MT_N]);
    mt[MT_N - 1] = mtTwistWord(mt[MT_N - 1], mt[0], mt[MT_M - 1]);
}

static void mtSeedWindow(uint32_t* mt, uint32_t seed) {
    mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
}

// Host-side generator state: mt holds a window of 624 raw words, p is the
// position of the next output inside it (p == 624: twist required).
struct MTState {
    uint32_t mt[MT_N];
    int p;

    void seed(uint32_t s) { mtSeedWindow(mt, s); p = MT_N; }

    void discard(uint64_t n) {
        uint64_t q = static_cast<uint64_t>(p) + n;
        while (q >= static_cast<uint64_t>(MT_N)) {
            mtTwist(mt);
            q -= MT_N;
        }
        p = static_cast<int>(q);
    }

    uint32_t next() {
        if (p >= MT_N) { mtTwist(mt); p = 0; }
        return mtTemper(mt[p++]);
    }

    // Window whose first word is the raw value of the next output
    void exportWindow(uint32_t* out) const {
        uint32_t tmp[MT_N];
        memcpy(tmp, mt, sizeof(tmp));
        mtTwist(tmp);
        if (p >= MT_N) { memcpy(out, tmp, sizeof(tmp)); return; }
        memcpy(out, mt + p, sizeof(uint32_t) * (MT_N - p));
        memcpy(out + (MT_N - p), tmp, sizeof(uint32_t) * p);
    }
};

using Poly = std::vector<uint64_t>;

static inline int polyBit(const Poly& a, size_t k) { return static_cast<int>((a[k >> 6] >> (k & 63)) & 1u); }

// a ^= b << shift (a must be large enough)
static void polyXorShifted(Poly& a, const Poly& b, size_t shift) {
    size_t ws = shift >> 6, bs = shift & 63;
    for (size_t w = 0; w < b.size(); ++w) {
        if (!b[w]) continue;
        if (w + ws < a.size()) a[w + ws] ^= b[w] << bs;
        if (bs && w + ws + 1 < a.size()) a[w + ws + 1] ^= b[w] >> (64 - bs);
    }
}

class MTJumper {
    Poly phi_;                     // characteristic polynomial, degree MT_DEG
    std::vector<Poly> phiShift_;   // phi << s for s in [0, 64)
    uint32_t seedWin_[MT_N];       // seeded window (x_0 .. x_623)

public:
    MTJumper() {
        // Berlekamp-Massey on the lowest output bit of the generator
        const size_t L2 = 2 * MT_DEG + 64;
        std::vector<uint8_t> s(L2);
        {
            MTState g;
            g.seed(MT_SEED);
            for (size_t n = 0; n < L2; ++n) s[n] = g.next() & 1u;
        }
        const size_t W = L2 / 64 + 4;
        Poly rv(W + 2, 0);  // reversed sequence: rv[k] = s[L2-1-k]
        for (size_t k = 0; k < L2; ++k)
            if (s[L2 - 1 - k]) rv[k >> 6] |= uint64_t(1) << (k & 63);
        auto get64 = [&](size_t pos) {
            size_t w = pos >> 6, b = pos & 63;
            uint64_t v = rv[w] >> b;
            if (b) v |= rv[w + 1] << (64 - b);
            return v;
        };
        Poly C(W, 0), B(W, 0), T;
        C[0] = 1; B[0] = 1;
        size_t L = 0, m = 1;
        for (size_t n = 0; n < L2; ++n) {
            size_t base = L2 - 1 - n;
            uint64_t acc = 0;
            size_t nw = L / 64 + 1;
            for (size_t w = 0; w < nw && base + 64 * w < 64 * (W + 1); ++w) acc ^= C[w] & get64(base + 64 * w);
            int d = __builtin_parityll(acc);
            if (!d) { ++m; continue; }
            if (2 * L <= n) {
                T = C;
                polyXorShifted(C, B, m);
                L = n + 1 - L;
                B = std::move(T);
                m = 1;
            } else {
                polyXorShifted(C, B, m);
                ++m;
            }
        }
        if (L != static_cast<size_t>(MT_DEG)) {
            fprintf(stderr, "MT jump-ahead: unexpected linear complexity %zu\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // phi(x) = x^L C(1/x)
        phi_.assign(MT_PW + 1, 0);
        for (size_t k = 0; k <= L; ++k)
            if (polyBit(C, L - k)) phi_[k >> 6] |= uint64_t(1) << (k & 63);
        phiShift_.resize(64);
        for (int sh = 0; sh < 64; ++sh) {
            phiShift_[sh].assign(MT_PW + 2, 0);
            polyXorShifted(phiShift_[sh], phi_, sh);
        }
        mtSeedWindow(seedWin_, MT_SEED);
    }

    // Reduce a (degree < 2*MT_DEG) modulo phi, result in a[0 .. MT_PW)
    void reduce(Poly& a) const {
        for (size_t w = a.size(); w-- > 0;) {
            while (a[w]) {
                size_t top = w * 64 + 63 - __builtin_clzll(a[w]);
                if (top < static_cast<size_t>(MT_DEG)) break;
                size_t sh = top - MT_DEG;
                const Poly& ps = phiShift_[sh & 63];
                size_t ws = sh >> 6;
                for (size_t k = 0; k < ps.size() && k + ws < a.size(); ++k) a[k + ws] ^= ps[k];
            }
        }
    }

    // x^e mod phi
    Poly powX(uint64_t e) const {
        Poly r(2 * MT_PW + 2, 0);
        r[0] = 1;
        if (e == 0) { r.resize(MT_PW); return r; }
        int topBit = 63 - __builtin_clzll(e);
        for (int b = topBit; b >= 0; --b) {
            // square: spread bits
            Poly sq(2 * MT_PW + 2, 0);
            for (int w = 0; w < MT_PW; ++w) {
                uint64_t v = r[w];
                if (!v) continue;
                for (int half = 0; half < 2; ++half) {
                    uint64_t x = (v >> (32 * half)) & 0xffffffffull;
                    x = (x | (x << 16)) & 0x0000ffff0000ffffull;
                    x = (x | (x << 8)) & 0x00ff00ff00ff00ffull;
                    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0full;
                    x = (x | (x << 2)) & 0x3333333333333333ull;
                    x = (x | (x << 1)) & 0x5555555555555555ull;
                    sq[2 * w + half] = x;
                }
            }
            reduce(sq);
            if ((e >> b) & 1u) {
                // multiply by x
                for (int w = MT_PW; w > 0; --w) sq[w] = (sq[w] << 1) | (sq[w - 1] >> 63);
                sq[0] <<= 1;
                reduce(sq);
            }
            r = std::move(sq);
        }
        r.resize(MT_PW);
        return r;
    }

    // State after `n` outputs of std::mt19937(MT_SEED)
    void jump(uint64_t n, MTState& st) const {
        // S1 = A * S0 (window x_1 .. x_624); then result = A^(623+n) S1
        uint32_t S1[MT_N];
        for (int q = 0; q < MT_N - 1; ++q) S1[q] = seedWin_[q + 1];
        S1[MT_N - 1] = mtTwistWord(seedWin_[0], seedWin_[1], seedWin_[MT_M]);
        Poly g = powX(static_cast<uint64_t>(MT_N - 1) + n);

        // Horner evaluation of g(A) S1 with a circular window
        uint32_t R[MT_N];
        int sR = 0;
        int top = -1;
        for (int k = MT_DEG - 1; k >= 0; --k)
            if (polyBit(g, k)) { top = k; break; }
        if (top < 0) {  // cannot happen for an irreducible phi
            fprintf(stderr, "MT jump-ahead: zero jump polynomial\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memcpy(R, S1, sizeof(R));
        for (int k = top - 1; k >= 0; --k) {
            R[sR] = mtTwistWord(R[sR], R[sR + 1 < MT_N ? sR + 1 : 0], R[(sR + MT_M) % MT_N]);
            sR = sR + 1 < MT_N ? sR + 1 : 0;
            if (polyBit(g, k)) {
                int n1 = MT_N - sR;
                for (int q = 0; q < n1; ++q) R[sR + q] ^= S1[q];
                for (int q = n1; q < MT_N; ++q) R[q - n1] ^= S1[q];
            }
        }
        for (int q = 0; q < MT_N; ++q) st.mt[q] = R[(sR + q) % MT_N];
        st.p = 0;
    }
};

// Generator state positioned `n` outputs after seeding
static void mtStateAt(uint64_t n, MTState& st, const MTJumper* jumper) {
    if (n < MT_JUMP_THRESHOLD || !jumper) {
        st.seed(MT_SEED);
        st.discard(n);
    } else {
        jumper->jump(n, st);
    }
}

// Generate a random point inside a triangle using barycentric coordinates
HD inline Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
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

// Möller-Trumbore with precomputed edges e1 = v1 - v0, e2 = v2 - v0
__device__ inline val_t rayTriangleIntersectEdges(const Vec3& orig, const Vec3& dir,
                                                  const Vec3& v0, const Vec3& e1, const Vec3& e2) {
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = __frcp_rn(det);  // correctly rounded, identical to 1.0f / det
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

struct EdgeTri {
    Vec3 v0, e1, e2;
};

// ============================================================================
// Flattened Octree (GPU)
// ============================================================================

struct FlatNode {
    Vec3 center, halfExtent;
    int child[8];      // -1 if absent
    uint32_t triStart; // leaf triangle range in the leaf index list
    uint32_t triCount; // > 0 for leaves
};

constexpr int OCT_STACK = 128;

struct FlatOctree {
    std::vector<FlatNode> nodes;
    std::vector<uint32_t> leafTris;
    int maxDepth = 0;

    int add(const Octree& o, int depth) {
        maxDepth = std::max(maxDepth, depth);
        int id = static_cast<int>(nodes.size());
        nodes.emplace_back();
        FlatNode n;
        n.center = o.center;
        n.halfExtent = o.halfExtent;
        n.triStart = static_cast<uint32_t>(leafTris.size());
        n.triCount = static_cast<uint32_t>(o.triangleIndices.size());
        for (size_t idx : o.triangleIndices) leafTris.push_back(static_cast<uint32_t>(idx));
        for (int i = 0; i < 8; ++i) n.child[i] = -1;
        if (o.triangleIndices.empty()) {
            for (int i = 0; i < 8; ++i)
                if (o.children[i]) n.child[i] = add(*o.children[i], depth + 1);
        }
        nodes[id] = n;
        return id;
    }
};

// Segment quantities shared by all box tests of one ray (same operations as
// Octree::rayIntersectsBox: d = (p2 - p1) * 0.5, c = p1 + d - center)
struct RaySeg {
    Vec3 d, m, ad;
};

__device__ inline RaySeg makeRaySeg(const Vec3& p1, const Vec3& p2) {
    RaySeg r;
    r.d = (p2 - p1) * 0.5f;
    r.m = p1 + r.d;
    r.ad = {fabsf(r.d.x), fabsf(r.d.y), fabsf(r.d.z)};
    return r;
}

__device__ inline bool rayIntersectsBox(const FlatNode& n, const RaySeg& s) {
    const Vec3& d = s.d;
    const Vec3& ad = s.ad;
    Vec3 c = s.m - n.center;
    const Vec3& h = n.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Warp-cooperative (packet) visibility test. Every lane carries its own ray and
// sees exactly the octree nodes its own box tests would reach (tracked with a
// lane mask per stack entry), so the outcome is identical to an independent
// per-ray traversal, but node/triangle loads are shared and divergence is low.
// Returns true if this lane's ray between two triangles is blocked.
__device__ bool isRayBlockedWarp(bool active, const Vec3& from, const Vec3& to,
                                 const FlatNode* __restrict__ nodes,
                                 const uint32_t* __restrict__ leafTris,
                                 const EdgeTri* __restrict__ etris,
                                 uint32_t srcTriIdx, uint32_t dstTriIdx,
                                 int* stackNode, unsigned* stackMask) {
    const int lane = threadIdx.x & 31;
    Vec3 dirNorm;
    RaySeg seg = makeRaySeg(from, to);
    val_t rayLen = ZERO, maxDist = ZERO;
    bool blocked = false;
    if (active) {
        Vec3 dir = to - from;
        rayLen = dir.norm();
        if (rayLen < EPSILON) {
            blocked = true;
            active = false;
        } else {
            dirNorm = dir / rayLen;
            maxDist = rayLen - EPSILON;
        }
    }
    unsigned todo = __ballot_sync(0xffffffffu, active);
    if (!todo) return blocked;

    int sp = 0;
    if (lane == 0) { stackNode[0] = 0; stackMask[0] = todo; }
    sp = 1;
    __syncwarp();
    while (sp > 0) {
        --sp;
        int node = stackNode[sp];
        unsigned mask = stackMask[sp] & todo;
        __syncwarp();
        if (!mask) continue;
        bool mine = (mask >> lane) & 1u;
        const FlatNode& n = nodes[node];
        if (n.triCount > 0) {
            for (uint32_t k = 0; k < n.triCount; ++k) {
                uint32_t idx = leafTris[n.triStart + k];
                if (mine && !blocked && idx != srcTriIdx && idx != dstTriIdx) {
                    const EdgeTri& tri = etris[idx];
                    val_t dist = rayTriangleIntersectEdges(from, dirNorm, tri.v0, tri.e1, tri.e2);
                    if (dist > EPSILON && dist < maxDist) blocked = true;
                }
            }
            todo &= ~__ballot_sync(0xffffffffu, blocked);
            if (!todo) break;
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = n.child[c];
                if (ch < 0) continue;
                bool hit = mine && rayIntersectsBox(nodes[ch], seg);
                unsigned m = __ballot_sync(0xffffffffu, hit);
                if (m) {
                    if (lane == 0) { stackNode[sp] = ch; stackMask[sp] = m; }
                    ++sp;
                }
            }
            __syncwarp();
        }
    }
    return blocked;
}

// ============================================================================
// Form Factor Computation (Kij) - GPU kernels
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // std::max(ZERO, c)
}

// Pairs of triangles facing the same direction are culled (and draw no randoms)
HD inline bool pairActive(uint32_t i, uint32_t j, const Vec3& ni, const Vec3& nj) {
    return i != j && !(ni.dot(nj) > 0.99f);
}

constexpr int ROW_THREADS = 256;
constexpr int FF_THREADS = 256;

// Number of non-culled pairs in each row
__global__ void countPairsKernel(const Triangle* __restrict__ tris, uint32_t N, uint32_t row0,
                                 uint32_t* __restrict__ counts) {
    __shared__ uint32_t total;
    uint32_t i = row0 + blockIdx.x;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    Vec3 ni = tris[i].normal();
    uint32_t cnt = 0;
    for (uint32_t j = threadIdx.x; j < N; j += blockDim.x)
        cnt += pairActive(i, j, ni, tris[j].normal()) ? 1u : 0u;
    atomicAdd(&total, cnt);
    __syncthreads();
    if (threadIdx.x == 0) counts[blockIdx.x] = total;
}

// Ordered compaction of the non-culled pairs of the rows [lb, lb+gridDim.x)
__global__ void compactPairsKernel(const Triangle* __restrict__ tris, uint32_t N, uint32_t row0,
                                   uint32_t lb, const uint64_t* __restrict__ pairPrefix,
                                   uint32_t* __restrict__ pairI, uint32_t* __restrict__ pairJ) {
    __shared__ uint32_t warpSums[ROW_THREADS / 32];
    __shared__ uint32_t running;
    uint32_t li = lb + blockIdx.x;
    uint32_t i = row0 + li;
    uint64_t out = pairPrefix[li] - pairPrefix[lb];
    Vec3 ni = tris[i].normal();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    for (uint32_t j0 = 0; j0 < N; j0 += blockDim.x) {
        uint32_t j = j0 + threadIdx.x;
        bool f = j < N && pairActive(i, j, ni, tris[j].normal());
        unsigned ballot = __ballot_sync(0xffffffffu, f);
        uint32_t pre = __popc(ballot & ((1u << lane) - 1u));
        if (lane == 0) warpSums[warp] = __popc(ballot);
        __syncthreads();
        uint32_t wOff = 0, tot = 0;
        for (int w = 0; w < ROW_THREADS / 32; ++w) {
            if (w < warp) wOff += warpSums[w];
            tot += warpSums[w];
        }
        if (f) {
            uint64_t pos = out + running + wOff + pre;
            pairI[pos] = i;
            pairJ[pos] = j;
        }
        __syncthreads();
        if (threadIdx.x == 0) running += tot;
        __syncthreads();
    }
}

// Expand each row's mt19937 stream (starting window in rowStates) to uniform floats
__global__ void generateRandomsKernel(const uint32_t* __restrict__ rowStates, uint32_t lb,
                                      const uint64_t* __restrict__ pairPrefix,
                                      float* __restrict__ rnd) {
    __shared__ uint32_t mt[MT_N];
    uint32_t li = lb + blockIdx.x;
    uint64_t outBase = (pairPrefix[li] - pairPrefix[lb]) * DRAWS_PER_PAIR;
    uint64_t nOut = (pairPrefix[li + 1] - pairPrefix[li]) * DRAWS_PER_PAIR;
    const uint32_t* st = rowStates + static_cast<size_t>(li) * MT_N;
    for (int k = threadIdx.x; k < MT_N; k += blockDim.x) mt[k] = st[k];
    __syncthreads();
    for (uint64_t base = 0; base < nOut; base += MT_N) {
        for (int k = threadIdx.x; k < MT_N; k += blockDim.x)
            if (base + k < nOut) rnd[outBase + base + k] = mtToUniform(mtTemper(mt[k]));
        if (base + MT_N >= nOut) break;
        // In-place twist, in three dependency phases
        __syncthreads();
        uint32_t v = 0;
        int k = threadIdx.x;
        // phase 1: k in [0, 227)
        if (k < MT_N - MT_M) v = mtTwistWord(mt[k], mt[k + 1], mt[k + MT_M]);
        __syncthreads();
        if (k < MT_N - MT_M) mt[k] = v;
        __syncthreads();
        // phase 2: k in [227, 454)
        int k2 = k + (MT_N - MT_M);
        if (k < MT_N - MT_M) v = mtTwistWord(mt[k2], mt[k2 + 1], mt[k2 + MT_M - MT_N]);
        __syncthreads();
        if (k < MT_N - MT_M) mt[k2] = v;
        __syncthreads();
        // phase 3: k in [454, 624)
        int k3 = k + 2 * (MT_N - MT_M);
        if (k3 < MT_N - 1) v = mtTwistWord(mt[k3], mt[k3 + 1], mt[k3 + MT_M - MT_N]);
        else if (k3 == MT_N - 1) v = mtTwistWord(mt[MT_N - 1], mt[0], mt[MT_M - 1]);
        __syncthreads();
        if (k3 < MT_N) mt[k3] = v;
        __syncthreads();
    }
}

// One half-warp (16 lanes = NUM_RAYS) per pair; lane r traces ray r.
__global__ void __launch_bounds__(FF_THREADS, 6) formFactorKernel(const uint32_t* __restrict__ pairI, const uint32_t* __restrict__ pairJ,
                                 uint64_t nPairs, const float* __restrict__ rnd,
                                 const Triangle* __restrict__ tris,
                                 const EdgeTri* __restrict__ etris,
                                 const FlatNode* __restrict__ nodes,
                                 const uint32_t* __restrict__ leafTris,
                                 float* __restrict__ kijLocal, uint32_t row0, uint32_t N) {
    static_assert(NUM_RAYS == 16, "kernel assumes 16 rays per pair");
    __shared__ int stackNode[FF_THREADS / 32][OCT_STACK];
    __shared__ unsigned stackMask[FF_THREADS / 32][OCT_STACK];
    const int warp = threadIdx.x >> 5;
    uint64_t gtid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t p = gtid / NUM_RAYS;
    int r = static_cast<int>(gtid % NUM_RAYS);
    val_t term = ZERO;
    uint32_t i = 0, j = 0;
    bool need = false;
    Vec3 pI, pJ;
    val_t cosPhiI = ZERO, cosPhiJ = ZERO, distSqr = ZERO;
    if (p < nPairs) {
        i = pairI[p];
        j = pairJ[p];
        const float4 uv = reinterpret_cast<const float4*>(rnd)[p * NUM_RAYS + r];
        const Triangle triI = tris[i];
        const Triangle triJ = tris[j];
        pI = randomPointInTriangle(triI, uv.x, uv.y);
        pJ = randomPointInTriangle(triJ, uv.z, uv.w);

        // Cheap rejections first; the result does not depend on the order
        Vec3 v = pJ - pI;
        distSqr = v.squaredNorm();
        if (!(distSqr < EPSILON)) {
            cosPhiI = cosPhi(v, triI.normal());
            cosPhiJ = cosPhi(-v, triJ.normal());
            need = !(cosPhiI <= ZERO || cosPhiJ <= ZERO);
        }
    }
    bool blocked = isRayBlockedWarp(need, pI, pJ, nodes, leafTris, etris, i, j,
                                    stackNode[warp], stackMask[warp]);
    if (need && !blocked) term = (cosPhiI * cosPhiJ) / (PI * distSqr);
    // Sequential, ray-ordered accumulation (bitwise identical to the serial loop)
    val_t kij = ZERO;
    for (int k = 0; k < NUM_RAYS; ++k) kij += __shfl_sync(0xffffffffu, term, k, NUM_RAYS);
    if (r == 0 && p < nPairs)
        kijLocal[static_cast<size_t>(i - row0) * N + j] = kij * INV_NUM_RAYS;
}

__global__ void countNonZeroKernel(const float* __restrict__ kij, size_t n,
                                   unsigned long long* __restrict__ result) {
    unsigned long long c = 0;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < n;
         k += static_cast<size_t>(gridDim.x) * blockDim.x)
        c += kij[k] > EPSILON ? 1ull : 0ull;
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(result, c);
}

// ============================================================================
// Tau (time delay) and propagation weights - GPU
// ============================================================================

HD inline int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Row-major weights w_ij = min(kij * A_j, 1) (0 where kij <= 0) and delays
__global__ void buildPropagationKernel(const float* __restrict__ kijLocal, const float* __restrict__ areas,
                                       const Vec3* __restrict__ centers, uint32_t N, uint32_t row0,
                                       uint32_t nLocal, float* __restrict__ w, uint8_t* __restrict__ tau) {
    size_t total = static_cast<size_t>(N) * nLocal;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t li = static_cast<uint32_t>(k / N);
        uint32_t j = static_cast<uint32_t>(k % N);
        uint32_t i = row0 + li;
        float wij = ZERO;
        int tauij = 0;
        if (i != j) {
            tauij = computeTau(centers[i], centers[j]);
            float kij = kijLocal[k];
            if (kij > ZERO) {
                float a = kij * areas[j];
                wij = (ONE < a) ? ONE : a;  // std::min(kij * area, ONE)
            }
        }
        w[k] = wij;
        tau[k] = static_cast<uint8_t>(tauij > 255 ? 255 : tauij);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation) - GPU kernel, one warp per local row
// ============================================================================

constexpr int PROP_THREADS = 256;
constexpr int PROP_CHUNK = 128;  // 32 lanes x 4 consecutive sources

// The contributions of 128 consecutive sources are computed in parallel and then
// accumulated strictly in source order (bitwise identical to the serial loop;
// skipped sources contribute +0).
__global__ void propagateKernel(const float* __restrict__ w, const uint8_t* __restrict__ tau,
                                const float* __restrict__ radB, uint32_t N, uint32_t row0,
                                uint32_t nLocal, int t, float rho, uint32_t sourceIndex,
                                int timeOff, float* __restrict__ out) {
    __shared__ float4 terms[PROP_THREADS / 32][PROP_CHUNK / 4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t li = blockIdx.x * (PROP_THREADS / 32) + warp;
    if (li >= nLocal) return;
    const float* wRow = w + static_cast<size_t>(li) * N;
    const uint8_t* tRow = tau + static_cast<size_t>(li) * N;
    float4* myTerms = terms[warp];

    val_t sumB = ZERO;
    for (uint32_t base = 0; base < N; base += PROP_CHUNK) {
        uint32_t j0 = base + 4 * lane;
        float c[4] = {ZERO, ZERO, ZERO, ZERO};
        if (j0 < N) {  // N is a multiple of 4
            float4 w4 = *reinterpret_cast<const float4*>(wRow + j0);
            uchar4 t4 = *reinterpret_cast<const uchar4*>(tRow + j0);
            float wv[4] = {w4.x, w4.y, w4.z, w4.w};
            int tv[4] = {t4.x, t4.y, t4.z, t4.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (wv[q] > ZERO && t >= tv[q]) {
                    val_t radJ = radB[static_cast<size_t>(t - tv[q]) * N + j0 + q];
                    if (radJ > ZERO) c[q] = wv[q] * radJ;
                }
            }
        }
        myTerms[lane] = make_float4(c[0], c[1], c[2], c[3]);
        __syncwarp();
#pragma unroll 8
        for (int q = 0; q < PROP_CHUNK / 4; ++q) {
            float4 v = myTerms[q];
            sumB += v.x;
            sumB += v.y;
            sumB += v.z;
            sumB += v.w;
        }
        __syncwarp();
    }
    if (lane == 0) {
        uint32_t i = row0 + li;
        val_t radE = (i == sourceIndex && t < timeOff) ? 1.0f : ZERO;
        out[li] = rho * sumB + radE;
    }
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (rank 0)
    unsigned long long nonZeroKij = 0;  // Global count of form factors > EPSILON

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI decomposition (rows [row0, row0 + nLocal) are owned by this rank)
    int rank = 0, nRanks = 1;
    uint32_t row0 = 0, nLocal = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    Triangle* dTris = nullptr;
    EdgeTri* dEdgeTris = nullptr;
    FlatNode* dNodes = nullptr;
    uint32_t* dLeafTris = nullptr;
    float* dAreas = nullptr;
    Vec3* dCenters = nullptr;
    float* dKij = nullptr;          // local rows x N
    float* dW = nullptr;            // local rows x N propagation weights
    uint8_t* dTau = nullptr;        // local rows x N time delays
    float* dRadB = nullptr;         // T x N

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.rank == 0;
    // Generate mesh (deterministic, replicated on all ranks)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    const size_t N = state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) state.areas[i] = state.triangles[i].area();

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    state.radB.assign(timesteps * N, ZERO);

    // Row decomposition
    state.rowCounts.resize(state.nRanks);
    state.rowDispls.resize(state.nRanks);
    for (int r = 0; r < state.nRanks; ++r) {
        size_t b = N * static_cast<size_t>(r) / state.nRanks;
        size_t e = N * static_cast<size_t>(r + 1) / state.nRanks;
        state.rowDispls[r] = static_cast<int>(b);
        state.rowCounts[r] = static_cast<int>(e - b);
    }
    state.row0 = static_cast<uint32_t>(state.rowDispls[state.rank]);
    state.nLocal = static_cast<uint32_t>(state.rowCounts[state.rank]);

    // Upload geometry
    FlatOctree flat;
    flat.add(state.octree, 0);
    if (7 * flat.maxDepth + 1 > OCT_STACK) {
        fprintf(stderr, "Octree too deep for traversal stack (%d)\n", flat.maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Vec3> centers(N);
    for (size_t i = 0; i < N; ++i) centers[i] = state.triangles[i].center();

    CUDA_CHECK(cudaMalloc(&state.dTris, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    std::vector<EdgeTri> edgeTris(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        edgeTris[i] = {t.a, t.b - t.a, t.c - t.a};
    }
    CUDA_CHECK(cudaMalloc(&state.dEdgeTris, N * sizeof(EdgeTri)));
    CUDA_CHECK(cudaMemcpy(state.dEdgeTris, edgeTris.data(), N * sizeof(EdgeTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dNodes, flat.nodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMemcpy(state.dNodes, flat.nodes.data(), flat.nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dLeafTris, std::max<size_t>(1, flat.leafTris.size()) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(state.dLeafTris, flat.leafTris.data(), flat.leafTris.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dAreas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dCenters, N * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(state.dCenters, centers.data(), N * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const bool root = state.rank == 0;
    if (root) printf("Computing form factors (Kij)...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t nLocal = state.nLocal, row0 = state.row0;

    // 1. Non-culled pair counts per row (GPU), shared among ranks
    std::vector<uint32_t> counts(N, 0);
    {
        uint32_t* dCounts = nullptr;
        CUDA_CHECK(cudaMalloc(&dCounts, std::max<uint32_t>(1, nLocal) * sizeof(uint32_t)));
        if (nLocal > 0) {
            countPairsKernel<<<nLocal, ROW_THREADS>>>(state.dTris, N, row0, dCounts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(counts.data() + row0, dCounts, nLocal * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(dCounts));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, counts.data(), state.rowCounts.data(),
                       state.rowDispls.data(), MPI_UINT32_T, MPI_COMM_WORLD);
    }

    // Global stream offset (in pairs) of the first pair of each row
    std::vector<uint64_t> globalPrefix(N + 1, 0);
    for (uint32_t i = 0; i < N; ++i) globalPrefix[i + 1] = globalPrefix[i] + counts[i];
    std::vector<uint64_t> localPrefix(nLocal + 1);
    for (uint32_t li = 0; li <= nLocal; ++li) localPrefix[li] = globalPrefix[row0 + li] - globalPrefix[row0];

    CUDA_CHECK(cudaMalloc(&state.dKij, std::max<size_t>(1, static_cast<size_t>(nLocal) * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dKij, 0, std::max<size_t>(1, static_cast<size_t>(nLocal) * N) * sizeof(float)));

    if (nLocal > 0) {
        // 2. Batches of rows for the GPU (bounded random-number buffer)
        const uint64_t maxBatchPairs = uint64_t(1) << 20;
        uint64_t maxRow = 0;
        for (uint32_t li = 0; li < nLocal; ++li) maxRow = std::max<uint64_t>(maxRow, counts[row0 + li]);
        const uint64_t bufPairs = std::max<uint64_t>(
            std::max<uint64_t>(std::min<uint64_t>(maxBatchPairs, localPrefix[nLocal]), maxRow), 1);
        std::vector<uint32_t> batchBegin;
        std::vector<uint32_t> batchOfRow(nLocal);
        for (uint32_t lb = 0; lb < nLocal;) {
            uint32_t le = lb + 1;
            while (le < nLocal && localPrefix[le + 1] - localPrefix[lb] <= bufPairs) ++le;
            for (uint32_t li = lb; li < le; ++li) batchOfRow[li] = static_cast<uint32_t>(batchBegin.size());
            batchBegin.push_back(lb);
            lb = le;
        }
        const size_t nBatches = batchBegin.size();
        batchBegin.push_back(nLocal);
        std::unique_ptr<std::atomic<uint32_t>[]> batchPending(new std::atomic<uint32_t>[nBatches]);
        for (size_t b = 0; b < nBatches; ++b) batchPending[b].store(batchBegin[b + 1] - batchBegin[b]);

        // 3. Chunks of rows whose generator states are reconstructed by one thread:
        //    one seek (discard or jump-ahead), then sequential discards row by row.
        const uint64_t chunkDraws = MT_JUMP_THRESHOLD;
        std::vector<uint32_t> chunkBegin;
        for (uint32_t li = 0; li < nLocal; ++li) {
            if (chunkBegin.empty() ||
                (localPrefix[li] - localPrefix[chunkBegin.back()]) * DRAWS_PER_PAIR >= chunkDraws)
                chunkBegin.push_back(li);
        }
        const size_t nChunks = chunkBegin.size();
        chunkBegin.push_back(nLocal);
        std::atomic<size_t> nextChunk{0};

        uint32_t* rowStates = nullptr;  // pinned, for asynchronous uploads
        CUDA_CHECK(cudaMallocHost(&rowStates, static_cast<size_t>(nLocal) * MT_N * sizeof(uint32_t)));
        std::unique_ptr<MTJumper> jumper;
        std::once_flag jumperOnce;

        auto processChunk = [&](size_t c) {
            MTState st;
            for (uint32_t li = chunkBegin[c]; li < chunkBegin[c + 1]; ++li) {
                uint64_t target = globalPrefix[row0 + li] * DRAWS_PER_PAIR;
                if (li == chunkBegin[c]) {
                    if (target >= MT_JUMP_THRESHOLD)
                        std::call_once(jumperOnce, [&] { jumper = std::make_unique<MTJumper>(); });
                    mtStateAt(target, st, jumper.get());
                } else {
                    st.discard(static_cast<uint64_t>(counts[row0 + li - 1]) * DRAWS_PER_PAIR);
                }
                st.exportWindow(&rowStates[static_cast<size_t>(li) * MT_N]);
                batchPending[batchOfRow[li]].fetch_sub(1, std::memory_order_release);
            }
        };
        auto tryProcessChunk = [&]() {
            size_t c = nextChunk.fetch_add(1);
            if (c >= nChunks) return false;
            processChunk(c);
            return true;
        };

        // 4. GPU buffers
        uint32_t *dStates = nullptr, *dPairI = nullptr, *dPairJ = nullptr;
        uint64_t* dPrefix = nullptr;
        float* dRnd = nullptr;
        CUDA_CHECK(cudaMalloc(&dStates, static_cast<size_t>(nLocal) * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dPrefix, localPrefix.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(dPrefix, localPrefix.data(), localPrefix.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&dPairI, bufPairs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dPairJ, bufPairs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dRnd, bufPairs * DRAWS_PER_PAIR * sizeof(float)));

        // 5. Pipeline: the master thread launches GPU batches as soon as their
        //    row states are ready (helping with chunks while waiting); all
        //    other OpenMP threads reconstruct generator states.
#pragma omp parallel
        {
            if (omp_get_thread_num() == 0) {
                for (size_t b = 0; b < nBatches; ++b) {
                    while (batchPending[b].load(std::memory_order_acquire) != 0) {
                        if (!tryProcessChunk()) std::this_thread::yield();
                    }
                    uint32_t lb = batchBegin[b], le = batchBegin[b + 1];
                    uint64_t nPairs = localPrefix[le] - localPrefix[lb];
                    CUDA_CHECK(cudaMemcpyAsync(dStates + static_cast<size_t>(lb) * MT_N,
                                               rowStates + static_cast<size_t>(lb) * MT_N,
                                               static_cast<size_t>(le - lb) * MT_N * sizeof(uint32_t),
                                               cudaMemcpyHostToDevice));
                    if (nPairs == 0) continue;
                    compactPairsKernel<<<le - lb, ROW_THREADS>>>(state.dTris, N, row0, lb, dPrefix, dPairI, dPairJ);
                    generateRandomsKernel<<<le - lb, ROW_THREADS>>>(dStates, lb, dPrefix, dRnd);
                    uint64_t blocks = (nPairs * NUM_RAYS + FF_THREADS - 1) / FF_THREADS;
                    formFactorKernel<<<static_cast<unsigned>(blocks), FF_THREADS>>>(
                        dPairI, dPairJ, nPairs, dRnd, state.dTris, state.dEdgeTris, state.dNodes, state.dLeafTris,
                        state.dKij, row0, N);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                while (tryProcessChunk()) {}
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFreeHost(rowStates));
        CUDA_CHECK(cudaFree(dStates));
        CUDA_CHECK(cudaFree(dPrefix));
        CUDA_CHECK(cudaFree(dPairI));
        CUDA_CHECK(cudaFree(dPairJ));
        CUDA_CHECK(cudaFree(dRnd));
    }

    // Count of non-zero form factors (for validation)
    unsigned long long localNonZero = 0;
    if (nLocal > 0) {
        unsigned long long* dCount = nullptr;
        CUDA_CHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
        countNonZeroKernel<<<1024, 256>>>(state.dKij, static_cast<size_t>(nLocal) * N, dCount);
        CUDA_CHECK(cudaMemcpy(&localNonZero, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dCount));
    }
    MPI_Allreduce(&localNonZero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (root) printf("  Progress: %u/%u triangles\n", N, N);
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    // Delays are combined with the form factors into the propagation operator
    // once the form factors are available (see buildPropagation).
}

void buildPropagation(SimulationState& state) {
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    size_t total = std::max<size_t>(1, static_cast<size_t>(state.nLocal) * N);
    CUDA_CHECK(cudaMalloc(&state.dW, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTau, total * sizeof(uint8_t)));
    if (state.nLocal > 0) {
        buildPropagationKernel<<<4096, 256>>>(state.dKij, state.dAreas, state.dCenters, N, state.row0,
                                              state.nLocal, state.dW, state.dTau);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(state.dKij));
    state.dKij = nullptr;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = state.rank == 0;
    if (root) printf("Running wave propagation simulation...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const size_t T = state.numTimesteps;
    const int timeOff = static_cast<int>(T / 2);
    const float rho = state.rho.empty() ? ZERO : state.rho[0];  // uniform reflectivity

    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, std::max<uint32_t>(1, state.nLocal) * sizeof(float)));

    for (size_t t = 0; t < T; ++t) {
        float* row = &state.radB[state.idxTN(t, 0)];
        if (state.nLocal > 0) {
            const uint32_t rowsPerBlock = PROP_THREADS / 32;
            propagateKernel<<<(state.nLocal + rowsPerBlock - 1) / rowsPerBlock, PROP_THREADS>>>(
                state.dW, state.dTau, state.dRadB, N, state.row0, state.nLocal, static_cast<int>(t), rho,
                static_cast<uint32_t>(state.sourceIndex), timeOff, dOut);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(row + state.row0, dOut, state.nLocal * sizeof(float), cudaMemcpyDeviceToHost));
        }
        if (state.nRanks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, row, state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        }
        CUDA_CHECK(cudaMemcpy(state.dRadB + state.idxTN(t, 0), row, N * sizeof(float), cudaMemcpyHostToDevice));

        if (root && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaFree(dOut));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    const size_t T = state.numTimesteps;
    std::vector<val_t> local(state.nLocal);
    // Size the thread team to the (small) amount of work
    const size_t work = static_cast<size_t>(state.nLocal) * T * T;
    const int nThreads = static_cast<int>(std::clamp<size_t>(work >> 20, 1, omp_get_max_threads()));

#pragma omp parallel for schedule(static) num_threads(nThreads)
    for (size_t li = 0; li < state.nLocal; ++li) {
        size_t i = state.row0 + li;
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

        local[li] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    if (state.rank == 0) state.distances.assign(state.numTriangles, ZERO);
    MPI_Gatherv(local.data(), static_cast<int>(state.nLocal), MPI_FLOAT, state.distances.data(),
                state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
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

    // One GPU per node-local rank
    int localRank = 0;
    {
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
    }
    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % nDevices));
    CUDA_CHECK(cudaFree(nullptr));  // create the context outside the timed regions

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelism: %d MPI ranks, %d OpenMP threads/rank, %d GPUs/node\n",
               nRanks, omp_get_max_threads(), nDevices);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nRanks = nRanks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);
    buildPropagation(state);
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

    // Release device memory
    CUDA_CHECK(cudaFree(state.dTris));
    CUDA_CHECK(cudaFree(state.dEdgeTris));
    CUDA_CHECK(cudaFree(state.dNodes));
    CUDA_CHECK(cudaFree(state.dLeafTris));
    CUDA_CHECK(cudaFree(state.dAreas));
    CUDA_CHECK(cudaFree(state.dCenters));
    CUDA_CHECK(cudaFree(state.dW));
    CUDA_CHECK(cudaFree(state.dTau));
    CUDA_CHECK(cudaFree(state.dRadB));

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

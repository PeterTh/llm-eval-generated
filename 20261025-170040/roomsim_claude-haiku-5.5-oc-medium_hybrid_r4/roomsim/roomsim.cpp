/**
 * Room Response Simulation Benchmark (hybrid MPI + OpenMP + CUDA)
 *
 * Models how sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *  - MPI:    the N x N form-factor matrix is partitioned by contiguous receiver rows
 *            (one block per rank). Each simulation timestep exchanges the radiosity
 *            row via MPI_Allgatherv so every rank holds the full radiosity history.
 *  - CUDA:   one GPU per rank (local rank modulo device count). Form factors (one thread
 *            per (receiver, emitter) pair, octree-accelerated visibility), time delays,
 *            and the per-timestep radiosity update all run on the device.
 *  - OpenMP: each rank runs several host threads. Each thread owns a contiguous slice of
 *            the rank's rows and drives its own CUDA stream pipeline, generating that
 *            slice's random draws concurrently. The O(N^2) active-pair census and the
 *            O(N * T^2) distance estimation are also OpenMP loops.
 *
 * Semantics are preserved exactly: the random-number stream is identical to the
 * sequential original. The single std::mt19937 stream is consumed in row-major order,
 * and each thread jumps directly to its starting draw (exact GF(2) jump-ahead, see
 * mt_jump). Floating-point evaluation order matches the original, so results are
 * bit-identical for any number of ranks and threads.
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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
#include <optional>
#include <random>
#include <sstream>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr val_t NO_HIT = 3.40282346638528859812e+38f;  // == std::numeric_limits<float>::max()

constexpr int BLOCK_SIZE = 256;
constexpr uint32_t INVALID_GROUP = 0xFFFFFFFFu;
constexpr int MAX_FF_THREADS = 16;      // host threads (RNG pipelines) per rank
constexpr size_t BATCH_PAIRS = 1u << 15;  // (receiver, emitter) pairs per form-factor launch (per thread)
constexpr size_t MIN_PAIRS_PER_THREAD = 1u << 21;  // minimum work before adding a host thread
constexpr uint64_t JUMP_MIN_DRAWS = 1ull << 22;    // skips longer than this use the MT jump table

// ============================================================================
// Math helpers shared by host and device (identical expression trees)
// ============================================================================

HD inline val_t absv(val_t x) {
#ifdef __CUDA_ARCH__
    return fabsf(x);
#else
    return std::fabs(x);
#endif
}

HD inline val_t sqrtv(val_t x) {
#ifdef __CUDA_ARCH__
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

// Same semantics as std::max / std::min for floats
HD inline val_t maxv(val_t a, val_t b) { return (a < b) ? b : a; }
HD inline val_t minv(val_t a, val_t b) { return (b < a) ? b : a; }

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
    HD val_t norm() const { return sqrtv(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return absv(x - o.x) < EPSILON && absv(y - o.y) < EPSILON && absv(z - o.z) < EPSILON;
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
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host only)
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
// Octree for Spatial Acceleration (built on host, flattened for the device)
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
};

// Flattened octree node for device traversal (stackless: parent link and slot in parent)
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];      // node index, or -1
    int32_t parent;        // -1 for the root
    int32_t slot;          // index of this node in parent->child[]
    uint32_t triBegin;     // leaf: range in FlatOctree::triIdx
    uint32_t triCount;     // leaf: number of triangles (0 for internal nodes)
};

struct FlatOctree {
    std::vector<DevNode> nodes;   // node 0 is the root
    std::vector<uint32_t> triIdx;
    int maxDepth = 0;
};

static int32_t flattenOctree(const Octree& n, FlatOctree& f, int depth, int32_t parent, int32_t slot) {
    const int32_t id = static_cast<int32_t>(f.nodes.size());
    f.nodes.push_back(DevNode{});
    f.maxDepth = std::max(f.maxDepth, depth);

    DevNode d{};
    d.center = n.center;
    d.halfExtent = n.halfExtent;
    for (int c = 0; c < 8; ++c) d.child[c] = -1;
    d.parent = parent;
    d.slot = slot;
    d.triBegin = 0;
    d.triCount = 0;

    if (!n.triangleIndices.empty()) {
        d.triBegin = static_cast<uint32_t>(f.triIdx.size());
        d.triCount = static_cast<uint32_t>(n.triangleIndices.size());
        for (size_t t : n.triangleIndices) f.triIdx.push_back(static_cast<uint32_t>(t));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (n.children[c]) d.child[c] = flattenOctree(*n.children[c], f, depth + 1, id, c);
        }
    }
    f.nodes[id] = d;
    return id;
}

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

// Exact Mersenne Twister (std::mt19937, seed 42) jump-ahead over GF(2).
//
// The MT19937 output bit planes satisfy a linear recurrence whose characteristic
// polynomial (degree 19937) is found by Berlekamp-Massey. The state at any draw index D
// is then obtained as sum_k (x^D mod P)_k * window_k, so a thread can start its part
// of the single global stream without generating the draws before it.
[[noreturn]] static void fatal(const char* msg);

namespace mt_jump {

constexpr uint32_t TEMPER_B = 0x9d2c5680u;
constexpr uint32_t TEMPER_C = 0xefc60000u;

inline uint32_t temper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & TEMPER_B;
    y ^= (y << 15) & TEMPER_C;
    y ^= y >> 18;
    return y;
}

inline uint32_t untemper(uint32_t y) {
    y ^= y >> 18;
    y ^= (y << 15) & TEMPER_C;
    uint32_t x = y;
    for (int i = 0; i < 5; ++i) x = y ^ ((x << 7) & TEMPER_B);
    y = x;
    x = y;
    for (int i = 0; i < 3; ++i) x = y ^ (x >> 11);
    return x;
}

// Berlekamp-Massey over GF(2). Returns linear complexity L; C[0..L] is the connection polynomial.
inline int berlekampMassey(const std::vector<uint8_t>& s, std::vector<uint8_t>& C) {
    const size_t n = s.size();
    std::vector<uint8_t> cur(n + 1, 0), prev(n + 1, 0), tmp;
    cur[0] = 1;
    prev[0] = 1;
    int L = 0;
    long m = -1;
    for (size_t i = 0; i < n; ++i) {
        uint8_t d = s[i];
        for (int j = 1; j <= L; ++j) d ^= cur[j] & s[i - j];
        if (d) {
            tmp = cur;
            const size_t shift = static_cast<size_t>(static_cast<long>(i) - m);
            const size_t deg = std::min(n - shift, i + 1);
            for (size_t j = 0; j <= deg; ++j) cur[j + shift] ^= prev[j];
            if (2 * L <= static_cast<long>(i)) {
                L = static_cast<int>(i) + 1 - L;
                m = static_cast<long>(i);
                prev = tmp;
            }
        }
    }
    cur.resize(static_cast<size_t>(L) + 1);
    C = cur;
    return L;
}

struct JumpTable {
    int L = 0;                   // linear complexity of the MT19937 output bit planes
    int W = 0;                   // 64-bit words per GF(2) polynomial of degree <= L
    std::vector<uint64_t> R;     // x^L == sum_{p in R} x^p (mod characteristic polynomial)
    std::vector<uint32_t> rows;  // rows[k*624 + j] = tempered output y_{k+j}, k < L, j < 624
};

// Build the jump table from the tempered output stream of std::mt19937(seed)
inline JumpTable buildJumpTable(uint32_t seed) {
    const size_t terms = 2 * 19937 + 1024;  // >= 2L + window length
    std::mt19937 g(seed);
    std::vector<uint32_t> y(terms);
    for (auto& v : y) v = g();

    std::vector<uint8_t> s(terms);
    for (size_t i = 0; i < terms; ++i) s[i] = static_cast<uint8_t>(y[i] & 1u);

    std::vector<uint8_t> C;
    JumpTable jt;
    jt.L = berlekampMassey(s, C);
    const int L = jt.L;
    if (L <= 0 || 2 * static_cast<size_t>(L) + 624 > terms) fatal("MT19937 jump table: unexpected linear complexity");
    jt.W = L / 64 + 1;

    jt.R.assign(static_cast<size_t>(jt.W), 0);
    for (int i = 1; i <= L; ++i) {
        if (C[i]) {
            const int p = L - i;
            jt.R[static_cast<size_t>(p >> 6)] |= 1ull << (p & 63);
        }
    }

    jt.rows.resize(static_cast<size_t>(L) * 624);
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < 624; ++j)
            jt.rows[static_cast<size_t>(k) * 624 + j] = y[static_cast<size_t>(k) + j];
    return jt;
}

// dst ^= src << sh (truncated to dst size)
inline void xorShifted(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, size_t sh) {
    const size_t dw = dst.size();
    for (size_t w = 0; w < src.size(); ++w) {
        const uint64_t v = src[w];
        if (!v) continue;
        const size_t pos = w * 64 + sh;
        const size_t q = pos >> 6;
        const unsigned r = static_cast<unsigned>(pos & 63);
        if (q < dw) dst[q] ^= v << r;
        if (r && q + 1 < dw) dst[q + 1] ^= v >> (64 - r);
    }
}

// a = a * x mod P
inline void mulxMod(const JumpTable& jt, std::vector<uint64_t>& a) {
    uint64_t carry = 0;
    for (int w = 0; w < jt.W; ++w) {
        const uint64_t nc = a[w] >> 63;
        a[w] = (a[w] << 1) | carry;
        carry = nc;
    }
    const int lw = jt.L >> 6;
    const unsigned lb = static_cast<unsigned>(jt.L & 63);
    if ((a[lw] >> lb) & 1ull) {
        a[lw] &= ~(1ull << lb);
        for (int w = 0; w < jt.W; ++w) a[w] ^= jt.R[w];
    }
}

// a = a^2 mod P
inline void squareMod(const JumpTable& jt, std::vector<uint64_t>& a, std::vector<uint64_t>& scratch) {
    const int L = jt.L;
    scratch.assign(static_cast<size_t>(2 * jt.W + 2), 0);
    for (int k = 0; k < L; ++k) {
        if ((a[k >> 6] >> (k & 63)) & 1ull) {
            const int p = 2 * k;
            scratch[p >> 6] |= 1ull << (p & 63);
        }
    }
    for (int t = 2 * L - 2; t >= L; --t) {
        if ((scratch[t >> 6] >> (t & 63)) & 1ull) {
            scratch[t >> 6] &= ~(1ull << (t & 63));
            xorShifted(scratch, jt.R, static_cast<size_t>(t - L));
        }
    }
    for (int w = 0; w < jt.W; ++w) a[w] = scratch[w];
    const int lw = L >> 6;
    const unsigned lb = static_cast<unsigned>(L & 63);
    a[lw] &= (lb == 0) ? 0ull : ((1ull << lb) - 1ull);
    for (int w = lw + 1; w < jt.W; ++w) a[w] = 0;
}

// x^D mod P
inline std::vector<uint64_t> xPowMod(const JumpTable& jt, uint64_t D) {
    std::vector<uint64_t> res(static_cast<size_t>(jt.W), 0), scratch;
    res[0] = 1;
    if (D == 0) return res;
    int hb = 63;
    while (!((D >> hb) & 1ull)) --hb;
    for (int b = hb; b >= 0; --b) {
        if (b < hb) squareMod(jt, res, scratch);
        if ((D >> b) & 1ull) mulxMod(jt, res);
    }
    return res;
}

// Untempered MT internal state for draws D..D+623 (the window preceding output D+624)
inline void mtWindowAt(const JumpTable& jt, uint64_t D, uint32_t out[624]) {
    const std::vector<uint64_t> d = xPowMod(jt, D);
    std::vector<uint32_t> acc(624, 0);
    for (int k = 0; k < jt.L; ++k) {
        if ((d[static_cast<size_t>(k >> 6)] >> (k & 63)) & 1ull) {
            const uint32_t* row = &jt.rows[static_cast<size_t>(k) * 624];
            for (int j = 0; j < 624; ++j) acc[j] ^= row[j];
        }
    }
    for (int j = 0; j < 624; ++j) out[j] = untemper(acc[j]);
}

// std::mt19937(seed) positioned so that its next output is draw index D
inline std::mt19937 engineAt(const JumpTable* jt, uint64_t D, uint32_t seed) {
    std::mt19937 e(seed);
    if (D < 624 || jt == nullptr) {
        e.discard(D);
        return e;
    }
    uint32_t st[624];
    mtWindowAt(*jt, D - 624, st);
    std::ostringstream os;
    for (int j = 0; j < 624; ++j) os << st[j] << ' ';
    os << 624;
    std::istringstream is(os.str());
    is >> e;
    return e;
}

}  // namespace mt_jump

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    // Stream of the global sequence starting at draw index `startDraw`
    RandomGenerator(const mt_jump::JumpTable* jt, uint64_t startDraw)
        : rng(mt_jump::engineAt(jt, startDraw, 42)), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Random point in triangle from two uniform draws (barycentric, same as original)
HD inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
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

HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (absv(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

// Cosine of angle between vector and normal, clamped at zero
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return maxv(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Device-side visibility (octree traversal)
// ============================================================================

struct SceneView {
    const Triangle* tris;
    const DevNode* nodes;
    const uint32_t* triIdx;
};

// Ray vs node bounding box (same test as the host octree)
__device__ inline bool boxHit(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {absv(d.x), absv(d.y), absv(d.z)};

    if (absv(c.x) > n.halfExtent.x + ad.x) return false;
    if (absv(c.y) > n.halfExtent.y + ad.y) return false;
    if (absv(c.z) > n.halfExtent.z + ad.z) return false;

    if (absv(d.y * c.z - d.z * c.y) > n.halfExtent.y * ad.z + n.halfExtent.z * ad.y + EPSILON) return false;
    if (absv(d.z * c.x - d.x * c.z) > n.halfExtent.z * ad.x + n.halfExtent.x * ad.z + EPSILON) return false;
    if (absv(d.x * c.y - d.y * c.x) > n.halfExtent.x * ad.y + n.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ inline bool leafBlocked(const SceneView& s, const DevNode& n,
                                   const Vec3& from, const Vec3& dirNorm, val_t rayLen,
                                   uint32_t srcIdx, uint32_t dstIdx) {
    for (uint32_t k = 0; k < n.triCount; ++k) {
        const uint32_t idx = s.triIdx[n.triBegin + k];
        if (idx == srcIdx || idx == dstIdx) continue;
        const Triangle& tri = s.tris[idx];
        const val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) return true;
    }
    return false;
}

// Is the segment from->to blocked by any triangle other than src/dst?
__device__ inline bool isRayBlockedDev(const SceneView& s, const Vec3& from, const Vec3& to,
                                       uint32_t srcIdx, uint32_t dstIdx) {
    const Vec3 dir = to - from;
    const val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    const Vec3 dirNorm = dir / rayLen;

    const DevNode& root = s.nodes[0];
    if (root.triCount > 0) return leafBlocked(s, root, from, dirNorm, rayLen, srcIdx, dstIdx);

    // Depth-first walk over the children of internal nodes; the root itself is not box-tested
    // (as in the host octree). Climbing back uses the parent link, so no stack is needed.
    int32_t cur = 0;
    int c = 0;
    while (true) {
        if (c >= 8) {
            if (cur == 0) break;
            const int32_t up = s.nodes[cur].slot;
            cur = s.nodes[cur].parent;
            c = up + 1;
            continue;
        }
        const int32_t ci = s.nodes[cur].child[c];
        ++c;
        if (ci < 0) continue;
        const DevNode& ch = s.nodes[ci];
        if (!boxHit(ch, from, to)) continue;
        if (ch.triCount > 0) {
            if (leafBlocked(s, ch, from, dirNorm, rayLen, srcIdx, dstIdx)) return true;
        } else {
            cur = ci;
            c = 0;
        }
    }
    return false;
}

// ============================================================================
// Form Factor Kernel (Kij)
// ============================================================================

struct FormFactorArgs {
    SceneView scene;
    const float* rng;        // NUM_RAYS*4 raw uniform draws per active pair group
    const uint32_t* slot;    // [j * batchRows + iB] -> group id or INVALID_GROUP
    float* kijT;             // [j * L + iL] form factors, receiver-local rows
    uint32_t* nonZero;       // count of kij > EPSILON
    int N;
    int L;                   // local rows on this rank
    int rowBase;             // global index of local row 0
    int batchStart;          // first local row of this batch
    int batchRows;           // rows in this batch
};

__device__ inline val_t computeKijDevice(const FormFactorArgs& a, uint32_t i, uint32_t j, uint32_t g) {
    const Triangle& triI = a.scene.tris[i];
    const Triangle& triJ = a.scene.tris[j];
    const float* r = a.rng + static_cast<size_t>(g) * (NUM_RAYS * 4);

    val_t kij = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const float* q = r + ray * 4;
        const Vec3 pI = pointInTriangle(triI, q[0], q[1]);
        const Vec3 pJ = pointInTriangle(triJ, q[2], q[3]);

        if (isRayBlockedDev(a.scene, pI, pJ, i, j)) continue;

        const Vec3 v = pJ - pI;
        const val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        const val_t cosPhiI = cosPhi(v, triI.normal());
        const val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    return kij * INV_NUM_RAYS;
}

__global__ void __launch_bounds__(BLOCK_SIZE, 4) formFactorKernel(FormFactorArgs a) {
    const size_t total = static_cast<size_t>(a.batchRows) * a.N;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = tid < total;

    val_t value = ZERO;
    if (valid) {
        const int iB = static_cast<int>(tid % a.batchRows);
        const uint32_t j = static_cast<uint32_t>(tid / a.batchRows);
        const int iL = a.batchStart + iB;
        const uint32_t i = static_cast<uint32_t>(a.rowBase + iL);
        const uint32_t g = a.slot[static_cast<size_t>(j) * a.batchRows + iB];
        if (g != INVALID_GROUP) value = computeKijDevice(a, i, j, g);
        a.kijT[static_cast<size_t>(j) * a.L + iL] = value;
    }

    // Warp-aggregated count of non-zero form factors
    const bool nz = valid && value > EPSILON;
    const unsigned mask = __ballot_sync(0xffffffffu, nz);
    if ((threadIdx.x & 31) == 0 && mask != 0) atomicAdd(a.nonZero, static_cast<unsigned>(__popc(mask)));
}

// ============================================================================
// Time Delay Kernel (Tau)
// ============================================================================

__global__ void tauKernel(const Triangle* tris, int* tauT, int N, int L, int rowBase) {
    const size_t total = static_cast<size_t>(L) * N;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int iL = static_cast<int>(tid % L);
    const uint32_t j = static_cast<uint32_t>(tid / L);
    const uint32_t i = static_cast<uint32_t>(rowBase + iL);

    int tau = 0;
    if (i != j) {
        const val_t dist = (tris[i].center() - tris[j].center()).norm();
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[tid] = tau;   // layout [j * L + iL]
}

// ============================================================================
// Radiosity Update Kernel (one timestep)
// ============================================================================

struct SimArgs {
    const float* kijT;     // [j * L + iL]
    const int* tauT;       // [j * L + iL]
    const float* areas;
    float* radB;           // [t * N + n], full replica on every rank
    int N;
    int L;
    int rowBase;
    int t;
    int timeOff;           // source emits for t < timeOff
    int source;
    float rho;
};

__global__ void simStepKernel(SimArgs a) {
    const int iL = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (iL >= a.L) return;
    const int i = a.rowBase + iL;

    val_t sumB = ZERO;
    for (int j = 0; j < a.N; ++j) {
        if (j == i) continue;

        const size_t p = static_cast<size_t>(j) * a.L + iL;
        const int tauij = a.tauT[p];

        // Skip if wave hasn't yet propagated from j to i
        if (a.t < tauij) continue;

        const val_t kij = a.kijT[p];
        if (kij <= ZERO) continue;

        // Radiosity from source triangle at the time emission occurred
        const size_t srcTime = static_cast<size_t>(a.t - tauij);
        const val_t radJ = a.radB[srcTime * a.N + j];
        if (radJ <= ZERO) continue;

        sumB += minv(kij * a.areas[j], ONE) * radJ;
    }

    const val_t radE = (a.t < a.timeOff && i == a.source) ? ONE : ZERO;
    a.radB[static_cast<size_t>(a.t) * a.N + i] = a.rho * sumB + radE;
}

// ============================================================================
// Error handling / MPI helpers
// ============================================================================

static int g_rank = 0;
static int g_size = 1;

#define RPRINTF(...) do { if (g_rank == 0) printf(__VA_ARGS__); } while (0)

[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "[rank %d] FATAL: %s\n", g_rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "[rank %d] CUDA error %s at %s:%d: %s\n", g_rank, #call,  \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

static size_t blocksFor(size_t n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

// Contiguous block of receiver rows owned by this rank
struct RowPartition {
    size_t begin;
    size_t end;
    size_t count() const { return end - begin; }
};

static RowPartition partitionRows(size_t N, int rank, int size) {
    return {N * static_cast<size_t>(rank) / size, N * static_cast<size_t>(rank + 1) / size};
}

// ============================================================================
// Room Response Simulation State (host, replicated on every rank)
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    val_t reflectivity = ZERO;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Radiosity history (T x N, row-major), full replica
    std::vector<val_t> distances;   // Computed distances from source
    std::vector<uint64_t> groupPrefix;  // groupPrefix[i] = #active (i,j) pairs for rows < i

    Octree octree;                  // Spatial acceleration structure (host build)
    FlatOctree flat;                // Flattened copy for the device

    size_t sourceIndex = 0;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Culling predicate used by the form-factor computation (identical to original)
static inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

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
    state.reflectivity = reflectivity;
    state.sourceIndex = sourceIdx % state.numTriangles;

    const size_t N = state.numTriangles;
    RPRINTF("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    RPRINTF("Building octree...\n");
    state.octree.build(state.triangles);
    state.flat = FlatOctree{};
    flattenOctree(state.octree, state.flat, 0, -1, -1);

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Radiosity history and distances
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Active (i,j) pair census (OpenMP). Drives the RNG stream layout: pair (i,j)
    // with i != j and not culled consumes NUM_RAYS*4 draws, in row-major order.
    std::vector<uint64_t> activeRow(N, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long ii = 0; ii < static_cast<long long>(N); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        uint64_t cnt = 0;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            if (!isCulled(state.triangles[i], state.triangles[j])) ++cnt;
        }
        activeRow[i] = cnt;
    }
    state.groupPrefix.assign(N + 1, 0);
    for (size_t i = 0; i < N; ++i) state.groupPrefix[i + 1] = state.groupPrefix[i] + activeRow[i];
}

// ============================================================================
// Device state
// ============================================================================

struct DeviceData {
    Triangle* tris = nullptr;
    DevNode* nodes = nullptr;
    uint32_t* triIdx = nullptr;
    float* areas = nullptr;
    float* kijT = nullptr;     // [j * L + iL]
    int* tauT = nullptr;       // [j * L + iL]
    float* radB = nullptr;     // [t * N + n]
    uint32_t* nonZero = nullptr;

    cudaStream_t stream = nullptr;            // simulation / tau stream
    std::vector<int> recvCounts, recvDispls;  // row partition for MPI_Allgatherv
};

static int selectDevice(int localRank) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) fatal("no CUDA devices visible to this rank");
    const int device = localRank % count;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

static void uploadScene(const SimulationState& st, const RowPartition& part, DeviceData& dev) {
    const size_t N = st.numTriangles;
    const size_t T = st.numTimesteps;
    const size_t L = part.count();

    CUDA_CHECK(cudaMalloc(&dev.tris, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(dev.tris, st.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev.nodes, st.flat.nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMemcpy(dev.nodes, st.flat.nodes.data(), st.flat.nodes.size() * sizeof(DevNode),
                          cudaMemcpyHostToDevice));

    const size_t triIdxCount = std::max<size_t>(1, st.flat.triIdx.size());
    CUDA_CHECK(cudaMalloc(&dev.triIdx, triIdxCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dev.triIdx, st.flat.triIdx.data(), st.flat.triIdx.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev.areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev.areas, st.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const size_t matCount = std::max<size_t>(1, N * L);
    CUDA_CHECK(cudaMalloc(&dev.kijT, matCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.tauT, matCount * sizeof(int)));

    const size_t radCount = std::max<size_t>(1, T * N);
    CUDA_CHECK(cudaMalloc(&dev.radB, radCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.radB, 0, radCount * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dev.nonZero, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dev.nonZero, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaStreamCreate(&dev.stream));

    dev.recvCounts.resize(g_size);
    dev.recvDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        RowPartition p = partitionRows(N, r, g_size);
        dev.recvCounts[r] = static_cast<int>(p.count());
        dev.recvDispls[r] = static_cast<int>(p.begin);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Fills one batch of raw RNG draws (in the exact order of the sequential original)
// and the slot table mapping (receiver-in-batch, emitter) -> draw group.
static uint32_t generateBatch(const SimulationState& st, RandomGenerator& rng,
                              size_t rowStart, int rows, float* out, uint32_t* slotT) {
    const size_t N = st.numTriangles;
    uint32_t groups = 0;
    size_t pos = 0;
    for (int iB = 0; iB < rows; ++iB) {
        const size_t i = rowStart + static_cast<size_t>(iB);
        const Triangle& triI = st.triangles[i];
        for (size_t j = 0; j < N; ++j) {
            uint32_t g = INVALID_GROUP;
            if (i != j && !isCulled(triI, st.triangles[j])) {
                g = groups++;
                for (int r = 0; r < NUM_RAYS; ++r) {
                    for (int k = 0; k < 4; ++k) out[pos++] = rng.rand();
                }
            }
            slotT[j * static_cast<size_t>(rows) + iB] = g;
        }
    }
    return groups;
}

static void computeTimeDelays(SimulationState& st, const RowPartition& part, DeviceData& dev) {
    RPRINTF("Computing time delays (Tau)...\n");
    const size_t N = st.numTriangles;
    const size_t total = part.count() * N;
    if (total == 0) return;
    tauKernel<<<blocksFor(total), BLOCK_SIZE, 0, dev.stream>>>(
        dev.tris, dev.tauT, static_cast<int>(N), static_cast<int>(part.count()), static_cast<int>(part.begin));
    CUDA_CHECK(cudaGetLastError());
}

// One form-factor pipeline per host thread: two streams with double-buffered staging so that
// host RNG/slot generation for one batch overlaps with the upload and kernel of the previous one.
struct FFLane {
    cudaStream_t stream[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    float* dRng[2] = {nullptr, nullptr};
    uint32_t* dSlot[2] = {nullptr, nullptr};
    float* hRng[2] = {nullptr, nullptr};
    uint32_t* hSlot[2] = {nullptr, nullptr};
};

// capPairs: maximum (receiver, emitter) pairs per batch
static void allocLane(FFLane& lane, size_t capPairs) {
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaStreamCreate(&lane.stream[s]));
        CUDA_CHECK(cudaEventCreateWithFlags(&lane.done[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&lane.dRng[s], capPairs * NUM_RAYS * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&lane.dSlot[s], capPairs * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&lane.hRng[s], capPairs * NUM_RAYS * 4 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&lane.hSlot[s], capPairs * sizeof(uint32_t)));
    }
}

static void freeLane(FFLane& lane) {
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaStreamDestroy(lane.stream[s]));
        CUDA_CHECK(cudaEventDestroy(lane.done[s]));
        CUDA_CHECK(cudaFree(lane.dRng[s]));
        CUDA_CHECK(cudaFree(lane.dSlot[s]));
        CUDA_CHECK(cudaFreeHost(lane.hRng[s]));
        CUDA_CHECK(cudaFreeHost(lane.hSlot[s]));
    }
}

struct FFShared {
    const SimulationState* st;
    const DeviceData* dev;
    const mt_jump::JumpTable* jump;   // may be null if no thread starts beyond JUMP_MIN_DRAWS
    RowPartition part;                // this rank's receiver rows
    int device;
};

// Computes form factors for receiver rows [sBegin, sEnd) (rank-local indices) on one lane
static void formFactorRange(const FFShared& sh, FFLane& lane, size_t sBegin, size_t sEnd, bool reportProgress) {
    const SimulationState& st = *sh.st;
    const DeviceData& dev = *sh.dev;
    const size_t N = st.numTriangles;
    const size_t L = sh.part.count();
    const size_t rowsHere = sEnd - sBegin;
    if (rowsHere == 0) return;

    CUDA_CHECK(cudaSetDevice(sh.device));

    // Each batch launches about BATCH_PAIRS threads (or one full receiver row block)
    const size_t batchRows = std::max<size_t>(1, std::min<size_t>(rowsHere, BATCH_PAIRS / std::max<size_t>(N, 1)));

    // This thread's stream starts at the draws consumed by all earlier receiver rows
    RandomGenerator rng(sh.jump, static_cast<uint64_t>(NUM_RAYS) * 4ull * st.groupPrefix[sh.part.begin + sBegin]);

    int batchIdx = 0;
    for (size_t start = sBegin; start < sEnd; start += batchRows, ++batchIdx) {
        const int s = batchIdx & 1;
        const int rows = static_cast<int>(std::min(batchRows, sEnd - start));

        // Wait until the buffers used two batches ago are free again
        if (batchIdx >= 2) CUDA_CHECK(cudaEventSynchronize(lane.done[s]));

        const uint32_t groups = generateBatch(st, rng, sh.part.begin + start, rows, lane.hRng[s], lane.hSlot[s]);

        CUDA_CHECK(cudaMemcpyAsync(lane.dRng[s], lane.hRng[s],
                                   static_cast<size_t>(groups) * NUM_RAYS * 4 * sizeof(float),
                                   cudaMemcpyHostToDevice, lane.stream[s]));
        CUDA_CHECK(cudaMemcpyAsync(lane.dSlot[s], lane.hSlot[s], static_cast<size_t>(rows) * N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, lane.stream[s]));

        FormFactorArgs args{};
        args.scene.tris = dev.tris;
        args.scene.nodes = dev.nodes;
        args.scene.triIdx = dev.triIdx;
        args.rng = lane.dRng[s];
        args.slot = lane.dSlot[s];
        args.kijT = dev.kijT;
        args.nonZero = dev.nonZero;
        args.N = static_cast<int>(N);
        args.L = static_cast<int>(L);
        args.rowBase = static_cast<int>(sh.part.begin);
        args.batchStart = static_cast<int>(start);
        args.batchRows = rows;

        formFactorKernel<<<blocksFor(static_cast<size_t>(rows) * N), BLOCK_SIZE, 0, lane.stream[s]>>>(args);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(lane.done[s], lane.stream[s]));

        if (reportProgress && g_rank == 0 && (batchIdx % 16 == 0 || start + rows == sEnd)) {
            printf("  Progress: %zu/%zu triangles (rank 0 rows)\n", start + rows - sBegin, rowsHere);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(lane.stream[0]));
    CUDA_CHECK(cudaStreamSynchronize(lane.stream[1]));
}

// Returns the global number of form factors above EPSILON
static uint64_t computeFormFactors(SimulationState& st, const RowPartition& part, DeviceData& dev,
                                   int device, int maxThreads) {
    RPRINTF("Computing form factors (Kij)...\n");
    const size_t N = st.numTriangles;
    const size_t L = part.count();

    if (L > 0) {
        // Threads only pay off when there is enough host RNG work to amortize the jump setup
        const size_t work = L * N;
        const size_t byWork = std::max<size_t>(1, work / MIN_PAIRS_PER_THREAD);
        const int T = static_cast<int>(std::max<size_t>(1, std::min<size_t>(std::min<size_t>(L, byWork),
                                                                           static_cast<size_t>(maxThreads))));

        // Thread starts far into the stream use the jump table; short skips are plain discards
        const uint64_t lastStart = 64ull * st.groupPrefix[part.begin + L * (T - 1) / T];
        std::optional<mt_jump::JumpTable> jump;
        if (lastStart >= JUMP_MIN_DRAWS) jump.emplace(mt_jump::buildJumpTable(42));

        // Resources are created once, before the parallel region (driver calls are serialized)
        const size_t capPairs = std::max<size_t>(BATCH_PAIRS, N);
        std::vector<FFLane> lanes(static_cast<size_t>(T));
        for (auto& lane : lanes) allocLane(lane, capPairs);

        FFShared sh{&st, &dev, jump ? &*jump : nullptr, part, device};

        #pragma omp parallel num_threads(T)
        {
            const int tid = omp_get_thread_num();
            const size_t sBegin = L * static_cast<size_t>(tid) / T;
            const size_t sEnd = L * static_cast<size_t>(tid + 1) / T;
            formFactorRange(sh, lanes[static_cast<size_t>(tid)], sBegin, sEnd, tid == 0);
        }

        for (auto& lane : lanes) freeLane(lane);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    uint32_t localCount = 0;
    CUDA_CHECK(cudaMemcpy(&localCount, dev.nonZero, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const uint64_t localNonZero = localCount;

    uint64_t globalNonZero = 0;
    MPI_Allreduce(&localNonZero, &globalNonZero, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    return globalNonZero;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

static void runSimulation(SimulationState& st, const RowPartition& part, DeviceData& dev) {
    RPRINTF("Running wave propagation simulation...\n");

    const size_t N = st.numTriangles;
    const size_t T = st.numTimesteps;
    const int L = static_cast<int>(part.count());
    const int timeOff = static_cast<int>(T / 2);

    for (size_t t = 0; t < T; ++t) {
        if (L > 0) {
            SimArgs a{};
            a.kijT = dev.kijT;
            a.tauT = dev.tauT;
            a.areas = dev.areas;
            a.radB = dev.radB;
            a.N = static_cast<int>(N);
            a.L = L;
            a.rowBase = static_cast<int>(part.begin);
            a.t = static_cast<int>(t);
            a.timeOff = timeOff;
            a.source = static_cast<int>(st.sourceIndex);
            a.rho = st.reflectivity;
            simStepKernel<<<blocksFor(static_cast<size_t>(L)), BLOCK_SIZE, 0, dev.stream>>>(a);
            CUDA_CHECK(cudaGetLastError());

            // Export this rank's slice of radiosity row t
            CUDA_CHECK(cudaMemcpy(st.radB.data() + t * N + part.begin, dev.radB + t * N + part.begin,
                                  static_cast<size_t>(L) * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Exchange slices so every rank holds the complete row t
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, st.radB.data() + t * N,
                       dev.recvCounts.data(), dev.recvDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(dev.radB + t * N, st.radB.data() + t * N, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            RPRINTF("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation, OpenMP over triangles)
// ============================================================================

void computeDistances(SimulationState& state) {
    RPRINTF("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t src = state.sourceIndex;

    #pragma omp parallel for schedule(dynamic, 16)
    for (long long ii = 0; ii < static_cast<long long>(N); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, src)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, uint64_t nonZeroKij) {
    RPRINTF("\nValidation:\n");

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
            RPRINTF("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            RPRINTF("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    RPRINTF("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    RPRINTF("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    RPRINTF("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        RPRINTF("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    RPRINTF("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        RPRINTF("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const size_t totalPairs = state.numTriangles * state.numTriangles;
    RPRINTF("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
            static_cast<unsigned long long>(nonZeroKij), totalPairs,
            100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalPairs));

    if (nonZeroKij == 0) {
        RPRINTF("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    RPRINTF("  Validation: PASSED\n");
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

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
            return finish(0);
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    // Host threads per rank: share the node's cores among the local ranks. Each thread drives
    // its own row slice (RNG stream + CUDA stream pipeline) on this rank's GPU.
    const int threadsPerRank = std::max(1, omp_get_num_procs() / localSize);
    omp_set_num_threads(threadsPerRank);
    const int ffThreads = std::min(MAX_FF_THREADS, threadsPerRank);
    const int device = selectDevice(localRank);

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    RPRINTF("Room Response Simulation Benchmark\n");
    RPRINTF("===================================\n");
    RPRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    RPRINTF("Timesteps: %d\n", timesteps);
    RPRINTF("Source triangle: %d\n", sourceIdx);
    RPRINTF("Reflectivity: %.2f\n", reflectivity);
    RPRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    RPRINTF("\n");

    // Initialize (replicated on every rank)
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const RowPartition part = partitionRows(N, g_rank, g_size);

    DeviceData dev;
    uploadScene(state, part, dev);

    RPRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, part, dev);
    const uint64_t nonZeroKij = computeFormFactors(state, part, dev, device, ffThreads);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    RPRINTF("Precomputation time: %ld ms\n", preDuration);
    RPRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, part, dev);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    RPRINTF("Simulation time: %ld ms\n", simDuration);
    RPRINTF("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    RPRINTF("Distance computation time: %ld ms\n", distDuration);
    RPRINTF("\n");

    // Total time
    long localTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    RPRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = N;
    size_t t = T;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    RPRINTF("\nPerformance:\n");
    RPRINTF("  Triangles: %zu\n", n);
    RPRINTF("  Timesteps: %zu\n", t);
    RPRINTF("  Form factor computations: %.2e\n", kijOps);
    RPRINTF("  Simulation operations: %.2e\n", simOps);
    RPRINTF("  Distance computations: %.2e\n", distOps);
    RPRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage (dense-matrix footprint of the problem, as in the original report)
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    RPRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    RPRINTF("  Result hash: %016lX\n", hash);
    RPRINTF("\n");

    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation (all ranks compute the same result; rank 0 prints)
    bool ok = true;
    if (validate) {
        ok = validateResults(state, nonZeroKij);
        if (!ok) return finish(1);
    }

    return finish(0);
}

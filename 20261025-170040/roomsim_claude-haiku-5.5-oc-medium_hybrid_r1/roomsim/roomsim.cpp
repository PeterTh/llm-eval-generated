/**
 * Room Response Simulation Benchmark
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
 * Parallelization (hybrid, all three levels are always used):
 *  - MPI:    receiver rows are block-distributed over ranks. Each rank owns the Kij/Tau
 *            rows of its receivers and advances their radiosity every timestep; the
 *            radiosity row of each timestep is exchanged with MPI_Allgatherv.
 *  - CUDA:   each rank's GPU evaluates its Kij rows (one thread per receiver row, so the
 *            per-row random stream stays sequential), its Tau rows, the wave propagation
 *            kernel, and the non-zero count used by validation. Each rank drives one GPU.
 *  - OpenMP: host-side work: per-row random-stream offsets and states, culling counts,
 *            and the cross-correlation distance computation.
 *
 * Semantics: the original code draws all form-factor samples from one std::mt19937(42)
 * stream in (i, j) order, consuming draws only for pairs that are not culled. To keep
 * the results identical while parallelizing over rows, every row's engine state is
 * obtained directly by GF(2) polynomial jump-ahead of the MT19937 sequence (see
 * mtjump below), so no draws are wasted and no sequential pass is needed.
 */

#include <algorithm>
#include <array>
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

#ifdef __CUDACC__
#define HD __host__ __device__
#define DEVICE_FN __device__ __forceinline__
#else
#define HD
#define DEVICE_FN
#endif

// ============================================================================
// Parallel runtime helpers
// ============================================================================

static int g_rank = 0;
static int g_size = 1;

#define RPRINTF(...)                 \
    do {                             \
        if (g_rank == 0) printf(__VA_ARGS__); \
    } while (0)

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "[rank %d] CUDA error '%s' at %s:%d: %s\n", g_rank,    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err_));           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
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
constexpr val_t VAL_MAX = 3.40282347e+38f;  // std::numeric_limits<float>::max()

HD inline val_t vmin(val_t a, val_t b) { return (b < a) ? b : a; }  // std::min semantics
HD inline val_t vmax(val_t a, val_t b) { return (a < b) ? b : a; }  // std::max semantics

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x_, val_t y_, val_t z_) : x(x_), y(y_), z(z_) {}
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

    HD Triangle() = default;
    HD Triangle(const Vec3& a_, const Vec3& b_, const Vec3& c_)
        : a(a_), b(b_), c(c_), _normal((b_ - a_).cross(c_ - a_).normalized()) {}

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
// Octree for Spatial Acceleration (built on host, traversed on device)
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

// Flattened octree node used by the device traversal
struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];
    int32_t triStart;
    int32_t triCount;   // > 0 only for leaves
};

constexpr int TRAVERSAL_STACK = 128;

static int32_t flattenOctree(const Octree& node, int depth, std::vector<FlatNode>& nodes,
                             std::vector<int32_t>& triIdx, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    int32_t id = static_cast<int32_t>(nodes.size());
    FlatNode fn{};
    fn.center = node.center;
    fn.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) fn.child[c] = -1;
    fn.triStart = 0;
    fn.triCount = 0;
    if (!node.triangleIndices.empty()) {
        fn.triStart = static_cast<int32_t>(triIdx.size());
        for (size_t idx : node.triangleIndices) triIdx.push_back(static_cast<int32_t>(idx));
        fn.triCount = static_cast<int32_t>(node.triangleIndices.size());
    }
    nodes.push_back(fn);
    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) {
            int32_t cid = flattenOctree(*node.children[c], depth + 1, nodes, triIdx, maxDepth);
            nodes[id].child[c] = cid;
        }
    }
    return id;
}

// ============================================================================
// Mersenne Twister jump-ahead (host)
//
// The original code draws Kij samples from std::mt19937(42) in (i, j) order, so
// each row's draws start at a known offset in that single stream. Engine states
// at arbitrary offsets are obtained from the output sequence alone:
//   - Berlekamp-Massey on the low output bit recovers the linear recurrence P(x)
//     (degree 19937) that every MT19937 output bit satisfies.
//   - o_n = XOR over set bits i of (x^n mod P) of o_i, so the outputs at any
//     position follow from the first 19937 outputs.
//   - The engine state for a block of 624 outputs is the untempered outputs.
// ============================================================================

namespace mtjump {

constexpr size_t MT_WORDS = 624;
constexpr uint32_t MT_SEED = 42;     // Same seed as RandomGenerator(42) in the original
constexpr size_t SEQ_LEN = 40130;    // Must exceed 2 * linear complexity (19937)

// Undo MT19937 tempering
inline uint32_t untemper(uint32_t y) {
    auto unRight = [](uint32_t v, int s) {
        uint32_t r = v;
        for (int i = 0; i < 32 / s + 1; ++i) r = v ^ (r >> s);
        return r;
    };
    auto unLeftAnd = [](uint32_t v, int s, uint32_t m) {
        uint32_t r = v;
        for (int i = 0; i < 32 / s + 1; ++i) r = v ^ ((r << s) & m);
        return r;
    };
    y = unRight(y, 18);
    y = unLeftAnd(y, 15, 0xefc60000u);
    y = unLeftAnd(y, 7, 0x9d2c5680u);
    y = unRight(y, 11);
    return y;
}

// c ^= src << sh  (bit-packed GF(2) polynomials, same word count)
inline void shiftXor(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, size_t sh) {
    size_t ws = sh / 64, bs = sh % 64, W = dst.size();
    for (size_t w = W; w-- > ws;) {
        size_t sw = w - ws;
        uint64_t v = src[sw] << bs;
        if (bs && sw > 0) v |= src[sw - 1] >> (64 - bs);
        dst[w] ^= v;
    }
}

// Berlekamp-Massey over GF(2). Returns the linear complexity L; C holds the
// connection polynomial (bit i = C_i, C_0 = 1).
inline size_t berlekampMassey(const std::vector<uint8_t>& s, std::vector<uint64_t>& C) {
    const size_t n = s.size();
    const size_t W = n / 64 + 2;
    std::vector<uint64_t> B(W, 0), T(W), H(W, 0);
    C.assign(W, 0);
    C[0] = 1;
    B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t k = 0; k < n; ++k) {
        // H bit i == s[k-i]
        for (size_t w = W - 1; w > 0; --w) H[w] = (H[w] << 1) | (H[w - 1] >> 63);
        H[0] = (H[0] << 1) | s[k];
        uint64_t par = 0;
        for (size_t w = 0; w < W; ++w) par ^= C[w] & H[w];
        if (__builtin_parityll(par) == 0) {
            ++m;
            continue;
        }
        T = C;
        shiftXor(C, B, m);
        if (2 * L <= k) {
            L = k + 1 - L;
            B = T;
            m = 1;
        } else {
            ++m;
        }
    }
    return L;
}

struct Jump {
    size_t L = 0;                 // recurrence degree
    size_t NW = 0;                // words per GF(2) polynomial (L/64 + 1)
    std::vector<uint64_t> plow;   // x^L mod P, i.e. P(x) - x^L
    std::vector<uint32_t> out;    // tempered outputs o_0 .. o_{SEQ_LEN-1}
};

inline Jump buildJump() {
    std::mt19937 eng(MT_SEED);
    Jump J;
    J.out.resize(SEQ_LEN);
    for (auto& v : J.out) v = eng();

    std::vector<uint8_t> bits(SEQ_LEN);
    for (size_t i = 0; i < SEQ_LEN; ++i) bits[i] = static_cast<uint8_t>(J.out[i] & 1u);

    std::vector<uint64_t> C;
    J.L = berlekampMassey(bits, C);
    if (2 * J.L > SEQ_LEN) {
        fprintf(stderr, "mtjump: recurrence degree %zu not determined by %zu outputs\n", J.L, SEQ_LEN);
        std::abort();
    }

    J.NW = J.L / 64 + 1;
    J.plow.assign(J.NW, 0);
    // P(x) = sum_i C_i x^(L-i); plow bit k = C_{L-k} for k < L
    for (size_t k = 0; k < J.L; ++k) {
        size_t i = J.L - k;
        if ((C[i / 64] >> (i % 64)) & 1ULL) J.plow[k / 64] |= 1ULL << (k % 64);
    }
    return J;
}

// a <- a * x mod P
inline void mulX(const Jump& J, std::vector<uint64_t>& a) {
    uint64_t carry = 0;
    for (size_t w = 0; w < J.NW; ++w) {
        uint64_t nc = a[w] >> 63;
        a[w] = (a[w] << 1) | carry;
        carry = nc;
    }
    size_t wl = J.L / 64, bl = J.L % 64;
    if ((a[wl] >> bl) & 1ULL) {
        a[wl] &= ~(1ULL << bl);
        for (size_t w = 0; w < J.NW; ++w) a[w] ^= J.plow[w];
    }
}

// a^2 mod P (squaring is linear over GF(2): spread bits, then reduce)
inline std::vector<uint64_t> squareMod(const Jump& J, const std::vector<uint64_t>& a) {
    const size_t R = 2 * J.NW + 2;
    std::vector<uint64_t> t(R, 0);
    for (size_t k = 0; k < J.L; ++k) {
        if ((a[k / 64] >> (k % 64)) & 1ULL) {
            size_t p = 2 * k;
            t[p / 64] |= 1ULL << (p % 64);
        }
    }
    for (int64_t k = 2 * static_cast<int64_t>(J.L) - 2; k >= static_cast<int64_t>(J.L); --k) {
        if ((t[k / 64] >> (k % 64)) & 1ULL) {
            t[k / 64] &= ~(1ULL << (k % 64));
            size_t s = static_cast<size_t>(k) - J.L;
            size_t ws = s / 64, bs = s % 64;
            for (size_t w = 0; w < J.NW; ++w) {
                uint64_t v = J.plow[w];
                if (!v) continue;
                t[w + ws] ^= v << bs;
                if (bs) t[w + ws + 1] ^= v >> (64 - bs);
            }
        }
    }
    return std::vector<uint64_t>(t.begin(), t.begin() + J.NW);
}

// x^m mod P
inline std::vector<uint64_t> powX(const Jump& J, uint64_t m) {
    std::vector<uint64_t> r(J.NW, 0);
    r[0] = 1;
    if (m == 0) return r;
    int hi = 63 - __builtin_clzll(m);
    bool first = true;
    for (int b = hi; b >= 0; --b) {
        if (!first) r = squareMod(J, r);
        first = false;
        if ((m >> b) & 1ULL) mulX(J, r);
    }
    return r;
}

// Standard MT19937 in-place twist (same as libstdc++)
inline void twist(uint32_t* mt) {
    for (size_t k = 0; k < MT_WORDS; ++k) {
        uint32_t y = (mt[k] & 0x80000000u) | (mt[(k + 1) % MT_WORDS] & 0x7fffffffu);
        mt[k] = mt[(k + 397) % MT_WORDS] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
}

// Advance an engine state (words, index p in [0, 624]) by n draws without producing them.
inline void advance(uint32_t* words, size_t& p, uint64_t n) {
    while (n > 0) {
        if (p == MT_WORDS) {
            twist(words);
            p = 0;
        }
        uint64_t take = std::min<uint64_t>(n, MT_WORDS - p);
        p += static_cast<size_t>(take);
        n -= take;
    }
}

// Engine words (post-twist state) for output block `block`, i.e. outputs
// [624*block, 624*block + 624). Consumers then start at index r within the block.
inline void stateForBlock(const Jump& J, uint64_t block, uint32_t* words) {
    std::vector<uint64_t> cur = powX(J, MT_WORDS * block);
    for (size_t j = 0; j < MT_WORDS; ++j) {
        uint32_t acc = 0;
        for (size_t w = 0; w < J.NW; ++w) {
            uint64_t x = cur[w];
            while (x) {
                acc ^= J.out[w * 64 + static_cast<size_t>(__builtin_ctzll(x))];
                x &= x - 1;
            }
        }
        words[j] = untemper(acc);
        mulX(J, cur);
    }
}

}  // namespace mtjump

// ============================================================================
// Device-side geometry and RNG
// ============================================================================

// Ray-box test (same arithmetic as the original Octree::rayIntersectsBox)
HD inline bool rayBoxHit(const Vec3& center, const Vec3& halfExtent, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Moller-Trumbore ray-triangle intersection
HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return VAL_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return VAL_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return VAL_MAX;

    return e2.dot(qvec) * invDet;
}

// Cosine between vector and normal (clamped at zero)
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return vmax(ZERO, v.dot(normal) / vNorm);
}

// MT19937 tempering and in-place twist (identical to libstdc++ std::mt19937)
DEVICE_FN uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

DEVICE_FN void mtTwist(uint32_t* mt) {
    for (int k = 0; k < 624; ++k) {
        uint32_t y = (mt[k] & 0x80000000u) | (mt[(k + 1) % 624] & 0x7fffffffu);
        mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
}

// Equivalent of std::uniform_real_distribution<float>(0,1)(std::mt19937&)
// (libstdc++ consumes exactly one 32-bit draw per value for float/mt19937).
DEVICE_FN val_t mtUniform(uint32_t* st, int& p) {
    if (p >= 624) {
        mtTwist(st);
        p = 0;
    }
    uint32_t y = mtTemper(st[p++]);
    val_t ret = static_cast<val_t>(y) / 4294967296.0f;
    if (ret >= 1.0f) ret = 0.99999994f;  // nextafter(1, 0)
    return ret;
}

// ============================================================================
// Device Ray Tracing and Form Factors
// ============================================================================

// Octree traversal for the visibility test (boolean result, so traversal order
// does not affect the answer; the same triangles are tested as in the original)
DEVICE_FN bool isRayBlockedDev(const Vec3& from, const Vec3& to,
                               const Triangle* tris, const FlatNode* nodes, const int32_t* triIdx,
                               int32_t srcIdx, int32_t dstIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int32_t stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const FlatNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (int k = 0; k < nd.triCount; ++k) {
                int32_t idx = triIdx[nd.triStart + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int32_t ch = nd.child[c];
                if (ch >= 0 && rayBoxHit(nodes[ch].center, nodes[ch].halfExtent, from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

DEVICE_FN Vec3 randomPointInTriangleDev(const Triangle& t, uint32_t* st, int& p) {
    val_t u = mtUniform(st, p);
    val_t v = mtUniform(st, p);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Form factor Kij between receiver i and emitter j. Consumes exactly 4 draws per
// ray, and no draws for culled pairs, matching the original sequential stream.
DEVICE_FN val_t computeKijDev(int i, int j, const Triangle* tris, const FlatNode* nodes,
                              const int32_t* triIdx, uint32_t* st, int& p) {
    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleDev(triI, st, p);
        Vec3 pJ = randomPointInTriangleDev(triJ, st, p);

        if (isRayBlockedDev(pI, pJ, tris, nodes, triIdx, i, j)) continue;

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
// Kernels
// ============================================================================

// One thread per (locally owned receiver row, emitter chunk). Each thread walks its chunk
// of emitters in order, continuing the random stream of the original sequential loop from
// the state recorded at the chunk start (states[g], index startIdx[g]).
__global__ void kijKernel(const Triangle* __restrict__ tris, const FlatNode* __restrict__ nodes,
                          const int32_t* __restrict__ triIdx, const uint32_t* __restrict__ states,
                          const int32_t* __restrict__ startIdx, int N, int i0, int nLocal,
                          int chunksPerRow, int chunkLen, float* __restrict__ kij) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nLocal * chunksPerRow) return;
    int li = g / chunksPerRow;
    int c = g - li * chunksPerRow;
    int i = i0 + li;
    int jBeg = c * chunkLen;
    int jEnd = min(N, jBeg + chunkLen);

    uint32_t st[mtjump::MT_WORDS];
    for (int k = 0; k < static_cast<int>(mtjump::MT_WORDS); ++k) {
        st[k] = states[static_cast<size_t>(g) * mtjump::MT_WORDS + k];
    }
    int p = startIdx[g];

    float* row = kij + static_cast<size_t>(li) * N;
    for (int j = jBeg; j < jEnd; ++j) {
        row[j] = (j == i) ? ZERO : computeKijDev(i, j, tris, nodes, triIdx, st, p);
    }
}

// Time delays Tau(i, j) = ceil(|c_i - c_j| / WAVE_SPEED) for locally owned rows
__global__ void tauKernel(const Triangle* __restrict__ tris, int N, int i0, int nLocal,
                          int32_t* __restrict__ tau) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;
    int i = i0 + li;
    Vec3 ci = tris[i].center();
    int32_t* row = tau + static_cast<size_t>(li) * N;
    for (int j = 0; j < N; ++j) {
        if (j == i) {
            row[j] = 0;
            continue;
        }
        val_t dist = (ci - tris[j].center()).norm();
        row[j] = static_cast<int32_t>(ceilf(dist * INV_WAVE_SPEED));
    }
}

// Wave propagation for one timestep t, for locally owned receivers.
// Each thread sums over emitters in the same order as the original code. Reads only
// radB rows < t, because distinct triangle centers give Tau >= 1, so all receivers of
// one timestep are independent and can run concurrently.
__global__ void simulateStepKernel(int t, int N, int i0, int nLocal, int src, int timeOff,
                                   const float* __restrict__ kijL, const int32_t* __restrict__ tauL,
                                   const float* __restrict__ areas, float rho,
                                   float* __restrict__ radB) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;
    int i = i0 + li;
    const float* kRow = kijL + static_cast<size_t>(li) * N;
    const int32_t* tRow = tauL + static_cast<size_t>(li) * N;

    float sumB = ZERO;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        int tauij = tRow[j];
        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float kij = kRow[j];
        if (kij <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        float radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += vmin(kij * areas[j], ONE) * radJ;
    }

    // Emission from the source (active for first half of timesteps)
    float radE = (t < timeOff && i == src) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE;
}

// Count entries above EPSILON (validation)
__global__ void countNonZeroKernel(const float* __restrict__ kij, size_t n,
                                   unsigned long long* __restrict__ count) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < n; k += stride) {
        if (kij[k] > EPSILON) ++local;
    }
    if (local) atomicAdd(count, local);
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
// Room Response Simulation State
// ============================================================================

// Row partition of receiver triangles over MPI ranks
struct Partition {
    std::vector<int> counts;   // rows per rank
    std::vector<int> displs;   // first row per rank
    int i0 = 0;                // first local row
    int nLocal = 0;            // number of local rows
};

static Partition makePartition(size_t N) {
    Partition p;
    p.counts.resize(g_size);
    p.displs.resize(g_size);
    int base = static_cast<int>(N) / g_size;
    int rem = static_cast<int>(N) % g_size;
    int off = 0;
    for (int r = 0; r < g_size; ++r) {
        p.counts[r] = base + (r < rem ? 1 : 0);
        p.displs[r] = off;
        off += p.counts[r];
    }
    p.i0 = p.displs[g_rank];
    p.nLocal = p.counts[g_rank];
    return p;
}

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;
    std::vector<val_t> radB;        // Reflected radiosity (T x N, full, all ranks)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    std::vector<FlatNode> flatNodes;
    std::vector<int32_t> flatTriIdx;
    int maxOctreeDepth = 0;

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Device-resident data owned by this rank
struct DeviceContext {
    Triangle* tris = nullptr;
    FlatNode* nodes = nullptr;
    int32_t* triIdx = nullptr;
    float* areas = nullptr;
    uint32_t* mtStates = nullptr;   // nLocal x 624
    int32_t* mtIndex = nullptr;     // nLocal
    float* kij = nullptr;           // nLocal x N
    int32_t* tau = nullptr;         // nLocal x N
    float* radB = nullptr;          // T x N
};

static void* allocDevice(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes > 0 ? bytes : 1));
    return ptr;
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
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    RPRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    RPRINTF("Building octree...\n");
    state.octree.build(state.triangles);

    state.flatNodes.clear();
    state.flatTriIdx.clear();
    state.maxOctreeDepth = 0;
    flattenOctree(state.octree, 0, state.flatNodes, state.flatTriIdx, state.maxOctreeDepth);
    if (1 + 8 * state.maxOctreeDepth > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep for device traversal stack\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize radiosity history (zero-filled, filled during simulation)
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Upload geometry and the octree to this rank's GPU
static void uploadGeometry(DeviceContext& dev, const SimulationState& state) {
    size_t N = state.numTriangles;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev.tris), N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(dev.tris, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));

    dev.nodes = static_cast<FlatNode*>(allocDevice(state.flatNodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMemcpy(dev.nodes, state.flatNodes.data(), state.flatNodes.size() * sizeof(FlatNode),
                          cudaMemcpyHostToDevice));

    dev.triIdx = static_cast<int32_t*>(allocDevice(state.flatTriIdx.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(dev.triIdx, state.flatTriIdx.data(), state.flatTriIdx.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    dev.areas = static_cast<float*>(allocDevice(N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev.areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
}

// Stream offset for each receiver row in the original sequential draw order.
// Row i starts after all draws of rows < i; a non-culled pair consumes 4 draws per ray.
static void computeRowStarts(const std::vector<Triangle>& tris, std::vector<uint64_t>& rowDraws) {
    const size_t N = tris.size();
    std::vector<uint64_t> cnt(N, 0);
#pragma omp parallel for schedule(dynamic, 16)
    for (long long i = 0; i < static_cast<long long>(N); ++i) {
        const Triangle& triI = tris[i];
        uint64_t c = 0;
        for (size_t j = 0; j < N; ++j) {
            if (j == static_cast<size_t>(i)) continue;
            if (!(triI.normal().dot(tris[j].normal()) > 0.99f)) ++c;
        }
        cnt[i] = c;
    }
    rowDraws.assign(N + 1, 0);
    for (size_t i = 0; i < N; ++i) {
        rowDraws[i + 1] = rowDraws[i] + cnt[i] * static_cast<uint64_t>(4 * NUM_RAYS);
    }
}

// Tau on the device (only the local rows)
static void computeTimeDelays(const SimulationState& state, DeviceContext& dev, const Partition& part) {
    RPRINTF("Computing time delays (Tau)...\n");
    size_t N = state.numTriangles;
    dev.tau = static_cast<int32_t*>(allocDevice(static_cast<size_t>(part.nLocal) * N * sizeof(int32_t)));
    if (part.nLocal > 0) {
        int block = 128;
        int grid = (part.nLocal + block - 1) / block;
        tauKernel<<<grid, block>>>(dev.tris, static_cast<int>(N), part.i0, part.nLocal, dev.tau);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Form factors on the device. Work items are (local row, emitter chunk) pairs so that the
// GPU has enough independent threads to hide memory latency during ray traversal. The
// random-stream state at the start of every chunk is computed on the host (OpenMP): each
// thread jumps once to its first row, then steps forward through the stream, recording the
// state at each chunk start. This reproduces the original single-stream draw order exactly.
static void computeFormFactors(const SimulationState& state, DeviceContext& dev, const Partition& part,
                               const mtjump::Jump& jump, int targetThreads) {
    RPRINTF("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const auto& tris = state.triangles;
    constexpr size_t MIN_CHUNK = 64;
    constexpr uint64_t DRAWS_PER_PAIR = 4 * NUM_RAYS;

    std::vector<uint64_t> rowDraws;
    computeRowStarts(tris, rowDraws);

    // Chunking: about targetThreads work items for this rank, chunks of at least MIN_CHUNK emitters
    const size_t nLocal = static_cast<size_t>(part.nLocal);
    size_t desiredChunks = std::max<size_t>(1, (static_cast<size_t>(targetThreads) + std::max<size_t>(1, nLocal) - 1) /
                                                   std::max<size_t>(1, nLocal));
    size_t chunkLen = std::max(MIN_CHUNK, (N + desiredChunks - 1) / desiredChunks);
    const int chunksPerRow = static_cast<int>((N + chunkLen - 1) / chunkLen);
    const int chunkLenI = static_cast<int>(chunkLen);
    const size_t numItems = nLocal * static_cast<size_t>(chunksPerRow);

    std::vector<uint32_t> hStates(numItems * mtjump::MT_WORDS);
    std::vector<int32_t> hIndex(numItems);
#pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int a = static_cast<int>(static_cast<long long>(part.nLocal) * tid / nt);
        const int b = static_cast<int>(static_cast<long long>(part.nLocal) * (tid + 1) / nt);
        if (a < b) {
            uint64_t off = rowDraws[part.i0 + a];
            uint32_t st[mtjump::MT_WORDS];
            mtjump::stateForBlock(jump, off / mtjump::MT_WORDS, st);
            size_t p = static_cast<size_t>(off % mtjump::MT_WORDS);
            for (int li = a; li < b; ++li) {
                const int i = part.i0 + li;
                const Triangle& triI = tris[i];
                for (int c = 0; c < chunksPerRow; ++c) {
                    const size_t g = static_cast<size_t>(li) * chunksPerRow + c;
                    std::memcpy(&hStates[g * mtjump::MT_WORDS], st, sizeof(st));
                    hIndex[g] = static_cast<int32_t>(p);

                    // Draws consumed by this chunk: 64 per non-culled pair (culled pairs draw nothing)
                    const size_t jBeg = static_cast<size_t>(c) * chunkLen;
                    const size_t jEnd = std::min(N, jBeg + chunkLen);
                    uint64_t notCulled = 0;
                    for (size_t j = jBeg; j < jEnd; ++j) {
                        if (j == static_cast<size_t>(i)) continue;
                        if (!(triI.normal().dot(tris[j].normal()) > 0.99f)) ++notCulled;
                    }
                    mtjump::advance(st, p, notCulled * DRAWS_PER_PAIR);
                }
            }
        }
    }

    dev.mtStates = static_cast<uint32_t*>(allocDevice(hStates.size() * sizeof(uint32_t)));
    dev.mtIndex = static_cast<int32_t*>(allocDevice(hIndex.size() * sizeof(int32_t)));
    dev.kij = static_cast<float*>(allocDevice(nLocal * N * sizeof(float)));
    if (numItems > 0) {
        CUDA_CHECK(cudaMemcpy(dev.mtStates, hStates.data(), hStates.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.mtIndex, hIndex.data(), hIndex.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        int block = 128;
        int grid = static_cast<int>((numItems + block - 1) / block);
        kijKernel<<<grid, block>>>(dev.tris, dev.nodes, dev.triIdx, dev.mtStates, dev.mtIndex,
                                   static_cast<int>(N), part.i0, part.nLocal, chunksPerRow, chunkLenI,
                                   dev.kij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    RPRINTF("  Progress: %zu/%zu triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, DeviceContext& dev, const Partition& part) {
    RPRINTF("Running wave propagation simulation...\n");
    size_t N = state.numTriangles;
    int T = static_cast<int>(state.numTimesteps);
    int timeOff = T / 2;  // source emits during the first half of the timesteps
    int src = static_cast<int>(state.sourceIndex);
    val_t rho = state.reflectivity;

    dev.radB = static_cast<float*>(allocDevice(static_cast<size_t>(T) * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.radB, 0, static_cast<size_t>(T) * N * sizeof(float)));

    std::vector<float> hSend(part.nLocal > 0 ? part.nLocal : 1);
    int block = 128;
    int grid = (part.nLocal + block - 1) / block;

    for (int t = 0; t < T; ++t) {
        if (part.nLocal > 0) {
            simulateStepKernel<<<grid, block>>>(t, static_cast<int>(N), part.i0, part.nLocal, src,
                                                timeOff, dev.kij, dev.tau, dev.areas, rho, dev.radB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hSend.data(), dev.radB + static_cast<size_t>(t) * N + part.i0,
                                  static_cast<size_t>(part.nLocal) * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Share this timestep's radiosity row among all ranks
        MPI_Allgatherv(hSend.data(), part.nLocal, MPI_FLOAT, state.radB.data() + static_cast<size_t>(t) * N,
                       part.counts.data(), part.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(dev.radB + static_cast<size_t>(t) * N,
                              state.radB.data() + static_cast<size_t>(t) * N, N * sizeof(float),
                              cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            RPRINTF("  Timestep %d/%d\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, const Partition& part) {
    RPRINTF("Computing distances via cross-correlation...\n");
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    size_t src = state.sourceIndex;

#pragma omp parallel for schedule(static)
    for (int li = 0; li < part.nLocal; ++li) {
        size_t i = static_cast<size_t>(part.i0 + li);
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[tt * N + i];
                val_t pS = state.radB[(tt - t) * N + src];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   part.counts.data(), part.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
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
    size_t totalKij = state.numTriangles * state.numTriangles;
    RPRINTF("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
            nonZeroKij, totalKij,
            100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalKij));

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
            if (g_rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    // Bind this rank to a GPU (ranks on a node are spread round-robin over its GPUs)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "[rank %d] roomsim requires at least one CUDA device\n", g_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    MPI_Comm_free(&localComm);

    // Share host cores between co-located ranks
    int cores = omp_get_num_procs();
    omp_set_num_threads(std::max(1, cores / localSize));

    RPRINTF("Room Response Simulation Benchmark\n");
    RPRINTF("===================================\n");
    RPRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    RPRINTF("Timesteps: %d\n", timesteps);
    RPRINTF("Source triangle: %d\n", sourceIdx);
    RPRINTF("Reflectivity: %.2f\n", reflectivity);
    RPRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    RPRINTF("MPI ranks: %d, OpenMP threads per rank: %d, GPUs per node in use: %d\n",
            g_size, omp_get_max_threads(), std::min(localSize, numDevices));
    RPRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    size_t N = state.numTriangles;
    Partition part = makePartition(N);

    RPRINTF("\n");

    DeviceContext dev;
    uploadGeometry(dev, state);

    // Random-stream jump table (shared read-only by all OpenMP threads)
    mtjump::Jump jump = mtjump::buildJump();

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = MPI_Wtime();

    computeTimeDelays(state, dev, part);
    // Work items per GPU for the form-factor kernel (shared by the ranks mapped to that GPU)
    const int ranksPerDevice = (localSize + numDevices - 1) / numDevices;
    const int targetThreads = (1 << 18) / std::max(1, ranksPerDevice);
    computeFormFactors(state, dev, part, jump, targetThreads);

    MPI_Barrier(MPI_COMM_WORLD);
    double endPre = MPI_Wtime();
    long preDuration = static_cast<long>((endPre - startPre) * 1000.0);

    RPRINTF("Precomputation time: %ld ms\n", preDuration);
    RPRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = MPI_Wtime();

    runSimulation(state, dev, part);

    MPI_Barrier(MPI_COMM_WORLD);
    double endSim = MPI_Wtime();
    long simDuration = static_cast<long>((endSim - startSim) * 1000.0);

    RPRINTF("Simulation time: %ld ms\n", simDuration);
    RPRINTF("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = MPI_Wtime();

    computeDistances(state, part);

    MPI_Barrier(MPI_COMM_WORLD);
    double endDist = MPI_Wtime();
    long distDuration = static_cast<long>((endDist - startDist) * 1000.0);

    RPRINTF("Distance computation time: %ld ms\n", distDuration);
    RPRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    RPRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
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

    // Memory usage
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

    // Validation
    int status = 0;
    if (validate) {
        // Count non-zero form factors across all ranks
        unsigned long long localNonZero = 0;
        unsigned long long* dCount = static_cast<unsigned long long*>(allocDevice(sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
        size_t localElems = static_cast<size_t>(part.nLocal) * N;
        if (localElems > 0) {
            countNonZeroKernel<<<256, 256>>>(dev.kij, localElems, dCount);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(&localNonZero, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dCount));

        unsigned long long nonZeroKij = 0;
        MPI_Allreduce(&localNonZero, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (!validateResults(state, nonZeroKij)) {
            status = 1;
        }
    }

    // Release device memory
    cudaFree(dev.tris);
    cudaFree(dev.nodes);
    cudaFree(dev.triIdx);
    cudaFree(dev.areas);
    cudaFree(dev.mtStates);
    cudaFree(dev.mtIndex);
    cudaFree(dev.kij);
    cudaFree(dev.tau);
    cudaFree(dev.radB);

    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    int rc = runBenchmark(argc, argv);

    MPI_Finalize();
    return rc;
}

/**
 * Room Response Simulation Benchmark (CUDA implementation)
 *
 * GPU-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization notes
 * ---------------------
 * Every compute phase runs on the GPU:
 *  - Form factors: one block per receiving triangle, 16 lanes per triangle pair
 *    (one lane per visibility ray); the octree is traversed warp-cooperatively
 *    (a shared stack of (node, lane mask) entries) so that control flow is uniform
 *    while every lane performs exactly the box and triangle tests of the original
 *    traversal.  The original consumes a single sequential std::mt19937 stream;
 *    that exact stream is reproduced on the GPU by computing the stream offset of
 *    every pair (prefix sums of the number of random values each pair consumes)
 *    and by generating the stream in parallel using GF(2) jump-ahead for MT19937
 *    (minimal polynomial via Berlekamp-Massey, x^J mod p(x) applied with a Horner
 *    scheme).  The random values are bit-identical to those of the original
 *    std::uniform_real_distribution<float>(0,1) over std::mt19937(42).
 *  - Time delays: one thread per pair.
 *  - Wave propagation: one thread per receiving triangle per timestep (the sum
 *    over emitters keeps the sequential accumulation order of the original).
 *  - Cross-correlation: one block per triangle, one thread per lag.
 *
 * Floating-point evaluation mirrors the host compiler's contraction choices for
 * the original code where a discrete decision depends on it (pair culling, time
 * delays, accumulation order), so results match the sequential program bit for bit.
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
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__), #call);                                   \
            exit(1);                                                                     \
        }                                                                                \
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
constexpr val_t VAL_MAX = 3.402823466e+38f;  // std::numeric_limits<float>::max()

// ============================================================================
// Vector and Triangle Types
// ============================================================================

// Three-term dot product evaluated as fma(z, fma(x, y*y)).  This mirrors the
// contraction pattern the host compiler generates for the original code, so that
// discrete decisions derived from dot products (back-face culling of pairs, which
// determines how many random numbers a pair consumes, and the time delays) are
// taken identically on the device.
__host__ __device__ __forceinline__ val_t dot3(val_t x, val_t y, val_t z, val_t ox, val_t oy, val_t oz) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(z, oz, __fmaf_rn(x, ox, __fmul_rn(y, oy)));
#else
    return std::fma(z, oz, std::fma(x, ox, y * oy));
#endif
}

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return dot3(x, y, z, o.x, o.y, o.z); }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return dot3(x, y, z, x, y, z); }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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

// Device-side triangle layout (16-byte aligned components)
struct DevTri {
    float4 a, b, c, n;
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
// Octree for Spatial Acceleration (built on the host, traversed on the device)
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
    // Flatten the tree into array form for the device (pre-order numbering).
    // A node is a leaf iff its triangle list is non-empty (as in the original traversal).
    // To keep device traversal to one memory round trip per visited node, each node
    // stores the boxes of its 8 children inline, and each leaf slot stores the
    // vertices of its triangle inline.
    struct Flat {
        std::vector<int4> childIdx;    // 2 per node: child indices 0..7 (-1 if absent)
        std::vector<float4> childBox;  // 16 per node: [2c] = child center, [2c+1] = child half extent
        std::vector<int2> leaf;        // (first slot, count) per node; count = 0 for inner nodes
        std::vector<int> slotIdx;      // triangle index per leaf slot
        std::vector<float4> slotVerts; // 3 per leaf slot: triangle vertices a, b, c
    };

    void flatten(Flat& f) const {
        const size_t myIdx = f.leaf.size();
        f.leaf.push_back(make_int2(static_cast<int>(f.slotIdx.size()), static_cast<int>(triangleIndices.size())));
        f.childIdx.resize(f.childIdx.size() + 2, make_int4(-1, -1, -1, -1));
        f.childBox.resize(f.childBox.size() + 16, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        for (size_t t : triangleIndices) {
            const Triangle& tri = (*allTriangles)[t];
            f.slotIdx.push_back(static_cast<int>(t));
            f.slotVerts.push_back(make_float4(tri.a.x, tri.a.y, tri.a.z, 0.0f));
            f.slotVerts.push_back(make_float4(tri.b.x, tri.b.y, tri.b.z, 0.0f));
            f.slotVerts.push_back(make_float4(tri.c.x, tri.c.y, tri.c.z, 0.0f));
        }
        for (int i = 0; i < 8; ++i) {
            if (!children[i]) continue;
            const int childNode = static_cast<int>(f.leaf.size());
            int* ci = reinterpret_cast<int*>(&f.childIdx[myIdx * 2]);
            ci[i] = childNode;
            const Octree& ch = *children[i];
            f.childBox[myIdx * 16 + 2 * i] = make_float4(ch.center.x, ch.center.y, ch.center.z, 0.0f);
            f.childBox[myIdx * 16 + 2 * i + 1] = make_float4(ch.halfExtent.x, ch.halfExtent.y, ch.halfExtent.z, 0.0f);
            children[i]->flatten(f);
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
// Random Number Generation: parallel reproduction of the std::mt19937 stream
// ============================================================================
//
// The original draws all random numbers from a single std::mt19937(42) through a
// std::uniform_real_distribution<float>(0, 1).  Each non-culled pair (i, j)
// consumes exactly NUM_RAYS * 4 values, so the stream offset of every pair is a
// prefix sum over the pair sequence.  The stream itself is produced on the GPU:
// the MT19937 state transition is linear over GF(2); using its minimal polynomial
// p(x) (found once via Berlekamp-Massey) the state after J steps is g(A)*state
// with g = x^J mod p, which a thread block evaluates with a Horner scheme.  The
// state space is partitioned into equally spaced segments by a doubling tree of
// such jumps; every segment is then expanded by one warp.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr uint32_t MT_SEED = 42u;

constexpr int RNG_RANDS_PER_PAIR = NUM_RAYS * 4;               // u, v for two points per ray
constexpr int RNG_BLOCKS_PER_SEG = 256;                         // MT blocks per expansion segment
constexpr int RNG_SEG_LEN = MT_N * RNG_BLOCKS_PER_SEG;          // random numbers per segment
constexpr int RNG_MAX_SEGS = 1024;                              // segments per stream chunk
constexpr long long RNG_CHUNK_LEN = static_cast<long long>(RNG_MAX_SEGS) * RNG_SEG_LEN;
constexpr int RNG_MAX_TREE_LEVELS = 40;                         // supports up to 2^40 segments
constexpr int JUMP_THREADS = 208;                               // 3 state words per thread
constexpr int JUMP_GROUP = 128;                                 // Horner steps per group (<= 227)

__host__ __device__ __forceinline__ uint32_t mtTwistWord(uint32_t x0, uint32_t x1, uint32_t x397) {
    uint32_t y = (x0 & MT_UPPER) | (x1 & MT_LOWER);
    return x397 ^ (y >> 1) ^ ((x1 & 1u) ? MT_MATRIX_A : 0u);
}

__host__ __device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::generate_canonical<float, 24>(mt19937) followed by uniform_real_distribution(0,1)
__device__ __forceinline__ float mtToCanonical(uint32_t u) {
    float f = __uint2float_rn(u) * (1.0f / 4294967296.0f);  // exact division by 2^32
    return (f >= 1.0f) ? 0.99999994f : f;                    // nextafter(1, 0)
}

// --- GF(2)[x] polynomial helpers (host) ---------------------------------------

using Poly = std::vector<uint64_t>;  // bit k = coefficient of x^k

static int polyWords(int bits) { return (bits + 63) / 64; }
static inline int polyBit(const Poly& p, int k) { return (p[k >> 6] >> (k & 63)) & 1; }
static inline void polySetBit(Poly& p, int k) { p[k >> 6] |= uint64_t(1) << (k & 63); }

// Berlekamp-Massey over GF(2): shortest LFSR generating bit sequence a[0..N).
// Returns the connection polynomial C (C[0] = 1) and its length L.
static void berlekampMassey(const Poly& a, int N, Poly& C, int& L) {
    const int W = polyWords(N + 1);
    C.assign(W, 0);
    Poly B(W + 1, 0), T;   // B is stored with one leading zero word (B[k] at index k + 1)
    C[0] = 1;
    B[1] = 1;
    L = 0;
    int m = 1;
    int degB = 0;          // upper bound on the degree of B

    Poly arev(W + 2, 0);  // arev[k] = a[N-1-k], zero padded
    for (int n = 0; n < N; ++n)
        if (polyBit(a, n)) polySetBit(arev, N - 1 - n);

    for (int n = 0; n < N; ++n) {
        // discrepancy d = sum_{i=0..L} c_i a_{n-i} = sum_i c_i arev[(N-1-n)+i]
        const int off = N - 1 - n;
        const int ow = off >> 6, ob = off & 63;
        const int cw = polyWords(L + 1);
        const int hs = (64 - ob) & 63;
        const uint64_t hmask = ob ? ~uint64_t(0) : uint64_t(0);
        const uint64_t* __restrict__ ar = arev.data() + ow;
        const uint64_t* __restrict__ cp = C.data();
        uint64_t acc = 0;
        for (int i = 0; i < cw; ++i) {
            uint64_t win = (ar[i] >> ob) | ((ar[i + 1] << hs) & hmask);
            acc ^= cp[i] & win;
        }
        if (__builtin_parityll(acc) == 0) {
            m++;
            continue;
        }
        const bool swap = (2 * L <= n);
        if (swap) T = C;
        // C ^= B * x^m
        {
            const int sw = m >> 6, sb = m & 63;
            const int ls = (64 - sb) & 63;
            const uint64_t lmask = sb ? ~uint64_t(0) : uint64_t(0);
            const uint64_t* __restrict__ bp = B.data() + 1 - sw;  // B[i - sw] at bp[i], B[i - sw - 1] at bp[i - 1]
            uint64_t* __restrict__ cq = C.data();
            const int iend = std::min(W, sw + polyWords(degB + 1) + 1);
            for (int i = sw; i < iend; ++i) {
                cq[i] ^= (bp[i] << sb) | ((bp[i - 1] >> ls) & lmask);
            }
        }
        if (swap) {
            degB = L;
            L = n + 1 - L;
            std::copy(T.begin(), T.end(), B.begin() + 1);
            m = 1;
        } else {
            m++;
        }
    }
}

// Arithmetic modulo a monic polynomial
struct PolyModulus {
    int deg = 0;
    int W = 0;
    std::vector<Poly> shifted;  // shifted[s] = m << s (W + 1 words)

    void init(const Poly& m, int d) {
        deg = d;
        W = polyWords(d + 1);
        shifted.assign(64, Poly(W + 1, 0));
        for (int s = 0; s < 64; ++s) {
            for (int i = 0; i < W; ++i) {
                uint64_t v = (i < static_cast<int>(m.size())) ? m[i] : 0;
                shifted[s][i] |= v << s;
                if (s) shifted[s][i + 1] |= v >> (64 - s);
            }
        }
    }

    // p mod m (in place); p may have any size
    void reduce(Poly& p) const {
        int top = static_cast<int>(p.size()) * 64 - 1;
        for (int k = top; k >= deg; --k) {
            if (!((p[k >> 6] >> (k & 63)) & 1)) continue;
            int sh = k - deg;
            int sw = sh >> 6, sb = sh & 63;
            const uint64_t* __restrict__ t = shifted[sb].data();
            uint64_t* __restrict__ q = p.data() + sw;
            const int n = W + (sb ? 1 : 0);
            for (int i = 0; i < n; ++i) q[i] ^= t[i];
        }
        p.resize(W);
    }

    Poly square(const Poly& a) const {
        Poly out(2 * W, 0);
        for (int i = 0; i < W; ++i) {
            uint64_t v = a[i];
            uint64_t lo = 0, hi = 0;
            for (int b = 0; b < 32; ++b) lo |= ((v >> b) & 1) << (2 * b);
            for (int b = 32; b < 64; ++b) hi |= ((v >> b) & 1) << (2 * (b - 32));
            out[2 * i] = lo;
            out[2 * i + 1] = hi;
        }
        reduce(out);
        return out;
    }

    // x^J mod m
    Poly powX(uint64_t J) const {
        Poly r(W, 0);
        r[0] = 1;
        int hb = 63;
        while (hb > 0 && !((J >> hb) & 1)) --hb;
        for (int b = hb; b >= 0; --b) {
            r = square(r);
            if ((J >> b) & 1) {
                Poly s(W + 1, 0);
                for (int i = 0; i < W; ++i) {
                    s[i] |= r[i] << 1;
                    s[i + 1] |= r[i] >> 63;
                }
                reduce(s);
                r = s;
            }
        }
        return r;
    }
};

// Host-side MT19937 state helpers
static void mtSeedState(uint32_t seed, std::vector<uint32_t>& s) {
    s.resize(MT_N);
    s[0] = seed;
    for (int i = 1; i < MT_N; ++i)
        s[i] = 1812433253u * (s[i - 1] ^ (s[i - 1] >> 30)) + static_cast<uint32_t>(i);
}

// Jump polynomials for the GPU stream generator
struct MtJumpTables {
    int polyWords32 = 0;          // uint32 words per polynomial
    int hornerSteps = 0;          // = deg(m)
    int levels = 0;               // number of jump polynomials
    std::vector<uint32_t> polys;  // polynomial l = x^(2^l * RNG_SEG_LEN) mod m
    std::vector<uint32_t> firstState;  // state producing output 0 of the stream (after the first twist)

    void build(int numLevels) {
        levels = std::max(1, std::min(numLevels, RNG_MAX_TREE_LEVELS));
        // Word sequence x_n of the generator seeded with MT_SEED
        std::vector<uint32_t> seed;
        mtSeedState(MT_SEED, seed);
        const int SEQ = 2 * (MT_N * 32) + 64;
        std::vector<uint32_t> xs(SEQ + MT_N);
        for (int i = 0; i < MT_N; ++i) xs[i] = seed[i];
        for (int n = 0; n + MT_N < static_cast<int>(xs.size()); ++n)
            xs[n + MT_N] = mtTwistWord(xs[n], xs[n + 1], xs[n + MT_M]);

        // Minimal polynomial of the bit sequence (bit 0 of x_n)
        Poly a(polyWords(SEQ), 0);
        for (int n = 0; n < SEQ; ++n)
            if (xs[n] & 1u) polySetBit(a, n);
        Poly C;
        int L;
        berlekampMassey(a, SEQ, C, L);

        // Characteristic polynomial m = x * reciprocal(C): the extra factor x makes
        // m annihilate the full 624x32-bit state (including the 31 unused bits).
        Poly m(polyWords(L + 2), 0);
        for (int k = 0; k <= L; ++k)
            if (polyBit(C, L - k)) polySetBit(m, k + 1);
        const int deg = L + 1;

        PolyModulus M;
        M.init(m, deg);
        hornerSteps = deg;
        polyWords32 = (deg + 31) / 32;

        // x^(RNG_SEG_LEN * 2^l) mod m for all tree levels (plus one zero word of
        // padding: the jump kernel fetches coefficient windows two words at a time)
        polys.assign(static_cast<size_t>(levels) * polyWords32 + 1, 0u);
        Poly g = M.powX(static_cast<uint64_t>(RNG_SEG_LEN));
        for (int l = 0; l < levels; ++l) {
            for (int k = 0; k < deg; ++k)
                if (polyBit(g, k)) polys[static_cast<size_t>(l) * polyWords32 + (k >> 5)] |= 1u << (k & 31);
            if (l + 1 < levels) g = M.square(g);
        }

        // State whose words are the (untempered) outputs 0..623 of the stream
        firstState.assign(xs.begin() + MT_N, xs.begin() + 2 * MT_N);
    }
};

// --- Device kernels -------------------------------------------------------------

// One block per jump task: states[dst] = g(A) * states[src], g = poly (Horner scheme,
// JUMP_GROUP steps at a time).  Logical word order: word 0 is the oldest state word.
__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(uint32_t* __restrict__ states, const int2* __restrict__ tasks,
             const uint32_t* __restrict__ poly, int steps) {
    __shared__ uint32_t xext[MT_N + JUMP_GROUP];
    __shared__ uint32_t h[MT_N];
    __shared__ uint32_t nw[JUMP_GROUP];

    const int tid = threadIdx.x;
    const int2 task = tasks[blockIdx.x];
    const uint32_t* x = states + static_cast<size_t>(task.x) * MT_N;

    for (int m = tid; m < MT_N; m += JUMP_THREADS) {
        xext[m] = x[m];
        h[m] = 0u;
    }
    __syncthreads();
    // Extend the source sequence: xext[624 + i] = x_{624+i}, i < JUMP_GROUP - 1
    for (int i = tid; i < JUMP_GROUP - 1; i += JUMP_THREADS)
        xext[MT_N + i] = mtTwistWord(xext[i], xext[i + 1], xext[i + MT_M]);
    __syncthreads();

    int idx = 0;               // h is a circular buffer: logical word m at h[(idx + m) % 624]
    int remaining = steps;     // exponents remaining-1 .. 0 still to process
    while (remaining > 0) {
        const int w = (remaining < JUMP_GROUP) ? remaining : JUMP_GROUP;
        const int khi = remaining - 1;

        // h = A^w h  (w <= 227: the w new words only depend on the current window)
        if (tid < w) {
            int i0 = idx + tid;
            uint32_t a0 = h[i0 % MT_N];
            uint32_t a1 = h[(i0 + 1) % MT_N];
            uint32_t a397 = h[(i0 + MT_M) % MT_N];
            nw[tid] = mtTwistWord(a0, a1, a397);
        }
        __syncthreads();
        if (tid < w) h[(idx + tid) % MT_N] = nw[tid];
        idx = (idx + w) % MT_N;
        __syncthreads();

        // h ^= sum_{k=klo..khi} c_k A^(k-klo) x, where word m of A^s x is xext[m + s].
        // The group's coefficients are fetched as a w-bit window (bit b <-> exponent klo + b).
        const int klo = khi - w + 1;
        const int wi = klo >> 5, sh = klo & 31;
        uint32_t acc0 = 0u, acc1 = 0u, acc2 = 0u;
        const uint32_t* x0 = xext + tid;
        const uint32_t* x1 = xext + tid + JUMP_THREADS;
        const uint32_t* x2 = xext + tid + 2 * JUMP_THREADS;
#pragma unroll
        for (int q = 0; q < JUMP_GROUP / 32; ++q) {
            const int bitBase = q * 32;
            if (bitBase >= w) break;
            uint32_t lo = poly[wi + q];
            uint32_t hi = poly[wi + q + 1];
            uint32_t cw = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
            if (w - bitBase < 32) cw &= (1u << (w - bitBase)) - 1u;
            while (cw) {
                const int b = bitBase + (__ffs(cw) - 1);
                cw &= cw - 1u;
                acc0 ^= x0[b];
                acc1 ^= x1[b];
                acc2 ^= x2[b];
            }
        }
        h[(idx + tid) % MT_N] ^= acc0;
        h[(idx + tid + JUMP_THREADS) % MT_N] ^= acc1;
        h[(idx + tid + 2 * JUMP_THREADS) % MT_N] ^= acc2;
        remaining -= w;
        __syncthreads();
    }

    uint32_t* out = states + static_cast<size_t>(task.y) * MT_N;
    for (int m = tid; m < MT_N; m += JUMP_THREADS) out[m] = h[(idx + m) % MT_N];
}

// One warp per segment: emit RNG_SEG_LEN canonical floats starting at the segment state.
constexpr int EXPAND_WARPS = 4;

__global__ void __launch_bounds__(EXPAND_WARPS * 32)
mtExpandKernel(const uint32_t* __restrict__ states, int numSegs, float* __restrict__ out) {
    __shared__ uint32_t buf[EXPAND_WARPS][2][MT_N];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int seg = blockIdx.x * EXPAND_WARPS + warp;
    if (seg >= numSegs) return;

    uint32_t* A = buf[warp][0];
    uint32_t* B = buf[warp][1];
    const uint32_t* s = states + static_cast<size_t>(seg) * MT_N;
    for (int m = lane; m < MT_N; m += 32) A[m] = s[m];
    __syncwarp();

    float* o = out + static_cast<size_t>(seg) * RNG_SEG_LEN;
    for (int b = 0; b < RNG_BLOCKS_PER_SEG; ++b) {
        for (int m = lane; m < MT_N; m += 32) o[b * MT_N + m] = mtToCanonical(mtTemper(A[m]));
        if (b + 1 == RNG_BLOCKS_PER_SEG) break;
        // Twist A -> B in dependency phases
        for (int i = lane; i < MT_N - MT_M; i += 32) B[i] = mtTwistWord(A[i], A[i + 1], A[i + MT_M]);
        __syncwarp();
        for (int i = (MT_N - MT_M) + lane; i < 2 * (MT_N - MT_M); i += 32)
            B[i] = mtTwistWord(A[i], A[i + 1], B[i - (MT_N - MT_M)]);
        __syncwarp();
        for (int i = 2 * (MT_N - MT_M) + lane; i < MT_N - 1; i += 32)
            B[i] = mtTwistWord(A[i], A[i + 1], B[i - (MT_N - MT_M)]);
        __syncwarp();
        if (lane == 0) B[MT_N - 1] = mtTwistWord(A[MT_N - 1], B[0], B[MT_M - 1]);
        __syncwarp();
        uint32_t* tmp = A; A = B; B = tmp;
    }
}

// ============================================================================
// Device geometry kernels
// ============================================================================

__device__ __forceinline__ Vec3 v3(const float4& f) { return Vec3(f.x, f.y, f.z); }

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ Vec3 randomPointInTriangle(const Vec3& a, const Vec3& b, const Vec3& c,
                                                      val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return a + ab * u + ac * v;
}

// Ray-Triangle Intersection (Moller-Trumbore algorithm)
__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Segment / axis-aligned box overlap test: same arithmetic as Octree::rayIntersectsBox
// in the original, evaluated branch-free (all six separating-axis tests are combined).
__device__ __forceinline__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                                 const Vec3& center, const Vec3& halfExtent) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    bool sep = (fabsf(c.x) > halfExtent.x + ad.x);
    sep |= (fabsf(c.y) > halfExtent.y + ad.y);
    sep |= (fabsf(c.z) > halfExtent.z + ad.z);
    sep |= (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON);
    sep |= (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON);
    sep |= (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON);
    return !sep;
}

struct DevOctree {
    const int4* childIdx;
    const float4* childBox;
    const int2* leaf;
    const int* slotIdx;
    const float4* slotVerts;
};

constexpr int OCTREE_STACK = 128;

// Warp-cooperative visibility test.  Every lane owns one ray (from -> to); the warp
// walks the union of the lanes' octree paths with a shared stack of (node, lane mask)
// entries.  A lane only tests a child box if it was active at the parent and only
// tests a leaf's triangles if its own segment passed the box test for that leaf, so
// each lane performs exactly the box and triangle tests of the sequential traversal.
// Returns true for lanes whose ray is blocked by some other triangle.
__device__ bool warpRayBlocked(const Vec3& from, const Vec3& to, bool laneActive,
                               const DevOctree& oct, int srcTriIdx, int dstTriIdx,
                               int2* __restrict__ stack) {
    const unsigned FULL = 0xffffffffu;
    const unsigned laneBit = 1u << (threadIdx.x & 31);

    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    const bool degenerate = rayLen < EPSILON;        // original: blocked
    Vec3 dirNorm = dir / (degenerate ? ONE : rayLen);

    unsigned blocked = __ballot_sync(FULL, laneActive && degenerate);
    unsigned alive = __ballot_sync(FULL, laneActive) & ~blocked;
    if (alive == 0u) return (blocked & laneBit) != 0u;

    int sp = 0;
    stack[sp++] = make_int2(0, static_cast<int>(alive));  // root (never box-tested, as in the original)

    while (sp > 0) {
        const int2 top = stack[--sp];
        const int node = top.x;
        unsigned mask = static_cast<unsigned>(top.y) & ~blocked;
        if (mask == 0u) continue;
        const bool inMask = (mask & laneBit) != 0u;
        const int2 lf = oct.leaf[node];
        if (lf.y > 0) {
            for (int k = 0; k < lf.y; ++k) {
                const int slot = lf.x + k;
                const int idx = oct.slotIdx[slot];
                const float4 va = oct.slotVerts[slot * 3];
                const float4 vb = oct.slotVerts[slot * 3 + 1];
                const float4 vc = oct.slotVerts[slot * 3 + 2];
                bool hit = false;
                if (inMask && idx != srcTriIdx && idx != dstTriIdx) {
                    val_t dist = rayTriangleIntersect(from, dirNorm, v3(va), v3(vb), v3(vc));
                    hit = (dist > EPSILON && dist < rayLen - EPSILON);
                }
                const unsigned h = __ballot_sync(FULL, hit);
                if (h) {
                    blocked |= h;
                    mask &= ~h;
                    if (mask == 0u) break;
                }
            }
        } else {
            const int4 ci0 = oct.childIdx[node * 2];
            const int4 ci1 = oct.childIdx[node * 2 + 1];
            const float4* box = oct.childBox + static_cast<size_t>(node) * 16;
#pragma unroll
            for (int g = 1; g >= 0; --g) {
                float4 cen[4], hf[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    cen[q] = box[(g * 4 + q) * 2];
                    hf[q] = box[(g * 4 + q) * 2 + 1];
                }
                const int4 ci = g ? ci1 : ci0;
#pragma unroll
                for (int q = 3; q >= 0; --q) {
                    const int ch = (q == 0) ? ci.x : (q == 1) ? ci.y : (q == 2) ? ci.z : ci.w;
                    if (ch < 0) continue;
                    const bool pass = inMask && rayIntersectsBox(from, to, v3(cen[q]), v3(hf[q]));
                    const unsigned m = __ballot_sync(FULL, pass);
                    if (m) stack[sp++] = make_int2(ch, static_cast<int>(m));
                }
            }
        }
    }
    return (blocked & laneBit) != 0u;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // std::max(ZERO, c)
}

__device__ __forceinline__ bool pairCulled(const Vec3& ni, const Vec3& nj) {
    return ni.dot(nj) > 0.99f;
}

// Number of random values consumed per row of the pair matrix
constexpr int ROW_THREADS = 256;

__global__ void __launch_bounds__(ROW_THREADS)
rowRandCountKernel(const DevTri* __restrict__ tris, int N, unsigned int* __restrict__ rowCount) {
    __shared__ int warpSum[ROW_THREADS / 32];
    const int i = blockIdx.x;
    const Vec3 ni = v3(tris[i].n);
    int cnt = 0;
    for (int j = threadIdx.x; j < N; j += ROW_THREADS) {
        if (j != i && !pairCulled(ni, v3(tris[j].n))) cnt++;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < ROW_THREADS / 32; ++w) total += warpSum[w];
        rowCount[i] = static_cast<unsigned int>(total) * RNG_RANDS_PER_PAIR;
    }
}

// Time delays tau_ij, stored transposed (tauT[j * N + i]) as bytes.
// dist <= 2 * radius = 20, so tau = ceil(dist / WAVE_SPEED) <= 40 fits in a byte.
__global__ void tauKernel(const DevTri* __restrict__ tris, int N, unsigned char* __restrict__ tauT) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= N) return;
    int tau = 0;
    if (i != j) {
        const DevTri ti = tris[i];
        const DevTri tj = tris[j];
        Vec3 ci = (v3(ti.a) + v3(ti.b) + v3(ti.c)) / 3.0f;
        Vec3 cj = (v3(tj.a) + v3(tj.b) + v3(tj.c)) / 3.0f;
        val_t dist = (ci - cj).norm();
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[static_cast<size_t>(j) * N + i] = static_cast<unsigned char>(tau);
}

// Form factors.  One block per row i; pairs (i, j) are processed 16 lanes per pair
// (one lane per ray).  Only pairs whose random-stream offset lies in
// [chunkStart, chunkStart + RNG_CHUNK_LEN) are processed by this launch.
constexpr int FF_THREADS = 256;
constexpr int FF_PAIRS_PER_PASS = FF_THREADS / NUM_RAYS;

__global__ void __launch_bounds__(FF_THREADS, 3)
formFactorKernel(const DevTri* __restrict__ tris, int N, DevOctree oct,
                 const unsigned long long* __restrict__ rowOff, int rowBase,
                 unsigned long long chunkStart, const float* __restrict__ rng,
                 float* __restrict__ kijT) {
    __shared__ int sFlag[FF_THREADS];
    __shared__ int sOff[FF_THREADS];
    __shared__ int sWarp[FF_THREADS / 32];
    __shared__ int2 sStack[FF_THREADS / 32][OCTREE_STACK];

    const int i = rowBase + blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const DevTri triI = tris[i];
    const Vec3 nI = v3(triI.n);
    const unsigned long long rowStart = rowOff[i];
    const unsigned long long chunkEnd = chunkStart + static_cast<unsigned long long>(RNG_CHUNK_LEN);

    int carry = 0;  // non-culled pairs before the current tile
    for (int tile = 0; tile < N; tile += FF_THREADS) {
        // --- flags + exclusive scan over the tile -------------------------------
        const int j0 = tile + tid;
        int flag = 0;
        if (j0 < N && j0 != i) flag = pairCulled(nI, v3(tris[j0].n)) ? 0 : 1;
        int incl = flag;
        for (int o = 1; o < 32; o <<= 1) {
            int v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 31) sWarp[warp] = incl;
        __syncthreads();
        int warpPrefix = 0;
        for (int w = 0; w < warp; ++w) warpPrefix += sWarp[w];
        const int tileTotal = sWarp[0] + sWarp[1] + sWarp[2] + sWarp[3] +
                              sWarp[4] + sWarp[5] + sWarp[6] + sWarp[7];
        sFlag[tid] = flag;
        sOff[tid] = carry + warpPrefix + incl - flag;
        __syncthreads();

        // Skip the tile if none of its pairs belongs to this chunk
        const unsigned long long tileLo = rowStart + static_cast<unsigned long long>(carry) * RNG_RANDS_PER_PAIR;
        const unsigned long long tileHi = rowStart + static_cast<unsigned long long>(carry + tileTotal) * RNG_RANDS_PER_PAIR;
        carry += tileTotal;
        if (tileTotal == 0 || tileHi <= chunkStart || tileLo >= chunkEnd) {
            __syncthreads();
            continue;
        }

        // --- process the tile, FF_PAIRS_PER_PASS pairs per pass ---------------
        for (int pass = 0; pass < FF_THREADS / FF_PAIRS_PER_PASS; ++pass) {
            const int q = pass * FF_PAIRS_PER_PASS + (tid / NUM_RAYS);
            const int r = tid % NUM_RAYS;
            const int j = tile + q;
            bool active = (j < N) && sFlag[q];
            unsigned long long off = 0;
            if (active) {
                off = rowStart + static_cast<unsigned long long>(sOff[q]) * RNG_RANDS_PER_PAIR;
                active = (off >= chunkStart) && (off < chunkEnd);
            }
            val_t contrib = ZERO;
            Vec3 pI, pJ, nJ;
            if (active) {
                const float* rp = rng + (off - chunkStart) + r * 4;
                const DevTri triJ = tris[j];
                nJ = v3(triJ.n);
                pI = randomPointInTriangle(v3(triI.a), v3(triI.b), v3(triI.c), rp[0], rp[1]);
                pJ = randomPointInTriangle(v3(triJ.a), v3(triJ.b), v3(triJ.c), rp[2], rp[3]);
            }
            const bool blocked = warpRayBlocked(pI, pJ, active, oct, i, active ? j : -1, sStack[warp]);
            if (active) {
                if (!blocked) {
                    Vec3 v = pJ - pI;
                    val_t distSqr = v.squaredNorm();
                    if (!(distSqr < EPSILON)) {
                        val_t cosPhiI = cosPhi(v, nI);
                        val_t cosPhiJ = cosPhi(-v, nJ);
                        if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                            contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                        }
                    }
                }
            }
            // Sum the 16 ray contributions in ray order (as the sequential loop does)
            const int groupBase = lane & ~(NUM_RAYS - 1);
            val_t kij = ZERO;
            for (int rr = 0; rr < NUM_RAYS; ++rr) {
                val_t c = __shfl_sync(0xffffffffu, contrib, groupBase + rr);
                kij += c;
            }
            if (active && r == 0) {
                kijT[static_cast<size_t>(j) * N + i] = kij * INV_NUM_RAYS;
            }
        }
        __syncthreads();
    }
}

// Wave propagation: one thread per receiving triangle i for timestep t.
// Radiosity arrays are stored transposed on the device: rad[n * T + t].
// The per-thread sum over emitters j is sequential and in order (as in the original);
// to hide memory latency despite that serial chain, a tile of SIM_TILE emitters is
// first staged through shared memory (coalesced loads) and all the tile's radiosity
// gathers are issued into registers before the accumulation.
constexpr int SIM_THREADS = 128;
constexpr int SIM_TILE = 64;

__global__ void __launch_bounds__(SIM_THREADS)
simStepKernel(int N, int T, int t,
              const float* __restrict__ kijT, const unsigned char* __restrict__ tauT,
              const float* __restrict__ areas, const float* __restrict__ rho,
              const float* __restrict__ radET, float* __restrict__ radBT) {
    __shared__ float sKij[SIM_TILE][SIM_THREADS];
    __shared__ unsigned char sTau[SIM_TILE][SIM_THREADS];
    __shared__ float sArea[SIM_TILE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * SIM_THREADS + tid;
    const bool valid = i < N;

    val_t sumB = ZERO;
    for (int j0 = 0; j0 < N; j0 += SIM_TILE) {
        const int nj = min(SIM_TILE, N - j0);
        __syncthreads();  // previous tile fully consumed
        if (valid) {
#pragma unroll
            for (int jj = 0; jj < SIM_TILE; ++jj) {
                const size_t e = static_cast<size_t>(j0 + min(jj, nj - 1)) * N + i;
                sKij[jj][tid] = kijT[e];
                sTau[jj][tid] = tauT[e];
            }
        }
        if (tid < SIM_TILE) sArea[tid] = areas[j0 + min(tid, nj - 1)];
        __syncthreads();
        if (valid) {
            val_t r[SIM_TILE];
#pragma unroll
            for (int jj = 0; jj < SIM_TILE; ++jj) {
                const int st = t - sTau[jj][tid];
                r[jj] = radBT[static_cast<size_t>(j0 + min(jj, nj - 1)) * T + (st > 0 ? st : 0)];
            }
            // Branch-free form of the original loop: skipped terms contribute an exact +0,
            // which leaves the sequential, in-order accumulation bit-identical.
#pragma unroll
            for (int jj = 0; jj < SIM_TILE; ++jj) {
                const int j = j0 + jj;
                const int st = t - sTau[jj][tid];
                const val_t kij = sKij[jj][tid];
                const val_t ka = kij * sArea[jj];
                // std::min(kij * area, ONE) * radJ, accumulated with separate multiply and add
                // (the original is compiled without contraction here).
                const val_t term = __fmul_rn((ONE < ka) ? ONE : ka, r[jj]);
                const bool use = (jj < nj) && (j != i) && (st >= 0) && !(kij <= ZERO) && !(r[jj] <= ZERO);
                sumB = __fadd_rn(sumB, use ? term : ZERO);
            }
        }
    }
    if (valid) {
        // rho * sumB + radE (contracted into one fused multiply-add, as in the original)
        radBT[static_cast<size_t>(i) * T + t] = __fmaf_rn(rho[i], sumB, radET[static_cast<size_t>(i) * T + t]);
    }
}

// Distance via cross-correlation: one block per triangle i, threads over lags t.
constexpr int DIST_THREADS = 256;

__global__ void __launch_bounds__(DIST_THREADS)
distanceKernel(int T, int sourceIndex, const float* __restrict__ radBT, float* __restrict__ distances) {
    __shared__ val_t sVal[DIST_THREADS];
    __shared__ int sIdx[DIST_THREADS];
    const int i = blockIdx.x;
    const float* rowB = radBT + static_cast<size_t>(i) * T;
    const float* rowS = radBT + static_cast<size_t>(sourceIndex) * T;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < T; t += DIST_THREADS) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = rowB[tt];
            val_t pS = rowS[tt - t];
            sum = __fadd_rn(sum, __fmul_rn(pS, pB));  // unfused, as in the original
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    sVal[threadIdx.x] = maxCorr;
    sIdx[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = DIST_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t v2 = sVal[threadIdx.x + s];
            int t2 = sIdx[threadIdx.x + s];
            val_t v1 = sVal[threadIdx.x];
            int t1 = sIdx[threadIdx.x];
            // strictly larger wins; on ties the earliest lag wins (first strict maximum)
            if (v2 > v1 || (v2 == v1 && t2 < t1)) {
                sVal[threadIdx.x] = v2;
                sIdx[threadIdx.x] = t2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(sVal[0] > ZERO ? sIdx[0] : 0);
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major; filled on download)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix; filled on download)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host build)

    size_t sourceIndex;

    // Device data
    DevTri* d_tris = nullptr;
    int4* d_childIdx = nullptr;
    float4* d_childBox = nullptr;
    int2* d_leaf = nullptr;
    int* d_slotIdx = nullptr;
    float4* d_slotVerts = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_kijT = nullptr;            // kij transposed: kijT[j * N + i]
    unsigned char* d_tauT = nullptr;    // tau transposed: tauT[j * N + i]
    float* d_radET = nullptr;           // radE transposed: [n * T + t]
    float* d_radBT = nullptr;           // radB transposed: [n * T + t]
    float* d_distances = nullptr;

    MtJumpTables jump;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    DevOctree devOctree() const {
        return DevOctree{d_childIdx, d_childBox, d_leaf, d_slotIdx, d_slotVerts};
    }
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
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
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // --- Device setup ----------------------------------------------------------
    const size_t N = state.numTriangles;
    const size_t T = timesteps;
    CUDA_CHECK(cudaFree(0));
    // Force loading of all kernels now (CUDA lazy module loading would otherwise
    // defer it to the first launch inside the timed phases).
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, mtJumpKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, mtExpandKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, rowRandCountKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, tauKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, formFactorKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, simStepKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, distanceKernel));
    }

    std::vector<DevTri> htris(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        htris[i].a = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        htris[i].b = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        htris[i].c = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        htris[i].n = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
    }
    Octree::Flat flat;
    state.octree.flatten(flat);

    CUDA_CHECK(cudaMalloc(&state.d_tris, N * sizeof(DevTri)));
    CUDA_CHECK(cudaMemcpy(state.d_tris, htris.data(), N * sizeof(DevTri), cudaMemcpyHostToDevice));
    auto upload = [](auto** dptr, const auto& vec) {
        using T = std::remove_pointer_t<std::remove_pointer_t<decltype(dptr)>>;
        const size_t bytes = std::max<size_t>(1, vec.size()) * sizeof(T);
        CUDA_CHECK(cudaMalloc(dptr, bytes));
        if (!vec.empty()) CUDA_CHECK(cudaMemcpy(*dptr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice));
    };
    upload(&state.d_childIdx, flat.childIdx);
    upload(&state.d_childBox, flat.childBox);
    upload(&state.d_leaf, flat.leaf);
    upload(&state.d_slotIdx, flat.slotIdx);
    upload(&state.d_slotVerts, flat.slotVerts);

    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_kijT, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_kijT, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tauT, N * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(state.d_tauT, 0, N * N * sizeof(unsigned char)));

    std::vector<float> radET(std::max<size_t>(1, T * N), ZERO);
    for (size_t t = 0; t < T; ++t)
        for (size_t n = 0; n < N; ++n) radET[n * T + t] = state.radE[state.idxTN(t, n)];
    CUDA_CHECK(cudaMalloc(&state.d_radET, radET.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_radET, radET.data(), radET.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_radBT, radET.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radBT, 0, radET.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Fill segment states slots [0, P) with stride RNG_SEG_LEN starting from slot 0,
// using a doubling tree of jumps (level l jumps by 2^l * RNG_SEG_LEN).
static void fillSegmentStates(uint32_t* d_states, int P, int2* d_tasks, std::vector<int2>& tasks,
                              const uint32_t* d_polys, const MtJumpTables& jump) {
    int levels = 0;
    while ((1 << levels) < P) levels++;
    for (int l = levels - 1; l >= 0; --l) {
        tasks.clear();
        const int step = 1 << l;
        for (int q = 0; q + step < P; q += 2 * step) tasks.push_back(make_int2(q, q + step));
        if (tasks.empty()) continue;
        CUDA_CHECK(cudaMemcpyAsync(d_tasks, tasks.data(), tasks.size() * sizeof(int2), cudaMemcpyHostToDevice));
        mtJumpKernel<<<static_cast<unsigned>(tasks.size()), JUMP_THREADS>>>(
            d_states, d_tasks, d_polys + static_cast<size_t>(l) * jump.polyWords32, jump.hornerSteps);
        CUDA_CHECK(cudaGetLastError());
    }
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);

    // Random stream layout: number of values consumed by each row of pairs
    unsigned int* d_rowCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowCount, N * sizeof(unsigned int)));
    rowRandCountKernel<<<N, ROW_THREADS>>>(state.d_tris, N, d_rowCount);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int> rowCount(N);
    CUDA_CHECK(cudaMemcpy(rowCount.data(), d_rowCount, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rowCount));

    std::vector<unsigned long long> rowOff(N + 1, 0ull);
    for (int i = 0; i < N; ++i) rowOff[i + 1] = rowOff[i] + rowCount[i];
    const unsigned long long totalRands = rowOff[N];

    unsigned long long* d_rowOff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowOff, (N + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_rowOff, rowOff.data(), (N + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Stream chunking: chunk c covers pair offsets [c * RNG_CHUNK_LEN, (c+1) * RNG_CHUNK_LEN);
    // the values of a pair extend at most RNG_RANDS_PER_PAIR beyond its offset, so each
    // chunk also expands the first segment of the next one.
    const unsigned long long numChunks = (totalRands + RNG_CHUNK_LEN - 1) / RNG_CHUNK_LEN;
    const unsigned long long totalSegs = (totalRands + RNG_RANDS_PER_PAIR + RNG_SEG_LEN - 1) / RNG_SEG_LEN;
    int treeLevels = 0;
    while ((1ull << treeLevels) < totalSegs) treeLevels++;

    // MT19937 jump tables (minimal polynomial + jump polynomials)
    state.jump.build(treeLevels);
    const MtJumpTables& jump = state.jump;
    uint32_t* d_polys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_polys, jump.polys.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_polys, jump.polys.data(), jump.polys.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t* d_states = nullptr;
    float* d_rng = nullptr;
    int2* d_tasks = nullptr;
    if (numChunks > 0) {
        const size_t bufSegs = static_cast<size_t>(std::min<unsigned long long>(RNG_MAX_SEGS + 1, totalSegs));
        CUDA_CHECK(cudaMalloc(&d_states, static_cast<size_t>(totalSegs) * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_rng, bufSegs * RNG_SEG_LEN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tasks, static_cast<size_t>(totalSegs) * sizeof(int2)));
        // Slot 0: the state whose words are stream outputs 0..623; the remaining
        // segment states (stride RNG_SEG_LEN) follow from a doubling tree of jumps.
        CUDA_CHECK(cudaMemcpy(d_states, jump.firstState.data(), MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        std::vector<int2> tasks;
        fillSegmentStates(d_states, static_cast<int>(totalSegs), d_tasks, tasks, d_polys, jump);
    }

    int nextProgressRow = 0;
    for (unsigned long long c = 0; c < numChunks; ++c) {
        const unsigned long long chunkStart = c * RNG_CHUNK_LEN;
        const unsigned long long chunkEnd = chunkStart + RNG_CHUNK_LEN;
        const unsigned long long segStart = c * RNG_MAX_SEGS;
        const int P = static_cast<int>(std::min<unsigned long long>(RNG_MAX_SEGS + 1, totalSegs - segStart));

        mtExpandKernel<<<(P + EXPAND_WARPS - 1) / EXPAND_WARPS, EXPAND_WARPS * 32>>>(
            d_states + segStart * MT_N, P, d_rng);
        CUDA_CHECK(cudaGetLastError());

        // Rows whose pair offsets intersect this chunk
        int ia = static_cast<int>(std::upper_bound(rowOff.begin() + 1, rowOff.end(), chunkStart) - rowOff.begin()) - 1;
        int ib = static_cast<int>(std::lower_bound(rowOff.begin(), rowOff.begin() + N, chunkEnd) - rowOff.begin());
        ia = std::max(0, std::min(ia, N));
        ib = std::max(ia, std::min(ib, N));
        if (ib > ia) {
            formFactorKernel<<<ib - ia, FF_THREADS>>>(state.d_tris, N, state.devOctree(), d_rowOff, ia,
                                                     chunkStart, d_rng, state.d_kijT);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Progress report for rows completed by this chunk
        while (nextProgressRow < N && rowOff[nextProgressRow + 1] <= chunkEnd) {
            const int i = nextProgressRow++;
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %d/%d triangles\n", i + 1, N);
            }
        }
    }
    while (nextProgressRow < N) {
        const int i = nextProgressRow++;
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            printf("  Progress: %d/%d triangles\n", i + 1, N);
        }
    }

    if (d_states) CUDA_CHECK(cudaFree(d_states));
    if (d_rng) CUDA_CHECK(cudaFree(d_rng));
    if (d_tasks) CUDA_CHECK(cudaFree(d_tasks));
    CUDA_CHECK(cudaFree(d_polys));
    CUDA_CHECK(cudaFree(d_rowOff));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);
    dim3 grid((N + 255) / 256, N);
    tauKernel<<<grid, 256>>>(state.d_tris, N, state.d_tauT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    for (int t = 0; t < T; ++t) {
        simStepKernel<<<(N + SIM_THREADS - 1) / SIM_THREADS, SIM_THREADS>>>(
            N, T, t, state.d_kijT, state.d_tauT, state.d_areas, state.d_rho, state.d_radET, state.d_radBT);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    distanceKernel<<<N, DIST_THREADS>>>(T, static_cast<int>(state.sourceIndex), state.d_radBT, state.d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, N * sizeof(float), cudaMemcpyDeviceToHost));
}

// Copy the full simulation results back to the host (needed for validation only)
void downloadResults(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    std::vector<float> kijT(N * N);
    CUDA_CHECK(cudaMemcpy(kijT.data(), state.d_kijT, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    state.kij.resize(N * N);
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i) state.kij[state.idx2d(i, j)] = kijT[j * N + i];

    std::vector<float> radBT(std::max<size_t>(1, T * N));
    CUDA_CHECK(cudaMemcpy(radBT.data(), state.d_radBT, radBT.size() * sizeof(float), cudaMemcpyDeviceToHost));
    state.radB.resize(T * N);
    for (size_t t = 0; t < T; ++t)
        for (size_t n = 0; n < N; ++n) state.radB[state.idxTN(t, n)] = radBT[n * T + t];
}

void releaseDevice(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_childIdx);
    cudaFree(state.d_childBox);
    cudaFree(state.d_leaf);
    cudaFree(state.d_slotIdx);
    cudaFree(state.d_slotVerts);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kijT);
    cudaFree(state.d_tauT);
    cudaFree(state.d_radET);
    cudaFree(state.d_radBT);
    cudaFree(state.d_distances);
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
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
    // This program uses a single GPU.  Unless the user selected devices explicitly,
    // restrict device enumeration to the first GPU: the CUDA driver initializes every
    // visible GPU on startup, which costs about a second per device when the
    // persistence daemon is not running.
    setenv("CUDA_VISIBLE_DEVICES", "0", 0);

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
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

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
    int rc = 0;
    if (validate) {
        downloadResults(state);
        if (!validateResults(state)) {
            rc = 1;
        }
    }

    releaseDevice(state);
    return rc;
}

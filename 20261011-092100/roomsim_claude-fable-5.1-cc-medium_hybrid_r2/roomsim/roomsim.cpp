/**
 * Room Response Simulation Benchmark -- hybrid MPI + OpenMP + CUDA version
 *
 * This is a parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy
 * ------------------------
 *  * MPI    : the triangle rows [0, N) are split into contiguous blocks, one per
 *             rank.  Every rank owns the form factors, time delays, simulation
 *             rows and distances of its block.  The radiosity row produced in
 *             every timestep is exchanged with an all-gather.
 *  * CUDA   : every rank drives one GPU.  Form factors (ray-traced visibility),
 *             time delays, wave propagation and the cross-correlation run on
 *             the device.
 *  * OpenMP : host-side preprocessing (culling mask / pair prefix counts, the
 *             mt19937 jump-ahead polynomial arithmetic, random state snapshots,
 *             reductions) is multi-threaded.
 *
 * Semantics: the program reproduces the original sequential algorithm exactly,
 * including the single mt19937(42) random stream that the original consumes in
 * row-major (i, j) order.  Each GPU warp that computes form factors owns a
 * contiguous range of (i, j) pairs and an mt19937 state that has been jumped
 * ahead (GF(2) polynomial jump-ahead) to exactly the position the original
 * program would have reached for the first pair of that range.  All floating
 * point accumulations are performed in the same order as in the original.
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
constexpr val_t VAL_MAX = 3.402823466e+38F;   // == std::numeric_limits<float>::max()

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ============================================================================
// Vector and Triangle Types (usable on host and device)
// ============================================================================

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

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct __align__(16) Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
// Octree for Spatial Acceleration (built on the host, flattened for the GPU)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

// Flattened octree node used by the device traversal (240 bytes, 16-byte aligned).
// The bounding boxes of the eight children are stored in the parent so that a
// visit of an internal node is a single batch of independent 128-bit loads.
struct __align__(16) FlatNode {
    val_t childBox[8][4];   // child i: center x, y, z and half extent x
    val_t childHalfYZ[8][2];// child i: half extent y, z
    int child[8];           // child node index or -1
    int triStart;           // offset into triangle index list (leaves only)
    int triCount;           // number of triangles (0 for internal nodes)
    int pad0, pad1;
};
static_assert(sizeof(FlatNode) == 240, "FlatNode layout");

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

    // Flatten the tree into arrays for the device traversal (depth-first order)
    void flatten(std::vector<FlatNode>& nodes, std::vector<int>& triList) const {
        nodes.clear();
        triList.clear();
        flattenNode(nodes, triList);
    }

private:
    int flattenNode(std::vector<FlatNode>& nodes, std::vector<int>& triList) const {
        int myIdx = static_cast<int>(nodes.size());
        nodes.emplace_back();
        FlatNode n;
        memset(&n, 0, sizeof(n));
        n.triStart = static_cast<int>(triList.size());
        n.triCount = static_cast<int>(triangleIndices.size());
        for (size_t idx : triangleIndices) triList.push_back(static_cast<int>(idx));
        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                const Octree& c = *children[i];
                n.childBox[i][0] = c.center.x; n.childBox[i][1] = c.center.y; n.childBox[i][2] = c.center.z;
                n.childBox[i][3] = c.halfExtent.x;
                n.childHalfYZ[i][0] = c.halfExtent.y; n.childHalfYZ[i][1] = c.halfExtent.z;
                n.child[i] = children[i]->flattenNode(nodes, triList);
            } else {
                n.child[i] = -1;
            }
        }
        nodes[myIdx] = n;
        return myIdx;
    }

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
// Random Number Generation: mt19937 (bit-exact std::mt19937 replacement)
// ============================================================================
//
// The original program draws all random numbers from a single std::mt19937
// seeded with 42 through std::uniform_real_distribution<float>(0, 1).  With
// libstdc++ this yields  float(word) / 2^32  (clamped below 1.0f) for every
// 32-bit output word.  Every non-culled (i, j) pair consumes exactly
// 4 * NUM_RAYS words, in row-major pair order.  The GPU form factor kernel
// reproduces this stream exactly: each warp owns a contiguous pair range and a
// raw mt19937 state window that has been positioned at the right stream offset.
//
// Raw sequence: x_0..x_623 is the seeded state, x_{k+624} = x_{k+397} ^ twist(x_k, x_{k+1}).
// Output number o (0-based) equals temper(x_{o+624}).  A "window" at raw offset
// W is the 624 words x_W..x_{W+623}; the standard in-place block regeneration
// applied to a window yields the next window x_{W+624}..x_{W+1247}.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_SEED = 42;
constexpr int64_t WORDS_PER_PAIR = 4 * NUM_RAYS;

__host__ __device__ inline uint32_t mtTwist(uint32_t a, uint32_t b) {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

__host__ __device__ inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0,1) applied to one mt19937 word (libstdc++)
__host__ __device__ inline val_t mtWordToFloat(uint32_t w) {
    val_t f = static_cast<val_t>(w) / 4294967296.0f;
    if (f >= 1.0f) f = 0.99999994f;  // std::nextafter(1.0f, 0.0f)
    return f;
}

static void mtSeed(uint32_t* st, uint32_t seed) {
    st[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        st[i] = 1812433253u * (st[i - 1] ^ (st[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

// Next window from previous window (separate buffers).  Written as three
// dependency-free phases so that the compiler can vectorize.
static void mtNextBlock(const uint32_t* __restrict__ old, uint32_t* __restrict__ nw) {
    #pragma omp simd
    for (int k = 0; k < MT_N - MT_M; ++k) {
        nw[k] = old[k + MT_M] ^ mtTwist(old[k], old[k + 1]);
    }
    #pragma omp simd safelen(MT_N - MT_M)
    for (int k = MT_N - MT_M; k < MT_N - 1; ++k) {
        nw[k] = nw[k - (MT_N - MT_M)] ^ mtTwist(old[k], old[k + 1]);
    }
    nw[MT_N - 1] = nw[MT_M - 1] ^ mtTwist(old[MT_N - 1], nw[0]);
}

// ----------------------------------------------------------------------------
// GF(2) polynomial arithmetic for mt19937 jump-ahead
// ----------------------------------------------------------------------------

using Bits = std::vector<uint64_t>;

static inline bool getBit(const Bits& w, size_t i) { return (w[i >> 6] >> (i & 63)) & 1u; }
static inline void setBit(Bits& w, size_t i) { w[i >> 6] |= 1ull << (i & 63); }
static int polyDegree(const Bits& w) {
    for (int i = static_cast<int>(w.size()) - 1; i >= 0; --i) {
        if (w[i]) return i * 64 + 63 - __builtin_clzll(w[i]);
    }
    return -1;
}

// dst ^= src << s
static void xorShl(Bits& dst, const Bits& src, size_t s) {
    size_t ws = s >> 6, bs = s & 63;
    for (size_t i = 0; i < src.size(); ++i) {
        uint64_t v = src[i];
        if (!v) continue;
        if (i + ws < dst.size()) dst[i + ws] ^= v << bs;
        if (bs && i + ws + 1 < dst.size()) dst[i + ws + 1] ^= v >> (64 - bs);
    }
}

// parity( popcount( a & (b >> s) ) ) over the first nwords words of a
static int dotShifted(const Bits& a, const Bits& b, size_t s, size_t nwords) {
    size_t ws = s >> 6, bs = s & 63;
    uint64_t acc = 0;
    for (size_t i = 0; i < nwords; ++i) {
        uint64_t lo = (i + ws < b.size()) ? b[i + ws] : 0;
        uint64_t hi = (i + ws + 1 < b.size()) ? b[i + ws + 1] : 0;
        uint64_t v = bs ? ((lo >> bs) | (hi << (64 - bs))) : lo;
        acc ^= a[i] & v;
    }
    return __builtin_parityll(acc);
}

// Berlekamp-Massey over GF(2).  Returns the minimal polynomial m(x) = sum a_j x^j
// of the bit sequence s, in the "forward" convention: sum_j a_j s_{k+j} = 0 for all k.
static Bits minimalPolynomial(const std::vector<uint8_t>& s) {
    size_t n = s.size();
    size_t W = n / 64 + 2;
    Bits R(W, 0);  // reversed sequence: bit j of R = s[n-1-j]
    for (size_t i = 0; i < n; ++i) if (s[n - 1 - i]) setBit(R, i);
    Bits C(W, 0), B(W, 0);
    C[0] = 1; B[0] = 1;
    size_t L = 0, m = 1;
    for (size_t k = 0; k < n; ++k) {
        int d = dotShifted(C, R, n - 1 - k, L / 64 + 1);
        if (!d) { ++m; continue; }
        if (2 * L <= k) {
            Bits T = C;
            xorShl(C, B, m);
            L = k + 1 - L;
            B = T;
            m = 1;
        } else {
            xorShl(C, B, m);
            ++m;
        }
    }
    Bits M(L / 64 + 1, 0);
    for (size_t i = 0; i <= L; ++i) if (getBit(C, i)) setBit(M, L - i);
    return M;
}

// Arithmetic modulo a fixed polynomial m(x) of degree D
struct PolyMod {
    Bits m;
    int D = 0;
    size_t nres = 0;             // words of a residue (deg < D)
    std::vector<Bits> shifted;   // m << s for s in [0, 64)

    void init(const Bits& mm) {
        m = mm;
        D = polyDegree(m);
        nres = static_cast<size_t>(D) / 64 + 1;
        shifted.resize(64);
        for (int s = 0; s < 64; ++s) {
            shifted[s].assign(m.size() + 1, 0);
            xorShl(shifted[s], m, static_cast<size_t>(s));
        }
    }

    size_t workWords() const { return 2 * nres + shifted[0].size() + 2; }

    // Reduce a (any degree, a.size() >= workWords()) in place to degree < D
    void reduce(Bits& a) const {
        const size_t dw = static_cast<size_t>(D) >> 6;
        for (long wi = static_cast<long>(a.size()) - 1; wi >= static_cast<long>(dw); --wi) {
            while (a[wi]) {
                int bit = 63 - __builtin_clzll(a[wi]);
                long b = wi * 64 + bit;
                if (b < D) return;
                size_t s = static_cast<size_t>(b - D);
                size_t ws = s >> 6, bs = s & 63;
                const Bits& ms = shifted[bs];
                uint64_t* dst = &a[ws];
                const uint64_t* src = ms.data();
                const size_t n = ms.size();
                for (size_t i = 0; i < n; ++i) dst[i] ^= src[i];
            }
        }
    }

    Bits square(const Bits& a) const {
        Bits r(workWords(), 0);
        int da = polyDegree(a);
        for (int i = 0; i <= da; ++i) if (getBit(a, i)) setBit(r, 2 * static_cast<size_t>(i));
        reduce(r);
        r.resize(nres);
        return r;
    }

    // a * x mod m
    Bits mulX(const Bits& a) const {
        Bits r(workWords(), 0);
        xorShl(r, a, 1);
        reduce(r);
        r.resize(nres);
        return r;
    }

    // x^e mod m
    Bits powX(uint64_t e) const {
        Bits r(nres, 0);
        r[0] = 1;
        if (e == 0) return r;
        int top = 63 - __builtin_clzll(e);
        for (int bit = top; bit >= 0; --bit) {
            r = square(r);
            if ((e >> bit) & 1ull) r = mulX(r);
        }
        return r;
    }
};

// Minimal polynomial of the raw mt19937 sequence (times x, to be safe for the
// unused low bits of the first state word).  This is a constant of mt19937.
static Bits mt19937AnnihilatorPolynomial() {
    const int nterms = 40000;
    const int nblocks = nterms / MT_N + 2;
    std::vector<uint32_t> seq(static_cast<size_t>(nblocks) * MT_N);
    mtSeed(seq.data(), MT_SEED);
    for (int b = 1; b < nblocks; ++b) mtNextBlock(&seq[(b - 1) * MT_N], &seq[b * MT_N]);
    std::vector<uint8_t> bits(nterms);
    for (int i = 0; i < nterms; ++i) bits[i] = static_cast<uint8_t>(seq[i] & 1u);
    Bits M = minimalPolynomial(bits);
    Bits M2(M.size() + 1, 0);
    xorShl(M2, M, 1);
    return M2;
}

// ----------------------------------------------------------------------------
// Device side: apply a jump polynomial g to a state window
//   y_w = XOR_{i : g_i = 1} x_{i+w},   w in [0, 624)
// where x is the raw sequence that starts with the source window.
// One thread block of MT_N threads per application.
// ----------------------------------------------------------------------------

constexpr int JUMP_OUT_PER_THREAD = 4;                   // output words per thread
constexpr int JUMP_BLOCK = MT_N / JUMP_OUT_PER_THREAD;   // 156 threads
constexpr int MAX_POLY_WORDS = 640;   // >= (deg+1)/32 for deg <= 19939
static_assert(MT_N % JUMP_OUT_PER_THREAD == 0, "jump block layout");

// Generate the window following `old` into `nw` (distinct buffers, whole block)
__device__ inline void mtGenBlockSync(const uint32_t* old, uint32_t* nw) {
    for (int k = threadIdx.x; k < MT_N - MT_M; k += blockDim.x) {
        nw[k] = old[k + MT_M] ^ mtTwist(old[k], old[k + 1]);
    }
    __syncthreads();
    for (int k = MT_N - MT_M + threadIdx.x; k < 2 * (MT_N - MT_M); k += blockDim.x) {
        nw[k] = nw[k - (MT_N - MT_M)] ^ mtTwist(old[k], old[k + 1]);
    }
    __syncthreads();
    for (int k = 2 * (MT_N - MT_M) + threadIdx.x; k < MT_N - 1; k += blockDim.x) {
        nw[k] = nw[k - (MT_N - MT_M)] ^ mtTwist(old[k], old[k + 1]);
    }
    __syncthreads();
    if (threadIdx.x == 0) nw[MT_N - 1] = nw[MT_M - 1] ^ mtTwist(old[MT_N - 1], nw[0]);
    __syncthreads();
}

__device__ void mtApplyPolyBlock(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                 const uint32_t* __restrict__ polyWords, int polyDeg,
                                 uint32_t* buf /* shared, 2*MT_N */, uint32_t* pw /* shared */) {
    const int tid = threadIdx.x;   // handles output words tid + u * JUMP_BLOCK, u < JUMP_OUT_PER_THREAD
    const int nPolyWords = polyDeg / 32 + 1;
    for (int k = tid; k < nPolyWords; k += blockDim.x) pw[k] = polyWords[k];
    for (int k = tid; k < MT_N; k += blockDim.x) buf[k] = src[k];
    __syncthreads();
    mtGenBlockSync(buf, buf + MT_N);

    uint32_t acc[JUMP_OUT_PER_THREAD];
    #pragma unroll
    for (int u = 0; u < JUMP_OUT_PER_THREAD; ++u) acc[u] = 0;
    const int nChunks = polyDeg / MT_N + 1;
    for (int c = 0; c < nChunks; ++c) {
        const int bitBase = c * MT_N;
        const int lim = min(MT_N, polyDeg + 1 - bitBase);
        // bitBase is a multiple of 624 = 19.5 * 32, so the chunk starts in the
        // middle of a polynomial word; mask the words to the chunk's bit range.
        const int wFirst = bitBase >> 5;
        const int wLast = (bitBase + lim - 1) >> 5;
        for (int wi = wFirst; wi <= wLast; ++wi) {
            uint32_t w = pw[wi];
            const int wbit0 = wi << 5;
            if (wbit0 < bitBase) w &= ~0u << (bitBase - wbit0);
            if (wbit0 + 32 > bitBase + lim) w &= (1u << (bitBase + lim - wbit0)) - 1u;
            while (w) {
                const int b = __ffs(w) - 1;
                const uint32_t* xp = buf + (wbit0 + b - bitBase) + tid;
                #pragma unroll
                for (int u = 0; u < JUMP_OUT_PER_THREAD; ++u) acc[u] ^= xp[u * JUMP_BLOCK];
                w &= w - 1;
            }
        }
        __syncthreads();
        if (c + 1 < nChunks) {
            uint32_t v[JUMP_OUT_PER_THREAD];
            #pragma unroll
            for (int u = 0; u < JUMP_OUT_PER_THREAD; ++u) v[u] = buf[MT_N + tid + u * JUMP_BLOCK];
            __syncthreads();
            #pragma unroll
            for (int u = 0; u < JUMP_OUT_PER_THREAD; ++u) buf[tid + u * JUMP_BLOCK] = v[u];
            __syncthreads();
            mtGenBlockSync(buf, buf + MT_N);
        }
    }
    #pragma unroll
    for (int u = 0; u < JUMP_OUT_PER_THREAD; ++u) dst[tid + u * JUMP_BLOCK] = acc[u];
}

// states[0] = apply(poly, seedWindow)        (single block)
__global__ void __launch_bounds__(JUMP_BLOCK)
mtJumpBaseKernel(const uint32_t* __restrict__ seedWindow, uint32_t* __restrict__ states,
                 const uint32_t* __restrict__ polyWords, int polyDeg) {
    __shared__ uint32_t buf[2 * MT_N];
    __shared__ uint32_t pw[MAX_POLY_WORDS];
    mtApplyPolyBlock(seedWindow, states, polyWords, polyDeg, buf, pw);
}

// Doubling round k: states[half + b] = apply(q_k, states[b]) for b in [0, count)
__global__ void __launch_bounds__(JUMP_BLOCK)
mtJumpRoundKernel(uint32_t* __restrict__ states, int half,
                  const uint32_t* __restrict__ polyWords, int polyDeg) {
    __shared__ uint32_t buf[2 * MT_N];
    __shared__ uint32_t pw[MAX_POLY_WORDS];
    const int b = blockIdx.x;
    mtApplyPolyBlock(states + static_cast<size_t>(b) * MT_N,
                     states + static_cast<size_t>(half + b) * MT_N, polyWords, polyDeg, buf, pw);
}

// In-place regeneration of a 624-word window held in shared memory by one warp
__device__ inline void mtRegenWarp(uint32_t* st, int lane) {
    // phase 1: k in [0, 227)
    for (int base = 0; base < MT_N - MT_M; base += 32) {
        const int k = base + lane;
        const bool act = k < MT_N - MT_M;
        uint32_t a = 0, b = 0, c = 0;
        if (act) { a = st[k]; b = st[k + 1]; c = st[k + MT_M]; }
        __syncwarp();
        if (act) st[k] = c ^ mtTwist(a, b);
        __syncwarp();
    }
    // phase 2: k in [227, 623)
    for (int base = MT_N - MT_M; base < MT_N - 1; base += 32) {
        const int k = base + lane;
        const bool act = k < MT_N - 1;
        uint32_t a = 0, b = 0, c = 0;
        if (act) { a = st[k]; b = st[k + 1]; c = st[k - (MT_N - MT_M)]; }
        __syncwarp();
        if (act) st[k] = c ^ mtTwist(a, b);
        __syncwarp();
    }
    // phase 3: k = 623
    if (lane == 0) st[MT_N - 1] = st[MT_M - 1] ^ mtTwist(st[MT_N - 1], st[0]);
    __syncwarp();
}

// ============================================================================
// Device geometry: ray-triangle intersection, ray-box test, visibility
// ============================================================================

// Möller-Trumbore ray-triangle intersection
__host__ __device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Segment / axis-aligned box separating axis test (identical to Octree::rayIntersectsBox)
__device__ inline bool rayIntersectsBoxDev(const Vec3& center, const Vec3& halfExtent,
                                           const Vec3& p1, const Vec3& p2) {
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

constexpr int TRAVERSAL_STACK = 96;

// Warp-synchronous visibility test.  Every lane owns one ray (from -> to).  The
// warp traverses the union of the octree nodes needed by its rays with a single
// shared stack of (node, lane mask) entries; a lane is part of a node's mask only
// if its ray passed the box tests of all ancestors, so every lane tests exactly
// the leaf triangles the original per-ray isRayBlocked() would test.  The
// returned value is the original decision: true if the ray is blocked.
__device__ bool isRayBlockedWarp(const Vec3& from, const Vec3& to, bool valid,
                                 int srcTriIdx, int dstTriIdx,
                                 const FlatNode* __restrict__ nodes,
                                 const int* __restrict__ triList,
                                 const Triangle* __restrict__ tris,
                                 int* __restrict__ sNode, uint32_t* __restrict__ sMask,
                                 int lane) {
    const uint32_t FULL = 0xffffffffu;
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    const bool degenerate = rayLen < EPSILON;
    Vec3 dirNorm = dir / (degenerate ? 1.0f : rayLen);
    bool blocked = degenerate;   // original: a zero-length ray counts as blocked

    uint32_t active = __ballot_sync(FULL, valid && !degenerate);
    int sp = 0;
    if (active) {
        if (lane == 0) { sNode[0] = 0; sMask[0] = active; }
        sp = 1;
    }
    __syncwarp();
    while (sp > 0) {
        --sp;
        const int nodeIdx = sNode[sp];
        uint32_t mask = sMask[sp];
        __syncwarp();
        mask &= __ballot_sync(FULL, !blocked);
        if (mask == 0) continue;
        const bool inMask = (mask >> lane) & 1u;

        const float4* np = reinterpret_cast<const float4*>(nodes + nodeIdx);
        const int4 hdr = __ldg(reinterpret_cast<const int4*>(np + 14));   // triStart, triCount
        const int triCount = hdr.y;
        if (triCount > 0) {
            // leaf: test triangles directly (four at a time for memory-level parallelism)
            const int triStart = hdr.x;
            for (int q0 = 0; q0 < triCount; q0 += 4) {
                int idx[4];
                bool hitAny = false;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    idx[u] = (q0 + u < triCount) ? __ldg(triList + triStart + q0 + u) : -1;
                }
                float4 t0[4], t1[4], t2[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4* tp = reinterpret_cast<const float4*>(tris + (idx[u] >= 0 ? idx[u] : 0));
                    t0[u] = __ldg(tp); t1[u] = __ldg(tp + 1); t2[u] = __ldg(tp + 2);
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const Vec3 v0(t0[u].x, t0[u].y, t0[u].z), v1(t0[u].w, t1[u].x, t1[u].y), v2(t1[u].z, t1[u].w, t2[u].x);
                    const val_t dist = rayTriangleIntersect(from, dirNorm, v0, v1, v2);
                    hitAny |= (idx[u] >= 0) && inMask && idx[u] != srcTriIdx && idx[u] != dstTriIdx &&
                              dist > EPSILON && dist < rayLen - EPSILON;
                }
                const uint32_t hm = __ballot_sync(FULL, hitAny);
                if (hm) {
                    blocked |= hitAny;   // Ray is blocked
                    mask &= ~hm;
                    if (mask == 0) break;
                }
            }
        } else {
            const int4 c0 = __ldg(reinterpret_cast<const int4*>(np + 12));
            const int4 c1 = __ldg(reinterpret_cast<const int4*>(np + 13));
            const int children[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
            float4 cb[8], hz[4];
            #pragma unroll
            for (int ci = 0; ci < 8; ++ci) cb[ci] = __ldg(np + ci);
            #pragma unroll
            for (int ci = 0; ci < 4; ++ci) hz[ci] = __ldg(np + 8 + ci);
            #pragma unroll
            for (int ci = 0; ci < 8; ++ci) {
                const int ch = children[ci];
                if (ch >= 0) {
                    const Vec3 center(cb[ci].x, cb[ci].y, cb[ci].z);
                    const Vec3 halfExtent(cb[ci].w, (ci & 1) ? hz[ci >> 1].z : hz[ci >> 1].x,
                                          (ci & 1) ? hz[ci >> 1].w : hz[ci >> 1].y);
                    const bool pass = inMask && rayIntersectsBoxDev(center, halfExtent, from, to);
                    const uint32_t m = __ballot_sync(FULL, pass);
                    if (m && sp < TRAVERSAL_STACK) {
                        if (lane == 0) { sNode[sp] = ch; sMask[sp] = m; }
                        ++sp;
                    }
                }
            }
        }
        __syncwarp();
    }
    return blocked;
}

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Form Factor Kernel (Kij)
// ============================================================================
//
// One warp per "generator": a contiguous range of non-culled (i, j) pairs in
// row-major order together with the mt19937 window at the stream position of
// the first pair.  Two pairs are processed per iteration (16 rays each, one
// ray per lane).  The ray contributions are accumulated by lanes 0 and 16 in
// ray order, exactly like the sequential loop of the original computeKij().

constexpr int KIJ_WARPS_PER_BLOCK = 4;
constexpr int KIJ_BLOCK = 32 * KIJ_WARPS_PER_BLOCK;
constexpr int PAIRS_PER_ITER = 2;
constexpr int OUT_WORDS = PAIRS_PER_ITER * WORDS_PER_PAIR;   // 128

struct KijParams {
    const Triangle* tris;        // all N triangles
    const FlatNode* nodes;       // flattened octree
    const int* triList;          // octree leaf triangle lists
    const uint32_t* mask;        // active-pair bitmask for local rows (1 = compute)
    const int64_t* rowStart;     // exclusive prefix of active pairs per local row (nloc + 1)
    const uint32_t* genStates;   // G x 624 mt19937 windows
    val_t* kij;                  // output, local rows: kij[il * N + j]
    int N;                       // total triangles
    int nloc;                    // local rows
    int rowOffset;               // first global row of this rank
    int maskWordsPerRow;
    int64_t pairsPerGen;
    int64_t numPairs;            // local active pairs
    int numGen;
};

__device__ inline bool maskActive(const uint32_t* __restrict__ mask, int wpr, int il, int j) {
    return (mask[static_cast<size_t>(il) * wpr + (j >> 5)] >> (j & 31)) & 1u;
}

// Generate a random point inside a triangle using barycentric coordinates
__host__ __device__ inline Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__global__ void __launch_bounds__(KIJ_BLOCK, 5)
kijKernel(const KijParams p) {
    __shared__ uint32_t stS[KIJ_WARPS_PER_BLOCK][MT_N];
    __shared__ uint32_t obS[KIJ_WARPS_PER_BLOCK][OUT_WORDS];
    __shared__ int stackNodeS[KIJ_WARPS_PER_BLOCK][TRAVERSAL_STACK];
    __shared__ uint32_t stackMaskS[KIJ_WARPS_PER_BLOCK][TRAVERSAL_STACK];

    const int warpInBlock = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * KIJ_WARPS_PER_BLOCK + warpInBlock;
    if (g >= p.numGen) return;

    uint32_t* st = stS[warpInBlock];
    uint32_t* ob = obS[warpInBlock];
    int* sNode = stackNodeS[warpInBlock];
    uint32_t* sMask = stackMaskS[warpInBlock];

    const int64_t P0 = static_cast<int64_t>(g) * p.pairsPerGen;
    const int64_t P1 = min(P0 + p.pairsPerGen, p.numPairs);
    if (P0 >= P1) return;

    for (int k = lane; k < MT_N; k += 32) st[k] = p.genStates[static_cast<size_t>(g) * MT_N + k];
    __syncwarp();
    int cursor = 0;

    // Locate the first pair (il, j) of this generator (uniform across the warp)
    int il;
    {
        int lo = 0, hi = p.nloc;  // find largest il with rowStart[il] <= P0
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (p.rowStart[mid] <= P0) lo = mid; else hi = mid;
        }
        il = lo;
    }
    int j = -1;
    {
        int64_t rem = P0 - p.rowStart[il];
        const uint32_t* mrow = p.mask + static_cast<size_t>(il) * p.maskWordsPerRow;
        int w = 0;
        int64_t cnt = 0;
        while (true) {
            uint32_t m = mrow[w];
            int pc = __popc(m);
            if (cnt + pc > rem) {
                int need = static_cast<int>(rem - cnt);
                for (int b = 0; b < 32; ++b) {
                    if ((m >> b) & 1u) {
                        if (need == 0) { j = w * 32 + b; break; }
                        --need;
                    }
                }
                break;
            }
            cnt += pc;
            ++w;
        }
    }

    const int N = p.N;
    for (int64_t pp = P0; pp < P1; pp += PAIRS_PER_ITER) {
        const int npairs = static_cast<int>(min(static_cast<int64_t>(PAIRS_PER_ITER), P1 - pp));

        // Fetch the random words of these pairs from the mt19937 window
        const int need = npairs * WORDS_PER_PAIR;
        int got = 0;
        while (got < need) {
            if (cursor == MT_N) {
                mtRegenWarp(st, lane);
                cursor = 0;
            }
            const int n = min(MT_N - cursor, need - got);
            for (int k = lane; k < n; k += 32) ob[got + k] = st[cursor + k];
            __syncwarp();
            cursor += n;
            got += n;
        }

        // Pair coordinates (uniform)
        const int il1 = il, j1 = j;
        int il2 = il, j2 = j;
        // advance to next active pair
        auto advance = [&](int& ai, int& aj) {
            do {
                ++aj;
                if (aj >= N) { aj = 0; ++ai; }
            } while (ai < p.nloc && !maskActive(p.mask, p.maskWordsPerRow, ai, aj));
        };
        if (npairs == 2) {
            advance(il2, j2);
            il = il2; j = j2;
            advance(il, j);
        } else {
            advance(il, j);
        }

        // One ray per lane (lanes 0-15: first pair, lanes 16-31: second pair)
        const int which = lane >> 4;
        const int r = lane & 15;
        const bool valid = which < npairs;
        const int cil = which ? il2 : il1;
        const int cj = valid ? (which ? j2 : j1) : j1;
        const int ci = p.rowOffset + (valid ? cil : il1);
        const Triangle triI = p.tris[ci];
        const Triangle triJ = p.tris[cj];
        const uint32_t* wd = ob + (valid ? which : 0) * WORDS_PER_PAIR + r * 4;
        const val_t uI = mtWordToFloat(mtTemper(wd[0]));
        const val_t vI = mtWordToFloat(mtTemper(wd[1]));
        const val_t uJ = mtWordToFloat(mtTemper(wd[2]));
        const val_t vJ = mtWordToFloat(mtTemper(wd[3]));
        const Vec3 pI = randomPointInTriangle(triI, uI, vI);
        const Vec3 pJ = randomPointInTriangle(triJ, uJ, vJ);

        const bool blocked = isRayBlockedWarp(pI, pJ, valid, ci, cj, p.nodes, p.triList, p.tris,
                                              sNode, sMask, lane);
        val_t contrib = ZERO;
        if (valid && !blocked) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosPhiI = cosPhi(v, triI.normal());
                val_t cosPhiJ = cosPhi(-v, triJ.normal());
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }

        // Ordered accumulation over the 16 rays of each pair (same order as original)
        val_t kij = ZERO;
        const int base = lane & 16;
        #pragma unroll
        for (int rr = 0; rr < NUM_RAYS; ++rr) {
            kij += __shfl_sync(0xffffffffu, contrib, base + rr);
        }
        if (lane == 0) {
            p.kij[static_cast<size_t>(il1) * N + j1] = kij * INV_NUM_RAYS;
        } else if (lane == 16 && npairs == 2) {
            p.kij[static_cast<size_t>(il2) * N + j2] = kij * INV_NUM_RAYS;
        }
        __syncwarp();
    }
}

// ============================================================================
// Tau (time delay) Kernel
// ============================================================================

__host__ __device__ inline int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void tauKernel(const Triangle* __restrict__ tris, int* __restrict__ tau,
                          int N, int nloc, int rowOffset) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int il = blockIdx.y;
    if (j >= N) return;
    const int i = rowOffset + il;
    tau[static_cast<size_t>(il) * N + j] = (i == j) ? 0 : computeTau(tris[i], tris[j]);
}

// ============================================================================
// Simulation Kernel (Wave Propagation), one timestep
// ============================================================================
//
// One warp per local row.  The 32 lanes evaluate 32 consecutive j terms; the
// terms are then accumulated in increasing j order so the float summation is
// identical to the sequential loop of the original.

constexpr int SIM_WARPS_PER_BLOCK = 4;
constexpr int SIM_BLOCK = 32 * SIM_WARPS_PER_BLOCK;

__global__ void __launch_bounds__(SIM_BLOCK)
simKernel(const val_t* __restrict__ kij, const int* __restrict__ tau,
          const val_t* __restrict__ areas, const val_t* __restrict__ rho,
          const val_t* __restrict__ radE, val_t* __restrict__ radB,
          int N, int nloc, int rowOffset, int t) {
    const int warpInBlock = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int il = blockIdx.x * SIM_WARPS_PER_BLOCK + warpInBlock;
    if (il >= nloc) return;
    const int i = rowOffset + il;

    val_t sumB = ZERO;
    for (int jb = 0; jb < N; jb += 32) {
        const int j = jb + lane;
        val_t term = ZERO;
        if (j < N && j != i) {
            const int tauij = tau[static_cast<size_t>(il) * N + j];
            if (t >= tauij) {
                const val_t kijv = kij[static_cast<size_t>(il) * N + j];
                if (kijv > ZERO) {
                    const val_t radJ = radB[static_cast<size_t>(t - tauij) * N + j];
                    if (radJ > ZERO) {
                        const val_t ka = kijv * areas[j];
                        term = ((ONE < ka) ? ONE : ka) * radJ;   // std::min(ka, ONE) * radJ
                    }
                }
            }
        }
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            sumB += __shfl_sync(0xffffffffu, term, l);
        }
    }
    if (lane == 0) {
        radB[static_cast<size_t>(t) * N + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
    }
}

// ============================================================================
// Distance Kernels (Cross-Correlation)
// ============================================================================

__global__ void corrKernel(const val_t* __restrict__ radB, val_t* __restrict__ corr,
                           int N, int T, int nloc, int rowOffset, int sourceIndex) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int il = blockIdx.y;
    if (t >= T) return;
    const int i = rowOffset + il;
    val_t sum = ZERO;
    for (int tt = t; tt < T; ++tt) {
        val_t pB = radB[static_cast<size_t>(tt) * N + i];
        val_t pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
        sum += pS * pB;
    }
    corr[static_cast<size_t>(il) * T + t] = sum;
}

__global__ void argmaxKernel(const val_t* __restrict__ corr, val_t* __restrict__ distances,
                             int T, int nloc) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= nloc) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        val_t s = corr[static_cast<size_t>(il) * T + t];
        if (s > maxCorr) {
            maxCorr = s;
            bestT = t;
        }
    }
    distances[il] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// Count entries > EPSILON (validation support)
__global__ void countNonZeroKernel(const val_t* __restrict__ data, size_t n,
                                   unsigned long long* __restrict__ count) {
    __shared__ unsigned int partial[256];
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int c = 0;
    for (; idx < n; idx += stride) c += (data[idx] > EPSILON) ? 1u : 0u;
    partial[threadIdx.x] = c;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(count, static_cast<unsigned long long>(partial[0]));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    int numTriangles = 0;
    int numTimesteps = 0;

    // MPI decomposition (contiguous row blocks)
    int rank = 0, size = 1;
    std::vector<int> rowOffsets;    // size + 1
    int rowOffset = 0;              // first local row
    int nloc = 0;                   // number of local rows

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N), host copy for validation
    std::vector<val_t> distances;   // Computed distances from source (all N)

    Octree octree;                  // Spatial acceleration structure
    std::vector<FlatNode> nodes;    // flattened octree
    std::vector<int> triList;

    size_t sourceIndex = 0;

    // mt19937 jump-ahead arithmetic (a constant of mt19937, prepared during initialization)
    PolyMod rngMod;

    // Active-pair bookkeeping for the form factor phase
    std::vector<uint32_t> mask;     // local rows x maskWordsPerRow
    int maskWordsPerRow = 0;
    std::vector<int64_t> rowStart;  // nloc + 1
    int64_t numPairsLocal = 0;
    int64_t pairOffsetGlobal = 0;   // active pairs in rows before rowOffset
    uint64_t nonZeroKij = 0;        // global count (filled for validation)

    // Device data
    Triangle* d_tris = nullptr;
    FlatNode* d_nodes = nullptr;
    int* d_triList = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radE = nullptr;        // T x N
    val_t* d_radB = nullptr;        // T x N
    val_t* d_kij = nullptr;         // nloc x N (local rows)
    int* d_tau = nullptr;           // nloc x N (local rows)
    val_t* d_dist = nullptr;        // nloc
    val_t* d_corr = nullptr;        // nloc x T
    val_t* h_rowLocal = nullptr;    // pinned, nloc
    val_t* h_rowFull = nullptr;     // pinned, N

    size_t idxTN(size_t t, size_t n) const { return t * static_cast<size_t>(numTriangles) + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh and build the octree; in parallel, prepare the (constant)
    // mt19937 jump-ahead polynomial arithmetic used by the form factor phase.
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
            state.triangles = std::move(mesh.triangles);
            if (state.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.triangles.size());

            // Build octree for spatial acceleration
            if (state.rank == 0) printf("Building octree...\n");
            state.octree.build(state.triangles);
            state.octree.flatten(state.nodes, state.triList);
        }
        #pragma omp section
        {
            state.rngMod.init(mt19937AnnihilatorPolynomial());
        }
    }
    state.numTriangles = static_cast<int>(state.triangles.size());
    state.numTimesteps = static_cast<int>(timesteps);
    state.sourceIndex = sourceIdx % state.triangles.size();
    const int N = state.numTriangles;
    const int T = state.numTimesteps;

    // Initialize areas
    state.areas.resize(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Row decomposition
    state.rowOffsets.resize(state.size + 1);
    for (int r = 0; r <= state.size; ++r) {
        state.rowOffsets[r] = static_cast<int>((static_cast<int64_t>(N) * r) / state.size);
    }
    state.rowOffset = state.rowOffsets[state.rank];
    state.nloc = state.rowOffsets[state.rank + 1] - state.rowOffset;
    const int nloc = state.nloc;

    state.distances.assign(N, ZERO);

    // Set source emission (active for first half of timesteps)
    std::vector<val_t> radE(static_cast<size_t>(T) * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Device allocations and uploads
    CUDA_CHECK(cudaMalloc(&state.d_tris, sizeof(Triangle) * N));
    CUDA_CHECK(cudaMemcpy(state.d_tris, state.triangles.data(), sizeof(Triangle) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, sizeof(FlatNode) * state.nodes.size()));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, state.nodes.data(), sizeof(FlatNode) * state.nodes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_triList, sizeof(int) * std::max<size_t>(1, state.triList.size())));
    if (!state.triList.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_triList, state.triList.data(), sizeof(int) * state.triList.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&state.d_areas, sizeof(val_t) * N));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), sizeof(val_t) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rho, sizeof(val_t) * N));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), sizeof(val_t) * N, cudaMemcpyHostToDevice));

    const size_t tnBytes = sizeof(val_t) * std::max<size_t>(1, static_cast<size_t>(T) * N);
    CUDA_CHECK(cudaMalloc(&state.d_radE, tnBytes));
    CUDA_CHECK(cudaMalloc(&state.d_radB, tnBytes));
    if (!radE.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_radE, radE.data(), sizeof(val_t) * radE.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(state.d_radB, 0, tnBytes));

    const size_t locBytes = std::max<size_t>(1, static_cast<size_t>(N) * nloc);
    CUDA_CHECK(cudaMalloc(&state.d_kij, sizeof(val_t) * locBytes));
    CUDA_CHECK(cudaMemset(state.d_kij, 0, sizeof(val_t) * locBytes));
    CUDA_CHECK(cudaMalloc(&state.d_tau, sizeof(int) * locBytes));
    CUDA_CHECK(cudaMemset(state.d_tau, 0, sizeof(int) * locBytes));
    CUDA_CHECK(cudaMalloc(&state.d_dist, sizeof(val_t) * std::max(1, nloc)));
    CUDA_CHECK(cudaMalloc(&state.d_corr, sizeof(val_t) * std::max<size_t>(1, static_cast<size_t>(nloc) * T)));
    CUDA_CHECK(cudaHostAlloc(&state.h_rowLocal, sizeof(val_t) * std::max(1, nloc), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&state.h_rowFull, sizeof(val_t) * N, cudaHostAllocDefault));
}

void releaseSimulation(SimulationState& state) {
    cudaFree(state.d_tris); cudaFree(state.d_nodes); cudaFree(state.d_triList);
    cudaFree(state.d_areas); cudaFree(state.d_rho); cudaFree(state.d_radE); cudaFree(state.d_radB);
    cudaFree(state.d_kij); cudaFree(state.d_tau); cudaFree(state.d_dist); cudaFree(state.d_corr);
    cudaFreeHost(state.h_rowLocal); cudaFreeHost(state.h_rowFull);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    if (state.nloc == 0) return;
    dim3 grid((state.numTriangles + 255) / 256, state.nloc);
    tauKernel<<<grid, 256>>>(state.d_tris, state.d_tau, state.numTriangles, state.nloc, state.rowOffset);
    CUDA_CHECK(cudaGetLastError());
}

// Upload a GF(2) polynomial as 32-bit words
static uint32_t* uploadPoly(const Bits& p, int& deg) {
    deg = polyDegree(p);
    std::vector<uint32_t> words(std::max<size_t>(1, p.size() * 2));
    for (size_t i = 0; i < p.size(); ++i) {
        words[2 * i] = static_cast<uint32_t>(p[i] & 0xffffffffu);
        words[2 * i + 1] = static_cast<uint32_t>(p[i] >> 32);
    }
    uint32_t* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(uint32_t) * words.size()));
    CUDA_CHECK(cudaMemcpy(d, words.data(), sizeof(uint32_t) * words.size(), cudaMemcpyHostToDevice));
    return d;
}

// Threshold (in raw mt19937 words) above which the polynomial jump-ahead is
// used instead of sequential generation with snapshots.
constexpr int64_t JUMP_THRESHOLD_WORDS = int64_t(1) << 28;

// Produce the G mt19937 windows for generators at raw offsets W0 + g * delta on the device.
static void prepareGeneratorStates(const PolyMod& mod, uint32_t* d_states, int G, int64_t W0, int64_t delta) {
    if (G <= 0) return;
    const int64_t Wend = W0 + static_cast<int64_t>(G - 1) * delta + MT_N;

    if (Wend <= JUMP_THRESHOLD_WORDS) {
        // Sequential generation with window snapshots (small problems)
        std::vector<uint32_t> states(static_cast<size_t>(G) * MT_N);
        std::vector<uint32_t> bufA(MT_N), bufB(MT_N), bufC(MT_N);
        uint32_t* blk = bufA.data();    // block bidx
        uint32_t* nxt = bufB.data();    // block bidx + 1
        uint32_t* tmp = bufC.data();
        mtSeed(blk, MT_SEED);
        mtNextBlock(blk, nxt);
        int64_t bidx = 0;
        for (int g = 0; g < G; ++g) {
            const int64_t Wg = W0 + static_cast<int64_t>(g) * delta;
            const int64_t b = Wg / MT_N;
            const int c = static_cast<int>(Wg % MT_N);
            while (bidx < b) {
                mtNextBlock(nxt, tmp);
                uint32_t* t = blk; blk = nxt; nxt = tmp; tmp = t;
                ++bidx;
            }
            uint32_t* dst = &states[static_cast<size_t>(g) * MT_N];
            memcpy(dst, blk + c, sizeof(uint32_t) * (MT_N - c));
            memcpy(dst + (MT_N - c), nxt, sizeof(uint32_t) * c);
        }
        CUDA_CHECK(cudaMemcpy(d_states, states.data(), sizeof(uint32_t) * states.size(), cudaMemcpyHostToDevice));
        return;
    }

    // Polynomial jump-ahead (large problems)
    Bits pBase, pDelta;
    #pragma omp parallel sections
    {
        #pragma omp section
        pBase = mod.powX(static_cast<uint64_t>(W0));
        #pragma omp section
        pDelta = mod.powX(static_cast<uint64_t>(delta));
    }
    int rounds = 0;
    while ((int64_t(1) << rounds) < G) ++rounds;
    std::vector<Bits> q(rounds);
    if (rounds > 0) {
        q[0] = pDelta;
        for (int k = 1; k < rounds; ++k) q[k] = mod.square(q[k - 1]);
    }

    uint32_t seed[MT_N];
    mtSeed(seed, MT_SEED);
    uint32_t* d_seed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seed, sizeof(uint32_t) * MT_N));
    CUDA_CHECK(cudaMemcpy(d_seed, seed, sizeof(uint32_t) * MT_N, cudaMemcpyHostToDevice));

    int degBase = 0;
    uint32_t* d_pBase = uploadPoly(pBase, degBase);
    mtJumpBaseKernel<<<1, JUMP_BLOCK>>>(d_seed, d_states, d_pBase, degBase);
    CUDA_CHECK(cudaGetLastError());

    for (int k = 0; k < rounds; ++k) {
        const int half = 1 << k;
        const int count = std::min(half, G - half);
        if (count <= 0) break;
        int deg = 0;
        uint32_t* d_q = uploadPoly(q[k], deg);
        mtJumpRoundKernel<<<count, JUMP_BLOCK>>>(d_states, half, d_q, deg);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_q));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_pBase));
    CUDA_CHECK(cudaFree(d_seed));
}

constexpr int MAX_GENERATORS = 8192;
constexpr int64_t MIN_PAIRS_PER_GENERATOR = 32;

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    const int N = state.numTriangles;
    const int nloc = state.nloc;

    // Active-pair mask (pairs that consume random numbers in the original: i != j and
    // normals not facing the same direction) and per-row counts for the whole mesh.
    std::vector<int> rowCount(N, 0);
    state.maskWordsPerRow = (N + 31) / 32;
    state.mask.assign(static_cast<size_t>(std::max(nloc, 1)) * state.maskWordsPerRow, 0u);
    const Triangle* tris = state.triangles.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const Vec3 ni = tris[i].normal();
        int cnt = 0;
        const bool local = (i >= state.rowOffset && i < state.rowOffset + nloc);
        uint32_t* mrow = local ? &state.mask[static_cast<size_t>(i - state.rowOffset) * state.maskWordsPerRow] : nullptr;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (ni.dot(tris[j].normal()) > 0.99f) continue;
            ++cnt;
            if (local) mrow[j >> 5] |= 1u << (j & 31);
        }
        rowCount[i] = cnt;
    }
    state.pairOffsetGlobal = 0;
    for (int i = 0; i < state.rowOffset; ++i) state.pairOffsetGlobal += rowCount[i];
    state.rowStart.assign(nloc + 1, 0);
    for (int il = 0; il < nloc; ++il) state.rowStart[il + 1] = state.rowStart[il] + rowCount[state.rowOffset + il];
    state.numPairsLocal = state.rowStart[nloc];

    if (state.numPairsLocal > 0) {
        // Generators: contiguous pair ranges of equal size
        int G = static_cast<int>(std::min<int64_t>(MAX_GENERATORS,
                     (state.numPairsLocal + MIN_PAIRS_PER_GENERATOR - 1) / MIN_PAIRS_PER_GENERATOR));
        G = std::max(G, 1);
        const int64_t pairsPerGen = (state.numPairsLocal + G - 1) / G;
        G = static_cast<int>((state.numPairsLocal + pairsPerGen - 1) / pairsPerGen);
        const int64_t delta = WORDS_PER_PAIR * pairsPerGen;
        const int64_t W0 = MT_N + WORDS_PER_PAIR * state.pairOffsetGlobal;

        uint32_t* d_states = nullptr;
        CUDA_CHECK(cudaMalloc(&d_states, sizeof(uint32_t) * static_cast<size_t>(G) * MT_N));
        prepareGeneratorStates(state.rngMod, d_states, G, W0, delta);

        uint32_t* d_mask = nullptr;
        int64_t* d_rowStart = nullptr;
        CUDA_CHECK(cudaMalloc(&d_mask, sizeof(uint32_t) * state.mask.size()));
        CUDA_CHECK(cudaMemcpy(d_mask, state.mask.data(), sizeof(uint32_t) * state.mask.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_rowStart, sizeof(int64_t) * state.rowStart.size()));
        CUDA_CHECK(cudaMemcpy(d_rowStart, state.rowStart.data(), sizeof(int64_t) * state.rowStart.size(), cudaMemcpyHostToDevice));

        KijParams p;
        p.tris = state.d_tris;
        p.nodes = state.d_nodes;
        p.triList = state.d_triList;
        p.mask = d_mask;
        p.rowStart = d_rowStart;
        p.genStates = d_states;
        p.kij = state.d_kij;
        p.N = N;
        p.nloc = nloc;
        p.rowOffset = state.rowOffset;
        p.maskWordsPerRow = state.maskWordsPerRow;
        p.pairsPerGen = pairsPerGen;
        p.numPairs = state.numPairsLocal;
        p.numGen = G;

        const int blocks = (G + KIJ_WARPS_PER_BLOCK - 1) / KIJ_WARPS_PER_BLOCK;
        kijKernel<<<blocks, KIJ_BLOCK>>>(p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_states));
        CUDA_CHECK(cudaFree(d_mask));
        CUDA_CHECK(cudaFree(d_rowStart));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (state.rank == 0) printf("  Progress: %d/%d triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    const int N = state.numTriangles;
    const int T = state.numTimesteps;
    const int nloc = state.nloc;

    std::vector<int> counts(state.size), displs(state.size);
    for (int r = 0; r < state.size; ++r) {
        counts[r] = state.rowOffsets[r + 1] - state.rowOffsets[r];
        displs[r] = state.rowOffsets[r];
    }

    for (int t = 0; t < T; ++t) {
        if (nloc > 0) {
            const int blocks = (nloc + SIM_WARPS_PER_BLOCK - 1) / SIM_WARPS_PER_BLOCK;
            simKernel<<<blocks, SIM_BLOCK>>>(state.d_kij, state.d_tau, state.d_areas, state.d_rho,
                                             state.d_radE, state.d_radB, N, nloc, state.rowOffset, t);
            CUDA_CHECK(cudaGetLastError());
        }
        if (state.size > 1) {
            // Exchange this timestep's radiosity row with all ranks
            val_t* rowDev = state.d_radB + static_cast<size_t>(t) * N;
            if (nloc > 0) {
                CUDA_CHECK(cudaMemcpy(state.h_rowLocal, rowDev + state.rowOffset,
                                      sizeof(val_t) * nloc, cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            MPI_Allgatherv(state.h_rowLocal, nloc, MPI_FLOAT, state.h_rowFull,
                           counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(rowDev, state.h_rowFull, sizeof(val_t) * N, cudaMemcpyHostToDevice));
        }

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    const int N = state.numTriangles;
    const int T = state.numTimesteps;
    const int nloc = state.nloc;

    if (nloc > 0) {
        if (T > 0) {
            dim3 grid((T + 127) / 128, nloc);
            corrKernel<<<grid, 128>>>(state.d_radB, state.d_corr, N, T, nloc, state.rowOffset,
                                      static_cast<int>(state.sourceIndex));
            CUDA_CHECK(cudaGetLastError());
        }
        argmaxKernel<<<(nloc + 255) / 256, 256>>>(state.d_corr, state.d_dist, T, nloc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.h_rowLocal, state.d_dist, sizeof(val_t) * nloc, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> counts(state.size), displs(state.size);
    for (int r = 0; r < state.size; ++r) {
        counts[r] = state.rowOffsets[r + 1] - state.rowOffsets[r];
        displs[r] = state.rowOffsets[r];
    }
    MPI_Allgatherv(state.h_rowLocal, nloc, MPI_FLOAT, state.distances.data(),
                   counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// Gather data needed for validation: global non-zero Kij count and the host radB copy
void gatherValidationData(SimulationState& state) {
    const int N = state.numTriangles;
    const int T = state.numTimesteps;
    unsigned long long local = 0;
    if (state.nloc > 0) {
        unsigned long long* d_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
        const size_t n = static_cast<size_t>(N) * state.nloc;
        const int blocks = static_cast<int>(std::min<size_t>(4096, (n + 255) / 256));
        countNonZeroKernel<<<blocks, 256>>>(state.d_kij, n, d_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&local, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_count));
    }
    unsigned long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = global;

    if (state.rank == 0) {
        state.radB.resize(static_cast<size_t>(T) * N);
        if (!state.radB.empty()) {
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, sizeof(val_t) * state.radB.size(),
                                  cudaMemcpyDeviceToHost));
        }
    }
}

// ============================================================================
// Validation (rank 0)
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");
    const size_t N = static_cast<size_t>(state.numTriangles);
    const size_t T = static_cast<size_t>(state.numTimesteps);

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < N; ++i) {
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = 0; t < T; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, N * N,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(N * N));

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
    for (size_t i = 0; i < static_cast<size_t>(state.numTriangles); ++i) {
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

static double wallSeconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int parseStatus = 0;  // 0 = run, 1 = help, 2 = error
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
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }
    if (timesteps < 0) timesteps = 0;

    // Select the GPU: one device per rank, round-robin over the node-local ranks
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        CUDA_CHECK(cudaFree(nullptr));  // establish the context
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", size, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.size = size;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = wallSeconds();

    computeTimeDelays(state);
    computeFormFactors(state);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double endPre = wallSeconds();
    long preDuration = static_cast<long>((endPre - startPre) * 1000.0);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = wallSeconds();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endSim = wallSeconds();
    long simDuration = static_cast<long>((endSim - startSim) * 1000.0);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = wallSeconds();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endDist = wallSeconds();
    long distDuration = static_cast<long>((endDist - startDist) * 1000.0);

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;

    if (rank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = static_cast<size_t>(state.numTriangles);
        size_t t = static_cast<size_t>(state.numTimesteps);
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
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        gatherValidationData(state);
        int ok = 1;
        if (rank == 0) ok = validateResults(state) ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) exitCode = 1;
    }

    releaseSimulation(state);
    MPI_Finalize();
    return exitCode;
}

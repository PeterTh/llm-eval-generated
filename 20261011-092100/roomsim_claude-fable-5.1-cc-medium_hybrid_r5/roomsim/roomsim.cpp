/**
 * Room Response Simulation Benchmark - Hybrid MPI + OpenMP + CUDA version
 *
 * Models how sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * Parallelization strategy
 * ------------------------
 *  - MPI:    the N x N pair matrices (Kij, Tau) are distributed by contiguous blocks of
 *            receiver rows across ranks. Each rank drives one GPU. The per-timestep
 *            radiosity row is exchanged with MPI_Allgatherv; distances are gathered at the end.
 *  - CUDA:   form factor Monte Carlo ray tracing (octree-accelerated), wave propagation
 *            and cross-correlation run as kernels on the rank's GPU.
 *  - OpenMP: host-side preparation (visibility culling / ray slot tables, time delays,
 *            random stream checkpointing, mesh/validation loops).
 *
 * Random numbers: the original code draws all Monte Carlo samples from a single
 * sequential std::mt19937 stream. To preserve exactly the same samples, the stream is
 * reproduced on the GPU: every receiver row gets a checkpoint of the generator state
 * (computed on the host using a GF(2) polynomial jump-ahead so that ranks and OpenMP
 * threads can start in the middle of the stream), and a kernel regenerates the row's
 * sub-stream in parallel before the ray tracing kernel consumes it.
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
#define OMPI_SKIP_MPICXX 1   // C bindings only
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
constexpr val_t VAL_MAX = 3.402823466e+38f;   // std::numeric_limits<float>::max()

// Random draws consumed per non-culled triangle pair (2 points x 2 coordinates x NUM_RAYS)
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;

static int g_rank = 0;
static int g_size = 1;

#define ROOT_PRINTF(...) do { if (g_rank == 0) { printf(__VA_ARGS__); fflush(stdout); } } while (0)

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "[rank %d] CUDA error %s at %s:%d\n", g_rank,                  \
                    cudaGetErrorString(err__), __FILE__, __LINE__);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

static void fatal(const char* msg) {
    fprintf(stderr, "[rank %d] FATAL: %s\n", g_rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

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

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// std::min semantics (returns first argument unless second is strictly smaller)
__host__ __device__ inline val_t minVal(val_t a, val_t b) { return b < a ? b : a; }
__host__ __device__ inline val_t maxVal(val_t a, val_t b) { return a < b ? b : a; }

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

// Flat octree node representation for the GPU
struct GNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];        // -1 if absent
    int triStart;        // offset into the flat leaf triangle index list
    int triCount;        // > 0 for leaves, 0 for internal nodes
};

struct FlatOctree {
    std::vector<GNode> nodes;
    std::vector<int> leafTris;
    int maxDepth = 0;

    void build(const Octree& root) {
        nodes.clear();
        leafTris.clear();
        maxDepth = 0;
        addNode(root, 0);
    }

private:
    int addNode(const Octree& n, int depth) {
        maxDepth = std::max(maxDepth, depth);
        int id = static_cast<int>(nodes.size());
        nodes.emplace_back();
        GNode g;
        g.center = n.center;
        g.halfExtent = n.halfExtent;
        for (int i = 0; i < 8; ++i) g.child[i] = -1;
        g.triStart = static_cast<int>(leafTris.size());
        g.triCount = static_cast<int>(n.triangleIndices.size());
        for (size_t idx : n.triangleIndices) leafTris.push_back(static_cast<int>(idx));
        nodes[id] = g;
        for (int i = 0; i < 8; ++i) {
            if (n.children[i]) {
                int c = addNode(*n.children[i], depth + 1);
                nodes[id].child[i] = c;
            }
        }
        return id;
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
// Random Number Generation: MT19937 with jump-ahead
//
// Bit-compatible re-implementation of std::mt19937 + std::uniform_real_distribution<float>
// (as implemented by libstdc++'s generate_canonical<float, 24>). The state is stored as a
// circular 624-word window; a "step" advances the window by one word. The jump-ahead uses
// the characteristic polynomial of the recurrence (computed once with Berlekamp-Massey)
// and evaluates x^J mod phi(x) at the state transition via Horner's rule.
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr float MT_INV_2POW32 = 2.3283064365386962890625e-10f;   // exactly 2^-32
constexpr float MT_ONE_MINUS_EPS = 0.99999994039535522461f;       // nextafter(1.0f, 0.0f)

__host__ __device__ inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// Convert a 32-bit draw to a float in [0, 1) exactly as libstdc++ does.
__host__ __device__ inline float mtCanonical(uint32_t x) {
    float r = static_cast<float>(x) * MT_INV_2POW32;
    if (r >= 1.0f) r = MT_ONE_MINUS_EPS;
    return r;
}

struct MT19937 {
    uint32_t mt[MT_N];
    int idx;

    MT19937() : idx(0) { std::memset(mt, 0, sizeof(mt)); }

    void seed(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < MT_N; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        idx = 0;
    }

    // Advance the window by one word; returns the newly generated (untempered) word.
    inline uint32_t stepRaw() {
        int i = idx;
        int i1 = i + 1; if (i1 == MT_N) i1 = 0;
        int im = i + MT_M; if (im >= MT_N) im -= MT_N;
        uint32_t y = (mt[i] & MT_UPPER) | (mt[i1] & MT_LOWER);
        uint32_t v = mt[im] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        mt[i] = v;
        idx = i1;
        return v;
    }

    inline uint32_t next() { return mtTemper(stepRaw()); }
    inline float nextFloat() { return mtCanonical(next()); }

    // Full in-place twist; equivalent to MT_N steps when idx == 0.
    // Written as three dependency-free loops so that the compiler can vectorize it.
    void twistAll() {
        uint32_t* __restrict__ m = mt;
        for (int i = 0; i < MT_N - MT_M; ++i) {
            uint32_t y = (m[i] & MT_UPPER) | (m[i + 1] & MT_LOWER);
            m[i] = m[i + MT_M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        }
        for (int i = MT_N - MT_M; i < MT_N - 1; ++i) {
            uint32_t y = (m[i] & MT_UPPER) | (m[i + 1] & MT_LOWER);
            m[i] = m[i + MT_M - MT_N] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        }
        {
            uint32_t y = (m[MT_N - 1] & MT_UPPER) | (m[0] & MT_LOWER);
            m[MT_N - 1] = m[MT_M - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        }
    }

    void discard(uint64_t n) {
        while (n > 0) {
            if (idx == 0 && n >= static_cast<uint64_t>(MT_N)) {
                twistAll();
                n -= MT_N;
            } else {
                stepRaw();
                --n;
            }
        }
    }

    // Copy the window out in sequence order (window element 0 first).
    void exportWindow(uint32_t* out) const {
        for (int m = 0; m < MT_N; ++m) out[m] = mt[(idx + m) % MT_N];
    }

    // XOR another generator's window (given in sequence order) into this window.
    void xorWindow(const uint32_t* w) {
        int first = MT_N - idx;
        for (int m = 0; m < first; ++m) mt[idx + m] ^= w[m];
        for (int m = first; m < MT_N; ++m) mt[m - first] ^= w[m];
    }
};

// --- GF(2) polynomial helpers (little-endian bit order, bit k = coefficient of x^k) ---
using Poly = std::vector<uint64_t>;

static inline bool polyBit(const Poly& p, size_t k) {
    size_t w = k >> 6;
    return w < p.size() && ((p[w] >> (k & 63)) & 1ull);
}
static inline void polySetBit(Poly& p, size_t k) { p[k >> 6] |= (1ull << (k & 63)); }

// Characteristic polynomial of the MT19937 recurrence, computed with Berlekamp-Massey
// on one bit of the generator's output sequence.
struct MTJump {
    Poly phi;      // characteristic polynomial
    size_t deg;    // degree of phi (19937)
    size_t words;  // words needed for polynomials of degree < deg

    void init() {
        const size_t nbits = 2 * 19937 + 64;
        // Sequence bits: lowest bit of the raw recurrence output.
        MT19937 g;
        g.seed(42);
        std::vector<uint8_t> s(nbits);
        for (size_t i = 0; i < nbits; ++i) s[i] = static_cast<uint8_t>(g.stepRaw() & 1u);

        // Bit-packed reversed sequence: R bit (nbits-1-i) = s[i]
        const size_t rw = (nbits + 63) / 64 + 1;
        std::vector<uint64_t> R(rw, 0);
        for (size_t i = 0; i < nbits; ++i) {
            size_t pos = nbits - 1 - i;
            if (s[i]) R[pos >> 6] |= (1ull << (pos & 63));
        }

        // Berlekamp-Massey over GF(2) with bit-packed connection polynomials
        const size_t pw = rw;
        Poly C(pw, 0), B(pw, 0), T(pw, 0);
        C[0] = 1; B[0] = 1;
        size_t L = 0, m = 1;
        for (size_t n = 0; n < nbits; ++n) {
            // d = sum_{k=0..L} C[k] * s[n-k]   (C[0]=1 contributes s[n])
            size_t off = nbits - 1 - n;      // R bit index of s[n]; s[n-k] is at off + k
            size_t ow = off >> 6, ob = off & 63;
            size_t lw = (L >> 6) + 1;
            uint64_t acc = 0;
            for (size_t w = 0; w < lw; ++w) {
                uint64_t lo = (ow + w < rw) ? R[ow + w] : 0;
                uint64_t hi = (ow + w + 1 < rw) ? R[ow + w + 1] : 0;
                uint64_t win = ob ? ((lo >> ob) | (hi << (64 - ob))) : lo;
                acc ^= win & C[w];
            }
            // Mask bits beyond L
            // (C has no bits beyond L by construction, so no masking needed)
            int d = __builtin_parityll(acc);
            if (d) {
                T = C;
                // C ^= B << m
                size_t sw = m >> 6, sb = m & 63;
                for (size_t w = pw; w-- > sw;) {
                    uint64_t v = B[w - sw] << sb;
                    if (sb && w - sw >= 1) v |= B[w - sw - 1] >> (64 - sb);
                    C[w] ^= v;
                }
                if (2 * L <= n) {
                    L = n + 1 - L;
                    B = T;
                    m = 1;
                } else {
                    ++m;
                }
            } else {
                ++m;
            }
        }
        if (L != 19937) fatal("MT19937 characteristic polynomial has unexpected degree");
        deg = L;
        words = (deg + 63) / 64 + 1;
        // phi(x) = x^L * C(1/x)  (reciprocal polynomial)
        phi.assign(words, 0);
        for (size_t k = 0; k <= L; ++k) {
            if (polyBit(C, k)) polySetBit(phi, L - k);
        }
        if (!polyBit(phi, 0)) fatal("MT19937 characteristic polynomial has zero constant term");
    }

    // r = (a * a) mod phi
    void squareMod(const Poly& a, Poly& r, Poly& tmp) const {
        tmp.assign(2 * words, 0);
        for (size_t k = 0; k < deg; ++k) {
            if (polyBit(a, k)) polySetBit(tmp, 2 * k);
        }
        reduce(tmp, 2 * deg);
        r.assign(words, 0);
        for (size_t w = 0; w < words; ++w) r[w] = tmp[w];
    }

    // a = (a * x) mod phi
    void mulXMod(Poly& a) const {
        uint64_t carry = 0;
        for (size_t w = 0; w < words; ++w) {
            uint64_t nc = a[w] >> 63;
            a[w] = (a[w] << 1) | carry;
            carry = nc;
        }
        if (polyBit(a, deg)) {
            for (size_t w = 0; w < words; ++w) a[w] ^= phi[w];
        }
    }

    // Reduce tmp (degree < maxDeg+1) modulo phi in place
    void reduce(Poly& tmp, size_t maxDeg) const {
        for (size_t k = maxDeg; k >= deg; --k) {
            if (polyBit(tmp, k)) {
                size_t shift = k - deg;
                size_t sw = shift >> 6, sb = shift & 63;
                for (size_t w = 0; w < words; ++w) {
                    uint64_t v = phi[w] << sb;
                    if (sb && w >= 1) v |= phi[w - 1] >> (64 - sb);
                    tmp[w + sw] ^= v;
                }
                if (sb) tmp[words + sw] ^= phi[words - 1] >> (64 - sb);
            }
            if (k == deg) break;
        }
    }

    // p = x^J mod phi
    Poly jumpPoly(uint64_t J) const {
        Poly res(words, 0), tmp;
        res[0] = 1;
        int top = 63;
        while (top >= 0 && !((J >> top) & 1ull)) --top;
        Poly sq;
        for (int b = top; b >= 0; --b) {
            squareMod(res, sq, tmp);
            res.swap(sq);
            if ((J >> b) & 1ull) mulXMod(res);
        }
        return res;
    }

    // Advance generator g by J steps.
    void jump(MT19937& g, uint64_t J) const {
        if (J == 0) return;
        Poly p = jumpPoly(J);
        uint32_t s[MT_N];
        g.exportWindow(s);
        MT19937 r;  // zero state
        for (size_t k = deg; ; --k) {
            r.stepRaw();
            if (polyBit(p, k)) r.xorWindow(s);
            if (k == 0) break;
        }
        g = r;
    }
};

// Verify that the re-implemented generator is bit-compatible with the standard library
// and that the jump-ahead is correct.
static void selfTestRng(const MTJump& jumper) {
    std::mt19937 ref(42);
    std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
    MT19937 mine;
    mine.seed(42);
    for (int i = 0; i < 200000; ++i) {
        val_t a = dist(ref);
        val_t b = mine.nextFloat();
        if (a != b) fatal("MT19937 re-implementation does not match std::mt19937");
    }
    const uint64_t J = 1000003;
    MT19937 g1, g2;
    g1.seed(42); g2.seed(42);
    g1.discard(J);
    jumper.jump(g2, J);
    for (int i = 0; i < 5000; ++i) {
        if (g1.next() != g2.next()) fatal("MT19937 jump-ahead self-test failed");
    }
}

// ============================================================================
// Device geometry helpers (Möller-Trumbore, visibility, form factor terms)
// ============================================================================

// Generate a point inside a triangle using barycentric coordinates from two uniform draws
__device__ inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Check if a segment intersects a node's bounding box
__device__ inline bool rayIntersectsBox(const Vec3& center, const Vec3& h, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

constexpr int OCTREE_STACK = 64;

// 16-byte vector loads of triangle vertices (Triangle is 48 bytes: a, b, c, normal)
__device__ __forceinline__ void loadTriangleVertices(const Triangle* __restrict__ tris, int idx,
                                                     Vec3& a, Vec3& b, Vec3& c) {
    const float4* p = reinterpret_cast<const float4*>(tris + idx);
    const float4 q0 = __ldg(p + 0);
    const float4 q1 = __ldg(p + 1);
    const float4 q2 = __ldg(p + 2);
    a = Vec3(q0.x, q0.y, q0.z);
    b = Vec3(q0.w, q1.x, q1.y);
    c = Vec3(q1.z, q1.w, q2.x);
}

// 16-byte vector loads of a node's bounding box (GNode is 64 bytes: center, halfExtent, ...)
__device__ __forceinline__ void loadNodeBox(const GNode* __restrict__ nodes, int nid,
                                            Vec3& center, Vec3& halfExtent) {
    const float4* p = reinterpret_cast<const float4*>(nodes + nid);
    const float4 q0 = __ldg(p + 0);
    const float4 q1 = __ldg(p + 1);
    center = Vec3(q0.x, q0.y, q0.z);
    halfExtent = Vec3(q0.w, q1.x, q1.y);
}

static_assert(sizeof(Triangle) == 48, "Triangle layout must be 48 bytes for vector loads");
static_assert(sizeof(GNode) == 64, "GNode layout must be 64 bytes for vector loads");

// Check if the segment between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const GNode* __restrict__ nodes,
                             const int* __restrict__ leafTris,
                             const Triangle* __restrict__ tris,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const int nid = stack[--sp];
        const int4* np = reinterpret_cast<const int4*>(nodes + nid);
        const int4 q3 = __ldg(np + 3);     // (child6, child7, triStart, triCount)
        const int triStart = q3.z;
        const int triCount = q3.w;
        if (triCount > 0) {
            for (int k = 0; k < triCount; ++k) {
                const int idx = __ldg(leafTris + triStart + k);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                Vec3 a, b, c;
                loadTriangleVertices(tris, idx, a, b, c);
                val_t dist = rayTriangleIntersect(from, dirNorm, a, b, c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            const int4 q1 = __ldg(np + 1);  // (halfExtent.y, halfExtent.z, child0, child1)
            const int4 q2 = __ldg(np + 2);  // (child2, child3, child4, child5)
            const int children[8] = {q1.z, q1.w, q2.x, q2.y, q2.z, q2.w, q3.x, q3.y};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int ch = children[c];
                if (ch >= 0) {
                    Vec3 center, halfExtent;
                    loadNodeBox(nodes, ch, center, halfExtent);
                    if (rayIntersectsBox(center, halfExtent, from, to) && sp < OCTREE_STACK) {
                        stack[sp++] = ch;
                    }
                }
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return maxVal(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// CUDA kernels
// ============================================================================

constexpr int GEN_WARPS = 8;   // rows (warps) per block in the stream generation kernel

// One warp per receiver row: regenerate the row's MT19937 sub-stream as floats.
// states: per-row windows in sequence order (MT_N words each); draws: words to generate.
__device__ inline void twistPhase(uint32_t* mt, int lo, int hi, int lane) {
    uint32_t v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int i = lo + lane + 32 * k;
        if (i < hi) {
            int i1 = i + 1; if (i1 == MT_N) i1 = 0;
            int im = i + MT_M; if (im >= MT_N) im -= MT_N;
            uint32_t y = (mt[i] & MT_UPPER) | (mt[i1] & MT_LOWER);
            v[k] = mt[im] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int i = lo + lane + 32 * k;
        if (i < hi) mt[i] = v[k];
    }
    __syncwarp();
}

__global__ void __launch_bounds__(GEN_WARPS * 32)
genStreamKernel(const uint32_t* __restrict__ states, const int* __restrict__ draws,
                float* __restrict__ rowbuf, size_t rowStride, int nRows) {
    __shared__ uint32_t smt[GEN_WARPS][MT_N];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEN_WARPS + warp;
    if (row >= nRows) return;

    uint32_t* mt = smt[warp];
    for (int k = lane; k < MT_N; k += 32) mt[k] = states[static_cast<size_t>(row) * MT_N + k];
    __syncwarp();

    const int total = draws[row];
    float* out = rowbuf + static_cast<size_t>(row) * rowStride;
    for (int produced = 0; produced < total; produced += MT_N) {
        // Dependency-ordered twist: phases chosen so every input is already final.
        twistPhase(mt, 0, MT_N - MT_M, lane);              // [0, 227)
        twistPhase(mt, MT_N - MT_M, 2 * (MT_N - MT_M), lane);  // [227, 454)
        twistPhase(mt, 2 * (MT_N - MT_M), MT_N - 1, lane); // [454, 623)
        twistPhase(mt, MT_N - 1, MT_N, lane);              // [623, 624)
        for (int k = lane; k < MT_N; k += 32) {
            int p = produced + k;
            if (p < total) out[p] = mtCanonical(mtTemper(mt[k]));
        }
        __syncwarp();
    }
}

// Form factor kernel: 16 lanes (one per ray) per (i, j) pair.
// slots[r*N + j] gives the index of the pair among the row's non-culled pairs, or -1.
__global__ void __launch_bounds__(256)
formFactorKernel(int N, int rowBase, int nRows, int nLocal, int localRowBase,
                 const int* __restrict__ slots, const float* __restrict__ rowbuf, size_t rowStride,
                 const Triangle* __restrict__ tris, const GNode* __restrict__ nodes,
                 const int* __restrict__ leafTris, float* __restrict__ kijT) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairLocal = tid >> 4;
    const int ray = static_cast<int>(tid & 15);
    const size_t totalPairs = static_cast<size_t>(nRows) * N;
    if (pairLocal >= totalPairs) return;   // whole warps exit together (N*16 is a multiple of 32)

    const int r = static_cast<int>(pairLocal / N);
    const int j = static_cast<int>(pairLocal - static_cast<size_t>(r) * N);
    const int i = rowBase + r;
    const int slot = slots[static_cast<size_t>(r) * N + j];

    val_t contrib = ZERO;
    if (slot >= 0) {
        const float* rnd = rowbuf + static_cast<size_t>(r) * rowStride +
                           static_cast<size_t>(slot) * DRAWS_PER_PAIR + ray * 4;
        const val_t uI = rnd[0], vI = rnd[1], uJ = rnd[2], vJ = rnd[3];
        const Triangle& triI = tris[i];
        const Triangle& triJ = tris[j];
        Vec3 pI = pointInTriangle(triI, uI, vI);
        Vec3 pJ = pointInTriangle(triJ, uJ, vJ);

        if (!isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) {
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
    }

    // Sequential accumulation in ray order (same order as the original loop)
    const int base = threadIdx.x & 16;
    val_t kij = ZERO;
#pragma unroll
    for (int k = 0; k < NUM_RAYS; ++k) {
        val_t c = __shfl_sync(0xffffffffu, contrib, base + k);
        kij += c;
    }
    if (ray == 0) {
        const int il = (i - localRowBase);
        kijT[static_cast<size_t>(j) * nLocal + il] = (slot >= 0) ? kij * INV_NUM_RAYS : ZERO;
    }
}

// Wave propagation for one timestep: one thread per local receiver row, accumulating in the
// same j order as the sequential code. kijT / tauT are stored column-major over the local
// rows ([j][il]) so that loads coalesce across threads; loads are issued in groups of
// SIM_UNROLL independent columns to expose memory-level parallelism.
constexpr int SIM_UNROLL = 16;

__global__ void __launch_bounds__(128)
simulationStepKernel(int t, int N, int nLocal, int rowStart,
                     const float* __restrict__ kijT, const uint8_t* __restrict__ tauT,
                     const float* __restrict__ areas, const float* __restrict__ rho,
                     const float* __restrict__ radE, float* radB) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= nLocal) return;
    const int i = rowStart + il;

    val_t sumB = ZERO;
    for (int j0 = 0; j0 < N; j0 += SIM_UNROLL) {
        int tauv[SIM_UNROLL];
        val_t kv[SIM_UNROLL];
        val_t rv[SIM_UNROLL];
#pragma unroll
        for (int k = 0; k < SIM_UNROLL; ++k) {
            const int j = j0 + k;
            const size_t p = static_cast<size_t>(j) * nLocal + il;
            tauv[k] = (j < N) ? static_cast<int>(__ldg(tauT + p)) : (t + 1);
        }
#pragma unroll
        for (int k = 0; k < SIM_UNROLL; ++k) {
            const int j = j0 + k;
            const size_t p = static_cast<size_t>(j) * nLocal + il;
            const bool arrived = (t >= tauv[k]) && (j != i);
            kv[k] = arrived ? __ldg(kijT + p) : ZERO;
        }
#pragma unroll
        for (int k = 0; k < SIM_UNROLL; ++k) {
            const int j = j0 + k;
            const bool active = (kv[k] > ZERO);
            // tau >= 1 for every pair, so only rows written by earlier timesteps are read
            rv[k] = active ? __ldg(radB + static_cast<size_t>(t - tauv[k]) * N + j) : ZERO;
        }
#pragma unroll
        for (int k = 0; k < SIM_UNROLL; ++k) {
            const int j = j0 + k;
            if (rv[k] > ZERO) {
                sumB += minVal(kv[k] * __ldg(areas + j), ONE) * rv[k];
            }
        }
    }
    radB[static_cast<size_t>(t) * N + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

// Cross-correlation: corr[t][il] = sum_tt radB[tt][i] * radB[tt - t][src]
__global__ void correlationKernel(int N, int T, int nLocal, int rowStart, int src,
                                  const float* __restrict__ radB, float* __restrict__ corr) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (il >= nLocal || t >= T) return;
    const int i = rowStart + il;
    val_t sum = ZERO;
    for (int tt = t; tt < T; ++tt) {
        val_t pB = radB[static_cast<size_t>(tt) * N + i];
        val_t pS = radB[static_cast<size_t>(tt - t) * N + src];
        sum += pS * pB;
    }
    corr[static_cast<size_t>(t) * nLocal + il] = sum;
}

__global__ void argmaxDistanceKernel(int T, int nLocal, const float* __restrict__ corr,
                                     float* __restrict__ dist) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= nLocal) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        val_t s = corr[static_cast<size_t>(t) * nLocal + il];
        if (s > maxCorr) {
            maxCorr = s;
            bestT = t;
        }
    }
    dist[il] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countNonZeroKernel(const float* __restrict__ kij, size_t n,
                                   unsigned long long* __restrict__ out) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long c = 0;
    for (size_t k = tid; k < n; k += stride) {
        if (kij[k] > EPSILON) ++c;
    }
    if (c) atomicAdd(out, c);
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// Visibility culling test shared by all host code paths (kept out of line so that the
// floating-point evaluation is identical wherever it is used).
__attribute__((noinline)) static bool pairCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated on all ranks)
    std::vector<val_t> distances;   // Computed distances from source (all triangles)

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;          // GPU representation

    size_t sourceIndex;

    // MPI row decomposition
    int rowStart = 0;
    int nLocal = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    Triangle* d_tris = nullptr;
    GNode* d_nodes = nullptr;
    int* d_leafTris = nullptr;
    float* d_kijT = nullptr;        // [N][nLocal] column-major form factors
    uint8_t* d_tauT = nullptr;      // [N][nLocal] column-major time delays
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radE = nullptr;        // [T][N]
    float* d_radB = nullptr;        // [T][N]

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

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    state.octree.build(state.triangles);
    state.flatOctree.build(state.octree);

    const int N = static_cast<int>(state.numTriangles);

    // Row decomposition across MPI ranks
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        int s = static_cast<int>((static_cast<long long>(N) * r) / g_size);
        int e = static_cast<int>((static_cast<long long>(N) * (r + 1)) / g_size);
        state.rowDispls[r] = s;
        state.rowCounts[r] = e - s;
    }
    state.rowStart = state.rowDispls[g_rank];
    state.nLocal = state.rowCounts[g_rank];

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload static data to the GPU
    const size_t nLocal = static_cast<size_t>(state.nLocal);
    CUDA_CHECK(cudaMalloc(&state.d_tris, sizeof(Triangle) * state.numTriangles));
    CUDA_CHECK(cudaMemcpy(state.d_tris, state.triangles.data(),
                          sizeof(Triangle) * state.numTriangles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, sizeof(GNode) * state.flatOctree.nodes.size()));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, state.flatOctree.nodes.data(),
                          sizeof(GNode) * state.flatOctree.nodes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_leafTris, sizeof(int) * std::max<size_t>(1, state.flatOctree.leafTris.size())));
    CUDA_CHECK(cudaMemcpy(state.d_leafTris, state.flatOctree.leafTris.data(),
                          sizeof(int) * state.flatOctree.leafTris.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_areas, sizeof(float) * state.numTriangles));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), sizeof(float) * state.numTriangles,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rho, sizeof(float) * state.numTriangles));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), sizeof(float) * state.numTriangles,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_radE, sizeof(float) * state.radE.size()));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), sizeof(float) * state.radE.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_radB, sizeof(float) * state.radB.size()));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, sizeof(float) * state.radB.size()));

    const size_t localPairs = std::max<size_t>(1, nLocal * state.numTriangles);
    CUDA_CHECK(cudaMalloc(&state.d_kijT, sizeof(float) * localPairs));
    CUDA_CHECK(cudaMemset(state.d_kijT, 0, sizeof(float) * localPairs));
    CUDA_CHECK(cudaMalloc(&state.d_tauT, sizeof(uint8_t) * localPairs));
    CUDA_CHECK(cudaMemset(state.d_tauT, 0, sizeof(uint8_t) * localPairs));
}

void freeSimulation(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodes);
    cudaFree(state.d_leafTris);
    cudaFree(state.d_kijT);
    cudaFree(state.d_tauT);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, const MTJump& jumper) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int nLocal = state.nLocal;
    const int rowStart = state.rowStart;
    const std::vector<Triangle>& tris = state.triangles;

    // --- 1. Random draws consumed by each local row (the original consumes DRAWS_PER_PAIR
    //        draws per non-culled off-diagonal pair, in row-major pair order) ---
    std::vector<int> draws(nLocal, 0);
    #pragma omp parallel for schedule(static)
    for (int il = 0; il < nLocal; ++il) {
        const int i = rowStart + il;
        int cnt = 0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            if (!pairCulled(tris[i], tris[j])) ++cnt;
        }
        draws[il] = cnt * DRAWS_PER_PAIR;
    }
    std::vector<uint64_t> rowOffset(nLocal + 1, 0);
    for (int il = 0; il < nLocal; ++il) rowOffset[il + 1] = rowOffset[il] + draws[il];

    uint64_t localTotal = rowOffset[nLocal];
    uint64_t rankStart = 0;
    MPI_Exscan(&localTotal, &rankStart, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) rankStart = 0;

    // --- 2. Generator checkpoint (window) at the start of each local row ---
    std::vector<uint32_t> rowStates(static_cast<size_t>(std::max(nLocal, 1)) * MT_N);
    #pragma omp parallel
    {
        const int nth = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int chunkBegin = static_cast<int>((static_cast<long long>(nLocal) * tid) / nth);
        const int chunkEnd = static_cast<int>((static_cast<long long>(nLocal) * (tid + 1)) / nth);
        if (chunkBegin < chunkEnd) {
            MT19937 g;
            g.seed(42);
            jumper.jump(g, rankStart + rowOffset[chunkBegin]);
            for (int il = chunkBegin; il < chunkEnd; ++il) {
                g.exportWindow(&rowStates[static_cast<size_t>(il) * MT_N]);
                g.discard(static_cast<uint64_t>(draws[il]));
            }
        }
    }

    if (nLocal == 0) return;

    // --- 3. Batched GPU processing with double buffering ---
    const size_t rowStride = static_cast<size_t>(N) * DRAWS_PER_PAIR;   // floats per row buffer
    const size_t rowBytes = rowStride * sizeof(float);
    const size_t budgetBytes = static_cast<size_t>(256) << 20;          // per buffer
    int batchRows = static_cast<int>(std::max<size_t>(1, budgetBytes / rowBytes));
    batchRows = std::min(batchRows, nLocal);

    cudaStream_t streams[2];
    uint32_t* d_states[2];
    int* d_draws[2];
    int* d_slots[2];
    float* d_rowbuf[2];
    uint32_t* h_states[2];
    int* h_draws[2];
    int* h_slots[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[b], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d_states[b], sizeof(uint32_t) * MT_N * batchRows));
        CUDA_CHECK(cudaMalloc(&d_draws[b], sizeof(int) * batchRows));
        CUDA_CHECK(cudaMalloc(&d_slots[b], sizeof(int) * static_cast<size_t>(N) * batchRows));
        CUDA_CHECK(cudaMalloc(&d_rowbuf[b], rowBytes * batchRows));
        CUDA_CHECK(cudaMallocHost(&h_states[b], sizeof(uint32_t) * MT_N * batchRows));
        CUDA_CHECK(cudaMallocHost(&h_draws[b], sizeof(int) * batchRows));
        CUDA_CHECK(cudaMallocHost(&h_slots[b], sizeof(int) * static_cast<size_t>(N) * batchRows));
    }

    int nextProgress = 0;
    for (int r0 = 0, b = 0; r0 < nLocal; r0 += batchRows, ++b) {
        const int buf = b & 1;
        const int nRows = std::min(batchRows, nLocal - r0);
        CUDA_CHECK(cudaStreamSynchronize(streams[buf]));   // buffers free for reuse

        // Host preparation (OpenMP): ray slot table and generator checkpoints
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < nRows; ++r) {
            const int i = rowStart + r0 + r;
            int* srow = h_slots[buf] + static_cast<size_t>(r) * N;
            int s = 0;
            for (int j = 0; j < N; ++j) {
                if (j == i || pairCulled(tris[i], tris[j])) {
                    srow[j] = -1;
                } else {
                    srow[j] = s++;
                }
            }
            h_draws[buf][r] = draws[r0 + r];
            std::memcpy(h_states[buf] + static_cast<size_t>(r) * MT_N,
                        &rowStates[static_cast<size_t>(r0 + r) * MT_N], sizeof(uint32_t) * MT_N);
        }

        CUDA_CHECK(cudaMemcpyAsync(d_states[buf], h_states[buf], sizeof(uint32_t) * MT_N * nRows,
                                   cudaMemcpyHostToDevice, streams[buf]));
        CUDA_CHECK(cudaMemcpyAsync(d_draws[buf], h_draws[buf], sizeof(int) * nRows,
                                   cudaMemcpyHostToDevice, streams[buf]));
        CUDA_CHECK(cudaMemcpyAsync(d_slots[buf], h_slots[buf], sizeof(int) * static_cast<size_t>(N) * nRows,
                                   cudaMemcpyHostToDevice, streams[buf]));

        const int genBlocks = (nRows + GEN_WARPS - 1) / GEN_WARPS;
        genStreamKernel<<<genBlocks, GEN_WARPS * 32, 0, streams[buf]>>>(
            d_states[buf], d_draws[buf], d_rowbuf[buf], rowStride, nRows);
        CUDA_CHECK(cudaGetLastError());

        const size_t threads = static_cast<size_t>(nRows) * N * NUM_RAYS;
        const unsigned blocks = static_cast<unsigned>((threads + 255) / 256);
        formFactorKernel<<<blocks, 256, 0, streams[buf]>>>(
            N, rowStart + r0, nRows, nLocal, rowStart, d_slots[buf], d_rowbuf[buf], rowStride,
            state.d_tris, state.d_nodes, state.d_leafTris, state.d_kijT);
        CUDA_CHECK(cudaGetLastError());

        const int done = r0 + nRows;
        if (done >= nextProgress + 100 || done == nLocal) {
            ROOT_PRINTF("  Progress: %d/%d triangles (rank 0 rows)\n", done, nLocal);
            nextProgress = (done / 100) * 100;
        }
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(streams[b]));
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFree(d_states[b]));
        CUDA_CHECK(cudaFree(d_draws[b]));
        CUDA_CHECK(cudaFree(d_slots[b]));
        CUDA_CHECK(cudaFree(d_rowbuf[b]));
        CUDA_CHECK(cudaFreeHost(h_states[b]));
        CUDA_CHECK(cudaFreeHost(h_draws[b]));
        CUDA_CHECK(cudaFreeHost(h_slots[b]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int nLocal = state.nLocal;
    const int rowStart = state.rowStart;
    if (nLocal == 0) return;

    // Delays are bounded by the room diameter (2 * radius / WAVE_SPEED = 40), so they are
    // stored as 8-bit values on the GPU to halve the memory traffic of the propagation kernel.
    std::vector<uint8_t> tauT(static_cast<size_t>(N) * nLocal, 0);
    int maxTau = 0;
    int minTau = std::numeric_limits<int>::max();
    #pragma omp parallel for schedule(static) reduction(max : maxTau) reduction(min : minTau)
    for (int j = 0; j < N; ++j) {
        uint8_t* col = tauT.data() + static_cast<size_t>(j) * nLocal;
        for (int il = 0; il < nLocal; ++il) {
            const int i = rowStart + il;
            if (i == j) continue;
            int tau = computeTau(state.triangles[i], state.triangles[j]);
            maxTau = std::max(maxTau, tau);
            minTau = std::min(minTau, tau);
            col[il] = static_cast<uint8_t>(tau);
        }
    }
    if (maxTau > 255 || minTau < 0) fatal("time delay does not fit in 8 bits");

    CUDA_CHECK(cudaMemcpy(state.d_tauT, tauT.data(), sizeof(uint8_t) * tauT.size(),
                          cudaMemcpyHostToDevice));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int nLocal = state.nLocal;
    const int rowStart = state.rowStart;

    float* h_row = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_row, sizeof(float) * N));

    const int blockSize = 128;
    const int blocks = (nLocal + blockSize - 1) / blockSize;

    for (int t = 0; t < T; ++t) {
        float* radBt = state.d_radB + static_cast<size_t>(t) * N;
        if (nLocal > 0) {
            simulationStepKernel<<<blocks, blockSize>>>(
                t, N, nLocal, rowStart, state.d_kijT, state.d_tauT, state.d_areas, state.d_rho,
                state.d_radE, state.d_radB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_row + rowStart, radBt + rowStart, sizeof(float) * nLocal,
                                  cudaMemcpyDeviceToHost));
        }
        // Exchange this timestep's radiosity row with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_row, state.rowCounts.data(),
                       state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        std::memcpy(&state.radB[state.idxTN(t, 0)], h_row, sizeof(float) * N);
        CUDA_CHECK(cudaMemcpy(radBt, h_row, sizeof(float) * N, cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            ROOT_PRINTF("  Timestep %d/%d\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaFreeHost(h_row));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int nLocal = state.nLocal;
    const int rowStart = state.rowStart;
    const int src = static_cast<int>(state.sourceIndex);

    std::vector<float> localDist(std::max(nLocal, 1), ZERO);
    if (nLocal > 0 && T > 0) {
        float* d_corr = nullptr;
        float* d_dist = nullptr;
        CUDA_CHECK(cudaMalloc(&d_corr, sizeof(float) * static_cast<size_t>(T) * nLocal));
        CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * nLocal));

        const int blockSize = 128;
        dim3 grid((nLocal + blockSize - 1) / blockSize, T);
        correlationKernel<<<grid, blockSize>>>(N, T, nLocal, rowStart, src, state.d_radB, d_corr);
        CUDA_CHECK(cudaGetLastError());
        argmaxDistanceKernel<<<(nLocal + blockSize - 1) / blockSize, blockSize>>>(T, nLocal, d_corr, d_dist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDist.data(), d_dist, sizeof(float) * nLocal, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_corr));
        CUDA_CHECK(cudaFree(d_dist));
    }

    MPI_Allgatherv(localDist.data(), nLocal, MPI_FLOAT, state.distances.data(),
                   state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// Number of form factors above EPSILON across all ranks (for validation)
long long countNonZeroFormFactors(const SimulationState& state) {
    unsigned long long local = 0;
    const size_t n = static_cast<size_t>(state.nLocal) * state.numTriangles;
    if (n > 0) {
        unsigned long long* d_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(4096, (n + 255) / 256));
        countNonZeroKernel<<<blocks, 256>>>(state.d_kijT, n, d_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&local, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_count));
    }
    unsigned long long total = 0;
    MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<long long>(total);
}

// ============================================================================
// Validation (rank 0; radB and distances are replicated there)
// ============================================================================

bool validateResults(const SimulationState& state, long long nonZeroKij) {
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
    const int N = static_cast<int>(state.numTriangles);
    #pragma omp parallel for schedule(static) reduction(+ : receivedEnergy)
    for (int i = 0; i < N; ++i) {
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

static void selectDevice() {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) fatal("no CUDA devices available");
    CUDA_CHECK(cudaSetDevice(localRank % devCount));
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
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    selectDevice();

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("Parallelization: %d MPI rank(s) x %d OpenMP thread(s) + CUDA\n",
                g_size, omp_get_max_threads());
    ROOT_PRINTF("\n");

    // Random stream reproduction machinery (characteristic polynomial + self test)
    MTJump jumper;
    auto startRng = std::chrono::high_resolution_clock::now();
    jumper.init();
    long rngDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - startRng).count();
    selfTestRng(jumper);

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state, jumper);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = rngDuration + preDuration + simDuration + distDuration;
    MPI_Allreduce(MPI_IN_PLACE, &totalTime, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", t);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");

    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        long long nonZeroKij = countNonZeroFormFactors(state);
        if (g_rank == 0 && !validateResults(state, nonZeroKij)) {
            status = 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    freeSimulation(state);
    MPI_Finalize();
    return status;
}

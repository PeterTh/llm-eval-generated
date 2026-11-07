/**
 * Room Response Simulation Benchmark -- hybrid MPI + OpenMP + CUDA version
 *
 * Room impulse response simulation using radiosity-based wave propagation.
 * It models how sound/light waves propagate between surfaces in a room by
 * computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * Parallelization strategy (semantics identical to the sequential reference):
 *
 *  - MPI:    triangles (rows of the Kij / Tau matrices) are block-distributed over
 *            the ranks.  Each rank owns one GPU and only ever stores its own rows.
 *            The per-timestep radiosity row is all-gathered, distances are gathered
 *            to rank 0 for output / validation.
 *  - CUDA:   form factors (ray casting against the flattened octree), time delays,
 *            wave propagation and the cross-correlation run as GPU kernels.
 *            Floating-point accumulation order of the reference is preserved
 *            (one thread per sequential sum).
 *  - OpenMP: host-side work: mesh statistics, validation reductions, and in
 *            particular MT19937 jump-ahead.  The reference draws all random numbers
 *            from ONE sequential mt19937(42).  To reproduce exactly that stream in
 *            parallel, every (i,j) pair's offset into the stream is computed from a
 *            prefix sum of the normal-culling mask (GPU block scan); the stream is
 *            cut into fixed-length substreams whose start states are computed by
 *            OpenMP workers with polynomial jump-ahead (characteristic polynomial via
 *            Berlekamp-Massey, applied with Horner's scheme; plain block stepping for
 *            small streams), and the substreams are then generated on the GPU, one
 *            warp per substream.  CPU state computation for chunk c+1 overlaps the
 *            GPU work of chunk c (double-buffered streams).
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <sched.h>

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

// Random draws consumed per (i,j) pair that is not culled by the normal test:
// NUM_RAYS rays * 2 points * 2 barycentric coordinates
constexpr uint64_t DRAWS_PER_PAIR = 4ull * NUM_RAYS;
constexpr uint32_t PAIR_CULLED = 0xFFFFFFFFu;

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

#define MPI_CHECK(call)                                                               \
    do {                                                                              \
        int err__ = (call);                                                           \
        if (err__ != MPI_SUCCESS) {                                                   \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,  \
                    __LINE__);                                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ============================================================================
// Vector and Triangle Types (shared between host and device)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD Vec3() : x(0), y(0), z(0) {}
    HD Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit Vec3(val_t v) : x(v), y(v), z(v) {}

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
// Octree for Spatial Acceleration (host-side construction)
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

// Flattened octree representation for the GPU
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];       // -1 if absent
    int leafStart;      // offset into leaf index array
    int leafCount;      // > 0 for leaves, 0 for internal nodes
};

constexpr int OCTREE_STACK = 128;

static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<uint32_t>& leafIdx, int depth, int& maxDepth) {
    int me = static_cast<int>(nodes.size());
    nodes.push_back(DevNode{});
    maxDepth = std::max(maxDepth, depth);

    DevNode dn{};
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    for (int i = 0; i < 8; ++i) dn.child[i] = -1;
    dn.leafStart = static_cast<int>(leafIdx.size());
    dn.leafCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) leafIdx.push_back(static_cast<uint32_t>(idx));

    if (node.triangleIndices.empty()) {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                dn.child[i] = flattenOctree(*node.children[i], nodes, leafIdx, depth + 1, maxDepth);
            }
        }
    }
    nodes[me] = dn;
    return me;
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
// Mersenne Twister (MT19937) -- bit-exact replica of std::mt19937 / libstdc++
// uniform_real_distribution<float>(0,1), with jump-ahead support
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_BITS = 19937;                 // effective state size in bits
constexpr int POLY_WORDS = (MT_BITS + 64) / 64; // 312 words hold degrees 0..19937

// Engine state as a circular window over the recurrence sequence.
struct MTState {
    uint32_t w[MT_N];
    int h;
};

// Advance the window by one step; returns the (untempered) word that leaves the window.
static inline uint32_t mtStep(MTState& s) {
    const uint32_t out = s.w[s.h];
    int h1 = s.h + 1; if (h1 == MT_N) h1 = 0;
    int hm = s.h + MT_M; if (hm >= MT_N) hm -= MT_N;
    const uint32_t y = (out & 0x80000000u) | (s.w[h1] & 0x7fffffffu);
    s.w[s.h] = s.w[hm] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    s.h = h1;
    return out;
}

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// libstdc++ std::uniform_real_distribution<float>(0,1) over mt19937:
// generate_canonical<float,24> takes exactly one 32-bit draw.
HD inline val_t mtToFloat(uint32_t x) {
    val_t r = static_cast<val_t>(x) / 4294967296.0f;
    if (r >= 1.0f) r = 0.99999994f;  // nextafter(1.0f, 0.0f)
    return r;
}

static MTState mtSeedState(uint32_t seed) {
    MTState s;
    s.w[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        s.w[i] = 1812433253u * (s.w[i - 1] ^ (s.w[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    s.h = 0;
    return s;
}

// State positioned at stream offset 0 (identical to std::mt19937(seed) right
// before its first output): the seed window advanced by one full regeneration.
static MTState mtBaseState(uint32_t seed) {
    MTState s = mtSeedState(seed);
    for (int i = 0; i < MT_N; ++i) mtStep(s);
    return s;
}

// Advance the state by n steps (full-block regeneration, ~1 ns per step).
static void mtAdvance(MTState& s, uint64_t n) {
    if (n >= static_cast<uint64_t>(MT_N)) {
        uint32_t tmp[MT_N];
        for (int i = 0; i < MT_N; ++i) {
            int k = s.h + i; if (k >= MT_N) k -= MT_N;
            tmp[i] = s.w[k];
        }
        std::memcpy(s.w, tmp, sizeof(tmp));
        s.h = 0;
        uint32_t* w = s.w;
        auto twist = [](uint32_t u, uint32_t v) {
            const uint32_t y = (u & 0x80000000u) | (v & 0x7fffffffu);
            return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        };
        while (n >= static_cast<uint64_t>(MT_N)) {
            for (int k = 0; k < MT_N - MT_M; ++k) w[k] = w[k + MT_M] ^ twist(w[k], w[k + 1]);
            for (int k = MT_N - MT_M; k < MT_N - 1; ++k) w[k] = w[k - (MT_N - MT_M)] ^ twist(w[k], w[k + 1]);
            w[MT_N - 1] = w[MT_M - 1] ^ twist(w[MT_N - 1], w[0]);
            n -= MT_N;
        }
    }
    while (n-- > 0) mtStep(s);
}

// --- GF(2) polynomial arithmetic modulo the MT characteristic polynomial ---

using Poly = std::vector<uint64_t>;

static inline bool polyBit(const uint64_t* p, int i) { return (p[i >> 6] >> (i & 63)) & 1u; }
static inline void polySetBit(uint64_t* p, int i) { p[i >> 6] |= (1ull << (i & 63)); }

// dst ^= src << shift   (src has srcWords words, dst has dstWords words)
static void xorShifted(uint64_t* __restrict__ dst, int dstWords,
                       const uint64_t* __restrict__ src, int srcWords, int shift) {
    const int wo = shift >> 6;
    const int bs = shift & 63;
    const int n = std::min(srcWords, dstWords - wo);
    if (n <= 0) return;
    uint64_t* d = dst + wo;
    if (bs == 0) {
        for (int w = 0; w < n; ++w) d[w] ^= src[w];
    } else {
        const int rs = 64 - bs;
        d[0] ^= src[0] << bs;
        for (int w = 1; w < n; ++w) d[w] ^= (src[w] << bs) | (src[w - 1] >> rs);
        if (wo + n < dstWords) d[n] ^= src[n - 1] >> rs;
    }
}

struct MTJump {
    Poly phi;                 // characteristic polynomial, degree MT_BITS
    std::vector<Poly> pow2;   // pow2[m] = x^(2^m) mod phi

    // Reduce a polynomial of up to 2*POLY_WORDS words modulo phi, in place.
    void reduce(uint64_t* tmp) const {
        for (int p = 2 * (MT_BITS - 1); p >= MT_BITS; --p) {
            if (polyBit(tmp, p)) xorShifted(tmp, 2 * POLY_WORDS, phi.data(), POLY_WORDS, p - MT_BITS);
        }
    }

    Poly mulmod(const Poly& a, const Poly& b) const {
        std::vector<uint64_t> tmp(2 * POLY_WORDS, 0);
        for (int i = 0; i < MT_BITS; ++i) {
            if (polyBit(a.data(), i)) xorShifted(tmp.data(), 2 * POLY_WORDS, b.data(), POLY_WORDS, i);
        }
        reduce(tmp.data());
        return Poly(tmp.begin(), tmp.begin() + POLY_WORDS);
    }

    Poly sqrmod(const Poly& a) const {
        std::vector<uint64_t> tmp(2 * POLY_WORDS, 0);
        for (int i = 0; i < MT_BITS; ++i) {
            if (polyBit(a.data(), i)) polySetBit(tmp.data(), 2 * i);
        }
        reduce(tmp.data());
        return Poly(tmp.begin(), tmp.begin() + POLY_WORDS);
    }

    // Ensure pow2[m] exists for all m <= maxM
    void ensurePow2(int maxM) {
        if (pow2.empty()) {
            Poly x(POLY_WORDS, 0);
            polySetBit(x.data(), 1);
            pow2.push_back(x);
        }
        while (static_cast<int>(pow2.size()) <= maxM) pow2.push_back(sqrmod(pow2.back()));
    }

    // x^e mod phi
    Poly power(uint64_t e) {
        int top = 0;
        while ((e >> top) > 1) ++top;
        ensurePow2(top);
        Poly r(POLY_WORDS, 0);
        polySetBit(r.data(), 0);  // 1
        bool first = true;
        for (int m = 0; m <= top; ++m) {
            if ((e >> m) & 1u) {
                if (first) { r = pow2[m]; first = false; }
                else r = mulmod(r, pow2[m]);
            }
        }
        return r;
    }

    // Apply g(A) to state s using Horner's scheme (A = one-step transition).
    // The accumulator is kept as a sliding window in a double-length buffer so the
    // XOR with s is a plain contiguous loop.
    static MTState apply(const Poly& g, const MTState& s) {
        uint32_t lin[MT_N];
        for (int i = 0; i < MT_N; ++i) {
            int k = s.h + i; if (k >= MT_N) k -= MT_N;
            lin[i] = s.w[k];
        }
        uint32_t buf[2 * MT_N];
        std::memset(buf, 0, sizeof(buf));
        int pos = 0;
        for (int k = MT_BITS - 1; k >= 0; --k) {
            uint32_t* w = buf + pos;
            const uint32_t y = (w[0] & 0x80000000u) | (w[1] & 0x7fffffffu);
            const uint32_t nw = w[MT_M] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
            if (pos + 1 + MT_N > 2 * MT_N) {
                std::memmove(buf, w + 1, (MT_N - 1) * sizeof(uint32_t));
                pos = 0;
                w = buf;
                w[MT_N - 1] = nw;
            } else {
                w[MT_N] = nw;
                ++pos;
                w = buf + pos;
            }
            if (polyBit(g.data(), k)) {
                for (int i = 0; i < MT_N; ++i) w[i] ^= lin[i];
            }
        }
        MTState r;
        std::memcpy(r.w, buf + pos, sizeof(r.w));
        r.h = 0;
        return r;
    }

    // Compute the characteristic polynomial of the recurrence with Berlekamp-Massey
    // on the least-significant-bit sequence of the recurrence words.
    void computeCharPoly() {
        const int NSEQ = 2 * MT_BITS + 128;
        const int RW = 2 * NSEQ / 64 + 8;
        std::vector<uint64_t> rseq(RW, 0);  // reversed sequence: bit t = s[NSEQ-1-t]
        {
            MTState s = mtSeedState(42);
            mtStep(s);  // x_0 is seed material, not produced by the recurrence: skip it
            for (int n = 0; n < NSEQ; ++n) {
                const uint32_t bit = mtStep(s) & 1u;
                if (bit) polySetBit(rseq.data(), NSEQ - 1 - n);
            }
        }
        const int CW = (NSEQ + 64) / 64 + 2;
        std::vector<uint64_t> C(CW, 0), B(CW, 0), T;
        C[0] = 1; B[0] = 1;
        int L = 0, m = 1;
        for (int n = 0; n < NSEQ; ++n) {
            const int off = NSEQ - 1 - n;
            const int wo = off >> 6, bs = off & 63;
            const int nw = L / 64 + 1;
            uint64_t acc = 0;
            for (int w = 0; w < nw; ++w) {
                uint64_t v = rseq[wo + w] >> bs;
                if (bs) v |= rseq[wo + w + 1] << (64 - bs);
                acc ^= C[w] & v;
            }
            const int d = __builtin_popcountll(acc) & 1;
            if (!d) {
                ++m;
            } else if (2 * L <= n) {
                T = C;
                xorShifted(C.data(), CW, B.data(), CW, m);
                L = n + 1 - L;
                B = T;
                m = 1;
            } else {
                xorShifted(C.data(), CW, B.data(), CW, m);
                ++m;
            }
        }
        if (L != MT_BITS) {
            fprintf(stderr, "MT19937 characteristic polynomial has unexpected degree %d\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Reciprocal: phi_k = C_{L-k}
        phi.assign(POLY_WORDS, 0);
        for (int k = 0; k <= L; ++k) {
            if (polyBit(C.data(), L - k)) polySetBit(phi.data(), k);
        }
    }

    // Verify the jump machinery against plain stepping
    void selfTest() {
        const uint64_t d = (1ull << 19) + 1;
        const Poly g = power(d);
        MTState base = mtBaseState(42);
        MTState jumped = apply(g, base);
        MTState stepped = base;
        for (uint64_t i = 0; i < d; ++i) mtStep(stepped);
        for (int i = 0; i < 16; ++i) {
            if (mtStep(jumped) != mtStep(stepped)) {
                fprintf(stderr, "MT19937 jump-ahead self test failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Substream start states.  The global mt19937(42) output stream is cut into
// substreams of L draws (L chosen at runtime); substream k starts at offset k*L.
// OpenMP worker p owns all substreams k == p (mod P) and walks them with a single
// precomputed jump polynomial x^(P*L), so no worker ever has to step through the
// stream serially.  Start states are handed to the GPU, which generates the draws.
// ----------------------------------------------------------------------------

class SubstreamStateProvider {
    struct Worker {
        MTState eng;
        int64_t posK = -1;     // substream index the engine is positioned at
    };

    // Streams below this size are walked with plain block stepping (a few ms);
    // above it the polynomial jump-ahead machinery (~30 ms setup) pays off.
    static constexpr uint64_t SEQUENTIAL_LIMIT = 1ull << 24;

    MTJump jump;
    MTState base;
    int P;
    int log2L;
    bool sequential;
    Poly stride;               // x^(P*L) mod phi
    std::vector<Worker> workers;

public:
    SubstreamStateProvider(int numWorkers, int log2SubstreamLen, uint64_t totalDraws)
        : base(mtBaseState(42)), P(std::max(1, numWorkers)), log2L(log2SubstreamLen),
          sequential(totalDraws <= SEQUENTIAL_LIMIT), workers(sequential ? 1 : P) {
        if (sequential) {
            workers[0].eng = base;
            workers[0].posK = 0;
            return;
        }
        jump.computeCharPoly();
        const uint64_t kMax = (totalDraws >> log2L) + 1;
        int bits = 0;
        while ((kMax >> bits) > 0) ++bits;
        jump.ensurePow2(log2L + bits + 1);
        jump.selfTest();
        stride = jump.power(static_cast<uint64_t>(P) << log2L);
    }

    // Start states of substreams [k0, k0 + K), substream-major: out[kk*MT_N + i]
    void fill(int64_t k0, int K, uint32_t* out) {
        if (sequential) {
            Worker& wk = workers[0];
            for (int64_t k = k0; k < k0 + K; ++k) {
                if (k < wk.posK) { wk.eng = base; wk.posK = 0; }
                mtAdvance(wk.eng, static_cast<uint64_t>(k - wk.posK) << log2L);
                wk.posK = k;
                uint32_t* dst = out + static_cast<size_t>(k - k0) * MT_N;
                for (int i = 0; i < MT_N; ++i) {
                    int idx = wk.eng.h + i; if (idx >= MT_N) idx -= MT_N;
                    dst[i] = wk.eng.w[idx];
                }
            }
            return;
        }

        #pragma omp parallel for schedule(static)
        for (int p = 0; p < P; ++p) {
            Worker& wk = workers[p];
            int64_t k = k0 + ((p - static_cast<int>(k0 % P)) + P) % P;
            for (; k < k0 + K; k += P) {
                if (wk.posK == k) {
                    // already positioned (chunk boundary inside this substream)
                } else if (wk.posK >= 0 && wk.posK + P == k) {
                    wk.eng = MTJump::apply(stride, wk.eng);
                } else {
                    wk.eng = base;
                    for (int m = 0; (k >> m) > 0; ++m) {
                        if ((k >> m) & 1) wk.eng = MTJump::apply(jump.pow2[log2L + m], wk.eng);
                    }
                }
                wk.posK = k;
                uint32_t* dst = out + static_cast<size_t>(k - k0) * MT_N;
                for (int i = 0; i < MT_N; ++i) {
                    int idx = wk.eng.h + i; if (idx >= MT_N) idx -= MT_N;
                    dst[i] = wk.eng.w[idx];
                }
            }
        }
    }
};

// ============================================================================
// Device code: ray casting, form factors, time delays, propagation, correlation
// ============================================================================

// Ray-Triangle Intersection (Moeller-Trumbore algorithm)
__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                       const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 3.402823466e+38f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e+38f;

    return e2.dot(qvec) * invDet;
}

// Check if a ray (segment p1->p2) intersects a node's bounding box
__device__ __forceinline__ bool rayIntersectsBox(const DevNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - nd.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3 he = nd.halfExtent;

    if (fabsf(c.x) > he.x + ad.x) return false;
    if (fabsf(c.y) > he.y + ad.y) return false;
    if (fabsf(c.z) > he.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
// (octree-accelerated, explicit traversal stack)
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevNode* __restrict__ nodes,
                             const uint32_t* __restrict__ leafIdx,
                             const float4* __restrict__ verts,
                             uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root is traversed without a box test (as in the reference)

    while (sp > 0) {
        const DevNode& nd = nodes[stack[--sp]];
        if (nd.leafCount > 0) {
            for (int k = 0; k < nd.leafCount; ++k) {
                const uint32_t idx = leafIdx[nd.leafStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const float4 a = verts[3 * idx], b = verts[3 * idx + 1], c = verts[3 * idx + 2];
                val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(a.x, a.y, a.z),
                                                  Vec3(b.x, b.y, b.z), Vec3(c.x, c.y, c.z));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;  // Ray is blocked
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int ci = nd.child[c];
                if (ci >= 0 && rayIntersectsBox(nodes[ci], from, to)) {
                    if (sp < OCTREE_STACK) stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // std::max(ZERO, c)
}

// Normal-based culling test (must be evaluated identically wherever it is used)
__device__ __forceinline__ bool pairCulled(const Vec3& ni, const Vec3& nj) {
    return ni.dot(nj) > 0.99f;
}

// Number of stream-consuming pairs per row (one block per row)
__global__ void rowDrawCountKernel(int N, int r0, const Triangle* __restrict__ tris,
                                   uint32_t* __restrict__ rowCount) {
    const int i = r0 + blockIdx.x;
    const Vec3 ni = tris[i].normal();
    unsigned cnt = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (j != i && !pairCulled(ni, tris[j].normal())) ++cnt;
    }
    __shared__ unsigned s[256];
    s[threadIdx.x] = cnt;
    __syncthreads();
    for (int d = blockDim.x / 2; d > 0; d >>= 1) {
        if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) rowCount[blockIdx.x] = s[0];
}

// Stream offset (relative to the chunk start D0) of every pair of a chunk of rows:
// block-wide exclusive scan of the culling mask along each row (one block per row).
__global__ void pairOffsetKernel(int c0, int N, const Triangle* __restrict__ tris,
                                 const uint64_t* __restrict__ rowDrawStart, uint64_t D0,
                                 uint32_t* __restrict__ pairOff) {   // [chunkRows][N]
    const int il = blockIdx.x;
    const int i = c0 + il;
    const Vec3 ni = tris[i].normal();
    uint32_t running = static_cast<uint32_t>(rowDrawStart[i] - D0);
    uint32_t* row = pairOff + static_cast<size_t>(il) * N;
    __shared__ uint32_t s[256];

    for (int j0 = 0; j0 < N; j0 += 256) {
        const int j = j0 + threadIdx.x;
        const uint32_t ok = (j < N && j != i && !pairCulled(ni, tris[j].normal())) ? 1u : 0u;
        s[threadIdx.x] = ok;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            const uint32_t v = (threadIdx.x >= static_cast<unsigned>(d)) ? s[threadIdx.x - d] : 0u;
            __syncthreads();
            s[threadIdx.x] += v;
            __syncthreads();
        }
        const uint32_t excl = s[threadIdx.x] - ok;
        if (j < N) row[j] = ok ? running + static_cast<uint32_t>(DRAWS_PER_PAIR) * excl : PAIR_CULLED;
        running += static_cast<uint32_t>(DRAWS_PER_PAIR) * s[255];
        __syncthreads();
    }
}

// Generate the tempered MT19937 stream for substreams [k0, k0 + K): one warp per
// substream.  The 624-word state lives in shared memory and is regenerated block-wise
// (the standard MT recurrence allows 227 words to be produced in parallel), outputs
// are written coalesced.  States are substream-major: states[kk*MT_N + i].
constexpr int MTGEN_WARPS = 8;
constexpr int MTGEN_BLOCK = 32 * MTGEN_WARPS;

__device__ __forceinline__ uint32_t mtTwist(uint32_t u, uint32_t v) {
    const uint32_t y = (u & 0x80000000u) | (v & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

__global__ void __launch_bounds__(MTGEN_BLOCK)
mtGenerateKernel(int K, int64_t k0, uint64_t L, uint64_t D0, uint64_t D1,
                 const uint32_t* __restrict__ states, uint32_t* __restrict__ out) {
    __shared__ uint32_t sw[MTGEN_WARPS][MT_N];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kk = blockIdx.x * MTGEN_WARPS + warp;
    if (kk >= K) return;
    uint32_t* w = sw[warp];

    const uint32_t* src = states + static_cast<size_t>(kk) * MT_N;
    for (int i = lane; i < MT_N; i += 32) w[i] = src[i];
    __syncwarp();

    uint64_t q = static_cast<uint64_t>(k0 + kk) * L;   // stream offset of w[0]
    const uint64_t qEnd = q + L;
    while (q < qEnd && q < D1) {
        // Emit this window
        const uint64_t cnt = (qEnd - q < MT_N) ? (qEnd - q) : MT_N;
        for (int i = lane; i < static_cast<int>(cnt); i += 32) {
            const uint64_t qq = q + i;
            if (qq >= D0 && qq < D1) out[qq - D0] = mtTemper(w[i]);
        }
        q += MT_N;
        if (q >= qEnd || q >= D1) break;

        // Regenerate.  All lanes run identical trip counts (warp-synchronous code).
        // Phase 1: words 0..226 depend on the old window only.
        constexpr int PH1 = MT_N - MT_M;                       // 227
        for (int it = 0; it < (PH1 + 31) / 32; ++it) {
            const int k = it * 32 + lane;
            const bool valid = k < PH1;
            const uint32_t nw = valid ? (w[k + MT_M] ^ mtTwist(w[k], w[k + 1])) : 0u;
            __syncwarp();
            if (valid) w[k] = nw;
            __syncwarp();
        }
        // Phase 2: words 227..622 depend on regenerated words k-227 (already final,
        // since k-227 <= k-32 lies in an earlier iteration) and old words k, k+1.
        constexpr int PH2 = MT_N - 1 - PH1;                    // 396
        for (int it = 0; it < (PH2 + 31) / 32; ++it) {
            const int k = PH1 + it * 32 + lane;
            const bool valid = k < MT_N - 1;
            const uint32_t nw = valid ? (w[k - PH1] ^ mtTwist(w[k], w[k + 1])) : 0u;
            __syncwarp();
            if (valid) w[k] = nw;
            __syncwarp();
        }
        // Last word wraps onto the regenerated w[0]
        if (lane == 0) w[MT_N - 1] = w[MT_M - 1] ^ mtTwist(w[MT_N - 1], w[0]);
        __syncwarp();
    }
}

// Form factor Kij for all pairs (i, j) of a chunk of rows.
// One lane per ray (NUM_RAYS lanes per pair): the rays of a pair are spatially
// coherent, which keeps the octree traversal of a warp mostly convergent.  The
// per-ray contributions are then summed in ray order through warp shuffles, so the
// accumulation is bit-identical to the sequential loop.  Pairs are enumerated local
// row fastest: the transposed output (kijT[j][i]) is written coalesced and triangle
// j is a warp-uniform load.
constexpr int FF_BLOCK = 128;   // 8 pairs per block
static_assert(FF_BLOCK % NUM_RAYS == 0 && 32 % NUM_RAYS == 0, "ray groups must not straddle warps");

__global__ void __launch_bounds__(FF_BLOCK)
formFactorKernel(int chunkRows, int rowBase, int rankRows, int N, int r0,
                 const Triangle* __restrict__ tris,
                 const float4* __restrict__ verts,
                 const DevNode* __restrict__ nodes,
                 const uint32_t* __restrict__ leafIdx,
                 const uint32_t* __restrict__ pairOff,   // [chunkRows][N]
                 const uint32_t* __restrict__ rng,
                 val_t* __restrict__ kijT) {
    const size_t gtid = static_cast<size_t>(blockIdx.x) * FF_BLOCK + threadIdx.x;
    const size_t pairIdx = gtid / NUM_RAYS;
    const int ray = threadIdx.x % NUM_RAYS;
    const bool valid = pairIdx < static_cast<size_t>(chunkRows) * N;

    int il = 0, j = 0, i = 0;
    bool active = false;
    uint32_t off = PAIR_CULLED;
    if (valid) {
        il = static_cast<int>(pairIdx % chunkRows);
        j = static_cast<int>(pairIdx / chunkRows);
        i = r0 + rowBase + il;
        off = pairOff[static_cast<size_t>(il) * N + j];
        active = (i != j && off != PAIR_CULLED);
    }

    val_t contrib = ZERO;
    if (active) {
        const Triangle triI = tris[i];
        const Triangle triJ = tris[j];
        const uint32_t* r = rng + off + 4 * ray;

        // Random point in triangle I (barycentric), then in triangle J
        val_t u = mtToFloat(r[0]);
        val_t v = mtToFloat(r[1]);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        const Vec3 pI = triI.a + (triI.b - triI.a) * u + (triI.c - triI.a) * v;

        u = mtToFloat(r[2]);
        v = mtToFloat(r[3]);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        const Vec3 pJ = triJ.a + (triJ.b - triJ.a) * u + (triJ.c - triJ.a) * v;

        if (!isRayBlocked(pI, pJ, nodes, leafIdx, verts, static_cast<uint32_t>(i),
                          static_cast<uint32_t>(j))) {
            const Vec3 d = pJ - pI;
            const val_t distSqr = d.squaredNorm();
            if (!(distSqr < EPSILON)) {
                const val_t cosPhiI = cosPhi(d, triI.normal());
                const val_t cosPhiJ = cosPhi(-d, triJ.normal());
                if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                    contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }
    }

    // Sequential (ray-ordered) accumulation; adding an exact zero for rays that were
    // skipped by the reference leaves the float sum unchanged.
    const int groupBase = (threadIdx.x & 31) & ~(NUM_RAYS - 1);
    val_t kij = ZERO;
    #pragma unroll
    for (int rr = 0; rr < NUM_RAYS; ++rr) {
        kij += __shfl_sync(0xFFFFFFFFu, contrib, groupBase + rr);
    }
    if (valid && ray == 0) {
        kijT[static_cast<size_t>(j) * rankRows + rowBase + il] = active ? kij * INV_NUM_RAYS : ZERO;
    }
}

// Time delays Tau (transposed layout tauT[j][i_local])
__global__ void timeDelayKernel(int rankRows, int N, int r0,
                                const Triangle* __restrict__ tris,
                                int* __restrict__ tauT) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<size_t>(rankRows) * N) return;
    const int il = static_cast<int>(tid % rankRows);
    const int j = static_cast<int>(tid / rankRows);
    const int i = r0 + il;
    int tau = 0;
    if (i != j) {
        val_t dist = (tris[i].center() - tris[j].center()).norm();
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[static_cast<size_t>(j) * rankRows + il] = tau;
}

// One timestep of wave propagation for the rows owned by this rank.
// One thread per row keeps the sequential accumulation order of the reference.
__global__ void propagationKernel(int rankRows, int N, int r0, int t, int sourceIndex, int timeOff,
                                  const val_t* __restrict__ kijT,
                                  const int* __restrict__ tauT,
                                  const val_t* __restrict__ areas,
                                  val_t rho,
                                  const val_t* __restrict__ radB,   // [T][N]
                                  val_t* __restrict__ radBRow) {    // [rankRows]
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= rankRows) return;
    const int i = r0 + il;

    val_t sumB = ZERO;
    #pragma unroll 8
    for (int j = 0; j < N; ++j) {
        const int tauij = tauT[static_cast<size_t>(j) * rankRows + il];
        const val_t kij = kijT[static_cast<size_t>(j) * rankRows + il];
        if (j == i) continue;
        if (t < tauij) continue;          // wave has not yet propagated from j to i
        if (kij <= ZERO) continue;
        const val_t radJ = radB[static_cast<size_t>(t - tauij) * N + j];
        if (radJ <= ZERO) continue;
        const val_t ka = kij * areas[j];
        const val_t m = (ONE < ka) ? ONE : ka;   // std::min(kij * area, ONE)
        sumB += m * radJ;
    }
    const val_t radE = (i == sourceIndex && t < timeOff) ? 1.0f : ZERO;
    radBRow[il] = rho * sumB + radE;
}

// Distance via discrete cross-correlation: one block per triangle, threads over lags.
__global__ void distanceKernel(int N, int T, int r0, int sourceIndex,
                               const val_t* __restrict__ radB,   // [T][N]
                               val_t* __restrict__ distances) {  // [rankRows]
    const int il = blockIdx.x;
    const int i = r0 + il;

    val_t bestSum = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            const val_t pB = radB[static_cast<size_t>(tt) * N + i];
            const val_t pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > bestSum) { bestSum = sum; bestT = t; }
    }

    __shared__ val_t sSum[256];
    __shared__ int sT[256];
    sSum[threadIdx.x] = bestSum;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const val_t oSum = sSum[threadIdx.x + s];
            const int oT = sT[threadIdx.x + s];
            // larger correlation wins; on ties the earlier lag (as in the sequential scan)
            if (oSum > sSum[threadIdx.x] || (oSum == sSum[threadIdx.x] && oT < sT[threadIdx.x])) {
                sSum[threadIdx.x] = oSum;
                sT[threadIdx.x] = oT;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[il] = WAVE_SPEED * static_cast<val_t>(sT[0]);
}

__global__ void countNonZeroKernel(size_t n, const val_t* __restrict__ data,
                                   unsigned long long* __restrict__ count) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int local = 0;
    for (size_t k = tid; k < n; k += stride) {
        if (data[k] > EPSILON) ++local;
    }
    if (local) atomicAdd(count, static_cast<unsigned long long>(local));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;             // Reflectivity (0.0 to 1.0), uniform
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source (complete on rank 0)

    Octree octree;                  // Spatial acceleration structure (host)
    std::vector<DevNode> octNodes;  // Flattened octree
    std::vector<uint32_t> octLeafIdx;

    size_t sourceIndex;

    // MPI decomposition
    int rank = 0, size = 1;
    int r0 = 0, r1 = 0;             // rows [r0, r1) owned by this rank
    int rankRows = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    Triangle* dTris = nullptr;
    float4* dVerts = nullptr;       // [3*N] vertices a,b,c (vector loads in the ray tests)
    DevNode* dNodes = nullptr;
    uint32_t* dLeafIdx = nullptr;
    val_t* dAreas = nullptr;
    val_t* dKijT = nullptr;         // [N][rankRows]
    int* dTauT = nullptr;           // [N][rankRows]
    val_t* dRadB = nullptr;         // [T][N]
    val_t* dRadBRow = nullptr;      // [rankRows]
    val_t* dDistances = nullptr;    // [rankRows]

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
    state.reflectivity = reflectivity;

    if (state.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    int maxDepth = 0;
    flattenOctree(state.octree, state.octNodes, state.octLeafIdx, 0, maxDepth);
    if (maxDepth * 7 + 1 > OCTREE_STACK) {
        fprintf(stderr, "Octree too deep for device traversal stack\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas
    const int N = static_cast<int>(state.numTriangles);
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Row decomposition over ranks
    state.rowCounts.resize(state.size);
    state.rowDispls.resize(state.size);
    for (int r = 0; r < state.size; ++r) {
        const int a = static_cast<int>(static_cast<int64_t>(N) * r / state.size);
        const int b = static_cast<int>(static_cast<int64_t>(N) * (r + 1) / state.size);
        state.rowDispls[r] = a;
        state.rowCounts[r] = b - a;
    }
    state.r0 = state.rowDispls[state.rank];
    state.rankRows = state.rowCounts[state.rank];
    state.r1 = state.r0 + state.rankRows;

    // Radiosity history (replicated), distances
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Device allocations
    const size_t R = static_cast<size_t>(state.rankRows);
    CUDA_CHECK(cudaMalloc(&state.dTris, state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), state.numTriangles * sizeof(Triangle),
                          cudaMemcpyHostToDevice));
    {
        std::vector<float4> verts(3 * state.numTriangles);
        for (size_t k = 0; k < state.numTriangles; ++k) {
            const Triangle& t = state.triangles[k];
            verts[3 * k + 0] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
            verts[3 * k + 1] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
            verts[3 * k + 2] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        }
        CUDA_CHECK(cudaMalloc(&state.dVerts, verts.size() * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(state.dVerts, verts.data(), verts.size() * sizeof(float4),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&state.dNodes, state.octNodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMemcpy(state.dNodes, state.octNodes.data(), state.octNodes.size() * sizeof(DevNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dLeafIdx, std::max<size_t>(1, state.octLeafIdx.size()) * sizeof(uint32_t)));
    if (!state.octLeafIdx.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dLeafIdx, state.octLeafIdx.data(),
                              state.octLeafIdx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&state.dAreas, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dKijT, std::max<size_t>(1, R * state.numTriangles) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dKijT, 0, std::max<size_t>(1, R * state.numTriangles) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTauT, std::max<size_t>(1, R * state.numTriangles) * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.dTauT, 0, std::max<size_t>(1, R * state.numTriangles) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, timesteps * state.numTriangles) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(1, timesteps * state.numTriangles) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadBRow, std::max<size_t>(1, R) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, std::max<size_t>(1, R) * sizeof(val_t)));
}

void freeSimulation(SimulationState& state) {
    cudaFree(state.dTris);
    cudaFree(state.dVerts);
    cudaFree(state.dNodes);
    cudaFree(state.dLeafIdx);
    cudaFree(state.dAreas);
    cudaFree(state.dKijT);
    cudaFree(state.dTauT);
    cudaFree(state.dRadB);
    cudaFree(state.dRadBRow);
    cudaFree(state.dDistances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    const size_t total = static_cast<size_t>(state.rankRows) * state.numTriangles;
    if (total == 0) return;
    const int block = 256;
    const size_t grid = (total + block - 1) / block;
    timeDelayKernel<<<static_cast<unsigned>(grid), block>>>(
        state.rankRows, static_cast<int>(state.numTriangles), state.r0, state.dTris, state.dTauT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int R = state.rankRows;

    // --- Stream bookkeeping: number of random draws consumed by each row ---------
    // A pair (i,j), i != j, consumes DRAWS_PER_PAIR draws unless culled by the
    // normal test.  Row counts are computed on the GPU and all-gathered.
    std::vector<uint32_t> localRowCount(std::max(1, R), 0);
    if (R > 0) {
        uint32_t* dRowCount = nullptr;
        CUDA_CHECK(cudaMalloc(&dRowCount, R * sizeof(uint32_t)));
        rowDrawCountKernel<<<R, 256>>>(N, state.r0, state.dTris, dRowCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localRowCount.data(), dRowCount, R * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dRowCount));
    }
    std::vector<uint32_t> rowCount(N);
    MPI_CHECK(MPI_Allgatherv(localRowCount.data(), R, MPI_UINT32_T, rowCount.data(),
                             state.rowCounts.data(), state.rowDispls.data(), MPI_UINT32_T,
                             MPI_COMM_WORLD));
    std::vector<uint64_t> rowDrawStart(N + 1);
    rowDrawStart[0] = 0;
    for (int i = 0; i < N; ++i) rowDrawStart[i + 1] = rowDrawStart[i] + DRAWS_PER_PAIR * rowCount[i];
    const uint64_t totalDraws = rowDrawStart[N];

    // --- Chunk / substream geometry ------------------------------------------------
    // Substreams per chunk scale with the OpenMP worker count: each worker handles
    // ~16 substreams per chunk (one ~0.8 ms jump each), which overlaps the GPU work.
    constexpr uint64_t CHUNK_DRAWS = 1ull << 26;   // 256 MB of random words per buffer
    constexpr uint64_t CHUNK_PAIRS = 1ull << 24;   // pair-offset table limit per chunk
    const int P = omp_get_max_threads();
    int log2Sub = 21;                                   // 32 substreams per chunk ...
    while (log2Sub > 16 && (CHUNK_DRAWS >> log2Sub) < static_cast<uint64_t>(16 * P)) --log2Sub;
    const uint64_t L = 1ull << log2Sub;                 // ... up to 1024 (16 MB per warp)

    // Substream start states (OpenMP + jump-ahead); all ranks set this up concurrently
    SubstreamStateProvider states(P, log2Sub, totalDraws);

    if (R == 0) return;

    uint64_t* dRowDrawStart = nullptr;
    CUDA_CHECK(cudaMalloc(&dRowDrawStart, (N + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dRowDrawStart, rowDrawStart.data(), (N + 1) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    // --- Chunking of the rank's rows ----------------------------------------------
    std::vector<std::pair<int, int>> chunks;       // [c0, c1) in global row indices
    {
        int c0 = state.r0;
        while (c0 < state.r1) {
            int c1 = c0 + 1;
            while (c1 < state.r1 &&
                   rowDrawStart[c1 + 1] - rowDrawStart[c0] <= CHUNK_DRAWS &&
                   static_cast<uint64_t>(c1 + 1 - c0) * N <= CHUNK_PAIRS) {
                ++c1;
            }
            chunks.emplace_back(c0, c1);
            c0 = c1;
        }
    }
    uint64_t maxChunkDraws = 1, maxChunkPairs = 1;
    int maxK = 1;
    for (const auto& ch : chunks) {
        const uint64_t D0 = rowDrawStart[ch.first], D1 = rowDrawStart[ch.second];
        maxChunkDraws = std::max(maxChunkDraws, D1 - D0);
        maxChunkPairs = std::max(maxChunkPairs, static_cast<uint64_t>(ch.second - ch.first) * N);
        if (D1 > D0) {
            maxK = std::max(maxK, static_cast<int>((D1 - 1) / L - D0 / L + 1));
        }
    }

    // --- Double-buffered pipeline: the CPU (OpenMP) computes the substream start
    //     states of chunk c+1 while the GPU generates draws / casts rays of chunk c --
    constexpr int NBUF = 2;
    uint32_t* hState[NBUF];
    uint32_t* dState[NBUF];
    uint32_t* dRng[NBUF];
    uint32_t* dOff[NBUF];
    cudaStream_t streams[NBUF];
    cudaEvent_t done[NBUF];
    const size_t stateWords = static_cast<size_t>(maxK) * MT_N;
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaHostAlloc(&hState[b], stateWords * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&dState[b], stateWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dRng[b], maxChunkDraws * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dOff[b], maxChunkPairs * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }

    int rowsDone = 0;
    int nextProgress = 100;
    for (size_t c = 0; c < chunks.size(); ++c) {
        const int b = static_cast<int>(c % NBUF);
        const int c0 = chunks[c].first, c1 = chunks[c].second;
        const int chunkRows = c1 - c0;
        const uint64_t D0 = rowDrawStart[c0];
        const uint64_t D1 = rowDrawStart[c1];

        // Wait until the GPU is done with this buffer set
        if (c >= static_cast<size_t>(NBUF)) CUDA_CHECK(cudaEventSynchronize(done[b]));

        // Pair offsets into the chunk's stream slice
        pairOffsetKernel<<<chunkRows, 256, 0, streams[b]>>>(c0, N, state.dTris, dRowDrawStart, D0, dOff[b]);
        CUDA_CHECK(cudaGetLastError());

        // Substream start states (CPU, OpenMP) -> GPU generation of the draws
        if (D1 > D0) {
            const int64_t k0 = static_cast<int64_t>(D0 / L);
            const int K = static_cast<int>((D1 - 1) / L - static_cast<uint64_t>(k0) + 1);
            states.fill(k0, K, hState[b]);
            CUDA_CHECK(cudaMemcpyAsync(dState[b], hState[b], static_cast<size_t>(K) * MT_N * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, streams[b]));
            const int genBlocks = (K + MTGEN_WARPS - 1) / MTGEN_WARPS;
            mtGenerateKernel<<<genBlocks, MTGEN_BLOCK, 0, streams[b]>>>(K, k0, L, D0, D1, dState[b], dRng[b]);
            CUDA_CHECK(cudaGetLastError());
        }

        // Ray casting for all pairs of the chunk (NUM_RAYS lanes per pair)
        const size_t totalThreads = static_cast<size_t>(chunkRows) * N * NUM_RAYS;
        const size_t grid = (totalThreads + FF_BLOCK - 1) / FF_BLOCK;
        formFactorKernel<<<static_cast<unsigned>(grid), FF_BLOCK, 0, streams[b]>>>(
            chunkRows, c0 - state.r0, R, N, state.r0, state.dTris, state.dVerts, state.dNodes,
            state.dLeafIdx, dOff[b], dRng[b], state.dKijT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(done[b], streams[b]));

        rowsDone += chunkRows;
        if (state.rank == 0 && (rowsDone >= nextProgress || rowsDone == R)) {
            printf("  Progress: %d/%d triangles (rank 0)\n", rowsDone, R);
            while (nextProgress <= rowsDone) nextProgress += 100;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaEventDestroy(done[b]));
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFree(dRng[b]));
        CUDA_CHECK(cudaFree(dOff[b]));
        CUDA_CHECK(cudaFree(dState[b]));
        CUDA_CHECK(cudaFreeHost(hState[b]));
    }
    CUDA_CHECK(cudaFree(dRowDrawStart));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int R = state.rankRows;
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = static_cast<int>(state.numTimesteps / 2);   // source active for first half

    std::vector<val_t> localRow(std::max(1, R));
    const int block = 128;
    const int grid = (R + block - 1) / block;

    for (int t = 0; t < T; ++t) {
        if (R > 0) {
            propagationKernel<<<grid, block>>>(R, N, state.r0, t, static_cast<int>(state.sourceIndex),
                                               timeOff, state.dKijT, state.dTauT, state.dAreas,
                                               state.reflectivity, state.dRadB, state.dRadBRow);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localRow.data(), state.dRadBRow, R * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        }
        val_t* fullRow = state.radB.data() + state.idxTN(t, 0);
        MPI_CHECK(MPI_Allgatherv(localRow.data(), R, MPI_FLOAT, fullRow, state.rowCounts.data(),
                                 state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(state.dRadB + state.idxTN(t, 0), fullRow, N * sizeof(val_t),
                              cudaMemcpyHostToDevice));

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int R = state.rankRows;
    const int T = static_cast<int>(state.numTimesteps);

    std::vector<val_t> local(std::max(1, R));
    if (R > 0 && T > 0) {
        distanceKernel<<<R, 256>>>(N, T, state.r0, static_cast<int>(state.sourceIndex),
                                   state.dRadB, state.dDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), state.dDistances, R * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Allgatherv(local.data(), R, MPI_FLOAT, state.distances.data(),
                             state.rowCounts.data(), state.rowDispls.data(), MPI_FLOAT,
                             MPI_COMM_WORLD));
}

// ============================================================================
// Validation
// ============================================================================

// Number of form factors above EPSILON over all ranks' rows (collective)
unsigned long long countNonZeroFormFactors(const SimulationState& state) {
    unsigned long long local = 0;
    const size_t n = static_cast<size_t>(state.rankRows) * state.numTriangles;
    if (n > 0) {
        unsigned long long* dCount = nullptr;
        CUDA_CHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
        const int block = 256;
        const size_t grid = std::min<size_t>((n + block - 1) / block, 4096);
        countNonZeroKernel<<<static_cast<unsigned>(grid), block>>>(n, state.dKijT, dCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&local, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dCount));
    }
    unsigned long long total = 0;
    MPI_CHECK(MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    return total;
}

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
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
    const int N = static_cast<int>(state.numTriangles);
    int receivedEnergy = 0;
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
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
    const size_t totalPairs = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, totalPairs,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalPairs));

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

static double wallTimeMs(MPI_Comm comm) {
    MPI_Barrier(comm);
    return MPI_Wtime() * 1000.0;
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

    // Parse command line arguments
    int parseStatus = 0;  // 0 = run, 1 = exit ok, 2 = exit error
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
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    // OpenMP threads: never oversubscribe the cores this rank is actually allowed to
    // run on (MPI launchers often bind ranks to a core or a socket).
    {
        int threads = omp_get_max_threads();
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            const int allowed = CPU_COUNT(&set);
            if (allowed > 0 && allowed < threads) threads = allowed;
        }
        omp_set_num_threads(std::max(1, threads));
    }

    // One GPU per rank (round-robin over the GPUs visible on this node)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int nodeRank = 0;
        MPI_Comm_rank(nodeComm, &nodeRank);
        MPI_Comm_free(&nodeComm);
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(nodeRank % devCount));
        CUDA_CHECK(cudaFree(nullptr));  // establish context now, outside the timed phases
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
        printf("Parallelization: %d MPI ranks x %d OpenMP threads, 1 CUDA device per rank\n",
               size, omp_get_max_threads());
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
    double startPre = wallTimeMs(MPI_COMM_WORLD);

    computeTimeDelays(state);
    computeFormFactors(state);

    double endPre = wallTimeMs(MPI_COMM_WORLD);
    double preElapsedMs = endPre - startPre;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &preElapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long preDuration = static_cast<long>(preElapsedMs);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    double startSim = wallTimeMs(MPI_COMM_WORLD);

    runSimulation(state);

    double endSim = wallTimeMs(MPI_COMM_WORLD);
    double simElapsedMs = endSim - startSim;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &simElapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long simDuration = static_cast<long>(simElapsedMs);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    double startDist = wallTimeMs(MPI_COMM_WORLD);

    computeDistances(state);

    double endDist = wallTimeMs(MPI_COMM_WORLD);
    double distElapsedMs = endDist - startDist;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &distElapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long distDuration = static_cast<long>(distElapsedMs);

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;

    unsigned long long nonZeroKij = 0;
    if (validate) nonZeroKij = countNonZeroFormFactors(state);

    int exitCode = 0;
    if (rank == 0) {
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
            if (!validateResults(state, nonZeroKij)) {
                exitCode = 1;
            }
        }
        fflush(stdout);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    freeSimulation(state);
    MPI_Finalize();
    return exitCode;
}

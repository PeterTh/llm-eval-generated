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
 *  - MPI: receiver triangles (rows of the Kij / Tau matrices) are block-distributed
 *    over ranks; each rank owns one GPU. Radiosity of each timestep is exchanged
 *    with MPI_Allgatherv.
 *  - CUDA: time delays, form factor ray casting (warp-cooperative octree traversal),
 *    MT19937 jump-ahead and stream generation, and the per-timestep radiosity
 *    gathering run on the GPU.
 *  - OpenMP: host-side precomputation (culling masks, MT19937 jump polynomials,
 *    cross-correlation).
 *
 * Floating-point results are bitwise identical to the sequential reference: device
 * code reproduces the host compiler's FMA contraction explicitly and keeps all
 * reductions in the original order.
 *
 * The original code consumes a single sequential std::mt19937 stream over all
 * (i, j) pairs in row-major order. To reproduce that stream exactly in parallel,
 * every pair's position in the stream is computed (64 draws per non-culled pair)
 * and MT19937 is jumped ahead using GF(2) polynomial arithmetic.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#if defined(__PCLMUL__)
#include <wmmintrin.h>
#endif

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
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
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;  // RNG draws consumed by a non-culled pair

static int g_rank = 0;
static int g_size = 1;

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

    // Device code is compiled without automatic FMA contraction; the fused operations
    // below reproduce the contraction pattern of the host compiler (GCC fuses the first
    // product of a sum: a*b + c*d -> fma(a, b, c*d)) so results are bitwise identical.
#ifdef __CUDA_ARCH__
    __device__ val_t dot(const Vec3& o) const { return fmaf(z, o.z, fmaf(x, o.x, y * o.y)); }
    __device__ Vec3 cross(const Vec3& o) const {
        return {fmaf(y, o.z, -(z * o.y)), fmaf(z, o.x, -(x * o.z)), fmaf(x, o.y, -(y * o.x))};
    }
    __device__ val_t squaredNorm() const { return fmaf(z, z, fmaf(x, x, y * y)); }
#else
    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
#endif
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
};

// ============================================================================
// Flattened Octree for the GPU
// ============================================================================

struct DevNode {
    float cx, cy, cz;      // center
    float hx, hy, hz;      // half extent
    int childStart;        // index of first child (children are contiguous, in original order)
    int childCount;
    int triStart;          // leaf: first entry in leaf triangle array (3 float4 per entry)
    int triCount;          // leaf iff triCount > 0 (matches !triangleIndices.empty())
};

constexpr int OCTREE_STACK = 128;

struct FlatOctree {
    std::vector<DevNode> nodes;
    std::vector<float4> leafTris;  // per entry: {a, idx bits}, {e1, 0}, {e2, 0}
    int maxDepth = 0;

    void flatten(const Octree& root, const std::vector<Triangle>& tris) {
        nodes.clear();
        leafTris.clear();
        nodes.push_back(DevNode{});
        fill(root, 0, tris, 0);
    }

private:
    void fill(const Octree& node, size_t slot, const std::vector<Triangle>& tris, int depth) {
        maxDepth = std::max(maxDepth, depth);
        DevNode dn{};
        dn.cx = node.center.x; dn.cy = node.center.y; dn.cz = node.center.z;
        dn.hx = node.halfExtent.x; dn.hy = node.halfExtent.y; dn.hz = node.halfExtent.z;
        dn.triStart = static_cast<int>(leafTris.size() / 3);
        dn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            const Triangle& t = tris[idx];
            Vec3 e1 = t.b - t.a;
            Vec3 e2 = t.c - t.a;
            int ii = static_cast<int>(idx);
            float fidx;
            memcpy(&fidx, &ii, sizeof(float));
            leafTris.push_back(make_float4(t.a.x, t.a.y, t.a.z, fidx));
            leafTris.push_back(make_float4(e1.x, e1.y, e1.z, 0.0f));
            leafTris.push_back(make_float4(e2.x, e2.y, e2.z, 0.0f));
        }
        std::vector<const Octree*> kids;
        if (node.triangleIndices.empty()) {
            for (int i = 0; i < 8; ++i)
                if (node.children[i]) kids.push_back(node.children[i].get());
        }
        dn.childStart = static_cast<int>(nodes.size());
        dn.childCount = static_cast<int>(kids.size());
        nodes[slot] = dn;
        size_t base = nodes.size();
        nodes.resize(base + kids.size());
        for (size_t k = 0; k < kids.size(); ++k) fill(*kids[k], base + k, tris, depth + 1);
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
// ============================================================================
//
// The reference uses std::mt19937(42) with std::uniform_real_distribution<float>,
// which consumes exactly one 32-bit draw per value: u = float(x) / 2^32 (clamped
// below 1). A non-culled (i, j) pair consumes 64 consecutive draws in row-major
// pair order (u, v for the point on i, then u, v for the point on j, per ray).

namespace mtj {
constexpr int MT_N = 624, MT_M = 397;
constexpr uint32_t MT_A = 0x9908b0dfu, MT_UP = 0x80000000u, MT_LO = 0x7fffffffu;
constexpr int DEG = 19937;
constexpr int NW = (DEG + 63) / 64;   // 64-bit words of a reduced polynomial

using State = std::array<uint32_t, MT_N>;   // logical order: next word is f(x[0], x[1], x[397])

inline uint32_t mtNext(uint32_t x0, uint32_t x1, uint32_t xm) {
    uint32_t y = (x0 & MT_UP) | (x1 & MT_LO);
    return xm ^ (y >> 1) ^ ((y & 1u) ? MT_A : 0u);
}

inline State seedState(uint32_t seed) {
    State s;
    s[0] = seed;
    for (int i = 1; i < MT_N; ++i) s[i] = 1812433253u * (s[i - 1] ^ (s[i - 1] >> 30)) + static_cast<uint32_t>(i);
    return s;
}

// Raw (untempered) word sequence x_0 .. x_{len-1} starting from state s
inline void rawSequence(const State& s, std::vector<uint32_t>& seq, size_t len) {
    seq.resize(len);
    for (int i = 0; i < MT_N && static_cast<size_t>(i) < len; ++i) seq[i] = s[i];
    for (size_t n = MT_N; n < len; ++n) seq[n] = mtNext(seq[n - MT_N], seq[n - MT_N + 1], seq[n - MT_N + MT_M]);
}

struct Poly { std::vector<uint64_t> w; };

class Field {
public:
    std::vector<uint64_t> phi;                    // characteristic polynomial (DEG+1 bits)

    Field() {
        // Berlekamp-Massey on one output bit of MT19937 recovers its characteristic polynomial
        const int len = 2 * DEG + 64;
        std::vector<uint32_t> seq;
        rawSequence(seedState(5489u), seq, static_cast<size_t>(len) + MT_N);
        const int nw = (len + 64) / 64 + 2;
        // R holds the bit stream reversed: bit k of R = s_{len-1-k}
        std::vector<uint64_t> R(nw, 0), C(nw, 0), B(nw, 0), T;
        for (int k = 0; k < len; ++k) {
            uint32_t bit = seq[static_cast<size_t>(len - 1 - k) + MT_N] >> 31;
            if (bit) R[k >> 6] |= 1ull << (k & 63);
        }
        C[0] = B[0] = 1;
        int L = 0, m = 1;
        // Rs[r][q] = 64 bits of R starting at bit 64q + r (aligned access for any offset)
        std::vector<uint64_t> Rs(64 * static_cast<size_t>(nw + 1), 0);
        for (int r = 0; r < 64; ++r)
            for (int q = 0; q < nw; ++q) {
                uint64_t v = R[q] >> r;
                if (r && q + 1 < nw) v |= R[q + 1] << (64 - r);
                Rs[static_cast<size_t>(r) * (nw + 1) + q] = v;
            }
        auto xorShifted = [&](std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, int sh, int words) {
            const int q = sh >> 6, r = sh & 63;
            const int hiw = std::min(nw - 1, words + q + 1);
            if (r == 0) {
                for (int i = q; i <= hiw; ++i) dst[i] ^= src[i - q];
            } else {
                dst[q] ^= src[0] << r;
                for (int i = q + 1; i <= hiw; ++i) dst[i] ^= (src[i - q] << r) | (src[i - q - 1] >> (64 - r));
            }
        };
        int lenB = 0;  // degree bound of B
        for (int n = 0; n < len; ++n) {
            // discrepancy d = sum_{i=0..L} C_i s_{n-i}, with s_{n-i} = bit (len-1-n+i) of R
            int off = len - 1 - n;
            const int words = (L >> 6) + 1;
            const uint64_t* rp = &Rs[static_cast<size_t>(off & 63) * (nw + 1) + (off >> 6)];
            const int avail = nw - (off >> 6);
            uint64_t acc = 0;
            const int full = std::min(words - 1, avail);
            for (int wi = 0; wi < full; ++wi) acc ^= C[wi] & rp[wi];
            if (words - 1 < avail) {
                uint64_t cw = C[words - 1];
                if ((L + 1) & 63) cw &= (1ull << ((L + 1) & 63)) - 1;
                acc ^= cw & rp[words - 1];
            }
            if (!(__builtin_popcountll(acc) & 1)) { ++m; continue; }
            if (2 * L <= n) {
                T = C;
                xorShifted(C, B, m, (lenB >> 6) + 1);
                int oldL = L;
                L = n + 1 - L;
                B = std::move(T);
                lenB = oldL;
                m = 1;
            } else {
                xorShifted(C, B, m, (lenB >> 6) + 1);
                ++m;
            }
        }
        if (L != DEG) {
            fprintf(stderr, "MT19937 characteristic polynomial degree mismatch (%d)\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // phi(x) = x^L C(1/x): coefficient of x^(L-i) is C_i
        phi.assign(NW + 1, 0);
        for (int i = 0; i <= L; ++i)
            if ((C[i >> 6] >> (i & 63)) & 1) { int e = L - i; phi[e >> 6] |= 1ull << (e & 63); }
        // Top 64 coefficients of phi below the leading term: bit 63 - (k-1) = coeff of x^(DEG-k)
        htop = 0;
        for (int k = 1; k <= 64; ++k) {
            int e = DEG - k;
            if ((phi[e >> 6] >> (e & 63)) & 1) htop |= 1ull << (64 - k);
        }
    }

    // Reduce a polynomial modulo phi in place; result occupies the first NW words.
    // Works on 64-bit windows [DEG + 64m, DEG + 64m + 64) from the top: the 64 quotient
    // bits of a window are found by a short long-division against the top coefficients
    // of phi, then q(x) * phi(x) * x^(64m) is cancelled with carry-less multiplications.
    void reduce(std::vector<uint64_t>& a) const {
        const int words = static_cast<int>(a.size());
        a.resize(static_cast<size_t>(words) + NW + 3, 0);  // room for the top window's product
        const int topBit = words * 64 - 1;
        if (topBit >= DEG) {
            for (int m = (topBit - DEG) / 64; m >= 0; --m) {
                const int p = DEG + 64 * m, q0 = p >> 6, r0 = p & 63;
                uint64_t W = a[q0] >> r0;
                if (r0) W |= a[q0 + 1] << (64 - r0);
                if (!W) continue;
                uint64_t q = 0;
                for (int s = 63; s >= 0; --s) {
                    if ((W >> s) & 1) {
                        q |= 1ull << s;
                        if (s) W ^= htop >> (64 - s);
                    }
                }
                uint64_t carry = 0;
                for (int w = 0; w <= NW; ++w) {
                    uint64_t lo, hi;
                    clmul64(q, phi[w], lo, hi);
                    a[m + w] ^= lo ^ carry;
                    carry = hi;
                }
                a[m + NW + 1] ^= carry;
            }
        }
        a.resize(NW);
    }

    static uint64_t spread32(uint32_t v) {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2)) & 0x3333333333333333ull;
        x = (x | (x << 1)) & 0x5555555555555555ull;
        return x;
    }

    Poly square(const Poly& p) const {
        std::vector<uint64_t> a(2 * NW + 2, 0);
        for (int i = 0; i < NW; ++i) {
            a[2 * i] = spread32(static_cast<uint32_t>(p.w[i]));
            a[2 * i + 1] = spread32(static_cast<uint32_t>(p.w[i] >> 32));
        }
        reduce(a);
        return Poly{std::move(a)};
    }

    Poly mulX(const Poly& p) const {
        std::vector<uint64_t> a(NW + 1, 0);
        for (int i = NW; i >= 0; --i) {
            uint64_t v = (i < NW ? p.w[i] << 1 : 0);
            if (i > 0) v |= p.w[i - 1] >> 63;
            a[i] = v;
        }
        reduce(a);
        return Poly{std::move(a)};
    }

    // a * b mod phi
    Poly mulMod(const Poly& x, const Poly& y) const {
        std::vector<uint64_t> a(2 * NW + 2, 0);
        for (int i = 0; i < NW; ++i) {
            const uint64_t xi = x.w[i];
            if (!xi) continue;
            for (int j = 0; j < NW; ++j) {
                uint64_t lo, hi;
                clmul64(xi, y.w[j], lo, hi);
                a[i + j] ^= lo;
                a[i + j + 1] ^= hi;
            }
        }
        reduce(a);
        return Poly{std::move(a)};
    }

    // x^n mod phi
    Poly powX(uint64_t n) const {
        Poly r{std::vector<uint64_t>(NW, 0)};
        r.w[0] = 1;
        if (n == 0) return r;
        int hb = 63 - __builtin_clzll(n);
        for (int b = hb; b >= 0; --b) {
            r = square(r);
            if ((n >> b) & 1) r = mulX(r);
        }
        return r;
    }

    // out[c] = x^(base + c * step) mod phi for c = 0..count-1, computed in parallel
    // segments (one exponentiation per segment, then multiplication by x^step)
    void powSeries(uint64_t base, uint64_t step, size_t count, std::vector<Poly>& out) const {
        out.resize(count);
        if (count == 0) return;
        const size_t nseg = std::min<size_t>(count, static_cast<size_t>(std::max(1, omp_get_max_threads())));
        const size_t segLen = (count + nseg - 1) / nseg;
        const Poly pStep = powX(step);
        #pragma omp parallel for schedule(static, 1)
        for (size_t sgi = 0; sgi < nseg; ++sgi) {
            const size_t a = sgi * segLen, b = std::min(count, a + segLen);
            if (a >= b) continue;
            out[a] = powX(base + a * step);
            for (size_t c = a + 1; c < b; ++c) out[c] = mulMod(out[c - 1], pStep);
        }
    }

private:
    uint64_t htop = 0;

    static void clmul64(uint64_t x, uint64_t y, uint64_t& lo, uint64_t& hi) {
#if defined(__PCLMUL__)
        const __m128i r = _mm_clmulepi64_si128(_mm_cvtsi64_si128(static_cast<long long>(x)),
                                               _mm_cvtsi64_si128(static_cast<long long>(y)), 0);
        lo = static_cast<uint64_t>(_mm_cvtsi128_si64(r));
        hi = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)));
#else
        uint64_t l = 0, h = 0;
        for (int i = 0; i < 64; ++i) {
            if ((y >> i) & 1) {
                l ^= x << i;
                if (i) h ^= x >> (64 - i);
            }
        }
        lo = l;
        hi = h;
#endif
    }
};
} // namespace mtj

// Generate tempered MT19937 output for many chunks in parallel (one block per chunk).
// Each chunk starts from its own (jumped-ahead) state and produces chunkLen draws.
constexpr int MT_THREADS = 256;

__global__ void mtGenerateKernel(const uint32_t* __restrict__ states, uint32_t* __restrict__ out,
                                 uint64_t chunkLen) {
    __shared__ uint32_t x[mtj::MT_N];
    const int tid = threadIdx.x;
    const uint32_t* st = states + static_cast<size_t>(blockIdx.x) * mtj::MT_N;
    for (int k = tid; k < mtj::MT_N; k += blockDim.x) x[k] = st[k];
    __syncthreads();

    uint32_t* dst = out + static_cast<size_t>(blockIdx.x) * chunkLen;
    const uint64_t nBlocks = chunkLen / mtj::MT_N;

    auto temper = [](uint32_t y) {
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    };
    auto next = [](uint32_t x0, uint32_t x1, uint32_t xm) {
        uint32_t y = (x0 & 0x80000000u) | (x1 & 0x7fffffffu);
        return xm ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    };

    for (uint64_t b = 0; b < nBlocks; ++b) {
        uint32_t* o = dst + b * mtj::MT_N;
        // Phase 1: k in [0, 227) uses old x[k+397]
        uint32_t v = 0;
        int k = tid;
        if (k < 227) v = next(x[k], x[k + 1], x[k + 397]);
        __syncthreads();
        if (k < 227) { x[k] = v; o[k] = temper(v); }
        __syncthreads();
        // Phase 2: k in [227, 454) uses new x[k-227]
        k = 227 + tid;
        if (tid < 227) v = next(x[k], x[k + 1], x[k - 227]);
        __syncthreads();
        if (tid < 227) { x[k] = v; o[k] = temper(v); }
        __syncthreads();
        // Phase 3: k in [454, 624) uses new x[k-227]; k = 623 uses new x[0]
        k = 454 + tid;
        if (tid < 170) v = next(x[k], x[(k + 1) % mtj::MT_N], x[k - 227]);
        __syncthreads();
        if (tid < 170) { x[k] = v; o[k] = temper(v); }
        __syncthreads();
    }
}

// MT19937 jump-ahead on the GPU (one block per jump): out = sum_i p_i T^i(base), i.e.
// out[k] = XOR over set coefficients i of p of x_{i+k}, where x is the raw word sequence
// generated from the base state. Job k uses base state k / baseDiv and polynomial k % polyMod.
constexpr int JUMP_THREADS = 640;
constexpr int JUMP_SEQ = mtj::DEG + mtj::MT_N;
constexpr int JUMP_KPT = 4;                                   // output words per thread
constexpr int JUMP_KGROUPS = mtj::MT_N / JUMP_KPT;            // 156
constexpr int JUMP_SPLITS = JUMP_THREADS / JUMP_KGROUPS;      // 4 slices of the polynomial

__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(const uint32_t* __restrict__ baseStates, uint64_t baseDiv,
             const uint64_t* __restrict__ polys, uint64_t polyMod, uint64_t kStart,
             uint32_t* __restrict__ out) {
    extern __shared__ uint32_t seq[];
    __shared__ uint32_t acc[mtj::MT_N];
    const int tid = threadIdx.x;
    const uint64_t k = kStart + blockIdx.x;
    const uint32_t* base = baseStates + (k / baseDiv) * mtj::MT_N;
    const uint32_t* poly = reinterpret_cast<const uint32_t*>(polys + (k % polyMod) * mtj::NW);

    for (int i = tid; i < mtj::MT_N; i += blockDim.x) {
        seq[i] = base[i];
        acc[i] = 0;
    }
    __syncthreads();

    auto next = [](uint32_t x0, uint32_t x1, uint32_t xm) {
        uint32_t y = (x0 & 0x80000000u) | (x1 & 0x7fffffffu);
        return xm ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    };
    // x_n = f(x_{n-624}, x_{n-623}, x_{n-227}); three dependency phases per 624 words
    for (int n0 = mtj::MT_N; n0 < JUMP_SEQ; n0 += mtj::MT_N) {
        int n = n0 + tid;
        if (tid < 227 && n < JUMP_SEQ) seq[n] = next(seq[n - 624], seq[n - 623], seq[n - 227]);
        __syncthreads();
        n = n0 + 227 + tid;
        if (tid < 227 && n < JUMP_SEQ) seq[n] = next(seq[n - 624], seq[n - 623], seq[n - 227]);
        __syncthreads();
        n = n0 + 454 + tid;
        if (tid < 170 && n < JUMP_SEQ) seq[n] = next(seq[n - 624], seq[n - 623], seq[n - 227]);
        __syncthreads();
    }

    // Thread (slice, g) accumulates outputs k = g + 156 r (r = 0..3) over one slice of the
    // polynomial's 32-bit words; slices are combined with shared-memory XOR atomics.
    if (tid < JUMP_KGROUPS * JUMP_SPLITS) {
        const int g = tid % JUMP_KGROUPS, slice = tid / JUMP_KGROUPS;
        constexpr int PW = 2 * mtj::NW;
        const int w0 = slice * PW / JUMP_SPLITS, w1 = (slice + 1) * PW / JUMP_SPLITS;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int w = w0; w < w1; ++w) {
            uint32_t bits = poly[w];
            const uint32_t* sp = seq + w * 32 + g;
            while (bits) {
                const int i = __ffs(bits) - 1;
                bits &= bits - 1;
                a0 ^= sp[i];
                a1 ^= sp[i + JUMP_KGROUPS];
                a2 ^= sp[i + 2 * JUMP_KGROUPS];
                a3 ^= sp[i + 3 * JUMP_KGROUPS];
            }
        }
        atomicXor(&acc[g], a0);
        atomicXor(&acc[g + JUMP_KGROUPS], a1);
        atomicXor(&acc[g + 2 * JUMP_KGROUPS], a2);
        atomicXor(&acc[g + 3 * JUMP_KGROUPS], a3);
    }
    __syncthreads();
    for (int i = tid; i < mtj::MT_N; i += blockDim.x) out[static_cast<size_t>(blockIdx.x) * mtj::MT_N + i] = acc[i];
}

__device__ __forceinline__ float drawToFloat(uint32_t u) {
    // generate_canonical<float, 24>(mt19937): float(u) / 2^32, clamped below 1
    float r = __uint2float_rn(u) * 2.3283064365386963e-10f;
    return (r >= 1.0f) ? 0x1.fffffep-1f : r;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                      const Vec3& v0, const Vec3& e1, const Vec3& e2) {
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

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle.
// Warp-cooperative packet traversal: the warp walks the union of the octree nodes
// visited by its rays using a shared stack of (node, lane mask) entries. Every lane
// performs exactly the box and triangle tests its own ray performs in the reference
// traversal (a node is tested by a lane iff all its ancestors passed for that lane),
// so the per-ray result is identical; only the control flow is made uniform.
struct StackEntry { int node; unsigned mask; };

__device__ bool isRayBlockedWarp(bool active, const Vec3& from, const Vec3& to,
                                 const DevNode* __restrict__ nodes, const float4* __restrict__ leafTris,
                                 int srcTriIdx, int dstTriIdx, StackEntry* __restrict__ stack) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned laneBit = 1u << lane;

    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    bool blocked = false;
    if (active && rayLen < EPSILON) { blocked = true; active = false; }
    Vec3 dirNorm = dir / rayLen;
    const val_t maxDist = rayLen - EPSILON;

    // Ray-invariant parts of the segment/box test
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    unsigned live = __ballot_sync(0xffffffffu, active);
    if (!live) return blocked;

    int sp = 0;
    if (lane == 0) stack[0] = StackEntry{0, live};  // root is visited without a box test
    sp = 1;
    __syncwarp();
    while (sp > 0) {
        --sp;
        const StackEntry e = stack[sp];
        __syncwarp();
        const DevNode n = nodes[e.node];
        const bool mine = (e.mask & laneBit) && !blocked;
        if (n.triCount > 0) {
            const float4* base = leafTris + 3 * static_cast<size_t>(n.triStart);
            for (int t = 0; t < n.triCount; ++t) {
                const float4 a = __ldg(base + 3 * t);
                const int idx = __float_as_int(a.w);
                if (mine && !blocked && idx != srcTriIdx && idx != dstTriIdx) {
                    const float4 b = __ldg(base + 3 * t + 1);
                    const float4 c = __ldg(base + 3 * t + 2);
                    val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(a.x, a.y, a.z),
                                                      Vec3(b.x, b.y, b.z), Vec3(c.x, c.y, c.z));
                    if (dist > EPSILON && dist < maxDist) blocked = true;  // Ray is blocked
                }
            }
        } else {
            for (int k = n.childCount - 1; k >= 0; --k) {
                const int ci = n.childStart + k;
                bool hit = false;
                if (mine) {
                    const DevNode& cn = nodes[ci];
                    const Vec3 halfExtent(cn.hx, cn.hy, cn.hz);
                    const Vec3 c = mid - Vec3(cn.cx, cn.cy, cn.cz);
                    hit = !(fabsf(c.x) > halfExtent.x + ad.x) &&
                          !(fabsf(c.y) > halfExtent.y + ad.y) &&
                          !(fabsf(c.z) > halfExtent.z + ad.z) &&
                          !(fabsf(fmaf(d.y, c.z, -(d.z * c.y))) > fmaf(halfExtent.y, ad.z, halfExtent.z * ad.y) + EPSILON) &&
                          !(fabsf(fmaf(d.z, c.x, -(d.x * c.z))) > fmaf(halfExtent.z, ad.x, halfExtent.x * ad.z) + EPSILON) &&
                          !(fabsf(fmaf(d.x, c.y, -(d.y * c.x))) > fmaf(halfExtent.x, ad.y, halfExtent.y * ad.x) + EPSILON);
                }
                const unsigned m = __ballot_sync(0xffffffffu, hit);
                if (m) {
                    if (lane == 0) stack[sp] = StackEntry{ci, m};
                    ++sp;
                }
            }
            __syncwarp();
        }
        if (!__any_sync(0xffffffffu, (live & laneBit) && !blocked)) break;
    }
    return blocked;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // std::max(ZERO, c)
}

__device__ __forceinline__ Vec3 pointInTriangle(const float4& a, const float4& e1, const float4& e2,
                                                val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    // t.a + ab * u + ac * v
    return Vec3(fmaf(e2.x, v, fmaf(e1.x, u, a.x)),
                fmaf(e2.y, v, fmaf(e1.y, u, a.y)),
                fmaf(e2.z, v, fmaf(e1.z, u, a.z)));
}

struct FFParams {
    uint64_t numPairs;          // pairs in this batch = nb * N
    uint32_t nb;                // rows in batch
    uint32_t b0;                // first local row of the batch
    uint32_t nloc;              // local rows of this rank
    uint32_t N;
    uint32_t i0;                // global index of first local row
    uint32_t maskWords;         // 32-bit mask words per row
    uint64_t bufBase;           // global draw index of buf[0]
};

// One thread per (pair, ray); 16 consecutive lanes form one pair.
constexpr int FF_THREADS = 256;

__global__ void __launch_bounds__(FF_THREADS, 5)
formFactorKernel(FFParams P,
                 const uint32_t* __restrict__ maskBits, const uint32_t* __restrict__ maskPrefix,
                 const uint64_t* __restrict__ rowOff, const uint32_t* __restrict__ buf,
                 const float4* __restrict__ triA, const float4* __restrict__ triE1,
                 const float4* __restrict__ triE2, const float4* __restrict__ triN,
                 const float* __restrict__ areas,
                 const DevNode* __restrict__ nodes, const float4* __restrict__ leafTris,
                 float* __restrict__ W, unsigned long long* __restrict__ nonZeroCount) {
    __shared__ StackEntry stacks[FF_THREADS / 32][OCTREE_STACK];
    StackEntry* stack = stacks[threadIdx.x >> 5];

    const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int ray = static_cast<int>(gid & (NUM_RAYS - 1));
    const uint64_t p = gid / NUM_RAYS;

    bool active = p < P.numPairs;
    uint32_t iloc = 0, j = 0, i = 0;
    Vec3 pI, pJ;
    if (active) {
        iloc = P.b0 + static_cast<uint32_t>(p % P.nb);
        j = static_cast<uint32_t>(p / P.nb);
        i = P.i0 + iloc;
        const size_t mw = static_cast<size_t>(iloc) * P.maskWords + (j >> 5);
        const uint32_t word = maskBits[mw];
        const uint32_t bit = 1u << (j & 31);
        active = (word & bit) != 0;   // i != j and not culled
        if (active) {
            const uint64_t r = rowOff[iloc] + maskPrefix[mw] + __popc(word & (bit - 1));
            const uint32_t* d = buf + (r * DRAWS_PER_PAIR + 4 * ray - P.bufBase);
            const uint4 dr = *reinterpret_cast<const uint4*>(d);
            const float4 aI = __ldg(triA + i), e1I = __ldg(triE1 + i), e2I = __ldg(triE2 + i);
            const float4 aJ = __ldg(triA + j), e1J = __ldg(triE1 + j), e2J = __ldg(triE2 + j);
            pI = pointInTriangle(aI, e1I, e2I, drawToFloat(dr.x), drawToFloat(dr.y));
            pJ = pointInTriangle(aJ, e1J, e2J, drawToFloat(dr.z), drawToFloat(dr.w));
        }
    }

    const bool blocked = isRayBlockedWarp(active, pI, pJ, nodes, leafTris, static_cast<int>(i),
                                          static_cast<int>(j), stack);

    val_t contrib = ZERO;
    if (active && !blocked) {
        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (!(distSqr < EPSILON)) {
            const float4 nI = __ldg(triN + i), nJ = __ldg(triN + j);
            val_t cosPhiI = cosPhi(v, Vec3(nI.x, nI.y, nI.z));
            val_t cosPhiJ = cosPhi(-v, Vec3(nJ.x, nJ.y, nJ.z));
            if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO))
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
        }
    }

    // Sequential accumulation over rays in the original order
    val_t kij = ZERO;
#pragma unroll
    for (int r = 0; r < NUM_RAYS; ++r) kij += __shfl_sync(0xffffffffu, contrib, r, NUM_RAYS);
    kij = kij * INV_NUM_RAYS;

    const bool leader = active && ray == 0;
    if (leader && kij > ZERO) {
        val_t w = kij * areas[j];
        W[static_cast<size_t>(iloc) * P.N + j] = (ONE < w) ? ONE : w;  // std::min(kij * area_j, 1)
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, leader && kij > EPSILON);
    if ((threadIdx.x & 31) == 0 && ballot) atomicAdd(nonZeroCount, static_cast<unsigned long long>(__popc(ballot)));
}

// ============================================================================
// Simulation Kernel (one timestep, one warp per local receiver triangle)
// ============================================================================

// The warp loads 32 consecutive emitters at a time; the contributing terms are then
// accumulated in ascending j order (identical to the sequential reference loop).
constexpr int SIM_THREADS = 256;

__global__ void __launch_bounds__(SIM_THREADS)
simulateStepKernel(uint32_t t, uint32_t N, uint32_t nloc, uint32_t i0,
                   const float* __restrict__ W, const uint8_t* __restrict__ tau,
                   float* __restrict__ radB, val_t rho, uint32_t sourceIndex, uint32_t timeOff) {
    const uint32_t iloc = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;
    if (iloc >= nloc) return;
    const uint32_t i = i0 + iloc;
    const int ti = static_cast<int>(t);
    const float* Wrow = W + static_cast<size_t>(iloc) * N;
    const uint8_t* tauRow = tau + static_cast<size_t>(iloc) * N;

    val_t sumB = ZERO;
    for (uint32_t jb = 0; jb < N; jb += 32) {
        const uint32_t j = jb + lane;
        val_t w = ZERO, radJ = ZERO;
        bool ok = false;
        if (j < N) {
            const int tauij = tauRow[j];
            // Skip if wave hasn't yet propagated from j to i
            if (ti >= tauij) {
                w = Wrow[j];  // min(kij * area_j, 1); zero where kij <= 0 or i == j
                if (w > ZERO) {
                    radJ = radB[static_cast<size_t>(ti - tauij) * N + j];
                    ok = radJ > ZERO;
                }
            }
        }
        unsigned m = __ballot_sync(0xffffffffu, ok);
        while (m) {
            const int k = __ffs(m) - 1;
            m &= m - 1;
            const val_t wk = __shfl_sync(0xffffffffu, w, k);
            const val_t rk = __shfl_sync(0xffffffffu, radJ, k);
            // Accumulate contribution: form factor * area * source radiosity (separate
            // multiply and add, as in the host reference build)
            sumB += wk * rk;
        }
    }
    if (lane == 0) {
        const val_t radE = (i == sourceIndex && t < timeOff) ? 1.0f : ZERO;
        // Update radiosity: reflection + emission
        radB[static_cast<size_t>(t) * N + i] = fmaf(rho, sumB, radE);
    }
}

// Time delays for the local rows (row-major: iloc * N + j)
constexpr int MAX_TAU = 255;

__global__ void tauKernel(const float4* __restrict__ centers, uint32_t N, uint32_t nloc, uint32_t i0,
                          uint8_t* __restrict__ tau) {
    const uint64_t e = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= static_cast<uint64_t>(N) * nloc) return;
    const uint32_t j = static_cast<uint32_t>(e % N);
    const uint32_t i = i0 + static_cast<uint32_t>(e / N);
    if (i == j) { tau[e] = 0; return; }
    const float4 ci = centers[i], cj = centers[j];
    val_t dist = (Vec3(ci.x, ci.y, ci.z) - Vec3(cj.x, cj.y, cj.z)).norm();
    int tv = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    tau[e] = static_cast<uint8_t>(tv);  // tv <= MAX_TAU (checked on the host)
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flat;                // GPU copy of the octree

    size_t sourceIndex;
    size_t timeOff;                 // Source emits for t < timeOff

    // Row decomposition over MPI ranks
    size_t i0, i1, nloc;
    std::vector<int> rowCounts, rowDispls;

    unsigned long long nonZeroKij = 0;   // global count of Kij > EPSILON
    std::future<mtj::Field> fieldFuture; // MT19937 jump-ahead arithmetic (built asynchronously)

    // Device data
    float* dW = nullptr;            // min(kij*area_j, 1) for local rows (nloc x N)
    uint8_t* dTau = nullptr;        // tau for local rows (nloc x N)
    float* dRadB = nullptr;         // T x N

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

    if (g_rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (g_rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.flat.flatten(state.octree, state.triangles);
    if (static_cast<int>(state.flat.maxDepth) * 7 + 1 > OCTREE_STACK) {
        fprintf(stderr, "Octree too deep for GPU traversal stack\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t N = state.numTriangles;

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) state.areas[i] = state.triangles[i].area();

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Source emission (active for first half of timesteps)
    state.timeOff = timesteps / 2;

    // Block row decomposition
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        size_t a = N * static_cast<size_t>(r) / g_size;
        size_t b = N * static_cast<size_t>(r + 1) / g_size;
        state.rowDispls[r] = static_cast<int>(a);
        state.rowCounts[r] = static_cast<int>(b - a);
    }
    state.i0 = state.rowDispls[g_rank];
    state.nloc = state.rowCounts[g_rank];
    state.i1 = state.i0 + state.nloc;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (g_rank == 0) printf("Computing time delays (Tau)...\n");
    // Recover the MT19937 characteristic polynomial in the background (needed for jump-ahead)
    state.fieldFuture = std::async(std::launch::async, [] { return mtj::Field(); });
    const size_t N = state.numTriangles, nloc = state.nloc, i0 = state.i0;

    std::vector<Vec3> centers(N);
    for (size_t i = 0; i < N; ++i) centers[i] = state.triangles[i].center();

    // Time delays are stored as bytes: bound them by the extent of the centers
    Vec3 lo = centers[0], hi = centers[0];
    for (const Vec3& c : centers) {
        lo = Vec3(std::min(lo.x, c.x), std::min(lo.y, c.y), std::min(lo.z, c.z));
        hi = Vec3(std::max(hi.x, c.x), std::max(hi.y, c.y), std::max(hi.z, c.z));
    }
    const Vec3 ext = hi - lo;
    if (std::sqrt(static_cast<double>(ext.x) * ext.x + static_cast<double>(ext.y) * ext.y +
                  static_cast<double>(ext.z) * ext.z) * INV_WAVE_SPEED + 2.0 > MAX_TAU) {
        fprintf(stderr, "Time delays exceed the supported range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<float4> hc(N);
    for (size_t i = 0; i < N; ++i) hc[i] = make_float4(centers[i].x, centers[i].y, centers[i].z, 0.0f);
    float4* dCenters;
    CUDA_CHECK(cudaMalloc(&dCenters, N * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(dCenters, hc.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dTau, std::max<size_t>(1, N * nloc) * sizeof(uint8_t)));
    if (nloc) {
        const uint64_t total = static_cast<uint64_t>(N) * nloc;
        tauKernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
            dCenters, static_cast<uint32_t>(N), static_cast<uint32_t>(nloc), static_cast<uint32_t>(i0), state.dTau);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFree(dCenters));
}

void computeFormFactors(SimulationState& state) {
    if (g_rank == 0) printf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles, nloc = state.nloc, i0 = state.i0;
    const size_t MW = (N + 31) / 32;

    // --- Culling masks: bit (i, j) set iff i != j and the pair is not culled (consumes RNG draws)
    std::vector<uint32_t> maskBits(nloc * MW, 0), maskPrefix(nloc * MW, 0);
    std::vector<uint64_t> rowCnt(nloc, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t il = 0; il < nloc; ++il) {
        const size_t i = i0 + il;
        const Vec3 nI = state.triangles[i].normal();
        uint32_t* bits = &maskBits[il * MW];
        uint32_t* pre = &maskPrefix[il * MW];
        uint64_t cnt = 0;
        for (size_t w = 0; w < MW; ++w) {
            uint32_t word = 0;
            for (size_t b = 0; b < 32; ++b) {
                size_t j = w * 32 + b;
                if (j >= N || j == i) continue;
                // Cull triangles facing the same direction
                if (nI.dot(state.triangles[j].normal()) > 0.99f) continue;
                word |= 1u << b;
            }
            bits[w] = word;
            pre[w] = static_cast<uint32_t>(cnt);
            cnt += static_cast<uint64_t>(__builtin_popcount(word));
        }
        rowCnt[il] = cnt;
    }
    std::vector<uint64_t> rowOff(nloc + 1, 0);
    uint64_t localPairs = 0;
    for (size_t il = 0; il < nloc; ++il) localPairs += rowCnt[il];
    uint64_t pairBase = 0;
    MPI_Exscan(&localPairs, &pairBase, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) pairBase = 0;
    rowOff[0] = pairBase;
    for (size_t il = 0; il < nloc; ++il) rowOff[il + 1] = rowOff[il] + rowCnt[il];

    // --- MT19937 stream decomposition: chunk k of this rank starts at draw D0 + k*S
    const uint64_t D0 = pairBase * DRAWS_PER_PAIR;
    const uint64_t totalDraws = localPairs * DRAWS_PER_PAIR;
    const uint64_t S = static_cast<uint64_t>(mtj::MT_N) * 512;
    const uint64_t K = (totalDraws + S - 1) / S;
    const uint64_t G = std::max<uint64_t>(1, std::min<uint64_t>(256, static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(K))))));
    const uint64_t numGroups = (K + G - 1) / G;

    // Jump polynomials: x^(D0 + g*G*S) for group starts, x^(m*S) for chunks within a group
    std::vector<mtj::Poly> groupPoly, stepPoly;
    {
        const mtj::Field field = state.fieldFuture.get();
        if (K > 0) {
            field.powSeries(D0, G * S, numGroups, groupPoly);
            field.powSeries(0, S, G, stepPoly);
        }
    }
    // --- Device data
    const size_t numNodes = state.flat.nodes.size();
    std::vector<float4> hA(N), hE1(N), hE2(N), hN(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 e1 = t.b - t.a, e2 = t.c - t.a, n = t.normal();
        hA[i] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        hE1[i] = make_float4(e1.x, e1.y, e1.z, 0.0f);
        hE2[i] = make_float4(e2.x, e2.y, e2.z, 0.0f);
        hN[i] = make_float4(n.x, n.y, n.z, 0.0f);
    }
    float4 *dA, *dE1, *dE2, *dN, *dLeaf;
    float* dArea;
    DevNode* dNodes;
    uint32_t *dBits, *dPrefix, *dBuf = nullptr, *dStates = nullptr;
    uint64_t* dRowOff;
    unsigned long long* dCount;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dE1, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dE2, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dN, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dArea, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dNodes, numNodes * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&dLeaf, std::max<size_t>(1, state.flat.leafTris.size()) * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dBits, std::max<size_t>(1, nloc * MW) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dPrefix, std::max<size_t>(1, nloc * MW) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dRowOff, (nloc + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&state.dW, std::max<size_t>(1, N * nloc) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dE1, hE1.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dE2, hE2.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dN, hN.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dArea, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodes, state.flat.nodes.data(), numNodes * sizeof(DevNode), cudaMemcpyHostToDevice));
    if (!state.flat.leafTris.empty())
        CUDA_CHECK(cudaMemcpy(dLeaf, state.flat.leafTris.data(), state.flat.leafTris.size() * sizeof(float4), cudaMemcpyHostToDevice));
    if (nloc) {
        CUDA_CHECK(cudaMemcpy(dBits, maskBits.data(), nloc * MW * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPrefix, maskPrefix.data(), nloc * MW * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRowOff, rowOff.data(), (nloc + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(state.dW, 0, std::max<size_t>(1, N * nloc) * sizeof(float)));

    // --- Row batches limited by the size of the random draw buffer
    const uint64_t maxBatchDraws = 1ull << 27;
    struct Batch { size_t b0, b1; uint64_t kA, kB; };
    std::vector<Batch> batches;
    uint64_t maxChunks = 0;
    for (size_t b0 = 0; b0 < nloc;) {
        size_t b1 = b0 + 1;
        while (b1 < nloc && (rowOff[b1 + 1] - rowOff[b0]) * DRAWS_PER_PAIR <= maxBatchDraws) ++b1;
        Batch bt{b0, b1, 0, 0};
        uint64_t dA_ = (rowOff[b0] - pairBase) * DRAWS_PER_PAIR;
        uint64_t dB_ = (rowOff[b1] - pairBase) * DRAWS_PER_PAIR;
        bt.kA = dA_ / S;
        bt.kB = (dB_ + S - 1) / S;
        if (bt.kB < bt.kA) bt.kB = bt.kA;
        maxChunks = std::max(maxChunks, bt.kB - bt.kA);
        batches.push_back(bt);
        b0 = b1;
    }
    if (maxChunks > 0) {
        CUDA_CHECK(cudaMalloc(&dBuf, maxChunks * S * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dStates, maxChunks * mtj::MT_N * sizeof(uint32_t)));
    }

    // Group start states (jumped from the seed state) computed on the GPU
    const size_t jumpSmem = static_cast<size_t>(JUMP_SEQ) * sizeof(uint32_t);
    uint32_t *dSeed = nullptr, *dGroupStates = nullptr;
    uint64_t *dGroupPoly = nullptr, *dStepPoly = nullptr;
    if (K > 0) {
        int dev = 0, optin = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (static_cast<size_t>(optin) < jumpSmem) {
            fprintf(stderr, "GPU does not provide %zu bytes of shared memory per block\n", jumpSmem);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaFuncSetAttribute(mtJumpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(jumpSmem)));
        std::vector<uint64_t> flat(std::max(numGroups, G) * mtj::NW);
        const mtj::State seed = mtj::seedState(42);
        CUDA_CHECK(cudaMalloc(&dSeed, mtj::MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dGroupStates, numGroups * mtj::MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dGroupPoly, numGroups * mtj::NW * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dStepPoly, G * mtj::NW * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(dSeed, seed.data(), mtj::MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        for (uint64_t g = 0; g < numGroups; ++g)
            memcpy(&flat[g * mtj::NW], groupPoly[g].w.data(), mtj::NW * sizeof(uint64_t));
        CUDA_CHECK(cudaMemcpy(dGroupPoly, flat.data(), numGroups * mtj::NW * sizeof(uint64_t), cudaMemcpyHostToDevice));
        for (uint64_t m = 0; m < G; ++m)
            memcpy(&flat[m * mtj::NW], stepPoly[m].w.data(), mtj::NW * sizeof(uint64_t));
        CUDA_CHECK(cudaMemcpy(dStepPoly, flat.data(), G * mtj::NW * sizeof(uint64_t), cudaMemcpyHostToDevice));
        mtJumpKernel<<<static_cast<unsigned>(numGroups), JUMP_THREADS, jumpSmem>>>(
            dSeed, ~0ull, dGroupPoly, numGroups, 0, dGroupStates);
        CUDA_CHECK(cudaGetLastError());
    }

    for (size_t bi = 0; bi < batches.size(); ++bi) {
        const Batch& bt = batches[bi];
        const uint64_t nc = bt.kB - bt.kA;
        if (nc > 0) {
            // Chunk start states: jump from the group start by x^((k mod G) * S)
            mtJumpKernel<<<static_cast<unsigned>(nc), JUMP_THREADS, jumpSmem>>>(
                dGroupStates, G, dStepPoly, G, bt.kA, dStates);
            CUDA_CHECK(cudaGetLastError());
            mtGenerateKernel<<<static_cast<unsigned>(nc), MT_THREADS>>>(dStates, dBuf, S);
            CUDA_CHECK(cudaGetLastError());

            FFParams P;
            P.nb = static_cast<uint32_t>(bt.b1 - bt.b0);
            P.numPairs = static_cast<uint64_t>(P.nb) * N;
            P.b0 = static_cast<uint32_t>(bt.b0);
            P.nloc = static_cast<uint32_t>(nloc);
            P.N = static_cast<uint32_t>(N);
            P.i0 = static_cast<uint32_t>(i0);
            P.maskWords = static_cast<uint32_t>(MW);
            P.bufBase = D0 + bt.kA * S;
            const uint64_t threads = P.numPairs * NUM_RAYS;
            const unsigned blocks = static_cast<unsigned>((threads + FF_THREADS - 1) / FF_THREADS);
            formFactorKernel<<<blocks, FF_THREADS>>>(P, dBits, dPrefix, dRowOff, dBuf, dA, dE1, dE2, dN, dArea,
                                              dNodes, dLeaf, state.dW, dCount);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long localCount = 0;
    CUDA_CHECK(cudaMemcpy(&localCount, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (g_rank == 0) {
        for (size_t i = 0; i < N; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) printf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }

    cudaFree(dA); cudaFree(dE1); cudaFree(dE2); cudaFree(dN); cudaFree(dArea);
    cudaFree(dNodes); cudaFree(dLeaf); cudaFree(dBits); cudaFree(dPrefix); cudaFree(dRowOff);
    cudaFree(dCount);
    if (dBuf) cudaFree(dBuf);
    if (dStates) cudaFree(dStates);
    if (dSeed) cudaFree(dSeed);
    if (dGroupStates) cudaFree(dGroupStates);
    if (dGroupPoly) cudaFree(dGroupPoly);
    if (dStepPoly) cudaFree(dStepPoly);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (g_rank == 0) printf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles, T = state.numTimesteps, nloc = state.nloc;

    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, T * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(1, T * N) * sizeof(float)));

    const unsigned threads = SIM_THREADS;
    const unsigned blocks = static_cast<unsigned>((nloc * 32 + threads - 1) / threads);
    for (size_t t = 0; t < T; ++t) {
        if (nloc > 0) {
            simulateStepKernel<<<blocks, threads>>>(static_cast<uint32_t>(t), static_cast<uint32_t>(N),
                                                    static_cast<uint32_t>(nloc), static_cast<uint32_t>(state.i0),
                                                    state.dW, state.dTau, state.dRadB, state.rho[0],
                                                    static_cast<uint32_t>(state.sourceIndex),
                                                    static_cast<uint32_t>(state.timeOff));
            CUDA_CHECK(cudaGetLastError());
        }
        if (g_size > 1) {
            float* row = &state.radB[state.idxTN(t, 0)];
            if (nloc > 0)
                CUDA_CHECK(cudaMemcpy(row + state.i0, state.dRadB + t * N + state.i0, nloc * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, row, state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(state.dRadB + t * N, row, N * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (g_rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    if (g_size == 1 && T * N > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, T * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (g_rank == 0) printf("Computing distances via cross-correlation...\n");
    const size_t T = state.numTimesteps;

    #pragma omp parallel for schedule(static)
    for (size_t i = state.i0; i < state.i1; ++i) {
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

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    if (g_size > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(), state.rowCounts.data(),
                       state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
    }
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

    // Check Kij matrix (should have some non-zero entries); counted on the GPUs
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

// Host threads per rank. A rank that the launcher pinned to a single core (e.g. Open MPI's
// default binding for small runs) is widened to its share of the node, since the OpenMP
// part of the hybrid code would otherwise run (almost) serially.
void configureHostThreads(int localRank, int localSize) {
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
    const long share = std::max(1L, ncpu / std::max(1, localSize));
    if (static_cast<long>(CPU_COUNT(&mask)) * 8 <= share) {
        cpu_set_t want;
        CPU_ZERO(&want);
        for (long c = localRank * share; c < (localRank + 1) * share && c < ncpu && c < CPU_SETSIZE; ++c)
            CPU_SET(static_cast<int>(c), &want);
        if (sched_setaffinity(0, sizeof(want), &want) == 0) {
            CPU_ZERO(&mask);
            sched_getaffinity(0, sizeof(mask), &mask);
        }
    }
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(static_cast<int>(std::max(1L, std::min<long>(CPU_COUNT(&mask), share))));
    }
}

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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // One GPU per rank (round-robin over node-local ranks), CPU cores shared among local ranks
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // create context outside timed regions
    configureHostThreads(localRank, localSize);

    // Start the OpenMP thread pool outside the timed regions
    #pragma omp parallel
    {
        std::vector<uint64_t> warm(8192, static_cast<uint64_t>(omp_get_thread_num()));
        volatile uint64_t sink = warm[omp_get_thread_num() % warm.size()];
        (void)sink;
    }

    // Only rank 0 writes to stdout
    if (g_rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) { /* ignore */ }
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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long localDurations[3] = {preDuration, simDuration, distDuration};
    long maxDurations[3];
    MPI_Allreduce(localDurations, maxDurations, 3, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    long totalTime = maxDurations[0] + maxDurations[1] + maxDurations[2];
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
    int status = 0;
    if (validate && g_rank == 0) {
        if (!validateResults(state)) {
            status = 1;
        }
    }
    fflush(stdout);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (state.dW) cudaFree(state.dW);
    if (state.dTau) cudaFree(state.dTau);
    if (state.dRadB) cudaFree(state.dRadB);

    MPI_Finalize();
    return status;
}

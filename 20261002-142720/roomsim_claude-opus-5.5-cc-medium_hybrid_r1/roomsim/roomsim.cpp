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
 *  - MPI: rows (receiver triangles) are block-distributed over ranks. Each rank
 *    owns complete rows of Kij/Tau, so the only communication is an allgather
 *    of one radiosity row per timestep plus a final gather of distances.
 *  - CUDA: form factors (ray casting through a flattened octree), the wave
 *    propagation and the cross-correlation run on one GPU per rank.
 *  - OpenMP: host-side O(N^2) work (culling / Tau) overlaps the GPU kernels.
 *
 * The form factors consume a single sequential std::mt19937(42) stream in the
 * original. To reproduce it exactly in parallel, the generator is jumped ahead
 * with GF(2) polynomial arithmetic (characteristic polynomial obtained via
 * Berlekamp-Massey), and each CUDA block regenerates its own stream segment.
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
constexpr val_t VAL_MAX = std::numeric_limits<val_t>::max();

static int g_rank = 0;
static int g_size = 1;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr HD Vec3() : x(0), y(0), z(0) {}
    constexpr HD Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr HD Vec3(val_t v) : x(v), y(v), z(v) {}

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

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
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
};

// Flattened octree for the GPU. Each internal node stores the boxes of its
// children contiguously (2 float4 per child: center|ref, halfExtent|count);
// count < 0 marks an internal child (ref = node id), otherwise the child is a
// leaf whose triangles are stored inline at leafTris[3*ref ...] as
// (v0|index, e1, e2) with e1 = v1 - v0, e2 = v2 - v0.
struct OctreeFlat {
    std::vector<int2> nodes;        // (first entry float4 index, number of children)
    std::vector<float4> entries;
    std::vector<float4> leafTris;
    int rootLeafStart = 0, rootLeafCount = 0;
    int maxDepth = 0;

    static float bitsToFloat(int v) { float f; std::memcpy(&f, &v, sizeof(f)); return f; }

    int addLeaf(const Octree& node, const std::vector<Triangle>& tris) {
        int start = static_cast<int>(leafTris.size() / 3);
        for (size_t idx : node.triangleIndices) {
            const Triangle& t = tris[idx];
            Vec3 e1 = t.b - t.a;
            Vec3 e2 = t.c - t.a;
            leafTris.push_back(make_float4(t.a.x, t.a.y, t.a.z, bitsToFloat(static_cast<int>(idx))));
            leafTris.push_back(make_float4(e1.x, e1.y, e1.z, 0.0f));
            leafTris.push_back(make_float4(e2.x, e2.y, e2.z, 0.0f));
        }
        return start;
    }

    int addInternal(const Octree& node, const std::vector<Triangle>& tris, int depth) {
        maxDepth = std::max(maxDepth, depth);
        int id = static_cast<int>(nodes.size());
        nodes.push_back(make_int2(0, 0));
        int k = 0;
        for (int c = 0; c < 8; ++c) k += node.children[c] ? 1 : 0;
        int first = static_cast<int>(entries.size());
        entries.resize(entries.size() + 2 * k);
        int e = first;
        for (int c = 0; c < 8; ++c) {
            const Octree* ch = node.children[c].get();
            if (!ch) continue;
            int ref, cnt;
            if (!ch->triangleIndices.empty()) {
                ref = addLeaf(*ch, tris);
                cnt = static_cast<int>(ch->triangleIndices.size());
            } else {
                ref = addInternal(*ch, tris, depth + 1);
                cnt = -1;
            }
            entries[e] = make_float4(ch->center.x, ch->center.y, ch->center.z, bitsToFloat(ref));
            entries[e + 1] = make_float4(ch->halfExtent.x, ch->halfExtent.y, ch->halfExtent.z, bitsToFloat(cnt));
            e += 2;
        }
        nodes[id] = make_int2(first, k);
        return id;
    }

    void build(const Octree& root, const std::vector<Triangle>& tris) {
        if (!root.triangleIndices.empty()) {
            rootLeafStart = addLeaf(root, tris);
            rootLeafCount = static_cast<int>(root.triangleIndices.size());
        } else {
            addInternal(root, tris, 0);
        }
        if (nodes.empty()) nodes.push_back(make_int2(0, 0));
        if (entries.empty()) entries.push_back(make_float4(0, 0, 0, 0));
        if (leafTris.empty()) leafTris.push_back(make_float4(0, 0, 0, 0));
    }
};

struct GpuOctree {
    const int2* nodes;
    const float4* entries;
    const float4* leafTris;
    int rootLeafStart, rootLeafCount;
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
// Random Number Generation: Mersenne Twister with GF(2) jump-ahead
// ============================================================================
//
// The original draws all random numbers from one std::mt19937(42) through
// std::uniform_real_distribution<float>(0,1), which consumes exactly one 32-bit
// output per float. Output n is temper(x[624 + n]) where x[0..623] is the
// seeded state and x[k] = x[k-227] ^ twist(x[k-624], x[k-623]).
//
// Pair (i,j) that is not culled consumes 4*NUM_RAYS = 64 consecutive outputs.
// 39 pairs * 64 = 2496 = 4 * 624 outputs form one "round", so round r is
// generated from the state window x[2496 r .. 2496 r + 624).

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DEG = 19937;                     // degree of characteristic polynomial
constexpr int POLY_WORDS = 312;                   // 64-bit words holding a poly of degree <= 19967
constexpr int MT_EXT_WINDOWS = 33;                // 33*624 >= 19937 + 624
constexpr int PAIRS_PER_ROUND = 39;
constexpr int RAND_PER_PAIR = 4 * NUM_RAYS;       // 64
constexpr int RAND_PER_ROUND = PAIRS_PER_ROUND * RAND_PER_PAIR;  // 2496
static_assert(RAND_PER_ROUND == 4 * MT_N, "round must be 4 MT windows");
static_assert(PAIRS_PER_ROUND * NUM_RAYS == MT_N, "one ray per thread");

HD inline uint32_t mtWord(uint32_t x0, uint32_t x1, uint32_t xm) {
    uint32_t y = (x0 & 0x80000000u) | (x1 & 0x7fffffffu);
    return xm ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0,1) on one mt19937 output (libstdc++)
HD inline val_t mtToFloat(uint32_t y) {
    val_t f = static_cast<val_t>(y) * 2.3283064365386962890625e-10f;  // / 2^32
    return f >= 1.0f ? 0x1.fffffep-1f : f;
}

static void mtSeedWindow(uint32_t seed, uint32_t* w) {
    w[0] = seed;
    for (uint32_t i = 1; i < MT_N; ++i) w[i] = 1812433253u * (w[i - 1] ^ (w[i - 1] >> 30)) + i;
}

using Poly = std::vector<uint64_t>;   // GF(2) polynomial, bit k = coefficient of x^k

// GF(2) polynomial arithmetic modulo the MT19937 characteristic polynomial
struct MtPoly {
    Poly p;                               // monic, degree MT_DEG
    std::vector<Poly> pShift;             // p << s, s = 0..63 (POLY_WORDS + 1 words)

    // Characteristic polynomial via Berlekamp-Massey on the MSB sequence
    void init() {
        const int nseq = 2 * MT_DEG + 64;
        std::vector<uint32_t> x(nseq);
        mtSeedWindow(42u, x.data());
        for (int k = MT_N; k < nseq; ++k) x[k] = mtWord(x[k - MT_N], x[k - MT_N + 1], x[k - (MT_N - MT_M)]);

        const int nw = (nseq + 63) / 64 + 2 * POLY_WORDS + 8;
        std::vector<uint64_t> R(nw, 0);   // R[j] = s[nseq-1-j]
        for (int k = 0; k < nseq; ++k) {
            if (x[k] >> 31) {
                int j = nseq - 1 - k;
                R[j >> 6] |= 1ull << (j & 63);
            }
        }
        // RS[s][w] = 64 bits of R starting at bit 64*w + s
        std::vector<std::vector<uint64_t>> RS(64, std::vector<uint64_t>(nw - 1));
        for (int s = 0; s < 64; ++s)
            for (int w = 0; w + 1 < nw; ++w)
                RS[s][w] = s ? (R[w] >> s) | (R[w + 1] << (64 - s)) : R[w];
        const int cw = 2 * POLY_WORDS + 4;
        std::vector<uint64_t> C(cw, 0), B(cw, 0), T(cw, 0);
        C[0] = 1; B[0] = 1;
        int L = 0, m = 1, LB = 0;   // LB: length of B (degree bound)
        auto xorShift = [&](std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, int sh, int srcBits) {
            int ws = sh >> 6, bs = sh & 63;
            int lim = std::min(srcBits / 64 + 1, cw);
            for (int q = 0; q < lim; ++q) {
                uint64_t v = src[q];
                if (q + ws < cw) dst[q + ws] ^= v << bs;
                if (bs && q + ws + 1 < cw) dst[q + ws + 1] ^= v >> (64 - bs);
            }
        };
        for (int n = 0; n < 2 * MT_DEG + 8; ++n) {
            int o = nseq - 1 - n;
            uint64_t acc = 0;
            int lw = L / 64 + 1;
            const uint64_t* rs = RS[o & 63].data() + (o >> 6);
            for (int q = 0; q < lw; ++q) acc ^= C[q] & rs[q];
            if (__builtin_popcountll(acc) & 1) {
                if (2 * L <= n) {
                    const int lw2 = L / 64 + 1;
                    std::copy(C.begin(), C.begin() + lw2, T.begin());
                    xorShift(C, B, m, LB);
                    std::swap(B, T);
                    std::fill(B.begin() + lw2, B.end(), 0);
                    LB = L;
                    L = n + 1 - L;
                    m = 1;
                } else {
                    xorShift(C, B, m, LB);
                    ++m;
                }
            } else {
                ++m;
            }
        }
        if (L != MT_DEG) {
            fprintf(stderr, "Berlekamp-Massey failed (L=%d)\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        p.assign(POLY_WORDS + 1, 0);
        for (int i = 0; i <= L; ++i) {
            if ((C[i >> 6] >> (i & 63)) & 1) {
                int k = L - i;
                p[k >> 6] |= 1ull << (k & 63);
            }
        }
        pShift.assign(64, Poly(POLY_WORDS + 2, 0));
        for (int s = 0; s < 64; ++s) {
            for (int q = 0; q <= POLY_WORDS; ++q) {
                pShift[s][q] ^= p[q] << s;
                if (s) pShift[s][q + 1] ^= p[q] >> (64 - s);
            }
        }
    }

    // Reduce a polynomial of degree < 2*MT_DEG (2*POLY_WORDS words) in place
    void reduce(Poly& a) const {
        for (int k = 2 * MT_DEG; k >= MT_DEG; --k) {
            if ((a[k >> 6] >> (k & 63)) & 1) {
                int off = k - MT_DEG;
                const uint64_t* ps = pShift[off & 63].data();
                uint64_t* dst = a.data() + (off >> 6);
                for (int q = 0; q < POLY_WORDS + 1; ++q) dst[q] ^= ps[q];
            }
        }
        a.resize(POLY_WORDS);
    }

    static uint64_t spread(uint32_t v) {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2)) & 0x3333333333333333ull;
        x = (x | (x << 1)) & 0x5555555555555555ull;
        return x;
    }

    Poly squareMod(const Poly& a) const {
        Poly r(2 * POLY_WORDS + 2, 0);
        for (int q = 0; q < POLY_WORDS; ++q) {
            r[2 * q] = spread(static_cast<uint32_t>(a[q]));
            r[2 * q + 1] = spread(static_cast<uint32_t>(a[q] >> 32));
        }
        reduce(r);
        return r;
    }

    void mulXMod(Poly& a) const {
        uint64_t carry = 0;
        for (int q = 0; q < POLY_WORDS; ++q) {
            uint64_t nc = a[q] >> 63;
            a[q] = (a[q] << 1) | carry;
            carry = nc;
        }
        if ((a[MT_DEG >> 6] >> (MT_DEG & 63)) & 1) {
            for (int q = 0; q < POLY_WORDS; ++q) a[q] ^= p[q];
        }
    }

    // x^e mod p
    Poly powX(uint64_t e) const {
        Poly r(POLY_WORDS, 0);
        r[0] = 1;
        if (e == 0) return r;
        int top = 63 - __builtin_clzll(e);
        for (int b = top; b >= 0; --b) {
            if (b != top) r = squareMod(r);
            if ((e >> b) & 1) mulXMod(r);
        }
        return r;
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& e1, const Vec3& e2) {
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

// ============================================================================
// Visibility Testing (Octree-accelerated, GPU)
// ============================================================================

constexpr int OCT_STACK = 64;

// Segment/box overlap test (mid = p1 + d, d = (p2 - p1) * 0.5, ad = |d|)
__device__ __forceinline__ bool rayIntersectsBox(const Vec3& center, const Vec3& h,
                                                 const Vec3& mid, const Vec3& d, const Vec3& ad) {
    Vec3 c = mid - center;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

__device__ __forceinline__ bool leafBlocks(const float4* __restrict__ leafTris, int start, int count,
                                           const Vec3& from, const Vec3& dirNorm, val_t rayLen,
                                           int srcTriIdx, int dstTriIdx) {
    const float4* lt = leafTris + 3 * start;
    for (int k = 0; k < count; ++k, lt += 3) {
        float4 A = __ldg(lt);
        int idx = __float_as_int(A.w);
        if (idx == srcTriIdx || idx == dstTriIdx) continue;
        float4 E1 = __ldg(lt + 1);
        float4 E2 = __ldg(lt + 2);
        val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(A.x, A.y, A.z),
                                          Vec3(E1.x, E1.y, E1.z), Vec3(E2.x, E2.y, E2.z));
        if (dist > EPSILON && dist < rayLen - EPSILON) return true;  // Ray is blocked
    }
    return false;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Warp-cooperative packet traversal: the warp walks the union of the nodes
// visited by its rays, but every stack entry carries the mask of lanes whose
// ray passed that node's box test, so each lane performs exactly the box and
// triangle tests of a per-ray traversal (and stops at its first hit).
// Must be called by all lanes in `warpMask` (warp-uniform control flow).
__device__ bool isRayBlockedWarp(bool active, const Vec3& from, const Vec3& to, const GpuOctree& oct,
                                 int srcTriIdx, int dstTriIdx, unsigned warpMask,
                                 int* __restrict__ stkNode, unsigned* __restrict__ stkMask) {
    const int lane = threadIdx.x & 31;
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    bool blocked = false;
    bool live = active;
    if (live && rayLen < EPSILON) {
        blocked = true;
        live = false;
    }
    Vec3 dirNorm = dir / rayLen;

    if (oct.rootLeafCount > 0) {
        if (live)
            blocked = leafBlocks(oct.leafTris, oct.rootLeafStart, oct.rootLeafCount, from, dirNorm, rayLen,
                                 srcTriIdx, dstTriIdx);
        __syncwarp(warpMask);
        return blocked;
    }

    const Vec3 d = (to - from) * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    unsigned liveMask = __ballot_sync(warpMask, live);
    int sp = 0;
    if (liveMask) {
        if (lane == 0) {
            stkNode[0] = 0;
            stkMask[0] = liveMask;
        }
        sp = 1;
    }
    __syncwarp(warpMask);
    while (sp > 0) {
        --sp;
        const int node = stkNode[sp];
        const unsigned m = stkMask[sp] & liveMask;
        __syncwarp(warpMask);
        if (!m) continue;
        const bool inNode = (m >> lane) & 1u;
        const int2 nd = __ldg(oct.nodes + node);
        const float4* e = oct.entries + nd.x;
        for (int c = 0; c < nd.y; ++c, e += 2) {
            const float4 A = __ldg(e);
            const float4 B = __ldg(e + 1);
            const bool pass = inNode && !blocked &&
                              rayIntersectsBox(Vec3(A.x, A.y, A.z), Vec3(B.x, B.y, B.z), mid, d, ad);
            const unsigned pm = __ballot_sync(warpMask, pass);
            if (!pm) continue;
            const int ref = __float_as_int(A.w);
            const int cnt = __float_as_int(B.w);
            if (cnt < 0) {
                if (lane == 0) {
                    stkNode[sp] = ref;
                    stkMask[sp] = pm;
                }
                ++sp;
                __syncwarp(warpMask);
            } else {
                const float4* lt = oct.leafTris + 3 * ref;
                for (int k = 0; k < cnt; ++k, lt += 3) {
                    const float4 T0 = __ldg(lt);
                    const float4 T1 = __ldg(lt + 1);
                    const float4 T2 = __ldg(lt + 2);
                    const int idx = __float_as_int(T0.w);
                    if (pass && !blocked && idx != srcTriIdx && idx != dstTriIdx) {
                        val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(T0.x, T0.y, T0.z),
                                                          Vec3(T1.x, T1.y, T1.z), Vec3(T2.x, T2.y, T2.z));
                        if (dist > EPSILON && dist < rayLen - EPSILON) blocked = true;  // Ray is blocked
                    }
                }
                liveMask = __ballot_sync(warpMask, live && !blocked);
            }
        }
        if (!liveMask) break;
    }
    __syncwarp(warpMask);
    return blocked;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

HD inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Generate one MT window (624 words) from the previous one; blockDim >= 624
__device__ __forceinline__ void twistWindow(const uint32_t* prev, uint32_t* next) {
    int w = threadIdx.x;
    if (w < MT_N - MT_M) next[w] = mtWord(prev[w], prev[w + 1], prev[w + MT_M]);
    __syncthreads();
    if (w >= MT_N - MT_M && w < 2 * (MT_N - MT_M))
        next[w] = mtWord(prev[w], prev[w + 1], next[w - (MT_N - MT_M)]);
    __syncthreads();
    if (w >= 2 * (MT_N - MT_M) && w < MT_N)
        next[w] = mtWord(prev[w], w + 1 < MT_N ? prev[w + 1] : next[0], next[w - (MT_N - MT_M)]);
    __syncthreads();
}

// dst = poly(T) applied to src (MT jump-ahead), one state per block iteration
__global__ void __launch_bounds__(MT_N)
mtJumpKernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int count,
             const uint64_t* __restrict__ poly, uint32_t* __restrict__ scratch) {
    __shared__ uint64_t sp[POLY_WORDS];
    for (int q = threadIdx.x; q < POLY_WORDS; q += blockDim.x) sp[q] = poly[q];
    uint32_t* ext = scratch + static_cast<size_t>(blockIdx.x) * MT_EXT_WINDOWS * MT_N;
    const int w = threadIdx.x;
    for (int item = blockIdx.x; item < count; item += gridDim.x) {
        ext[w] = src[static_cast<size_t>(item) * MT_N + w];
        __syncthreads();
        for (int k = 1; k < MT_EXT_WINDOWS; ++k) twistWindow(ext + (k - 1) * MT_N, ext + k * MT_N);
        uint32_t acc = 0;
        for (int q = 0; q < POLY_WORDS; ++q) {
            uint64_t bits = sp[q];
            while (bits) {
                int b = __ffsll(static_cast<long long>(bits)) - 1;
                bits &= bits - 1;
                acc ^= ext[64 * q + b + w];
            }
        }
        dst[static_cast<size_t>(item) * MT_N + w] = acc;
        __syncthreads();
    }
}

// Generate random numbers for one chunk: block b produces rounds
// [b*Rs + c*Rc, b*Rs + (c+1)*Rc) of this rank's round range into slots
// buf[(b*Rc + k) * 2496 ...] and keeps its MT state across chunks.
__global__ void __launch_bounds__(MT_N)
mtGenerateKernel(uint32_t* __restrict__ states, long long roundBase, long long Rs, long long roundEnd,
                 long long chunk, int Rc, val_t* __restrict__ buf) {
    __shared__ uint32_t win[5][MT_N];
    const int t = threadIdx.x;
    uint32_t* st = states + static_cast<size_t>(blockIdx.x) * MT_N;
    long long first = static_cast<long long>(blockIdx.x) * Rs + chunk * Rc;   // local round offset
    int nr = static_cast<int>(max(0LL, min(static_cast<long long>(Rc),
                                           min(Rs - chunk * Rc, roundEnd - roundBase - first))));
    if (nr <= 0) return;
    win[0][t] = st[t];
    __syncthreads();
    val_t* out = buf + static_cast<size_t>(blockIdx.x) * Rc * RAND_PER_ROUND;
    for (int k = 0; k < nr; ++k) {
        for (int q = 0; q < 4; ++q) twistWindow(win[q], win[q + 1]);
        for (int q = 0; q < 4; ++q)
            out[static_cast<size_t>(k) * RAND_PER_ROUND + q * MT_N + t] = mtToFloat(mtTemper(win[q + 1][t]));
        win[0][t] = win[4][t];
        __syncthreads();
    }
    st[t] = win[0][t];
}

struct KijParams {
    long long roundBase;         // first round handled by this rank
    long long roundsPerBlock;    // Rs: rounds per generator block
    long long roundEnd;          // exclusive
    long long pairBegin, pairEnd;// global pair range owned by this rank
    const long long* rowStart;   // global prefix count of non-culled pairs (N+1)
    const long long* culledOff;  // local rows (R+1)
    const int* culledCols;
    int r0, r1, n;
    const Triangle* tris;
    GpuOctree oct;
    val_t* kij;                  // local rows, R x N
};

constexpr int TRACE_WARPS = 8;
constexpr int TASKS_PER_ROUND = (PAIRS_PER_ROUND + 1) / 2;   // two pairs (32 rays) per warp task

// Ray tracing of one chunk: independent warps fetch tasks dynamically
__global__ void __launch_bounds__(TRACE_WARPS * 32)
kijTraceKernel(KijParams P, const val_t* __restrict__ buf, long long chunk, int Rc, int G,
               unsigned long long* __restrict__ counter) {
    __shared__ int stkNode[TRACE_WARPS][OCT_STACK];
    __shared__ unsigned stkMask[TRACE_WARPS][OCT_STACK];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int half = lane >> 4;
    const int ray = lane & 15;
    const unsigned long long totalTasks = static_cast<unsigned long long>(G) * Rc * TASKS_PER_ROUND;

    while (true) {
        unsigned long long task = 0;
        if (lane == 0) task = atomicAdd(counter, 1ull);
        task = __shfl_sync(0xffffffffu, task, 0);
        if (task >= totalTasks) break;

        const int slot = static_cast<int>(task / TASKS_PER_ROUND);
        const int p = 2 * static_cast<int>(task % TASKS_PER_ROUND) + half;   // pair within round
        const int b = slot / Rc;
        const int k = slot % Rc;
        const long long localRound = static_cast<long long>(b) * P.roundsPerBlock + chunk * Rc + k;
        const long long r = P.roundBase + localRound;
        const long long pidx = r * PAIRS_PER_ROUND + p;
        const bool active = p < PAIRS_PER_ROUND && chunk * Rc + k < P.roundsPerBlock &&
                            r < P.roundEnd && pidx >= P.pairBegin && pidx < P.pairEnd;

        int i = 0, j = 0;
        if (active) {
            int lo = P.r0, hi = P.r1 - 1;   // largest row with rowStart <= pidx
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (P.rowStart[mid] <= pidx) lo = mid; else hi = mid - 1;
            }
            i = lo;
            long long kk = pidx - P.rowStart[i];
            const int* cols = P.culledCols + P.culledOff[i - P.r0];
            int cnt = static_cast<int>(P.culledOff[i - P.r0 + 1] - P.culledOff[i - P.r0]);
            int a = 0, e = cnt;              // number of q with cols[q] - q <= kk
            while (a < e) {
                int mid = (a + e) >> 1;
                if (cols[mid] - mid <= kk) a = mid + 1; else e = mid;
            }
            j = static_cast<int>(kk) + a;
        }

        float4 rr = make_float4(0.f, 0.f, 0.f, 0.f);
        if (active)
            rr = __ldg(reinterpret_cast<const float4*>(
                buf + static_cast<size_t>(slot) * RAND_PER_ROUND + p * RAND_PER_PAIR + ray * 4));

        const Triangle triI = P.tris[i];
        const Triangle triJ = P.tris[j];
        Vec3 pI = pointInTriangle(triI, rr.x, rr.y);
        Vec3 pJ = pointInTriangle(triJ, rr.z, rr.w);
        bool blocked = isRayBlockedWarp(active, pI, pJ, P.oct, i, j, 0xffffffffu,
                                        stkNode[warp], stkMask[warp]);
        val_t c = ZERO;
        if (active && !blocked) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosPhiI = cosPhi(v, triI.normal());
                val_t cosPhiJ = cosPhi(-v, triJ.normal());
                if (cosPhiI > ZERO && cosPhiJ > ZERO) c = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }

        // Sum the 16 rays of each pair in ray order (as the sequential loop does)
        val_t kij = ZERO;
        for (int q = 0; q < NUM_RAYS; ++q) kij += __shfl_sync(0xffffffffu, c, (half << 4) + q);
        if (ray == 0 && active) P.kij[static_cast<size_t>(i - P.r0) * P.n + j] = kij * INV_NUM_RAYS;
    }
}

// Convert Kij into propagation weights min(Kij*area_j, 1) and count non-zero Kij
__global__ void weightKernel(val_t* __restrict__ kij, const val_t* __restrict__ areas,
                             size_t total, int n, unsigned long long* nonZero) {
    unsigned long long cnt = 0;
    for (size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<size_t>(gridDim.x) * blockDim.x) {
        val_t k = kij[e];
        if (k > EPSILON) ++cnt;
        kij[e] = k > ZERO ? fminf(k * areas[e % n], ONE) : ZERO;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
    if ((threadIdx.x & 31) == 0 && cnt) atomicAdd(nonZero, cnt);
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
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

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Distribution: this rank owns receiver rows [rowBegin, rowEnd)
    int rowBegin = 0, rowEnd = 0;
    std::vector<int> rowCounts, rowDispls;
    unsigned long long nonZeroKij = 0;

    // Device data
    Triangle* dTris = nullptr;
    int2* dNodes = nullptr;
    float4* dEntries = nullptr;
    float4* dLeafTris = nullptr;
    GpuOctree oct{};
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dW = nullptr;            // local rows: min(Kij*area_j,1), 0 if Kij<=0
    uint16_t* dTau = nullptr;       // local rows
    val_t* dRadB = nullptr;         // T x N
    val_t* dDist = nullptr;         // local rows
    cudaStream_t stream = nullptr;
    int numSMs = 1;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    int localRows() const { return rowEnd - rowBegin; }
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

    const size_t n = state.numTriangles;

    // Initialize areas
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // Row distribution over ranks
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        size_t b = n * static_cast<size_t>(r) / g_size;
        size_t e = n * static_cast<size_t>(r + 1) / g_size;
        state.rowDispls[r] = static_cast<int>(b);
        state.rowCounts[r] = static_cast<int>(e - b);
    }
    state.rowBegin = state.rowDispls[g_rank];
    state.rowEnd = state.rowBegin + state.rowCounts[g_rank];

    // Device setup
    OctreeFlat flat;
    flat.build(state.octree, state.triangles);
    if (7 * flat.maxDepth + 1 > OCT_STACK) {
        fprintf(stderr, "Octree too deep for traversal stack (%d)\n", flat.maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&state.numSMs, cudaDevAttrMultiProcessorCount, dev));

    const size_t R = static_cast<size_t>(state.localRows());
    CUDA_CHECK(cudaMalloc(&state.dTris, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, flat.nodes.size() * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&state.dEntries, flat.entries.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dLeafTris, flat.leafTris.size() * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dW, std::max<size_t>(1, R * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, std::max<size_t>(1, R * n) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, timesteps * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDist, std::max<size_t>(1, R) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, flat.nodes.data(), flat.nodes.size() * sizeof(int2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dEntries, flat.entries.data(), flat.entries.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dLeafTris, flat.leafTris.data(), flat.leafTris.size() * sizeof(float4), cudaMemcpyHostToDevice));
    state.oct = GpuOctree{state.dNodes, state.dEntries, state.dLeafTris, flat.rootLeafStart, flat.rootLeafCount};
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dW, 0, std::max<size_t>(1, R * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(1, timesteps * n) * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase (Tau on host with OpenMP, Kij on GPU, overlapped)
// ============================================================================

void precompute(SimulationState& state) {
    const int n = static_cast<int>(state.numTriangles);
    const int r0 = state.rowBegin, r1 = state.rowEnd;
    const int R = r1 - r0;
    const std::vector<Triangle>& tris = state.triangles;

    // MT19937 characteristic polynomial (independent of the mesh), overlapped with culling
    MtPoly mp;
    std::thread bmThread([&mp] { mp.init(); });

    // --- Culling: pairs whose normals face the same way consume no random numbers
    std::vector<std::vector<int>> culled(R);
    std::vector<long long> localCounts(R);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int li = 0; li < R; ++li) {
        int i = r0 + li;
        const Vec3 ni = tris[i].normal();
        auto& lst = culled[li];
        for (int j = 0; j < n; ++j) {
            if (i == j || ni.dot(tris[j].normal()) > 0.99f) lst.push_back(j);
        }
        localCounts[li] = n - static_cast<long long>(lst.size());
    }

    std::vector<long long> allCounts(n);
    {
        std::vector<int> cnts(g_size), displs(g_size);
        for (int r = 0; r < g_size; ++r) { cnts[r] = state.rowCounts[r]; displs[r] = state.rowDispls[r]; }
        MPI_Allgatherv(localCounts.data(), R, MPI_LONG_LONG, allCounts.data(), cnts.data(), displs.data(),
                       MPI_LONG_LONG, MPI_COMM_WORLD);
    }
    std::vector<long long> rowStart(n + 1, 0);
    for (int i = 0; i < n; ++i) rowStart[i + 1] = rowStart[i] + allCounts[i];

    std::vector<long long> culledOff(R + 1, 0);
    for (int li = 0; li < R; ++li) culledOff[li + 1] = culledOff[li] + static_cast<long long>(culled[li].size());
    std::vector<int> culledCols(std::max<long long>(1, culledOff[R]));
    #pragma omp parallel for schedule(static)
    for (int li = 0; li < R; ++li)
        std::copy(culled[li].begin(), culled[li].end(), culledCols.begin() + culledOff[li]);

    const long long pairBegin = rowStart[r0], pairEnd = rowStart[r1];
    const long long roundBase = pairBegin / PAIRS_PER_ROUND;
    const long long roundEnd = (pairEnd + PAIRS_PER_ROUND - 1) / PAIRS_PER_ROUND;
    const long long localRounds = pairEnd > pairBegin ? roundEnd - roundBase : 0;

    // --- Kij on the GPU
    long long* dRowStart = nullptr;
    long long* dCulledOff = nullptr;
    int* dCulledCols = nullptr;
    uint32_t* dStates = nullptr;
    uint32_t* dSeed = nullptr;
    uint64_t* dPolys = nullptr;
    uint32_t* dScratch = nullptr;

    val_t* dRand = nullptr;
    unsigned long long* dCounters = nullptr;

    if (localRounds > 0) {
        // Generator blocks (one MT jump each); Rs rounds per block
        long long targetBlocks = static_cast<long long>(state.numSMs) * 4;
        long long Rs = (localRounds + targetBlocks - 1) / targetBlocks;
        int G = static_cast<int>((localRounds + Rs - 1) / Rs);
        int levels = 0;
        while ((1 << levels) < G) ++levels;
        // Chunk: Rc rounds per generator block, ~256 MB of random numbers
        const long long budgetRounds = (256ll << 20) / (RAND_PER_ROUND * static_cast<long long>(sizeof(val_t)));
        int Rc = static_cast<int>(std::max(1ll, std::min(Rs, budgetRounds / G)));
        const long long numChunks = (Rs + Rc - 1) / Rc;

        // Jump polynomials: A = x^(2496*roundBase), Q_b = x^(2496*Rs*2^b)
        bmThread.join();
        std::vector<Poly> polys(levels + 1);
        #pragma omp parallel sections
        {
            #pragma omp section
            { polys[0] = mp.powX(static_cast<uint64_t>(RAND_PER_ROUND) * roundBase); }
            #pragma omp section
            {
                if (levels > 0) {
                    polys[1] = mp.powX(static_cast<uint64_t>(RAND_PER_ROUND) * Rs);
                    for (int b = 1; b < levels; ++b) polys[b + 1] = mp.squareMod(polys[b]);
                }
            }
        }

        std::vector<uint64_t> polyFlat(static_cast<size_t>(levels + 1) * POLY_WORDS);
        for (int b = 0; b <= levels; ++b)
            std::copy(polys[b].begin(), polys[b].begin() + POLY_WORDS, polyFlat.begin() + static_cast<size_t>(b) * POLY_WORDS);
        uint32_t seed[MT_N];
        mtSeedWindow(42u, seed);

        const int jumpGrid = std::min(G, state.numSMs * 2);
        CUDA_CHECK(cudaMalloc(&dRowStart, (n + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&dCulledOff, (R + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&dCulledCols, culledCols.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dStates, static_cast<size_t>(G) * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dSeed, MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dPolys, polyFlat.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dScratch, static_cast<size_t>(jumpGrid) * MT_EXT_WINDOWS * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dRand, static_cast<size_t>(G) * Rc * RAND_PER_ROUND * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dCounters, numChunks * sizeof(unsigned long long)));

        cudaStream_t s = state.stream;
        CUDA_CHECK(cudaMemsetAsync(dCounters, 0, numChunks * sizeof(unsigned long long), s));
        CUDA_CHECK(cudaMemcpyAsync(dRowStart, rowStart.data(), (n + 1) * sizeof(long long), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dCulledOff, culledOff.data(), (R + 1) * sizeof(long long), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dCulledCols, culledCols.data(), culledCols.size() * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dSeed, seed, MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dPolys, polyFlat.data(), polyFlat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, s));

        // Starting states of all generator blocks via a binary jump tree
        mtJumpKernel<<<1, MT_N, 0, s>>>(dSeed, dStates, 1, dPolys, dScratch);
        for (int b = 0; b < levels; ++b) {
            int first = 1 << b;
            int count = std::min(first, G - first);
            mtJumpKernel<<<std::min(count, jumpGrid), MT_N, 0, s>>>(
                dStates, dStates + static_cast<size_t>(first) * MT_N, count,
                dPolys + static_cast<size_t>(b + 1) * POLY_WORDS, dScratch);
        }
        CUDA_CHECK(cudaGetLastError());

        KijParams P;
        P.roundBase = roundBase;
        P.roundsPerBlock = Rs;
        P.roundEnd = roundEnd;
        P.pairBegin = pairBegin;
        P.pairEnd = pairEnd;
        P.rowStart = dRowStart;
        P.culledOff = dCulledOff;
        P.culledCols = dCulledCols;
        P.r0 = r0;
        P.r1 = r1;
        P.n = n;
        P.tris = state.dTris;
        P.oct = state.oct;
        P.kij = state.dW;

        int traceBlocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&traceBlocksPerSM, kijTraceKernel,
                                                                 TRACE_WARPS * 32, 0));
        const int traceGrid = std::max(1, traceBlocksPerSM) * state.numSMs;
        for (long long c = 0; c < numChunks; ++c) {
            mtGenerateKernel<<<G, MT_N, 0, s>>>(dStates, roundBase, Rs, roundEnd, c, Rc, dRand);
            kijTraceKernel<<<traceGrid, TRACE_WARPS * 32, 0, s>>>(P, dRand, c, Rc, G, dCounters + c);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (bmThread.joinable()) bmThread.join();

    // --- Tau on the host (overlaps with the GPU form factor kernel)
    if (g_rank == 0) printf("Computing time delays (Tau)...\n");
    if (R > 0) {
        uint16_t* hTau = nullptr;
        const size_t total = static_cast<size_t>(R) * n;
        CUDA_CHECK(cudaMallocHost(&hTau, total * sizeof(uint16_t)));
        int maxTau = 0;
        #pragma omp parallel for schedule(static) reduction(max : maxTau)
        for (int li = 0; li < R; ++li) {
            int i = r0 + li;
            uint16_t* row = hTau + static_cast<size_t>(li) * n;
            for (int j = 0; j < n; ++j) {
                int tau = (i == j) ? 0 : computeTau(tris[i], tris[j]);
                maxTau = std::max(maxTau, tau);
                row[j] = static_cast<uint16_t>(tau);
            }
        }
        if (maxTau > 65535) {
            fprintf(stderr, "Tau exceeds 16-bit storage\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaMemcpyAsync(state.dTau, hTau, total * sizeof(uint16_t), cudaMemcpyHostToDevice, state.stream));

        if (g_rank == 0) printf("Computing form factors (Kij)...\n");
        unsigned long long* dNonZero = nullptr;
        CUDA_CHECK(cudaMalloc(&dNonZero, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(dNonZero, 0, sizeof(unsigned long long), state.stream));
        weightKernel<<<state.numSMs * 8, 256, 0, state.stream>>>(state.dW, state.dAreas, total, n, dNonZero);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long nz = 0;
        CUDA_CHECK(cudaMemcpyAsync(&nz, dNonZero, sizeof(nz), cudaMemcpyDeviceToHost, state.stream));
        CUDA_CHECK(cudaStreamSynchronize(state.stream));
        state.nonZeroKij = nz;
        CUDA_CHECK(cudaFree(dNonZero));
        CUDA_CHECK(cudaFreeHost(hTau));
    } else if (g_rank == 0) {
        printf("Computing form factors (Kij)...\n");
    }

    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    cudaFree(dRowStart);
    cudaFree(dCulledOff);
    cudaFree(dCulledCols);
    cudaFree(dStates);
    cudaFree(dSeed);
    cudaFree(dPolys);
    cudaFree(dScratch);
    cudaFree(dRand);
    cudaFree(dCounters);

    unsigned long long totalNZ = 0;
    MPI_Allreduce(&state.nonZeroKij, &totalNZ, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = totalNZ;

    if (g_rank == 0) {
        for (size_t i = 0; i < state.numTriangles; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
            }
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

constexpr int SIM_WARPS = 8;

// One warp per receiver row; lanes stride over emitters, fixed-order reduction
__global__ void __launch_bounds__(SIM_WARPS * 32)
simStepKernel(const val_t* __restrict__ W, const uint16_t* __restrict__ tau,
              val_t* __restrict__ radB, const val_t* __restrict__ rho,
              int t, int n, int r0, int R, int src, int timeOff) {
    int lane = threadIdx.x & 31;
    int li = blockIdx.x * SIM_WARPS + (threadIdx.x >> 5);
    if (li >= R) return;
    int i = r0 + li;
    const val_t* wRow = W + static_cast<size_t>(li) * n;
    const uint16_t* tRow = tau + static_cast<size_t>(li) * n;
    val_t sumB = ZERO;
    for (int j = lane; j < n; j += 32) {
        int tauij = tRow[j];
        if (t < tauij || j == i) continue;
        val_t w = wRow[j];
        if (w <= ZERO) continue;
        val_t radJ = radB[static_cast<size_t>(t - tauij) * n + j];
        if (radJ <= ZERO) continue;
        sumB += w * radJ;
    }
    for (int o = 16; o > 0; o >>= 1) sumB += __shfl_down_sync(0xffffffffu, sumB, o);
    if (lane == 0) {
        val_t radE = (i == src && t < timeOff) ? 1.0f : 0.0f;
        radB[static_cast<size_t>(t) * n + i] = rho[i] * sumB + radE;
    }
}

void runSimulation(SimulationState& state) {
    if (g_rank == 0) printf("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int R = state.localRows();
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = T / 2;
    cudaStream_t s = state.stream;
    const int grid = std::max(1, (R + SIM_WARPS - 1) / SIM_WARPS);

    for (int t = 0; t < T; ++t) {
        if (R > 0) {
            simStepKernel<<<grid, SIM_WARPS * 32, 0, s>>>(state.dW, state.dTau, state.dRadB, state.dRho,
                                                         t, n, state.rowBegin, R,
                                                         static_cast<int>(state.sourceIndex), timeOff);
            CUDA_CHECK(cudaGetLastError());
        }
        val_t* hRow = state.radB.data() + state.idxTN(t, 0);
        val_t* dRow = state.dRadB + state.idxTN(t, 0);
        if (g_size > 1) {
            if (R > 0)
                CUDA_CHECK(cudaMemcpyAsync(hRow + state.rowBegin, dRow + state.rowBegin, R * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hRow, state.rowCounts.data(),
                           state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpyAsync(dRow, hRow, n * sizeof(val_t), cudaMemcpyHostToDevice, s));
        }

        if (g_rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    if (g_size == 1 && T > 0) {
        CUDA_CHECK(cudaMemcpyAsync(state.radB.data(), state.dRadB, static_cast<size_t>(T) * n * sizeof(val_t),
                                   cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distanceKernel(const val_t* __restrict__ radB, val_t* __restrict__ dist,
                               int n, int T, int r0, int R, int src) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= R) return;
    int i = r0 + li;
    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = 0; t < T; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + src];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    dist[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    if (g_rank == 0) printf("Computing distances via cross-correlation...\n");

    const int R = state.localRows();
    std::vector<val_t> local(std::max(R, 1));
    if (R > 0) {
        distanceKernel<<<(R + 127) / 128, 128, 0, state.stream>>>(
            state.dRadB, state.dDist, static_cast<int>(state.numTriangles),
            static_cast<int>(state.numTimesteps), state.rowBegin, R, static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(local.data(), state.dDist, R * sizeof(val_t), cudaMemcpyDeviceToHost, state.stream));
        CUDA_CHECK(cudaStreamSynchronize(state.stream));
    }
    MPI_Gatherv(local.data(), R, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
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

static void selectDevice() {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &local);
    int localRank = 0;
    MPI_Comm_rank(local, &localRank);
    int localSize = 1;
    MPI_Comm_size(local, &localSize);
    MPI_Comm_free(&local);
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devCount));
    // Share the node's cores among the ranks placed on it
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    CUDA_CHECK(cudaFree(nullptr));   // create context
}

static int finish(int code) {
    fflush(stdout);
    MPI_Finalize();
    return code;
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
            return finish(0);
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    selectDevice();

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_rank == 0) {
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
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (g_rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    precompute(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (g_rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (g_rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    int rc = 0;
    if (g_rank == 0) {
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
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return finish(rc);
}

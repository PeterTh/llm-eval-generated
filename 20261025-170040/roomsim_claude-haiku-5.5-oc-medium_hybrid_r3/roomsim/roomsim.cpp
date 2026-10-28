/**
 * Room Response Simulation Benchmark
 * 
 * Room impulse response simulation using radiosity-based wave propagation. It models
 * how sound/light waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization (hybrid, unconditional):
 *  - MPI:    triangles (receiver/emitter rows) are block-distributed across ranks, one
 *            GPU per rank. Each timestep's radiosity row is exchanged with MPI_Allgatherv.
 *  - CUDA:   form factors (ray casting + octree traversal), the wave-propagation update
 *            and the cross-correlation run on each rank's GPU.
 *  - OpenMP: host-side work runs on CPU threads: time-delay tiles, the per-row pair
 *            counts, and generation of the random samples for the form factor pipeline.
 *
 * Semantics are preserved exactly. The original single sequential mt19937(42) stream is
 * reproduced: each pair's samples are placed at their exact stream position using exact
 * mt19937 jump-ahead (MtJump), so that generation parallelizes. Culled pairs consume no
 * random numbers, every other pair consumes exactly 4*NUM_RAYS draws. All floating-point
 * accumulations keep the original order.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cstdarg>
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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Error handling and process identity
// ============================================================================

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int e_ = (call);                                                       \
        if (e_ != MPI_SUCCESS) {                                               \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(e_));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

static int g_rank = 0;
static int g_size = 1;

// Console output is produced by rank 0 only
void rootPrintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void rootPrintf(const char* fmt, ...) {
    if (g_rank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

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

// Random numbers consumed per form factor pair (4 per ray: two barycentric samples per point)
constexpr size_t DRAWS_PER_PAIR = NUM_RAYS * 4;
// Form factor pipeline: pairs are grouped into RNG blocks of FF_BLOCK consecutive pairs (in the
// original stream order). FF_BLOCKS_PER_BATCH blocks are generated concurrently and handed to
// the GPU as one batch.
constexpr size_t FF_BLOCK = size_t(1) << 13;
constexpr size_t FF_BLOCKS_PER_BATCH = 64;
constexpr size_t FF_BATCH = FF_BLOCK * FF_BLOCKS_PER_BATCH;
constexpr uint32_t NO_ITEM = 0xFFFFFFFFu;  // padding entry in a batch (skipped by the kernel)
constexpr int BLOCK = 128;
constexpr int DEV_STACK = 128;            // Traversal stack depth on device

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
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
// Octree for Spatial Acceleration (built on host, flattened for the GPU)
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

// Node layout used on the device. Leaves have triCount > 0 and reference
// triStart..triStart+triCount in the flattened index array.
struct DNode {
    float cx, cy, cz;
    float hx, hy, hz;
    int child[8];
    uint32_t triStart;
    uint32_t triCount;
};

struct FlatOctree {
    std::vector<DNode> nodes;      // nodes[0] is the root
    std::vector<uint32_t> tris;    // flattened leaf triangle indices
    int maxDepth = 0;
};

int flattenOctreeNode(const Octree& node, FlatOctree& flat, int depth) {
    const int id = static_cast<int>(flat.nodes.size());
    flat.nodes.push_back(DNode{});
    flat.maxDepth = std::max(flat.maxDepth, depth);

    DNode d{};
    d.cx = node.center.x;
    d.cy = node.center.y;
    d.cz = node.center.z;
    d.hx = node.halfExtent.x;
    d.hy = node.halfExtent.y;
    d.hz = node.halfExtent.z;
    for (int c = 0; c < 8; ++c) d.child[c] = -1;
    d.triStart = 0;
    d.triCount = 0;

    if (!node.triangleIndices.empty()) {
        d.triStart = static_cast<uint32_t>(flat.tris.size());
        d.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) flat.tris.push_back(static_cast<uint32_t>(idx));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                d.child[c] = flattenOctreeNode(*node.children[c], flat, depth + 1);
            }
        }
    }
    flat.nodes[id] = d;
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

// Mersenne Twister, bit-identical to std::mt19937, with access to the raw
// (untempered) state words needed for exact jump-ahead.
class MersenneTwister {
public:
    using result_type = uint32_t;
    static constexpr int N = 624;
    static constexpr int MM = 397;

    explicit MersenneTwister(uint32_t seed) {
        mt_[0] = seed;
        for (int i = 1; i < N; ++i) {
            mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        mti_ = N;
    }

    // Engine whose next raw word is w[0]; w holds N consecutive raw words of the stream
    static MersenneTwister fromWindow(const uint32_t* w) {
        MersenneTwister g(0);
        std::memcpy(g.mt_, w, sizeof(g.mt_));
        g.mti_ = 0;
        return g;
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<uint32_t>::max(); }

    uint32_t nextRaw() {
        if (mti_ >= N) twist();
        return mt_[mti_++];
    }

    result_type operator()() {
        uint32_t y = nextRaw();
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

private:
    static uint32_t mix(uint32_t a, uint32_t b) {
        constexpr uint32_t UPPER = 0x80000000u;
        constexpr uint32_t LOWER = 0x7fffffffu;
        constexpr uint32_t MATRIX_A = 0x9908b0dfu;
        uint32_t y = (a & UPPER) | (b & LOWER);
        return (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }

    void twist() {
        int k = 0;
        for (; k < N - MM; ++k) mt_[k] = mt_[k + MM] ^ mix(mt_[k], mt_[k + 1]);
        for (; k < N - 1; ++k) mt_[k] = mt_[k + MM - N] ^ mix(mt_[k], mt_[k + 1]);
        mt_[N - 1] = mt_[MM - 1] ^ mix(mt_[N - 1], mt_[0]);
        mti_ = 0;
    }

    uint32_t mt_[N];
    int mti_;
};

// Exact jump-ahead for the raw mt19937(42) word sequence s_0, s_1, ...
//
// The raw sequence obeys a linear recurrence over GF(2) with characteristic polynomial P of
// degree L (found with Berlekamp-Massey on the sequence itself). Therefore for any k
//     s_{m+k} = sum_j a_j s_{m+j},   where x^k mod P = sum_j a_j x^j,
// so the 624-word engine window at any stream position can be computed without generating
// the intervening outputs.
class MtJump {
public:
    using Poly = std::vector<uint64_t>;

    explicit MtJump(size_t seqLen = 44000) : seq_(seqLen) {
        MersenneTwister g(42);
        for (auto& w : seq_) w = g.nextRaw();

        // Bit lane 0 of the sequence; its minimal polynomial has the full degree of the recurrence
        std::vector<uint8_t> bits(seq_.size());
        for (size_t k = 0; k < seq_.size(); ++k) bits[k] = static_cast<uint8_t>(seq_[k] & 1u);

        std::vector<uint64_t> C;
        int L = 0;
        berlekampMassey(bits, C, L);
        if (2 * static_cast<size_t>(L) + 624 > seq_.size()) {
            fprintf(stderr, "MtJump: sequence too short for recurrence of degree %d\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        L_ = L;
        W_ = static_cast<size_t>(L) / 64 + 2;

        // Characteristic polynomial P(x) = sum_t c_{L-t} x^t (bits 0..L)
        P_.assign(W_, 0);
        for (int t = 0; t <= L; ++t) {
            size_t src = static_cast<size_t>(L - t);
            if ((C[src / 64] >> (src % 64)) & 1u) P_[t / 64] |= (uint64_t(1) << (t % 64));
        }
    }

    // Window of 624 raw words starting at absolute stream position q
    void windowAt(uint64_t q, uint32_t* out) const {
        std::vector<uint32_t> idx = setBitIndices(powX(q));
        for (int jp = 0; jp < MersenneTwister::N; ++jp) {
            uint32_t acc = 0;
            for (uint32_t i : idx) acc ^= seq_[static_cast<size_t>(jp) + i];
            out[jp] = acc;
        }
    }

    // Indices i with (x^s mod P)_i = 1; reusable with advance() for jumps of length s
    std::vector<uint32_t> jumpIndices(uint64_t s) const { return setBitIndices(powX(s)); }

    // Given the window at position p, write the window at position p + s (idx = jumpIndices(s))
    void advance(const uint32_t* cur, const std::vector<uint32_t>& idx, uint32_t* out) const {
        std::vector<uint32_t> S(static_cast<size_t>(L_) + MersenneTwister::N);
        MersenneTwister e = MersenneTwister::fromWindow(cur);
        for (auto& w : S) w = e.nextRaw();
        for (int jp = 0; jp < MersenneTwister::N; ++jp) {
            uint32_t acc = 0;
            for (uint32_t i : idx) acc ^= S[static_cast<size_t>(jp) + i];
            out[jp] = acc;
        }
    }

private:
    static void berlekampMassey(const std::vector<uint8_t>& s, std::vector<uint64_t>& C, int& Lout) {
        const size_t n = s.size();
        const size_t W = n / 64 + 2;
        std::vector<uint64_t> B(W, 0), T(W), R(W, 0);
        C.assign(W, 0);
        C[0] = 1;
        B[0] = 1;
        int L = 0;
        size_t m = 1;
        for (size_t k = 0; k < n; ++k) {
            // R bit i = s_{k-i}
            for (size_t w = W - 1; w > 0; --w) R[w] = (R[w] << 1) | (R[w - 1] >> 63);
            R[0] = (R[0] << 1) | s[k];

            uint64_t acc = 0;
            for (size_t w = 0; w < W; ++w) acc ^= C[w] & R[w];
            if (__builtin_parityll(acc) == 0) {
                ++m;
                continue;
            }

            T = C;
            const size_t ws = m / 64;
            const unsigned bs = static_cast<unsigned>(m % 64);
            for (size_t w = 0; w + ws < W; ++w) {
                C[w + ws] ^= B[w] << bs;
                if (bs && w + ws + 1 < W) C[w + ws + 1] ^= B[w] >> (64 - bs);
            }
            if (2 * static_cast<size_t>(L) <= k) {
                L = static_cast<int>(k) + 1 - L;
                B = T;
                m = 1;
            } else {
                ++m;
            }
        }
        Lout = L;
        C.resize(W);
    }

    // dst ^= src << sh
    static void xorShifted(Poly& dst, const Poly& src, size_t sh) {
        const size_t ws = sh / 64;
        const unsigned bs = static_cast<unsigned>(sh % 64);
        for (size_t k = 0; k < src.size(); ++k) {
            if (!src[k]) continue;
            size_t d = k + ws;
            if (d < dst.size()) dst[d] ^= src[k] << bs;
            if (bs && d + 1 < dst.size()) dst[d + 1] ^= src[k] >> (64 - bs);
        }
    }

    // a * x mod P (a has degree < L)
    Poly mulX(Poly a) const {
        for (size_t w = W_ - 1; w > 0; --w) a[w] = (a[w] << 1) | (a[w - 1] >> 63);
        a[0] <<= 1;
        if ((a[L_ / 64] >> (L_ % 64)) & 1u) xorShifted(a, P_, 0);
        return a;
    }

    // a * a mod P (a has degree < L)
    Poly sqrMod(const Poly& a) const {
        Poly t(2 * W_, 0);
        for (int i = 0; i < L_; ++i) {
            if ((a[i / 64] >> (i % 64)) & 1u) {
                size_t d = 2 * static_cast<size_t>(i);
                t[d / 64] |= uint64_t(1) << (d % 64);
            }
        }
        for (int d = 2 * L_ - 2; d >= L_; --d) {
            if ((t[d / 64] >> (d % 64)) & 1u) xorShifted(t, P_, static_cast<size_t>(d - L_));
        }
        t.resize(W_);
        return t;
    }

    // x^q mod P
    Poly powX(uint64_t q) const {
        Poly r(W_, 0);
        r[0] = 1;
        if (q == 0) return r;
        int hb = 63 - __builtin_clzll(q);
        for (int b = hb; b >= 0; --b) {
            r = sqrMod(r);
            if ((q >> b) & 1u) r = mulX(r);
        }
        return r;
    }

    std::vector<uint32_t> setBitIndices(const Poly& a) const {
        std::vector<uint32_t> idx;
        for (int i = 0; i < L_; ++i) {
            if ((a[i / 64] >> (i % 64)) & 1u) idx.push_back(static_cast<uint32_t>(i));
        }
        return idx;
    }

    std::vector<uint32_t> seq_;   // raw words s_0 .. s_{len-1}
    int L_ = 0;                   // recurrence degree
    size_t W_ = 0;                // words per polynomial
    Poly P_;                      // characteristic polynomial, bits 0..L
};

// ============================================================================
// Host-side geometry helpers
// ============================================================================

// Culling of triangles facing the same direction (no random draws are consumed)
inline bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Device Types and Kernels
// ============================================================================

struct DVec3 {
    float x, y, z;
};

struct DTri {
    DVec3 a, b, c, n;
};

__device__ __forceinline__ DVec3 operator+(DVec3 a, DVec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ DVec3 operator-(DVec3 a, DVec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ DVec3 operator-(DVec3 a) { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ DVec3 operator*(DVec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ DVec3 operator/(DVec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }

__device__ __forceinline__ float dotD(DVec3 a, DVec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ DVec3 crossD(DVec3 a, DVec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ float sqrNormD(DVec3 a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
__device__ __forceinline__ float normD(DVec3 a) { return sqrtf(sqrNormD(a)); }

// Mirrors randomPointInTriangle with pre-drawn barycentric samples (u, v)
__device__ __forceinline__ DVec3 pointInTriangle(const DTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    DVec3 ab = t.b - t.a;
    DVec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Mirrors Octree::rayIntersectsBox
__device__ __forceinline__ bool boxHitD(const DNode& n, DVec3 p1, DVec3 p2) {
    DVec3 center = {n.cx, n.cy, n.cz};
    DVec3 halfExtent = {n.hx, n.hy, n.hz};
    DVec3 d = (p2 - p1) * 0.5f;
    DVec3 c = p1 + d - center;
    DVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Mirrors rayTriangleIntersect (Moller-Trumbore)
__device__ __forceinline__ float rayTriangleIntersectD(DVec3 orig, DVec3 dir, DVec3 v0, DVec3 v1, DVec3 v2) {
    DVec3 e1 = v1 - v0;
    DVec3 e2 = v2 - v0;
    DVec3 pvec = crossD(dir, e2);
    float det = dotD(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    DVec3 tvec = orig - v0;
    float u = dotD(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    DVec3 qvec = crossD(tvec, e1);
    float v = dotD(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return dotD(e2, qvec) * invDet;
}

// Mirrors isRayBlocked + Octree::applyToTris (iterative; result is a pure OR over candidates)
__device__ bool isRayBlockedD(DVec3 from, DVec3 to, uint32_t srcIdx, uint32_t dstIdx,
                              const DNode* __restrict__ nodes,
                              const uint32_t* __restrict__ triIdx,
                              const DTri* __restrict__ tris) {
    DVec3 dir = to - from;
    float rayLen = normD(dir);
    if (rayLen < EPSILON) return true;
    DVec3 dirNorm = dir / rayLen;

    int stack[DEV_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root is not box-tested (matches applyToTris on the root)

    while (sp > 0) {
        const DNode& n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (uint32_t q = 0; q < n.triCount; ++q) {
                uint32_t idx = triIdx[n.triStart + q];
                if (idx == srcIdx || idx == dstIdx) continue;
                const DTri& tri = tris[idx];
                float dist = rayTriangleIntersectD(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = n.child[c];
                if (ch >= 0 && boxHitD(nodes[ch], from, to)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

// Mirrors cosPhi
__device__ __forceinline__ float cosPhiD(DVec3 v, DVec3 normal) {
    float vNorm = normD(v);
    if (vNorm <= EPSILON) return ZERO;
    float c = dotD(v, normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // std::max(ZERO, ...)
}

// One thread per (i, j) pair. The random samples for this pair were drawn on the
// host from the shared stream in the same order as the original computeKij.
__global__ void formFactorKernel(const DTri* __restrict__ tris,
                                 const DNode* __restrict__ nodes,
                                 const uint32_t* __restrict__ triIdx,
                                 const uint2* __restrict__ items,
                                 const float* __restrict__ rnd,
                                 int count, int rowBegin, int nLocal,
                                 float* __restrict__ kijT) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    if (items[k].x == NO_ITEM) return;  // padding

    const uint32_t i = items[k].x;
    const uint32_t j = items[k].y;
    const DTri& triI = tris[i];
    const DTri& triJ = tris[j];
    const float* r = rnd + static_cast<size_t>(k) * DRAWS_PER_PAIR;

    float kij = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        DVec3 pI = pointInTriangle(triI, r[4 * ray + 0], r[4 * ray + 1]);
        DVec3 pJ = pointInTriangle(triJ, r[4 * ray + 2], r[4 * ray + 3]);

        if (isRayBlockedD(pI, pJ, i, j, nodes, triIdx, tris)) continue;

        DVec3 v = pJ - pI;
        float distSqr = sqrNormD(v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = cosPhiD(v, triI.n);
        float cosPhiJ = cosPhiD(-v, triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    // Column-major-by-emitter layout kijT[j * nLocal + (i - rowBegin)] for coalesced access
    kijT[static_cast<size_t>(j) * nLocal + (i - rowBegin)] = kij * INV_NUM_RAYS;
}

// One thread per locally owned receiver i. Mirrors the timestep update of runSimulation.
// radB[t] for all emitters must already hold earlier timesteps; tau >= 1 for i != j.
__global__ void simulationKernel(int t, int N, int nLocal, int rowBegin, int srcIdx, int timeOff,
                                 const float* __restrict__ kijT,
                                 const int* __restrict__ tauT,
                                 const float* __restrict__ areas,
                                 float rho,
                                 float* __restrict__ radB) {
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= nLocal) return;
    const int i = rowBegin + il;

    float sumB = ZERO;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        int tauij = tauT[static_cast<size_t>(j) * nLocal + il];
        if (t < tauij) continue;  // wave has not yet propagated from j to i

        float kij = kijT[static_cast<size_t>(j) * nLocal + il];
        if (kij <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t - tauij);
        float radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        float a = kij * areas[j];
        sumB += (ONE < a ? ONE : a) * radJ;  // std::min(a, ONE) * radJ
    }

    float radE = (t < timeOff && i == srcIdx) ? ONE : ZERO;
    radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE;
}

// One thread per locally owned receiver i. Mirrors computeDistances.
__global__ void distanceKernel(int T, int N, int nLocal, int rowBegin, int srcIdx,
                               const float* __restrict__ radB,
                               float* __restrict__ distOut) {
    int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= nLocal) return;
    const int i = rowBegin + il;

    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * N + i];
            float pS = radB[static_cast<size_t>(tt - t) * N + srcIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distOut[il] = WAVE_SPEED * static_cast<float>(bestT);
}

__global__ void countNonZeroKernel(const float* __restrict__ kijT, size_t total,
                                   unsigned long long* __restrict__ count) {
    size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long c = 0;
    for (; k < total; k += stride) {
        if (kijT[k] > EPSILON) ++c;
    }
    if (c) atomicAdd(count, c);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    val_t reflectivity = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity history (T x N), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source (N), replicated on all ranks

    Octree octree;                  // Spatial acceleration structure (host build)

    // Row block owned by this rank: receivers/emitters i in [rowBegin, rowEnd)
    size_t rowBegin = 0;
    size_t rowEnd = 0;
    std::vector<int> rowCounts;     // Rows per rank
    std::vector<int> rowDispls;     // Row offsets per rank

    size_t nLocal() const { return rowEnd - rowBegin; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Device-resident data for this rank
struct DeviceContext {
    DTri* tris = nullptr;
    DNode* nodes = nullptr;
    uint32_t* triIdx = nullptr;
    float* areas = nullptr;
    float* kijT = nullptr;         // [j * nLocal + (i - rowBegin)]
    int* tauT = nullptr;           // [j * nLocal + (i - rowBegin)]
    float* radB = nullptr;         // T x N
    float* distOut = nullptr;      // nLocal

    // Double-buffered form-factor pipeline
    uint2* items[2] = {nullptr, nullptr};
    float* rnd[2] = {nullptr, nullptr};
    uint2* hItems[2] = {nullptr, nullptr};   // pinned host
    float* hRnd[2] = {nullptr, nullptr};     // pinned host
    cudaEvent_t done[2] = {nullptr, nullptr};
};

template <typename T>
T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, FlatOctree& flat, int subdivisions,
                          size_t timesteps, size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (identical on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    rootPrintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration and flatten it for the device
    rootPrintf("Building octree...\n");
    state.octree.build(state.triangles);
    flat = FlatOctree{};
    flattenOctreeNode(state.octree, flat, 0);
    if (8 * (flat.maxDepth + 2) > DEV_STACK) {
        fprintf(stderr, "Octree too deep for device traversal stack\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Radiosity history and distances (replicated; filled by allgathers)
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Row-block decomposition across MPI ranks
    const size_t N = state.numTriangles;
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        size_t b = static_cast<size_t>(r) * N / g_size;
        size_t e = static_cast<size_t>(r + 1) * N / g_size;
        state.rowCounts[r] = static_cast<int>(e - b);
        state.rowDispls[r] = static_cast<int>(b);
    }
    state.rowBegin = static_cast<size_t>(g_rank) * N / g_size;
    state.rowEnd = static_cast<size_t>(g_rank + 1) * N / g_size;
}

void setupDevice(const SimulationState& st, DeviceContext& ctx, const FlatOctree& flat) {
    const size_t N = st.numTriangles;
    const size_t nL = st.nLocal();

    // Geometry (identical on every device)
    std::vector<DTri> hTris(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = st.triangles[i];
        hTris[i].a = {t.a.x, t.a.y, t.a.z};
        hTris[i].b = {t.b.x, t.b.y, t.b.z};
        hTris[i].c = {t.c.x, t.c.y, t.c.z};
        Vec3 n = t.normal();
        hTris[i].n = {n.x, n.y, n.z};
    }
    ctx.tris = deviceAlloc<DTri>(N);
    CUDA_CHECK(cudaMemcpy(ctx.tris, hTris.data(), N * sizeof(DTri), cudaMemcpyHostToDevice));

    ctx.nodes = deviceAlloc<DNode>(flat.nodes.size());
    CUDA_CHECK(cudaMemcpy(ctx.nodes, flat.nodes.data(), flat.nodes.size() * sizeof(DNode),
                          cudaMemcpyHostToDevice));

    ctx.triIdx = deviceAlloc<uint32_t>(flat.tris.size());
    CUDA_CHECK(cudaMemcpy(ctx.triIdx, flat.tris.data(), flat.tris.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    ctx.areas = deviceAlloc<float>(N);
    CUDA_CHECK(cudaMemcpy(ctx.areas, st.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Local form-factor and time-delay blocks (N x nLocal)
    ctx.kijT = deviceAlloc<float>(N * nL);
    CUDA_CHECK(cudaMemset(ctx.kijT, 0, std::max<size_t>(N * nL, 1) * sizeof(float)));
    ctx.tauT = deviceAlloc<int>(N * nL);

    ctx.radB = deviceAlloc<float>(st.numTimesteps * N);
    CUDA_CHECK(cudaMemset(ctx.radB, 0, std::max<size_t>(st.numTimesteps * N, 1) * sizeof(float)));
    ctx.distOut = deviceAlloc<float>(nL);

    for (int s = 0; s < 2; ++s) {
        ctx.items[s] = deviceAlloc<uint2>(FF_BATCH);
        ctx.rnd[s] = deviceAlloc<float>(FF_BATCH * DRAWS_PER_PAIR);
        CUDA_CHECK(cudaMallocHost(&ctx.hItems[s], FF_BATCH * sizeof(uint2)));
        CUDA_CHECK(cudaMallocHost(&ctx.hRnd[s], FF_BATCH * DRAWS_PER_PAIR * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.done[s], cudaEventDisableTiming));
    }
}

void freeDevice(DeviceContext& ctx) {
    cudaFree(ctx.tris);
    cudaFree(ctx.nodes);
    cudaFree(ctx.triIdx);
    cudaFree(ctx.areas);
    cudaFree(ctx.kijT);
    cudaFree(ctx.tauT);
    cudaFree(ctx.radB);
    cudaFree(ctx.distOut);
    for (int s = 0; s < 2; ++s) {
        cudaFree(ctx.items[s]);
        cudaFree(ctx.rnd[s]);
        cudaFreeHost(ctx.hItems[s]);
        cudaFreeHost(ctx.hRnd[s]);
        cudaEventDestroy(ctx.done[s]);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Number of form-factor pairs (non-diagonal, not culled) in each row, as prefix sums:
// pre[r] = number of such pairs in rows [0, r). These are exactly the mt19937 draw groups
// that the original sequential code consumes before row r.
std::vector<uint64_t> rowPairPrefix(const SimulationState& st, size_t rowLimit) {
    const long long N = static_cast<long long>(st.numTriangles);
    std::vector<uint64_t> cnt(rowLimit, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long i = 0; i < static_cast<long long>(rowLimit); ++i) {
        uint64_t c = 0;
        for (long long j = 0; j < N; ++j) {
            if (j == i) continue;
            if (!isCulled(st.triangles[i], st.triangles[j])) ++c;
        }
        cnt[i] = c;
    }
    std::vector<uint64_t> pre(rowLimit + 1, 0);
    for (size_t r = 0; r < rowLimit; ++r) pre[r + 1] = pre[r] + cnt[r];
    return pre;
}

void computeTimeDelays(const SimulationState& st, DeviceContext& ctx) {
    rootPrintf("Computing time delays (Tau)...\n");
    const size_t N = st.numTriangles;
    const size_t nL = st.nLocal();
    if (nL == 0) return;

    constexpr size_t TILE = 256;
    std::vector<int> tile(TILE * N);
    for (size_t il0 = 0; il0 < nL; il0 += TILE) {
        const size_t rows = std::min(TILE, nL - il0);
        #pragma omp parallel for schedule(static)
        for (long long j = 0; j < static_cast<long long>(N); ++j) {
            for (size_t q = 0; q < rows; ++q) {
                size_t i = st.rowBegin + il0 + q;
                tile[static_cast<size_t>(j) * TILE + q] =
                    (i == static_cast<size_t>(j)) ? 0 : computeTau(st.triangles[i], st.triangles[j]);
            }
        }
        // tile is [j][q]; scatter into device layout [j][il0 + q] with pitch nL
        CUDA_CHECK(cudaMemcpy2D(ctx.tauT + il0, nL * sizeof(int), tile.data(), TILE * sizeof(int),
                                rows * sizeof(int), N, cudaMemcpyHostToDevice));
    }
}

// Form factors for this rank's rows.
//
// The original code draws all random samples from one mt19937(42) stream, consuming
// DRAWS_PER_PAIR draws for every non-culled pair in row-major order. Here the pairs of this
// rank are split into blocks of FF_BLOCK consecutive pairs; block b starts at the exact stream
// position 64 * (global pair index), which MtJump computes directly. Blocks are generated in
// parallel on the host (OpenMP) and consumed by the GPU in batches, double-buffered so that
// host generation overlaps device work.
void computeFormFactors(const SimulationState& st, DeviceContext& ctx) {
    rootPrintf("Computing form factors (Kij)...\n");
    const size_t N = st.numTriangles;
    const size_t r0 = st.rowBegin;
    const size_t r1 = st.rowEnd;
    const int nL = static_cast<int>(st.nLocal());
    if (nL == 0) return;

    const MtJump jump;
    const std::vector<uint64_t> pre = rowPairPrefix(st, r1);
    const uint64_t G0 = pre[r0];                 // first global pair index of this rank
    const uint64_t numPairs = pre[r1] - pre[r0];
    const uint64_t numBlocks = (numPairs + FF_BLOCK - 1) / FF_BLOCK;
    const uint64_t numBatches = (numBlocks + FF_BLOCKS_PER_BATCH - 1) / FF_BLOCKS_PER_BATCH;
    const size_t B = FF_BLOCKS_PER_BATCH;

    // Jump by one batch worth of blocks (B * FF_BLOCK pairs)
    const std::vector<uint32_t> batchJump = jump.jumpIndices(DRAWS_PER_PAIR * FF_BATCH);

    std::vector<uint32_t> windows(B * MersenneTwister::N);
    std::vector<char> windowReady(B, 0);

    int slot = 0;
    for (uint64_t k = 0; k < numBatches; ++k) {
        // Previous user of this host/device buffer (two batches ago) must have finished
        CUDA_CHECK(cudaEventSynchronize(ctx.done[slot]));

        uint2* hItems = ctx.hItems[slot];
        float* hRnd = ctx.hRnd[slot];

        #pragma omp parallel for schedule(static, 1)
        for (int jb = 0; jb < static_cast<int>(B); ++jb) {
            const uint64_t b = k * B + static_cast<uint64_t>(jb);
            uint2* items = hItems + static_cast<size_t>(jb) * FF_BLOCK;
            float* rnd = hRnd + static_cast<size_t>(jb) * FF_BLOCK * DRAWS_PER_PAIR;

            if (b >= numBlocks) {
                for (size_t q = 0; q < FF_BLOCK; ++q) items[q] = make_uint2(NO_ITEM, 0);
                continue;
            }

            uint32_t* win = windows.data() + static_cast<size_t>(jb) * MersenneTwister::N;
            const uint64_t g0 = G0 + b * FF_BLOCK;        // global pair index of block start
            const size_t cnt = static_cast<size_t>(std::min<uint64_t>(FF_BLOCK, pre[r1] - g0));
            if (!windowReady[jb]) {
                jump.windowAt(DRAWS_PER_PAIR * g0, win);
                windowReady[jb] = 1;
            }

            MersenneTwister eng = MersenneTwister::fromWindow(win);
            std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);

            // Locate the first pair of the block: row with pre[row] <= g0 < pre[row + 1]
            size_t row = static_cast<size_t>(
                std::upper_bound(pre.begin() + r0, pre.begin() + r1 + 1, g0) - pre.begin()) - 1;
            size_t i = row;
            size_t j = 0;
            uint64_t skip = g0 - pre[row];
            while (true) {
                if (j != i && !isCulled(st.triangles[i], st.triangles[j])) {
                    if (skip == 0) break;
                    --skip;
                }
                ++j;
            }

            for (size_t q = 0; q < cnt; ++q) {
                items[q] = make_uint2(static_cast<unsigned>(i), static_cast<unsigned>(j));
                float* r = rnd + q * DRAWS_PER_PAIR;
                for (size_t d = 0; d < DRAWS_PER_PAIR; ++d) r[d] = dist(eng);
                if (q + 1 < cnt) {
                    do {
                        if (++j == N) {
                            ++i;
                            j = 0;
                        }
                    } while (j == i || isCulled(st.triangles[i], st.triangles[j]));
                }
            }
            for (size_t q = cnt; q < FF_BLOCK; ++q) items[q] = make_uint2(NO_ITEM, 0);

            // Advance this slot's window to the next block it will serve (jb + B blocks later)
            if (b + B < numBlocks) {
                uint32_t next[MersenneTwister::N];
                jump.advance(win, batchJump, next);
                std::memcpy(win, next, sizeof(next));
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(ctx.items[slot], hItems, FF_BATCH * sizeof(uint2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(ctx.rnd[slot], hRnd, FF_BATCH * DRAWS_PER_PAIR * sizeof(float),
                                   cudaMemcpyHostToDevice));
        formFactorKernel<<<static_cast<unsigned int>(FF_BATCH / BLOCK), BLOCK>>>(
            ctx.tris, ctx.nodes, ctx.triIdx, ctx.items[slot], ctx.rnd[slot],
            static_cast<int>(FF_BATCH), static_cast<int>(r0), nL, ctx.kijT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ctx.done[slot], 0));
        slot ^= 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    rootPrintf("  Progress: %zu/%zu triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& st, DeviceContext& ctx) {
    rootPrintf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(st.numTriangles);
    const int T = static_cast<int>(st.numTimesteps);
    const int nL = static_cast<int>(st.nLocal());
    const int r0 = static_cast<int>(st.rowBegin);
    const int src = static_cast<int>(st.sourceIndex);
    const int timeOff = T / 2;
    const unsigned int grid = static_cast<unsigned int>((nL + BLOCK - 1) / BLOCK);

    for (int t = 0; t < T; ++t) {
        if (nL > 0) {
            simulationKernel<<<grid, BLOCK>>>(t, N, nL, r0, src, timeOff, ctx.kijT, ctx.tauT,
                                              ctx.areas, st.reflectivity, ctx.radB);
            CUDA_CHECK(cudaGetLastError());
        }

        // Exchange this timestep's local radiosity block with all ranks
        val_t* rowH = st.radB.data() + st.idxTN(t, 0);
        if (nL > 0) {
            CUDA_CHECK(cudaMemcpy(rowH + r0, ctx.radB + st.idxTN(t, 0) + r0,
                                  static_cast<size_t>(nL) * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rowH, st.rowCounts.data(),
                                 st.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(ctx.radB + st.idxTN(t, 0), rowH,
                              st.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            rootPrintf("  Timestep %zu/%zu\n", static_cast<size_t>(t + 1), st.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& st, DeviceContext& ctx) {
    rootPrintf("Computing distances via cross-correlation...\n");
    const int nL = static_cast<int>(st.nLocal());
    if (nL > 0) {
        unsigned int grid = static_cast<unsigned int>((nL + BLOCK - 1) / BLOCK);
        distanceKernel<<<grid, BLOCK>>>(static_cast<int>(st.numTimesteps),
                                        static_cast<int>(st.numTriangles), nL,
                                        static_cast<int>(st.rowBegin),
                                        static_cast<int>(st.sourceIndex), ctx.radB, ctx.distOut);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<val_t> local(static_cast<size_t>(nL));
    if (nL > 0) {
        CUDA_CHECK(cudaMemcpy(local.data(), ctx.distOut, local.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    MPI_CHECK(MPI_Allgatherv(local.data(), nL, MPI_FLOAT, st.distances.data(), st.rowCounts.data(),
                             st.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD));
}

// Number of form factors above EPSILON, summed over all ranks (validation only)
unsigned long long countNonZeroFormFactors(const SimulationState& st, const DeviceContext& ctx) {
    unsigned long long* dCount = deviceAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
    const size_t total = st.numTriangles * st.nLocal();
    if (total > 0) {
        unsigned int grid = static_cast<unsigned int>(std::min<size_t>((total + BLOCK - 1) / BLOCK, 65535));
        countNonZeroKernel<<<grid, BLOCK>>>(ctx.kijT, total, dCount);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long hCount = 0;
    CUDA_CHECK(cudaMemcpy(&hCount, dCount, sizeof(hCount), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCount));

    unsigned long long global = 0;
    MPI_CHECK(MPI_Allreduce(&hCount, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    return global;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
    rootPrintf("\nValidation:\n");

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
            rootPrintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rootPrintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rootPrintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rootPrintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rootPrintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rootPrintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    rootPrintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rootPrintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const unsigned long long totalKij =
        static_cast<unsigned long long>(state.numTriangles) * state.numTriangles;
    rootPrintf("  Non-zero form factors: %llu/%llu (%.2f%%)\n", nonZeroKij, totalKij,
               100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalKij));

    if (nonZeroKij == 0) {
        rootPrintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rootPrintf("  Validation: PASSED\n");
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
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -n <num>     Target number of triangles (default: 320)\n");
    rootPrintf("               Actual count will be rounded to nearest icosphere level:\n");
    rootPrintf("               20, 80, 320, 1280, 5120, 20480\n");
    rootPrintf("  -t <num>     Number of timesteps (default: 50)\n");
    rootPrintf("  -s <num>     Source triangle index (default: 0)\n");
    rootPrintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -o           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
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
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rootPrintf("Room Response Simulation Benchmark\n");
    rootPrintf("===================================\n");
    rootPrintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rootPrintf("Timesteps: %d\n", timesteps);
    rootPrintf("Source triangle: %d\n", sourceIdx);
    rootPrintf("Reflectivity: %.2f\n", reflectivity);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rootPrintf("MPI ranks: %d, OpenMP threads/rank: %d\n", g_size, omp_get_max_threads());
    rootPrintf("\n");

    // Initialize
    SimulationState state;
    FlatOctree flat;
    initializeSimulation(state, flat, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    DeviceContext ctx;
    setupDevice(state, ctx, flat);

    rootPrintf("\n");

    // Precomputation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, ctx);
    computeFormFactors(state, ctx);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rootPrintf("Precomputation time: %ld ms\n", static_cast<long>(preDuration));
    rootPrintf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, ctx);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rootPrintf("Simulation time: %ld ms\n", static_cast<long>(simDuration));
    rootPrintf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, ctx);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rootPrintf("Distance computation time: %ld ms\n", static_cast<long>(distDuration));
    rootPrintf("\n");

    // Total time
    long localTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_CHECK(MPI_Reduce(&localTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
    rootPrintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    rootPrintf("\nPerformance:\n");
    rootPrintf("  Triangles: %zu\n", n);
    rootPrintf("  Timesteps: %zu\n", t);
    rootPrintf("  Form factor computations: %.2e\n", kijOps);
    rootPrintf("  Simulation operations: %.2e\n", simOps);
    rootPrintf("  Distance computations: %.2e\n", distOps);
    rootPrintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    rootPrintf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    rootPrintf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
    rootPrintf("\n");

    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        unsigned long long nonZeroKij = countNonZeroFormFactors(state, ctx);
        if (!validateResults(state, nonZeroKij)) {
            status = 1;
        }
    }

    freeDevice(ctx);
    return status;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &g_size));

    // Local rank within the node selects the GPU and shares the CPU cores
    MPI_Comm shm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &shm));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(shm, &localRank));
    MPI_CHECK(MPI_Comm_size(shm, &localSize));
    MPI_CHECK(MPI_Comm_free(&shm));

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "rank %d: no CUDA devices available\n", g_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    int rc = runBenchmark(argc, argv);

    MPI_CHECK(MPI_Finalize());
    return rc;
}

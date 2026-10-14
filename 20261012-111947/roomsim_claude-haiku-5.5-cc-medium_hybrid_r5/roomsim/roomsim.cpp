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
 * Parallelization (hybrid, all mandatory):
 *  - MPI:    receiver rows are block-distributed across ranks. Each rank owns the
 *            Kij/Tau rows of its block and the full radiosity history is replicated
 *            by an all-gather after every timestep.
 *  - CUDA:   form factors (the dominant cost: N^2 pairs x NUM_RAYS octree-accelerated
 *            visibility rays) are computed on the GPU, one thread per pair.
 *  - OpenMP: host-side loops (Tau, wave propagation, distances, sample staging).
 *
 * Random sampling uses the same single Mersenne Twister stream as the sequential
 * algorithm. Each rank's rows, and each thread's slice of them, start at their own
 * position in that stream via MT19937 jump-ahead, so results are bit-identical to the
 * sequential run regardless of rank or thread count.
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
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

// Functions shared by the host and the GPU kernels
#define RS_HD __host__ __device__

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
constexpr val_t NO_HIT = std::numeric_limits<val_t>::max();

// RNG draws consumed per non-culled pair: two points (u, v each) per ray, per point
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
// Form-factor pairs staged per GPU batch (bounds pinned/device buffer sizes)
constexpr size_t FF_PAIRS_PER_CHUNK = size_t(1) << 16;
// Host threads that generate sample draws for a rank's form factors
constexpr size_t FF_STAGING_THREADS = 16;
// Seed of the sample stream (sequential in the original program)
constexpr uint32_t SAMPLE_SEED = 42;
// Traversal stack size on the GPU; must cover 8 * octree depth + 1
constexpr int MAX_TRAVERSAL_STACK = 128;
constexpr int GPU_BLOCK_SIZE = 256;

// Rank bookkeeping (set in main)
int g_rank = 0;
int g_size = 1;

// Console output is produced by the root rank only
#define ROOT_PRINTF(...) do { if (g_rank == 0) printf(__VA_ARGS__); } while (0)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n", g_rank,      \
                    __FILE__, __LINE__, cudaGetErrorString(err_));              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    RS_HD constexpr Vec3() : x(0), y(0), z(0) {}
    RS_HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    RS_HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    RS_HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    RS_HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    RS_HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    RS_HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    RS_HD Vec3 operator-() const { return {-x, -y, -z}; }

    RS_HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    RS_HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    RS_HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    RS_HD val_t norm() const { return sqrtf(squaredNorm()); }
    RS_HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    RS_HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    RS_HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    RS_HD Vec3 center() const { return (a + b + c) / 3.0f; }
    RS_HD Vec3 normal() const { return _normal; }

    RS_HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    RS_HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

// Flattened octree node, laid out for upload to the GPU
struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];     // Index into FlatOctree::nodes, -1 if absent
    uint32_t triBegin;    // Range in FlatOctree::tris (leaves only)
    uint32_t triCount;    // Zero for internal nodes
};

struct FlatOctree {
    std::vector<FlatNode> nodes;  // nodes[0] is the root
    std::vector<idx_t> tris;      // Triangle indices of all leaves
    size_t maxDepth = 0;
};

// Non-owning view of a flattened octree plus the triangle list (device-usable)
struct OctreeView {
    const FlatNode* nodes;
    const idx_t* tris;
    const Triangle* all;
};

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

    // Copy the tree into flat arrays for GPU traversal (same leaves and boxes)
    FlatOctree flatten() const {
        FlatOctree flat;
        flattenNode(*this, flat, 0);
        if (8 * flat.maxDepth + 1 > static_cast<size_t>(MAX_TRAVERSAL_STACK)) {
            fprintf(stderr, "Octree depth %zu exceeds GPU traversal stack\n", flat.maxDepth);
            std::abort();
        }
        return flat;
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

    static int32_t flattenNode(const Octree& node, FlatOctree& flat, size_t depth) {
        int32_t id = static_cast<int32_t>(flat.nodes.size());
        flat.nodes.emplace_back();
        flat.maxDepth = std::max(flat.maxDepth, depth);

        FlatNode fn{};
        fn.center = node.center;
        fn.halfExtent = node.halfExtent;
        fn.triBegin = static_cast<uint32_t>(flat.tris.size());
        fn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) flat.tris.push_back(static_cast<idx_t>(idx));
        for (int i = 0; i < 8; ++i) {
            fn.child[i] = node.children[i] ? flattenNode(*node.children[i], flat, depth + 1) : -1;
        }
        flat.nodes[id] = fn;
        return id;
    }
};

// Check if a ray segment intersects a node's bounding box
RS_HD bool rayIntersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
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

// Mersenne Twister jump-ahead. Positioning the sample stream by discarding draws costs
// time proportional to the position, so for large offsets the engine state is computed
// directly. The engine is linear over GF(2): the state of block M (624 outputs per block)
// is sum_k r_k B_k, where B_k is block k and r(y) = y^M mod psi(y), psi being the
// minimal polynomial of the block sequence (recovered with Berlekamp-Massey).
namespace mt_jump {

constexpr size_t BLOCK = 624;             // Outputs per mt19937 block (state size)
constexpr size_t BM_TERMS = 40064;        // Above 2 x 19937, the Mersenne Twister recurrence degree
using Block = std::array<uint32_t, BLOCK>;
using Bits = std::vector<uint64_t>;       // Bit-packed GF(2) polynomial, bit i = coefficient of y^i

// Regenerate a block in place (same recurrence as the standard library engine)
void twist(Block& x) {
    constexpr size_t M = 397;
    constexpr uint32_t MATRIX_A = 0x9908b0dfu, UPPER = 0x80000000u, LOWER = 0x7fffffffu;
    for (size_t k = 0; k < BLOCK; ++k) {
        uint32_t y = (x[k] & UPPER) | (x[(k + 1) % BLOCK] & LOWER);
        x[k] = x[(k + M) % BLOCK] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }
}

// Standard Mersenne Twister tempering
uint32_t temper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// dst ^= src << shift (truncated to dst's size)
void xorShiftedInto(Bits& dst, const Bits& src, size_t shift) {
    const size_t wordOff = shift / 64;
    const unsigned bitOff = shift % 64;
    for (size_t i = 0; i < src.size() && i + wordOff < dst.size(); ++i) {
        dst[i + wordOff] ^= src[i] << bitOff;
        if (bitOff && i + wordOff + 1 < dst.size()) {
            dst[i + wordOff + 1] ^= src[i] >> (64 - bitOff);
        }
    }
}

// Berlekamp-Massey over GF(2): returns the degree L and connection polynomial C (C[0] = 1)
size_t berlekampMassey(const std::vector<uint8_t>& s, Bits& C) {
    const size_t words = s.size() / 64 + 2;
    Bits B(words, 0), T, R(words, 0);  // R bit i holds s[n - i]
    C.assign(words, 0);
    C[0] = 1;
    B[0] = 1;
    size_t L = 0;
    long m = -1;

    for (size_t n = 0; n < s.size(); ++n) {
        for (size_t w = words - 1; w > 0; --w) R[w] = (R[w] << 1) | (R[w - 1] >> 63);
        R[0] <<= 1;
        if (n > 0) R[0] |= static_cast<uint64_t>(s[n - 1]) << 1;

        uint64_t parity = 0;
        for (size_t w = 0; w < words; ++w) parity ^= __builtin_popcountll(C[w] & R[w]);
        if ((s[n] ^ (parity & 1)) == 0) continue;

        T = C;
        xorShiftedInto(C, B, static_cast<size_t>(static_cast<long>(n) - m));
        if (2 * L <= n) {
            L = n + 1 - L;
            B = T;
            m = static_cast<long>(n);
        }
    }
    return L;
}

// Reduction rules for psi(y) = y^L + sum_{i=1..L} C_i y^(L-i): returns P with y^L = P(y)
Bits reductionOf(const Bits& C, size_t L) {
    Bits P((L + 63) / 64 + 1, 0);
    for (size_t k = 0; k < L; ++k) {
        if ((C[(L - k) / 64] >> ((L - k) % 64)) & 1) P[k / 64] |= uint64_t(1) << (k % 64);
    }
    return P;
}

// Reduce a polynomial modulo psi, leaving only coefficients below L
void reduceMod(Bits& a, size_t L, const Bits& P) {
    for (size_t j = a.size() * 64; j-- > L;) {
        if (!((a[j / 64] >> (j % 64)) & 1)) continue;
        a[j / 64] ^= uint64_t(1) << (j % 64);
        xorShiftedInto(a, P, j - L);
    }
}

// r(y) = y^e mod psi(y)
Bits yPowMod(uint64_t e, size_t L, const Bits& P) {
    const size_t words = (L + 63) / 64;
    Bits r(words, 0);
    r[0] = 1;
    bool started = false;
    for (int b = 63; b >= 0; --b) {
        if (started) {
            Bits sq(2 * words + 1, 0);
            for (size_t i = 0; i < L; ++i) {
                if ((r[i / 64] >> (i % 64)) & 1) sq[(2 * i) / 64] |= uint64_t(1) << ((2 * i) % 64);
            }
            reduceMod(sq, L, P);
            sq.resize(words);
            r = std::move(sq);
        }
        if ((e >> b) & 1) {
            started = true;
            Bits shifted(words + 1, 0);
            for (size_t i = 0; i < words; ++i) {
                shifted[i] ^= r[i] << 1;
                shifted[i + 1] ^= r[i] >> 63;
            }
            reduceMod(shifted, L, P);
            shifted.resize(words);
            r = std::move(shifted);
        }
    }
    return r;
}

// Recurrence of the block sequence of the sample stream, computed once
struct Recurrence {
    size_t degree;        // L
    Bits reduction;       // y^L mod psi(y), as coefficients below L
    Block firstBlock;     // Block 0 of the stream
};

const Recurrence& recurrence() {
    static const Recurrence rec = [] {
        // Raw state of a fresh engine, then block 0 = its first regeneration
        Block b0{};
        std::ostringstream os;
        os << std::mt19937(SAMPLE_SEED);
        std::istringstream is(os.str());
        for (auto& w : b0) is >> w;
        twist(b0);

        // Minimal polynomial of the block sequence, from bit 0 of each block's first output
        std::vector<uint8_t> bits(BM_TERMS);
        Block blk = b0;
        for (auto& bit : bits) {
            bit = static_cast<uint8_t>(temper(blk[0]) & 1);
            twist(blk);
        }
        Bits C;
        const size_t L = berlekampMassey(bits, C);
        return Recurrence{L, reductionOf(C, L), b0};
    }();
    return rec;
}

// Engine whose next output is draw number `offset` of the sample stream
std::mt19937 engineAt(uint64_t offset) {
    const Recurrence& rec = recurrence();

    // Block (offset / BLOCK) as a combination of the first L blocks
    const Bits r = yPowMod(offset / BLOCK, rec.degree, rec.reduction);
    Block target{};
    Block blk = rec.firstBlock;
    for (size_t k = 0; k < rec.degree; ++k) {
        if ((r[k / 64] >> (k % 64)) & 1) {
            for (size_t i = 0; i < BLOCK; ++i) target[i] ^= blk[i];
        }
        twist(blk);
    }

    // Load the block as raw state with position 0, then skip the remainder within the block
    std::ostringstream os;
    for (auto w : target) os << w << ' ';
    os << 0;
    std::istringstream is(os.str());
    std::mt19937 engine;
    is >> engine;
    engine.discard(offset % BLOCK);
    return engine;
}

}  // namespace mt_jump

class RandomGenerator {
    std::mt19937 rng;
public:
    // Stream positioned at draw number `offset` (each rand() consumes exactly one engine output)
    explicit RandomGenerator(uint64_t offset = 0) : rng(SAMPLE_SEED) {
        constexpr uint64_t JUMP_THRESHOLD = uint64_t(1) << 26;
        if (offset < JUMP_THRESHOLD) {
            rng.discard(offset);
        } else {
            rng = mt_jump::engineAt(offset);
        }
    }
    // Same value as std::uniform_real_distribution<float>(0, 1) on one engine output u:
    // float(u) / 2^32, clamped below 1. Computed directly because the library's
    // long-double conversion dominated the sampling cost.
    val_t rand() {
        val_t r = static_cast<val_t>(rng()) * 0x1p-32f;
        return r < ONE ? r : 0x1.fffffep-1f;
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

RS_HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                  const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle.
// Walks the flattened octree with an explicit stack: O(log N) instead of O(N).
RS_HD bool isRayBlocked(const Vec3& from, const Vec3& to,
                        const OctreeView& oct,
                        size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int32_t stack[MAX_TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;  // Root box is not tested, matching the recursive octree walk

    while (sp > 0) {
        const FlatNode& node = oct.nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                idx_t idx = oct.tris[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = oct.all[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int32_t child = node.child[c];
                if (child >= 0 && rayIntersectsBox(oct.nodes[child], from, to)) {
                    stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
RS_HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Point inside a triangle from two uniform samples (barycentric coordinates)
RS_HD Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Two triangles facing the same direction exchange no energy (host and GPU cull identically)
bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Form factor Kij between triangle i (receiver) and triangle j (emitter), using the
// sample stream draws[0..DRAWS_PER_PAIR) in the order: pI.u, pI.v, pJ.u, pJ.v per ray
RS_HD val_t computeKijFromDraws(size_t idxI, size_t idxJ, const Triangle* triangles,
                                const OctreeView& oct, const val_t* draws) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r, draws += 4) {
        Vec3 pI = pointInTriangle(triI, draws[0], draws[1]);
        Vec3 pJ = pointInTriangle(triJ, draws[2], draws[3]);

        if (isRayBlocked(pI, pJ, oct, idxI, idxJ)) continue;

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

// One thread per (row, emitter) pair of a batch. Pairs with drawBase < 0 (diagonal or culled)
// are zero. Results go to the rank's owned Kij rows, which start at rowBegin.
__global__ void formFactorKernel(const Triangle* triangles, OctreeView oct,
                                 const val_t* draws, const int32_t* drawBase,
                                 const int32_t* rowIds, size_t numRows, size_t N,
                                 size_t rowBegin, val_t* kij) {
    size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= numRows * N) return;

    size_t i = static_cast<size_t>(rowIds[g / N]);
    size_t j = g % N;
    int32_t base = drawBase[g];
    val_t value = ZERO;
    if (i != j && base >= 0) {
        value = computeKijFromDraws(i, j, triangles, oct, draws + base);
    }
    kij[(i - rowBegin) * N + j] = value;
}

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    // Receiver rows [rowBegin, rowEnd) owned by this rank
    size_t rowBegin;
    size_t rowEnd;
    std::vector<int> rowCounts;   // Per-rank row counts (for all-gathers)
    std::vector<int> rowDispls;   // Per-rank first row

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors, owned rows (localRows x N, row-major)
    std::vector<int> tau;           // Time delays, owned rows (localRows x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated
    std::vector<val_t> distances;   // Computed distances from source, replicated

    FlatOctree octree;              // Spatial acceleration structure

    size_t sourceIndex;

    size_t numOwnedRows() const { return rowEnd - rowBegin; }
    size_t idxLocal(size_t localRow, size_t j) const { return localRow * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (replicated on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    // Contiguous block of receiver rows per rank
    state.rowCounts.resize(g_size);
    state.rowDispls.resize(g_size);
    for (int r = 0; r < g_size; ++r) {
        size_t begin = static_cast<size_t>(r) * state.numTriangles / g_size;
        size_t end = static_cast<size_t>(r + 1) * state.numTriangles / g_size;
        state.rowDispls[r] = static_cast<int>(begin);
        state.rowCounts[r] = static_cast<int>(end - begin);
    }
    state.rowBegin = static_cast<size_t>(state.rowDispls[g_rank]);
    state.rowEnd = state.rowBegin + static_cast<size_t>(state.rowCounts[g_rank]);

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    Octree octree;
    octree.build(state.triangles);
    state.octree = octree.flatten();

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Non-culled pairs (i != j) of each row in [0, rowEnd); each consumes DRAWS_PER_PAIR draws
std::vector<uint64_t> rowPairCounts(const SimulationState& state, size_t rowEnd) {
    const size_t N = state.numTriangles;
    std::vector<uint64_t> counts(rowEnd, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < rowEnd; ++i) {
        uint64_t pairs = 0;
        for (size_t j = 0; j < N; ++j) {
            if (i != j && !isCulled(state.triangles[i], state.triangles[j])) ++pairs;
        }
        counts[i] = pairs;
    }
    return counts;
}

void computeFormFactors(SimulationState& state) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t rows = state.numOwnedRows();
    state.kij.assign(rows * N, ZERO);
    if (rows == 0) return;

    // The sample stream is sequential over rows. The rank's rows are split into contiguous
    // slices, one per staging thread; each slice has a generator positioned at its first draw
    // (computed by jump-ahead), so draws are produced in parallel in exactly the stream order.
    const size_t T = std::min(FF_STAGING_THREADS, static_cast<size_t>(omp_get_max_threads()));
    std::vector<size_t> sliceBegin(T + 1);
    for (size_t t = 0; t <= T; ++t) sliceBegin[t] = t * rows / T;

    const std::vector<uint64_t> counts = rowPairCounts(state, state.rowEnd);
    std::vector<uint64_t> drawsBeforeRow(state.rowEnd + 1, 0);
    for (size_t i = 0; i < state.rowEnd; ++i) {
        drawsBeforeRow[i + 1] = drawsBeforeRow[i] + static_cast<uint64_t>(DRAWS_PER_PAIR) * counts[i];
    }

    std::vector<std::unique_ptr<RandomGenerator>> gens(T);
    #pragma omp parallel for schedule(static, 1)
    for (size_t t = 0; t < T; ++t) {
        if (sliceBegin[t + 1] > sliceBegin[t]) {
            gens[t] = std::make_unique<RandomGenerator>(drawsBeforeRow[state.rowBegin + sliceBegin[t]]);
        }
    }

    // Device geometry and octree
    Triangle* dTriangles = nullptr;
    FlatNode* dNodes = nullptr;
    idx_t* dOctTris = nullptr;
    val_t* dKij = nullptr;
    CUDA_CHECK(cudaMalloc(&dTriangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(dTriangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dNodes, state.octree.nodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMemcpy(dNodes, state.octree.nodes.data(), state.octree.nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dOctTris, state.octree.tris.size() * sizeof(idx_t)));
    CUDA_CHECK(cudaMemcpy(dOctTris, state.octree.tris.data(), state.octree.tris.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dKij, rows * N * sizeof(val_t)));
    OctreeView view{dNodes, dOctTris, dTriangles};

    // Two slots so the host can stage the next batch while the GPU works on the current one
    struct Slot {
        cudaStream_t stream;
        cudaEvent_t done;
        val_t* hDraws;
        int32_t* hBase;
        int32_t* hRowIds;
        val_t* dDraws;
        int32_t* dBase;
        int32_t* dRowIds;
    };
    // A chunk takes up to chunkRowsPerSlice consecutive rows from every slice
    const size_t chunkRowsPerSlice = std::max<size_t>(1, FF_PAIRS_PER_CHUNK / (T * N));
    const size_t maxChunkRows = std::min(rows, T * chunkRowsPerSlice);
    const size_t maxDraws = maxChunkRows * N * DRAWS_PER_PAIR;
    Slot slots[2];
    for (Slot& s : slots) {
        CUDA_CHECK(cudaStreamCreate(&s.stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost(&s.hDraws, maxDraws * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&s.hBase, maxChunkRows * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&s.hRowIds, maxChunkRows * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&s.dDraws, maxDraws * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&s.dBase, maxChunkRows * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&s.dRowIds, maxChunkRows * sizeof(int32_t)));
    }

    size_t maxSliceRows = 0;
    for (size_t t = 0; t < T; ++t) maxSliceRows = std::max(maxSliceRows, sliceBegin[t + 1] - sliceBegin[t]);
    const size_t chunks = (maxSliceRows + chunkRowsPerSlice - 1) / chunkRowsPerSlice;

    size_t rowsDone = 0;
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        Slot& s = slots[chunk % 2];

        // Previous batch using this slot must be finished before reusing its buffers
        CUDA_CHECK(cudaEventSynchronize(s.done));

        // Rows of this chunk: slice t contributes its next chunkRowsPerSlice rows
        std::vector<size_t> lo(T), hi(T), firstRow(T + 1, 0);
        for (size_t t = 0; t < T; ++t) {
            lo[t] = std::min(sliceBegin[t + 1], sliceBegin[t] + chunk * chunkRowsPerSlice);
            hi[t] = std::min(sliceBegin[t + 1], lo[t] + chunkRowsPerSlice);
            firstRow[t + 1] = firstRow[t] + (hi[t] - lo[t]);
        }
        const size_t nRows = firstRow[T];

        // Pass 1: pairs per slice, so each slice knows where its draws go in the staging buffer
        std::vector<uint64_t> slicePairs(T, 0);
        #pragma omp parallel for schedule(static, 1)
        for (size_t t = 0; t < T; ++t) {
            for (size_t r = lo[t]; r < hi[t]; ++r) {
                size_t i = state.rowBegin + r;
                for (size_t j = 0; j < N; ++j) {
                    if (i != j && !isCulled(state.triangles[i], state.triangles[j])) slicePairs[t]++;
                }
            }
        }
        std::vector<size_t> drawOffset(T + 1, 0);
        for (size_t t = 0; t < T; ++t) {
            drawOffset[t + 1] = drawOffset[t] + static_cast<size_t>(slicePairs[t]) * DRAWS_PER_PAIR;
        }

        // Pass 2: draw samples in stream order (each slice continues its own generator)
        #pragma omp parallel for schedule(static, 1)
        for (size_t t = 0; t < T; ++t) {
            size_t cursor = drawOffset[t];
            for (size_t r = lo[t]; r < hi[t]; ++r) {
                size_t i = state.rowBegin + r;
                size_t chunkRow = firstRow[t] + (r - lo[t]);
                s.hRowIds[chunkRow] = static_cast<int32_t>(i);
                for (size_t j = 0; j < N; ++j) {
                    int32_t base = -1;
                    if (i != j && !isCulled(state.triangles[i], state.triangles[j])) {
                        base = static_cast<int32_t>(cursor);
                        for (int k = 0; k < DRAWS_PER_PAIR; ++k) s.hDraws[cursor++] = gens[t]->rand();
                    }
                    s.hBase[chunkRow * N + j] = base;
                }
            }
        }

        const size_t nPairs = nRows * N;
        CUDA_CHECK(cudaMemcpyAsync(s.dDraws, s.hDraws, drawOffset[T] * sizeof(val_t), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.dBase, s.hBase, nPairs * sizeof(int32_t), cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.dRowIds, s.hRowIds, nRows * sizeof(int32_t), cudaMemcpyHostToDevice, s.stream));

        unsigned int blocks = static_cast<unsigned int>((nPairs + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE);
        formFactorKernel<<<blocks, GPU_BLOCK_SIZE, 0, s.stream>>>(
            dTriangles, view, s.dDraws, s.dBase, s.dRowIds, nRows, N, state.rowBegin, dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(s.done, s.stream));

        rowsDone += nRows;
        if (g_rank == 0) {
            printf("  Progress: %zu/%zu triangles (rank 0 rows)\n", rowsDone, rows);
        }
    }

    for (Slot& s : slots) CUDA_CHECK(cudaStreamSynchronize(s.stream));
    CUDA_CHECK(cudaMemcpy(state.kij.data(), dKij, rows * N * sizeof(val_t), cudaMemcpyDeviceToHost));

    for (Slot& s : slots) {
        CUDA_CHECK(cudaFreeHost(s.hDraws));
        CUDA_CHECK(cudaFreeHost(s.hBase));
        CUDA_CHECK(cudaFreeHost(s.hRowIds));
        CUDA_CHECK(cudaFree(s.dDraws));
        CUDA_CHECK(cudaFree(s.dBase));
        CUDA_CHECK(cudaFree(s.dRowIds));
        CUDA_CHECK(cudaEventDestroy(s.done));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    CUDA_CHECK(cudaFree(dKij));
    CUDA_CHECK(cudaFree(dOctTris));
    CUDA_CHECK(cudaFree(dNodes));
    CUDA_CHECK(cudaFree(dTriangles));
}

// Time delay (in timesteps) between owned receivers and all emitters
int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    const size_t rows = state.numOwnedRows();
    state.tau.assign(rows * N, 0);

    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < rows; ++li) {
        size_t i = state.rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.tau[state.idxLocal(li, j)] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t rows = state.numOwnedRows();

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        // Rows of this timestep only read earlier timesteps (tau >= 1), so owned rows are independent
        #pragma omp parallel for schedule(static)
        for (size_t li = 0; li < rows; ++li) {
            size_t i = state.rowBegin + li;
            val_t sumB = ZERO;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;

                int tauij = state.tau[state.idxLocal(li, j)];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[state.idxLocal(li, j)];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        // Share this timestep's radiosity so every rank has the full history
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                       state.radB.data() + t * N, state.rowCounts.data(), state.rowDispls.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    const size_t rows = state.numOwnedRows();
    std::vector<val_t> localDistances(rows, ZERO);

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t li = 0; li < rows; ++li) {
        size_t i = state.rowBegin + li;
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDistances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(localDistances.data(), static_cast<int>(rows), MPI_FLOAT,
                   state.distances.data(), state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

// Runs on the root rank, which holds the full distance and radiosity history.
// nonZeroKij is the global count of non-zero form factors (reduced across ranks).
bool validateResults(const SimulationState& state, size_t nonZeroKij) {
    ROOT_PRINTF("\nValidation:\n");

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
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
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

// Maximum over all ranks of a per-rank duration in milliseconds
long maxAcrossRanks(long localMs) {
    long globalMs = localMs;
    MPI_Allreduce(&localMs, &globalMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    return globalMs;
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

    // One GPU per rank (ranks on a node share the devices round-robin) and
    // OpenMP threads divided among the ranks sharing a node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "[rank %d] No CUDA devices found\n", g_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("MPI ranks: %d (GPUs: %d per node, OpenMP threads per rank: %d)\n",
                g_size, numDevices, omp_get_max_threads());
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    long preDuration = maxAcrossRanks(std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count());

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    long simDuration = maxAcrossRanks(std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count());

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    long distDuration = maxAcrossRanks(std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count());

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
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

    // Memory usage (per rank: owned rows of Kij and Tau, replicated radiosity)
    size_t memKij = state.numOwnedRows() * n * sizeof(val_t);
    size_t memTau = state.numOwnedRows() * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage (per rank): %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
    int valid = 1;
    if (validate) {
        size_t localNonZeroKij = 0;
        for (val_t k : state.kij) {
            if (k > EPSILON) localNonZeroKij++;
        }
        unsigned long long nonZeroKij = 0;
        unsigned long long localCount = localNonZeroKij;
        MPI_Allreduce(&localCount, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (g_rank == 0) {
            valid = validateResults(state, static_cast<size_t>(nonZeroKij)) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}

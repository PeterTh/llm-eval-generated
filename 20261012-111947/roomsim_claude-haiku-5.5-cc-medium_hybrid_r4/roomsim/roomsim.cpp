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
 * Parallelization (hybrid MPI + OpenMP + CUDA):
 *   - MPI:    receiver triangles (rows i) are block-distributed over ranks. Each rank
 *             owns the form factors, time delays and wave updates of its rows. The
 *             reflected radiosity of each timestep is exchanged with an allgather.
 *   - CUDA:   the form factor rows of each rank are computed on that rank's GPU, one
 *             thread per receiver row, running a device port of the visibility test.
 *   - OpenMP: host loops (time delays, wave updates, cross-correlation, RNG jump-ahead
 *             set-up) are multithreaded over the rank's rows.
 *
 * The random stream of the sequential original is reproduced exactly (see "Exact
 * random-stream support"), so the results are identical to the sequential program.
 */

#include <algorithm>
#include <array>
#include <cfloat>
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

#define HD __host__ __device__

// ============================================================================
// Error Handling and Process Identity
// ============================================================================

static int gRank = 0;
static int gSize = 1;

#define ROOT_PRINTF(...) do { if (gRank == 0) printf(__VA_ARGS__); } while (0)

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
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

// Host-side octree used for construction only
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

// Flattened octree node (POD) that can be traversed on the device.
// A node is a leaf when triCount > 0; otherwise child[] holds node indices (-1 = none).
struct FlatOctreeNode {
    Vec3 center, halfExtent;
    int32_t child[8];
    uint32_t triBegin, triCount;
};

// Copies the host octree into flat arrays; node 0 is the root.
int32_t flattenOctree(const Octree& node, std::vector<FlatOctreeNode>& nodes,
                      std::vector<uint32_t>& triIdx) {
    const int32_t id = static_cast<int32_t>(nodes.size());
    nodes.push_back({});

    FlatOctreeNode flat{};
    flat.center = node.center;
    flat.halfExtent = node.halfExtent;
    for (int32_t& c : flat.child) c = -1;

    if (!node.triangleIndices.empty()) {
        flat.triBegin = static_cast<uint32_t>(triIdx.size());
        flat.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) triIdx.push_back(static_cast<uint32_t>(idx));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) flat.child[c] = flattenOctree(*node.children[c], nodes, triIdx);
        }
    }

    nodes[id] = flat;
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
// Exact random-stream support (MT19937 jump-ahead)
// ============================================================================
//
// The sequential original draws every sample from a single std::mt19937 stream
// (seed 42), in row-major (i, j) order; culled pairs draw nothing. The draws of
// row i therefore start at a position that depends on the culling of all earlier
// rows. To compute rows independently while reproducing that stream exactly, each
// row's engine state is reconstructed by jump-ahead:
//
//  - The MT19937 output sequence obeys a linear recurrence over GF(2) with a
//    primitive characteristic polynomial Q of degree L (found with
//    Berlekamp-Massey on the output bits).
//  - For a position p, x^p mod Q gives the coefficients that express the 624
//    outputs starting at p as combinations of the first L outputs.
//  - Untempering those 624 outputs yields the engine state at p.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_SEED = 42;
constexpr size_t MT_DEGREE_BOUND = 19937;  // Degree of the MT19937 characteristic polynomial

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

inline uint32_t mtUntemper(uint32_t y) {
    y ^= (y >> 18);
    y ^= (y << 15) & 0xefc60000u;
    uint32_t x = y;
    for (int i = 0; i < 5; ++i) x = y ^ ((x << 7) & 0x9d2c5680u);
    y = x;
    return y ^ (y >> 11) ^ (y >> 22);
}

// Polynomial arithmetic over GF(2), bit-packed into 64-bit words
struct JumpAhead {
    size_t deg = 0;                            // L: degree of the characteristic polynomial
    size_t words = 0;                          // words holding a residue modulo Q
    std::vector<uint64_t> q;                   // Q(x), including the x^L term
    std::vector<uint32_t> base;                // Tempered outputs w_0 .. w_{L-1}
    std::vector<std::vector<uint64_t>> xpow;   // x^(2^k) mod Q
};

inline bool gf2Bit(const uint64_t* v, size_t k) { return (v[k >> 6] >> (k & 63)) & 1; }

// dst ^= src << shift
inline void gf2XorShifted(uint64_t* dst, const uint64_t* src, size_t srcWords, size_t shift) {
    const size_t ws = shift >> 6;
    const size_t bs = shift & 63;
    for (size_t w = 0; w < srcWords; ++w) {
        const uint64_t v = src[w];
        if (!v) continue;
        dst[w + ws] ^= v << bs;
        if (bs) dst[w + ws + 1] ^= v >> (64 - bs);
    }
}

// out = a * b mod Q
void gf2MulMod(const JumpAhead& ja, const uint64_t* a, const uint64_t* b, uint64_t* out) {
    const size_t W = ja.words;
    std::vector<uint64_t> prod(2 * W + 1, 0);
    for (size_t w = 0; w < W; ++w) {
        uint64_t v = a[w];
        while (v) {
            const size_t bit = static_cast<size_t>(__builtin_ctzll(v));
            v &= v - 1;
            gf2XorShifted(prod.data(), b, W, w * 64 + bit);
        }
    }
    for (size_t k = 2 * ja.deg - 2; k >= ja.deg; --k) {
        if (gf2Bit(prod.data(), k)) gf2XorShifted(prod.data(), ja.q.data(), W, k - ja.deg);
    }
    std::copy(prod.begin(), prod.begin() + static_cast<std::ptrdiff_t>(W), out);
}

// out = x^e mod Q
void gf2PowX(const JumpAhead& ja, uint64_t e, uint64_t* out) {
    std::vector<uint64_t> acc(ja.words, 0), tmp(ja.words, 0);
    acc[0] = 1;
    for (size_t k = 0; e != 0; ++k, e >>= 1) {
        if (e & 1) {
            gf2MulMod(ja, acc.data(), ja.xpow[k].data(), tmp.data());
            acc.swap(tmp);
        }
    }
    std::copy(acc.begin(), acc.end(), out);
}

// Engine state (untempered) of the MT19937 stream at position p, where x^p mod Q = X.
// The engine state is the 624 outputs starting at p, untempered.
void gf2EngineWindow(const JumpAhead& ja, const uint64_t* X, uint32_t* mtOut) {
    const size_t W = ja.words;
    std::vector<uint64_t> D(X, X + W);
    for (int t = 0; t < MT_N; ++t) {
        uint32_t word = 0;
        for (size_t w = 0; w < W; ++w) {
            uint64_t v = D[w];
            while (v) {
                const size_t k = w * 64 + static_cast<size_t>(__builtin_ctzll(v));
                v &= v - 1;
                word ^= ja.base[k];
            }
        }
        mtOut[t] = mtUntemper(word);

        // D <- x * D mod Q
        for (size_t w = W - 1; w > 0; --w) D[w] = (D[w] << 1) | (D[w - 1] >> 63);
        D[0] <<= 1;
        if (gf2Bit(D.data(), ja.deg)) {
            for (size_t w = 0; w < W; ++w) D[w] ^= ja.q[w];
        }
    }
}

// Berlekamp-Massey over GF(2). Returns connection polynomial C (C[0] = 1) with
// s[n] = sum_{k=1..L} C[k] s[n-k] for all n >= L.
// Bit-packed: the sequence is stored reversed so that the window s[i], s[i-1], ...
// is a forward bit range, and the discrepancy is a parity over word-wise ANDs.
std::vector<uint8_t> berlekampMassey(const std::vector<uint8_t>& s, size_t& Lout) {
    const size_t n = s.size();
    const size_t words = n / 64 + 2;               // bits 0..n of C and B
    const size_t padWords = 2 * (n / 64) + 4;      // room for windows and shifted XORs
    std::vector<uint64_t> rev(padWords, 0);        // rev bit (n-1-k) holds s[k]
    for (size_t k = 0; k < n; ++k) {
        if (s[k]) {
            const size_t pos = n - 1 - k;
            rev[pos >> 6] |= 1ull << (pos & 63);
        }
    }

    std::vector<uint64_t> C(padWords, 0), B(padWords, 0), T;
    C[0] = 1;
    B[0] = 1;
    size_t L = 0;
    long m = -1;
    for (size_t i = 0; i < n; ++i) {
        // Discrepancy d = sum_{k=0..L} C[k] s[i-k] (mod 2); s[i-k] is rev bit n-1-i+k
        const size_t base = n - 1 - i;
        uint64_t parity = 0;
        for (size_t w = 0; w <= L / 64; ++w) {
            const size_t pos = base + 64 * w;
            const size_t ws = pos >> 6;
            const size_t bs = pos & 63;
            uint64_t window = rev[ws] >> bs;
            if (bs) window |= rev[ws + 1] << (64 - bs);
            parity ^= C[w] & window;
        }
        if (__builtin_parityll(parity)) {
            T = C;
            const size_t shift = static_cast<size_t>(static_cast<long>(i) - m);
            gf2XorShifted(C.data(), B.data(), words, shift);
            if (2 * L <= i) {
                L = i + 1 - L;
                m = static_cast<long>(i);
                B = T;
            }
        }
    }

    std::vector<uint8_t> out(L + 1);
    for (size_t k = 0; k <= L; ++k) out[k] = static_cast<uint8_t>((C[k >> 6] >> (k & 63)) & 1u);
    Lout = L;
    return out;
}

// Builds the jump-ahead tables. Rank 0 runs Berlekamp-Massey; the result is broadcast.
JumpAhead buildJumpAhead(uint64_t maxPosition) {
    JumpAhead ja;
    constexpr size_t NB = 2 * MT_DEGREE_BOUND + 64;  // Output bits used by Berlekamp-Massey

    uint64_t L = 0;
    std::vector<uint64_t> qTmp;
    if (gRank == 0) {
        std::mt19937 gen(MT_SEED);
        std::vector<uint8_t> bits(NB);
        for (size_t k = 0; k < NB; ++k) bits[k] = static_cast<uint8_t>(gen() & 1u);

        size_t degree = 0;
        const std::vector<uint8_t> C = berlekampMassey(bits, degree);
        if (2 * degree > NB) {
            fprintf(stderr, "Jump-ahead: recurrence degree %zu exceeds the bit budget\n", degree);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        L = degree;
        qTmp.assign(L / 64 + 1, 0);
        // Q(x) = sum_k C[k] x^(L-k)
        for (size_t k = 0; k <= L; ++k) {
            if (C[k]) {
                const size_t bit = L - k;
                qTmp[bit >> 6] |= 1ull << (bit & 63);
            }
        }
    }
    MPI_Bcast(&L, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    ja.deg = static_cast<size_t>(L);
    ja.words = ja.deg / 64 + 1;
    if (gRank != 0) qTmp.assign(ja.words, 0);
    MPI_Bcast(qTmp.data(), static_cast<int>(ja.words), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    ja.q = qTmp;

    // Base outputs w_0 .. w_{L-1}
    std::mt19937 gen(MT_SEED);
    ja.base.resize(ja.deg);
    for (size_t k = 0; k < ja.deg; ++k) ja.base[k] = static_cast<uint32_t>(gen());

    // Squarings x^(2^k) mod Q for every exponent bit that can occur
    size_t bitsNeeded = 1;
    while ((uint64_t{1} << bitsNeeded) <= maxPosition) ++bitsNeeded;
    ja.xpow.assign(bitsNeeded, std::vector<uint64_t>(ja.words, 0));
    ja.xpow[0][0] = 2;  // x
    for (size_t k = 1; k < bitsNeeded; ++k) gf2MulMod(ja, ja.xpow[k - 1].data(), ja.xpow[k - 1].data(), ja.xpow[k].data());
    return ja;
}

// ============================================================================
// Random Number Generation (device side)
// ============================================================================

// Per-thread MT19937 state, as used by one receiver row on the device
struct MTState {
    uint32_t mt[MT_N];
    uint32_t idx;
};

__device__ inline uint32_t mtNextRaw(MTState& s) {
    if (s.idx >= static_cast<uint32_t>(MT_N)) {
        for (int k = 0; k < MT_N; ++k) {
            const uint32_t y = (s.mt[k] & 0x80000000u) | (s.mt[(k + 1) % MT_N] & 0x7fffffffu);
            s.mt[k] = s.mt[(k + MT_M) % MT_N] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        s.idx = 0;
    }
    return mtTemper(s.mt[s.idx++]);
}

// Same value as std::uniform_real_distribution<float>(0,1) over std::mt19937
__device__ inline val_t mtUniform(MTState& s) {
    const val_t u = static_cast<val_t>(mtNextRaw(s)) / 4294967296.0f;
    return u < ONE ? u : 0.99999994f;  // largest float below 1
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
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

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                      const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Ray-box slab test of a flattened octree node (separating axis form)
__device__ bool nodeIntersectsRay(const FlatOctreeNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

constexpr int TRAVERSAL_STACK_SIZE = 128;

// Check if a ray between two triangles is blocked by any other triangle.
// Traverses the flattened octree, returning true as soon as a blocking triangle is found.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const FlatOctreeNode* nodes, const uint32_t* triIdx, const Triangle* tris,
                             uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    uint32_t stack[TRAVERSAL_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root
    while (sp > 0) {
        const FlatOctreeNode& node = nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                const uint32_t idx = triIdx[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                const val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int32_t ch = node.child[c];
                if (ch >= 0 && nodeIntersectsRay(nodes[ch], from, to)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter).
// `draws` holds the 4 uniform numbers of each ray (NUM_RAYS * 4 values, in stream order).
// The caller guarantees that the pair is neither the diagonal nor culled.
__device__ val_t computeKij(size_t idxI, size_t idxJ,
                            const FlatOctreeNode* nodes, const uint32_t* triIdx, const Triangle* tris,
                            const val_t* draws) {
    const Triangle& triI = tris[idxI];
    const Triangle& triJ = tris[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* d = draws + 4 * r;
        Vec3 pI = randomPointInTriangle(triI, d[0], d[1]);
        Vec3 pJ = randomPointInTriangle(triJ, d[2], d[3]);

        if (isRayBlocked(pI, pJ, nodes, triIdx, tris, static_cast<uint32_t>(idxI), static_cast<uint32_t>(idxJ))) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI._normal);
        val_t cosPhiJ = cosPhi(-v, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// Stage 1: one thread per row of the chunk. Walks the row's engine stream and stores
// the uniform numbers of every non-skipped pair consecutively, starting at rowOffset.
// prefix[r * N + j] receives the number of drawing pairs before j in row r.
__global__ void drawKernel(size_t chunkRows, size_t rowBegin, size_t N, size_t maskWords,
                           const uint64_t* skipBits, const uint32_t* rowStates,
                           const uint64_t* rowOffset, val_t* draws, uint32_t* prefix) {
    const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (r >= chunkRows) return;

    const size_t rl = rowBegin + r;  // row index local to the rank
    MTState rng;
    for (int k = 0; k < MT_N; ++k) rng.mt[k] = rowStates[rl * MT_N + k];
    rng.idx = 0;

    const uint64_t* skip = skipBits + rl * maskWords;
    uint32_t count = 0;
    for (size_t j = 0; j < N; ++j) {
        prefix[r * N + j] = count;
        const bool skipPair = (skip[j >> 6] >> (j & 63)) & 1u;
        if (skipPair) continue;
        val_t* out = draws + rowOffset[r] + static_cast<size_t>(count) * 4 * NUM_RAYS;
        for (int k = 0; k < 4 * NUM_RAYS; ++k) out[k] = mtUniform(rng);
        ++count;
    }
}

// Stage 2: one thread per (row, emitter) pair of the chunk.
__global__ void pairKernel(size_t chunkRows, size_t rowBegin, size_t i0, size_t N, size_t maskWords,
                           const Triangle* tris, const FlatOctreeNode* nodes, const uint32_t* triIdx,
                           const uint64_t* skipBits, const uint64_t* rowOffset,
                           const val_t* draws, const uint32_t* prefix, val_t* kij) {
    const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (p >= chunkRows * N) return;

    const size_t r = p / N;
    const size_t j = p % N;
    const size_t rl = rowBegin + r;
    const size_t i = i0 + rl;

    const bool skipPair = (skipBits[rl * maskWords + (j >> 6)] >> (j & 63)) & 1u;
    if (skipPair) {
        kij[rl * N + j] = ZERO;
        return;
    }
    const val_t* d = draws + rowOffset[r] + static_cast<size_t>(prefix[r * N + j]) * 4 * NUM_RAYS;
    kij[rl * N + j] = computeKij(i, j, nodes, triIdx, tris, d);
}

// ============================================================================
// Simulation State
// ============================================================================

// Splits N rows into contiguous blocks; rank r owns [begin, begin + count)
void partitionRows(size_t N, int rank, int size, size_t& begin, size_t& count) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    count = base + (r < rem ? 1 : 0);
    begin = r * base + std::min(r, rem);
}

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;

    size_t rowBegin;   // First receiver row owned by this rank
    size_t rowCount;   // Number of receiver rows owned by this rank

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle (replicated)
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0) (replicated)
    std::vector<val_t> kij;         // Form factors of owned rows (rowCount x N, row-major)
    std::vector<int> tau;           // Time delays of owned rows (rowCount x N, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N, replicated)
    std::vector<val_t> radB;        // Reflected radiosity (T x N, replicated)
    std::vector<val_t> distances;   // Computed distances (N, replicated after gather)

    std::vector<FlatOctreeNode> octNodes;  // Flattened octree for the device
    std::vector<uint32_t> octTris;

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
    partitionRows(state.numTriangles, gRank, gSize, state.rowBegin, state.rowCount);

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration and flatten it for the device
    ROOT_PRINTF("Building octree...\n");
    Octree octree;
    octree.build(state.triangles);
    state.octNodes.clear();
    state.octTris.clear();
    flattenOctree(octree, state.octNodes, state.octTris);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.assign(state.rowCount * state.numTriangles, ZERO);
    state.tau.assign(state.rowCount * state.numTriangles, 0);
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < state.rowCount; ++r) {
        const size_t i = state.rowBegin + r;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.tau[r * N + j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// Pair (i, j) is skipped when i == j or when the facing normals are nearly parallel
// (culled). skipBits holds one bit per pair (row-major, maskWords words per row),
// and drawCount[i] is the number of pairs of row i that draw random numbers.
void buildSkipBits(const std::vector<Triangle>& tris, size_t N, size_t maskWords,
                   std::vector<uint64_t>& skipBits, std::vector<uint64_t>& drawCount) {
    skipBits.assign(N * maskWords, 0);
    drawCount.assign(N, 0);

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i) {
        uint64_t* row = skipBits.data() + i * maskWords;
        const Vec3 ni = tris[i]._normal;
        uint64_t skipped = 0;
        for (size_t j = 0; j < N; ++j) {
            const bool skip = (i == j) || (ni.dot(tris[j]._normal) > 0.99f);
            if (skip) {
                row[j >> 6] |= 1ull << (j & 63);
                ++skipped;
            }
        }
        drawCount[i] = N - skipped;
    }
}

// Computes form factors of the rank's rows on its GPU
void computeFormFactors(SimulationState& state, const JumpAhead& ja,
                        const std::vector<uint64_t>& skipBits, size_t maskWords,
                        const std::vector<uint64_t>& drawPos) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t R = state.rowCount;
    const size_t i0 = state.rowBegin;

    // Engine state at the start of every owned row (jump-ahead, parallel over threads)
    std::vector<uint32_t> rowStates(R * MT_N, 0);
    #pragma omp parallel
    {
        const size_t nthr = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t b0 = R * tid / nthr;
        const size_t b1 = R * (tid + 1) / nthr;
        if (b0 < b1) {
            std::vector<uint64_t> X(ja.words), next(ja.words), step(ja.words);
            std::map<uint64_t, std::vector<uint64_t>> stepCache;
            gf2PowX(ja, drawPos[i0 + b0], X.data());
            for (size_t r = b0; r < b1; ++r) {
                gf2EngineWindow(ja, X.data(), &rowStates[r * MT_N]);
                if (r + 1 < b1) {
                    const uint64_t draws = drawPos[i0 + r + 1] - drawPos[i0 + r];
                    auto it = stepCache.find(draws);
                    if (it == stepCache.end()) {
                        gf2PowX(ja, draws, step.data());
                        it = stepCache.emplace(draws, step).first;
                    }
                    gf2MulMod(ja, X.data(), it->second.data(), next.data());
                    X.swap(next);
                }
            }
        }
    }

    // Device buffers: triangles, octree and the rank's cull bits, states and results
    Triangle* dTris = nullptr;
    FlatOctreeNode* dNodes = nullptr;
    uint32_t* dTriIdx = nullptr;
    uint64_t* dSkip = nullptr;
    uint32_t* dStates = nullptr;
    val_t* dKij = nullptr;
    CUDA_CHECK(cudaMalloc(&dTris, state.triangles.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&dNodes, state.octNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&dTriIdx, state.octTris.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dSkip, std::max<size_t>(R * maskWords, 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dStates, std::max<size_t>(R * MT_N, 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dKij, std::max<size_t>(R * N, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(dTris, state.triangles.data(), state.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodes, state.octNodes.data(), state.octNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTriIdx, state.octTris.data(), state.octTris.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (R > 0) {
        CUDA_CHECK(cudaMemcpy(dSkip, skipBits.data() + i0 * maskWords, R * maskWords * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dStates, rowStates.data(), R * MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Rows are processed in chunks so that the materialized uniform numbers
    // (64 per drawing pair) fit in a share of the free device memory. A larger chunk
    // runs more row streams concurrently, which is what keeps the GPU busy.
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t rowDrawBytes = 4 * NUM_RAYS * N * sizeof(val_t);
    const size_t chunkRowsCap = std::min(std::max<size_t>(R, 1),
                                         std::max<size_t>(1, (freeBytes * 3 / 5) / rowDrawBytes));

    uint64_t* dOffset = nullptr;
    val_t* dDraws = nullptr;
    uint32_t* dPrefix = nullptr;
    CUDA_CHECK(cudaMalloc(&dOffset, (chunkRowsCap + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dDraws, std::max<size_t>(chunkRowsCap * N * 4 * NUM_RAYS, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dPrefix, std::max<size_t>(chunkRowsCap * N, 1) * sizeof(uint32_t)));

    constexpr int THREADS = 256;
    for (size_t c0 = 0; c0 < R; c0 += chunkRowsCap) {
        const size_t chunkRows = std::min(chunkRowsCap, R - c0);

        // Offset of each row's first draw within this chunk's draw buffer
        std::vector<uint64_t> offset(chunkRows);
        for (size_t r = 0; r < chunkRows; ++r) offset[r] = drawPos[i0 + c0 + r] - drawPos[i0 + c0];
        CUDA_CHECK(cudaMemcpy(dOffset, offset.data(), chunkRows * sizeof(uint64_t), cudaMemcpyHostToDevice));

        const int drawBlocks = static_cast<int>((chunkRows + 31) / 32);
        drawKernel<<<drawBlocks, 32>>>(chunkRows, c0, N, maskWords, dSkip, dStates, dOffset, dDraws, dPrefix);
        CUDA_CHECK(cudaGetLastError());

        const size_t pairs = chunkRows * N;
        const int pairBlocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
        pairKernel<<<pairBlocks, THREADS>>>(chunkRows, c0, i0, N, maskWords, dTris, dNodes, dTriIdx,
                                            dSkip, dOffset, dDraws, dPrefix, dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(state.kij.data() + c0 * N, dKij + c0 * N, chunkRows * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dTris));
    CUDA_CHECK(cudaFree(dNodes));
    CUDA_CHECK(cudaFree(dTriIdx));
    CUDA_CHECK(cudaFree(dSkip));
    CUDA_CHECK(cudaFree(dStates));
    CUDA_CHECK(cudaFree(dKij));
    CUDA_CHECK(cudaFree(dOffset));
    CUDA_CHECK(cudaFree(dDraws));
    CUDA_CHECK(cudaFree(dPrefix));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Per-rank element counts and displacements for a row-block distribution
void rowBlockLayout(size_t N, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(gSize, 0);
    displs.assign(gSize, 0);
    for (int r = 0; r < gSize; ++r) {
        size_t begin = 0, count = 0;
        partitionRows(N, r, gSize, begin, count);
        counts[r] = static_cast<int>(count);
        displs[r] = static_cast<int>(begin);
    }
}

void runSimulation(SimulationState& state, const std::vector<int>& counts, const std::vector<int>& displs) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < state.rowCount; ++r) {
            const size_t i = state.rowBegin + r;
            val_t sumB = ZERO;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;

                int tauij = state.tau[r * N + j];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[r * N + j];
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

        // Share this timestep's radiosity of every rank's rows
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.radB.data() + t * N,
                       counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, const std::vector<int>& counts, const std::vector<int>& displs) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < state.rowCount; ++r) {
        const size_t i = state.rowBegin + r;
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

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   counts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, size_t nonZeroKij) {
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
    size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
           nonZeroKij, totalKij,
           100.0f * nonZeroKij / static_cast<val_t>(totalKij));

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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);

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
            if (gRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (gRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Bind each rank to one GPU of its node and share the node's cores between its ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, gRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, ranksOnNode = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &ranksOnNode);
    MPI_Comm_free(&nodeComm);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / ranksOnNode));
    }
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("MPI ranks: %d, OpenMP threads per rank: %d, GPUs per rank: 1\n", gSize, omp_get_max_threads());
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    const size_t N = state.numTriangles;
    const size_t maskWords = (N + 63) / 64;
    std::vector<int> counts, displs;
    rowBlockLayout(N, counts, displs);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = MPI_Wtime();

    computeTimeDelays(state);

    std::vector<uint64_t> skipBits, drawCount;
    buildSkipBits(state.triangles, N, maskWords, skipBits, drawCount);
    std::vector<uint64_t> drawPos(N + 1, 0);
    for (size_t i = 0; i < N; ++i) drawPos[i + 1] = drawPos[i] + 64 * drawCount[i];
    JumpAhead ja = buildJumpAhead(drawPos[N]);

    computeFormFactors(state, ja, skipBits, maskWords, drawPos);

    MPI_Barrier(MPI_COMM_WORLD);
    double endPre = MPI_Wtime();
    auto preDuration = static_cast<long>((endPre - startPre) * 1000.0);

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = MPI_Wtime();

    runSimulation(state, counts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    double endSim = MPI_Wtime();
    auto simDuration = static_cast<long>((endSim - startSim) * 1000.0);

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = MPI_Wtime();

    computeDistances(state, counts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    double endDist = MPI_Wtime();
    auto distDuration = static_cast<long>((endDist - startDist) * 1000.0);

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Non-zero form factor count over the whole matrix
    uint64_t localNonZeroKij = 0;
    for (val_t k : state.kij) {
        if (k > EPSILON) localNonZeroKij++;
    }
    uint64_t nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

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

    // Memory usage (whole problem, as in the sequential program)
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    int exitCode = 0;
    if (gRank == 0) {
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
            if (!validateResults(state, static_cast<size_t>(nonZeroKij))) {
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}

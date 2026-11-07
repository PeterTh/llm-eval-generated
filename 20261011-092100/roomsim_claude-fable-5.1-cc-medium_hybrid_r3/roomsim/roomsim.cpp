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
 * Parallelization strategy
 * ------------------------
 *  - MPI: the N x N pair matrices (Kij, Tau) are distributed by contiguous row
 *    blocks over the ranks; every rank drives one GPU. Only O(N) data (one
 *    radiosity row per timestep, the final distances) is exchanged.
 *  - CUDA: the O(N^2) form factors with octree-accelerated visibility, the
 *    O(N^2 T) wave propagation and the O(N T^2) cross-correlation run as GPU
 *    kernels on the rank-local row block.
 *  - OpenMP: host-side preparation (geometry upload, time delay matrix,
 *    validation scans).
 *
 * Reproducing the sequential random stream
 * ----------------------------------------
 * The original code draws all Monte-Carlo samples from one sequential
 * std::mt19937(42). The number of samples per (i,j) pair is deterministic
 * (0 if the pair is normal-culled, 2*2*NUM_RAYS otherwise), so the stream
 * offset of every pair is known from a prefix count. An MT19937 jump-ahead
 * (characteristic polynomial via Berlekamp-Massey, x^(2^k) mod P table on the
 * host; polynomial application and parallel twists on the GPU) lets every
 * generator segment start at an arbitrary stream offset, so the samples are
 * bit-identical to the sequential program while being generated on the GPU.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
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
constexpr int RNG_PER_PAIR = 4 * NUM_RAYS;  // random numbers consumed per visible pair
constexpr uint32_t RNG_SEED = 42;

// GPU tiling of the flattened (row, column) pair space
constexpr int PAIR_TILE = 256;
// Maximum number of visible pairs per streamed chunk (-> 32 floats each)
constexpr size_t CHUNK_PAIRS = size_t(4) << 20;

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

// Flattened octree for the GPU. Children of a node are stored contiguously.
// A node is a leaf iff triCount > 0 (same convention as the pointer tree).
struct DNode {
    float cx, cy, cz;      // center
    float hx, hy, hz;      // half extent
    int triStart, triCount;
    int childStart, childCount;
};
// GPU layout: two 16-byte words per node.
//   a = (cx, cy, cz, hx)
//   b = (hy, hz, start, count) with start/count stored as int bits;
//   count > 0: leaf with `count` triangles at nodeTris[start..]
//   count < 0: inner node with -count children at nodes[start..]
struct DNodePacked {
    float4 a, b;
};

struct FlatOctree {
    std::vector<DNode> nodes;
    std::vector<int> tris;
    std::vector<DNodePacked> packed;

    void build(const Octree& root) {
        nodes.clear();
        tris.clear();
        nodes.push_back(DNode{});
        fill(root, 0);
        packed.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            const DNode& n = nodes[i];
            DNodePacked p;
            p.a = make_float4(n.cx, n.cy, n.cz, n.hx);
            int start = n.triCount > 0 ? n.triStart : n.childStart;
            int count = n.triCount > 0 ? n.triCount : -n.childCount;
            float fs, fc;
            memcpy(&fs, &start, sizeof(float));
            memcpy(&fc, &count, sizeof(float));
            p.b = make_float4(n.hy, n.hz, fs, fc);
            packed[i] = p;
        }
    }

private:
    void fill(const Octree& n, int self) {
        DNode d{};
        d.cx = n.center.x; d.cy = n.center.y; d.cz = n.center.z;
        d.hx = n.halfExtent.x; d.hy = n.halfExtent.y; d.hz = n.halfExtent.z;
        d.triStart = static_cast<int>(tris.size());
        d.triCount = static_cast<int>(n.triangleIndices.size());
        for (size_t t : n.triangleIndices) tris.push_back(static_cast<int>(t));
        int childCount = 0;
        for (int i = 0; i < 8; ++i) if (n.children[i]) childCount++;
        d.childStart = static_cast<int>(nodes.size());
        d.childCount = childCount;
        nodes[self] = d;
        for (int i = 0; i < childCount; ++i) nodes.push_back(DNode{});
        int k = 0;
        for (int i = 0; i < 8; ++i) {
            if (n.children[i]) {
                fill(*n.children[i], d.childStart + k);
                k++;
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
// Random Number Generation: MT19937 with jump-ahead
// ============================================================================
//
// Bit-exact re-implementation of std::mt19937 (as used by the original
// RandomGenerator, seed 42) in circular-buffer form, plus an O(1)-per-bit
// jump-ahead so that arbitrary contiguous segments of the sequential random
// stream can be produced by independent threads.

struct MTState {
    uint32_t x[624];
    int h;

    void seed(uint32_t s) {
        x[0] = s;
        for (int i = 1; i < 624; ++i)
            x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
        h = 0;
    }
    inline uint32_t step() {
        int h1 = h + 1; if (h1 >= 624) h1 -= 624;
        int hm = h + 397; if (hm >= 624) hm -= 624;
        uint32_t y = (x[h] & 0x80000000u) | (x[h1] & 0x7fffffffu);
        uint32_t v = x[hm] ^ (y >> 1) ^ ((x[h1] & 1u) ? 0x9908b0dfu : 0u);
        x[h] = v;
        h = h1;
        return v;
    }
    static inline uint32_t temper(uint32_t y) {
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }
};

using Poly = std::vector<uint64_t>;  // bit i = coefficient of x^i

static inline int polyDeg(const Poly& p) {
    for (int w = static_cast<int>(p.size()) - 1; w >= 0; --w)
        if (p[w]) return w * 64 + 63 - __builtin_clzll(p[w]);
    return -1;
}
static inline bool polyBit(const Poly& p, int i) { return (p[i >> 6] >> (i & 63)) & 1; }
static inline void polySet(Poly& p, int i) { p[i >> 6] |= (1ull << (i & 63)); }

// Berlekamp-Massey over GF(2): minimal connection polynomial of a bit sequence.
static Poly berlekampMassey(const std::vector<uint8_t>& s, int& Lout) {
    const int n = static_cast<int>(s.size());
    const int W = n / 64 + 2;
    Poly C(W, 0), B(W, 0), T(W, 0);
    C[0] = 1; B[0] = 1;
    int L = 0, m = 1;
    // Reversed packed sequence: bit k of R = s[n-1-k]
    std::vector<uint64_t> R(W + 2, 0);
    for (int i = 0; i < n; ++i)
        if (s[i]) R[(n - 1 - i) >> 6] |= 1ull << ((n - 1 - i) & 63);
    auto getR = [&](long b) -> uint64_t {
        int wi = static_cast<int>(b >> 6), sh = static_cast<int>(b & 63);
        uint64_t lo = wi < static_cast<int>(R.size()) ? R[wi] : 0;
        uint64_t hi = (wi + 1) < static_cast<int>(R.size()) ? R[wi + 1] : 0;
        return sh ? ((lo >> sh) | (hi << (64 - sh))) : lo;
    };
    for (int N = 0; N < n; ++N) {
        const long base = n - 1 - N;
        uint64_t acc = 0;
        const int lw = L / 64 + 1;
        for (int w = 0; w < lw; ++w) acc ^= C[w] & getR(base + 64L * w);
        if (__builtin_parityll(acc)) {
            T = C;
            const int ws = m >> 6, bs = m & 63;
            for (int w = W - 1; w >= ws; --w) {
                uint64_t v = B[w - ws] << bs;
                if (bs && w - ws - 1 >= 0) v |= B[w - ws - 1] >> (64 - bs);
                C[w] ^= v;
            }
            if (2 * L <= N) { L = N + 1 - L; B = T; m = 1; }
            else m++;
        } else {
            m++;
        }
    }
    Lout = L;
    return C;
}

// Arithmetic in GF(2)[x] / P(x)
struct ModPoly {
    Poly P;
    int deg, words;
    explicit ModPoly(const Poly& p) : P(p) { deg = polyDeg(p); words = deg / 64 + 1; }

    void reduce(Poly& r) const {
        for (int i = polyDeg(r); i >= deg; --i) {
            if (polyBit(r, i)) {
                const int sh = i - deg, ws = sh >> 6, bs = sh & 63;
                for (int w = 0; w <= words; ++w) {
                    const int tw = w + ws;
                    if (tw >= static_cast<int>(r.size())) break;
                    r[tw] ^= P[w] << bs;
                    if (bs && tw + 1 < static_cast<int>(r.size())) r[tw + 1] ^= P[w] >> (64 - bs);
                }
            }
        }
    }
    Poly mul(const Poly& a, const Poly& b) const {
        Poly r(2 * words + 2, 0);
        for (int i = 0; i < deg; ++i) {
            if (polyBit(a, i)) {
                const int ws = i >> 6, bs = i & 63;
                for (int w = 0; w < words; ++w) {
                    r[w + ws] ^= b[w] << bs;
                    if (bs) r[w + ws + 1] ^= b[w] >> (64 - bs);
                }
            }
        }
        reduce(r);
        r.resize(words + 1);
        return r;
    }
};

// Builds the table x^(2^k) mod P(x), where P is the characteristic polynomial
// of the MT19937 state transition. Jumping the state by J outputs is then the
// application of the polynomials for the set bits of J (done on the GPU).
constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DEG = 19937;
constexpr int MT_PW = MT_DEG / 64 + 1;  // 64-bit words per residue polynomial

constexpr int MT_JUMP_BITS = 48;        // table covers jumps below 2^48 outputs

class MTJumpTable {
    Poly P;
    int deg = 0;
public:
    std::vector<uint64_t> table;  // MT_JUMP_BITS x MT_PW words
    int nb = MT_JUMP_BITS;

    MTJumpTable() {
        MTState s;
        s.seed(5489u);
        s.step();  // x_0 carries 31 ignored bits; start the bit sequence at x_1
        const int n = 2 * MT_DEG + 128;
        std::vector<uint8_t> seq(n);
        for (int i = 0; i < n; ++i) {
            seq[i] = static_cast<uint8_t>(s.x[s.h] & 1u);
            s.step();
        }
        int L = 0;
        Poly C = berlekampMassey(seq, L);
        if (L != MT_DEG) {
            fprintf(stderr, "MT19937 characteristic polynomial has unexpected degree %d\n", L);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        deg = L;
        // Characteristic polynomial is the reciprocal of the connection polynomial
        P.assign(L / 64 + 2, 0);
        for (int k = 0; k <= L; ++k)
            if (polyBit(C, L - k)) polySet(P, k);
        ModPoly mp(P);
        table.assign(static_cast<size_t>(nb) * MT_PW, 0);
        Poly cur(mp.words + 1, 0);
        polySet(cur, 1);
        for (int k = 0; k < nb; ++k) {
            for (int w = 0; w < MT_PW; ++w) table[static_cast<size_t>(k) * MT_PW + w] = cur[w];
            if (k + 1 < nb) cur = mp.mul(cur, cur);
        }
    }
};

// ============================================================================
// Device MT19937: state jump and stream generation
// ============================================================================

__device__ __forceinline__ uint32_t mtRecur(uint32_t x0, uint32_t x1, uint32_t xm) {
    uint32_t y = (x0 & 0x80000000u) | (x1 & 0x7fffffffu);
    return xm ^ (y >> 1) ^ ((x1 & 1u) ? 0x9908b0dfu : 0u);
}
__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}
// std::uniform_real_distribution<float>(0,1) on mt19937 (libstdc++ generate_canonical)
__device__ __forceinline__ float mtToFloat(uint32_t u) {
    float r = __uint2float_rn(u) * 2.3283064365386963e-10f;  // exact / 2^32
    return (r >= 1.0f) ? 0.99999994f : r;
}

constexpr int MT_BLOCK = 256;
constexpr int MT_SCRATCH = MT_N + MT_DEG;  // words of scratch per segment

// Apply the polynomial g (MT_PW words, degree < MT_DEG) to the state S held
// in shared memory: S <- g(A) S, where A is the one-step transition. Uses the
// identity (A^i S)[k] = x_{i+k}: the x sequence is generated into X (scratch
// of MT_SCRATCH words, parallel twists of 227 values) and XOR-combined.
__device__ void mtApplyPoly(const uint64_t* __restrict__ g, uint32_t* S, uint32_t* R, uint32_t* X) {
    const int tid = threadIdx.x;
    for (int k = tid; k < MT_N; k += MT_BLOCK) X[k] = S[k];
    __syncthreads();
    for (int s0 = 0; s0 < MT_DEG; s0 += MT_N - MT_M) {
        const int n = s0 + tid;
        if (tid < MT_N - MT_M && n < MT_DEG) X[MT_N + n] = mtRecur(X[n], X[n + 1], X[n + MT_M]);
        __syncthreads();
    }
    uint32_t a0 = 0, a1 = 0, a2 = 0;
    const bool has2 = (tid + 2 * MT_BLOCK) < MT_N;
    for (int w = 0; w < MT_PW; ++w) {
        uint64_t word = g[w];
        while (word) {
            const int i = w * 64 + (__ffsll(static_cast<long long>(word)) - 1);
            word &= word - 1;
            a0 ^= X[i + tid];
            a1 ^= X[i + tid + MT_BLOCK];
            if (has2) a2 ^= X[i + tid + 2 * MT_BLOCK];
        }
    }
    R[tid] = a0;
    R[tid + MT_BLOCK] = a1;
    if (has2) R[tid + 2 * MT_BLOCK] = a2;
    __syncthreads();
    for (int k = tid; k < MT_N; k += MT_BLOCK) S[k] = R[k];
    __syncthreads();
}

// Single block: states[0] = generator state at stream offset `base` (from seed).
__global__ void __launch_bounds__(MT_BLOCK)
mtJumpBaseKernel(const uint64_t* __restrict__ pow2, int nb, unsigned long long base, uint32_t seed,
                 uint32_t* __restrict__ scratch, uint32_t* __restrict__ states) {
    __shared__ uint32_t S[MT_N];
    __shared__ uint32_t R[MT_N];
    const int tid = threadIdx.x;
    if (tid == 0) {
        S[0] = seed;
        for (int i = 1; i < MT_N; ++i)
            S[i] = 1812433253u * (S[i - 1] ^ (S[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    __syncthreads();
    for (int b = 0; b < nb; ++b)
        if ((base >> b) & 1ull) mtApplyPoly(pow2 + static_cast<size_t>(b) * MT_PW, S, R, scratch);
    for (int k = tid; k < MT_N; k += MT_BLOCK) states[k] = S[k];
}

// Doubling level: states[half + g] = states[g] advanced by half * segLen
// (polynomial x^(half*segLen) mod P, passed as `g`), for g < gridDim.x.
__global__ void __launch_bounds__(MT_BLOCK)
mtJumpLevelKernel(const uint64_t* __restrict__ g, int half, uint32_t* __restrict__ scratch,
                  uint32_t* __restrict__ states) {
    __shared__ uint32_t S[MT_N];
    __shared__ uint32_t R[MT_N];
    const int tid = threadIdx.x;
    const int src = blockIdx.x;
    for (int k = tid; k < MT_N; k += MT_BLOCK) S[k] = states[static_cast<size_t>(src) * MT_N + k];
    __syncthreads();
    mtApplyPoly(g, S, R, scratch + static_cast<size_t>(blockIdx.x) * MT_SCRATCH);
    for (int k = tid; k < MT_N; k += MT_BLOCK) states[static_cast<size_t>(half + src) * MT_N + k] = S[k];
}

// One block per segment: emit uniform floats [segStart, segEnd) of the stream.
__global__ void __launch_bounds__(MT_BLOCK)
mtGenKernel(const uint32_t* __restrict__ states, unsigned long long segLen, unsigned long long total,
            float* __restrict__ out) {
    __shared__ uint32_t B[2 * MT_N];
    const int tid = threadIdx.x;
    const unsigned long long segStart = static_cast<unsigned long long>(blockIdx.x) * segLen;
    if (segStart >= total) return;
    const unsigned long long segEnd = min(segStart + segLen, total);
    for (int k = tid; k < MT_N; k += MT_BLOCK) B[k] = states[static_cast<size_t>(blockIdx.x) * MT_N + k];
    __syncthreads();
    for (unsigned long long pos = segStart; pos < segEnd; pos += MT_N) {
        for (int s0 = 0; s0 < MT_N; s0 += MT_N - MT_M) {
            const int n = s0 + tid;
            if (tid < MT_N - MT_M && n < MT_N) B[MT_N + n] = mtRecur(B[n], B[n + 1], B[n + MT_M]);
            __syncthreads();
        }
        for (int k = tid; k < MT_N; k += MT_BLOCK) {
            const unsigned long long p = pos + k;
            if (p < segEnd) out[p] = mtToFloat(mtTemper(B[MT_N + k]));
        }
        __syncthreads();
        for (int k = tid; k < MT_N; k += MT_BLOCK) B[k] = B[MT_N + k];
        __syncthreads();
    }
}

// ============================================================================
// Device geometry helpers
// ============================================================================

__device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return f3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float3 operator-(float3 a) { return f3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float norm3(float3 a) { return sqrtf(dot3(a, a)); }

// Normal-culling test shared by the counting and the form-factor kernels.
// Uses non-fused arithmetic so both kernels evaluate it identically.
__device__ __forceinline__ bool pairVisible(const float4* __restrict__ tA, const float4* __restrict__ tB,
                                            const float4* __restrict__ tC, int i, int j) {
    if (i == j) return false;
    float4 ai = tA[i], bi = tB[i], ci = tC[i];
    float4 aj = tA[j], bj = tB[j], cj = tC[j];
    float d = __fadd_rn(__fadd_rn(__fmul_rn(ai.w, aj.w), __fmul_rn(bi.w, bj.w)), __fmul_rn(ci.w, cj.w));
    return !(d > 0.99f);
}

// Möller-Trumbore (same operation order as the host reference)
__device__ __forceinline__ float rayTriangleIntersectDev(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    float invDet = 1.0f / det;
    float3 tvec = orig - v0;
    float u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    float3 qvec = cross3(tvec, e1);
    float v = dot3(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return dot3(e2, qvec) * invDet;
}

__device__ __forceinline__ bool rayIntersectsBoxDev(float4 na, float4 nb, float3 d, float3 pd, float3 ad) {
    const float hx = na.w, hy = nb.x, hz = nb.y;
    float3 c = pd - f3(na.x, na.y, na.z);
    if (fabsf(c.x) > hx + ad.x) return false;
    if (fabsf(c.y) > hy + ad.y) return false;
    if (fabsf(c.z) > hz + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > hy * ad.z + hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > hz * ad.x + hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > hx * ad.y + hy * ad.x + EPSILON) return false;
    return true;
}

// Octree-accelerated occlusion query (iterative DFS, same visited set as the
// recursive host version; the result is order independent).
__device__ bool isRayBlockedDev(float3 from, float3 to, int srcTri, int dstTri,
                                const DNodePacked* __restrict__ nodes, const int* __restrict__ nodeTris,
                                const float4* __restrict__ tA, const float4* __restrict__ tB,
                                const float4* __restrict__ tC) {
    float3 dir = to - from;
    float rayLen = norm3(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = dir / rayLen;

    float3 d = (to - from) * 0.5f;
    float3 pd = from + d;
    float3 ad = f3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const float4 nb = nodes[stack[--sp]].b;
        const int start = __float_as_int(nb.z);
        const int count = __float_as_int(nb.w);
        if (count > 0) {
            for (int k = 0; k < count; ++k) {
                int idx = nodeTris[start + k];
                if (idx == srcTri || idx == dstTri) continue;
                float4 a = tA[idx], b = tB[idx], c = tC[idx];
                float dist = rayTriangleIntersectDev(from, dirNorm, f3(a.x, a.y, a.z), f3(b.x, b.y, b.z), f3(c.x, c.y, c.z));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int k = 0; k < -count; ++k) {
                const int ch = start + k;
                const float4 ca = nodes[ch].a, cb = nodes[ch].b;
                if (rayIntersectsBoxDev(ca, cb, d, pd, ad)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

__device__ __forceinline__ float cosPhiDev(float3 v, float3 normal) {
    float vNorm = norm3(v);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, dot3(v, normal) / vNorm);
}

__device__ __forceinline__ float3 randomPointInTriangleDev(float3 a, float3 b, float3 c, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = b - a;
    float3 ac = c - a;
    return a + ab * u + ac * v;
}

// ============================================================================
// Kernels
// ============================================================================

// Number of visible (RNG consuming) pairs per tile of PAIR_TILE flattened pairs
__global__ void tileCountKernel(const float4* __restrict__ tA, const float4* __restrict__ tB,
                                const float4* __restrict__ tC, int N, int nLocal, int rowStart,
                                int* __restrict__ counts) {
    __shared__ int warpCounts[PAIR_TILE / 32];
    const long long p = static_cast<long long>(blockIdx.x) * PAIR_TILE + threadIdx.x;
    bool vis = false;
    if (p < static_cast<long long>(nLocal) * N) {
        int iL = static_cast<int>(p / N);
        int j = static_cast<int>(p - static_cast<long long>(iL) * N);
        vis = pairVisible(tA, tB, tC, iL + rowStart, j);
    }
    unsigned ballot = __ballot_sync(0xffffffffu, vis);
    if ((threadIdx.x & 31) == 0) warpCounts[threadIdx.x >> 5] = __popc(ballot);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int w = 0; w < PAIR_TILE / 32; ++w) s += warpCounts[w];
        counts[blockIdx.x] = s;
    }
}

// Form factors for tiles [tileStart, tileStart + gridDim.x)
__global__ void __launch_bounds__(PAIR_TILE, 5)
formFactorKernel(const float4* __restrict__ tA, const float4* __restrict__ tB, const float4* __restrict__ tC,
                 const DNodePacked* __restrict__ nodes, const int* __restrict__ nodeTris,
                 const float* __restrict__ rng, const long long* __restrict__ tilePrefix,
                 long long chunkBase, int tileStart, int N, int nLocal, int rowStart,
                 float* __restrict__ kij) {
    __shared__ int warpCounts[PAIR_TILE / 32];
    const int tile = tileStart + blockIdx.x;
    const long long p = static_cast<long long>(tile) * PAIR_TILE + threadIdx.x;
    const bool inRange = p < static_cast<long long>(nLocal) * N;
    int iL = 0, j = 0, i = 0;
    bool vis = false;
    if (inRange) {
        iL = static_cast<int>(p / N);
        j = static_cast<int>(p - static_cast<long long>(iL) * N);
        i = iL + rowStart;
        vis = pairVisible(tA, tB, tC, i, j);
    }
    // Exclusive prefix of visible pairs inside the tile
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned ballot = __ballot_sync(0xffffffffu, vis);
    if (lane == 0) warpCounts[warp] = __popc(ballot);
    __syncthreads();
    int before = 0;
    for (unsigned w = 0; w < warp; ++w) before += warpCounts[w];
    before += __popc(ballot & ((1u << lane) - 1u));

    if (!inRange) return;
    // Stored transposed (kij[j * nLocal + iL]) so the propagation kernel can
    // walk j sequentially with coalesced accesses.
    const size_t outIdx = static_cast<size_t>(j) * nLocal + iL;
    if (!vis) {
        kij[outIdx] = ZERO;
        return;
    }

    const size_t off = static_cast<size_t>(tilePrefix[tile] - chunkBase + before) * RNG_PER_PAIR;
    const float4* r4 = reinterpret_cast<const float4*>(rng + off);

    float4 a4 = tA[i], b4 = tB[i], c4 = tC[i];
    float3 aI = f3(a4.x, a4.y, a4.z), bI = f3(b4.x, b4.y, b4.z), cI = f3(c4.x, c4.y, c4.z);
    float3 nI = f3(a4.w, b4.w, c4.w);
    a4 = tA[j]; b4 = tB[j]; c4 = tC[j];
    float3 aJ = f3(a4.x, a4.y, a4.z), bJ = f3(b4.x, b4.y, b4.z), cJ = f3(c4.x, c4.y, c4.z);
    float3 nJ = f3(a4.w, b4.w, c4.w);

    float k = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        float4 u = r4[r];
        float3 pI = randomPointInTriangleDev(aI, bI, cI, u.x, u.y);
        float3 pJ = randomPointInTriangleDev(aJ, bJ, cJ, u.z, u.w);

        if (isRayBlockedDev(pI, pJ, i, j, nodes, nodeTris, tA, tB, tC)) continue;

        float3 v = pJ - pI;
        float distSqr = dot3(v, v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = cosPhiDev(v, nI);
        float cosPhiJ = cosPhiDev(-v, nJ);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        k += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    kij[outIdx] = k * INV_NUM_RAYS;
}

// Count kij > EPSILON and replace kij by the propagation weight min(kij*area_j, 1)
__global__ void weightKernel(float* __restrict__ kij, const float* __restrict__ areas, int nLocal,
                             size_t total, unsigned long long* __restrict__ nonZero) {
    __shared__ int warpCounts[8];
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int nz = 0;
    if (p < total) {
        float k = kij[p];
        nz = (k > EPSILON) ? 1 : 0;
        int j = static_cast<int>(p / nLocal);
        kij[p] = fminf(__fmul_rn(k, areas[j]), ONE);
    }
    unsigned ballot = __ballot_sync(0xffffffffu, nz);
    if ((threadIdx.x & 31) == 0) warpCounts[threadIdx.x >> 5] = __popc(ballot);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) s += warpCounts[w];
        if (s) atomicAdd(nonZero, static_cast<unsigned long long>(s));
    }
}

// One timestep of radiosity propagation for the local rows. One thread per
// row walks j in order, so the accumulation order and rounding (separate
// multiply/add per term, fused multiply-add for the final reflection) match
// the sequential reference. Matrices are stored transposed (j-major) so the
// per-j accesses of consecutive rows coalesce.
template <typename TauT>
__global__ void simStepKernel(const float* __restrict__ wT, const TauT* __restrict__ tauT,
                              const float* __restrict__ radB, int t, int N, int nLocal, int rowStart,
                              float rho, int srcIdx, int timeOff, float* __restrict__ outRow) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nLocal) return;
    float sum = ZERO;
    #pragma unroll 8
    for (int j = 0; j < N; ++j) {
        const size_t idx = static_cast<size_t>(j) * nLocal + row;
        const int tj = static_cast<int>(tauT[idx]);
        const float wj = wT[idx];
        if (t < tj || wj <= ZERO) continue;
        const float radJ = radB[static_cast<size_t>(t - tj) * N + j];
        if (radJ <= ZERO) continue;
        sum = __fadd_rn(sum, __fmul_rn(wj, radJ));
    }
    const int i = rowStart + row;
    const float radE = (i == srcIdx && t < timeOff) ? ONE : ZERO;
    outRow[row] = fmaf(rho, sum, radE);
}

// radBT[row][t] = radB[t][rowStart + row]
__global__ void transposeRadKernel(const float* __restrict__ radB, int N, int T, int nLocal, int rowStart,
                                   float* __restrict__ radBT) {
    __shared__ float tileS[32][33];
    int t0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int k = ty; k < 32; k += blockDim.y) {
        int t = t0 + k, r = r0 + tx;
        tileS[k][tx] = (t < T && r < nLocal) ? radB[static_cast<size_t>(t) * N + rowStart + r] : ZERO;
    }
    __syncthreads();
    for (int k = ty; k < 32; k += blockDim.y) {
        int r = r0 + k, t = t0 + tx;
        if (r < nLocal && t < T) radBT[static_cast<size_t>(r) * T + t] = tileS[tx][k];
    }
}

// Cross-correlation distance estimate, one block per local row.
// The reference loop was auto-vectorized by the host compiler as an in-order
// reduction: blocks of 8 then 4 products are rounded individually and added
// sequentially, and up to 3 trailing terms use a fused multiply-add; loops
// shorter than 8 are fully scalar (separate multiply/add). Reproduced here.
__device__ __forceinline__ float crossCorr(const float* __restrict__ srcCol, const float* __restrict__ col,
                                           int t, int T) {
    const int count = T - t;
    float sum = ZERO;
    int k = 0;
    if (count >= 8) {
        const int n8 = count & ~7;
        for (; k < n8; ++k) sum = __fadd_rn(sum, __fmul_rn(srcCol[k], col[t + k]));
        const int rem = count - n8;
        if (rem >= 4) {
            for (int e = k + 4; k < e; ++k) sum = __fadd_rn(sum, __fmul_rn(srcCol[k], col[t + k]));
        }
        for (; k < count; ++k) sum = fmaf(srcCol[k], col[t + k], sum);
    } else {
        for (; k < count; ++k) sum = __fadd_rn(sum, __fmul_rn(srcCol[k], col[t + k]));
    }
    return sum;
}

__global__ void distanceKernel(const float* __restrict__ radBT, const float* __restrict__ srcCol,
                               int T, float* __restrict__ dist) {
    __shared__ float bestVal[1024];
    __shared__ int bestIdx[1024];
    const int row = blockIdx.x;
    const float* col = radBT + static_cast<size_t>(row) * T;
    float bv = ZERO;
    int bt = INT_MAX;  // INT_MAX = no candidate
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float sum = crossCorr(srcCol, col, t, T);
        if (sum > bv) { bv = sum; bt = t; }  // t ascending per thread -> first max
    }
    bestVal[threadIdx.x] = bv;
    bestIdx[threadIdx.x] = bt;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mv = ZERO;
        int mt = 0;
        for (int k = 0; k < static_cast<int>(blockDim.x); ++k) {
            if (bestIdx[k] == INT_MAX) continue;
            if (bestVal[k] > mv || (bestVal[k] == mv && bestIdx[k] < mt)) { mv = bestVal[k]; mt = bestIdx[k]; }
        }
        dist[row] = WAVE_SPEED * static_cast<float>(mt);
    }
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
    val_t rho;                      // Reflectivity (0.0 to 1.0), uniform
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source (gathered on rank 0)

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;

    size_t sourceIndex;

    // MPI decomposition
    int rank = 0, nranks = 1;
    int rowStart = 0, nLocal = 0;
    std::vector<int> rowCounts, rowDispls;

    // Device data
    float4* dTA = nullptr; float4* dTB = nullptr; float4* dTC = nullptr;
    DNodePacked* dNodes = nullptr; int* dNodeTris = nullptr;
    float* dAreas = nullptr;
    float* dKij = nullptr;          // N x nLocal (j-major): form factors, later propagation weights
    void* dTau = nullptr;           // N x nLocal (j-major): uint8 or uint16
    bool tau8 = true;
    float* dRadB = nullptr;         // T x N
    float* dRowOut = nullptr;       // nLocal
    uint64_t* dPow2 = nullptr;      // MT19937 jump polynomials x^(2^k) mod P
    cudaStream_t stream = nullptr;     // compute stream
    cudaStream_t genStream = nullptr;  // random stream generation (overlapped)

    unsigned long long nonZeroKij = 0;  // global count (rank 0)

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

    if (state.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.flatOctree.build(state.octree);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho = reflectivity;

    // Row decomposition over ranks
    const int N = static_cast<int>(state.numTriangles);
    state.rowCounts.resize(state.nranks);
    state.rowDispls.resize(state.nranks);
    for (int r = 0; r < state.nranks; ++r) {
        int s = static_cast<int>(static_cast<long long>(N) * r / state.nranks);
        int e = static_cast<int>(static_cast<long long>(N) * (r + 1) / state.nranks);
        state.rowDispls[r] = s;
        state.rowCounts[r] = e - s;
    }
    state.rowStart = state.rowDispls[state.rank];
    state.nLocal = state.rowCounts[state.rank];

    // Replicated radiosity history (filled row by row during the simulation)
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Upload static geometry
    std::vector<float4> hA(N), hB(N), hC(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 n = t.normal();
        hA[i] = make_float4(t.a.x, t.a.y, t.a.z, n.x);
        hB[i] = make_float4(t.b.x, t.b.y, t.b.z, n.y);
        hC[i] = make_float4(t.c.x, t.c.y, t.c.z, n.z);
    }
    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(&state.dTA, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTB, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTC, N * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(state.dTA, hA.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTB, hB.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTC, hC.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dNodes, state.flatOctree.packed.size() * sizeof(DNodePacked)));
    CUDA_CHECK(cudaMemcpy(state.dNodes, state.flatOctree.packed.data(),
                          state.flatOctree.packed.size() * sizeof(DNodePacked), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dNodeTris, std::max<size_t>(1, state.flatOctree.tris.size()) * sizeof(int)));
    if (!state.flatOctree.tris.empty())
        CUDA_CHECK(cudaMemcpy(state.dNodeTris, state.flatOctree.tris.data(),
                              state.flatOctree.tris.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.dAreas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const size_t localPairs = static_cast<size_t>(std::max(state.nLocal, 1)) * N;
    CUDA_CHECK(cudaMalloc(&state.dKij, localPairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * state.numTriangles * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * state.numTriangles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRowOut, std::max(state.nLocal, 1) * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&state.genStream));

    // Jump-ahead polynomials of the MT19937 generator (problem independent)
    {
        MTJumpTable jumpTable;
        CUDA_CHECK(cudaMalloc(&state.dPow2, jumpTable.table.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(state.dPow2, jumpTable.table.data(), jumpTable.table.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    }

    // Force (lazy) module loading of all kernels now, outside the timed phases
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, tileCountKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, formFactorKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, weightKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, simStepKernel<uint8_t>));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, simStepKernel<uint16_t>));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, transposeRadKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, distanceKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, mtJumpBaseKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, mtJumpLevelKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&fa, mtGenKernel));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int nLocal = state.nLocal;
    std::vector<Vec3> centers(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) centers[i] = state.triangles[i].center();

    // Local block of the delay matrix (row-major here, uploaded j-major);
    // find the maximum to pick a compact storage type
    std::vector<int> tauLocal(static_cast<size_t>(std::max(nLocal, 1)) * N, 0);
    int maxTau = 0;
    #pragma omp parallel for schedule(static) reduction(max : maxTau)
    for (int iL = 0; iL < nLocal; ++iL) {
        const int i = state.rowStart + iL;
        int* row = tauLocal.data() + static_cast<size_t>(iL) * N;
        for (int j = 0; j < N; ++j) {
            if (i == j) { row[j] = 0; continue; }
            val_t dist = (centers[i] - centers[j]).norm();
            int t = static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
            row[j] = t;
            maxTau = std::max(maxTau, t);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &maxTau, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    state.tau8 = maxTau <= 255;

    const size_t localPairs = static_cast<size_t>(std::max(nLocal, 1)) * N;
    if (state.tau8) {
        std::vector<uint8_t> packed(localPairs);
        #pragma omp parallel for schedule(static)
        for (int iL = 0; iL < nLocal; ++iL)
            for (int j = 0; j < N; ++j)
                packed[static_cast<size_t>(j) * nLocal + iL] = static_cast<uint8_t>(tauLocal[static_cast<size_t>(iL) * N + j]);
        CUDA_CHECK(cudaMalloc(&state.dTau, localPairs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(state.dTau, packed.data(), localPairs * sizeof(uint8_t), cudaMemcpyHostToDevice));
    } else {
        std::vector<uint16_t> packed(localPairs);
        #pragma omp parallel for schedule(static)
        for (int iL = 0; iL < nLocal; ++iL)
            for (int j = 0; j < N; ++j)
                packed[static_cast<size_t>(j) * nLocal + iL] =
                    static_cast<uint16_t>(std::min(tauLocal[static_cast<size_t>(iL) * N + j], 65535));
        CUDA_CHECK(cudaMalloc(&state.dTau, localPairs * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(state.dTau, packed.data(), localPairs * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int nLocal = state.nLocal;
    const size_t localPairs = static_cast<size_t>(nLocal) * N;
    const int numTiles = static_cast<int>((localPairs + PAIR_TILE - 1) / PAIR_TILE);

    // 1. Count visible pairs per tile -> exclusive prefix (global stream layout)
    std::vector<long long> tilePrefix(numTiles + 1, 0);
    long long* dTilePrefix = nullptr;
    CUDA_CHECK(cudaMalloc(&dTilePrefix, (numTiles + 1) * sizeof(long long)));
    if (numTiles > 0) {
        int* dCounts = nullptr;
        CUDA_CHECK(cudaMalloc(&dCounts, numTiles * sizeof(int)));
        tileCountKernel<<<numTiles, PAIR_TILE, 0, state.stream>>>(state.dTA, state.dTB, state.dTC, N, nLocal,
                                                                  state.rowStart, dCounts);
        CUDA_CHECK(cudaGetLastError());
        std::vector<int> counts(numTiles);
        CUDA_CHECK(cudaMemcpyAsync(counts.data(), dCounts, numTiles * sizeof(int), cudaMemcpyDeviceToHost, state.stream));
        CUDA_CHECK(cudaStreamSynchronize(state.stream));
        CUDA_CHECK(cudaFree(dCounts));
        for (int t = 0; t < numTiles; ++t) tilePrefix[t + 1] = tilePrefix[t] + counts[t];
    }
    const long long localVisible = tilePrefix[numTiles];
    long long visibleBefore = 0;
    MPI_Exscan(&localVisible, &visibleBefore, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (state.rank == 0) visibleBefore = 0;
    long long totalVisible = 0;
    MPI_Allreduce(&localVisible, &totalVisible, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(dTilePrefix, tilePrefix.data(), (numTiles + 1) * sizeof(long long), cudaMemcpyHostToDevice));

    // 2. Jump-ahead table for the sequential RNG reproduction (built at init)
    const uint64_t streamTotal = static_cast<uint64_t>(totalVisible) * RNG_PER_PAIR;
    if (streamTotal >= (1ull << (MT_JUMP_BITS - 1))) {
        if (state.rank == 0) fprintf(stderr, "Random stream too long for the jump table\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const uint64_t* dPow2 = state.dPow2;

    // 3. Chunked: generate the stream segment on the GPU, then the visibility kernel
    struct Chunk { int tileA, tileB; long long base, count; };
    std::vector<Chunk> chunks;
    {
        int a = 0;
        while (a < numTiles) {
            int b = a;
            while (b < numTiles && (tilePrefix[b + 1] - tilePrefix[a]) <= static_cast<long long>(CHUNK_PAIRS)) b++;
            if (b == a) b = a + 1;  // single tile larger than the chunk limit (cannot happen: tile <= 256)
            chunks.push_back({a, b, tilePrefix[a], tilePrefix[b] - tilePrefix[a]});
            a = b;
        }
    }
    long long maxChunkPairs = 0;
    for (const auto& c : chunks) maxChunkPairs = std::max(maxChunkPairs, c.count);
    const size_t bufFloats = static_cast<size_t>(std::max<long long>(maxChunkPairs, 1)) * RNG_PER_PAIR;

    constexpr int SEG_LOG2_MIN = 11;      // >= 2048 floats per generator segment
    constexpr int MAX_SEG_LOG2 = 11;      // <= 2048 segments per chunk
    float* dRng[2] = {nullptr, nullptr};
    uint32_t* dStates = nullptr;
    uint32_t* dScratch = nullptr;
    cudaEvent_t evGen[2], evKij[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dRng[b], bufFloats * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&evGen[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evKij[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evKij[b], state.stream));
    }
    CUDA_CHECK(cudaMalloc(&dStates, static_cast<size_t>(1 << MAX_SEG_LOG2) * MT_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dScratch, static_cast<size_t>(1 << (MAX_SEG_LOG2 - 1)) * MT_SCRATCH * sizeof(uint32_t)));

    // Generation of chunk c+1 (genStream) overlaps the visibility kernel of
    // chunk c (compute stream); the two RNG buffers alternate.
    for (size_t c = 0; c < chunks.size(); ++c) {
        const Chunk& ch = chunks[c];
        const int b = static_cast<int>(c & 1);
        const unsigned long long len = static_cast<unsigned long long>(ch.count) * RNG_PER_PAIR;
        CUDA_CHECK(cudaStreamWaitEvent(state.genStream, evKij[b], 0));  // buffer b free
        if (len > 0) {
            const unsigned long long base =
                (static_cast<unsigned long long>(visibleBefore) + static_cast<unsigned long long>(ch.base)) * RNG_PER_PAIR;
            // Segment length 2^s so that the segment start offsets g*2^s are
            // reached by a doubling tree of single polynomial applications.
            int segLog2 = SEG_LOG2_MIN;
            while ((len + (1ull << segLog2) - 1) >> segLog2 > (1ull << MAX_SEG_LOG2)) segLog2++;
            const unsigned long long segLen = 1ull << segLog2;
            const int segments = static_cast<int>((len + segLen - 1) >> segLog2);
            mtJumpBaseKernel<<<1, MT_BLOCK, 0, state.genStream>>>(dPow2, MT_JUMP_BITS, base, RNG_SEED, dScratch, dStates);
            CUDA_CHECK(cudaGetLastError());
            for (int half = 1; half < segments; half <<= 1) {
                int lvl = 0;
                while ((1 << lvl) < half) lvl++;
                const int blocks = std::min(half, segments - half);
                mtJumpLevelKernel<<<blocks, MT_BLOCK, 0, state.genStream>>>(
                    dPow2 + static_cast<size_t>(segLog2 + lvl) * MT_PW, half, dScratch, dStates);
                CUDA_CHECK(cudaGetLastError());
            }
            mtGenKernel<<<segments, MT_BLOCK, 0, state.genStream>>>(dStates, segLen, len, dRng[b]);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(evGen[b], state.genStream));
        CUDA_CHECK(cudaStreamWaitEvent(state.stream, evGen[b], 0));

        formFactorKernel<<<ch.tileB - ch.tileA, PAIR_TILE, 0, state.stream>>>(
            state.dTA, state.dTB, state.dTC, state.dNodes, state.dNodeTris, dRng[b], dTilePrefix,
            ch.base, ch.tileA, N, nLocal, state.rowStart, state.dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evKij[b], state.stream));

        if (state.rank == 0) {
            size_t rowsDone = std::min<size_t>(static_cast<size_t>(ch.tileB) * PAIR_TILE / N, nLocal);
            size_t shown = (c + 1 == chunks.size()) ? state.numTriangles
                                                    : std::min(state.numTriangles, rowsDone * state.nranks);
            printf("  Progress: %zu/%zu triangles\n", shown, state.numTriangles);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(dRng[b]));
        CUDA_CHECK(cudaEventDestroy(evGen[b]));
        CUDA_CHECK(cudaEventDestroy(evKij[b]));
    }
    CUDA_CHECK(cudaFree(dStates));
    CUDA_CHECK(cudaFree(dScratch));
    CUDA_CHECK(cudaFree(dTilePrefix));

    // 4. Count non-zero form factors and convert to propagation weights in place
    unsigned long long* dNonZero = nullptr;
    CUDA_CHECK(cudaMalloc(&dNonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dNonZero, 0, sizeof(unsigned long long)));
    if (localPairs > 0) {
        const int blocks = static_cast<int>((localPairs + 255) / 256);
        weightKernel<<<blocks, 256, 0, state.stream>>>(state.dKij, state.dAreas, nLocal, localPairs, dNonZero);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpyAsync(&localNonZero, dNonZero, sizeof(unsigned long long), cudaMemcpyDeviceToHost, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    CUDA_CHECK(cudaFree(dNonZero));
    MPI_Reduce(&localNonZero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int nLocal = state.nLocal;
    const int timeOff = T / 2;
    const int threadsPerBlock = 128;
    const int blocks = (nLocal + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> rowLocal(std::max(nLocal, 1));

    for (int t = 0; t < T; ++t) {
        float* radRow = state.radB.data() + static_cast<size_t>(t) * N;
        if (nLocal > 0) {
            if (state.tau8) {
                simStepKernel<uint8_t><<<blocks, threadsPerBlock, 0, state.stream>>>(
                    state.dKij, static_cast<const uint8_t*>(state.dTau), state.dRadB, t, N, nLocal,
                    state.rowStart, state.rho, static_cast<int>(state.sourceIndex), timeOff, state.dRowOut);
            } else {
                simStepKernel<uint16_t><<<blocks, threadsPerBlock, 0, state.stream>>>(
                    state.dKij, static_cast<const uint16_t*>(state.dTau), state.dRadB, t, N, nLocal,
                    state.rowStart, state.rho, static_cast<int>(state.sourceIndex), timeOff, state.dRowOut);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(rowLocal.data(), state.dRowOut, nLocal * sizeof(float),
                                       cudaMemcpyDeviceToHost, state.stream));
            CUDA_CHECK(cudaStreamSynchronize(state.stream));
        }
        // Exchange the new radiosity row
        MPI_Allgatherv(rowLocal.data(), nLocal, MPI_FLOAT, radRow, state.rowCounts.data(),
                       state.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(state.dRadB + static_cast<size_t>(t) * N, radRow, N * sizeof(float),
                                   cudaMemcpyHostToDevice, state.stream));

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int nLocal = state.nLocal;

    std::vector<float> distLocal(std::max(nLocal, 1), ZERO);
    if (nLocal > 0 && T > 0) {
        float* dRadBT = nullptr;
        float* dSrcCol = nullptr;
        float* dDist = nullptr;
        CUDA_CHECK(cudaMalloc(&dRadBT, static_cast<size_t>(nLocal) * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dSrcCol, T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDist, nLocal * sizeof(float)));

        std::vector<float> srcCol(T);
        for (int t = 0; t < T; ++t) srcCol[t] = state.radB[state.idxTN(t, state.sourceIndex)];
        CUDA_CHECK(cudaMemcpyAsync(dSrcCol, srcCol.data(), T * sizeof(float), cudaMemcpyHostToDevice, state.stream));

        dim3 tb(32, 8);
        dim3 tg((T + 31) / 32, (nLocal + 31) / 32);
        transposeRadKernel<<<tg, tb, 0, state.stream>>>(state.dRadB, N, T, nLocal, state.rowStart, dRadBT);
        CUDA_CHECK(cudaGetLastError());

        const int threads = std::min(1024, std::max(32, ((T + 31) / 32) * 32));
        distanceKernel<<<nLocal, threads, 0, state.stream>>>(dRadBT, dSrcCol, T, dDist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(distLocal.data(), dDist, nLocal * sizeof(float), cudaMemcpyDeviceToHost, state.stream));
        CUDA_CHECK(cudaStreamSynchronize(state.stream));
        CUDA_CHECK(cudaFree(dRadBT));
        CUDA_CHECK(cudaFree(dSrcCol));
        CUDA_CHECK(cudaFree(dDist));
    }
    MPI_Gatherv(distLocal.data(), nLocal, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
                state.rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (rank 0)
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
    #pragma omp parallel for schedule(static) reduction(+ : receivedEnergy)
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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

static void releaseDevice(SimulationState& state) {
    cudaFree(state.dTA); cudaFree(state.dTB); cudaFree(state.dTC);
    cudaFree(state.dNodes); cudaFree(state.dNodeTris); cudaFree(state.dAreas);
    cudaFree(state.dKij); cudaFree(state.dTau); cudaFree(state.dRadB); cudaFree(state.dRowOut);
    cudaFree(state.dPow2);
    if (state.stream) cudaStreamDestroy(state.stream);
    if (state.genStream) cudaStreamDestroy(state.genStream);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank, selected by the rank's position on its node
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount <= 0) {
            if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
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
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", nranks, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nranks = nranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    long localTotalTime = preDuration + simDuration + distDuration;
    long totalTime = 0;
    MPI_Reduce(&localTotalTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        // Total time
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
        fflush(stdout);
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    releaseDevice(state);
    MPI_Finalize();
    return rc;
}

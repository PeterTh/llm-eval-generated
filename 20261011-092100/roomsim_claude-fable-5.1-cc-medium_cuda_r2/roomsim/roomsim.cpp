/**
 * Room Response Simulation Benchmark (CUDA version)
 *
 * This is a CUDA-parallel implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
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
 *   - Form factors: 16 lanes trace the 16 rays of a pair (two pairs per warp),
 *     walking the octree with a parametric octant traversal; the per-ray terms
 *     are then accumulated in the reference order.  The sequential reference
 *     consumes a single std::mt19937 stream in pair order; to keep bit-identical
 *     semantics the very same stream is produced on the GPU: the host derives
 *     the MT19937 characteristic polynomial (Berlekamp-Massey) and jump
 *     polynomials x^(S*2^k) mod phi, a GPU Horner kernel computes the generator
 *     state at every chunk boundary, and a generation kernel expands every chunk
 *     in parallel (double buffered against the form factor kernel).
 *   - Time delays, wave propagation (one thread per receiver per timestep, packed
 *     weights/delays) and cross-correlation (one block per triangle) are plain
 *     data-parallel kernels that keep the reference accumulation order inside
 *     each sum.
 */

#include <algorithm>
#include <array>
#include <chrono>
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

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#define HD __host__ __device__

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;  // u,v for both endpoints of each ray

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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
// Octree for Spatial Acceleration (host build, flattened for the device)
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

// Flat device representation of an octree node (64 bytes).  Child bounding
// boxes are derived from the parent's bounds exactly like Octree::buildNode
// does, so a child only has to be loaded once it is actually visited.
struct __align__(16) OctNode {
    float minx, miny, minz;  // node bounds
    float maxx, maxy, maxz;
    int triStart;            // offset into the leaf triangle list (3 float4 per entry)
    int triCount;            // > 0 for leaves (leaves always hold >= 1 triangle)
    int child[8];            // child node indices, -1 if absent
};

static int flattenOctree(const Octree& node, const std::vector<Triangle>& tris,
                         std::vector<OctNode>& nodes, std::vector<float4>& leafGeom) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();
    OctNode n;
    n.minx = node.minBound.x; n.miny = node.minBound.y; n.minz = node.minBound.z;
    n.maxx = node.maxBound.x; n.maxy = node.maxBound.y; n.maxz = node.maxBound.z;
    n.triStart = static_cast<int>(leafGeom.size() / 3);
    n.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) {
        const Triangle& t = tris[idx];
        float idxBits;
        int ii = static_cast<int>(idx);
        memcpy(&idxBits, &ii, sizeof(float));
        leafGeom.push_back(make_float4(t.a.x, t.a.y, t.a.z, idxBits));
        leafGeom.push_back(make_float4(t.b.x, t.b.y, t.b.z, 0.0f));
        leafGeom.push_back(make_float4(t.c.x, t.c.y, t.c.z, 0.0f));
    }
    for (int i = 0; i < 8; ++i) {
        n.child[i] = node.children[i] ? flattenOctree(*node.children[i], tris, nodes, leafGeom) : -1;
    }
    nodes[myIdx] = n;
    return myIdx;
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
// Random Number Generation: MT19937 reproduced on the GPU
//
// The reference implementation draws std::uniform_real_distribution<float>
// from a single std::mt19937(42) in pair order.  To reproduce the exact same
// stream in parallel, the generator is treated as the GF(2)-linear recurrence
// it is: the characteristic polynomial phi (degree 19937) is recovered with
// Berlekamp-Massey, jump polynomials x^(S*2^k) mod phi are obtained by repeated
// squaring, and a Horner evaluation over the recurrence (on the GPU) yields the
// state at any chunk boundary.  Each chunk is then expanded independently.
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DEG = 19937;             // degree of the characteristic polynomial
constexpr int MT_PW = (MT_DEG + 63) / 64; // 312 64-bit words hold a polynomial of degree < 19937
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;

// One step of the MT19937 recurrence: x_{k+624} from x_k, x_{k+1}, x_{k+397}
HD inline uint32_t mtNext(uint32_t xk, uint32_t xk1, uint32_t xk397) {
    uint32_t y = (xk & MT_UPPER) | (xk1 & MT_LOWER);
    return xk397 ^ (y >> 1) ^ ((xk1 & 1u) ? MT_MATRIX_A : 0u);
}

HD inline uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// std::uniform_real_distribution<float>(0,1) over mt19937 == generate_canonical<float,24>:
// one 32-bit draw, float(x) / 2^32, clamped below 1.
__device__ inline float mtToUnitFloat(uint32_t x) {
    float f = __uint2float_rn(x) * 2.3283064365386962890625e-10f;  // exact power-of-two scale
    if (f >= 1.0f) f = 0.99999994f;  // nextafter(1.0f, 0.0f)
    return f;
}

// Host-side MT19937 state (seeded exactly like std::mt19937)
struct MTState {
    uint32_t x[MT_N];
    explicit MTState(uint32_t seed) {
        x[0] = seed;
        for (int i = 1; i < MT_N; ++i) {
            x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }
    void twist() {
        for (int k = 0; k < MT_N; ++k) {
            x[k] = mtNext(x[k], x[(k + 1) % MT_N], x[(k + MT_M) % MT_N]);
        }
    }
};

// ----- GF(2)[x] polynomial helpers (64-bit word packed, LSB = x^0) -----

// Berlekamp-Massey over GF(2).  Returns the connection polynomial C (c_0 = 1)
// such that b_n = sum_{i=1..L} c_i b_{n-i}, and sets L.
static std::vector<uint64_t> berlekampMassey(const std::vector<uint8_t>& bits, int NB, int& Lout) {
    const int W = NB / 64 + 16;
    std::vector<uint64_t> R(W, 0);  // reversed sequence: bit j = b_{NB-1-j}
    for (int n = 0; n < NB; ++n) {
        if (bits[n]) {
            int j = NB - 1 - n;
            R[j >> 6] |= 1ull << (j & 63);
        }
    }
    std::vector<uint64_t> C(W, 0), B(W, 0), T;
    C[0] = 1; B[0] = 1;
    int L = 0, m = 1;
    int cWords = 0, bWords = 0;  // highest possibly non-zero word index of C / B

    auto shiftXor = [&](std::vector<uint64_t>& dst, const std::vector<uint64_t>& src, int srcWords, int sh) {
        int q = sh >> 6, b = sh & 63;
        for (int k = std::min(W - 1 - q, srcWords); k >= 0; --k) {
            uint64_t v = src[k] << b;
            if (b && k > 0) v |= src[k - 1] >> (64 - b);
            dst[k + q] ^= v;
        }
    };

    for (int n = 0; n < NB; ++n) {
        // discrepancy d = sum_{i=0..L} c_i b_{n-i}; b_{n-i} = R[NB-1-n+i]
        int o = NB - 1 - n;
        int q = o >> 6, bsh = o & 63;
        int nw = (L >> 6) + 1;
        uint64_t acc = 0;
        for (int k = 0; k < nw; ++k) {
            uint64_t w = R[q + k] >> bsh;
            if (bsh) w |= R[q + k + 1] << (64 - bsh);
            acc ^= w & C[k];
        }
        int d = __builtin_parityll(acc);
        if (d) {
            if (2 * L <= n) {
                T = C;
                int tWords = cWords;
                shiftXor(C, B, bWords, m);
                cWords = std::min(W - 1, std::max(cWords, bWords + (m >> 6) + 1));
                L = n + 1 - L;
                B = T;
                bWords = tWords;
                m = 1;
            } else {
                shiftXor(C, B, bWords, m);
                cWords = std::min(W - 1, std::max(cWords, bWords + (m >> 6) + 1));
                m++;
            }
        } else {
            m++;
        }
    }
    Lout = L;
    return C;
}

// Spread the 32 bits of x to even bit positions of a 64-bit word
static inline uint64_t spreadBits(uint64_t x) {
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2))  & 0x3333333333333333ull;
    x = (x | (x << 1))  & 0x5555555555555555ull;
    return x;
}

struct PolyMod {
    static constexpr int TW = MT_PW + 1;  // 313 words per shifted copy of phi
    std::vector<uint64_t> phi;            // degree MT_DEG, MT_PW words
    std::vector<uint64_t> tab;            // tab[b] = phi << b, b = 0..63

    explicit PolyMod(const std::vector<uint64_t>& p) : phi(p), tab(64 * TW, 0) {
        for (int b = 0; b < 64; ++b) {
            uint64_t* t = &tab[b * TW];
            for (int k = 0; k < MT_PW; ++k) {
                t[k] |= phi[k] << b;
                if (b) t[k + 1] |= phi[k] >> (64 - b);
            }
        }
    }

    // Reduce a polynomial of degree < 2*MT_DEG (2*MT_PW words) modulo phi, in place
    void reduce(uint64_t* a) const {
        for (int w = 2 * MT_PW - 1; w >= MT_PW - 1; --w) {
            for (;;) {
                uint64_t x = a[w];
                if (w == MT_PW - 1) x &= ~((1ull << (MT_DEG & 63)) - 1);  // only bits of degree >= MT_DEG
                if (!x) break;
                int bit = 63 - __builtin_clzll(x);
                int shift = w * 64 + bit - MT_DEG;
                int q = shift >> 6, b = shift & 63;
                const uint64_t* t = &tab[b * TW];
                uint64_t* dst = a + q;
                for (int k = 0; k < TW; ++k) dst[k] ^= t[k];
            }
        }
    }

    // p <- p^2 mod phi  (p has MT_PW words)
    void square(std::vector<uint64_t>& p) const {
        std::vector<uint64_t> sq(2 * MT_PW + 8, 0);
        for (int k = 0; k < MT_PW; ++k) {
            sq[2 * k] = spreadBits(p[k] & 0xFFFFFFFFull);
            sq[2 * k + 1] = spreadBits(p[k] >> 32);
        }
        reduce(sq.data());
        for (int k = 0; k < MT_PW; ++k) p[k] = sq[k];
    }
};

// Compute the MT19937 characteristic polynomial from the output stream of a
// freshly seeded generator (any seed works; the minimal polynomial is the same).
static std::vector<uint64_t> computeMTCharPoly() {
    const int NB = 2 * MT_DEG + 64;
    const int NEXTRA = 512;
    std::vector<uint8_t> bits(NB + NEXTRA);
    std::mt19937 gen(12345u);
    for (int n = 0; n < NB + NEXTRA; ++n) bits[n] = static_cast<uint8_t>(gen() & 1u);

    int L = 0;
    std::vector<uint64_t> C = berlekampMassey(bits, NB, L);
    if (L != MT_DEG) {
        fprintf(stderr, "MT19937 minimal polynomial has unexpected degree %d\n", L);
        exit(1);
    }
    // Sanity check: the recurrence must predict the extra bits
    for (int n = NB; n < NB + NEXTRA; ++n) {
        int d = 0;
        for (int i = 0; i <= L; ++i) {
            if ((C[i >> 6] >> (i & 63)) & 1ull) d ^= bits[n - i];
        }
        if (d) {
            fprintf(stderr, "MT19937 minimal polynomial verification failed\n");
            exit(1);
        }
    }
    // Characteristic polynomial phi(x) = x^L C(1/x): phi_i = c_{L-i}
    std::vector<uint64_t> phi(MT_PW, 0);
    for (int i = 0; i <= L; ++i) {
        int src = L - i;
        if ((C[src >> 6] >> (src & 63)) & 1ull) phi[i >> 6] |= 1ull << (i & 63);
    }
    return phi;
}

// ----- GPU kernels for the generator -----

// Horner evaluation of g(M) applied to a state vector:  Wout[c] = g(M) W[c - stride]
// Processed 16 polynomial coefficients at a time:
//   acc <- M^16 acc + sum_{b<16} g_{16m+b} (M^b W)      for m = top..0
// where M^b W (b < 16) are precomputed in shared memory and M^16 shifts the
// 624-word window by 16 (the 16 new words only depend on the old window).
constexpr int JUMP_RADIX = 16;

constexpr int JUMP_THREADS = 640;  // >= MT_N: one ring slot per thread

__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(const uint32_t* __restrict__ g, const uint32_t* W, uint32_t* Wout,
             int cStart, int stride) {
    __shared__ uint32_t gsh[MT_N];
    __shared__ uint32_t V[JUMP_RADIX][MT_N];
    __shared__ uint32_t ringA[MT_N];
    __shared__ uint32_t ringB[MT_N];

    const int c = cStart + blockIdx.x;
    const int src = c - stride;
    const int tid = threadIdx.x;

    for (int k = tid; k < MT_N; k += blockDim.x) {
        gsh[k] = g[k];
        V[0][k] = W[static_cast<size_t>(src) * MT_N + k];
        ringA[k] = 0;
    }
    __syncthreads();
    for (int b = 1; b < JUMP_RADIX; ++b) {
        for (int k = tid; k < MT_N; k += blockDim.x) {
            V[b][k] = (k + 1 < MT_N) ? V[b - 1][k + 1]
                                     : mtNext(V[b - 1][0], V[b - 1][1], V[b - 1][MT_M]);
        }
        __syncthreads();
    }

    auto digit = [&](int m) -> uint32_t {
        uint32_t w = gsh[m >> 1];
        return (m & 1) ? (w >> 16) : (w & 0xffffu);
    };
    int top = (MT_DEG + JUMP_RADIX - 1) / JUMP_RADIX - 1;
    while (top > 0 && digit(top) == 0u) --top;

    uint32_t* A = ringA;
    uint32_t* B = ringB;
    int base = 0;  // window element k lives at slot (base + k) % 624
    for (int m = top; m >= 0; --m) {
        const uint32_t h = digit(m);
        int baseNew = base + JUMP_RADIX; if (baseNew >= MT_N) baseNew -= MT_N;
        for (int slot = tid; slot < MT_N; slot += blockDim.x) {
            int kk = slot - base; if (kk < 0) kk += MT_N;          // old element index
            uint32_t v;
            if (kk < JUMP_RADIX) {
                int s1 = slot + 1; if (s1 >= MT_N) s1 -= MT_N;
                int s397 = slot + MT_M; if (s397 >= MT_N) s397 -= MT_N;
                v = mtNext(A[slot], A[s1], A[s397]);
            } else {
                v = A[slot];
            }
            int kn = kk - JUMP_RADIX; if (kn < 0) kn += MT_N;     // new element index
            #pragma unroll
            for (int b = 0; b < JUMP_RADIX; ++b) {
                if ((h >> b) & 1u) v ^= V[b][kn];
            }
            B[slot] = v;
        }
        base = baseNew;
        __syncthreads();
        uint32_t* tmp = A; A = B; B = tmp;
    }
    for (int k = tid; k < MT_N; k += blockDim.x) {
        int slot = base + k; if (slot >= MT_N) slot -= MT_N;
        Wout[static_cast<size_t>(c) * MT_N + k] = A[slot];
    }
}

// Expand chunk c (S outputs, starting at state W[c]) into the batch buffer.
__global__ void __launch_bounds__(256)
mtGenKernel(const uint32_t* __restrict__ W, uint32_t* __restrict__ out,
            int cFirst, unsigned long long S, unsigned long long batchStart) {
    __shared__ uint32_t sA[MT_N];
    __shared__ uint32_t sB[MT_N];

    const int c = cFirst + blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t* A = sA;
    uint32_t* B = sB;

    for (int k = tid; k < MT_N; k += blockDim.x) A[k] = W[static_cast<size_t>(c) * MT_N + k];
    __syncthreads();

    uint32_t* dst = out + (static_cast<unsigned long long>(c) * S - batchStart);
    unsigned long long produced = 0;
    for (;;) {
        // Emit the tempered current window
        for (int k = tid; k < MT_N; k += blockDim.x) {
            if (produced + k < S) dst[produced + k] = mtTemper(A[k]);
        }
        produced += MT_N;
        if (produced >= S) break;

        // Twist A -> B in three dependency phases
        if (tid < MT_N - MT_M) {                       // k in [0, 227)
            int k = tid;
            B[k] = mtNext(A[k], A[k + 1], A[k + MT_M]);
        }
        __syncthreads();
        if (tid < MT_N - MT_M) {                       // k in [227, 454)
            int k = tid + (MT_N - MT_M);
            B[k] = mtNext(A[k], A[k + 1], B[k - (MT_N - MT_M)]);
        }
        __syncthreads();
        if (tid < MT_N - 2 * (MT_N - MT_M)) {          // k in [454, 624)
            int k = tid + 2 * (MT_N - MT_M);
            uint32_t xk1 = (k + 1 < MT_N) ? A[k + 1] : B[0];
            B[k] = mtNext(A[k], xk1, B[k - (MT_N - MT_M)]);
        }
        __syncthreads();
        uint32_t* tmp = A; A = B; B = tmp;
    }
}

// ============================================================================
// Device geometry helpers
// ============================================================================

// Generate a random point inside a triangle using barycentric coordinates
__device__ inline Vec3 randomPointInTriangle(const Vec3& a, const Vec3& b, const Vec3& c, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return a + ab * u + ac * v;
}

// Ray-Triangle Intersection (Möller-Trumbore algorithm)
__device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

__device__ inline Vec3 ld3(const float4* p, int i) {
    float4 v = __ldg(p + i);
    return Vec3(v.x, v.y, v.z);
}

// Check if a ray between two triangles is blocked by any other triangle.
//
// Octree-accelerated: the segment is walked through the octants it actually
// crosses (parametric octree traversal), which visits every leaf whose box the
// segment touches (with a small tolerance) and therefore finds exactly the
// same intersections as the reference's box-test recursion; the per-triangle
// intersection test is the reference's.
constexpr int TRAV_STACK = 64;
constexpr float TRAV_TOL = 1e-5f;   // parametric tolerance around octant boundaries

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const OctNode* __restrict__ nodes, const float4* __restrict__ leafGeom,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Segment parametrization p(t) = from + t*dir, t in [0,1]; inverse direction
    // (zero components get a huge finite value so no NaNs appear).
    const float invx = dir.x != 0.0f ? 1.0f / dir.x : 1e30f;
    const float invy = dir.y != 0.0f ? 1.0f / dir.y : 1e30f;
    const float invz = dir.z != 0.0f ? 1.0f / dir.z : 1e30f;

    int stack[TRAV_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root is visited unconditionally, like the reference

    while (sp > 0) {
        const OctNode n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            const float4* g = leafGeom + static_cast<size_t>(n.triStart) * 3;
            for (int t = 0; t < n.triCount; ++t) {
                float4 a = __ldg(g + 3 * t);
                int idx = __float_as_int(a.w);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                float4 b = __ldg(g + 3 * t + 1);
                float4 c = __ldg(g + 3 * t + 2);
                val_t dist = rayTriangleIntersect(from, dirNorm, Vec3(a.x, a.y, a.z),
                                                  Vec3(b.x, b.y, b.z), Vec3(c.x, c.y, c.z));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
            continue;
        }

        // Parametric interval of the segment inside this (padded) node box
        float ta = 0.0f, tb = 1.0f;
        {
            float u = (n.minx - EPSILON - from.x) * invx, v = (n.maxx + EPSILON - from.x) * invx;
            ta = fmaxf(ta, fminf(u, v)); tb = fminf(tb, fmaxf(u, v));
            u = (n.miny - EPSILON - from.y) * invy; v = (n.maxy + EPSILON - from.y) * invy;
            ta = fmaxf(ta, fminf(u, v)); tb = fminf(tb, fmaxf(u, v));
            u = (n.minz - EPSILON - from.z) * invz; v = (n.maxz + EPSILON - from.z) * invz;
            ta = fmaxf(ta, fminf(u, v)); tb = fminf(tb, fmaxf(u, v));
        }
        if (ta > tb) continue;

        // Midplane crossings (node center exactly as Octree::buildNode computes it).
        // For each axis: crossing parameter tm (3.0 = never) and side at t = ta.
        auto axisInfo = [&](float mn, float mx, float d, float inv, float o, float& tm, bool& hi) {
            const float ctr = (mn + mx) * 0.5f;
            if (d > 0.0f)      { tm = (ctr - o) * inv; hi = ta > tm; }
            else if (d < 0.0f) { tm = (ctr - o) * inv; hi = ta < tm; }
            else {  // parallel to the midplane: fixed side, or both if (nearly) on it
                hi = o > ctr;
                tm = (fabsf(o - ctr) <= EPSILON) ? ta : 3.0f;
            }
            if (!(tm > ta - TRAV_TOL && tm < tb + TRAV_TOL)) tm = 3.0f;
        };
        float tmx, tmy, tmz;
        bool hx, hy, hz;
        axisInfo(n.minx, n.maxx, dir.x, invx, from.x, tmx, hx);
        axisInfo(n.miny, n.maxy, dir.y, invy, from.y, tmy, hy);
        axisInfo(n.minz, n.maxz, dir.z, invz, from.z, tmz, hz);
        int oct = (hx ? 1 : 0) | (hy ? 2 : 0) | (hz ? 4 : 0);

        // Sort the (up to three) crossings; non-crossings (3.0) sort last.
        float c0 = tmx, c1 = tmy, c2 = tmz;
        int a0 = 1, a1 = 2, a2 = 4;
        if (c1 < c0) { float t = c0; c0 = c1; c1 = t; int x = a0; a0 = a1; a1 = x; }
        if (c2 < c1) { float t = c1; c1 = c2; c2 = t; int x = a1; a1 = a2; a2 = x; }
        if (c1 < c0) { float t = c0; c0 = c1; c1 = t; int x = a0; a0 = a1; a1 = x; }

        // Walk the crossed octants in order (the segment leaves the node at tb)
        int ch = n.child[oct];
        if (ch >= 0) stack[sp++] = ch;
        if (c0 < 3.0f) {
            oct ^= a0; ch = n.child[oct];
            if (ch >= 0) stack[sp++] = ch;
            if (c1 < 3.0f) {
                oct ^= a1; ch = n.child[oct];
                if (ch >= 0) stack[sp++] = ch;
                if (c2 < 3.0f) {
                    oct ^= a2; ch = n.child[oct];
                    if (ch >= 0) stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ inline val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Form Factor Kernel (Kij)
//
// One block handles a tile of FF_TILE consecutive j for a single row i.  The
// position of each pair in the global random stream is tileStart[tile] plus the
// pair's rank among the non-culled pairs of the tile (block-wide prefix sum).
// The 16 rays of a pair are traced by 16 lanes of a warp (two pairs per warp),
// and the per-ray terms are then accumulated in the reference order.  Only
// pairs whose draws fall inside the currently resident batch are computed.
// ============================================================================

constexpr int FF_TILE = 256;
constexpr int FF_PAIRS_PER_PASS = FF_TILE / NUM_RAYS;  // 16 pairs per pass

__global__ void __launch_bounds__(FF_TILE)
formFactorKernel(const float4* __restrict__ triA, const float4* __restrict__ triB,
                 const float4* __restrict__ triC, const float4* __restrict__ triN,
                 const OctNode* __restrict__ nodes, const float4* __restrict__ leafGeom,
                 const uint32_t* __restrict__ flagBits, int N, int wordsPerRow, int tilesPerRow,
                 const unsigned long long* __restrict__ tileStart, int tileFirst,
                 const uint32_t* __restrict__ rnd, unsigned long long batchStart,
                 unsigned long long batchEnd, float* __restrict__ kij) {
    __shared__ int warpTot[FF_TILE / 32];
    __shared__ int sRank[FF_TILE];
    __shared__ unsigned char sActive[FF_TILE];

    const int tid = threadIdx.x;
    const int tile = tileFirst + blockIdx.x;
    const int i = tile / tilesPerRow;
    const int j0 = (tile - i * tilesPerRow) * FF_TILE;
    {
        const int j = j0 + tid;
        const bool active = (j < N) &&
            ((flagBits[static_cast<size_t>(i) * wordsPerRow + (j >> 5)] >> (j & 31)) & 1u);
        // Block-wide exclusive scan of 'active'
        const int lane = tid & 31;
        const int warp = tid >> 5;
        const unsigned ballot = __ballot_sync(0xffffffffu, active);
        if (lane == 0) warpTot[warp] = __popc(ballot);
        __syncthreads();
        int rank = __popc(ballot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w) rank += warpTot[w];
        sRank[tid] = rank;
        sActive[tid] = active ? 1 : 0;
    }
    __syncthreads();

    const unsigned long long tileBase = tileStart[tile];
    const Vec3 aI = ld3(triA, i), bI = ld3(triB, i), cI = ld3(triC, i), nI = ld3(triN, i);
    const int ray = tid & (NUM_RAYS - 1);

    for (int pass = 0; pass < FF_TILE / FF_PAIRS_PER_PASS; ++pass) {
        const int p = pass * FF_PAIRS_PER_PASS + (tid / NUM_RAYS);
        const int j = j0 + p;
        bool doPair = sActive[p] != 0;
        unsigned long long off = 0;
        if (doPair) {
            off = (tileBase + sRank[p]) * static_cast<unsigned long long>(RANDS_PER_PAIR);
            doPair = (off >= batchStart) && (off < batchEnd);
        }

        val_t term = ZERO;
        if (doPair) {
            const uint4 q = *reinterpret_cast<const uint4*>(rnd + (off - batchStart) + 4 * ray);
            const Vec3 aJ = ld3(triA, j), bJ = ld3(triB, j), cJ = ld3(triC, j), nJ = ld3(triN, j);
            Vec3 pI = randomPointInTriangle(aI, bI, cI, mtToUnitFloat(q.x), mtToUnitFloat(q.y));
            Vec3 pJ = randomPointInTriangle(aJ, bJ, cJ, mtToUnitFloat(q.z), mtToUnitFloat(q.w));

            if (!isRayBlocked(pI, pJ, nodes, leafGeom, i, j)) {
                Vec3 v = pJ - pI;
                val_t distSqr = v.squaredNorm();
                if (distSqr >= EPSILON) {
                    val_t cosPhiI = cosPhiDev(v, nI);
                    val_t cosPhiJ = cosPhiDev(-v, nJ);
                    if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                        term = (cosPhiI * cosPhiJ) / (PI * distSqr);
                    }
                }
            }
        }

        // Accumulate the 16 ray terms in reference order (skipped rays add +0)
        const int laneBase = (tid & 31) & ~(NUM_RAYS - 1);
        val_t k = ZERO;
        #pragma unroll
        for (int r = 0; r < NUM_RAYS; ++r) {
            k += __shfl_sync(0xffffffffu, term, laneBase + r);
        }
        if (doPair && ray == 0) kij[static_cast<size_t>(i) * N + j] = k * INV_NUM_RAYS;
    }
}

// ============================================================================
// Cull-flag kernel
//
// Pair (i, j) consumes random draws iff i != j and dot(nI, nJ) <= 0.99.  The
// flags decide the position of every pair in the random stream, so they must
// match the reference's host evaluation bit for bit.  Pairs whose dot product
// is within CULL_MARGIN of the threshold (far wider than any rounding
// difference) are collected and re-evaluated on the host with the reference
// expression; all others are unambiguous.
// ============================================================================

constexpr float CULL_MARGIN = 1e-4f;

__global__ void cullFlagKernel(const float4* __restrict__ triN, int N, int wordsPerRow,
                               uint32_t* __restrict__ flagBits,
                               unsigned long long* __restrict__ ambiguous, int ambCap,
                               int* __restrict__ ambCount) {
    const int i = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = false;
    if (j < N && j != i) {
        const Vec3 nI = ld3(triN, i);
        const Vec3 nJ = ld3(triN, j);
        const float d = nI.dot(nJ);
        active = !(d > 0.99f);
        if (fabsf(d - 0.99f) < CULL_MARGIN) {
            int slot = atomicAdd(ambCount, 1);
            if (slot < ambCap) ambiguous[slot] = (static_cast<unsigned long long>(i) << 32) | static_cast<unsigned>(j);
        }
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, active);
    if ((threadIdx.x & 31) == 0 && j < N) {
        flagBits[static_cast<size_t>(i) * wordsPerRow + (j >> 5)] = ballot;
    }
}

// ============================================================================
// Tau (time delay) Kernel
// ============================================================================

__global__ void tauKernel(const float4* __restrict__ triA, const float4* __restrict__ triB,
                          const float4* __restrict__ triC, int N, int* __restrict__ tau) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= static_cast<size_t>(N) * N) return;
    const int i = static_cast<int>(p / N);
    const int j = static_cast<int>(p - static_cast<size_t>(i) * N);
    if (i == j) { tau[p] = 0; return; }
    Vec3 ci = (ld3(triA, i) + ld3(triB, i) + ld3(triC, i)) / 3.0f;
    Vec3 cj = (ld3(triA, j) + ld3(triB, j) + ld3(triC, j)) / 3.0f;
    val_t dist = (ci - cj).norm();
    tau[p] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Simulation data packing
//
// The simulation reads, for every receiver i and every source j, the weight
// min(kij[i][j]*area[j], 1) and the delay tau[i][j].  Both are stored in a
// "j-quad" layout: four consecutive j per receiver packed into one float4 /
// one 4-byte delay group, i.e. index = (j/4) * (4N) + 4i + (j%4).  Reads are
// then coalesced across receivers and need 4x fewer load instructions.  Delays
// use 8 bits when the largest delay allows it (32 bits otherwise).
// ============================================================================

__global__ void maxTauKernel(const int* __restrict__ tau, size_t total, int* __restrict__ maxTau) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int m = 0;
    for (; p < total; p += static_cast<size_t>(gridDim.x) * blockDim.x) m = max(m, __ldg(tau + p));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(maxTau, m);
}

template <typename TauT>
__global__ void packSimKernel(const float* __restrict__ kij, const float* __restrict__ areas,
                              const int* __restrict__ tau, int N, int Nq,
                              float4* __restrict__ kijQ, TauT* __restrict__ tauQ) {
    const int i = blockIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= Nq) return;
    float w[4];
    TauT d[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int j = 4 * q + c;
        if (j < N) {
            w[c] = fminf(kij[static_cast<size_t>(i) * N + j] * areas[j], ONE);
            d[c] = static_cast<TauT>(tau[static_cast<size_t>(i) * N + j]);
        } else {  // padding: never arrives, zero weight
            w[c] = ZERO;
            d[c] = static_cast<TauT>(sizeof(TauT) == 1 ? 255 : 0x7fffffff);
        }
    }
    const size_t base = (static_cast<size_t>(q) * N + i) * 4;
    kijQ[base / 4] = make_float4(w[0], w[1], w[2], w[3]);
    tauQ[base + 0] = d[0]; tauQ[base + 1] = d[1]; tauQ[base + 2] = d[2]; tauQ[base + 3] = d[3];
}

// ============================================================================
// Simulation Kernel (one timestep; one thread per receiving triangle)
//
// The reference skips pairs whose wave has not arrived yet, whose form factor
// is zero or whose source radiosity is zero.  All of those contribute exactly
// +0 to a non-negative sum, so the loop is written branch-free (contribution
// forced to zero) which lets the loads of many j pipeline.  The accumulation
// order over j is the reference's.
// ============================================================================

template <typename TauT>
__global__ void __launch_bounds__(64)
simulationStepKernel(int N, int Nq, int T, int t,
                     const float4* __restrict__ kijQ,  // j-quad layout, see packSimKernel
                     const TauT* __restrict__ tauQ,    // same layout
                     const float* __restrict__ rho,
                     const float* __restrict__ radE,
                     float* __restrict__ radB,         // [T][N] (reference layout)
                     float* __restrict__ radBT) {      // [N][T] (time-major copy for reads)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    val_t sumB = ZERO;

    #pragma unroll 4
    for (int q = 0; q < Nq; ++q) {
        const size_t base = (static_cast<size_t>(q) * N + i) * 4;
        const float4 w = __ldg(kijQ + base / 4);
        int d[4];
        if (sizeof(TauT) == 1) {
            const uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(tauQ + base));
            d[0] = packed & 0xffu; d[1] = (packed >> 8) & 0xffu; d[2] = (packed >> 16) & 0xffu; d[3] = packed >> 24;
        } else {
            const int4 packed = __ldg(reinterpret_cast<const int4*>(tauQ + base));
            d[0] = packed.x; d[1] = packed.y; d[2] = packed.z; d[3] = packed.w;
        }
        const float wv[4] = {w.x, w.y, w.z, w.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int j = 4 * q + c;
            // Contribution only once the wave from j has arrived (never from i itself)
            const bool valid = (j != i) && (t >= d[c]);
            const int srcTime = valid ? (t - d[c]) : 0;
            const int jj = valid ? j : 0;
            val_t radJ = __ldg(radBT + static_cast<size_t>(jj) * T + srcTime);
            radJ = valid ? radJ : ZERO;
            // Accumulate contribution: form factor * area * source radiosity
            sumB += wv[c] * radJ;
        }
    }

    // Update radiosity: reflection + emission
    const val_t v = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
    radB[static_cast<size_t>(t) * N + i] = v;
    radBT[static_cast<size_t>(i) * T + t] = v;
}

// ============================================================================
// Distance Kernel (cross-correlation; one block per triangle)
// ============================================================================

constexpr int DIST_BLOCK = 256;

__global__ void __launch_bounds__(DIST_BLOCK)
distanceKernel(int T, const float* __restrict__ radBT, int sourceIndex,
               float* __restrict__ distances) {
    __shared__ float sCorr[DIST_BLOCK];
    __shared__ int sT[DIST_BLOCK];

    const float* colB = radBT + static_cast<size_t>(blockIdx.x) * T;
    const float* colS = radBT + static_cast<size_t>(sourceIndex) * T;
    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (int t = threadIdx.x; t < T; t += DIST_BLOCK) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            val_t pB = __ldg(colB + tt);
            val_t pS = __ldg(colS + (tt - t));
            sum += pS * pB;
        }
        if (sum > maxCorr) {  // strict: earliest t wins among equal maxima
            maxCorr = sum;
            bestT = t;
        }
    }

    sCorr[threadIdx.x] = maxCorr;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = DIST_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float oc = sCorr[threadIdx.x + s];
            int ot = sT[threadIdx.x + s];
            float mc = sCorr[threadIdx.x];
            int mt = sT[threadIdx.x];
            if (oc > mc || (oc == mc && ot < mt)) {
                sCorr[threadIdx.x] = oc;
                sT[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[blockIdx.x] = WAVE_SPEED * static_cast<val_t>(sT[0]);
}

// Trivial kernel used to absorb module loading / clock ramp-up outside the timed regions
__global__ void warmupKernel(int* p) { if (threadIdx.x == 0) *p = 1; }

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    std::vector<uint64_t> mtCharPoly;  // MT19937 characteristic polynomial (input independent)

    size_t sourceIndex;

    // Device-side data
    float4* dTriA = nullptr;
    float4* dTriB = nullptr;
    float4* dTriC = nullptr;
    float4* dTriN = nullptr;
    OctNode* dNodes = nullptr;
    float4* dLeafGeom = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dKij = nullptr;          // row-major
    float4* dKijQ = nullptr;        // packed simulation weights (see packSimKernel)
    void* dTauQ = nullptr;          // packed simulation delays (uint8 or int)
    bool tau8 = false;              // delays fit in 8 bits
    int* dTau = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;
    float* dRadBT = nullptr;        // time-major copy of radB
    float* dDistances = nullptr;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
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
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Input-independent constant of the random number generator
    state.mtCharPoly = computeMTCharPoly();

    // ---- Upload everything to the device ----
    const size_t N = state.numTriangles;
    CUDA_CHECK(cudaFree(nullptr));  // create context outside the timed regions

    std::vector<float4> hA(N), hB(N), hC(N), hN(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        hA[i] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        hB[i] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        hC[i] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        hN[i] = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
    }
    std::vector<OctNode> nodes;
    std::vector<float4> leafGeom;
    flattenOctree(state.octree, state.triangles, nodes, leafGeom);

    CUDA_CHECK(cudaMalloc(&state.dTriA, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriB, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriC, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriN, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, nodes.size() * sizeof(OctNode)));
    CUDA_CHECK(cudaMalloc(&state.dLeafGeom, std::max<size_t>(3, leafGeom.size()) * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dKij, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadBT, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.dTriA, hA.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriB, hB.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriC, hC.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriN, hN.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, nodes.data(), nodes.size() * sizeof(OctNode), cudaMemcpyHostToDevice));
    if (!leafGeom.empty())
        CUDA_CHECK(cudaMemcpy(state.dLeafGeom, leafGeom.data(), leafGeom.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dKij, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), timesteps * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dRadBT, 0, timesteps * N * sizeof(float)));

    // Warm up the device (module load, clocks) so the timed phases measure the work
    {
        int* dTmp = nullptr;
        CUDA_CHECK(cudaMalloc(&dTmp, sizeof(int)));
        warmupKernel<<<1, 32>>>(dTmp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dTmp));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

static inline int ceilLog2(unsigned long long v) {
    int k = 0;
    while ((1ull << k) < v) ++k;
    return k;
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int tilesPerRow = (N + FF_TILE - 1) / FF_TILE;
    const int wordsPerRow = (N + 31) / 32;
    const size_t numTiles = static_cast<size_t>(N) * tilesPerRow;

    // ---- Cull flags (GPU + host fix-up of near-threshold pairs) and tile offsets ----
    // A pair consumes RANDS_PER_PAIR draws iff i != j and the normals are not
    // nearly parallel; the stream position of every pair follows from that.
    std::vector<uint32_t> flagBits(static_cast<size_t>(N) * wordsPerRow, 0u);
    std::vector<unsigned long long> tileStart(numTiles + 1, 0ull);
    {
        uint32_t* dBits = nullptr;
        unsigned long long* dAmb = nullptr;
        int* dAmbCount = nullptr;
        const int ambCap = static_cast<int>(std::min<size_t>(static_cast<size_t>(N) * N, static_cast<size_t>(N) * N / 64 + 4096));
        CUDA_CHECK(cudaMalloc(&dBits, flagBits.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dAmb, static_cast<size_t>(ambCap) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dAmbCount, sizeof(int)));
        CUDA_CHECK(cudaMemset(dAmbCount, 0, sizeof(int)));
        dim3 fg((N + 255) / 256, N);
        cullFlagKernel<<<fg, 256>>>(state.dTriN, N, wordsPerRow, dBits, dAmb, ambCap, dAmbCount);
        CUDA_CHECK(cudaGetLastError());
        int ambCount = 0;
        CUDA_CHECK(cudaMemcpy(&ambCount, dAmbCount, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(flagBits.data(), dBits, flagBits.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (ambCount > ambCap) {
            // Pathological geometry: fall back to evaluating every pair on the host
            for (int i = 0; i < N; ++i) {
                const Vec3 nI = state.triangles[i].normal();
                uint32_t* row = &flagBits[static_cast<size_t>(i) * wordsPerRow];
                for (int j = 0; j < N; ++j) {
                    bool act = (i != j) && !(nI.dot(state.triangles[j].normal()) > 0.99f);
                    if (act) row[j >> 5] |= 1u << (j & 31); else row[j >> 5] &= ~(1u << (j & 31));
                }
            }
        } else if (ambCount > 0) {
            std::vector<unsigned long long> amb(ambCount);
            CUDA_CHECK(cudaMemcpy(amb.data(), dAmb, amb.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (unsigned long long e : amb) {
                const int i = static_cast<int>(e >> 32);
                const int j = static_cast<int>(e & 0xffffffffu);
                // Reference expression, evaluated by the host compiler
                const bool act = !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f);
                uint32_t& w = flagBits[static_cast<size_t>(i) * wordsPerRow + (j >> 5)];
                if (act) w |= 1u << (j & 31); else w &= ~(1u << (j & 31));
            }
        }
        CUDA_CHECK(cudaFree(dBits));
        CUDA_CHECK(cudaFree(dAmb));
        CUDA_CHECK(cudaFree(dAmbCount));
    }
    static_assert(FF_TILE % 32 == 0, "tiles must cover whole flag words");
    for (int i = 0; i < N; ++i) {
        const uint32_t* row = &flagBits[static_cast<size_t>(i) * wordsPerRow];
        for (int k = 0; k < tilesPerRow; ++k) {
            unsigned long long cnt = 0;
            const int w0 = k * (FF_TILE / 32);
            const int w1 = std::min(wordsPerRow, w0 + FF_TILE / 32);
            for (int w = w0; w < w1; ++w) cnt += __builtin_popcount(row[w]);
            tileStart[static_cast<size_t>(i) * tilesPerRow + k + 1] = cnt;
        }
    }
    for (size_t t = 0; t < numTiles; ++t) tileStart[t + 1] += tileStart[t];
    const unsigned long long totalPairs = tileStart[numTiles];
    const unsigned long long totalRands = totalPairs * RANDS_PER_PAIR;

    uint32_t* dFlagBits = nullptr;
    unsigned long long* dTileStart = nullptr;
    CUDA_CHECK(cudaMalloc(&dFlagBits, flagBits.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dTileStart, tileStart.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(dFlagBits, flagBits.data(), flagBits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTileStart, tileStart.data(), tileStart.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    if (totalRands == 0) {
        for (int i = 0; i < N; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) printf("  Progress: %d/%d triangles\n", i + 1, N);
        }
        CUDA_CHECK(cudaFree(dFlagBits));
        CUDA_CHECK(cudaFree(dTileStart));
        return;
    }

    // ---- Chunking of the random stream ----
    // S outputs per chunk (power of two, multiple of RANDS_PER_PAIR): at least
    // 2^18 so that small problems need few doubling rounds, at most ~2048 chunks.
    const int sExp = std::max(18, ceilLog2((totalRands + 2047) / 2048));
    const unsigned long long S = 1ull << sExp;
    const int C = static_cast<int>((totalRands + S - 1) / S);
    const int K = ceilLog2(static_cast<unsigned long long>(C));  // doubling rounds

    // ---- Jump polynomials: g_k = x^(S * 2^k) mod phi ----
    PolyMod pm(state.mtCharPoly);
    std::vector<uint64_t> p(MT_PW, 0);
    p[0] = 2;  // x
    for (int s = 0; s < sExp; ++s) pm.square(p);
    std::vector<uint32_t> hG(static_cast<size_t>(std::max(K, 1)) * MT_N, 0u);
    for (int k = 0; k < K; ++k) {
        for (int w = 0; w < MT_PW; ++w) {
            hG[static_cast<size_t>(k) * MT_N + 2 * w] = static_cast<uint32_t>(p[w]);
            hG[static_cast<size_t>(k) * MT_N + 2 * w + 1] = static_cast<uint32_t>(p[w] >> 32);
        }
        pm.square(p);
    }

    // ---- Chunk start states on the device (W_0 = seeded state after the first twist) ----
    MTState mt(42);
    mt.twist();
    uint32_t* dG = nullptr;
    uint32_t* dW = nullptr;
    CUDA_CHECK(cudaMalloc(&dG, hG.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dW, static_cast<size_t>(C) * MT_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dG, hG.data(), hG.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, mt.x, MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    for (int k = 0; k < K; ++k) {
        const int cStart = 1 << k;
        const int cEnd = std::min(C, 2 << k);
        if (cEnd <= cStart) break;
        mtJumpKernel<<<cEnd - cStart, JUMP_THREADS>>>(dG + static_cast<size_t>(k) * MT_N, dW, dW, cStart, cStart);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Batched generation + form factor evaluation (double buffered) ----
    const unsigned long long maxBufRands = std::max<unsigned long long>(1ull << 27, S);  // 512 MB of draws
    const int G = static_cast<int>(std::min<unsigned long long>(maxBufRands / S, static_cast<unsigned long long>(C)));
    const int numBatches = (C + G - 1) / G;
    const int numBufs = numBatches > 1 ? 2 : 1;

    uint32_t* dRand[2] = {nullptr, nullptr};
    cudaStream_t sGen, sFF;
    cudaEvent_t evGen[2], evFF[2];
    CUDA_CHECK(cudaStreamCreate(&sGen));
    CUDA_CHECK(cudaStreamCreate(&sFF));
    for (int b = 0; b < numBufs; ++b) {
        CUDA_CHECK(cudaMalloc(&dRand[b], static_cast<size_t>(G) * S * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&evGen[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evFF[b], cudaEventDisableTiming));
    }

    auto launchGen = [&](int batch) {
        const int buf = batch % 2;
        const int cFirst = batch * G;
        const int cCount = std::min(G, C - cFirst);
        if (batch >= 2) CUDA_CHECK(cudaStreamWaitEvent(sGen, evFF[buf], 0));  // buffer free again
        mtGenKernel<<<cCount, 256, 0, sGen>>>(dW, dRand[buf], cFirst, S, static_cast<unsigned long long>(cFirst) * S);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evGen[buf], sGen));
    };

    // The Horner kernels ran on the legacy default stream; make sGen wait for them.
    CUDA_CHECK(cudaDeviceSynchronize());
    launchGen(0);

    int printedRows = 0;
    for (int batch = 0; batch < numBatches; ++batch) {
        const int buf = batch % 2;
        const int cFirst = batch * G;
        const int cCount = std::min(G, C - cFirst);
        const unsigned long long batchStart = static_cast<unsigned long long>(cFirst) * S;
        const unsigned long long batchEnd = batchStart + static_cast<unsigned long long>(cCount) * S;

        // Tiles overlapping [batchStart, batchEnd) in the draw stream
        auto tileBegin = [&](size_t t) { return tileStart[t] * RANDS_PER_PAIR; };
        size_t tFirst = 0, tLast = numTiles;  // tLast exclusive
        {
            // first tile whose end > batchStart
            size_t lo = 0, hi = numTiles;
            while (lo < hi) { size_t mid = (lo + hi) / 2; if (tileBegin(mid + 1) > batchStart) hi = mid; else lo = mid + 1; }
            tFirst = lo;
            // first tile whose begin >= batchEnd
            lo = tFirst; hi = numTiles;
            while (lo < hi) { size_t mid = (lo + hi) / 2; if (tileBegin(mid) >= batchEnd) hi = mid; else lo = mid + 1; }
            tLast = lo;
        }

        CUDA_CHECK(cudaStreamWaitEvent(sFF, evGen[buf], 0));
        if (tLast > tFirst) {
            formFactorKernel<<<static_cast<unsigned>(tLast - tFirst), FF_TILE, 0, sFF>>>(
                state.dTriA, state.dTriB, state.dTriC, state.dTriN, state.dNodes, state.dLeafGeom,
                dFlagBits, N, wordsPerRow, tilesPerRow, dTileStart, static_cast<int>(tFirst),
                dRand[buf], batchStart, batchEnd, state.dKij);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(evFF[buf], sFF));

        if (batch + 1 < numBatches) launchGen(batch + 1);

        CUDA_CHECK(cudaEventSynchronize(evFF[buf]));

        // Progress: rows whose draws are entirely consumed by now
        int doneRows = (batch + 1 == numBatches) ? N : static_cast<int>(tLast / tilesPerRow);
        for (int i = printedRows; i < doneRows; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) printf("  Progress: %d/%d triangles\n", i + 1, N);
        }
        printedRows = doneRows;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = 0; b < numBufs; ++b) {
        CUDA_CHECK(cudaFree(dRand[b]));
        CUDA_CHECK(cudaEventDestroy(evGen[b]));
        CUDA_CHECK(cudaEventDestroy(evFF[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(sGen));
    CUDA_CHECK(cudaStreamDestroy(sFF));
    CUDA_CHECK(cudaFree(dG));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dFlagBits));
    CUDA_CHECK(cudaFree(dTileStart));

}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);
    const size_t total = static_cast<size_t>(N) * N;
    const int block = 256;
    const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    tauKernel<<<grid, block>>>(state.dTriA, state.dTriB, state.dTriC, N, state.dTau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Pack form factors and delays into the layout used by the simulation kernel
void prepareSimulation(SimulationState& state) {
    const int N = static_cast<int>(state.numTriangles);
    const int Nq = (N + 3) / 4;
    const size_t total = static_cast<size_t>(N) * N;

    int* dMax = nullptr;
    CUDA_CHECK(cudaMalloc(&dMax, sizeof(int)));
    CUDA_CHECK(cudaMemset(dMax, 0, sizeof(int)));
    maxTauKernel<<<1024, 256>>>(state.dTau, total, dMax);
    CUDA_CHECK(cudaGetLastError());
    int maxTau = 0;
    CUDA_CHECK(cudaMemcpy(&maxTau, dMax, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMax));
    state.tau8 = maxTau < 255;

    CUDA_CHECK(cudaMalloc(&state.dKijQ, static_cast<size_t>(Nq) * N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTauQ, static_cast<size_t>(Nq) * N * 4 * (state.tau8 ? sizeof(uint8_t) : sizeof(int))));
    dim3 pg((Nq + 127) / 128, N);
    if (state.tau8) {
        packSimKernel<uint8_t><<<pg, 128>>>(state.dKij, state.dAreas, state.dTau, N, Nq,
                                            state.dKijQ, static_cast<uint8_t*>(state.dTauQ));
    } else {
        packSimKernel<int><<<pg, 128>>>(state.dKij, state.dAreas, state.dTau, N, Nq,
                                        state.dKijQ, static_cast<int*>(state.dTauQ));
    }
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
    const int Nq = (N + 3) / 4;
    const int block = 64;
    const unsigned grid = static_cast<unsigned>((N + block - 1) / block);

    for (int t = 0; t < T; ++t) {
        if (state.tau8) {
            simulationStepKernel<uint8_t><<<grid, block>>>(N, Nq, T, t, state.dKijQ,
                static_cast<const uint8_t*>(state.dTauQ), state.dRho, state.dRadE, state.dRadB, state.dRadBT);
        } else {
            simulationStepKernel<int><<<grid, block>>>(N, Nq, T, t, state.dKijQ,
                static_cast<const int*>(state.dTauQ), state.dRho, state.dRadE, state.dRadB, state.dRadBT);
        }
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

    distanceKernel<<<N, DIST_BLOCK>>>(T, state.dRadBT, static_cast<int>(state.sourceIndex), state.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy device results back into the host-side state
void downloadResults(SimulationState& state, bool withMatrices) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, T * N * sizeof(float), cudaMemcpyDeviceToHost));
    if (withMatrices) {
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dKij, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.tau.data(), state.dTau, N * N * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

void releaseDevice(SimulationState& state) {
    cudaFree(state.dTriA); cudaFree(state.dTriB); cudaFree(state.dTriC); cudaFree(state.dTriN);
    cudaFree(state.dNodes); cudaFree(state.dLeafGeom);
    cudaFree(state.dAreas); cudaFree(state.dRho);
    cudaFree(state.dKij); cudaFree(state.dKijQ); cudaFree(state.dTauQ); cudaFree(state.dTau);
    cudaFree(state.dRadE); cudaFree(state.dRadB); cudaFree(state.dRadBT); cudaFree(state.dDistances);
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
    prepareSimulation(state);

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

    // Bring results back to the host (full matrices only when validation needs them)
    downloadResults(state, validate);

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
    bool ok = true;
    if (validate) {
        ok = validateResults(state);
    }

    releaseDevice(state);
    return ok ? 0 : 1;
}

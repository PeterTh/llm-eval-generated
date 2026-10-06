/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * GPU parallelization (CUDA, all visible GPUs for large meshes, each GPU owning
 * a block of receiving triangles):
 * - Form factors: one thread per ray, warp-cooperative octree traversal. The
 *   sequential std::mt19937 stream is reproduced bit-exactly: every non-culled
 *   pair consumes a fixed number of random numbers, so the stream position of
 *   each pair is known (prefix sum), and segment start states are obtained by
 *   exact mt19937 jump-ahead, allowing parallel generation.
 * - Simulation: per timestep, receivers in parallel, each sum accumulated in
 *   the original source order (results are identical to the sequential code).
 * - Cross-correlation: one thread per triangle.
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
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

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

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// ============================================================================
// Random Number Generation (bit-exact GPU replica of std::mt19937 +
// std::uniform_real_distribution<float>(0, 1) as used by the sequential code)
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_NM = MT_N - MT_M;   // 227
constexpr uint32_t MT_MATRIX = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr int RANDS_PER_PAIR = NUM_RAYS * 4;   // u,v for pI and u,v for pJ per ray
constexpr int MT_GEN_THREADS = 256;            // >= MT_NM

// Seeded initial state, identical to std::mt19937(seed)
static void mtSeedState(uint32_t seed, uint32_t* st) {
    st[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        st[i] = 1812433253u * (st[i - 1] ^ (st[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

__device__ __forceinline__ uint32_t mtTwist(uint32_t cur, uint32_t next, uint32_t far) {
    uint32_t y = (cur & MT_UPPER) | (next & MT_LOWER);
    return far ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX : 0u);
}

// Converts a raw mt19937 output to the float produced by
// uniform_real_distribution<float>(0,1) (libstdc++ generate_canonical)
__device__ __forceinline__ float mtTemperToFloat(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    float r = __fmul_rn(__uint2float_rn(y), 2.3283064365386963e-10f);  // / 2^32 (exact)
    if (r >= 1.0f) r = __uint_as_float(0x3f7fffffu);                     // nextafter(1, 0)
    return r;
}

// Generates mt19937 output for consecutive segments, one CTA per segment.
// Segment b starts from the 624-word state states[b] (a state at a refresh
// boundary) and produces up to segGens refreshes (624 outputs each) into
// out[b * segGens * 624 ...]. The twist of one refresh is split into three
// dependency-free phases.
__global__ void __launch_bounds__(MT_GEN_THREADS)
mtGenerateKernel(const uint32_t* __restrict__ states, float* __restrict__ out,
                 uint32_t segGens, unsigned long long totalGens) {
    __shared__ uint32_t s[2][MT_N];
    const int tid = threadIdx.x;
    const unsigned long long firstGen = static_cast<unsigned long long>(blockIdx.x) * segGens;
    if (firstGen >= totalGens) return;
    const uint32_t numGens = static_cast<uint32_t>(
        totalGens - firstGen < segGens ? totalGens - firstGen : segGens);
    const uint32_t* state = states + static_cast<size_t>(blockIdx.x) * MT_N;
    out += firstGen * MT_N;

    for (int k = tid; k < MT_N; k += blockDim.x) s[0][k] = state[k];
    __syncthreads();

    int cur = 0;
    for (uint32_t g = 0; g < numGens; ++g) {
        const uint32_t* O = s[cur];
        uint32_t* N = s[cur ^ 1];
        float* o = out + static_cast<size_t>(g) * MT_N;

        if (tid < MT_NM) {
            int k = tid;
            uint32_t v = mtTwist(O[k], O[k + 1], O[k + MT_M]);
            N[k] = v;
            o[k] = mtTemperToFloat(v);
        }
        __syncthreads();
        if (tid < MT_NM) {
            int k = tid + MT_NM;
            uint32_t v = mtTwist(O[k], O[k + 1], N[k - MT_NM]);
            N[k] = v;
            o[k] = mtTemperToFloat(v);
        }
        __syncthreads();
        if (tid < MT_N - 2 * MT_NM) {
            int k = tid + 2 * MT_NM;
            uint32_t next = (k == MT_N - 1) ? N[0] : O[k + 1];
            uint32_t v = mtTwist(O[k], next, N[k - MT_NM]);
            N[k] = v;
            o[k] = mtTemperToFloat(v);
        }
        __syncthreads();
        cur ^= 1;
    }
}

// ----------------------------------------------------------------------------
// mt19937 jump-ahead (exact): the state after S further outputs equals
// q(A) applied to the current state, with q(x) = x^S mod phi(x) and phi the
// characteristic polynomial of the generator. With y[0..623] = state words and
// y[m] = y[m-227] ^ twist(y[m-624], y[m-623]), the jumped state is
// Z[k] = XOR_{i : q_i = 1} y[i + k]  (only the upper bit of Z[0] is relevant,
// exactly as for any mt19937 state).
// ----------------------------------------------------------------------------

constexpr int MT_DEG = 19937;
constexpr int MT_JUMP_SEQ = MT_DEG + MT_N;   // y[0 .. MT_DEG + 622] needed
constexpr int MT_JUMP_THREADS = 640;
constexpr int MT_SEG_LOG2_GENS = 12;                       // refreshes per segment = 4096
constexpr uint32_t MT_SEG_GENS = 1u << MT_SEG_LOG2_GENS;
constexpr unsigned long long MT_SEG_WORDS = static_cast<unsigned long long>(MT_SEG_GENS) * MT_N;
constexpr unsigned long long PAIRS_PER_SEG = MT_SEG_WORDS / RANDS_PER_PAIR;   // exact
static_assert(MT_SEG_WORDS % RANDS_PER_PAIR == 0, "segments must hold whole pairs");

// Applies the jump polynomial given by its set-bit positions to the state of
// CTA b: dst[b] = q(A) src[b]. One CTA per jump.
__global__ void __launch_bounds__(MT_JUMP_THREADS)
mtJumpKernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
             const uint16_t* __restrict__ jumpBits, int numJumpBits) {
    extern __shared__ uint32_t y[];   // MT_JUMP_SEQ words
    const int tid = threadIdx.x;
    src += static_cast<size_t>(blockIdx.x) * MT_N;
    dst += static_cast<size_t>(blockIdx.x) * MT_N;
    for (int k = tid; k < MT_N; k += blockDim.x) y[k] = src[k];
    __syncthreads();

    // Extend the sequence in batches of 227 independent words
    for (int m0 = MT_N; m0 < MT_JUMP_SEQ; m0 += MT_NM) {
        int m = m0 + tid;
        if (tid < MT_NM && m < MT_JUMP_SEQ) y[m] = mtTwist(y[m - MT_N], y[m - MT_N + 1], y[m - MT_NM]);
        __syncthreads();
    }
    if (tid >= MT_N) return;

    const uint2* bits4 = reinterpret_cast<const uint2*>(jumpBits);
    const int n4 = numJumpBits >> 2;
    uint32_t z0 = 0, z1 = 0;
#pragma unroll 4
    for (int b = 0; b < n4; ++b) {
        uint2 o = __ldg(&bits4[b]);
        z0 ^= y[(o.x & 0xffffu) + tid] ^ y[(o.x >> 16) + tid];
        z1 ^= y[(o.y & 0xffffu) + tid] ^ y[(o.y >> 16) + tid];
    }
    for (int b = n4 * 4; b < numJumpBits; ++b) z0 ^= y[__ldg(&jumpBits[b]) + tid];
    dst[tid] = z0 ^ z1;
}

// Exponents of the (135-term) characteristic polynomial phi(x) of mt19937,
// obtained with Berlekamp-Massey from the generator's output and re-checked
// at runtime by mtCheckCharPoly().
static const int MT_PHI_TERMS[] = {
    0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681, 3908, 4135, 4362,
    4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752, 7979, 8206, 9505, 9901, 9969,
    10128, 10693, 10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374, 11384, 11485,
    11611, 11712, 11717, 11838, 11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509,
    12620, 12673, 12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243,
    13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093, 14151, 14209,
    14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774, 14779, 14953, 15001, 15006,
    15059, 15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682, 15909, 15967,
    16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590, 16595, 16817, 16822,
    16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783, 17841, 17952,
    18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314, 19937};
constexpr int MT_PHI_NUM_TERMS = static_cast<int>(sizeof(MT_PHI_TERMS) / sizeof(MT_PHI_TERMS[0]));

// Verifies sum_k phi_k * s[n + k] = 0 on a stretch of an output bit sequence
static bool mtCheckCharPoly() {
    const int checks = 2048;
    std::vector<uint32_t> y(MT_N + MT_DEG + checks + 1);
    mtSeedState(5489u, y.data());
    for (size_t m = MT_N; m < y.size(); ++m) {
        uint32_t v = (y[m - MT_N] & MT_UPPER) | (y[m - MT_N + 1] & MT_LOWER);
        y[m] = y[m - MT_NM] ^ (v >> 1) ^ ((v & 1u) ? MT_MATRIX : 0u);
    }
    for (int n = 0; n < checks; ++n) {
        uint32_t acc = 0;
        for (int t = 0; t < MT_PHI_NUM_TERMS; ++t) acc ^= y[MT_N + n + MT_PHI_TERMS[t]];
        if (acc != 0) return false;   // all 32 bit-sequences must satisfy it
    }
    return true;
}

// a(x) mod phi(x) for deg(a) < 2 * MT_DEG (bit i of a = coefficient of x^i),
// eliminating 64 coefficients at a time using the sparsity of phi
static void gf2ReduceModPhi(std::vector<uint64_t>& a) {
    const int topWord = MT_DEG >> 6, topBit = MT_DEG & 63;
    for (int t = static_cast<int>(a.size()) - 1; t >= topWord; --t) {
        uint64_t v = a[t];
        if (t == topWord) v &= ~((1ull << topBit) - 1);
        if (!v) continue;
        a[t] ^= v;
        const long base = 64L * t - MT_DEG;   // x^(64t + b) = x^(base + b) * x^MT_DEG
        for (int k = 0; k < MT_PHI_NUM_TERMS - 1; ++k) {
            long pos = base + MT_PHI_TERMS[k];
            if (pos < 0) {
                a[0] ^= v >> (-pos);
                continue;
            }
            size_t w = static_cast<size_t>(pos) >> 6;
            int sh = static_cast<int>(pos & 63);
            a[w] ^= v << sh;
            if (sh) a[w + 1] ^= v >> (64 - sh);
        }
    }
}

// Set-bit positions of q_t(x) = x^(624 * 2^(log2Gens + t)) mod phi(x) for
// t = 0..levels, i.e. the jump polynomials for 1, 2, 4, ... segments
static std::vector<std::vector<uint16_t>> mtJumpPolys(int log2Gens, int levels) {
    const size_t AW = (2 * MT_DEG + 63) / 64 + 1;
    std::vector<uint64_t> q(AW, 0), sq(AW);
    q[MT_N >> 6] = 1ull << (MT_N & 63);   // x^624
    // Squaring over GF(2) spreads the bits: bit i -> bit 2i
    uint16_t spread[256];
    for (int v = 0; v < 256; ++v) {
        uint16_t r = 0;
        for (int b = 0; b < 8; ++b) r |= static_cast<uint16_t>(((v >> b) & 1) << (2 * b));
        spread[v] = r;
    }
    auto spread32 = [&](uint32_t v) {
        return static_cast<uint64_t>(spread[v & 0xff]) | (static_cast<uint64_t>(spread[(v >> 8) & 0xff]) << 16) |
               (static_cast<uint64_t>(spread[(v >> 16) & 0xff]) << 32) |
               (static_cast<uint64_t>(spread[v >> 24]) << 48);
    };
    auto square = [&]() {
        std::fill(sq.begin(), sq.end(), 0);
        for (size_t w = 0; w < AW / 2; ++w) {
            uint64_t v = q[w];
            if (!v) continue;
            sq[2 * w] = spread32(static_cast<uint32_t>(v));
            sq[2 * w + 1] = spread32(static_cast<uint32_t>(v >> 32));
        }
        gf2ReduceModPhi(sq);
        q.swap(sq);
    };
    for (int it = 0; it < log2Gens; ++it) square();
    std::vector<std::vector<uint16_t>> polys;
    for (int t = 0; t <= levels; ++t) {
        if (t > 0) square();
        std::vector<uint16_t> bits;
        for (int w = 0; w <= (MT_DEG - 1) >> 6; ++w) {
            for (uint64_t v = q[w]; v; v &= v - 1) {
                int i = 64 * w + __builtin_ctzll(v);
                if (i < MT_DEG) bits.push_back(static_cast<uint16_t>(i));
            }
        }
        polys.push_back(std::move(bits));
    }
    return polys;
}

// ============================================================================
// Device geometry
// ============================================================================

struct DevTri {
    float4 v0;   // a.xyz
    float4 e1;   // (b - a).xyz
    float4 e2;   // (c - a).xyz
};

// Octree node: box center/half extent; c.w = index of the first child (inner
// node, children are contiguous) or of the first leaf triangle record (leaf);
// h.w = number of children or (number of triangles | NODE_LEAF)
struct DevNode {
    float4 c;
    float4 h;
};
constexpr int NODE_LEAF = 1 << 30;

__device__ __forceinline__ float3 f3sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Same-direction culling test, with the exact operation order of the host build
__device__ __forceinline__ bool culledPair(float3 nI, float3 nJ) {
    float d = __fmaf_rn(nI.z, nJ.z, __fmaf_rn(nI.x, nJ.x, __fmul_rn(nI.y, nJ.y)));
    return d > 0.99f;
}

// Tau computation with the exact operation order of the host build
__device__ __forceinline__ int tauFromCenters(float3 ci, float3 cj) {
    float dx = __fsub_rn(ci.x, cj.x);
    float dy = __fsub_rn(ci.y, cj.y);
    float dz = __fsub_rn(ci.z, cj.z);
    float sq = __fmaf_rn(dz, dz, __fmaf_rn(dx, dx, __fmul_rn(dy, dy)));
    float dist = __fsqrt_rn(sq);
    return static_cast<int>(ceilf(__fmul_rn(dist, INV_WAVE_SPEED)));
}

__device__ __forceinline__ float rayTriDev(float3 orig, float3 dir, const DevTri& t) {
    const float BIG = 3.402823466e+38f;
    float3 v0 = make_float3(t.v0.x, t.v0.y, t.v0.z);
    float3 e1 = make_float3(t.e1.x, t.e1.y, t.e1.z);
    float3 e2 = make_float3(t.e2.x, t.e2.y, t.e2.z);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);
    if (fabsf(det) < EPSILON) return BIG;
    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return BIG;
    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return BIG;
    return f3dot(e2, qvec) * invDet;
}

// Segment/box test; mid = p1 + d with d = (p2 - p1) / 2
__device__ __forceinline__ bool rayBoxDev(const DevNode& nd, float3 mid, float3 d, float3 ad) {
    float3 c = make_float3(mid.x - nd.c.x, mid.y - nd.c.y, mid.z - nd.c.z);
    if (fabsf(c.x) > nd.h.x + ad.x) return false;
    if (fabsf(c.y) > nd.h.y + ad.y) return false;
    if (fabsf(c.z) > nd.h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > nd.h.y * ad.z + nd.h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.h.z * ad.x + nd.h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.h.x * ad.y + nd.h.y * ad.x + EPSILON) return false;
    return true;
}

constexpr int TRAVERSAL_STACK = 128;

// Octree-accelerated visibility test, warp-cooperative ("packet") traversal.
// Must be called by all 32 lanes of a warp, each with its own ray. The warp
// shares one stack of (node, lane mask) entries; a lane takes part in a node
// only if its own ray reached it, so every lane visits exactly the nodes the
// sequential per-ray traversal visits. The result does not depend on the
// visiting order.
__device__ bool isRayBlockedWarp(float3 from, float3 to, const DevNode* __restrict__ nodes,
                                 const DevTri* __restrict__ leafTris, int src, int dst, int4* stack) {
    const int lane = threadIdx.x & 31;
    float3 dir = f3sub(to, from);
    float rayLen = sqrtf(f3dot(dir, dir));
    bool blocked = rayLen < EPSILON;
    float3 dirN = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);
    float3 d = make_float3(dir.x * 0.5f, dir.y * 0.5f, dir.z * 0.5f);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    float3 mid = make_float3(from.x + d.x, from.y + d.y, from.z + d.z);
    float limit = rayLen - EPSILON;

    unsigned alive = __ballot_sync(0xffffffffu, !blocked);
    int sp = 0;
    if (alive) {   // root: visited without a box test
        if (lane == 0) {
            const DevNode root = nodes[0];
            stack[0] = make_int4(__float_as_int(root.c.w), __float_as_int(root.h.w), static_cast<int>(alive), 0);
        }
        sp = 1;
    }
    __syncwarp();
    while (sp > 0) {
        --sp;
        const int4 e = stack[sp];
        __syncwarp();
        const unsigned m = static_cast<unsigned>(e.z) & alive;
        if (!m) continue;
        const bool mine = (m >> lane) & 1u;
        if (e.y & NODE_LEAF) {
            if (mine) {
                const int cnt = e.y & ~NODE_LEAF;
                for (int k = 0; k < cnt; ++k) {
                    const DevTri t = leafTris[e.x + k];
                    const int idx = __float_as_int(t.v0.w);
                    if (idx == src || idx == dst) continue;
                    float dist = rayTriDev(from, dirN, t);
                    if (dist > EPSILON && dist < limit) {
                        blocked = true;
                        break;
                    }
                }
            }
            alive &= ~__ballot_sync(0xffffffffu, blocked);
        } else {
            for (int k = 0; k < e.y; ++k) {
                const DevNode ch = nodes[e.x + k];
                const bool hit = mine && rayBoxDev(ch, mid, d, ad);
                const unsigned cm = __ballot_sync(0xffffffffu, hit);
                if (cm) {
                    if (lane == 0)
                        stack[sp] = make_int4(__float_as_int(ch.c.w), __float_as_int(ch.h.w), static_cast<int>(cm), 0);
                    ++sp;
                }
            }
            __syncwarp();
        }
    }
    return blocked;
}

__device__ __forceinline__ float3 pointInTri(const DevTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return make_float3(t.v0.x + t.e1.x * u + t.e2.x * v,
                       t.v0.y + t.e1.y * u + t.e2.y * v,
                       t.v0.z + t.e1.z * u + t.e2.z * v);
}

// ============================================================================
// Kernels: pair enumeration, form factors, time delays
// ============================================================================

// One warp per (row i, block of 32 columns): mask and number of non-culled pairs
__global__ void countPairsKernel(const float3* __restrict__ normals, int n, int nb,
                                 unsigned long long numTasks, unsigned long long* __restrict__ counts,
                                 uint32_t* __restrict__ masks) {
    unsigned long long w = (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= numTasks) return;
    int i = static_cast<int>(w / nb);
    int j = static_cast<int>(w % nb) * 32 + lane;
    bool valid = j < n && j != i && !culledPair(normals[i], normals[j]);
    unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) {
        counts[w] = __popc(mask);
        masks[w] = mask;
    }
}

// One thread per ray, NUM_RAYS consecutive threads per non-culled pair.
// Pairs [p0, p1) of the global (row-major) pair sequence are processed; the
// random numbers of pair p start at rnd[(p - p0) * RANDS_PER_PAIR]. kij holds
// the rows r0.. of the form factor matrix.
constexpr int FF_THREADS = 128;
static_assert(32 % NUM_RAYS == 0, "a warp must hold whole pairs");

__global__ void __launch_bounds__(FF_THREADS)
formFactorKernel(const DevTri* __restrict__ tris, const float3* __restrict__ normals,
                 const DevNode* __restrict__ nodes, const DevTri* __restrict__ leafTris, int n, int nb,
                 unsigned long long numTasks, const unsigned long long* __restrict__ excl,
                 const uint32_t* __restrict__ masks,
                 unsigned long long p0, unsigned long long p1, int r0,
                 const float* __restrict__ rnd, float* __restrict__ kij) {
    const unsigned long long gid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int r = static_cast<int>(gid % NUM_RAYS);
    const int lane = threadIdx.x & 31;
    const bool active = p0 + gid / NUM_RAYS < p1;
    const unsigned long long p = active ? p0 + gid / NUM_RAYS : p1 - 1;

    // Locate the warp task containing pair p: excl[w] <= p < excl[w + 1]
    unsigned long long lo = 0, hi = numTasks;   // excl[numTasks] = total > p
    while (hi - lo > 1) {
        unsigned long long mid = (lo + hi) >> 1;
        if (__ldg(&excl[mid]) <= p) lo = mid; else hi = mid;
    }
    const unsigned long long w = lo;
    uint32_t m = __ldg(&masks[w]);
    for (unsigned long long k = p - __ldg(&excl[w]); k > 0; --k) m &= m - 1;
    const int i = static_cast<int>(w / nb);
    const int j = static_cast<int>(w % nb) * 32 + (__ffs(m) - 1);

    const DevTri tI = tris[i];
    const DevTri tJ = tris[j];
    const float3 nI = normals[i];
    const float3 nJ = normals[j];
    const float4 q = __ldg(reinterpret_cast<const float4*>(rnd + (p - p0) * RANDS_PER_PAIR) + r);
    const float3 pI = pointInTri(tI, q.x, q.y);
    const float3 pJ = pointInTri(tJ, q.z, q.w);

    __shared__ int4 sStack[FF_THREADS / 32][TRAVERSAL_STACK];
    int4* stack = sStack[threadIdx.x >> 5];

    float contrib = ZERO;   // skipped rays contribute exactly nothing
    if (!isRayBlockedWarp(pI, pJ, nodes, leafTris, i, j, stack)) {
        float3 v = f3sub(pJ, pI);
        float distSqr = f3dot(v, v);
        if (distSqr >= EPSILON) {
            float vNorm = sqrtf(distSqr);
            if (vNorm > EPSILON) {
                float cI = fmaxf(ZERO, f3dot(v, nI) / vNorm);
                float cJ = fmaxf(ZERO, -f3dot(v, nJ) / vNorm);
                if (cI > ZERO && cJ > ZERO) contrib = (cI * cJ) / (PI * distSqr);
            }
        }
    }

    // Accumulate in ray order, as the sequential loop does
    const int groupBase = lane & ~(NUM_RAYS - 1);
    float acc = ZERO;
#pragma unroll
    for (int k = 0; k < NUM_RAYS; ++k) acc += __shfl_sync(0xffffffffu, contrib, groupBase + k);
    if (r == 0 && active) kij[static_cast<size_t>(i - r0) * n + j] = acc * INV_NUM_RAYS;
}

// Builds the transposed coupling weights of the receivers i = r0 .. r0+nloc-1
// used by the simulation: wT[j][i - r0] = min(kij[i][j] * area[j], 1), or 0 if
// kij <= 0 (kij holds the local rows)
constexpr int TILE = 32;
__global__ void couplingKernel(const float* __restrict__ kij, const float* __restrict__ areas,
                               int n, int r0, int nloc, float* __restrict__ wT) {
    __shared__ float tile[TILE][TILE + 1];
    int bi = blockIdx.y * TILE;   // local row block (i - r0)
    int bj = blockIdx.x * TILE;   // column block (j)
    int tx = threadIdx.x, ty = threadIdx.y;   // 32 x 8
    for (int r = ty; r < TILE; r += blockDim.y) {
        int ii = bi + r, j = bj + tx;
        tile[r][tx] = (ii < nloc && j < n) ? kij[static_cast<size_t>(ii) * n + j] : ZERO;
    }
    __syncthreads();
    for (int r = ty; r < TILE; r += blockDim.y) {
        int j = bj + r, ii = bi + tx;
        if (ii < nloc && j < n) {
            float k = tile[tx][r];
            float w = (r0 + ii != j && k > ZERO) ? fminf(__fmul_rn(k, areas[j]), ONE) : ZERO;
            wT[static_cast<size_t>(j) * nloc + ii] = w;
        }
    }
}

// Time delays of the receivers i = r0 .. r0+nloc-1: tauT[j][i - r0] = tau(i, j)
__global__ void tauKernel(const float3* __restrict__ centers, int n, int r0, int nloc,
                          unsigned char* __restrict__ tauT) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(n) * nloc) return;
    int j = static_cast<int>(idx / nloc), i = r0 + static_cast<int>(idx % nloc);
    tauT[idx] = static_cast<unsigned char>((i != j) ? tauFromCenters(centers[i], centers[j]) : 0);
}

// ============================================================================
// Kernels: wave propagation and cross-correlation
// ============================================================================

// One block per 32 receiving triangles (of the local receivers r0 ..
// r0+nloc-1, whose weights wT / tauT are stored as [j][i - r0]). Producer warps compute the coupling
// terms w_ij * radB[t - tau_ij][j] of a tile of sources into shared memory
// (coalesced, many loads in flight); warp 0 adds them to each receiver's sum
// sequentially in the original j order (skipped terms are exactly +0).
constexpr int SIM_I = 32;
constexpr int SIM_PRODUCER_WARPS = 8;
constexpr int SIM_TJ = 32 * SIM_PRODUCER_WARPS;   // each producer thread: SIM_I terms per tile
constexpr int SIM_THREADS = 32 * (SIM_PRODUCER_WARPS + 1);
constexpr size_t SIM_SMEM = 2ull * SIM_TJ * SIM_I * sizeof(float);   // double-buffered tile

__global__ void __launch_bounds__(SIM_THREADS)
simulationStepKernel(const float* __restrict__ wT, const unsigned char* __restrict__ tauT,
                     const float* __restrict__ rho, const float* __restrict__ radE,
                     float* __restrict__ radB, int n, int r0, int nloc, int t) {
    extern __shared__ float simTerms[];
    auto terms = reinterpret_cast<float (*)[SIM_TJ][SIM_I]>(simTerms);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int i0 = blockIdx.x * SIM_I;   // local receiver index
    const int numTiles = (n + SIM_TJ - 1) / SIM_TJ;

    // Producer thread (warp p, lane l) handles i = i0 + l and the sources
    // jj = p + SIM_PRODUCER_WARPS * k of the tile; loads are batched for
    // memory-level parallelism.
    auto produce = [&](int tile, int buf) {
        const int pw = (tid >> 5) - 1;
        const int i = i0 + lane;
        const int j0 = tile * SIM_TJ + pw;
        int tauv[SIM_I];
        float wv[SIM_I];
#pragma unroll
        for (int k = 0; k < SIM_I; ++k) {
            const int j = j0 + SIM_PRODUCER_WARPS * k;
            tauv[k] = 0;
            wv[k] = ZERO;
            if (j < n && i < nloc && j != r0 + i) {
                const size_t o = static_cast<size_t>(j) * nloc + i;
                tauv[k] = tauT[o];
                wv[k] = wT[o];
            }
        }
#pragma unroll
        for (int k = 0; k < SIM_I; ++k) {
            const int j = j0 + SIM_PRODUCER_WARPS * k;
            float term = ZERO;
            if (t >= tauv[k] && wv[k] > ZERO) {
                const float radJ = radB[static_cast<size_t>(t - tauv[k]) * n + j];
                if (radJ > ZERO) term = __fmul_rn(wv[k], radJ);
            }
            terms[buf][pw + SIM_PRODUCER_WARPS * k][lane] = term;
        }
    };

    float sumB = ZERO;
    if (tid >= 32) produce(0, 0);
    __syncthreads();
    for (int tile = 0; tile < numTiles; ++tile) {
        const int buf = tile & 1;
        if (tid >= 32) {
            if (tile + 1 < numTiles) produce(tile + 1, buf ^ 1);
        } else {
            const int cnt = min(SIM_TJ, n - tile * SIM_TJ);
            if (cnt == SIM_TJ) {
#pragma unroll 16
                for (int jj = 0; jj < SIM_TJ; ++jj) sumB = __fadd_rn(sumB, terms[buf][jj][tid]);
            } else {
                for (int jj = 0; jj < cnt; ++jj) sumB = __fadd_rn(sumB, terms[buf][jj][tid]);
            }
        }
        __syncthreads();
    }
    const int i = r0 + i0 + tid;
    if (tid < 32 && i0 + tid < nloc) {
        const size_t o = static_cast<size_t>(t) * n + i;
        radB[o] = __fmaf_rn(rho[i], sumB, radE[o]);
    }
}

__global__ void distanceKernel(const float* __restrict__ radB, int n, int T, int src,
                               float* __restrict__ distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * n + i];
            float pS = radB[static_cast<size_t>(tt - t) * n + src];
            sum = __fadd_rn(sum, __fmul_rn(pS, pB));
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Per-GPU data. Every GPU owns the receiving triangles (rows) r0 .. r1-1: it
// computes their form factors, time delays and radiosity updates.
struct DeviceContext {
    int device = 0;
    int r0 = 0, r1 = 0;

    DevTri* dTris = nullptr;
    float3* dNormals = nullptr;
    float3* dCenters = nullptr;
    DevNode* dNodes = nullptr;
    DevTri* dLeafTris = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;            // full T x N copy (exchanged every timestep)
    float* dKij = nullptr;             // local rows x N
    float* dWT = nullptr;              // N x local rows
    unsigned char* dTauT = nullptr;    // N x local rows
    float* dDistances = nullptr;

    // Form factor workspace
    unsigned long long* dCounts = nullptr;
    unsigned long long* dExcl = nullptr;
    uint32_t* dMasks = nullptr;
    void* dScanTmp = nullptr;
    size_t scanTmpBytes = 0;
    uint32_t* dMtCur = nullptr;
    uint32_t* dMtStates[2] = {nullptr, nullptr};
    float* dRnd[2] = {nullptr, nullptr};
    cudaEvent_t genDone[2] = {nullptr, nullptr};
    cudaEvent_t compDone[2] = {nullptr, nullptr};

    cudaStream_t sComp = nullptr;      // main work stream
    cudaStream_t sGen = nullptr;       // high-priority random number stream
    cudaEvent_t stepDone[2] = {nullptr, nullptr};   // by timestep parity

    int nloc() const { return r1 - r0; }
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    std::vector<DeviceContext> devs;   // time delays live on the GPUs (tauT)
    float* hRadB = nullptr;            // pinned exchange buffer (T x N)

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

static float intBitsToFloat(int v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

// Flatten the octree into device-friendly arrays (BFS, children contiguous,
// leaf triangles stored inline with their original index in v0.w)
static void flattenOctree(const Octree& root, const std::vector<DevTri>& tris,
                          std::vector<DevNode>& nodes, std::vector<DevTri>& leafTris) {
    std::vector<const Octree*> order;
    order.push_back(&root);
    nodes.clear();
    leafTris.clear();
    for (size_t q = 0; q < order.size(); ++q) {
        const Octree* o = order[q];
        int first, count;
        if (!o->triangleIndices.empty()) {
            first = static_cast<int>(leafTris.size());
            count = static_cast<int>(o->triangleIndices.size()) | NODE_LEAF;
            for (size_t idx : o->triangleIndices) {
                DevTri t = tris[idx];
                t.v0.w = intBitsToFloat(static_cast<int>(idx));
                leafTris.push_back(t);
            }
        } else {
            first = static_cast<int>(order.size());
            count = 0;
            for (int c = 0; c < 8; ++c) {
                if (o->children[c]) {
                    order.push_back(o->children[c].get());
                    ++count;
                }
            }
        }
        DevNode nd;
        nd.c = make_float4(o->center.x, o->center.y, o->center.z, intBitsToFloat(first));
        nd.h = make_float4(o->halfExtent.x, o->halfExtent.y, o->halfExtent.z, intBitsToFloat(count));
        nodes.push_back(nd);
    }
}

static int octreeDepth(const Octree& o) {
    int d = 0;
    if (o.triangleIndices.empty()) {
        for (int c = 0; c < 8; ++c)
            if (o.children[c]) d = std::max(d, octreeDepth(*o.children[c]));
    }
    return d + 1;
}

// ============================================================================
// Initialization
// ============================================================================

template <typename T>
static T* deviceUpload(const std::vector<T>& v) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// Rows per GPU below which additional GPUs are not worth their overhead
constexpr int MIN_ROWS_PER_GPU = 1280;

// Random stream layout: the sequence is split into segments of MT_SEG_GENS
// generator refreshes; a chunk of work covers CHUNK_SEGS segments (plus one
// for a non-aligned start). Segment start states are obtained by exact
// jump-ahead, so all segments of a chunk are generated in parallel.
constexpr int CHUNK_LOG2_SEGS = 5;
constexpr unsigned long long CHUNK_SEGS = 1ull << CHUNK_LOG2_SEGS;
constexpr unsigned long long PAIRS_PER_CHUNK = CHUNK_SEGS * PAIRS_PER_SEG;

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

    const size_t n = state.numTriangles;

    // Initialize areas
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    // Initialize matrices
    state.kij.resize(n * n, ZERO);
    state.radE.resize(timesteps * n, ZERO);
    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Device data
    std::vector<DevTri> tris(n);
    std::vector<float3> normals(n), centers(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 e1 = t.b - t.a, e2 = t.c - t.a;
        tris[i].v0 = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        tris[i].e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
        tris[i].e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
        Vec3 nn = t.normal();
        normals[i] = make_float3(nn.x, nn.y, nn.z);
        Vec3 c = t.center();
        centers[i] = make_float3(c.x, c.y, c.z);
    }
    std::vector<DevNode> nodes;
    std::vector<DevTri> leafTris;
    flattenOctree(state.octree, tris, nodes, leafTris);
    if (octreeDepth(state.octree) * 7 + 2 > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep for traversal stack\n");
        exit(1);
    }

    // GPUs: all visible devices, rows split evenly (load all kernels up front
    // so they are not loaded lazily inside the timed phases)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    const int numGpus = std::max(1, std::min(deviceCount, static_cast<int>(n) / MIN_ROWS_PER_GPU));
    state.devs.resize(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        dc.device = g;
        dc.r0 = static_cast<int>(n * g / numGpus);
        dc.r1 = static_cast<int>(n * (g + 1) / numGpus);
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(0));
        for (int h = 0; h < numGpus; ++h) {
            int can = 0;
            if (h != g && cudaDeviceCanAccessPeer(&can, g, h) == cudaSuccess && can) {
                cudaDeviceEnablePeerAccess(h, 0);
                cudaGetLastError();   // ignore "already enabled"
            }
        }
        int prLeast = 0, prGreatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLeast, &prGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&dc.sComp, cudaStreamNonBlocking, prLeast));
        CUDA_CHECK(cudaStreamCreateWithPriority(&dc.sGen, cudaStreamNonBlocking, prGreatest));
        for (int b = 0; b < 2; ++b) CUDA_CHECK(cudaEventCreateWithFlags(&dc.stepDone[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaFuncSetAttribute(simulationStepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(SIM_SMEM)));

        const size_t nloc = static_cast<size_t>(dc.nloc());
        dc.dTris = deviceUpload(tris);
        dc.dNormals = deviceUpload(normals);
        dc.dCenters = deviceUpload(centers);
        dc.dNodes = deviceUpload(nodes);
        dc.dLeafTris = deviceUpload(leafTris);
        dc.dAreas = deviceUpload(state.areas);
        dc.dRho = deviceUpload(state.rho);
        dc.dRadE = deviceUpload(state.radE);
        dc.dRadB = deviceUpload(state.radB);
        CUDA_CHECK(cudaMalloc(&dc.dKij, std::max<size_t>(1, nloc * n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dc.dWT, std::max<size_t>(1, nloc * n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dc.dTauT, std::max<size_t>(1, nloc * n) * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&dc.dDistances, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(dc.dKij, 0, std::max<size_t>(1, nloc * n) * sizeof(float)));

        // Form factor workspace (pair enumeration and random number buffers)
        const size_t nb = (n + 31) / 32;
        const size_t numTasks = n * nb;
        CUDA_CHECK(cudaMalloc(&dc.dCounts, (numTasks + 1) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dc.dExcl, (numTasks + 1) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dc.dMasks, numTasks * sizeof(uint32_t)));
        cub::DeviceScan::ExclusiveSum(nullptr, dc.scanTmpBytes, dc.dCounts, dc.dExcl, numTasks + 1);
        CUDA_CHECK(cudaMalloc(&dc.dScanTmp, dc.scanTmpBytes));
        const unsigned long long maxPairs = static_cast<unsigned long long>(nloc) * n;
        const unsigned long long bufSegs = std::min(
            CHUNK_SEGS + 1, (maxPairs * RANDS_PER_PAIR + MT_SEG_WORDS - 1) / MT_SEG_WORDS + 1);
        CUDA_CHECK(cudaMalloc(&dc.dMtCur, MT_N * sizeof(uint32_t)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dc.dMtStates[b], bufSegs * MT_N * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&dc.dRnd[b], bufSegs * MT_SEG_WORDS * sizeof(float)));
            CUDA_CHECK(cudaEventCreateWithFlags(&dc.genDone[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&dc.compDone[b], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaFuncSetAttribute(mtJumpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(MT_JUMP_SEQ * sizeof(uint32_t))));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Pinned host memory for the device -> host results and the per-timestep
    // exchange of radiosities between GPUs
    CUDA_CHECK(cudaHostRegister(state.kij.data(), state.kij.size() * sizeof(float), cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostAlloc(&state.hRadB, std::max<size_t>(1, state.radB.size()) * sizeof(float),
                             cudaHostAllocPortable));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int nb = (n + 31) / 32;
    const unsigned long long numTasks = static_cast<unsigned long long>(n) * nb;
    const int numGpus = static_cast<int>(state.devs.size());

    const int jumpSmem = MT_JUMP_SEQ * sizeof(uint32_t);

    struct Work {
        unsigned long long p0 = 0, p1 = 0;   // pair range of the GPU's rows
        unsigned long long seg0 = 0;         // segment containing the first pair
        unsigned long long segOff = 0;       // word offset of the first pair in it
        unsigned long long numChunks = 0;
        std::vector<uint16_t*> dJumpBits;
    };
    std::vector<Work> work(numGpus);

    // 1. Enumerate non-culled pairs (on every GPU): each consumes exactly
    //    RANDS_PER_PAIR random numbers of the sequential stream, in (i, j)
    //    row-major order.
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        CUDA_CHECK(cudaSetDevice(dc.device));
        CUDA_CHECK(cudaMemsetAsync(dc.dCounts + numTasks, 0, sizeof(unsigned long long), dc.sComp));
        const int threads = 256;
        unsigned long long blocks = (numTasks * 32 + threads - 1) / threads;
        countPairsKernel<<<static_cast<unsigned>(blocks), threads, 0, dc.sComp>>>(
            dc.dNormals, n, nb, numTasks, dc.dCounts, dc.dMasks);
        CUDA_CHECK(cudaGetLastError());
        size_t tmpBytes = dc.scanTmpBytes;
        cub::DeviceScan::ExclusiveSum(dc.dScanTmp, tmpBytes, dc.dCounts, dc.dExcl, numTasks + 1, dc.sComp);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long totalPairs = 0;
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        Work& wk = work[g];
        CUDA_CHECK(cudaSetDevice(dc.device));
        CUDA_CHECK(cudaMemcpyAsync(&wk.p0, dc.dExcl + static_cast<size_t>(dc.r0) * nb,
                                   sizeof(unsigned long long), cudaMemcpyDeviceToHost, dc.sComp));
        CUDA_CHECK(cudaMemcpyAsync(&wk.p1, dc.dExcl + static_cast<size_t>(dc.r1) * nb,
                                   sizeof(unsigned long long), cudaMemcpyDeviceToHost, dc.sComp));
        CUDA_CHECK(cudaStreamSynchronize(dc.sComp));
        totalPairs = std::max(totalPairs, wk.p1);
        wk.seg0 = wk.p0 * RANDS_PER_PAIR / MT_SEG_WORDS;
        wk.segOff = wk.p0 * RANDS_PER_PAIR % MT_SEG_WORDS;
        wk.numChunks = (wk.p1 - wk.p0 + PAIRS_PER_CHUNK - 1) / PAIRS_PER_CHUNK;
    }

    // 2. Jump polynomials for 2^t segments, t = 0..levels
    const unsigned long long totalSegs = (totalPairs * RANDS_PER_PAIR + MT_SEG_WORDS - 1) / MT_SEG_WORDS;
    int levels = CHUNK_LOG2_SEGS;
    while ((1ull << levels) < totalSegs) ++levels;
    std::vector<std::vector<uint16_t>> jumpPolys;
    if (totalSegs > 1) {
        if (!mtCheckCharPoly()) {
            fprintf(stderr, "mt19937 characteristic polynomial check failed\n");
            exit(1);
        }
        jumpPolys = mtJumpPolys(MT_SEG_LOG2_GENS, levels);
    }
    auto launchJump = [&](int g, const uint32_t* src, uint32_t* dst, unsigned jumps, int level) {
        mtJumpKernel<<<jumps, MT_JUMP_THREADS, jumpSmem, state.devs[g].sGen>>>(
            src, dst, work[g].dJumpBits[level], static_cast<int>(jumpPolys[level].size()));
        CUDA_CHECK(cudaGetLastError());
    };

    // 3. Per-GPU generator setup: start state of the GPU's first segment
    uint32_t seedState[MT_N];
    mtSeedState(42u, seedState);
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        Work& wk = work[g];
        CUDA_CHECK(cudaSetDevice(dc.device));
        for (auto& bits : jumpPolys) wk.dJumpBits.push_back(deviceUpload(bits));
        CUDA_CHECK(cudaMemcpyAsync(dc.dMtCur, seedState, sizeof(seedState), cudaMemcpyHostToDevice, dc.sGen));
        for (int t = 0; t <= levels; ++t)
            if ((wk.seg0 >> t) & 1ull) launchJump(g, dc.dMtCur, dc.dMtCur, 1, t);
    }

    // 4. Chunked pipeline per GPU: while the ray tracing of chunk c runs, the
    //    start states and random numbers of chunk c+1 are produced on the
    //    high-priority stream.
    unsigned long long maxChunks = 0;
    for (const Work& wk : work) maxChunks = std::max(maxChunks, wk.numChunks);
    for (unsigned long long c = 0; c < maxChunks; ++c) {
        for (int g = 0; g < numGpus; ++g) {
            DeviceContext& dc = state.devs[g];
            Work& wk = work[g];
            if (c >= wk.numChunks) continue;
            CUDA_CHECK(cudaSetDevice(dc.device));
            const int b = static_cast<int>(c & 1);
            const unsigned long long p0 = wk.p0 + c * PAIRS_PER_CHUNK;
            const unsigned long long p1 = std::min(wk.p1, p0 + PAIRS_PER_CHUNK);
            const unsigned long long words = wk.segOff + (p1 - p0) * RANDS_PER_PAIR;
            const unsigned long long gens = (words + MT_N - 1) / MT_N;
            const unsigned long long segs = (gens + MT_SEG_GENS - 1) / MT_SEG_GENS;

            if (c >= 2) CUDA_CHECK(cudaStreamWaitEvent(dc.sGen, dc.compDone[b], 0));
            // Segment start states: states[0] = chunk start, then doubling
            CUDA_CHECK(cudaMemcpyAsync(dc.dMtStates[b], dc.dMtCur, MT_N * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice, dc.sGen));
            for (int t = 0; (1ull << t) < segs; ++t) {
                const unsigned long long have = 1ull << t;
                launchJump(g, dc.dMtStates[b], dc.dMtStates[b] + have * MT_N,
                           static_cast<unsigned>(std::min(have, segs - have)), t);
            }
            if (c + 1 < wk.numChunks)   // start state of the next chunk
                launchJump(g, dc.dMtStates[b], dc.dMtCur, 1, CHUNK_LOG2_SEGS);
            mtGenerateKernel<<<static_cast<unsigned>(segs), MT_GEN_THREADS, 0, dc.sGen>>>(
                dc.dMtStates[b], dc.dRnd[b], MT_SEG_GENS, gens);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(dc.genDone[b], dc.sGen));

            CUDA_CHECK(cudaStreamWaitEvent(dc.sComp, dc.genDone[b], 0));
            const unsigned long long threadsTotal = (p1 - p0) * NUM_RAYS;
            const unsigned long long blocks = (threadsTotal + FF_THREADS - 1) / FF_THREADS;
            formFactorKernel<<<static_cast<unsigned>(blocks), FF_THREADS, 0, dc.sComp>>>(
                dc.dTris, dc.dNormals, dc.dNodes, dc.dLeafTris, n, nb, numTasks, dc.dExcl, dc.dMasks,
                p0, p1, dc.r0, dc.dRnd[b] + wk.segOff, dc.dKij);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(dc.compDone[b], dc.sComp));
        }
    }

    // 5. Coupling weights for the simulation (transposed for coalesced access)
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        CUDA_CHECK(cudaSetDevice(dc.device));
        dim3 block(TILE, 8);
        dim3 grid((n + TILE - 1) / TILE, (dc.nloc() + TILE - 1) / TILE);
        couplingKernel<<<grid, block, 0, dc.sComp>>>(dc.dKij, dc.dAreas, n, dc.r0, dc.nloc(), dc.dWT);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int g = 0; g < numGpus; ++g) {
        DeviceContext& dc = state.devs[g];
        CUDA_CHECK(cudaSetDevice(dc.device));
        CUDA_CHECK(cudaMemcpyAsync(state.kij.data() + static_cast<size_t>(dc.r0) * n, dc.dKij,
                                   static_cast<size_t>(dc.nloc()) * n * sizeof(float),
                                   cudaMemcpyDeviceToHost, dc.sComp));
    }
    for (int g = 0; g < numGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(state.devs[g].device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < n; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == n) {
            printf("  Progress: %d/%d triangles\n", i + 1, n);
        }
    }

    for (int g = 0; g < numGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(state.devs[g].device));
        for (auto* p : work[g].dJumpBits) CUDA_CHECK(cudaFree(p));
    }
    CUDA_CHECK(cudaSetDevice(0));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    for (DeviceContext& dc : state.devs) {
        CUDA_CHECK(cudaSetDevice(dc.device));
        const int threads = 256;
        const size_t blocks = (static_cast<size_t>(n) * dc.nloc() + threads - 1) / threads;
        tauKernel<<<static_cast<unsigned>(blocks), threads, 0, dc.sComp>>>(dc.dCenters, n, dc.r0, dc.nloc(),
                                                                           dc.dTauT);
        CUDA_CHECK(cudaGetLastError());
    }
    for (DeviceContext& dc : state.devs) {
        CUDA_CHECK(cudaSetDevice(dc.device));
        CUDA_CHECK(cudaStreamSynchronize(dc.sComp));
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int numGpus = static_cast<int>(state.devs.size());

    // Every GPU updates its receivers; with several GPUs, the new values
    // radB[t][r0..r1) are exchanged through pinned host memory before step t+1
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        for (int g = 0; g < numGpus; ++g) {
            DeviceContext& dc = state.devs[g];
            CUDA_CHECK(cudaSetDevice(dc.device));
            if (t > 0 && numGpus > 1) {
                for (int h = 0; h < numGpus; ++h)
                    if (h != g) CUDA_CHECK(cudaStreamWaitEvent(dc.sComp, state.devs[h].stepDone[(t - 1) & 1], 0));
                const size_t prev = (t - 1) * n;
                CUDA_CHECK(cudaMemcpyAsync(dc.dRadB + prev, state.hRadB + prev, n * sizeof(float),
                                           cudaMemcpyHostToDevice, dc.sComp));
            }
            const int blocks = (dc.nloc() + SIM_I - 1) / SIM_I;
            simulationStepKernel<<<blocks, SIM_THREADS, SIM_SMEM, dc.sComp>>>(
                dc.dWT, dc.dTauT, dc.dRho, dc.dRadE, dc.dRadB, n, dc.r0, dc.nloc(), static_cast<int>(t));
            CUDA_CHECK(cudaGetLastError());
            const size_t off = t * n + dc.r0;
            CUDA_CHECK(cudaMemcpyAsync(state.hRadB + off, dc.dRadB + off, dc.nloc() * sizeof(float),
                                       cudaMemcpyDeviceToHost, dc.sComp));
            CUDA_CHECK(cudaEventRecord(dc.stepDone[t & 1], dc.sComp));
        }
    }
    for (int g = 0; g < numGpus; ++g) {
        CUDA_CHECK(cudaSetDevice(state.devs[g].device));
        CUDA_CHECK(cudaStreamSynchronize(state.devs[g].sComp));
    }
    CUDA_CHECK(cudaSetDevice(0));
    std::copy(state.hRadB, state.hRadB + state.radB.size(), state.radB.begin());
    if (numGpus > 1 && state.numTimesteps > 0) {   // last row for the distance computation
        const size_t last = (state.numTimesteps - 1) * n;
        CUDA_CHECK(cudaMemcpyAsync(state.devs[0].dRadB + last, state.hRadB + last, n * sizeof(float),
                                   cudaMemcpyHostToDevice, state.devs[0].sComp));
    }

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    DeviceContext& d0 = state.devs[0];
    CUDA_CHECK(cudaSetDevice(d0.device));
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;
    distanceKernel<<<blocks, threads, 0, d0.sComp>>>(d0.dRadB, n, static_cast<int>(state.numTimesteps),
                                                     static_cast<int>(state.sourceIndex), d0.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(state.distances.data(), d0.dDistances, n * sizeof(float),
                               cudaMemcpyDeviceToHost, d0.sComp));
    CUDA_CHECK(cudaStreamSynchronize(d0.sComp));
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
            return 1;
        }
    }

    return 0;
}

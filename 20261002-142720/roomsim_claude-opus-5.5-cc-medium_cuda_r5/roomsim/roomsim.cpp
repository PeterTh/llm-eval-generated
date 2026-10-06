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
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
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
// CUDA Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

// ============================================================================
// Random Number Generation (bit-exact std::mt19937 + uniform_real_distribution)
//
// The sequential code draws all random numbers from a single mt19937 stream
// (seed 42), 64 draws per non-culled (i, j) pair in row-major order. To keep
// identical results, the GPU regenerates exactly this stream: the host only
// runs the (cheap) MT twist to record a state snapshot every K twists, and GPU
// blocks regenerate and temper the stream segments in parallel from those
// snapshots. Each pair then reads its 64 values at offset 64 * rank(i, j).
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;

static void mtSeed(uint32_t* x, uint32_t seed) {
    x[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

static inline void mtTwist(uint32_t* __restrict x) {
    for (int k = 0; k < MT_N - MT_M; ++k) {
        uint32_t y = (x[k] & MT_UPPER) | (x[k + 1] & MT_LOWER);
        x[k] = x[k + MT_M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
    }
    for (int k = MT_N - MT_M; k < MT_N - 1; ++k) {
        uint32_t y = (x[k] & MT_UPPER) | (x[k + 1] & MT_LOWER);
        x[k] = x[k + (MT_M - MT_N)] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
    }
    uint32_t y = (x[MT_N - 1] & MT_UPPER) | (x[0] & MT_LOWER);
    x[MT_N - 1] = x[MT_M - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
}

__device__ __forceinline__ float mtTemperToFloat(uint32_t z) {
    z ^= (z >> 11);
    z ^= (z << 7) & 0x9d2c5680u;
    z ^= (z << 15) & 0xefc60000u;
    z ^= (z >> 18);
    // generate_canonical<float, 24>(mt19937) followed by uniform [0, 1)
    float r = __uint2float_rn(z) * 2.3283064365386963e-10f;  // 2^-32
    if (r >= 1.0f) r = 0x1.fffffep-1f;
    return r;
}

// One block per snapshot: regenerate K twists and write the tempered floats
// whose global word index lies in [wordBegin, wordEnd).
__global__ void __launch_bounds__(256)
generateRandomsKernel(const uint32_t* __restrict__ snapshots, size_t firstSnap, int twistsPerSnap,
                      uint64_t wordBegin, uint64_t wordEnd, float* __restrict__ out) {
    __shared__ uint32_t x[MT_N];
    const size_t snap = firstSnap + blockIdx.x;
    const uint32_t* src = snapshots + static_cast<size_t>(blockIdx.x) * MT_N;
    for (int k = threadIdx.x; k < MT_N; k += blockDim.x) x[k] = src[k];
    __syncthreads();

    const int tid = threadIdx.x;
    uint64_t roundBase = static_cast<uint64_t>(snap) * twistsPerSnap * MT_N;
    for (int r = 0; r < twistsPerSnap; ++r, roundBase += MT_N) {
        if (roundBase >= wordEnd) break;
        // Phase 1: k in [0, 227) uses old x[k], x[k+1], x[k+397]
        // Phase 2: k in [227, 454) uses old x[k], x[k+1], new x[k-227]
        // Phase 3: k in [454, 624) uses old x[k], x[k+1 or 0 (new)], new x[k-227]
        const int phaseStart[3] = {0, 227, 454};
        const int phaseEnd[3] = {227, 454, 624};
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            int k = phaseStart[p] + tid;
            uint32_t v = 0;
            bool active = k < phaseEnd[p];
            if (active) {
                uint32_t next = (k + 1 < MT_N) ? x[k + 1] : x[0];
                uint32_t y = (x[k] & MT_UPPER) | (next & MT_LOWER);
                uint32_t other = (k < MT_N - MT_M) ? x[k + MT_M] : x[k + MT_M - MT_N];
                v = other ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
            }
            __syncthreads();
            if (active) x[k] = v;
            __syncthreads();
        }
        if (roundBase + MT_N > wordBegin) {
            for (int k = tid; k < MT_N; k += blockDim.x) {
                uint64_t w = roundBase + k;
                if (w >= wordBegin && w < wordEnd) out[w - wordBegin] = mtTemperToFloat(x[k]);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// MT19937 jump-ahead (GF(2) polynomial arithmetic), used to let several host
// threads produce disjoint, exactly positioned parts of the snapshot sequence.
// ----------------------------------------------------------------------------

using Gf2Poly = std::vector<uint64_t>;  // little-endian bit vector of coefficients

static inline int gf2Bit(const Gf2Poly& p, size_t i) { return static_cast<int>((p[i >> 6] >> (i & 63)) & 1u); }

// Characteristic polynomial phi(t) of the MT19937 state recurrence (degree
// 19937, little-endian coefficient bits), obtained with Berlekamp-Massey.
constexpr int MT_CHAR_POLY_DEGREE = 19937;
static const uint64_t MT_CHAR_POLY[312] = {
    0x0000000000000001ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000002000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000100ull, 0x0000000000000000ull,
    0x0002000000000000ull, 0x0000080000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000004000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x2000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000200000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0100000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000008000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x4000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000200000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000010ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000008000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000400ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000020000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000020000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000002ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000020000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000002000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0020000000000000ull, 0x0000002000000000ull, 0x0000000080000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000100ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000080000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000004000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000200000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000200000000000ull, 0x0002000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000010000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000020ull,
    0x0000000000000200ull, 0x0000000000000000ull, 0x0000010000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000020000ull, 0x0000000000200800ull, 0x0000000000008000ull,
    0x0200000000000000ull, 0x0100400000000000ull, 0x0000000000000000ull, 0x0000000020000000ull,
    0x0000000000000000ull, 0x0000000008000000ull, 0x0000000000000000ull, 0x0000000000000021ull,
    0x4000000000000000ull, 0x0000020000000000ull, 0x0000010000000000ull, 0x0000000020000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0020000000000000ull,
    0x0000800000000000ull, 0x0000020000000000ull, 0x0000000000000000ull, 0x0000000021000000ull,
    0x0000000000000000ull, 0x0000000000001000ull, 0x0800000000000002ull, 0x0020000000000001ull,
    0x0000000000000000ull, 0x0000020000000000ull, 0x0000000840000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000020000ull, 0x0800000000000042ull, 0x0020000000000000ull,
    0x0000000000000000ull, 0x0000001000000000ull, 0x0000000000000000ull, 0x0000000021000000ull,
    0x0000000000000000ull, 0x0000000000000080ull, 0x0000000000000002ull, 0x0020000000000001ull,
    0x0000040000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000002000ull, 0x0000000000000080ull, 0x0000000000000002ull, 0x0021000000000000ull,
    0x0000000000000000ull, 0x0000001000000000ull, 0x0000000000000000ull, 0x0000000001080000ull,
    0x0000000000002000ull, 0x0000000000000000ull, 0x0840000000000002ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000020000000000ull, 0x0000000042000000ull, 0x0000000000080000ull,
    0x0000000000002000ull, 0x1000000000000000ull, 0x0000000000000000ull, 0x0021000000000000ull,
    0x0000000000000000ull, 0x0000000080000000ull, 0x0000000002000000ull, 0x0000000001000000ull,
    0x0000000000002000ull, 0x0000000000000004ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000002000000000ull, 0x0000000080000000ull, 0x0000000002000000ull, 0x0000000000000000ull,
    0x0000000000002100ull, 0x1000000000000000ull, 0x0000000000000000ull, 0x0001080000000000ull,
    0x0000002000000000ull, 0x0000000000000000ull, 0x0000000002000000ull, 0x0000000000084000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0042000000000000ull, 0x0000080000000000ull,
    0x0000002000000000ull, 0x0000000000000000ull, 0x0000000000100000ull, 0x0000000000000000ull,
    0x0000000000000100ull, 0x0080000000000000ull, 0x0002000000000000ull, 0x0000000000000000ull,
    0x0000002000000000ull, 0x0000000004000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x2000000000000000ull, 0x0080000000000000ull, 0x0002000000000000ull, 0x0000000000000000ull,
    0x0000000100000000ull, 0x0000000000000000ull, 0x0000000000100000ull, 0x0000000000000000ull,
    0x2000000000000008ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000004000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000200ull,
    0x0000000000000008ull, 0x0000000000000000ull, 0x0000100000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000008000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0004000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull,
    0x0000000000000000ull, 0x0000000000000000ull, 0x0000000000000000ull, 0x0000000200000000ull,
};

// Computes t^exponent mod phi. phi is sparse and its second-highest term is
// far below the leading one, so the reduction folds 64 bits at a time.
static Gf2Poly gf2PowT(const Gf2Poly& phi, int deg, uint64_t exponent) {
    std::vector<int> terms;  // exponents of phi below the leading term
    for (int k = 0; k < deg; ++k) {
        if (gf2Bit(phi, k)) terms.push_back(k);
    }
    if (terms.empty() || deg - terms.back() < 64) {
        fprintf(stderr, "Unsupported characteristic polynomial\n");
        exit(1);
    }
    const size_t pw = static_cast<size_t>(deg) / 64 + 1;

    auto get64 = [](const Gf2Poly& a, size_t pos) -> uint64_t {
        size_t w = pos >> 6, b = pos & 63;
        uint64_t lo = a[w] >> b;
        if (b && w + 1 < a.size()) lo |= a[w + 1] << (64 - b);
        return lo;
    };
    auto xorAt = [](Gf2Poly& a, size_t pos, uint64_t v) {
        size_t w = pos >> 6, b = pos & 63;
        a[w] ^= v << b;
        if (b && w + 1 < a.size()) a[w + 1] ^= v >> (64 - b);
    };
    // Reduce a polynomial of degree <= topBit modulo phi
    auto reduce = [&](Gf2Poly& a, size_t topBit) {
        size_t hi = topBit;
        while (hi >= static_cast<size_t>(deg)) {
            const size_t lo = std::max<size_t>(deg, hi >= 63 ? hi - 63 : 0);
            const size_t cnt = hi - lo + 1;
            uint64_t v = get64(a, lo);
            if (cnt < 64) v &= (uint64_t(1) << cnt) - 1;
            if (v) {
                xorAt(a, lo, v);  // clear the chunk
                const size_t base = lo - deg;
                for (int e : terms) xorAt(a, base + e, v);
            }
            if (lo == static_cast<size_t>(deg)) break;
            hi = lo - 1;
        }
    };
    auto spread = [](uint32_t v32) {
        uint64_t v = v32;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
        v = (v | (v << 2)) & 0x3333333333333333ull;
        v = (v | (v << 1)) & 0x5555555555555555ull;
        return v;
    };

    Gf2Poly g(2 * pw + 2, 0), sq(2 * pw + 2, 0);
    g[0] = 1;
    for (int bit = 63; bit >= 0; --bit) {
        // g = g^2 mod phi
        std::fill(sq.begin(), sq.end(), 0);
        for (size_t w = 0; w < pw; ++w) {
            sq[2 * w] = spread(static_cast<uint32_t>(g[w]));
            sq[2 * w + 1] = spread(static_cast<uint32_t>(g[w] >> 32));
        }
        reduce(sq, 2 * static_cast<size_t>(deg));
        std::swap(g, sq);
        if ((exponent >> bit) & 1u) {
            // g = g * t mod phi
            for (size_t w = g.size() - 1; w > 0; --w) g[w] = (g[w] << 1) | (g[w - 1] >> 63);
            g[0] <<= 1;
            reduce(g, static_cast<size_t>(deg));
        }
    }
    g.resize(pw);
    return g;
}

// out = state advanced by the jump polynomial g (i.e. by the exponent used to compute g)
static void mtJump(const uint32_t* in, uint32_t* out, const Gf2Poly& g, int deg) {
    uint32_t cur[MT_N], acc[MT_N] = {};
    std::memcpy(cur, in, sizeof(cur));
    int p = 0;
    for (int i = 0; i < deg; ++i) {
        if (gf2Bit(g, i)) {
            for (int m = 0; m < MT_N - p; ++m) acc[m] ^= cur[p + m];
            for (int m = MT_N - p; m < MT_N; ++m) acc[m] ^= cur[p + m - MT_N];
        }
        int p1 = p + 1 == MT_N ? 0 : p + 1;
        int pm = p + MT_M < MT_N ? p + MT_M : p + MT_M - MT_N;
        uint32_t y = (cur[p] & MT_UPPER) | (cur[p1] & MT_LOWER);
        cur[p] = cur[pm] ^ (y >> 1) ^ ((0u - (y & 1u)) & MT_MATRIX_A);
        p = p1;
    }
    std::memcpy(out, acc, sizeof(acc));
}

// Host-side producer of MT state snapshots (runs concurrently with the GPU).
// Snapshot s is the generator state after s * twistsPerSnap twists. Snapshots
// are grouped into fixed-size blocks that are distributed round-robin over
// the producer threads (the GPUs consume them in order). A thread reaches its
// next block by an exact jump-ahead over the blocks of the other threads.
// Snapshots live in a ring buffer; producers wait for consumed space.
class SnapshotProducer {
public:
    static constexpr uint64_t MAX_BLOCK_TWISTS = uint64_t(1) << 16;

    // batchRanges: snapshot range [first, second) of every consumer batch (non-decreasing).
    // blockTwists and twistsPerSnap are powers of two, twistsPerSnap <= blockTwists.
    SnapshotProducer(size_t numSnaps, int twistsPerSnap, uint64_t blockTwists, int numThreads,
                     std::vector<std::pair<size_t, size_t>> batchRanges)
        : numSnaps_(numSnaps), twistsPerSnap_(twistsPerSnap), blockTwists_(blockTwists),
          batchRanges_(std::move(batchRanges)), batchCopied_(batchRanges_.size(), 0) {
        blockSnaps_ = static_cast<size_t>(blockTwists_ / twistsPerSnap);
        numBlocks_ = (numSnaps + blockSnaps_ - 1) / blockSnaps_;
        numThreads_ = static_cast<int>(std::min<size_t>(std::max(numThreads, 1), std::max<size_t>(numBlocks_, 1)));
        const size_t ringBlocks = std::max<size_t>(2 * numThreads_ + 2, (size_t(16384) + blockSnaps_ - 1) / blockSnaps_);
        capacity_ = std::min(ringBlocks * blockSnaps_, std::max<size_t>(numBlocks_, 1) * blockSnaps_);
        ring_.reset(new uint32_t[capacity_ * MT_N]);
        progress_ = std::make_unique<std::atomic<size_t>[]>(std::max<size_t>(numBlocks_, 1));
        for (size_t b = 0; b < numBlocks_; ++b) progress_[b].store(0);
        phi_.assign(MT_CHAR_POLY, MT_CHAR_POLY + 312);
        if (numThreads_ > 1) {
            // Jump over (numThreads - 1) blocks, needed after each block
            workers_.emplace_back([this] {
                skipPoly_ = gf2PowT(phi_, MT_CHAR_POLY_DEGREE, (numThreads_ - 1) * blockTwists_ * MT_N);
                skipReady_.store(1, std::memory_order_release);
            });
        }
        for (int t = 0; t < numThreads_; ++t) workers_.emplace_back([this, t] { run(t); });
    }
    ~SnapshotProducer() {
        released_.store(numSnaps_ + capacity_, std::memory_order_release);  // unblock producers
        for (auto& w : workers_) w.join();
    }
    // Copy the snapshots of a batch to dst (waiting for them). Ring space is
    // released up to the first batch that has not been copied yet.
    void copy(size_t batch, uint32_t* dst) {
        const size_t begin = batchRanges_[batch].first, end = batchRanges_[batch].second;
        for (size_t b = begin / blockSnaps_; b < numBlocks_ && b * blockSnaps_ < end; ++b) {
            const size_t need = std::min({end, numSnaps_, (b + 1) * blockSnaps_}) - b * blockSnaps_;
            while (progress_[b].load(std::memory_order_acquire) < need) std::this_thread::yield();
        }
        for (size_t s = begin; s < end;) {
            const size_t pos = s % capacity_;
            const size_t cnt = std::min(end - s, capacity_ - pos);
            std::memcpy(dst + (s - begin) * MT_N, ring_.get() + pos * MT_N, cnt * MT_N * sizeof(uint32_t));
            s += cnt;
        }
        std::lock_guard<std::mutex> lock(releaseMutex_);
        batchCopied_[batch] = 1;
        while (firstUncopied_ < batchRanges_.size() && batchCopied_[firstUncopied_]) ++firstUncopied_;
        released_.store(firstUncopied_ < batchRanges_.size() ? batchRanges_[firstUncopied_].first : numSnaps_,
                        std::memory_order_release);
    }

private:
    template <typename T>
    static void spinUntil(const std::atomic<T>& flag) {
        while (!flag.load(std::memory_order_acquire)) std::this_thread::yield();
    }

    void run(int t) {
        alignas(64) uint32_t x[MT_N];
        mtSeed(x, 42);
        if (t > 0) {
            // Start of block t: jump directly from the seeded state
            const Gf2Poly startPoly = gf2PowT(phi_, MT_CHAR_POLY_DEGREE, t * blockTwists_ * MT_N);
            uint32_t start[MT_N];
            mtJump(x, start, startPoly, MT_CHAR_POLY_DEGREE);
            std::memcpy(x, start, sizeof(x));
        }

        for (size_t b = t; b < numBlocks_; b += numThreads_) {
            const size_t lo = b * blockSnaps_, hi = std::min(numSnaps_, lo + blockSnaps_);
            // Wait until the ring slots of this block have been consumed
            while (lo + blockSnaps_ > released_.load(std::memory_order_acquire) + capacity_) {
                std::this_thread::yield();
            }
            for (size_t s = lo; s < hi; ++s) {
                std::memcpy(ring_.get() + (s % capacity_) * MT_N, x, sizeof(x));
                progress_[b].store(s - lo + 1, std::memory_order_release);
                for (int r = 0; r < twistsPerSnap_; ++r) mtTwist(x);
            }
            // x is now the start of block b + 1; skip the other threads' blocks
            if (b + numThreads_ < numBlocks_ && numThreads_ > 1) {
                spinUntil(skipReady_);
                uint32_t next[MT_N];
                mtJump(x, next, skipPoly_, MT_CHAR_POLY_DEGREE);
                std::memcpy(x, next, sizeof(x));
            }
        }
    }

    size_t numSnaps_;
    int twistsPerSnap_;
    uint64_t blockTwists_;
    size_t blockSnaps_ = 1, numBlocks_ = 0, capacity_ = 1;
    int numThreads_ = 1;
    std::unique_ptr<uint32_t[]> ring_;
    std::unique_ptr<std::atomic<size_t>[]> progress_;
    std::vector<std::pair<size_t, size_t>> batchRanges_;
    std::vector<char> batchCopied_;
    size_t firstUncopied_ = 0;
    std::mutex releaseMutex_;
    std::atomic<size_t> released_{0};
    Gf2Poly phi_, skipPoly_;
    std::atomic<int> skipReady_{0};
    std::vector<std::thread> workers_;
};

// ============================================================================
// GPU Geometry Data (flattened octree)
// ============================================================================

struct GpuNode {
    float4 c;   // center.xyz, halfExtent.x
    float4 h;   // halfExtent.y, halfExtent.z, (int) first, (int) count
                // inner node: children are nodes [first, first + count)
                // leaf:       triangles are leaf entries [first, first - count), count < 0
};

struct GpuGeometry {
    std::vector<GpuNode> nodes;    // breadth-first: the children of a node are contiguous
    std::vector<float4> leafTris;  // per leaf entry: a (w = tri index), b - a, c - a
    int maxDepth = 0;
};

static float hostIntAsFloat(int v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

static void flattenOctree(const Octree& root, const std::vector<Triangle>& tris, GpuGeometry& geo) {
    std::vector<std::pair<const Octree*, int>> order{{&root, 0}};
    order.reserve(4 * tris.size());
    geo.nodes.clear();
    geo.leafTris.clear();
    geo.nodes.reserve(4 * tris.size());
    geo.leafTris.reserve(24 * tris.size());
    geo.maxDepth = 0;
    for (size_t q = 0; q < order.size(); ++q) {
        const Octree* node = order[q].first;
        const int depth = order[q].second;
        geo.maxDepth = std::max(geo.maxDepth, depth);
        int first, count;
        if (!node->triangleIndices.empty()) {
            first = static_cast<int>(geo.leafTris.size() / 3);
            count = -static_cast<int>(node->triangleIndices.size());
            for (size_t idx : node->triangleIndices) {
                const Triangle& t = tris[idx];
                Vec3 e1 = t.b - t.a, e2 = t.c - t.a;
                geo.leafTris.push_back(make_float4(t.a.x, t.a.y, t.a.z, hostIntAsFloat(static_cast<int>(idx))));
                geo.leafTris.push_back(make_float4(e1.x, e1.y, e1.z, 0.f));
                geo.leafTris.push_back(make_float4(e2.x, e2.y, e2.z, 0.f));
            }
        } else {
            first = static_cast<int>(order.size());
            count = 0;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    order.emplace_back(node->children[i].get(), depth + 1);
                    ++count;
                }
            }
        }
        GpuNode g;
        g.c = make_float4(node->center.x, node->center.y, node->center.z, node->halfExtent.x);
        g.h = make_float4(node->halfExtent.y, node->halfExtent.z, hostIntAsFloat(first), hostIntAsFloat(count));
        geo.nodes.push_back(g);
    }
}

// ============================================================================
// Device Geometry Routines (same math as the sequential version)
// ============================================================================

// The host reference build (gcc, -O3 -march=native, GNU mode) contracts
// "a * b + c" patterns into FMAs. The helpers below reproduce exactly that
// rounding so that device results match the reference bit for bit; device
// code is compiled with -fmad=false so nothing else gets contracted.

__device__ __forceinline__ float3 f3(float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y), __fsub_rn(a.z, b.z));
}
__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y), __fadd_rn(a.z, b.z));
}
__device__ __forceinline__ float3 mul3(float3 a, float s) {
    return make_float3(__fmul_rn(a.x, s), __fmul_rn(a.y, s), __fmul_rn(a.z, s));
}
// x*o.x + y*o.y + z*o.z
__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return __fmaf_rn(a.z, b.z, __fmaf_rn(a.x, b.x, __fmul_rn(a.y, b.y)));
}
// {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x}
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(__fmaf_rn(a.y, b.z, -__fmul_rn(a.z, b.y)),
                       __fmaf_rn(a.z, b.x, -__fmul_rn(a.x, b.z)),
                       __fmaf_rn(a.x, b.y, -__fmul_rn(a.y, b.x)));
}
// a + ab*u + ac*v
__device__ __forceinline__ float3 baryPoint(float3 a, float3 ab, float3 ac, float u, float v) {
    return make_float3(__fmaf_rn(ac.x, v, __fmaf_rn(ab.x, u, a.x)),
                       __fmaf_rn(ac.y, v, __fmaf_rn(ab.y, u, a.y)),
                       __fmaf_rn(ac.z, v, __fmaf_rn(ab.z, u, a.z)));
}

// Segment p1 -> p2 as used by the box test: d = (p2 - p1) * 0.5, mid = p1 + d
struct BoxRay {
    float3 d, ad, mid;
};

__device__ __forceinline__ BoxRay makeBoxRay(float3 p1, float3 p2) {
    BoxRay r;
    r.d = mul3(sub3(p2, p1), 0.5f);
    r.ad = make_float3(fabsf(r.d.x), fabsf(r.d.y), fabsf(r.d.z));
    r.mid = add3(p1, r.d);
    return r;
}

__device__ __forceinline__ bool gpuRayIntersectsBox(const GpuNode& n, const BoxRay& r) {
    const float3 d = r.d;
    const float3 c = sub3(r.mid, make_float3(n.c.x, n.c.y, n.c.z));
    const float hx = n.c.w, hy = n.h.x, hz = n.h.y;
    const float adx = r.ad.x, ady = r.ad.y, adz = r.ad.z;

    if (fabsf(c.x) > __fadd_rn(hx, adx)) return false;
    if (fabsf(c.y) > __fadd_rn(hy, ady)) return false;
    if (fabsf(c.z) > __fadd_rn(hz, adz)) return false;

    if (fabsf(__fmaf_rn(d.y, c.z, -__fmul_rn(d.z, c.y))) >
        __fadd_rn(__fmaf_rn(hy, adz, __fmul_rn(hz, ady)), EPSILON)) return false;
    if (fabsf(__fmaf_rn(d.z, c.x, -__fmul_rn(d.x, c.z))) >
        __fadd_rn(__fmaf_rn(hz, adx, __fmul_rn(hx, adz)), EPSILON)) return false;
    if (fabsf(__fmaf_rn(d.x, c.y, -__fmul_rn(d.y, c.x))) >
        __fadd_rn(__fmaf_rn(hx, ady, __fmul_rn(hy, adx)), EPSILON)) return false;
    return true;
}

// Moller-Trumbore with precomputed edges e1 = v1 - v0, e2 = v2 - v0
__device__ __forceinline__ float gpuRayTriangleIntersect(float3 orig, float3 dir, float3 v0, float3 e1, float3 e2) {
    const float NO_HIT = 3.402823466e+38f;
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);
    if (fabsf(det) < EPSILON) return NO_HIT;

    float invDet = __frcp_rn(det);  // == 1.0f / det (both correctly rounded)
    float3 tvec = sub3(orig, v0);
    float u = __fmul_rn(dot3(tvec, pvec), invDet);
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    float3 qvec = cross3(tvec, e1);
    float v = __fmul_rn(dot3(dir, qvec), invDet);
    if (v < 0.0f || __fadd_rn(u, v) > 1.0f) return NO_HIT;

    return __fmul_rn(dot3(e2, qvec), invDet);
}

// Traversal stack: every expansion pops one node and pushes at most 7 more
// than it pops, so depth * 7 + 1 entries suffice (checked on the host).
constexpr int OCTREE_STACK = 64;
constexpr int KIJ_THREADS_MAX = 256;  // block size of the callers of gpuWarpIsRayBlocked

// Warp-synchronous ("packet") any-hit query: each lane owns one ray and the
// warp walks the octree together, carrying the mask of lanes whose ray passed
// every box test on the path. A lane therefore tests exactly the leaves the
// sequential per-ray traversal would visit; only the visiting order differs,
// which does not affect the boolean result. Must be called by the full warp.
__device__ bool gpuWarpIsRayBlocked(bool participate, float3 from, float3 to,
                                    const GpuNode* __restrict__ nodes, const float4* __restrict__ leafTris,
                                    int srcIdx, int dstIdx) {
    const unsigned laneBit = 1u << (threadIdx.x & 31);
    float3 dir = sub3(to, from);
    float rayLen = __fsqrt_rn(dot3(dir, dir));
    bool blocked = participate && rayLen < EPSILON;
    // Kept in shared memory: under register pressure the compiler would
    // otherwise recompute these divisions inside the triangle loop
    __shared__ float3 dirNormShared[KIJ_THREADS_MAX];
    dirNormShared[threadIdx.x] =
        make_float3(__fdiv_rn(dir.x, rayLen), __fdiv_rn(dir.y, rayLen), __fdiv_rn(dir.z, rayLen));
    const float maxDist = __fsub_rn(rayLen, EPSILON);
    const BoxRay boxRay = makeBoxRay(from, to);

    unsigned alive = __ballot_sync(0xffffffffu, participate && !blocked);

    auto testLeaf = [&](const GpuNode& leaf, unsigned laneMask) {
        if ((laneMask & laneBit) && !blocked) {
            const int first = __float_as_int(leaf.h.z);
            const int count = -__float_as_int(leaf.h.w);
            const float4* tp = leafTris + 3 * static_cast<size_t>(first);
            for (int t = 0; t < count; ++t, tp += 3) {
                float4 a = __ldg(tp);
                int idx = __float_as_int(a.w);
                if (idx == srcIdx || idx == dstIdx) continue;
                float dist = gpuRayTriangleIntersect(from, dirNormShared[threadIdx.x], f3(a), f3(__ldg(tp + 1)),
                                                     f3(__ldg(tp + 2)));
                if (dist > EPSILON && dist < maxDist) {
                    blocked = true;
                    break;
                }
            }
        }
        alive &= ~__ballot_sync(0xffffffffu, blocked);
    };

    // The root itself is never box-tested (as in the sequential version)
    const GpuNode root = nodes[0];
    if (__float_as_int(root.h.w) < 0) {
        if (alive) testLeaf(root, alive);
        return blocked;
    }

    int stackNode[OCTREE_STACK];
    unsigned stackMask[OCTREE_STACK];
    int sp = 0;
    stackNode[0] = 0;
    stackMask[0] = alive;
    sp = alive ? 1 : 0;
    while (sp > 0) {
        --sp;
        const unsigned mask = stackMask[sp] & alive;
        if (!mask) continue;
        const GpuNode nd = nodes[stackNode[sp]];
        const int first = __float_as_int(nd.h.z);
        const int count = __float_as_int(nd.h.w);
        for (int c = 0; c < count; ++c) {
            const GpuNode child = nodes[first + c];
            const bool hit = (mask & alive & laneBit) && gpuRayIntersectsBox(child, boxRay);
            const unsigned hitMask = __ballot_sync(0xffffffffu, hit);
            if (!hitMask) continue;
            if (__float_as_int(child.h.w) < 0) {
                testLeaf(child, hitMask);
                if (!alive) return blocked;
            } else {
                stackNode[sp] = first + c;
                stackMask[sp] = hitMask;
                ++sp;
            }
        }
    }
    return blocked;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

constexpr int KIJ_TILE = 128;

// Pair (i, j) consumes random numbers iff i != j and the normals are not
// (nearly) parallel. Must be evaluated identically everywhere.
__device__ __forceinline__ bool pairActive(const float4* __restrict__ normals, int i, int j) {
    if (i == j) return false;
    float4 a = __ldg(normals + i);
    float4 b = __ldg(normals + j);
    float d = dot3(f3(a), f3(b));
    return !(d > 0.99f);
}

// Count active pairs per (row, tile)
__global__ void countActiveKernel(const float4* __restrict__ normals, int n, int numTiles,
                                  uint32_t* __restrict__ tileCounts) {
    const int j = blockIdx.x * KIJ_TILE + threadIdx.x;
    for (int i = blockIdx.y; i < n; i += gridDim.y) {
        bool active = j < n && pairActive(normals, i, j);
        int cnt = __syncthreads_count(active);
        if (threadIdx.x == 0) tileCounts[static_cast<size_t>(i) * numTiles + blockIdx.x] = cnt;
    }
}

// A block handles one row i and a tile of KIJ_TILE columns j. The active pairs
// of the tile are compacted, then each half-warp traces the NUM_RAYS rays of one
// pair (one ray per lane): rays of the same pair follow nearly identical paths
// through the octree, which keeps the warps coherent.
constexpr int KIJ_THREADS = 256;
static_assert(KIJ_THREADS <= KIJ_THREADS_MAX, "shared ray directions");
static_assert(NUM_RAYS == 16, "one half-warp per pair");

__global__ void __launch_bounds__(KIJ_THREADS, 4)
formFactorKernel(int n, int rowBegin, int numTiles,
                 const float4* __restrict__ tris,      // 4 per triangle: a, b, c, normal
                 const float4* __restrict__ normals,
                 const float* __restrict__ areas,
                 const GpuNode* __restrict__ nodes, const float4* __restrict__ leafTris,
                 const uint64_t* __restrict__ tileRank, uint64_t batchRankBegin,
                 const float* __restrict__ randoms,
                 float* __restrict__ weights,           // row-major rows [rowBegin, ...) x n
                 unsigned long long* __restrict__ nonZeroCount) {
    __shared__ int pairJ[KIJ_TILE];
    __shared__ int warpCounts[KIJ_TILE / 32];
    __shared__ unsigned blockNonZero;
    __shared__ int nextPair;  // work distribution among the warps of the block
    const int i = rowBegin + blockIdx.y;
    const int tileStart = blockIdx.x * KIJ_TILE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* outRow = weights + static_cast<size_t>(blockIdx.y) * n;

    // Compact the active pairs of this tile (in j order => consecutive ranks)
    bool active = false;
    unsigned ballot = 0;
    if (tid < KIJ_TILE) {
        const int j = tileStart + tid;
        active = j < n && pairActive(normals, i, j);
        ballot = __ballot_sync(0xffffffffu, active);
        if (lane == 0) warpCounts[warp] = __popc(ballot);
        if (!active && j < n) outRow[j] = 0.0f;
    }
    if (tid == 0) {
        blockNonZero = 0;
        nextPair = 0;
    }
    __syncthreads();
    int activeCount = 0;
#pragma unroll
    for (int w = 0; w < KIJ_TILE / 32; ++w) activeCount += warpCounts[w];
    if (active) {
        int local = __popc(ballot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w) local += warpCounts[w];
        pairJ[local] = tileStart + tid;
    }
    __syncthreads();

    const uint64_t tileBase = tileRank[static_cast<size_t>(i) * numTiles + blockIdx.x] - batchRankBegin;
    const int half = lane >> 4, ray = lane & 15;
    const float4* ti = tris + 4 * static_cast<size_t>(i);
    const float3 aI = f3(ti[0]), abI = sub3(f3(ti[1]), aI), acI = sub3(f3(ti[2]), aI), nI = f3(ti[3]);
    const float4* rnd4 = reinterpret_cast<const float4*>(randoms);

    for (;;) {
        // Each warp grabs the next two pairs
        int q0 = 0;
        if (lane == 0) q0 = atomicAdd(&nextPair, 2);
        q0 = __shfl_sync(0xffffffffu, q0, 0);
        if (q0 >= activeCount) break;
        const int q = q0 + half;
        const bool valid = q < activeCount;
        float term = 0.0f;
        int j = -1;
        float3 pI = make_float3(0.f, 0.f, 0.f), pJ = pI, nJ = pI;
        if (valid) {
            j = pairJ[q];
            const float4 rq = rnd4[(tileBase + q) * NUM_RAYS + ray];  // uI, vI, uJ, vJ
            const float4* tj = tris + 4 * static_cast<size_t>(j);
            const float3 aJ = f3(tj[0]), abJ = sub3(f3(tj[1]), aJ), acJ = sub3(f3(tj[2]), aJ);
            nJ = f3(tj[3]);

            float u = rq.x, v = rq.y;
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            pI = baryPoint(aI, abI, acI, u, v);
            u = rq.z; v = rq.w;
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            pJ = baryPoint(aJ, abJ, acJ, u, v);
        }
        const bool blocked = gpuWarpIsRayBlocked(valid, pI, pJ, nodes, leafTris, i, j);
        if (valid && !blocked) {
            const float3 dv = sub3(pJ, pI);
            const float distSqr = dot3(dv, dv);
            const float vNorm = sqrtf(distSqr);
            if (distSqr >= EPSILON && vNorm > EPSILON) {
                const float cosPhiI = fmaxf(0.0f, dot3(dv, nI) / vNorm);
                const float cosPhiJ = fmaxf(0.0f, dot3(make_float3(-dv.x, -dv.y, -dv.z), nJ) / vNorm);
                if (cosPhiI > 0.0f && cosPhiJ > 0.0f) term = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }
        // Accumulate in the original ray order (adding +0 for skipped rays is exact)
        float kij = 0.0f;
#pragma unroll
        for (int r = 0; r < NUM_RAYS; ++r) kij += __shfl_sync(0xffffffffu, term, (half << 4) + r);
        kij *= INV_NUM_RAYS;
        if (valid && ray == 0) {
            // Effective coupling used by the simulation: min(kij * area_j, 1), 0 if kij <= 0
            outRow[j] = kij > 0.0f ? fminf(kij * areas[j], ONE) : 0.0f;
            if (kij > EPSILON) atomicAdd(&blockNonZero, 1u);
        }
    }
    __syncthreads();
    if (tid == 0 && blockNonZero) atomicAdd(nonZeroCount, static_cast<unsigned long long>(blockNonZero));
}

// ============================================================================
// Simulation Coefficients
// ============================================================================

// Store the rows (receivers) of a batch, computed row-major as W[r][j], into
// the transposed coefficient matrix of the GPU: coef[j * pitch + col0 + r],
// so that the simulation reads them coalesced.
__global__ void transposeWeightsKernel(int n, int rows, const float* __restrict__ weights, float* __restrict__ coef,
                                       size_t pitch, int col0) {
    __shared__ float tile[32][33];
    int j = blockIdx.x * 32 + threadIdx.x;
    int r = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        if (j < n && r + k < rows) tile[threadIdx.y + k][threadIdx.x] = weights[static_cast<size_t>(r + k) * n + j];
    }
    __syncthreads();
    r = blockIdx.y * 32 + threadIdx.x;
    j = blockIdx.x * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        if (r < rows && j + k < n) coef[static_cast<size_t>(j + k) * pitch + col0 + r] = tile[threadIdx.x][threadIdx.y + k];
    }
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source
    uint64_t nonZeroKij = 0;        // Number of form factors > EPSILON

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Per-GPU data. Every GPU computes the form-factor rows of a subset of the
    // receivers and then simulates exactly those receivers.
    struct Device {
        int id;
        cudaStream_t stream;     // form factors / simulation
        cudaStream_t genStream;  // random-number generation (overlaps the form factors)
        float4* dTris = nullptr;    // a, b, c, normal per triangle
        float4* dNormals = nullptr;
        float4* dCenters = nullptr;
        float* dAreas = nullptr;
        float* dRho = nullptr;
        GpuNode* dNodes = nullptr;
        float4* dLeafTris = nullptr;
        std::vector<int> rows;      // owned receivers (global indices, local order)
        int* dRows = nullptr;
        float* dSimCoef = nullptr;  // [j * coefPitch + local row]: min(kij * area_j, 1) (0 if kij <= 0)
        size_t coefPitch = 0;       // row capacity of dSimCoef
        float* dRadB = nullptr;     // full T x N radiosity history (replicated)
        float* dOwnNew = nullptr;   // new radiosity of the owned receivers (one step)
        float* dAllNew = nullptr;   // new radiosity of all receivers, in gather order
        int* dGatherRows = nullptr; // global index of every gather position
    };
    std::vector<Device> devices;

    // Temporary buffers released after the timed phases (device id, pointer)
    std::vector<std::pair<int, void*>> deferredDeviceFree;
    float* dDistances = nullptr;    // GPU 0

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Initialize the CUDA contexts of all GPUs up front
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }
    state.devices.resize(numDevices);
    for (int d = 0; d < numDevices; ++d) {
        state.devices[d].id = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaStreamCreateWithFlags(&state.devices[d].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&state.devices[d].genStream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaSetDevice(0));

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

// Upload geometry and per-triangle data to the GPU
static void uploadGeometry(SimulationState& state) {
    const size_t n = state.numTriangles;
    std::vector<float4> tris(4 * n), normals(n), centers(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 nr = t.normal(), c = t.center();
        tris[4 * i + 0] = make_float4(t.a.x, t.a.y, t.a.z, 0.f);
        tris[4 * i + 1] = make_float4(t.b.x, t.b.y, t.b.z, 0.f);
        tris[4 * i + 2] = make_float4(t.c.x, t.c.y, t.c.z, 0.f);
        tris[4 * i + 3] = make_float4(nr.x, nr.y, nr.z, 0.f);
        normals[i] = tris[4 * i + 3];
        centers[i] = make_float4(c.x, c.y, c.z, 0.f);
    }
    GpuGeometry geo;
    flattenOctree(state.octree, state.triangles, geo);
    if (geo.maxDepth * 7 + 1 > OCTREE_STACK) {
        fprintf(stderr, "Octree too deep for the GPU traversal stack (depth %d)\n", geo.maxDepth);
        exit(1);
    }

    auto upload = [&](SimulationState::Device& dev) {
        CUDA_CHECK(cudaSetDevice(dev.id));
        dev.dTris = deviceAlloc<float4>(tris.size());
        dev.dNormals = deviceAlloc<float4>(n);
        dev.dAreas = deviceAlloc<float>(n);
        dev.dCenters = deviceAlloc<float4>(n);
        dev.dRho = deviceAlloc<float>(n);
        dev.dNodes = deviceAlloc<GpuNode>(geo.nodes.size());
        dev.dLeafTris = deviceAlloc<float4>(geo.leafTris.size());
        CUDA_CHECK(cudaMemcpy(dev.dTris, tris.data(), tris.size() * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dNormals, normals.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dAreas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dCenters, centers.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dRho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dNodes, geo.nodes.data(), geo.nodes.size() * sizeof(GpuNode),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.dLeafTris, geo.leafTris.data(), geo.leafTris.size() * sizeof(float4),
                              cudaMemcpyHostToDevice));
    };
    std::vector<std::thread> uploaders;
    for (auto& dev : state.devices) uploaders.emplace_back(upload, std::ref(dev));
    for (auto& t : uploaders) t.join();
    CUDA_CHECK(cudaSetDevice(0));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

static std::mutex deferredFreeMutex;

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int numTiles = (n + KIJ_TILE - 1) / KIJ_TILE;
    const int numDevices = static_cast<int>(state.devices.size());
    const SimulationState::Device& dev0 = state.devices[0];

    // 1. Rank of every (row, tile) among active pairs => offsets into the RNG stream
    CUDA_CHECK(cudaSetDevice(0));
    uint32_t* dTileCounts = deviceAlloc<uint32_t>(static_cast<size_t>(n) * numTiles);
    countActiveKernel<<<dim3(numTiles, std::min(n, 65535)), KIJ_TILE>>>(dev0.dNormals, n, numTiles, dTileCounts);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> tileCounts(static_cast<size_t>(n) * numTiles);
    CUDA_CHECK(cudaMemcpy(tileCounts.data(), dTileCounts, tileCounts.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    state.deferredDeviceFree.emplace_back(0, dTileCounts);

    std::vector<uint64_t> tileRank(static_cast<size_t>(n) * numTiles + 1);
    uint64_t acc = 0;
    for (size_t k = 0; k < tileCounts.size(); ++k) {
        tileRank[k] = acc;
        acc += tileCounts[k];
    }
    tileRank.back() = acc;
    const uint64_t totalPairs = acc;
    auto rowRank = [&](int i) { return tileRank[static_cast<size_t>(i) * numTiles]; };
    auto rowRankAt = [&](int i) { return i < n ? rowRank(i) : totalPairs; };

    // 2. Random-number stream layout: snapshot spacing is a power of two
    //    (dividing the producer block size) such that a full batch is
    //    regenerated by >= ~256 parallel GPU blocks
    const uint64_t totalWords = totalPairs * RANDS_PER_PAIR;
    const uint64_t totalTwists = (totalWords + MT_N - 1) / MT_N;
    const uint64_t rowWords = static_cast<uint64_t>(n) * RANDS_PER_PAIR;
    const uint64_t maxBatchWords = std::max<uint64_t>(rowWords, uint64_t(1) << 25);
    const uint64_t targetTwists = std::max<uint64_t>(16, maxBatchWords / (uint64_t(MT_N) * 256));
    uint64_t twistsPow2 = 16;
    while (twistsPow2 * 2 <= targetTwists && twistsPow2 * 2 <= SnapshotProducer::MAX_BLOCK_TWISTS) twistsPow2 <<= 1;
    const int twistsPerSnap = static_cast<int>(twistsPow2);
    const uint64_t wordsPerSnap = static_cast<uint64_t>(twistsPerSnap) * MT_N;
    const size_t numSnaps = static_cast<size_t>((totalTwists + twistsPerSnap - 1) / twistsPerSnap);

    // 3. Row batches (bounded random-number buffer), dynamically scheduled over all
    //    GPUs; batches shrink towards the end (guided scheduling) to balance the
    //    tail and grow during the first rounds so that the GPUs start early.
    const uint64_t minBatchWords = std::max<uint64_t>(rowWords, uint64_t(1) << 21);
    struct Batch {
        int rowBegin, rowEnd;
        uint64_t rankBegin, wordBegin, wordEnd;
        size_t snapBegin, snapEnd;
    };
    std::vector<Batch> batches;
    std::vector<std::pair<size_t, size_t>> batchSnaps;
    for (int rowBegin = 0; rowBegin < n;) {
        const uint64_t remaining = (totalPairs - rowRank(rowBegin)) * RANDS_PER_PAIR;
        const size_t round = batches.size() / numDevices;
        const uint64_t rampWords = round < 4 ? (uint64_t(1) << 21) << round : maxBatchWords;
        const uint64_t upper = std::min(maxBatchWords, std::max(rampWords, minBatchWords));
        const uint64_t target =
            std::clamp<uint64_t>(remaining / (static_cast<uint64_t>(numDevices) * 4), minBatchWords, upper);
        int rowEnd = rowBegin + 1;
        while (rowEnd < n && (rowRankAt(rowEnd + 1) - rowRank(rowBegin)) * RANDS_PER_PAIR <= target) {
            ++rowEnd;
        }
        Batch b;
        b.rowBegin = rowBegin;
        b.rowEnd = rowEnd;
        b.rankBegin = rowRank(rowBegin);
        b.wordBegin = b.rankBegin * RANDS_PER_PAIR;
        b.wordEnd = rowRankAt(rowEnd) * RANDS_PER_PAIR;
        b.snapBegin = static_cast<size_t>(b.wordBegin / wordsPerSnap);
        b.snapEnd = b.wordEnd > b.wordBegin ? static_cast<size_t>((b.wordEnd + wordsPerSnap - 1) / wordsPerSnap)
                                            : b.snapBegin;
        batches.push_back(b);
        batchSnaps.emplace_back(b.snapBegin, b.snapEnd);
        rowBegin = rowEnd;
    }
    // Buffer sizes from the largest actual batch
    uint64_t slotWords = 1;
    size_t maxBatchSnaps = 1, maxBatchRows = 1;
    for (const Batch& b : batches) {
        maxBatchRows = std::max<size_t>(maxBatchRows, b.rowEnd - b.rowBegin);
        slotWords = std::max<uint64_t>(slotWords, b.wordEnd - b.wordBegin);
        maxBatchSnaps = std::max(maxBatchSnaps, b.snapEnd - b.snapBegin);
    }

    // 4. Produce MT snapshots on the host, concurrently with the GPU work
    //    Up to 16 threads with >= ~4 blocks each; a block is a power of two of
    //    twists between max(twistsPerSnap, 4096) and 65536.
    const int hwThreads = static_cast<int>(std::max(2u, std::thread::hardware_concurrency()));
    const int maxProducers = std::max(1, std::min(16, hwThreads / 2));
    uint64_t blockTwists = std::max<uint64_t>(twistsPerSnap, 4096);
    while (blockTwists * 2 <= SnapshotProducer::MAX_BLOCK_TWISTS && blockTwists * 2 * maxProducers * 4 <= totalTwists) {
        blockTwists <<= 1;
    }
    const int producerThreads = static_cast<int>(
        std::clamp<uint64_t>(totalTwists / (2 * blockTwists), 1, static_cast<uint64_t>(maxProducers)));
    SnapshotProducer producer(numSnaps, twistsPerSnap, blockTwists, producerThreads, batchSnaps);

    // 5. One host thread per GPU pulls batches in order. Each GPU has two
    //    buffer slots so that the random numbers of a batch are generated
    //    (genStream) while the previous batch still runs (stream). A GPU keeps
    //    the rows it computes: it simulates exactly those receivers later.
    constexpr size_t SLOTS = 2;
    std::atomic<size_t> nextBatch{0};
    std::vector<unsigned long long> nonZeroPerDevice(numDevices, 0);

    auto worker = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        SimulationState::Device& dev = state.devices[d];
        auto devAlloc = [&](auto* tag, size_t count) {
            auto* ptr = deviceAlloc<std::remove_pointer_t<decltype(tag)>>(count);
            std::lock_guard<std::mutex> lock(deferredFreeMutex);
            state.deferredDeviceFree.emplace_back(d, ptr);
            return ptr;
        };
        uint64_t* dTileRank = devAlloc(static_cast<uint64_t*>(nullptr), tileRank.size());
        float* dRandoms = devAlloc(static_cast<float*>(nullptr), SLOTS * slotWords);
        uint32_t* dSnaps = devAlloc(static_cast<uint32_t*>(nullptr), SLOTS * maxBatchSnaps * MT_N);
        float* dBatchRows = devAlloc(static_cast<float*>(nullptr), SLOTS * maxBatchRows * static_cast<size_t>(n));
        dev.rows.clear();
        // Capacity for the owned receivers: about an even share, grown on demand
        dev.coefPitch = std::min<size_t>(n, (5 * static_cast<size_t>(n)) / (4 * numDevices) + maxBatchRows);
        dev.dSimCoef = devAlloc(static_cast<float*>(nullptr), static_cast<size_t>(n) * dev.coefPitch);
        unsigned long long* dNonZero = devAlloc(static_cast<unsigned long long*>(nullptr), 1);
        std::vector<uint32_t> hSnaps(maxBatchSnaps * MT_N);
        cudaEvent_t genDone[SLOTS], ffDone[SLOTS];
        for (size_t k = 0; k < SLOTS; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&genDone[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ffDone[k], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaMemcpyAsync(dTileRank, tileRank.data(), tileRank.size() * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, dev.stream));
        CUDA_CHECK(cudaMemsetAsync(dNonZero, 0, sizeof(unsigned long long), dev.stream));

        size_t issued = 0;
        for (;; ++issued) {
            // Back-pressure: at most SLOTS batches in flight per GPU (dynamic load balance)
            const size_t slot = issued % SLOTS;
            if (issued >= SLOTS) CUDA_CHECK(cudaEventSynchronize(ffDone[slot]));
            const size_t bi = nextBatch.fetch_add(1);
            if (bi >= batches.size()) break;
            const Batch& b = batches[bi];
            float* randoms = dRandoms + slot * slotWords;
            if (b.wordEnd > b.wordBegin) {
                uint32_t* snaps = dSnaps + slot * maxBatchSnaps * MT_N;
                producer.copy(bi, hSnaps.data());
                CUDA_CHECK(cudaStreamWaitEvent(dev.genStream, ffDone[slot], 0));
                CUDA_CHECK(cudaMemcpyAsync(snaps, hSnaps.data(),
                                           (b.snapEnd - b.snapBegin) * MT_N * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice, dev.genStream));
                generateRandomsKernel<<<static_cast<unsigned>(b.snapEnd - b.snapBegin), 256, 0, dev.genStream>>>(
                    snaps, b.snapBegin, twistsPerSnap, b.wordBegin, b.wordEnd, randoms);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(genDone[slot], dev.genStream));
                CUDA_CHECK(cudaStreamWaitEvent(dev.stream, genDone[slot], 0));
            } else {
                producer.copy(bi, hSnaps.data());  // empty; keeps the release order
            }
            const int rows = b.rowEnd - b.rowBegin;
            const size_t col0 = dev.rows.size();
            if (col0 + rows > dev.coefPitch) {
                // Grow the coefficient matrix (stream-ordered copy of the columns so far)
                const size_t newPitch = std::min<size_t>(n, std::max(2 * dev.coefPitch, col0 + rows));
                float* grown = devAlloc(static_cast<float*>(nullptr), static_cast<size_t>(n) * newPitch);
                CUDA_CHECK(cudaMemcpy2DAsync(grown, newPitch * sizeof(float), dev.dSimCoef,
                                             dev.coefPitch * sizeof(float), col0 * sizeof(float), n,
                                             cudaMemcpyDeviceToDevice, dev.stream));
                dev.dSimCoef = grown;
                dev.coefPitch = newPitch;
            }
            for (int r = b.rowBegin; r < b.rowEnd; ++r) dev.rows.push_back(r);
            float* out = dBatchRows + slot * maxBatchRows * static_cast<size_t>(n);
            formFactorKernel<<<dim3(numTiles, rows), KIJ_THREADS, 0, dev.stream>>>(
                n, b.rowBegin, numTiles, dev.dTris, dev.dNormals, dev.dAreas, dev.dNodes, dev.dLeafTris, dTileRank,
                b.rankBegin, randoms, out, dNonZero);
            CUDA_CHECK(cudaGetLastError());
            transposeWeightsKernel<<<dim3((n + 31) / 32, (rows + 31) / 32), dim3(32, 8), 0, dev.stream>>>(
                n, rows, out, dev.dSimCoef, dev.coefPitch, static_cast<int>(col0));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ffDone[slot], dev.stream));
        }

        // Global indices of the owned receivers
        const int m = static_cast<int>(dev.rows.size());
        dev.dRows = devAlloc(static_cast<int*>(nullptr), dev.rows.size());
        if (m > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dev.dRows, dev.rows.data(), m * sizeof(int), cudaMemcpyHostToDevice,
                                       dev.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK(cudaMemcpy(&nonZeroPerDevice[d], dNonZero, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < SLOTS; ++k) {
            CUDA_CHECK(cudaEventDestroy(genDone[k]));
            CUDA_CHECK(cudaEventDestroy(ffDone[k]));
        }
    };
    std::vector<std::thread> workers;
    for (int d = 0; d < numDevices; ++d) workers.emplace_back(worker, d);
    for (auto& t : workers) t.join();

    for (int row = 100; row <= n; row += 100) printf("  Progress: %d/%d triangles\n", row, n);
    if (n % 100 != 0) printf("  Progress: %d/%d triangles\n", n, n);

    uint64_t nonZero = 0;
    for (unsigned long long nz : nonZeroPerDevice) nonZero += nz;
    state.nonZeroKij = nonZero;
    CUDA_CHECK(cudaSetDevice(0));
}

// Uploads the geometry (incl. triangle centers) to all GPUs. The delays
// tau_ij = ceil(|c_i - c_j| / WAVE_SPEED) are cheap to evaluate, so the
// simulation recomputes them on the fly instead of streaming an N x N matrix.
void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    uploadGeometry(state);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// A block handles SIM_RECV owned receivers (one per lane). For each tile of
// SIM_TILE emitters j, all warps compute the contributions w_ij * radB_j in
// parallel, then warp 0 accumulates them in the original sequential j order
// (a skipped term contributes +0, which leaves the sum unchanged), so results
// match the reference bit for bit.
constexpr int SIM_RECV = 32;
constexpr int SIM_WARPS = 32;
constexpr int SIM_TILE = 256;

__global__ void __launch_bounds__(SIM_RECV * SIM_WARPS)
simulationStepKernel(int n, int m, int t, const int* __restrict__ rows, const float* __restrict__ simCoef, size_t pitch,
                     const float4* __restrict__ centers, const float* __restrict__ rho, int source, int timeOff,
                     float* __restrict__ radB, float* __restrict__ ownNew) {
    __shared__ float contrib[SIM_TILE][SIM_RECV];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int li = blockIdx.x * SIM_RECV + lane;
    const bool valid = li < m;
    const int i = valid ? rows[li] : -1;
    const float3 ci = valid ? f3(centers[i]) : make_float3(0.f, 0.f, 0.f);
    float sumB = 0.0f;

    for (int tile = 0; tile < n; tile += SIM_TILE) {
#pragma unroll 4
        for (int k = 0; k < SIM_TILE / SIM_WARPS; ++k) {
            const int jl = warp * (SIM_TILE / SIM_WARPS) + k;
            const int j = tile + jl;
            float p = 0.0f;
            if (valid && j < n && j != i) {
                const float w = simCoef[static_cast<size_t>(j) * pitch + li];
                const float3 d = sub3(ci, f3(__ldg(centers + j)));
                const int tauij = static_cast<int>(ceilf(sqrtf(dot3(d, d)) * INV_WAVE_SPEED));
                if (t >= tauij && w > 0.0f) {
                    const float radJ = radB[static_cast<size_t>(t - tauij) * n + j];
                    if (radJ > 0.0f) p = __fmul_rn(w, radJ);
                }
            }
            contrib[jl][lane] = p;
        }
        __syncthreads();
        if (warp == 0) {
            const int count = min(SIM_TILE, n - tile);
#pragma unroll 16
            for (int k = 0; k < count; ++k) sumB = __fadd_rn(sumB, contrib[k][lane]);
        }
        __syncthreads();
    }
    if (warp == 0 && valid) {
        // Emission: the source is active during the first half of the timesteps
        const float radE = (i == source && t < timeOff) ? 1.0f : 0.0f;
        const float value = __fmaf_rn(rho[i], sumB, radE);
        radB[static_cast<size_t>(t) * n + i] = value;
        ownNew[li] = value;
    }
}

__global__ void scatterRadiosityKernel(int n, const int* __restrict__ gatherRows, const float* __restrict__ allNew,
                                       float* __restrict__ radBRow) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) radBRow[gatherRows[k]] = allNew[k];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int numDevices = static_cast<int>(state.devices.size());
    const int timeOff = T / 2;
    const int source = static_cast<int>(state.sourceIndex);

    // Gather order: the owned receivers of GPU 0, then GPU 1, ...
    std::vector<int> gatherRows;
    std::vector<size_t> offset(numDevices + 1, 0);
    for (int d = 0; d < numDevices; ++d) {
        offset[d] = gatherRows.size();
        gatherRows.insert(gatherRows.end(), state.devices[d].rows.begin(), state.devices[d].rows.end());
    }
    offset[numDevices] = gatherRows.size();

    // Every GPU computes its receivers for step t; the new values are then
    // exchanged through the host (the GPUs need no peer access) before step
    // t + 1. Two buffers alternate, so a buffer is only rewritten after every
    // GPU has passed the next barrier, i.e. has staged its copy of it.
    std::vector<float> exchange[2] = {std::vector<float>(n), std::vector<float>(n)};
    std::barrier stepBarrier(numDevices);

    auto worker = [&](int d) {
        CUDA_CHECK(cudaSetDevice(d));
        SimulationState::Device& dev = state.devices[d];
        const int m = static_cast<int>(dev.rows.size());
        dev.dRadB = deviceAlloc<float>(static_cast<size_t>(T) * n);
        dev.dOwnNew = deviceAlloc<float>(m);
        dev.dAllNew = deviceAlloc<float>(n);
        dev.dGatherRows = deviceAlloc<int>(n);
        CUDA_CHECK(cudaMemcpyAsync(dev.dGatherRows, gatherRows.data(), n * sizeof(int), cudaMemcpyHostToDevice,
                                   dev.stream));
        for (int t = 0; t < T; ++t) {
            std::vector<float>& buf = exchange[t & 1];
            float* radBRow = dev.dRadB + static_cast<size_t>(t) * n;
            if (m > 0) {
                simulationStepKernel<<<(m + SIM_RECV - 1) / SIM_RECV, SIM_RECV * SIM_WARPS, 0, dev.stream>>>(
                    n, m, t, dev.dRows, dev.dSimCoef, dev.coefPitch, dev.dCenters, dev.dRho, source, timeOff,
                    dev.dRadB, dev.dOwnNew);
                CUDA_CHECK(cudaGetLastError());
            }
            if (numDevices > 1) {
                if (m > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(buf.data() + offset[d], dev.dOwnNew, m * sizeof(float),
                                               cudaMemcpyDeviceToHost, dev.stream));
                    CUDA_CHECK(cudaStreamSynchronize(dev.stream));
                }
                stepBarrier.arrive_and_wait();
                CUDA_CHECK(cudaMemcpyAsync(dev.dAllNew, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice,
                                           dev.stream));
                scatterRadiosityKernel<<<(n + 255) / 256, 256, 0, dev.stream>>>(n, dev.dGatherRows, dev.dAllNew,
                                                                                radBRow);
                CUDA_CHECK(cudaGetLastError());
            }
            if (d == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) printf("  Timestep %d/%d\n", t + 1, T);
        }
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));
    };
    std::vector<std::thread> workers;
    for (int d = 0; d < numDevices; ++d) workers.emplace_back(worker, d);
    for (auto& w : workers) w.join();

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.devices[0].dRadB, state.radB.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void transposeKernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        if (x < cols && y + k < rows) tile[threadIdx.y + k][threadIdx.x] = in[static_cast<size_t>(y + k) * cols + x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        if (x < rows && y + k < cols) out[static_cast<size_t>(y + k) * rows + x] = tile[threadIdx.x][threadIdx.y + k];
    }
}

constexpr int DIST_THREADS = 128;

// One block per triangle; threads split the lags t, each summed in sequential order.
__global__ void __launch_bounds__(DIST_THREADS)
distanceKernel(int n, int T, int src, const float* __restrict__ radBT, float* __restrict__ distances) {
    __shared__ float bestVal[DIST_THREADS];
    __shared__ int bestIdx[DIST_THREADS];
    const int i = blockIdx.x;
    const float* pB = radBT + static_cast<size_t>(i) * T;
    const float* pS = radBT + static_cast<size_t>(src) * T;

    float myBest = 0.0f;
    int myT = 0;
    bool found = false;
    for (int t = threadIdx.x; t < T; t += DIST_THREADS) {
        float sum = 0.0f;
        for (int tt = t; tt < T; ++tt) sum = __fadd_rn(sum, __fmul_rn(pS[tt - t], pB[tt]));
        if (sum > 0.0f && (!found || sum > myBest)) {
            myBest = sum;
            myT = t;
            found = true;
        }
    }
    bestVal[threadIdx.x] = found ? myBest : 0.0f;
    bestIdx[threadIdx.x] = found ? myT : -1;
    __syncthreads();
    for (int s = DIST_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            int oi = bestIdx[threadIdx.x + s];
            int mi = bestIdx[threadIdx.x];
            float ov = bestVal[threadIdx.x + s];
            float mv = bestVal[threadIdx.x];
            if (oi >= 0 && (mi < 0 || ov > mv || (ov == mv && oi < mi))) {
                bestVal[threadIdx.x] = ov;
                bestIdx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int bestT = bestIdx[0] < 0 ? 0 : bestIdx[0];
        distances[i] = WAVE_SPEED * static_cast<float>(bestT);
    }
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    state.dDistances = deviceAlloc<float>(n);
    float* dRadBT = deviceAlloc<float>(static_cast<size_t>(n) * T);
    if (T > 0) {
        transposeKernel<<<dim3((n + 31) / 32, (T + 31) / 32), dim3(32, 8)>>>(state.devices[0].dRadB, dRadBT, T, n);
        CUDA_CHECK(cudaGetLastError());
    }
    distanceKernel<<<n, DIST_THREADS>>>(n, T, static_cast<int>(state.sourceIndex), dRadBT, state.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, n * sizeof(float), cudaMemcpyDeviceToHost));
    state.deferredDeviceFree.emplace_back(0, dRadBT);
}

// Release all GPU resources (after the timed phases)
void releaseDeviceResources(SimulationState& state) {
    for (auto& [dev, ptr] : state.deferredDeviceFree) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(ptr));
    }
    state.deferredDeviceFree.clear();
    for (auto& d : state.devices) {
        CUDA_CHECK(cudaSetDevice(d.id));
        for (void* ptr : {static_cast<void*>(d.dTris), static_cast<void*>(d.dNormals),
                          static_cast<void*>(d.dCenters), static_cast<void*>(d.dAreas), static_cast<void*>(d.dRho),
                          static_cast<void*>(d.dNodes), static_cast<void*>(d.dLeafTris),
                          static_cast<void*>(d.dRadB), static_cast<void*>(d.dOwnNew), static_cast<void*>(d.dAllNew),
                          static_cast<void*>(d.dGatherRows)}) {
            CUDA_CHECK(cudaFree(ptr));
        }
        CUDA_CHECK(cudaStreamDestroy(d.stream));
        CUDA_CHECK(cudaStreamDestroy(d.genStream));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(state.dDistances));
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

    // Check Kij matrix (should have some non-zero entries; counted on the GPU)
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

    releaseDeviceResources(state);

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

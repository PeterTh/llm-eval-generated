/**
 * Room Response Simulation Benchmark (CUDA-parallel implementation)
 *
 * This is a CUDA GPU-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization notes:
 * - Form factors: one GPU thread per (i,j) triangle pair, traversing a
 *   flattened octree. The Monte Carlo random number stream is generated on the
 *   host with semantics identical to the original std::mt19937 +
 *   std::uniform_real_distribution<float> sequence (each pair consumes a fixed
 *   64 floats unless culled), so sampling is bit-identical to the sequential
 *   code. Generation is double-buffered and overlapped with GPU compute.
 * - Time delays: one GPU thread per pair.
 * - Wave propagation: one kernel launch per timestep (timesteps are inherently
 *   sequential since tau >= 1), one block per receiving triangle with a
 *   parallel reduction over source triangles.
 * - Cross-correlation distances: one block per triangle, threads parallelize
 *   over candidate lags with an argmax reduction (ties -> smallest lag, as in
 *   the sequential code).
 */

#include <algorithm>
#include <array>
#include <cfloat>
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
// CUDA Error Handling
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host-only)
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

// ============================================================================
// Flattened Octree Representation for GPU Traversal
// ============================================================================

struct DevNode {
    Vec3 center, halfExtent;
    int child[8];      // Index into node array, -1 if absent
    int triStart;      // Start index into flattened leaf triangle index array
    int triCount;      // Number of triangles (>0 identifies a leaf, as in Octree)
};

static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<int>& leafTris) {
    int idx = static_cast<int>(nodes.size());
    nodes.push_back(DevNode{});

    DevNode dn;
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    dn.triStart = 0;
    dn.triCount = 0;
    for (int i = 0; i < 8; ++i) dn.child[i] = -1;

    if (!node.triangleIndices.empty()) {
        dn.triStart = static_cast<int>(leafTris.size());
        dn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t t : node.triangleIndices) leafTris.push_back(static_cast<int>(t));
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                dn.child[i] = flattenOctree(*node.children[i], nodes, leafTris);
            }
        }
    }

    nodes[idx] = dn;
    return idx;
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

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Fast reimplementation of std::mt19937 used to generate the Monte Carlo
// stream in bulk. Verified at runtime against the standard library sequence;
// if it were ever to differ, generation falls back to RandomGenerator.
class FastMT19937 {
    static constexpr int STATE = 624;

    uint32_t mt[STATE];
    int mti;

    static inline uint32_t twistWord(uint32_t hi, uint32_t lo, uint32_t far) {
        uint32_t y = (hi & 0x80000000u) | (lo & 0x7fffffffu);
        return far ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }

    void twist() {
        for (int i = 0; i < 227; ++i) mt[i] = twistWord(mt[i], mt[i + 1], mt[i + 397]);
        for (int i = 227; i < 623; ++i) mt[i] = twistWord(mt[i], mt[i + 1], mt[i - 227]);
        mt[623] = twistWord(mt[623], mt[0], mt[396]);
        mti = 0;
    }

    // Advance one round of raw state out-of-place (cur may not alias prev).
    // Identical sequence to the in-place twist above.
    static void twistTo(const uint32_t* prev, uint32_t* cur) {
        for (int i = 0; i < 227; ++i) cur[i] = twistWord(prev[i], prev[i + 1], prev[i + 397]);
        for (int i = 227; i < 623; ++i) cur[i] = twistWord(prev[i], prev[i + 1], cur[i - 227]);
        cur[623] = twistWord(prev[623], cur[0], cur[396]);
    }

    // Temper a raw state word and convert it exactly like libstdc++
    // std::generate_canonical<float, 24, mt19937>: one 32-bit draw scaled by
    // 2^-32, clamped to nextafter(1, 0).
    static inline float convertOne(uint32_t y) {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        float f = static_cast<float>(y) * (1.0f / 4294967296.0f);
        return f >= 1.0f ? 0.99999994f : f;  // 0.99999994f = nextafterf(1, 0)
    }

    // Temper + convert raw words to canonical floats in place (vectorizable).
    static void temperConvertRange(uint32_t* buf, size_t count) {
        for (size_t q = 0; q < count; ++q) {
            float f = convertOne(buf[q]);
            uint32_t bits;
            memcpy(&bits, &f, sizeof bits);
            buf[q] = bits;
        }
    }

public:
    explicit FastMT19937(uint32_t seed) {
        mt[0] = seed;
        for (int i = 1; i < STATE; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        mti = STATE;
    }

    // Bulk-generate canonical floats. Raw state rounds are generated serially
    // (the recurrence is inherently sequential) straight into the destination
    // buffer; the independent temper + convert pass then runs multithreaded.
    void fill(float* dst, size_t n) {
        uint32_t* raw = reinterpret_cast<uint32_t*>(dst);
        size_t k = 0;

        // Leftover tempered outputs from the current round
        while (mti < STATE && k < n) {
            float f = convertOne(mt[mti++]);
            memcpy(&raw[k++], &f, sizeof(float));
        }

        // Full rounds, written raw into the destination and converted below
        size_t fullRounds = (n - k) / STATE;
        if (fullRounds > 0) {
            const size_t rawStart = k;
            twistTo(mt, raw + k);
            for (size_t r = 1; r < fullRounds; ++r) {
                twistTo(raw + k + (r - 1) * STATE, raw + k + r * STATE);
            }
            // The last generated round is the new engine state, fully consumed
            memcpy(mt, raw + k + (fullRounds - 1) * STATE, sizeof(mt));
            mti = STATE;
            k += fullRounds * STATE;

            const size_t rawCount = k - rawStart;
            unsigned hw = std::thread::hardware_concurrency();
            size_t numWorkers = std::min<size_t>(hw ? hw : 1, 16);
            if (rawCount < (1u << 20)) numWorkers = 1;
            if (numWorkers <= 1) {
                temperConvertRange(raw + rawStart, rawCount);
            } else {
                std::vector<std::thread> workers;
                size_t per = (rawCount + numWorkers - 1) / numWorkers;
                for (size_t w = 0; w < numWorkers; ++w) {
                    size_t lo = w * per;
                    size_t hi = std::min(lo + per, rawCount);
                    if (lo >= hi) break;
                    workers.emplace_back(temperConvertRange, raw + rawStart + lo, hi - lo);
                }
                for (auto& th : workers) th.join();
            }
        }

        // Tail (partial round)
        if (k < n) {
            twist();
            while (k < n) {
                float f = convertOne(mt[mti++]);
                memcpy(&raw[k++], &f, sizeof(float));
            }
        }
    }
};

// Produces the exact float sequence of RandomGenerator(seed), in bulk.
class ExactRandomStream {
    FastMT19937 fast;
    RandomGenerator slow;
    bool useFast;
public:
    explicit ExactRandomStream(uint32_t seed) : fast(seed), slow(seed) {
        // Verify the fast path reproduces the standard library sequence.
        FastMT19937 f2(seed);
        RandomGenerator s2(seed);
        std::vector<float> probe(65536);
        f2.fill(probe.data(), probe.size());
        useFast = true;
        for (size_t k = 0; k < probe.size(); ++k) {
            if (probe[k] != s2.rand()) {
                useFast = false;
                break;
            }
        }
    }

    void fill(float* dst, size_t n) {
        if (useFast) {
            fast.fill(dst, n);
        } else {
            for (size_t k = 0; k < n; ++k) dst[k] = slow.rand();
        }
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (Octree-accelerated, GPU)
// ============================================================================

// Check if a ray intersects a node's bounding box (same math as Octree::rayIntersectsBox)
__device__ bool rayIntersectsBoxDev(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > n.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > n.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > n.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.halfExtent.y * ad.z + n.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.halfExtent.z * ad.x + n.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.halfExtent.x * ad.y + n.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative traversal of the flattened octree (equivalent to Octree::applyToTris).
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevNode* nodes, const int* leafTris,
                             const Triangle* triangles,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;  // Root is visited unconditionally, as in the original

    while (sp > 0) {
        const DevNode n = nodes[stack[--sp]];

        if (n.triCount > 0) {
            // Leaf: test triangles directly
            for (int k = 0; k < n.triCount; ++k) {
                int idx = leafTris[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Interior: descend to children whose box the ray intersects
            for (int i = 0; i < 8; ++i) {
                int ci = n.child[i];
                if (ci >= 0 && rayIntersectsBoxDev(nodes[ci], from, to)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) - GPU kernel
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

constexpr uint32_t RAND_OFFSET_SKIP = 0xFFFFFFFFu;
constexpr size_t FLOATS_PER_PAIR = NUM_RAYS * 4;  // RNG floats consumed per non-culled pair

// Warp-cooperative kernel: each (i,j) pair is handled by a group of NUM_RAYS
// (16) lanes, one ray per lane, within a chunk of rows [rowStart,
// rowStart+rows). randOffsets[p] gives the pair's offset into the chunk's
// random float buffer (each active pair consumes exactly NUM_RAYS * 4 floats,
// matching the sequential consumption order of the original code), or
// RAND_OFFSET_SKIP for pairs that are culled or on the diagonal (their kij
// stays 0). Ray terms are summed in ray order via warp shuffles so the result
// matches the sequential accumulation.
__global__ void kijKernel(int rowStart, long long pairCount, int numTriangles,
                          const Triangle* __restrict__ triangles,
                          const DevNode* __restrict__ nodes,
                          const int* __restrict__ leafTris,
                          const float* __restrict__ rands,
                          const uint32_t* __restrict__ randOffsets,
                          float* __restrict__ kij) {
    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long p = tid / NUM_RAYS;   // pair index within chunk
    int lane = static_cast<int>(tid % NUM_RAYS);  // ray index

    uint32_t off = RAND_OFFSET_SKIP;
    if (p < pairCount) off = randOffsets[p];
    bool active = (off != RAND_OFFSET_SKIP);

    val_t term = ZERO;
    int i = 0, j = 0;

    if (active) {
        i = rowStart + static_cast<int>(p / numTriangles);
        j = static_cast<int>(p % numTriangles);

        const Triangle triI = triangles[i];
        const Triangle triJ = triangles[j];

        // This lane's 4 random floats (coalesced: the pair's 64 floats are
        // consecutive, ray r owns floats [off + 4r, off + 4r + 3])
        const float4 rv = reinterpret_cast<const float4*>(rands + off)[lane];

        // Random point in triangle I (matches randomPointInTriangle)
        val_t u = rv.x;
        val_t v = rv.y;
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        Vec3 pI = triI.a + (triI.b - triI.a) * u + (triI.c - triI.a) * v;

        // Random point in triangle J
        val_t u2 = rv.z;
        val_t v2 = rv.w;
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        Vec3 pJ = triJ.a + (triJ.b - triJ.a) * u2 + (triJ.c - triJ.a) * v2;

        if (!isRayBlocked(pI, pJ, nodes, leafTris, triangles, i, j)) {
            Vec3 d = pJ - pI;
            val_t distSqr = d.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosPhiI = cosPhiDev(d, triI.normal());
                val_t cosPhiJ = cosPhiDev(-d, triJ.normal());
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    term = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }
    }

    // Sum the 16 ray terms in ray order (identical rounding to the sequential
    // loop). All warp lanes participate in the shuffles.
    val_t kijSum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        kijSum += __shfl_sync(0xffffffffu, term, r, NUM_RAYS);
    }

    if (active && lane == 0) {
        kij[static_cast<size_t>(i) * numTriangles + j] = kijSum * INV_NUM_RAYS;
    }
}

// ============================================================================
// Tau (time delay) Computation - GPU kernel
// ============================================================================

__global__ void tauKernel(int numTriangles, const Triangle* __restrict__ triangles,
                          int* __restrict__ tau) {
    long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(numTriangles) * numTriangles;
    if (p >= total) return;

    int i = static_cast<int>(p / numTriangles);
    int j = static_cast<int>(p % numTriangles);
    if (i == j) {
        tau[p] = 0;
        return;
    }

    val_t dist = (triangles[i].center() - triangles[j].center()).norm();
    tau[p] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Simulation Phase (Wave Propagation) - GPU kernel, one launch per timestep
// ============================================================================

constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(int t, int numTriangles,
                              const float* __restrict__ kij,
                              const int* __restrict__ tau,
                              const float* __restrict__ areas,
                              const float* __restrict__ rho,
                              const float* __restrict__ radE,
                              float* __restrict__ radB) {
    int i = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[SIM_BLOCK];

    size_t rowBase = static_cast<size_t>(i) * numTriangles;
    float sumB = ZERO;

    for (int j = tid; j < numTriangles; j += SIM_BLOCK) {
        if (j == i) continue;

        int tauij = tau[rowBase + j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        float k = kij[rowBase + j];
        if (k <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        float radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    shared[tid] = sumB;
    __syncthreads();

    for (int s = SIM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        size_t idx = static_cast<size_t>(t) * numTriangles + i;
        // Update radiosity: reflection + emission
        radB[idx] = rho[i] * shared[0] + radE[idx];
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation) - GPU kernel
// ============================================================================

constexpr int DIST_BLOCK = 128;

__global__ void distKernel(int numTriangles, int numTimesteps, int sourceIndex,
                           const float* __restrict__ radB,
                           float* __restrict__ distances) {
    int i = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float bestCorr[DIST_BLOCK];
    __shared__ int bestLag[DIST_BLOCK];

    float maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay; each thread handles a
    // strided subset of lags, summing over tt in the original sequential order.
    for (int t = tid; t < numTimesteps; t += DIST_BLOCK) {
        float sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            float pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    bestCorr[tid] = maxCorr;
    bestLag[tid] = bestT;
    __syncthreads();

    // Argmax reduction; ties resolve to the smallest lag (matches sequential
    // strict-greater-than scan from t = 0 upward).
    for (int s = DIST_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (bestCorr[tid + s] > bestCorr[tid] ||
                (bestCorr[tid + s] == bestCorr[tid] && bestLag[tid + s] < bestLag[tid])) {
                bestCorr[tid] = bestCorr[tid + s];
                bestLag[tid] = bestLag[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestLag[0]);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct GpuBuffers {
    Triangle* triangles = nullptr;
    DevNode* nodes = nullptr;
    int* leafTris = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* kij = nullptr;
    int* tau = nullptr;
    float* radE = nullptr;
    float* radB = nullptr;
    float* distances = nullptr;

    // Multi-GPU resources for the form factor phase (index 0 = device 0,
    // whose static buffers alias the ones above). All allocated at init.
    size_t ngpu = 1;
    size_t rowsPerChunk = 0;
    std::vector<Triangle*> devTris;
    std::vector<DevNode*> devNodes;
    std::vector<int*> devLeafTris;
    std::vector<float*> devKij;
    std::vector<std::array<float*, 2>> hRand, dRand;      // pinned host / device staging
    std::vector<std::array<uint32_t*, 2>> hOff, dOff;
    std::vector<std::array<cudaStream_t, 2>> streams;
};

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

    size_t sourceIndex;

    GpuBuffers gpu;                 // Device-side mirrors of the above

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Initialize the CUDA context up front so it isn't attributed to compute time
    CUDA_CHECK(cudaFree(nullptr));

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

    // Flatten octree and upload all static data to the GPU
    std::vector<DevNode> nodes;
    std::vector<int> leafTris;
    flattenOctree(state.octree, nodes, leafTris);

    size_t n = state.numTriangles;
    GpuBuffers& g = state.gpu;
    CUDA_CHECK(cudaMalloc(&g.triangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&g.nodes, nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&g.leafTris, leafTris.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.kij, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.radE, timesteps * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.radB, timesteps * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.distances, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(g.triangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.nodes, nodes.data(), nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.leafTris, leafTris.data(), leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.radE, state.radE.data(), timesteps * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.kij, 0, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.tau, 0, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemset(g.radB, 0, timesteps * n * sizeof(float)));

    // Set up the multi-GPU form factor pipeline: chunk geometry, per-device
    // replicas of the static data, and double-buffered staging buffers.
    const size_t floatBudget = 64ull << 20;  // floats per staging buffer (256 MB)
    g.rowsPerChunk = std::max<size_t>(1, floatBudget / (FLOATS_PER_PAIR * n));
    g.rowsPerChunk = std::min(g.rowsPerChunk, n);
    const size_t maxPairs = g.rowsPerChunk * n;
    const size_t maxFloats = maxPairs * FLOATS_PER_PAIR;
    const size_t numChunks = (n + g.rowsPerChunk - 1) / g.rowsPerChunk;

    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    g.ngpu = std::min<size_t>(std::max(numGpus, 1), numChunks);

    g.devTris.assign(g.ngpu, g.triangles);
    g.devNodes.assign(g.ngpu, g.nodes);
    g.devLeafTris.assign(g.ngpu, g.leafTris);
    g.devKij.assign(g.ngpu, g.kij);
    for (size_t dev = 1; dev < g.ngpu; ++dev) {
        CUDA_CHECK(cudaSetDevice(static_cast<int>(dev)));
        CUDA_CHECK(cudaMalloc(&g.devTris[dev], n * sizeof(Triangle)));
        CUDA_CHECK(cudaMalloc(&g.devNodes[dev], nodes.size() * sizeof(DevNode)));
        CUDA_CHECK(cudaMalloc(&g.devLeafTris[dev], leafTris.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g.devKij[dev], n * n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(g.devTris[dev], state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.devNodes[dev], nodes.data(), nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.devLeafTris[dev], leafTris.data(), leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(g.devKij[dev], 0, n * n * sizeof(float)));
    }

    g.hRand.resize(g.ngpu);
    g.hOff.resize(g.ngpu);
    g.dRand.resize(g.ngpu);
    g.dOff.resize(g.ngpu);
    g.streams.resize(g.ngpu);
    for (size_t dev = 0; dev < g.ngpu; ++dev) {
        CUDA_CHECK(cudaSetDevice(static_cast<int>(dev)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMallocHost(&g.hRand[dev][b], maxFloats * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&g.hOff[dev][b], maxPairs * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&g.dRand[dev][b], maxFloats * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&g.dOff[dev][b], maxPairs * sizeof(uint32_t)));
            CUDA_CHECK(cudaStreamCreate(&g.streams[dev][b]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    const int N = static_cast<int>(n);
    GpuBuffers& g = state.gpu;

    // The Monte Carlo stream: reproduces RandomGenerator(42) exactly. Each
    // non-culled (i,j) pair (i != j) consumes NUM_RAYS * 4 floats in row-major
    // pair order, exactly as the sequential loop does.
    ExactRandomStream rngStream(42);

    // Chunks of rows are dispatched round-robin across all GPUs,
    // double-buffered per GPU, so host RNG generation overlaps GPU compute on
    // every device. All buffers were allocated during initialization.
    const size_t rowsPerChunk = g.rowsPerChunk;
    const size_t ngpu = g.ngpu;

    size_t chunk = 0;
    for (size_t r0 = 0; r0 < n; r0 += rowsPerChunk, ++chunk) {
        size_t r1 = std::min(r0 + rowsPerChunk, n);
        size_t pairCount = (r1 - r0) * n;
        size_t dev = chunk % ngpu;
        int buf = static_cast<int>((chunk / ngpu) % 2);
        CUDA_CHECK(cudaSetDevice(static_cast<int>(dev)));

        // Wait until this buffer's previous chunk has been consumed by the GPU
        CUDA_CHECK(cudaStreamSynchronize(g.streams[dev][buf]));

        // Compute per-pair random offsets, in the exact sequential consumption
        // order: pairs with i == j or facing the same direction (culled in
        // computeKij before any sampling) consume no random numbers.
        uint32_t* off = g.hOff[dev][buf];
        size_t running = 0;
        for (size_t i = r0; i < r1; ++i) {
            const Vec3 ni = state.triangles[i].normal();
            for (size_t j = 0; j < n; ++j) {
                size_t p = (i - r0) * n + j;
                if (i == j || ni.dot(state.triangles[j].normal()) > 0.99f) {
                    off[p] = RAND_OFFSET_SKIP;
                } else {
                    off[p] = static_cast<uint32_t>(running);
                    running += FLOATS_PER_PAIR;
                }
            }
        }

        // Generate this chunk's slice of the random stream
        rngStream.fill(g.hRand[dev][buf], running);

        // Upload and launch
        CUDA_CHECK(cudaMemcpyAsync(g.dOff[dev][buf], g.hOff[dev][buf], pairCount * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, g.streams[dev][buf]));
        if (running > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g.dRand[dev][buf], g.hRand[dev][buf], running * sizeof(float),
                                       cudaMemcpyHostToDevice, g.streams[dev][buf]));
        }

        int threads = 128;
        long long blocks = (static_cast<long long>(pairCount) * NUM_RAYS + threads - 1) / threads;
        kijKernel<<<static_cast<unsigned>(blocks), threads, 0, g.streams[dev][buf]>>>(
            static_cast<int>(r0), static_cast<long long>(pairCount), N,
            g.devTris[dev], g.devNodes[dev], g.devLeafTris[dev],
            g.dRand[dev][buf], g.dOff[dev][buf], g.devKij[dev]);
        CUDA_CHECK(cudaGetLastError());

        for (size_t i = r0; i < r1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }

    for (size_t dev = 0; dev < ngpu; ++dev) {
        CUDA_CHECK(cudaSetDevice(static_cast<int>(dev)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Gather each device's chunk rows into the host matrix (used by
    // validation), then mirror the complete matrix to device 0 for the
    // simulation phase.
    chunk = 0;
    for (size_t r0 = 0; r0 < n; r0 += rowsPerChunk, ++chunk) {
        size_t r1 = std::min(r0 + rowsPerChunk, n);
        size_t dev = chunk % ngpu;
        CUDA_CHECK(cudaSetDevice(static_cast<int>(dev)));
        CUDA_CHECK(cudaMemcpy(state.kij.data() + r0 * n, g.devKij[dev] + r0 * n,
                              (r1 - r0) * n * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaSetDevice(0));
    if (ngpu > 1) {
        CUDA_CHECK(cudaMemcpy(g.kij, state.kij.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const int N = static_cast<int>(state.numTriangles);
    long long total = static_cast<long long>(N) * N;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    tauKernel<<<static_cast<unsigned>(blocks), threads>>>(N, state.gpu.triangles, state.gpu.tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    GpuBuffers& g = state.gpu;

    // Timesteps are inherently sequential (tau >= 1 for i != j, so timestep t
    // only reads rows < t); all triangles within a timestep run in parallel.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<N, SIM_BLOCK>>>(static_cast<int>(t), N,
                                        g.kij, g.tau, g.areas, g.rho, g.radE, g.radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Mirror results back to the host (used by validation)
    CUDA_CHECK(cudaMemcpy(state.radB.data(), g.radB,
                          state.numTimesteps * state.numTriangles * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    distKernel<<<N, DIST_BLOCK>>>(N, static_cast<int>(state.numTimesteps),
                                  static_cast<int>(state.sourceIndex),
                                  state.gpu.radB, state.gpu.distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.gpu.distances,
                          state.numTriangles * sizeof(float), cudaMemcpyDeviceToHost));
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

/**
 * Room Response Simulation Benchmark (CUDA)
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
 * GPU parallelization:
 * - Form factors: one thread per (receiver, emitter) pair. The Monte Carlo random
 *   samples are drawn from the same mt19937 stream, in the same order, as the original
 *   sequential code (the host generates the stream in batches, overlapped with GPU work),
 *   so results are identical to the sequential algorithm.
 * - Wave propagation: one thread per receiver triangle per timestep. The sum over
 *   emitters is still performed in the original order by each thread.
 * - Distance cross-correlation: one thread per triangle.
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
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA helpers
// ============================================================================

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Owning device allocation
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { release(); }

    void allocate(size_t n) {
        release();
        count = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
    void upload(const T* host, size_t n) {
        allocate(n);
        if (n > 0) CUDA_CHECK(cudaMemcpy(ptr, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    T* get() const { return ptr; }
    size_t size() const { return count; }

private:
    void release() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        count = 0;
    }
    T* ptr = nullptr;
    size_t count = 0;
};

// Owning pinned (page-locked) host allocation, used for async host->device copies
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }

    void allocate(size_t n) {
        if (ptr) cudaFreeHost(ptr);
        ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr), std::max<size_t>(n, 1) * sizeof(T)));
    }
    T* get() const { return ptr; }

private:
    T* ptr = nullptr;
};

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
constexpr val_t NO_HIT = FLT_MAX;         // Returned by ray/triangle test when there is no hit

constexpr int BLOCK_SIZE = 256;           // Threads per block for all kernels
// Random numbers consumed per (non-culled) form factor pair: 2 points x 2 values x NUM_RAYS rays
constexpr uint32_t DRAWS_PER_PAIR = NUM_RAYS * 4;
// Upper bound on number of (receiver, emitter) pairs processed per form factor batch
constexpr size_t BATCH_PAIR_BUDGET = size_t(1) << 18;
constexpr int OCTREE_STACK_SIZE = 128;

// Min/max with the same semantics as std::min/std::max (used in the original code)
HD inline val_t minVal(val_t a, val_t b) { return (b < a) ? b : a; }
HD inline val_t maxVal(val_t a, val_t b) { return (a < b) ? b : a; }

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
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

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
// Octree for Spatial Acceleration (built on host, traversed on device)
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

// Flattened octree node for device traversal
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];   // -1 if no child
    uint32_t triBegin;  // Leaf: range in triIdx
    uint32_t triCount;  // Leaf: number of triangles (0 for internal nodes)
};

// Read-only view of the flattened octree
struct DevOctree {
    const DevNode* nodes;
    const uint32_t* triIdx;
};

static int flattenOctreeNode(const Octree& node, std::vector<DevNode>& nodes,
                             std::vector<uint32_t>& triIdx, int depth, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    int me = static_cast<int>(nodes.size());
    nodes.push_back(DevNode{});

    DevNode d{};
    d.center = node.center;
    d.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) d.child[c] = -1;
    d.triBegin = static_cast<uint32_t>(triIdx.size());
    d.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) triIdx.push_back(static_cast<uint32_t>(idx));

    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) {
            d.child[c] = flattenOctreeNode(*node.children[c], nodes, triIdx, depth + 1, maxDepth);
        }
    }
    nodes[me] = d;
    return me;
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
// Random Number Generation (host-side stream, consumed in the original order)
// ============================================================================

// Standard MT19937 (identical sequence to std::mt19937), with the output tempering and
// float conversion done in a vectorizable block loop. Float values are identical to
// std::uniform_real_distribution<float>(0,1) driven by std::mt19937: one 32-bit engine
// output scaled by 2^-32 (exact), clamped below 1 as the standard library does.
class RandomGenerator {
    static constexpr uint32_t MT_N = 624;
    static constexpr uint32_t MT_M = 397;

    uint32_t mt[MT_N];
    uint32_t pos = MT_N;

    static inline uint32_t twistStep(uint32_t u, uint32_t v) {
        const uint32_t y = (u & 0x80000000u) | (v & 0x7fffffffu);
        return (y >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
    }

    void twist() {
        uint32_t* __restrict x = mt;
        for (uint32_t i = 0; i < MT_N - MT_M; ++i) x[i] = x[i + MT_M] ^ twistStep(x[i], x[i + 1]);
        for (uint32_t i = MT_N - MT_M; i < MT_N - 1; ++i) x[i] = x[i + MT_M - MT_N] ^ twistStep(x[i], x[i + 1]);
        x[MT_N - 1] = x[MT_M - 1] ^ twistStep(x[MT_N - 1], x[0]);
        pos = 0;
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) {
        mt[0] = seed;
        for (uint32_t i = 1; i < MT_N; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }

    // Fills `out` with the next n values of the stream
    void fill(val_t* __restrict out, size_t n) {
        while (n) {
            if (pos == MT_N) twist();
            const size_t m = std::min<size_t>(n, MT_N - pos);
            const uint32_t* __restrict src = mt + pos;
            for (size_t k = 0; k < m; ++k) {
                uint32_t y = src[k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                val_t f = static_cast<val_t>(y) * (1.0f / 4294967296.0f);
                out[k] = (f >= ONE) ? 0.99999994f : f;  // nextafter(1, 0)
            }
            pos += static_cast<uint32_t>(m);
            out += m;
            n -= m;
        }
    }
};

// ============================================================================
// Device geometry and visibility routines
// ============================================================================

// Point inside triangle from two uniform samples (barycentric coordinates)
HD inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Möller-Trumbore ray/triangle intersection; returns NO_HIT if there is no hit
HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Check if a ray segment intersects a node's bounding box (slab/SAT test)
HD inline bool rayIntersectsBox(const Vec3& boxCenter, const Vec3& halfExtent,
                                const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - boxCenter;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle (octree-accelerated)
HD bool isRayBlocked(const Vec3& from, const Vec3& to, const DevOctree& oct,
                     const Triangle* triangles, uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Iterative traversal; the result is a pure "any triangle blocks" predicate, so the
    // visiting order does not affect the outcome.
    uint32_t stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root

    while (sp > 0) {
        const DevNode& node = oct.nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                uint32_t idx = oct.triIdx[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = node.child[c];
                if (ch >= 0 && rayIntersectsBox(oct.nodes[ch].center, oct.nodes[ch].halfExtent, from, to)) {
                    stack[sp++] = static_cast<uint32_t>(ch);
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
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return maxVal(ZERO, v.dot(normal) / vNorm);
}

// Pair (i,j) contributes to Kij only if i != j and the triangles do not face the same way
HD inline bool isFormFactorPair(const Triangle* triangles, uint32_t i, uint32_t j) {
    if (i == j) return false;
    return !(triangles[i].normal().dot(triangles[j].normal()) > 0.99f);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter).
// `draws` points to the NUM_RAYS x 4 uniform samples reserved for this pair.
HD inline val_t computeKij(uint32_t idxI, uint32_t idxJ, const Triangle* triangles,
                           const DevOctree& oct, const float4* draws) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const float4 s = draws[r];
        Vec3 pI = pointInTriangle(triI, s.x, s.y);
        Vec3 pJ = pointInTriangle(triJ, s.z, s.w);

        if (isRayBlocked(pI, pJ, oct, triangles, idxI, idxJ)) continue;

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

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD inline int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Block-wide exclusive scan helper
// ============================================================================

template <int BS>
__device__ __forceinline__ int blockExclusiveScan(int val, int* warpSums, int& total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    int inc = val;
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, inc, off);
        if (lane >= off) inc += n;
    }
    if (lane == 31) warpSums[warp] = inc;
    __syncthreads();

    if (warp == 0) {
        int w = (lane < BS / 32) ? warpSums[lane] : 0;
        for (int off = 1; off < 32; off <<= 1) {
            int n = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += n;
        }
        if (lane < BS / 32) warpSums[lane] = w;
    }
    __syncthreads();

    total = warpSums[BS / 32 - 1];
    int prefix = (warp > 0) ? warpSums[warp - 1] : 0;
    return inc - val + prefix;
}

// ============================================================================
// Kernels
// ============================================================================

// Number of form factor pairs that consume random samples, per receiver row
__global__ void countFormFactorPairsKernel(uint32_t N, const Triangle* triangles, uint32_t* counts) {
    __shared__ uint32_t rowCount;
    const uint32_t i = blockIdx.x;
    if (threadIdx.x == 0) rowCount = 0;
    __syncthreads();

    uint32_t local = 0;
    for (uint32_t j = threadIdx.x; j < N; j += blockDim.x) {
        if (isFormFactorPair(triangles, i, j)) ++local;
    }
    atomicAdd(&rowCount, local);
    __syncthreads();
    if (threadIdx.x == 0) counts[i] = rowCount;
}

// For each kept pair of rows [r0, r0+rows), compute its exclusive rank within its row.
// Output layout: rankT[j * rows + r].  One block per row.
__global__ void formFactorRankKernel(uint32_t r0, uint32_t rows, uint32_t N,
                                     const Triangle* triangles, uint32_t* rankT) {
    __shared__ int warpSums[BLOCK_SIZE / 32];
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const uint32_t i = r0 + r;

    int running = 0;
    for (uint32_t c = 0; c < N; c += BLOCK_SIZE) {
        const uint32_t j = c + threadIdx.x;
        const int keep = (j < N && isFormFactorPair(triangles, i, j)) ? 1 : 0;
        int total = 0;
        const int ex = blockExclusiveScan<BLOCK_SIZE>(keep, warpSums, total);
        if (keep) rankT[static_cast<size_t>(j) * rows + r] = static_cast<uint32_t>(running + ex);
        running += total;
        __syncthreads();
    }
}

// Form factors for rows [r0, r0+rows). One thread per (receiver, emitter) entry.
// Writes the transposed matrix: kijT[j * N + i] = Kij, so that the simulation can read
// emitter-major data coalesced across receivers.
__global__ void formFactorKernel(uint32_t r0, uint32_t rows, uint32_t N,
                                 const uint32_t* rowOffsets, const uint32_t* rankT,
                                 const float4* draws, const Triangle* triangles,
                                 DevOctree oct, val_t* kijT) {
    const size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= static_cast<size_t>(N) * rows) return;

    const uint32_t j = static_cast<uint32_t>(g / rows);
    const uint32_t r = static_cast<uint32_t>(g % rows);
    const uint32_t i = r0 + r;

    val_t kij = ZERO;
    if (isFormFactorPair(triangles, i, j)) {
        const uint32_t rank = rankT[static_cast<size_t>(j) * rows + r];
        // Row offsets and per-pair draw blocks are multiples of 64 floats (16 float4)
        const size_t base = (static_cast<size_t>(rowOffsets[r]) >> 2) +
                            static_cast<size_t>(rank) * (DRAWS_PER_PAIR / 4);
        kij = computeKij(i, j, triangles, oct, draws + base);
    }
    kijT[static_cast<size_t>(j) * N + i] = kij;
}

// Counts entries of the (transposed) form factor matrix above EPSILON
__global__ void countNonZeroKijKernel(size_t total, const val_t* kijT, unsigned int* count) {
    __shared__ unsigned int blockCount;
    if (threadIdx.x == 0) blockCount = 0;
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int local = 0;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += stride) {
        if (kijT[k] > EPSILON) ++local;
    }
    atomicAdd(&blockCount, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(count, blockCount);
}

// One simulation timestep: radB[t][i] for all receivers i. Each thread sums over emitters
// j in the original order.
__global__ void simulateTimestepKernel(size_t N, size_t t, size_t src, size_t timeOff, val_t rho,
                                       const Triangle* triangles, const val_t* areas,
                                       const val_t* kijT, val_t* radB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;
    const Triangle& triI = triangles[i];

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = computeTau(triI, triangles[j]);

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = kijT[j * N + i];
        if (kij <= ZERO) continue;

        // Radiosity from source triangle at the time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += minVal(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission (source emits during the first half)
    val_t radE = (t < timeOff && i == src) ? ONE : ZERO;
    radB[t * N + i] = rho * sumB + radE;
}

// Cross-correlation distance estimate for each triangle
__global__ void distanceKernel(size_t N, size_t T, size_t src, const val_t* radB, val_t* distances) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + src];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    val_t reflectivity = ZERO;

    // Host copies
    std::vector<Triangle> triangles;  // Mesh (also uploaded to device)
    std::vector<val_t> areas;         // Area of each triangle
    std::vector<val_t> radB;          // Radiosity (T x N), copied back after simulation
    std::vector<val_t> distances;     // Computed distances from source

    Octree octree;                    // Spatial acceleration structure (host build)

    // Device data
    DeviceBuffer<Triangle> dTriangles;
    DeviceBuffer<val_t> dAreas;
    DeviceBuffer<DevNode> dNodes;
    DeviceBuffer<uint32_t> dTriIdx;
    DeviceBuffer<val_t> dKijT;        // Form factors, transposed (N x N): [j * N + i] = Kij
    DeviceBuffer<val_t> dRadB;        // Radiosity (T x N): [t * N + i]
    DeviceBuffer<val_t> dDistances;
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
    state.reflectivity = reflectivity;

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    std::vector<DevNode> nodes;
    std::vector<uint32_t> triIdx;
    int maxDepth = 0;
    flattenOctreeNode(state.octree, nodes, triIdx, 0, maxDepth);
    if (8 * maxDepth + 1 > OCTREE_STACK_SIZE) {
        fprintf(stderr, "Octree too deep for traversal stack (depth %d)\n", maxDepth);
        exit(EXIT_FAILURE);
    }

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Upload geometry and acceleration structure
    state.dTriangles.upload(state.triangles.data(), N);
    state.dAreas.upload(state.areas.data(), N);
    state.dNodes.upload(nodes.data(), nodes.size());
    state.dTriIdx.upload(triIdx.data(), triIdx.size());

    // Allocate and zero matrices
    state.dKijT.allocate(N * N);
    CUDA_CHECK(cudaMemset(state.dKijT.get(), 0, N * N * sizeof(val_t)));
    state.dRadB.allocate(T * N);
    if (T * N > 0) CUDA_CHECK(cudaMemset(state.dRadB.get(), 0, T * N * sizeof(val_t)));
    state.dDistances.allocate(N);
    state.radB.assign(T * N, ZERO);
    state.distances.assign(N, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);

    // 1. Number of pairs that consume random samples in each row (determines stream layout)
    DeviceBuffer<uint32_t> dRowCounts;
    dRowCounts.allocate(N);
    countFormFactorPairsKernel<<<N, BLOCK_SIZE>>>(N, state.dTriangles.get(), dRowCounts.get());
    CUDA_CHECK_LAST();
    std::vector<uint32_t> rowCounts(N);
    CUDA_CHECK(cudaMemcpy(rowCounts.data(), dRowCounts.get(), N * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // 2. Partition rows into batches. Each batch has a contiguous slice of the random stream.
    struct Batch {
        uint32_t r0;
        uint32_t rows;
        size_t draws;
    };
    std::vector<Batch> batches;
    size_t maxRows = 0;
    size_t maxDraws = 0;
    const size_t rowsPerBatch = std::max<size_t>(1, BATCH_PAIR_BUDGET / N);
    for (size_t r0 = 0; r0 < N; r0 += rowsPerBatch) {
        size_t rows = std::min(rowsPerBatch, N - r0);
        size_t kept = 0;
        for (size_t r = 0; r < rows; ++r) kept += rowCounts[r0 + r];
        batches.push_back({static_cast<uint32_t>(r0), static_cast<uint32_t>(rows), kept * DRAWS_PER_PAIR});
        maxRows = std::max(maxRows, rows);
        maxDraws = std::max(maxDraws, kept * DRAWS_PER_PAIR);
    }

    // 3. Double-buffered pipeline: host generates the random stream for batch b while the
    //    GPU processes batch b-1.
    PinnedBuffer<val_t> hDraws[2];
    PinnedBuffer<uint32_t> hOffsets[2];
    DeviceBuffer<val_t> dDraws[2];
    DeviceBuffer<uint32_t> dOffsets[2];
    DeviceBuffer<uint32_t> dRank[2];
    cudaStream_t streams[2];
    cudaEvent_t copyDone[2];
    for (int k = 0; k < 2; ++k) {
        hDraws[k].allocate(maxDraws);
        hOffsets[k].allocate(maxRows);
        dDraws[k].allocate(std::max<size_t>(maxDraws, 4));
        dOffsets[k].allocate(maxRows);
        dRank[k].allocate(maxRows * N);
        CUDA_CHECK(cudaStreamCreate(&streams[k]));
        CUDA_CHECK(cudaEventCreateWithFlags(&copyDone[k], cudaEventDisableTiming));
    }

    RandomGenerator rng(42);
    const Triangle* dTris = state.dTriangles.get();
    DevOctree oct{state.dNodes.get(), state.dTriIdx.get()};

    for (size_t b = 0; b < batches.size(); ++b) {
        const int k = static_cast<int>(b & 1);
        const Batch& batch = batches[b];

        // Host buffer k may still be read by the copy issued for batch b-2
        CUDA_CHECK(cudaEventSynchronize(copyDone[k]));

        // Row offsets (in floats) within this batch's draw buffer
        uint32_t* off = hOffsets[k].get();
        size_t acc = 0;
        for (uint32_t r = 0; r < batch.rows; ++r) {
            off[r] = static_cast<uint32_t>(acc);
            acc += static_cast<size_t>(rowCounts[batch.r0 + r]) * DRAWS_PER_PAIR;
        }

        // Generate this batch's slice of the random stream, in the original consumption order
        val_t* hd = hDraws[k].get();
        rng.fill(hd, batch.draws);

        CUDA_CHECK(cudaMemcpyAsync(dDraws[k].get(), hd, batch.draws * sizeof(val_t),
                                   cudaMemcpyHostToDevice, streams[k]));
        CUDA_CHECK(cudaMemcpyAsync(dOffsets[k].get(), off, batch.rows * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, streams[k]));
        CUDA_CHECK(cudaEventRecord(copyDone[k], streams[k]));

        formFactorRankKernel<<<batch.rows, BLOCK_SIZE, 0, streams[k]>>>(
            batch.r0, batch.rows, N, dTris, dRank[k].get());
        CUDA_CHECK_LAST();

        const size_t work = static_cast<size_t>(N) * batch.rows;
        const unsigned int grid = static_cast<unsigned int>((work + BLOCK_SIZE - 1) / BLOCK_SIZE);
        formFactorKernel<<<grid, BLOCK_SIZE, 0, streams[k]>>>(
            batch.r0, batch.rows, N, dOffsets[k].get(), dRank[k].get(),
            reinterpret_cast<const float4*>(dDraws[k].get()), dTris, oct, state.dKijT.get());
        CUDA_CHECK_LAST();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaEventDestroy(copyDone[k]));
        CUDA_CHECK(cudaStreamDestroy(streams[k]));
    }

    for (uint32_t i = 0; i < N; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            printf("  Progress: %u/%u triangles\n", i + 1, N);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t src = state.sourceIndex;
    const size_t timeOff = T / 2;
    const unsigned int grid = static_cast<unsigned int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t t = 0; t < T; ++t) {
        simulateTimestepKernel<<<grid, BLOCK_SIZE>>>(N, t, src, timeOff, state.reflectivity,
                                                     state.dTriangles.get(), state.dAreas.get(),
                                                     state.dKijT.get(), state.dRadB.get());
        CUDA_CHECK_LAST();

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (T * N > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB.get(), T * N * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const unsigned int grid = static_cast<unsigned int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    distanceKernel<<<grid, BLOCK_SIZE>>>(N, state.numTimesteps, state.sourceIndex,
                                         state.dRadB.get(), state.dDistances.get());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances.get(), N * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Validation
// ============================================================================

size_t countNonZeroKij(const SimulationState& state) {
    const size_t total = state.numTriangles * state.numTriangles;
    DeviceBuffer<unsigned int> dCount;
    dCount.allocate(1);
    CUDA_CHECK(cudaMemset(dCount.get(), 0, sizeof(unsigned int)));
    const unsigned int grid = static_cast<unsigned int>(std::min<size_t>((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535));
    countNonZeroKijKernel<<<std::max(grid, 1u), BLOCK_SIZE>>>(total, state.dKijT.get(), dCount.get());
    CUDA_CHECK_LAST();
    unsigned int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, dCount.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return count;
}

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
            if (state.radB[t * state.numTriangles + i] > EPSILON) {
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
    const int nonZeroKij = static_cast<int>(countNonZeroKij(state));
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

    // Precomputation (time delays are evaluated on the device during propagation)
    auto startPre = std::chrono::high_resolution_clock::now();

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

    // Memory usage (logical footprint of the N x N matrices and radiosity arrays)
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

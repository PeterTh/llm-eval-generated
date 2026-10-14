/**
 * Room Response Simulation Benchmark
 *
 * This is an implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallel decomposition (hybrid MPI + OpenMP + CUDA):
 *  - MPI:    the receiver rows i of the N x N Kij and Tau matrices are split into
 *            contiguous blocks, one block per rank. Wave propagation is sequential
 *            in time, so after every timestep the ranks allgather their slice of
 *            the radiosity history; every rank keeps the full history.
 *  - CUDA:   each rank runs its block on its GPU: visibility ray casting against a
 *            flattened octree for Kij, Tau, the per-timestep radiosity update, and
 *            the cross-correlation distances.
 *  - OpenMP: the stream-offset counting for the random generator, and the overlap of
 *            host-side random number generation for the next chunk with GPU work
 *            on the current chunk.
 *
 * Semantics are preserved: the random number stream of the sequential algorithm
 * is reproduced exactly (each rank skips ahead to the position of its first row),
 * and the floating-point operations run in the same order with contraction disabled
 * on both host and device, so the results match the sequential reference.
 */

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

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
constexpr size_t DRAWS_PER_PAIR = NUM_RAYS * 4;   // Two points (u, v) per ray, for receiver and emitter
constexpr size_t DRAW_BUDGET = size_t(1) << 25;    // Random values per pipeline chunk (bounds buffer memory)
constexpr unsigned THREADS_PER_BLOCK = 128;
constexpr int OCTREE_STACK_SIZE = 128;            // Depth-first traversal needs at most 7 * depth + 1 entries
constexpr int MAX_DEVICE_OCTREE_DEPTH = 15;

// Host-side printing restricted to the root rank
int gRank = 0;

void rootPrintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void rootPrintf(const char* fmt, ...) {
    if (gRank != 0) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

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

// Same semantics as std::min / std::max for floats, usable on the device
HD inline val_t hmin(val_t a, val_t b) { return b < a ? b : a; }
HD inline val_t hmax(val_t a, val_t b) { return a < b ? b : a; }

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

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

static_assert(std::is_trivially_copyable_v<Triangle>, "Triangles are copied to the device as raw data");

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

// Flat, device-friendly form of the octree. Node 0 is the root.
struct FlatNode {
    Vec3 center, halfExtent;
    int32_t child[8];          // Index of child node, or -1
    uint32_t leafBegin;        // Start of this leaf's triangles in leafTris
    uint32_t leafCount;        // Number of triangles in this leaf (0 for interior nodes)
};

struct FlatOctree {
    std::vector<FlatNode> nodes;
    std::vector<uint32_t> leafTris;
    int maxDepth = 0;
};

uint32_t flattenOctree(const Octree& node, FlatOctree& out, int depth) {
    const uint32_t self = static_cast<uint32_t>(out.nodes.size());
    out.nodes.emplace_back();
    out.maxDepth = std::max(out.maxDepth, depth);

    FlatNode& fn = out.nodes[self];
    fn.center = node.center;
    fn.halfExtent = node.halfExtent;
    std::fill(std::begin(fn.child), std::end(fn.child), -1);
    fn.leafBegin = 0;
    fn.leafCount = 0;

    if (!node.triangleIndices.empty()) {
        fn.leafBegin = static_cast<uint32_t>(out.leafTris.size());
        fn.leafCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) out.leafTris.push_back(static_cast<uint32_t>(idx));
        return self;
    }

    for (int i = 0; i < 8; ++i) {
        if (node.children[i]) {
            const uint32_t childIdx = flattenOctree(*node.children[i], out, depth + 1);
            out.nodes[self].child[i] = static_cast<int32_t>(childIdx);
        }
    }
    return self;
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

// MT19937 with the exact output sequence of std::mt19937. The twist is written so that the
// compiler can vectorize it, which roughly halves the cost of generating the random stream.
class MT19937 {
public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0u; }
    static constexpr result_type max() { return 0xffffffffu; }

    explicit MT19937(uint32_t seed) {
        state[0] = seed;
        for (int i = 1; i < N; ++i) {
            state[i] = 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        index = N;
    }

    result_type operator()() {
        if (index >= N) twistState();
        if (!tempered) temper();
        return out[index++];
    }

    // Advance by n outputs. Skipped outputs are never tempered, only the state is twisted.
    void discard(uint64_t n) {
        while (n > 0) {
            if (index >= N) twistState();
            uint64_t take = std::min<uint64_t>(n, static_cast<uint64_t>(N - index));
            index += static_cast<int>(take);
            n -= take;
        }
    }

private:
    static constexpr int N = 624, M = 397;
    static constexpr uint32_t UPPER = 0x80000000u, LOWER = 0x7fffffffu, MATRIX = 0x9908b0dfu;

    static uint32_t twistWord(uint32_t u, uint32_t v) {
        uint32_t y = (u & UPPER) | (v & LOWER);
        return (y >> 1) ^ (MATRIX & (0u - (v & 1u)));
    }

    // Advance the state by one block. The new state goes to a separate buffer so the first
    // loops have no loop-carried dependency and can be vectorized.
    void twistState() {
        uint32_t next[N];
        for (int k = 0; k < N - M; ++k) next[k] = state[k + M] ^ twistWord(state[k], state[k + 1]);
        for (int k = N - M; k < N - 1; ++k) next[k] = next[k + M - N] ^ twistWord(state[k], state[k + 1]);
        next[N - 1] = next[M - 1] ^ twistWord(state[N - 1], next[0]);
        for (int k = 0; k < N; ++k) state[k] = next[k];
        index = 0;
        tempered = false;
    }

    // Produce the tempered outputs of the current block
    void temper() {
        for (int k = 0; k < N; ++k) {
            uint32_t y = state[k];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            out[k] = y;
        }
        tempered = true;
    }

    uint32_t state[N];
    uint32_t out[N];
    int index;
    bool tempered = false;
};

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    // Advance the stream by n engine outputs (each rand() consumes exactly one)
    void skip(uint64_t n) { rng.discard(n); }
};

// ============================================================================
// Geometry Helpers (shared by host and device)
// ============================================================================

// Point inside a triangle from barycentric random values (u, v)
HD Vec3 samplePoint(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-triangle intersection (Möller-Trumbore algorithm); returns FLT_MAX on a miss
HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Slab-style ray vs. axis-aligned box test (separating axis formulation)
HD bool boxRayHit(const Vec3& center, const Vec3& halfExtent, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return hmax(ZERO, v.dot(normal) / vNorm);
}

// Form factor pairs whose normals (almost) coincide get no contribution and consume no random values
inline bool formFactorCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// ============================================================================
// Visibility Testing (device, octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle.
// Traverses the flattened octree with an explicit stack; the result is a
// pure "any blocker" predicate, so the traversal order does not affect it.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const FlatNode* nodes, const uint32_t* leafTris, const Triangle* tris,
                             size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int32_t stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // Root

    while (sp > 0) {
        const FlatNode& node = nodes[stack[--sp]];

        if (node.leafCount > 0) {
            for (uint32_t k = 0; k < node.leafCount; ++k) {
                size_t idx = leafTris[node.leafBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
            continue;
        }

        for (int c = 0; c < 8; ++c) {
            int32_t child = node.child[c];
            if (child >= 0 && boxRayHit(nodes[child].center, nodes[child].halfExtent, from, to)) {
                stack[sp++] = child;
            }
        }
    }
    return false;
}

// ============================================================================
// Device Memory Helpers
// ============================================================================

struct CudaFree {
    void operator()(void* p) const { if (p) cudaFree(p); }
};

template <typename T>
using DevArray = std::unique_ptr<T, CudaFree>;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

template <typename T>
DevArray<T> deviceAlloc(size_t count) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return DevArray<T>(static_cast<T*>(p));
}

template <typename T>
DevArray<T> toDevice(const std::vector<T>& host) {
    DevArray<T> dev = deviceAlloc<T>(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(dev.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev;
}

inline unsigned blocksFor(size_t n) {
    return static_cast<unsigned>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Matrices Tau and Kij are stored column-major over the local rows: element (local row li, column j)
// is at j * localRows + li. Threads of a warp then touch consecutive memory in the simulation loop.

// Tau (time delay) for the local rows [rowBegin, rowBegin + localRows)
__global__ void tauKernel(const Triangle* tris, size_t N, size_t rowBegin, size_t localRows, int* tau) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= localRows * N) return;

    size_t j = gid / localRows;
    size_t i = rowBegin + gid % localRows;

    int value = 0;
    if (i != j) {
        val_t dist = (tris[i].center() - tris[j].center()).norm();
        value = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tau[gid] = value;
}

// Form factors for one chunk of receiver rows. slots[gid] is the offset of the pair's random values in draws
// (-1 if the pair does not sample), with gid = ri * N + j for chunk row ri and column j.
__global__ void formFactorKernel(const Triangle* tris, const FlatNode* nodes, const uint32_t* leafTris,
                                 const int* slots, const val_t* draws,
                                 size_t rowBegin, size_t rowCount, size_t N,
                                 size_t localRowBegin, size_t localRows,
                                 val_t* kij, unsigned long long* nonZeroCount) {
    size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (gid >= rowCount * N) return;

    size_t ri = gid / N;
    size_t j = gid % N;
    size_t i = rowBegin + ri;

    val_t kijV = ZERO;
    int slot = slots[gid];

    if (slot >= 0) {
        const Triangle& triI = tris[i];
        const Triangle& triJ = tris[j];

        for (int r = 0; r < NUM_RAYS; ++r) {
            const val_t* d = draws + slot + 4 * r;
            Vec3 pI = samplePoint(triI, d[0], d[1]);
            Vec3 pJ = samplePoint(triJ, d[2], d[3]);

            if (isRayBlocked(pI, pJ, nodes, leafTris, tris, i, j)) continue;

            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr < EPSILON) continue;

            val_t cosPhiI = cosPhi(v, triI.normal());
            val_t cosPhiJ = cosPhi(-v, triJ.normal());

            if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

            kijV += (cosPhiI * cosPhiJ) / (PI * distSqr);
        }
        kijV = kijV * INV_NUM_RAYS;
    }

    kij[j * localRows + localRowBegin + ri] = kijV;
    if (kijV > EPSILON) atomicAdd(nonZeroCount, 1ULL);
}

// One timestep of wave propagation for the local receiver rows. Reads the full radiosity
// history (all earlier timesteps) and writes radB[t, i] for the local rows only.
__global__ void simulationStepKernel(size_t t, size_t N, size_t rowBegin, size_t localRows,
                                     size_t srcIdx, size_t timeOff,
                                     const val_t* kij, const int* tau,
                                     const val_t* areas, const val_t* rho, val_t* radB) {
    size_t li = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (li >= localRows) return;

    size_t i = rowBegin + li;
    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[j * localRows + li];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kijV = kij[j * localRows + li];
        if (kijV <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += hmin(kijV * areas[j], ONE) * radJ;
    }

    // Emission is active for the first half of the timesteps at the source triangle
    val_t radE = (t < timeOff && i == srcIdx) ? ONE : ZERO;
    radB[t * N + i] = rho[i] * sumB + radE;
}

// Cross-correlation distance for the local rows
__global__ void distancesKernel(size_t T, size_t N, size_t rowBegin, size_t localRows, size_t srcIdx,
                                const val_t* radB, val_t* distances) {
    size_t li = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (li >= localRows) return;

    size_t i = rowBegin + li;
    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;

        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + srcIdx];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Simulation State
// ============================================================================

// Contiguous block of receiver rows owned by one rank
struct Partition {
    size_t begin = 0;
    size_t end = 0;
    size_t size() const { return end - begin; }
};

Partition partitionRows(size_t n, int rank, int numRanks) {
    Partition p;
    p.begin = n * static_cast<size_t>(rank) / static_cast<size_t>(numRanks);
    p.end = n * static_cast<size_t>(rank + 1) / static_cast<size_t>(numRanks);
    return p;
}

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated on every rank
    std::vector<val_t> distances;   // Computed distances from source (replicated)

    Octree octree;                  // Spatial acceleration structure (host, used to build flatOctree)
    FlatOctree flatOctree;          // Device-ready form of octree

    size_t sourceIndex;
    size_t timeOff;                 // Source emits for timesteps [0, timeOff)
    unsigned long long nonZeroKij = 0;  // Count of form factors above EPSILON (global)

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Per-rank device data and the row layout used for MPI exchanges
struct GpuState {
    int deviceId = 0;
    Partition rows;
    size_t rowsPerChunk = 1;
    std::vector<int> rowCounts;     // MPI layout of receiver rows over all ranks
    std::vector<int> rowDispls;

    DevArray<Triangle> triangles;
    DevArray<FlatNode> nodes;
    DevArray<uint32_t> leafTris;
    DevArray<val_t> areas;
    DevArray<val_t> rho;
    DevArray<val_t> kij;            // Local rows x N, column-major (see kernels)
    DevArray<int> tau;              // Local rows x N, column-major (see kernels)
    DevArray<val_t> radB;           // T x N, full history
    DevArray<val_t> distances;      // Local rows
    DevArray<int> slots;            // Per chunk
    DevArray<val_t> draws;          // Per chunk
    DevArray<unsigned long long> nonZeroCount;
};

// Random values for a block of receiver rows, generated in the sequential stream order
struct RandomChunk {
    size_t rowBegin = 0;
    size_t rowCount = 0;
    std::vector<int> slots;         // Per (row, column): offset into draws, or -1
    std::vector<val_t> draws;       // DRAWS_PER_PAIR values per sampling pair
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
    state.timeOff = timesteps / 2;

    rootPrintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    rootPrintf("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctree(state.octree, state.flatOctree, 0);
    if (state.flatOctree.maxDepth > MAX_DEVICE_OCTREE_DEPTH) {
        fprintf(stderr, "Octree depth %d exceeds device traversal limit\n", state.flatOctree.maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize radiosity history and distances
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
}

// Row layout of all ranks, as MPI counts/displacements of receiver rows
void buildRowLayout(size_t n, int numRanks, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(numRanks);
    displs.resize(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        Partition p = partitionRows(n, r, numRanks);
        counts[r] = static_cast<int>(p.size());
        displs[r] = static_cast<int>(p.begin);
    }
}

GpuState createGpuState(const SimulationState& state, int rank, int numRanks, int deviceId) {
    GpuState gpu;
    gpu.deviceId = deviceId;
    gpu.rows = partitionRows(state.numTriangles, rank, numRanks);
    buildRowLayout(state.numTriangles, numRanks, gpu.rowCounts, gpu.rowDispls);

    const size_t N = state.numTriangles;
    const size_t localRows = gpu.rows.size();
    const size_t T = state.numTimesteps;

    CUDA_CHECK(cudaSetDevice(deviceId));

    // Chunk size is chosen so that one chunk's random values stay within DRAW_BUDGET
    gpu.rowsPerChunk = std::max<size_t>(1, std::min(localRows, DRAW_BUDGET / (DRAWS_PER_PAIR * N)));

    gpu.triangles = toDevice(state.triangles);
    gpu.nodes = toDevice(state.flatOctree.nodes);
    gpu.leafTris = toDevice(state.flatOctree.leafTris);
    gpu.areas = toDevice(state.areas);
    gpu.rho = toDevice(state.rho);

    gpu.kij = deviceAlloc<val_t>(localRows * N);
    gpu.tau = deviceAlloc<int>(localRows * N);
    gpu.radB = deviceAlloc<val_t>(T * N);
    gpu.distances = deviceAlloc<val_t>(localRows);
    gpu.slots = deviceAlloc<int>(gpu.rowsPerChunk * N);
    gpu.draws = deviceAlloc<val_t>(gpu.rowsPerChunk * N * DRAWS_PER_PAIR);
    gpu.nonZeroCount = deviceAlloc<unsigned long long>(1);

    CUDA_CHECK(cudaMemset(gpu.radB.get(), 0, std::max<size_t>(T * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(gpu.nonZeroCount.get(), 0, sizeof(unsigned long long)));

    return gpu;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Number of receiver/emitter pairs in rows [rowBegin, rowEnd) that draw random values
size_t countSamplingPairs(const std::vector<Triangle>& tris, size_t rowBegin, size_t rowEnd) {
    const size_t N = tris.size();
    size_t count = 0;
    #pragma omp parallel for reduction(+ : count) schedule(dynamic, 16)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i != j && !formFactorCulled(tris[i], tris[j])) ++count;
        }
    }
    return count;
}

// Generate the random values for a block of rows, in the order the sequential code consumes them
void fillChunk(RandomChunk& chunk, const std::vector<Triangle>& tris, RandomGenerator& rng,
               size_t rowBegin, size_t rowCount) {
    const size_t N = tris.size();
    chunk.rowBegin = rowBegin;
    chunk.rowCount = rowCount;
    chunk.slots.assign(rowCount * N, -1);
    chunk.draws.clear();
    chunk.draws.reserve(rowCount * N * DRAWS_PER_PAIR);  // Upper bound; keeps capacity across chunks

    for (size_t ri = 0; ri < rowCount; ++ri) {
        const size_t i = rowBegin + ri;
        for (size_t j = 0; j < N; ++j) {
            if (i == j || formFactorCulled(tris[i], tris[j])) continue;

            chunk.slots[ri * N + j] = static_cast<int>(chunk.draws.size());
            const size_t base = chunk.draws.size();
            chunk.draws.resize(base + DRAWS_PER_PAIR);
            for (size_t k = 0; k < DRAWS_PER_PAIR; ++k) {
                chunk.draws[base + k] = rng.rand();
            }
        }
    }
}

void computeTimeDelays(const SimulationState& state, GpuState& gpu) {
    rootPrintf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    const size_t localRows = gpu.rows.size();
    const size_t n = localRows * N;

    CUDA_CHECK(cudaSetDevice(gpu.deviceId));
    if (n > 0) {
        tauKernel<<<blocksFor(n), THREADS_PER_BLOCK>>>(gpu.triangles.get(), N, gpu.rows.begin, localRows,
                                                       gpu.tau.get());
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Form factors for one chunk of the local rows, on the GPU
void processFormFactorChunk(GpuState& gpu, const RandomChunk& chunk, size_t N) {
    CUDA_CHECK(cudaSetDevice(gpu.deviceId));

    const size_t n = chunk.rowCount * N;
    CUDA_CHECK(cudaMemcpy(gpu.slots.get(), chunk.slots.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.draws.get(), chunk.draws.data(), chunk.draws.size() * sizeof(val_t),
                          cudaMemcpyHostToDevice));

    formFactorKernel<<<blocksFor(n), THREADS_PER_BLOCK>>>(
        gpu.triangles.get(), gpu.nodes.get(), gpu.leafTris.get(), gpu.slots.get(), gpu.draws.get(),
        chunk.rowBegin, chunk.rowCount, N, chunk.rowBegin - gpu.rows.begin, gpu.rows.size(),
        gpu.kij.get(), gpu.nonZeroCount.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state, GpuState& gpu) {
    rootPrintf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t localRows = gpu.rows.size();
    const size_t numChunks = (localRows + gpu.rowsPerChunk - 1) / gpu.rowsPerChunk;

    // The random stream is shared by all pairs in row-major order. Start at this rank's first row.
    RandomGenerator rng(42);
    rng.skip(DRAWS_PER_PAIR * countSamplingPairs(state.triangles, 0, gpu.rows.begin));

    auto chunkRows = [&](size_t k, size_t& begin, size_t& count) {
        begin = gpu.rows.begin + k * gpu.rowsPerChunk;
        count = std::min(gpu.rowsPerChunk, gpu.rows.end - begin);
    };

    // Pipeline: the host generates the random values of chunk k+1 while the GPU processes chunk k
    RandomChunk chunks[2];
    if (numChunks > 0) {
        size_t begin = 0, count = 0;
        chunkRows(0, begin, count);
        fillChunk(chunks[0], state.triangles, rng, begin, count);
    }

    for (size_t k = 0; k < numChunks; ++k) {
        RandomChunk& current = chunks[k % 2];
        RandomChunk& next = chunks[(k + 1) % 2];

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (k + 1 < numChunks) {
                    size_t begin = 0, count = 0;
                    chunkRows(k + 1, begin, count);
                    fillChunk(next, state.triangles, rng, begin, count);
                }
            }
            #pragma omp section
            {
                processFormFactorChunk(gpu, current, N);
            }
        }

        rootPrintf("  Progress (rank 0): %zu/%zu receiver rows\n",
                   std::min(localRows, (k + 1) * gpu.rowsPerChunk), localRows);
    }

    CUDA_CHECK(cudaSetDevice(gpu.deviceId));
    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpy(&localNonZero, gpu.nonZeroCount.get(), sizeof(localNonZero), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&localNonZero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, GpuState& gpu) {
    rootPrintf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t localRows = gpu.rows.size();

    CUDA_CHECK(cudaSetDevice(gpu.deviceId));

    for (size_t t = 0; t < T; ++t) {
        if (localRows > 0) {
            simulationStepKernel<<<blocksFor(localRows), THREADS_PER_BLOCK>>>(
                t, N, gpu.rows.begin, localRows, state.sourceIndex, state.timeOff,
                gpu.kij.get(), gpu.tau.get(), gpu.areas.get(), gpu.rho.get(), gpu.radB.get());
            CUDA_CHECK(cudaGetLastError());
        }

        // Share this timestep's radiosity across ranks; every rank needs the full row for the next step
        CUDA_CHECK(cudaMemcpy(state.radB.data() + state.idxTN(t, gpu.rows.begin),
                              gpu.radB.get() + state.idxTN(t, gpu.rows.begin),
                              localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.radB.data() + state.idxTN(t, 0),
                       gpu.rowCounts.data(), gpu.rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(gpu.radB.get() + state.idxTN(t, 0), state.radB.data() + state.idxTN(t, 0),
                              N * sizeof(val_t), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            rootPrintf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuState& gpu) {
    rootPrintf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t localRows = gpu.rows.size();

    CUDA_CHECK(cudaSetDevice(gpu.deviceId));
    if (localRows > 0) {
        distancesKernel<<<blocksFor(localRows), THREADS_PER_BLOCK>>>(
            state.numTimesteps, N, gpu.rows.begin, localRows, state.sourceIndex,
            gpu.radB.get(), gpu.distances.get());
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<val_t> localDistances(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localDistances.data(), gpu.distances.get(), localRows * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(localDistances.data(), static_cast<int>(localRows), MPI_FLOAT,
                   state.distances.data(), gpu.rowCounts.data(), gpu.rowDispls.data(), MPI_FLOAT,
                   MPI_COMM_WORLD);
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

struct Options {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
};

int runBenchmark(const Options& opt, int rank, int numRanks, int deviceId) {
    int subdivisions = getSubdivisionsForTriangleCount(opt.targetTriangles);

    rootPrintf("Room Response Simulation Benchmark\n");
    rootPrintf("===================================\n");
    rootPrintf("Target triangles: %d (using %d subdivisions)\n", opt.targetTriangles, subdivisions);
    rootPrintf("Timesteps: %d\n", opt.timesteps);
    rootPrintf("Source triangle: %d\n", opt.sourceIdx);
    rootPrintf("Reflectivity: %.2f\n", opt.reflectivity);
    rootPrintf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
    rootPrintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(opt.timesteps),
                         static_cast<size_t>(opt.sourceIdx), opt.reflectivity);

    GpuState gpu = createGpuState(state, rank, numRanks, deviceId);

    rootPrintf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, gpu);
    computeFormFactors(state, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rootPrintf("Precomputation time: %ld ms\n", preDuration);
    rootPrintf("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rootPrintf("Simulation time: %ld ms\n", simDuration);
    rootPrintf("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rootPrintf("Distance computation time: %ld ms\n", distDuration);
    rootPrintf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
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
    rootPrintf("  Result hash: %016lX\n", hash);
    rootPrintf("\n");

    // Print results for external validation
    if (opt.printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        if (rank == 0) print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (opt.validate && rank == 0) {
        if (!validateResults(state)) {
            status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    gRank = rank;

    Options opt;

    // Parse command line arguments
    int parseStatus = -1;  // -1: run, 0: help, 1: error
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            opt.targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            opt.timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            opt.sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            opt.reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            opt.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            parseStatus = 0;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
            break;
        }
    }

    if (parseStatus != -1) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus;
    }

    // One GPU per rank, round-robin over the GPUs visible on the node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    int deviceId = localRank % numDevices;

    int status = runBenchmark(opt, rank, numRanks, deviceId);

    MPI_Finalize();
    return status;
}

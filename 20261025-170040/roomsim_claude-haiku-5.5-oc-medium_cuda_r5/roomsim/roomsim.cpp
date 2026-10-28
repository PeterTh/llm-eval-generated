/**
 * Room Response Simulation Benchmark (CUDA)
 *
 * This implementation of room impulse response simulation using radiosity-based
 * wave propagation. It models how sound/light waves propagate between surfaces
 * in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * GPU parallelization:
 *  - Form factors: one thread block per receiver triangle; visibility rays are
 *    traced against a flattened octree on the device. The per-pair random
 *    samples come from the same std::mt19937 stream, in the same order, as the
 *    sequential code (generated on the host in pipelined chunks).
 *  - Tau: one thread per (receiver, emitter) pair.
 *  - Wave propagation: one kernel launch per timestep, one thread per receiver.
 *  - Cross-correlation: one thread per (triangle, lag) pair, then a per-triangle
 *    arg-max scan.
 * Per-output summation order is preserved, so results match the sequential code.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <climits>
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

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
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
constexpr val_t NO_HIT = FLT_MAX;
constexpr uint32_t DRAWS_PER_RAY = 4;     // u,v for emitter point and u,v for receiver point
constexpr uint32_t DRAWS_PER_PAIR = DRAWS_PER_RAY * NUM_RAYS;
constexpr size_t CHUNK_DRAWS = size_t(1) << 24;  // Host->device random chunk (floats)
constexpr uint32_t FF_BLOCK = 512;        // Threads per receiver row (form factors)
constexpr uint32_t SIM_BLOCK = 256;

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

// Same semantics as std::min / std::max for arithmetic types (usable on device)
HD inline val_t sMin(val_t a, val_t b) { return (b < a) ? b : a; }
HD inline val_t sMax(val_t a, val_t b) { return (a < b) ? b : a; }

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

// Flattened octree node used on the device.
// A node is a leaf iff triCount > 0; leaves reference triIdx[triStart .. triStart+triCount).
// Internal nodes reference their children by index (-1 = no child).
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];
    uint32_t triStart;
    uint32_t triCount;
};

constexpr int OCTREE_STACK = 64;   // Enough for the depth of the subdivided icosphere (<= 8 levels)
constexpr int MAX_DEVICE_OCTREE_DEPTH = 7;

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

// Append the subtree rooted at `node` to the flat arrays; returns its node index.
// Root is always index 0.
uint32_t flattenOctree(const Octree& node, int depth,
                       std::vector<DevNode>& nodes, std::vector<uint32_t>& tris) {
    if (depth > MAX_DEVICE_OCTREE_DEPTH) {
        fprintf(stderr, "Octree depth %d exceeds device traversal limit\n", depth);
        exit(EXIT_FAILURE);
    }
    const uint32_t id = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();

    DevNode dn{};
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) dn.child[c] = -1;
    dn.triStart = 0;
    dn.triCount = 0;

    if (!node.triangleIndices.empty()) {
        dn.triStart = static_cast<uint32_t>(tris.size());
        dn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) tris.push_back(static_cast<uint32_t>(idx));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                dn.child[c] = static_cast<int32_t>(
                    flattenOctree(*node.children[c], depth + 1, nodes, tris));
            }
        }
    }
    nodes[id] = dn;
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
//
// The sample stream is the same as std::mt19937 + std::uniform_real_distribution<float>
// (libstdc++) with seed 42: the stream is consumed in the same order as the sequential
// code, so the generated samples are identical.
// ============================================================================

// Mersenne Twister MT19937 producing exactly the std::mt19937 sequence. The state is
// regenerated one 624-word block at a time using loops the compiler can vectorize.
class MT19937Stream {
public:
    explicit MT19937Stream(uint32_t seed) : pos_(N) {
        state_[0] = seed;
        for (uint32_t i = 1; i < N; ++i) {
            state_[i] = 1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i;
        }
    }

    // Write the next n 32-bit outputs to out
    void fill(uint32_t* out, size_t n) {
        size_t k = 0;
        while (k < n) {
            if (pos_ >= N) {
                regenerate();
                pos_ = 0;
            }
            const size_t take = std::min<size_t>(N - pos_, n - k);
            for (size_t j = 0; j < take; ++j) out[k + j] = temper(state_[pos_ + j]);
            pos_ += static_cast<int>(take);
            k += take;
        }
    }

private:
    static constexpr uint32_t N = 624;
    static constexpr uint32_t M = 397;

    static inline uint32_t twist(uint32_t u, uint32_t v, uint32_t x) {
        const uint32_t y = (u & 0x80000000u) | (v & 0x7fffffffu);
        return x ^ (y >> 1) ^ ((0u - (v & 1u)) & 0x9908b0dfu);
    }

    static inline uint32_t temper(uint32_t y) {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    void regenerate() {
        const uint32_t* __restrict o = state_;
        uint32_t* __restrict nw = next_;
        // Same recurrence as the in-place algorithm: entries wrapping past the end
        // read the already-regenerated values.
        for (uint32_t i = 0; i < N - M; ++i) nw[i] = twist(o[i], o[i + 1], o[i + M]);
        for (uint32_t i = N - M; i < 2 * (N - M); ++i) nw[i] = twist(o[i], o[i + 1], nw[i - (N - M)]);
        for (uint32_t i = 2 * (N - M); i < N - 1; ++i) nw[i] = twist(o[i], o[i + 1], nw[i - (N - M)]);
        nw[N - 1] = twist(o[N - 1], nw[0], nw[M - 1]);
        for (uint32_t i = 0; i < N; ++i) state_[i] = nw[i];
    }

    uint32_t state_[N];
    uint32_t next_[N];
    uint32_t pos_;
};

// Uniform float in [0, 1), bit-identical to libstdc++'s uniform_real_distribution<float>
// applied to a 32-bit engine output.
inline val_t uniformFromWord(uint32_t w) {
    val_t r = static_cast<val_t>(w) / 4294967296.0f;
    if (r >= ONE) r = std::nextafter(ONE, ZERO);
    return r;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (octree-accelerated, device side)
// ============================================================================

// Check if a ray's bounding test passes for a node (separating-axis box test)
HD bool rayHitsNodeBox(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    const Vec3& center = n.center;
    const Vec3& halfExtent = n.halfExtent;
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

// Check if a segment between two triangles is blocked by any other triangle
HD bool isRayBlocked(const Vec3& from, const Vec3& to,
                     const DevNode* nodes, const uint32_t* triIdx, const Triangle* tris,
                     uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Depth-first traversal equivalent to Octree::applyToTris from the root.
    // The root's own box is not tested; children are tested before being visited.
    // Blocking is an any-of test, so the visiting order does not affect the result.
    uint32_t stack[OCTREE_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevNode& n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (uint32_t k = 0; k < n.triCount; ++k) {
                const uint32_t idx = triIdx[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int32_t ch = n.child[c];
                if (ch >= 0 && rayHitsNodeBox(nodes[ch], from, to)) {
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

// Cull triangles facing the same direction
HD bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return sMax(ZERO, v.dot(normal) / vNorm);
}

// Random point in triangle from two uniform samples (barycentric coordinates)
HD Vec3 pointFromUV(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Form factor Kij (receiver i, emitter j). `draws` points at the
// DRAWS_PER_PAIR samples for this pair, in the order the sequential code draws them.
HD val_t computeKijFromDraws(uint32_t idxI, uint32_t idxJ,
                             const Triangle* tris, const DevNode* nodes, const uint32_t* triIdx,
                             const float* draws) {
    const Triangle& triI = tris[idxI];
    const Triangle& triJ = tris[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const float* d = draws + r * DRAWS_PER_RAY;
        Vec3 pI = pointFromUV(triI, d[0], d[1]);
        Vec3 pJ = pointFromUV(triJ, d[2], d[3]);

        if (isRayBlocked(pI, pJ, nodes, triIdx, tris, idxI, idxJ)) continue;

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

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Device Memory Helper
// ============================================================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    void allocate(size_t count) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = count;
        if (count_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }

    void zero() {
        if (count_) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    void upload(const std::vector<T>& host) {
        allocate(host.size());
        if (count_) CUDA_CHECK(cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(std::vector<T>& host) const {
        host.resize(count_);
        if (count_) CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;

    // Host-side data
    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<DevNode> octreeNodes;
    std::vector<uint32_t> octreeTris;
    std::vector<val_t> distances;   // Computed distances from source
    std::vector<val_t> radB;        // Reflected radiosity (T x N), filled for validation
    std::vector<val_t> kij;         // Form factors (N x N, row-major), filled for validation

    // Device-side data
    DeviceBuffer<Triangle> dTriangles;
    DeviceBuffer<val_t> dAreas;
    DeviceBuffer<val_t> dRho;
    DeviceBuffer<DevNode> dNodes;
    DeviceBuffer<uint32_t> dTriIdx;
    DeviceBuffer<uint32_t> dRowCount;
    DeviceBuffer<uint64_t> dRowDrawStart;
    DeviceBuffer<val_t> dKij;       // N x N, row-major (receiver row)
    DeviceBuffer<val_t> dW;         // N x N, transposed: dW[j*N+i] = min(Kij*Aj, 1)
    DeviceBuffer<int> dTau;         // N x N, transposed: dTau[j*N+i] = Tau(i,j)
    DeviceBuffer<val_t> dRadE;      // T x N
    DeviceBuffer<val_t> dRadB;      // T x N
    DeviceBuffer<val_t> dSums;      // N x T correlation sums
    DeviceBuffer<val_t> dDist;      // N

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Kernels
// ============================================================================

// Number of non-culled emitters j != i for each receiver i (one block per row)
__global__ void countRowsKernel(const Triangle* tris, uint32_t N, uint32_t* rowCount) {
    __shared__ uint32_t partial;
    const uint32_t i = blockIdx.x;
    if (threadIdx.x == 0) partial = 0;
    __syncthreads();

    uint32_t local = 0;
    for (uint32_t j = threadIdx.x; j < N; j += blockDim.x) {
        if (j != i && !isCulled(tris[i], tris[j])) ++local;
    }
    atomicAdd(&partial, local);
    __syncthreads();
    if (threadIdx.x == 0) rowCount[i] = partial;
}

// Form factors for receiver rows [rowBase, rowBase + gridDim.x).
// One block per row. Emitters are processed in tiles of FF_BLOCK; a block-wide
// exclusive scan gives each non-culled emitter its rank within the row, which
// selects its sample block in `draws` (the chunk of the global random stream).
__global__ void __launch_bounds__(FF_BLOCK)
formFactorChunkKernel(const Triangle* tris, const DevNode* nodes, const uint32_t* triIdx,
                                      const uint32_t* rowCount, const uint64_t* rowDrawStart,
                                      uint64_t chunkStart, const float* draws, uint32_t rowBase,
                                      val_t* kij, uint32_t N, uint32_t* errFlag) {
    __shared__ uint32_t warpSums[FF_BLOCK / 32];
    __shared__ uint32_t tileTotal;

    const uint32_t i = rowBase + blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint64_t drawBase = rowDrawStart[i] - chunkStart;
    const uint32_t expected = rowCount[i];

    uint32_t running = 0;
    for (uint32_t tile = 0; tile < N; tile += FF_BLOCK) {
        const uint32_t j = tile + tid;
        const uint32_t flag = (j < N && j != i && !isCulled(tris[i], tris[j])) ? 1u : 0u;

        // Inclusive warp scan
        uint32_t incl = flag;
        for (uint32_t off = 1; off < 32; off <<= 1) {
            const uint32_t n = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= off) incl += n;
        }
        if (lane == 31) warpSums[warp] = incl;
        __syncthreads();
        if (tid == 0) {
            uint32_t s = 0;
            for (uint32_t w = 0; w < FF_BLOCK / 32; ++w) {
                const uint32_t t = warpSums[w];
                warpSums[w] = s;
                s += t;
            }
            tileTotal = s;
        }
        __syncthreads();

        const uint32_t total = tileTotal;
        const uint32_t rank = running + (incl - flag) + warpSums[warp];
        if (flag) {
            if (rank < expected) {
                const float* d = draws + drawBase + static_cast<uint64_t>(rank) * DRAWS_PER_PAIR;
                kij[static_cast<size_t>(i) * N + j] =
                    computeKijFromDraws(i, j, tris, nodes, triIdx, d);
            } else {
                atomicOr(errFlag, 1u);
            }
        }
        running += total;
        __syncthreads();
    }
    if (running != expected) atomicOr(errFlag, 2u);
}

// Tau for all ordered pairs; also records the minimum off-diagonal delay.
__global__ void timeDelayKernel(const Triangle* tris, uint32_t N, int* tauT, int* minOffDiag) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;  // receiver
    const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;  // emitter
    if (i >= N || j >= N) return;
    int tau = 0;
    if (i != j) {
        tau = computeTau(tris[i], tris[j]);
        atomicMin(minOffDiag, tau);
    }
    tauT[static_cast<size_t>(j) * N + i] = tau;
}

// W[j][i] = min(Kij * Aj, 1), transposed so that threads with consecutive i read contiguously.
__global__ void weightKernel(const val_t* kij, const val_t* areas, val_t* wT, uint32_t N) {
    __shared__ val_t tile[32][33];
    const uint32_t col = blockIdx.x * 32 + threadIdx.x;  // emitter j (input column)
    const uint32_t row = blockIdx.y * 32 + threadIdx.y;  // receiver i (input row)
    for (uint32_t k = 0; k < 32; k += 8) {
        const uint32_t r = row + k;
        if (r < N && col < N) tile[threadIdx.y + k][threadIdx.x] = kij[static_cast<size_t>(r) * N + col];
    }
    __syncthreads();
    for (uint32_t k = 0; k < 32; k += 8) {
        const uint32_t jo = blockIdx.x * 32 + threadIdx.y + k;
        const uint32_t io = blockIdx.y * 32 + threadIdx.x;
        if (jo < N && io < N) {
            wT[static_cast<size_t>(jo) * N + io] = sMin(tile[threadIdx.x][threadIdx.y + k] * areas[jo], ONE);
        }
    }
}

// One timestep of radiosity propagation. Thread i computes row t for receiver i,
// accumulating over emitters j in ascending order (same as the sequential code).
__global__ void simulateStepKernel(uint32_t t, uint32_t N, const val_t* wT, const int* tauT,
                                   const val_t* rho, const val_t* radE, val_t* radB) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;
    for (uint32_t j = 0; j < N; ++j) {
        if (j == i) continue;

        const size_t k = static_cast<size_t>(j) * N + i;
        const int tauij = tauT[k];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        const val_t w = wT[k];
        if (w <= ZERO) continue;

        // Radiosity from source triangle at the time when emission occurred
        const size_t srcTime = t - static_cast<size_t>(tauij);
        const val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += w * radJ;
    }

    radB[static_cast<size_t>(t) * N + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

// Cross-correlation sums: sums[i*T + t] = sum_{tt>=t} radB[tt-t][src] * radB[tt][i]
__global__ void correlationSumKernel(uint32_t N, uint32_t T, uint32_t src,
                                     const val_t* radB, val_t* sums) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(N) * T) return;
    const uint32_t i = static_cast<uint32_t>(idx / T);
    const uint32_t t = static_cast<uint32_t>(idx % T);

    val_t sum = ZERO;
    for (uint32_t tt = t; tt < T; ++tt) {
        const val_t pS = radB[static_cast<size_t>(tt - t) * N + src];
        const val_t pB = radB[static_cast<size_t>(tt) * N + i];
        sum += pS * pB;
    }
    sums[idx] = sum;
}

// Arg-max over lags (first maximum wins, threshold starts at zero)
__global__ void bestDelayKernel(uint32_t N, uint32_t T, const val_t* sums, val_t* distances) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const val_t s = sums[static_cast<size_t>(i) * T + t];
        if (s > maxCorr) {
            maxCorr = s;
            bestT = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

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

    const size_t N = state.numTriangles;
    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration, then flatten for the device
    printf("Building octree...\n");
    {
        Octree octree;
        octree.build(state.triangles);
        state.octreeNodes.clear();
        state.octreeTris.clear();
        flattenOctree(octree, 0, state.octreeNodes, state.octreeTris);
    }

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Set source emission (active for first half of timesteps)
    state.radE.assign(timesteps * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    state.distances.assign(N, ZERO);
    state.radB.clear();
    state.kij.clear();

    // Device allocations and uploads
    state.dTriangles.upload(state.triangles);
    state.dAreas.upload(state.areas);
    state.dRho.upload(state.rho);
    state.dNodes.upload(state.octreeNodes);
    state.dTriIdx.upload(state.octreeTris);
    state.dRowCount.allocate(N);
    state.dRowDrawStart.allocate(N + 1);
    state.dKij.allocate(N * N);
    state.dKij.zero();
    state.dW.allocate(N * N);
    state.dTau.allocate(N * N);
    state.dRadE.upload(state.radE);
    state.dRadB.allocate(timesteps * N);
    state.dRadB.zero();
    state.dDist.allocate(N);
    state.dDist.zero();
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);

    DeviceBuffer<int> dMinTau;
    dMinTau.allocate(1);
    int initMin = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dMinTau.get(), &initMin, sizeof(int), cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((N + 15) / 16, (N + 15) / 16);
    timeDelayKernel<<<grid, block>>>(state.dTriangles.get(), N, state.dTau.get(), dMinTau.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int minTau = INT_MAX;
    CUDA_CHECK(cudaMemcpy(&minTau, dMinTau.get(), sizeof(int), cudaMemcpyDeviceToHost));
    // The wave update reads radiosity of earlier timesteps only if every delay >= 1
    if (N > 1 && minTau < 1) {
        fprintf(stderr, "Zero time delay between distinct triangles; unsupported\n");
        exit(EXIT_FAILURE);
    }
}

struct FormFactorChunk {
    uint32_t rowBegin;
    uint32_t rowEnd;
    uint64_t drawBegin;   // offset in the global draw stream (floats)
    uint64_t drawCount;   // number of floats in this chunk
};

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);

    // 1) Number of non-culled emitters per receiver row, and the draw offset of each row.
    //    The sequential code consumes DRAWS_PER_PAIR samples only for non-culled pairs,
    //    in row-major (i, j) order; this reproduces that layout exactly.
    countRowsKernel<<<N, FF_BLOCK>>>(state.dTriangles.get(), N, state.dRowCount.get());
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint32_t> rowCount;
    state.dRowCount.download(rowCount);
    std::vector<uint64_t> rowDrawStart(N + 1, 0);
    for (uint32_t i = 0; i < N; ++i) {
        rowDrawStart[i + 1] = rowDrawStart[i] + static_cast<uint64_t>(rowCount[i]) * DRAWS_PER_PAIR;
    }
    CUDA_CHECK(cudaMemcpy(state.dRowDrawStart.get(), rowDrawStart.data(),
                          (N + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // 2) Split rows into chunks whose draws fit in one host/device buffer
    std::vector<FormFactorChunk> chunks;
    for (uint32_t r = 0; r < N;) {
        uint32_t r1 = r;
        uint64_t cnt = 0;
        while (r1 < N && (cnt == 0 || cnt + rowDrawStart[r1 + 1] - rowDrawStart[r1] <= CHUNK_DRAWS)) {
            cnt += rowDrawStart[r1 + 1] - rowDrawStart[r1];
            ++r1;
        }
        chunks.push_back({r, r1, rowDrawStart[r], cnt});
        r = r1;
    }

    // 3) Pipeline: host generates the next chunk of the random stream while the GPU
    //    processes the previous one (double buffered).
    constexpr size_t NBUF = 2;
    cudaStream_t streams[NBUF];
    cudaEvent_t done[NBUF];
    float* hBuf[NBUF];
    DeviceBuffer<float> dBuf[NBUF];
    for (size_t b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hBuf[b]), CHUNK_DRAWS * sizeof(float)));
        dBuf[b].allocate(CHUNK_DRAWS);
    }

    DeviceBuffer<uint32_t> dErr;
    dErr.allocate(1);
    dErr.zero();

    MT19937Stream rng(42);
    constexpr size_t WORD_BATCH = size_t(1) << 16;
    std::vector<uint32_t> words(WORD_BATCH);
    uint32_t reported = 0;
    auto report = [&](uint32_t rowsDone) {
        for (uint32_t k = reported + 1; k <= rowsDone; ++k) {
            if (k % 100 == 0 || k == N) printf("  Progress: %u/%u triangles\n", k, N);
        }
        if (rowsDone > reported) reported = rowsDone;
    };

    for (size_t c = 0; c < chunks.size(); ++c) {
        const size_t b = c % NBUF;
        if (c >= NBUF) {
            // Buffer b is free once the kernel of chunk c-NBUF has finished
            CUDA_CHECK(cudaEventSynchronize(done[b]));
            report(chunks[c - NBUF].rowEnd);
        }

        const FormFactorChunk& ch = chunks[c];
        float* dst = hBuf[b];
        for (uint64_t k = 0; k < ch.drawCount; k += WORD_BATCH) {
            const size_t m = static_cast<size_t>(std::min<uint64_t>(WORD_BATCH, ch.drawCount - k));
            rng.fill(words.data(), m);
            for (size_t q = 0; q < m; ++q) dst[k + q] = uniformFromWord(words[q]);
        }
        if (ch.drawCount) {
            CUDA_CHECK(cudaMemcpyAsync(dBuf[b].get(), dst, ch.drawCount * sizeof(float),
                                       cudaMemcpyHostToDevice, streams[b]));
        }

        formFactorChunkKernel<<<ch.rowEnd - ch.rowBegin, FF_BLOCK, 0, streams[b]>>>(
            state.dTriangles.get(), state.dNodes.get(), state.dTriIdx.get(),
            state.dRowCount.get(), state.dRowDrawStart.get(), ch.drawBegin,
            dBuf[b].get(), ch.rowBegin, state.dKij.get(), N, dErr.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(done[b], streams[b]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    report(N);

    uint32_t err = 0;
    CUDA_CHECK(cudaMemcpy(&err, dErr.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (err) {
        fprintf(stderr, "Form factor sample layout mismatch (error flags 0x%x)\n", err);
        exit(EXIT_FAILURE);
    }

    for (size_t b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaEventDestroy(done[b]));
        CUDA_CHECK(cudaFreeHost(hBuf[b]));
    }

    // 4) Pre-weighted, transposed form factors used by the wave propagation
    const dim3 block(32, 8);
    const dim3 grid((N + 31) / 32, (N + 31) / 32);
    weightKernel<<<grid, block>>>(state.dKij.get(), state.dAreas.get(), state.dW.get(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);
    const uint32_t blocks = (N + SIM_BLOCK - 1) / SIM_BLOCK;

    for (uint32_t t = 0; t < T; ++t) {
        simulateStepKernel<<<blocks, SIM_BLOCK>>>(t, N, state.dW.get(), state.dTau.get(),
                                                  state.dRho.get(), state.dRadE.get(),
                                                  state.dRadB.get());
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", static_cast<size_t>(t + 1), state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);
    const uint32_t src = static_cast<uint32_t>(state.sourceIndex);

    if (T > 0) {
        state.dSums.allocate(static_cast<size_t>(N) * T);
        const size_t total = static_cast<size_t>(N) * T;
        const uint32_t blocks = static_cast<uint32_t>((total + 255) / 256);
        correlationSumKernel<<<blocks, 256>>>(N, T, src, state.dRadB.get(), state.dSums.get());
        CUDA_CHECK(cudaGetLastError());
    }
    bestDelayKernel<<<(N + 255) / 256, 256>>>(N, T, state.dSums.get(), state.dDist.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    state.dDist.download(state.distances);
}

// Copy the data needed by validateResults back to the host
void downloadForValidation(SimulationState& state) {
    state.dRadB.download(state.radB);
    state.dKij.download(state.kij);
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

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("===================================\n");
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
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

    // Copy back data for result printing / validation
    if (validate) {
        downloadForValidation(state);
    }

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

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
 *  - Tau and Kij are computed with one CUDA thread per (receiver, emitter) pair.
 *  - Kij ray casting traverses a flattened octree on the device.
 *  - The random stream for Kij is the exact sequential std::mt19937(42) stream of the
 *    original code. The host generates it in row-major pair order (culled pairs consume
 *    no numbers) in chunks, overlapped with device execution, so results are identical.
 *  - Wave propagation is parallel over receivers per timestep. Timesteps remain
 *    sequential (true data dependency). Each receiver's sum over emitters keeps the
 *    original summation order, so results are bit-identical.
 *  - Cross-correlation is parallel over (lag, triangle), followed by a per-triangle argmax.
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
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

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
constexpr val_t VAL_MAX = 3.402823466e+38f;   // std::numeric_limits<float>::max()

constexpr int THREADS_PER_BLOCK = 256;

// Host/device min/max with the exact semantics of std::min / std::max.
HD inline val_t vmin(val_t a, val_t b) { return b < a ? b : a; }
HD inline val_t vmax(val_t a, val_t b) { return a < b ? b : a; }
HD inline val_t vabs(val_t x) { return fabsf(x); }

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
// Octree for Spatial Acceleration (host build)
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
// Flattened Octree (device-friendly layout)
// ============================================================================

struct DNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];      // Child node indices, -1 if absent
    uint32_t triStart;     // Start in leafTris (leaf nodes only)
    uint32_t triCount;     // Number of triangles; > 0 iff leaf
};

// Maximum traversal depth supported by the device stack (7 * depth + 8 entries).
constexpr int MAX_TRAVERSAL_DEPTH = 12;
constexpr int TRAVERSAL_STACK_CAP = 96;

int flattenOctree(const Octree& node, std::vector<DNode>& nodes, std::vector<uint32_t>& leafTris,
                  int depth, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    const int myIdx = static_cast<int>(nodes.size());
    nodes.push_back(DNode{});

    DNode dn{};
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) dn.child[c] = -1;
    dn.triStart = 0;
    dn.triCount = 0;

    if (!node.triangleIndices.empty()) {
        dn.triStart = static_cast<uint32_t>(leafTris.size());
        dn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) leafTris.push_back(static_cast<uint32_t>(idx));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                dn.child[c] = flattenOctree(*node.children[c], nodes, leafTris, depth + 1, maxDepth);
            }
        }
    }
    nodes[myIdx] = dn;
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
// Random Number Generation
// ============================================================================

// Raw 32-bit stream of std::mt19937. Each sample of the original
// std::uniform_real_distribution<float>(0,1) consumes exactly one 32-bit output u and
// equals float(u) * 2^-32 (clamped to nextafter(1,0)); the device performs that conversion
// (see unitFloatFromBits), so the host only stages raw words.
class RandomGenerator {
    std::mt19937 rng;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed) {}
    uint32_t nextBits() { return static_cast<uint32_t>(rng()); }
};

// Bit-exact equivalent of std::uniform_real_distribution<float>(0,1) fed by one mt19937 output
HD inline val_t unitFloatFromBits(uint32_t u) {
    val_t r = static_cast<val_t>(u) * 0x1p-32f;
    return r >= ONE ? 0x1.fffffep-1f : r;
}

// ============================================================================
// Device Memory Helper
// ============================================================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t n) {
        release();
        size_ = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    }
    void zero() {
        if (size_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }
    void upload(const T* src, size_t n) {
        if (n > 0) CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* dst, size_t n) const {
        if (n > 0) CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// Device Geometry Routines (shared semantics with the original host code)
// ============================================================================

// Point in triangle from two uniform samples (same logic as original randomPointInTriangle)
HD inline Vec3 pointInTriangleUV(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-triangle intersection (Möller-Trumbore)
HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (vabs(det) < EPSILON) return VAL_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return VAL_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return VAL_MAX;

    return e2.dot(qvec) * invDet;
}

// Ray vs node bounding box (separating axis), identical to Octree::rayIntersectsBox
HD inline bool nodeRayIntersectsBox(const DNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {vabs(d.x), vabs(d.y), vabs(d.z)};
    const Vec3& he = n.halfExtent;

    if (vabs(c.x) > he.x + ad.x) return false;
    if (vabs(c.y) > he.y + ad.y) return false;
    if (vabs(c.z) > he.z + ad.z) return false;

    if (vabs(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (vabs(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (vabs(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Visibility test between points 'from' (on triangle src) and 'to' (on triangle dst).
// Mirrors isRayBlocked + Octree::applyToTris: root box is not tested, children are tested
// before descending or testing leaf triangles. The result is an "any triangle blocks"
// predicate, so traversal order does not affect the outcome.
__device__ bool isRayBlockedDev(const Vec3& from, const Vec3& to,
                                const DNode* nodes, const uint32_t* leafTris,
                                const Triangle* tris, uint32_t srcIdx, uint32_t dstIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    auto leafBlocked = [&](const DNode& n) -> bool {
        for (uint32_t k = 0; k < n.triCount; ++k) {
            uint32_t idx = leafTris[n.triStart + k];
            if (idx == srcIdx || idx == dstIdx) continue;
            const Triangle& tri = tris[idx];
            val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        }
        return false;
    };

    const DNode& root = nodes[0];
    if (root.triCount > 0) return leafBlocked(root);

    int stack[TRAVERSAL_STACK_CAP];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DNode& n = nodes[stack[--sp]];
        for (int c = 0; c < 8; ++c) {
            const int ch = n.child[c];
            if (ch < 0) continue;
            const DNode& cn = nodes[ch];
            if (!nodeRayIntersectsBox(cn, from, to)) continue;
            if (cn.triCount > 0) {
                if (leafBlocked(cn)) return true;
            } else {
                stack[sp++] = ch;
            }
        }
    }
    return false;
}

// Cosine of angle between vector and normal (clamped at zero)
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return vmax(ZERO, v.dot(normal) / vNorm);
}

// Host-side pair cull test (same expression as the original computeKij)
inline bool formFactorCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

constexpr uint32_t NOT_VALID = 0xFFFFFFFFu;
constexpr size_t DRAWS_PER_PAIR = static_cast<size_t>(NUM_RAYS) * 4;   // u,v for emitter & receiver per ray
constexpr size_t DRAW_CHUNK_FLOATS = size_t(1) << 24;                  // host/device staging chunk (64 MB)

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;

    // Host data
    std::vector<Triangle> triangles;
    std::vector<Vec3> centers;      // Triangle centers
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix, row-major)
    std::vector<val_t> distances;   // Computed distances from source
    Octree octree;                  // Spatial acceleration structure (host build)
    std::vector<DNode> flatNodes;   // Flattened octree
    std::vector<uint32_t> leafTris; // Flattened leaf triangle lists

    // Device data
    DeviceBuffer<Triangle> dTris;
    DeviceBuffer<Vec3> dCenters;
    DeviceBuffer<val_t> dAreas;
    DeviceBuffer<val_t> dRho;
    DeviceBuffer<DNode> dNodes;
    DeviceBuffer<uint32_t> dLeafTris;
    DeviceBuffer<val_t> dKij;       // Form factors, stored transposed: kij[j * N + i] (receiver i, emitter j)
    DeviceBuffer<int> dTau;         // Time delays, stored transposed: tau[j * N + i]
    DeviceBuffer<val_t> dRadE;      // Emission radiosity (T x N)
    DeviceBuffer<val_t> dRadB;      // Reflected radiosity (T x N)
    DeviceBuffer<val_t> dCorr;      // Cross-correlation scores (T x N)
    DeviceBuffer<val_t> dDistances; // Distances (N)

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

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    state.flatNodes.clear();
    state.leafTris.clear();
    int maxDepth = 0;
    flattenOctree(state.octree, state.flatNodes, state.leafTris, 0, maxDepth);
    if (maxDepth > MAX_TRAVERSAL_DEPTH) {
        fprintf(stderr, "Octree depth %d exceeds supported maximum %d\n", maxDepth, MAX_TRAVERSAL_DEPTH);
        exit(EXIT_FAILURE);
    }

    // Triangle centers, areas, reflectivity
    state.centers.resize(N);
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.centers[i] = state.triangles[i].center();
        state.areas[i] = state.triangles[i].area();
    }
    state.rho.assign(N, reflectivity);

    // Source emission (active for first half of timesteps)
    state.radE.assign(T * N, ZERO);
    state.distances.assign(N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Device allocation and upload
    state.dTris.allocate(N);
    state.dTris.upload(state.triangles.data(), N);
    state.dCenters.allocate(N);
    state.dCenters.upload(state.centers.data(), N);
    state.dAreas.allocate(N);
    state.dAreas.upload(state.areas.data(), N);
    state.dRho.allocate(N);
    state.dRho.upload(state.rho.data(), N);
    state.dNodes.allocate(state.flatNodes.size());
    state.dNodes.upload(state.flatNodes.data(), state.flatNodes.size());
    state.dLeafTris.allocate(state.leafTris.size());
    state.dLeafTris.upload(state.leafTris.data(), state.leafTris.size());

    state.dKij.allocate(N * N);
    state.dKij.zero();
    state.dTau.allocate(N * N);
    state.dTau.zero();
    state.dRadE.allocate(T * N);
    state.dRadE.upload(state.radE.data(), T * N);
    state.dRadB.allocate(T * N);
    state.dRadB.zero();
    state.dCorr.allocate(T * N);
    state.dCorr.zero();
    state.dDistances.allocate(N);
    state.dDistances.zero();
}

// ============================================================================
// Precomputation Phase
// ============================================================================

__global__ void tauKernel(int* tauT, const Vec3* centers, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    size_t i = idx % N;   // receiver (fastest varying -> coalesced writes)
    size_t j = idx / N;   // emitter
    int tau = 0;
    if (i != j) {
        val_t dist = (centers[i] - centers[j]).norm();
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[j * N + i] = tau;
}

__global__ void formFactorKernel(val_t* kijT, const uint32_t* draws, const uint32_t* offs,
                                 size_t rows, size_t r0, size_t N,
                                 const Triangle* tris, const DNode* nodes, const uint32_t* leafTris) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= rows * N) return;
    size_t li = p % rows;        // local receiver row (fastest varying)
    size_t j = p / rows;         // emitter
    size_t i = r0 + li;

    uint32_t off = offs[j * rows + li];
    if (off == NOT_VALID) {
        kijT[j * N + i] = ZERO;
        return;
    }

    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];
    const uint32_t* rnd = draws + off;

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = pointInTriangleUV(triI, unitFloatFromBits(rnd[4 * r + 0]), unitFloatFromBits(rnd[4 * r + 1]));
        Vec3 pJ = pointInTriangleUV(triJ, unitFloatFromBits(rnd[4 * r + 2]), unitFloatFromBits(rnd[4 * r + 3]));

        if (isRayBlockedDev(pI, pJ, nodes, leafTris, tris, static_cast<uint32_t>(i), static_cast<uint32_t>(j))) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijT[j * N + i] = kij * INV_NUM_RAYS;
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;
    const size_t total = N * N;
    const unsigned blocks = static_cast<unsigned>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    tauKernel<<<blocks, THREADS_PER_BLOCK>>>(state.dTau.get(), state.dCenters.get(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const Triangle* tris = state.triangles.data();

    // Rows per chunk, bounded by the staging capacity (each valid pair uses DRAWS_PER_PAIR draws)
    const size_t rowsPerChunk = std::max<size_t>(1, std::min(N, DRAW_CHUNK_FLOATS / (DRAWS_PER_PAIR * N)));
    const size_t maxDraws = rowsPerChunk * N * DRAWS_PER_PAIR;

    // Double-buffered pinned staging so host RNG generation overlaps device compute
    uint32_t* hDraws[2];
    uint32_t* hOffs[2];
    cudaEvent_t copied[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hDraws[b]), maxDraws * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hOffs[b]), rowsPerChunk * N * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&copied[b], cudaEventDisableTiming));
    }
    DeviceBuffer<uint32_t> dDraws(maxDraws);
    DeviceBuffer<uint32_t> dOffs(rowsPerChunk * N);

    // Same random stream as the original sequential code: one generator, seed 42,
    // consumed in row-major (i, j) order, 4 draws per ray, 16 rays per valid pair.
    RandomGenerator rng(42);

    size_t chunk = 0;
    for (size_t r0 = 0; r0 < N; r0 += rowsPerChunk, ++chunk) {
        const size_t rows = std::min(rowsPerChunk, N - r0);
        const int b = static_cast<int>(chunk & 1);
        if (chunk >= 2) CUDA_CHECK(cudaEventSynchronize(copied[b]));   // buffer b free for reuse

        uint32_t* draws = hDraws[b];
        uint32_t* offs = hOffs[b];
        size_t nd = 0;
        for (size_t li = 0; li < rows; ++li) {
            const size_t i = r0 + li;
            const Triangle& triI = tris[i];
            for (size_t j = 0; j < N; ++j) {
                uint32_t code = NOT_VALID;
                if (i != j && !formFactorCulled(triI, tris[j])) {
                    code = static_cast<uint32_t>(nd);
                    for (size_t k = 0; k < DRAWS_PER_PAIR; ++k) draws[nd++] = rng.nextBits();
                }
                offs[j * rows + li] = code;
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(dDraws.get(), draws, nd * sizeof(uint32_t), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(dOffs.get(), offs, rows * N * sizeof(uint32_t), cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaEventRecord(copied[b], 0));

        const size_t total = rows * N;
        const unsigned blocks = static_cast<unsigned>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        formFactorKernel<<<blocks, THREADS_PER_BLOCK>>>(state.dKij.get(), dDraws.get(), dOffs.get(),
                                                        rows, r0, N, state.dTris.get(),
                                                        state.dNodes.get(), state.dLeafTris.get());
        CUDA_CHECK(cudaGetLastError());

        for (size_t li = 0; li < rows; ++li) {
            const size_t i = r0 + li;
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, N);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventDestroy(copied[b]));
        CUDA_CHECK(cudaFreeHost(hDraws[b]));
        CUDA_CHECK(cudaFreeHost(hOffs[b]));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One timestep: each thread computes the radiosity of one receiver triangle i.
// Writes only radB[t][i]; reads radB from earlier timesteps (tau >= 1 for distinct triangles).
__global__ void waveStepKernel(size_t t, size_t N, const int* tauT, const val_t* kijT,
                               const val_t* areas, const val_t* rho,
                               const val_t* radE, val_t* radB) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tauT[j * N + i];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = kijT[j * N + i];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += vmin(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const unsigned blocks = static_cast<unsigned>((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    for (size_t t = 0; t < T; ++t) {
        waveStepKernel<<<blocks, THREADS_PER_BLOCK>>>(t, N, state.dTau.get(), state.dKij.get(),
                                                      state.dAreas.get(), state.dRho.get(),
                                                      state.dRadE.get(), state.dRadB.get());
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// corr[t][i] = sum_{tt >= t} radB[tt][i] * radB[tt - t][src]
__global__ void correlationKernel(val_t* corr, const val_t* radB, size_t N, size_t T, size_t src) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= T * N) return;
    size_t t = idx / N;
    size_t i = idx % N;

    val_t sum = ZERO;
    for (size_t tt = t; tt < T; ++tt) {
        val_t pB = radB[tt * N + i];
        val_t pS = radB[(tt - t) * N + src];
        sum += pS * pB;
    }
    corr[t * N + i] = sum;
}

// Per triangle: first lag (in increasing order) achieving the maximum correlation
__global__ void bestLagKernel(val_t* distances, const val_t* corr, size_t N, size_t T) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < T; ++t) {
        val_t sum = corr[t * N + i];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    const size_t totalCorr = T * N;
    if (totalCorr > 0) {
        correlationKernel<<<static_cast<unsigned>((totalCorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                            THREADS_PER_BLOCK>>>(state.dCorr.get(), state.dRadB.get(), N, T, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
    }

    if (N > 0) {
        bestLagKernel<<<static_cast<unsigned>((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                        THREADS_PER_BLOCK>>>(state.dDistances.get(), state.dCorr.get(), N, T);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    state.dDistances.download(state.distances.data(), N);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const std::vector<val_t>& kij,
                     const std::vector<val_t>& radB) {
    printf("\nValidation:\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < N; ++i) {
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t t = 0; t < T; ++t) {
            if (radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int nonZeroKij = 0;
    for (size_t i = 0; i < N * N; ++i) {
        if (kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, N * N,
           100.0f * nonZeroKij / static_cast<val_t>(N * N));

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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }

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
        std::vector<val_t> kijHost(n * n);
        std::vector<val_t> radBHost(t * n);
        state.dKij.download(kijHost.data(), n * n);
        state.dRadB.download(radBHost.data(), t * n);
        if (!validateResults(state, kijHost, radBHost)) {
            return 1;
        }
    }

    return 0;
}

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
 * Mesh generation and octree construction run on the host; all per-pair,
 * per-timestep and per-triangle work runs on the GPU.
 */

#include <cuda_runtime.h>

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
#include <utility>
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
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr uint64_t RNG_SEED = 42;         // Base seed for form factor sampling
constexpr int PROGRESS_INTERVAL = 100;    // Receiver rows per form factor launch (progress granularity)
constexpr unsigned int THREADS_PER_BLOCK = 256;

// ============================================================================
// CUDA Error Handling and Device Memory
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Minimal RAII wrapper for a device array
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = o.ptr_;
            count_ = o.count_;
            o.ptr_ = nullptr;
            o.count_ = 0;
        }
        return *this;
    }

    T* data() const { return ptr_; }
    size_t size() const { return count_; }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        host.resize(count_);
        if (count_ > 0)
            CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (count_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Number of blocks needed to cover n threads (kernels bounds-check their index,
// so a single block is launched even when n == 0)
inline unsigned int numBlocks(size_t n) {
    size_t blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    return static_cast<unsigned int>(blocks > 0 ? blocks : 1);
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
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
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
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

// Flat, device-friendly octree node. Leaves have triCount > 0 and reference
// a range of leafTriangles; internal nodes have triCount == 0.
struct OctreeNode {
    Vec3 center, halfExtent;
    int32_t child[8];       // Index of child node, or -1 if absent
    uint32_t triBegin;      // First entry in leafTriangles (leaves only)
    uint32_t triCount;      // Number of triangles in leaf (0 for internal nodes)
};

// Builds the octree on the host; the result is uploaded to the device for traversal
class Octree {
public:
    std::vector<OctreeNode> nodes;          // nodes[0] is the root
    std::vector<uint32_t> leafTriangles;    // Triangle indices, grouped per leaf

    void build(const std::vector<Triangle>& triangles) {
        nodes.clear();
        leafTriangles.clear();
        if (triangles.empty()) return;

        // Compute bounding box
        Vec3 minBound = triangles[0].a;
        Vec3 maxBound = triangles[0].a;
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
        std::vector<uint32_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = static_cast<uint32_t>(i);

        buildNode(triangles, allIndices, minBound, maxBound);
    }

private:
    int32_t buildNode(const std::vector<Triangle>& triangles, const std::vector<uint32_t>& indices,
                      const Vec3& nodeMin, const Vec3& nodeMax) {
        OctreeNode node{};
        node.halfExtent = (nodeMax - nodeMin) * 0.5f;
        node.center = (nodeMin + nodeMax) * 0.5f;
        for (auto& c : node.child) c = -1;

        int32_t nodeIdx = static_cast<int32_t>(nodes.size());
        nodes.push_back(node);

        auto makeLeaf = [&]() {
            nodes[nodeIdx].triBegin = static_cast<uint32_t>(leafTriangles.size());
            nodes[nodeIdx].triCount = static_cast<uint32_t>(indices.size());
            leafTriangles.insert(leafTriangles.end(), indices.begin(), indices.end());
            return nodeIdx;
        };

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (nodeMax - nodeMin).norm() < MAX_OCTREE_LEAF_SIZE) {
            return makeLeaf();
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = node.halfExtent * 0.5f;
        std::vector<uint32_t> childIndices[8];

        for (uint32_t idx : indices) {
            const Triangle& tri = triangles[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = node.center;
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
            return makeLeaf();
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = node.center;
                Vec3 childMax = node.center;
                childMin.x = (i & 1) ? node.center.x : nodeMin.x;
                childMax.x = (i & 1) ? nodeMax.x : node.center.x;
                childMin.y = (i & 2) ? node.center.y : nodeMin.y;
                childMax.y = (i & 2) ? nodeMax.y : node.center.y;
                childMin.z = (i & 4) ? node.center.z : nodeMin.z;
                childMax.z = (i & 4) ? nodeMax.z : node.center.z;

                int32_t childIdx = buildNode(triangles, childIndices[i], childMin, childMax);
                nodes[nodeIdx].child[i] = childIdx;
            }
        }
        return nodeIdx;
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
// Random Number Generation (counter-based, one independent stream per pair)
// ============================================================================

// SplitMix64 stream. Each (receiver, emitter) pair gets its own stream, so the
// samples do not depend on the order in which pairs are processed.
struct PairRng {
    uint64_t state;

    HD explicit PairRng(uint64_t pairIndex) {
        // Derive a well-mixed per-pair seed from the global seed and pair index
        uint64_t z = (pairIndex + RNG_SEED) * 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        state = z ^ (z >> 31);
    }

    // Uniform value in [0, 1) with 24 bits of precision
    HD val_t rand() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);
    }
};

// Generate a random point inside a triangle using barycentric coordinates
template <typename Rng>
HD Vec3 randomPointInTriangle(const Triangle& t, Rng& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                               const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray intersects a node's bounding box
HD bool rayIntersectsBox(const OctreeNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};
    const Vec3& h = node.halfExtent;

    if (std::abs(c.x) > h.x + ad.x) return false;
    if (std::abs(c.y) > h.y + ad.y) return false;
    if (std::abs(c.z) > h.z + ad.z) return false;

    if (std::abs(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (std::abs(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (std::abs(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Explicit traversal stack size; the octree depth is bounded by MAX_OCTREE_LEAF_SIZE,
// so at most 7 entries per level are live at once.
constexpr int OCTREE_STACK_SIZE = 128;

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const OctreeNode* nodes, const uint32_t* leafTris, const Triangle* tris,
                             uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // Root (not box-tested, matching the original traversal)

    while (sp > 0) {
        const OctreeNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                uint32_t idx = leafTris[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int child = node.child[c];
                if (child >= 0 && rayIntersectsBox(nodes[child], from, to)) {
                    stack[sp++] = child;
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
__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return 0.0f;
    return std::max(0.0f, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
__device__ val_t computeKij(size_t idxI, size_t idxJ,
                            const Triangle* tris,
                            const OctreeNode* nodes, const uint32_t* leafTris,
                            PairRng& rng) {
    const Triangle& triI = tris[idxI];
    const Triangle& triJ = tris[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return 0.0f;

    val_t kij = 0.0f;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, nodes, leafTris, tris,
                         static_cast<uint32_t>(idxI), static_cast<uint32_t>(idxJ))) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Matrices indexed by (receiver i, emitter j) are stored transposed: element (i, j)
// lives at j * N + i. Consecutive threads handle consecutive receivers, so this
// layout makes both the writes in the precomputation kernels and the reads in the
// propagation kernel coalesced.
struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;    // Host copy of the mesh (used to build the octree)
    std::vector<val_t> distances;       // Computed distances from source (host copy of results)
    Octree octree;                      // Host-side spatial acceleration structure

    size_t sourceIndex;

    // Device-resident data
    DeviceBuffer<Triangle> d_triangles;
    DeviceBuffer<OctreeNode> d_nodes;
    DeviceBuffer<uint32_t> d_leafTriangles;
    DeviceBuffer<val_t> d_areas;        // Area of each triangle
    DeviceBuffer<val_t> d_rho;          // Reflectivity (0.0 to 1.0)
    DeviceBuffer<val_t> d_kij;          // Form factors (N x N, transposed)
    DeviceBuffer<int> d_tau;            // Time delays (N x N, transposed)
    DeviceBuffer<val_t> d_radE;         // Emission radiosity (T x N matrix)
    DeviceBuffer<val_t> d_radB;         // Reflected radiosity (T x N matrix)
    DeviceBuffer<val_t> d_corr;         // Cross-correlation values (N x T matrix)
    DeviceBuffer<val_t> d_distances;
};

// ============================================================================
// GPU Kernels
// ============================================================================

// Form factors for a batch of receivers [rowBegin, rowBegin + rowCount) against
// all emitters. Thread k handles receiver rowBegin + k % rowCount and emitter k / rowCount.
__global__ void formFactorKernel(size_t rowBegin, size_t rowCount, size_t N,
                                 const Triangle* tris, const OctreeNode* nodes,
                                 const uint32_t* leafTris, val_t* kijT) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= rowCount * N) return;

    size_t i = rowBegin + k % rowCount;
    size_t j = k / rowCount;

    val_t kij = 0.0f;
    if (i != j) {
        PairRng rng(static_cast<uint64_t>(i) * N + j);
        kij = computeKij(i, j, tris, nodes, leafTris, rng);
    }
    kijT[j * N + i] = kij;
}

// Time delays for all (receiver, emitter) pairs
__global__ void timeDelayKernel(size_t N, const Triangle* tris, int* tauT) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= N * N) return;

    size_t i = k % N;
    size_t j = k / N;

    tauT[j * N + i] = (i == j) ? 0 : computeTau(tris[i], tris[j]);
}

// One wave propagation timestep: radB[t, i] for all receivers i. Every
// contribution uses radB at an earlier time (tau >= 1 for i != j), so all
// receivers within a timestep are independent.
__global__ void propagateKernel(int t, size_t N, const int* tauT, const val_t* kijT,
                                const val_t* areas, const val_t* rho,
                                const val_t* radE, val_t* radB) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = 0.0f;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tauT[j * N + i];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        val_t kij = kijT[j * N + i];
        if (kij <= 0.0f) continue;

        // Get radiosity from source triangle at time when emission occurred
        int srcTime = t - tauij;
        val_t radJ = radB[static_cast<size_t>(srcTime) * N + j];
        if (radJ <= 0.0f) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += std::min(kij * areas[j], 1.0f) * radJ;
    }

    // Update radiosity: reflection + emission
    radB[static_cast<size_t>(t) * N + i] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

// Unnormalized cross-correlation for every (triangle i, lag t) pair:
// corr[i * T + t] = sum_{tt >= t} radB[tt, i] * radB[tt - t, source]
__global__ void correlationKernel(size_t N, size_t T, size_t sourceIndex,
                                  const val_t* radB, val_t* corr) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= N * T) return;

    size_t i = k / T;
    size_t t = k % T;

    val_t sum = 0.0f;
    for (size_t tt = t; tt < T; ++tt) {
        val_t pB = radB[tt * N + i];
        val_t pS = radB[(tt - t) * N + sourceIndex];
        sum += pS * pB;
    }
    corr[k] = sum;
}

// Pick the lag with the largest correlation for each triangle (first maximum wins)
__global__ void bestLagKernel(size_t N, size_t T, const val_t* corr, val_t* distances) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = 0.0f;
    int bestT = 0;
    for (size_t t = 0; t < T; ++t) {
        val_t sum = corr[i * T + t];
        if (sum > maxCorr) {
            maxCorr = sum;
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    const size_t N = state.numTriangles;
    const size_t T = timesteps;

    // Initialize areas
    std::vector<val_t> areas(N);
    for (size_t i = 0; i < N; ++i) {
        areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    std::vector<val_t> rho(N, reflectivity);

    // Set source emission (active for first half of timesteps)
    std::vector<val_t> radE(T * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[t * N + state.sourceIndex] = 1.0f;
    }

    // Upload geometry and acceleration structure
    state.d_triangles = DeviceBuffer<Triangle>(N);
    state.d_triangles.upload(state.triangles);
    state.d_nodes = DeviceBuffer<OctreeNode>(state.octree.nodes.size());
    state.d_nodes.upload(state.octree.nodes);
    state.d_leafTriangles = DeviceBuffer<uint32_t>(state.octree.leafTriangles.size());
    state.d_leafTriangles.upload(state.octree.leafTriangles);

    // Upload per-triangle data and source emission
    state.d_areas = DeviceBuffer<val_t>(N);
    state.d_areas.upload(areas);
    state.d_rho = DeviceBuffer<val_t>(N);
    state.d_rho.upload(rho);
    state.d_radE = DeviceBuffer<val_t>(T * N);
    state.d_radE.upload(radE);

    // Allocate matrices and results (zeroed to match the original initial state)
    state.d_kij = DeviceBuffer<val_t>(N * N);
    state.d_kij.zero();
    state.d_tau = DeviceBuffer<int>(N * N);
    state.d_tau.zero();
    state.d_radB = DeviceBuffer<val_t>(T * N);
    state.d_radB.zero();
    state.d_corr = DeviceBuffer<val_t>(N * T);
    state.d_distances = DeviceBuffer<val_t>(N);
    state.distances.assign(N, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;

    // Process receivers in batches so progress can be reported as before
    for (size_t i0 = 0; i0 < N; i0 += PROGRESS_INTERVAL) {
        size_t rows = std::min(static_cast<size_t>(PROGRESS_INTERVAL), N - i0);

        formFactorKernel<<<numBlocks(rows * N), THREADS_PER_BLOCK>>>(
            i0, rows, N,
            state.d_triangles.data(), state.d_nodes.data(), state.d_leafTriangles.data(),
            state.d_kij.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        size_t done = i0 + rows;
        if (done % PROGRESS_INTERVAL == 0 || done == N) {
            printf("  Progress: %zu/%zu triangles\n", done, N);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;

    timeDelayKernel<<<numBlocks(N * N), THREADS_PER_BLOCK>>>(
        N, state.d_triangles.data(), state.d_tau.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    for (size_t t = 0; t < T; ++t) {
        propagateKernel<<<numBlocks(N), THREADS_PER_BLOCK>>>(
            static_cast<int>(t), N,
            state.d_tau.data(), state.d_kij.data(), state.d_areas.data(), state.d_rho.data(),
            state.d_radE.data(), state.d_radB.data());
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    correlationKernel<<<numBlocks(N * T), THREADS_PER_BLOCK>>>(
        N, T, state.sourceIndex, state.d_radB.data(), state.d_corr.data());
    CUDA_CHECK(cudaGetLastError());

    bestLagKernel<<<numBlocks(N), THREADS_PER_BLOCK>>>(
        N, T, state.d_corr.data(), state.d_distances.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    state.d_distances.download(state.distances);
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

    // Copy radiosity and form factors back for inspection
    std::vector<val_t> radB;
    state.d_radB.download(radB);

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (radB[t * state.numTriangles + i] > EPSILON) {
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
    std::vector<val_t> kij;
    state.d_kij.download(kij);

    int nonZeroKij = 0;
    for (size_t i = 0; i < kij.size(); ++i) {
        if (kij[i] > EPSILON) nonZeroKij++;
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

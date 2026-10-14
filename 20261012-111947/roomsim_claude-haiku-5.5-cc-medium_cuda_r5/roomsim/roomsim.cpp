/**
 * Room Response Simulation Benchmark
 *
 * This is a CUDA-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate between
 * surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization: each form factor (Kij) is a block of threads, one per (receiver, ray)
 * pair, and visibility is tested with a flattened octree on the device. Time delays, the
 * wave propagation timestep, and the cross-correlation are all device kernels. The random
 * sample stream is produced on the host in exactly the order the sequential code consumed
 * it (and overlapped with the GPU), and per-element sums keep the sequential order, so the
 * results are identical to the serial version.
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
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Owning device buffer (move-only)
template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceArray() { release(); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceArray& operator=(DeviceArray&& o) noexcept {
        if (this != &o) {
            release();
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
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

private:
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }

    T* ptr_ = nullptr;
    size_t count_ = 0;
};

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
constexpr val_t NO_HIT = std::numeric_limits<val_t>::max();

constexpr unsigned BLOCK_SIZE = 256;
constexpr unsigned RECV_PER_BLOCK = BLOCK_SIZE / NUM_RAYS;  // Receivers per form-factor block
static_assert(RECV_PER_BLOCK * NUM_RAYS == BLOCK_SIZE, "form factor block must be full");
constexpr size_t DRAWS_PER_PAIR = 4 * NUM_RAYS;          // 2 points x (u, v) per ray
constexpr uint64_t RNG_CHUNK_FLOATS = 1ull << 25;        // Host staging granularity

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

    Vec3 center() const { return (a + b + c) / 3.0f; }
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

// Device-friendly octree node, stored in preorder. A node with triCount > 0 is a leaf.
// `skip` is the index just past this node's subtree, so a traversal can jump over a
// subtree without a stack (children of an internal node start at index + 1).
struct DeviceNode {
    Vec3 center;
    Vec3 halfExtent;
    uint32_t skip;
    uint32_t triBegin;
    uint32_t triCount;
};

// Triangle copy stored in leaf order, so a leaf's triangles are contiguous in memory
struct LeafTri {
    Vec3 a, b, c;
    uint32_t idx;      // Index into the global triangle list
};

struct FlatOctree {
    std::vector<DeviceNode> nodes;   // nodes[0] is the root
    std::vector<LeafTri> leafTris;   // triangles of all leaves, concatenated
};

// Copy the pointer-based octree into preorder arrays
void flattenOctree(const Octree& node, const std::vector<Triangle>& triangles, FlatOctree& flat) {
    const size_t self = flat.nodes.size();
    DeviceNode dn{};
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    dn.triBegin = static_cast<uint32_t>(flat.leafTris.size());
    dn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) {
        const Triangle& tri = triangles[idx];
        flat.leafTris.push_back({tri.a, tri.b, tri.c, static_cast<uint32_t>(idx)});
    }
    flat.nodes.push_back(dn);

    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) flattenOctree(*node.children[c], triangles, flat);
    }
    flat.nodes[self].skip = static_cast<uint32_t>(flat.nodes.size());
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
// Random Number Generation (host; the stream order defines the results)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (Octree-accelerated, device)
// ============================================================================

// Slab-style test of a segment against a node's bounding box
__device__ bool rayIntersectsBox(const DeviceNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - nd.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = nd.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Read-only views of the scene used by the kernels
struct SceneView {
    const Triangle* tris;
    const DeviceNode* nodes;
    const LeafTri* leafTris;
    uint32_t numNodes;
};

// Check if the segment between two triangles is blocked by any other triangle.
// Stackless preorder walk: a node's box gates its subtree (the root's box is not tested).
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const SceneView& sc,
                             size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    uint32_t n = 0;
    while (n < sc.numNodes) {
        const DeviceNode& nd = sc.nodes[n];
        if (nd.triCount > 0) {
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                const LeafTri& lt = sc.leafTris[nd.triBegin + k];
                if (lt.idx == srcTriIdx || lt.idx == dstTriIdx) continue;

                val_t dist = rayTriangleIntersect(from, dirNorm, lt.a, lt.b, lt.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;  // Ray is blocked
            }
            n = nd.skip;
        } else {
            // Descend into the children if this box is hit, otherwise skip the subtree
            n = (n == 0 || rayIntersectsBox(nd, from, to)) ? n + 1 : nd.skip;
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Triangles facing the same direction exchange no energy (form factor is zero)
__host__ __device__ bool isCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Map uniform samples (u, v) to a point in the triangle
__device__ Vec3 samplePointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// One ray's contribution to Kij between triangle i (receiver) and j (emitter).
// `d` holds 4 samples: (u, v) for the point on i, then (u, v) for the point on j.
// Returns 0 for rays that do not contribute.
__device__ val_t rayContribution(const Triangle& triI, const Triangle& triJ, size_t idxI, size_t idxJ,
                                 const SceneView& sc, const val_t* d) {
    Vec3 pI = samplePointInTriangle(triI, d[0], d[1]);
    Vec3 pJ = samplePointInTriangle(triJ, d[2], d[3]);

    if (isRayBlocked(pI, pJ, sc, idxI, idxJ)) return ZERO;

    Vec3 v = pJ - pI;
    val_t distSqr = v.squaredNorm();
    if (distSqr < EPSILON) return ZERO;

    val_t cosPhiI = cosPhi(v, triI.normal());
    val_t cosPhiJ = cosPhi(-v, triJ.normal());

    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) return ZERO;

    return (cosPhiI * cosPhiJ) / (PI * distSqr);
}

// Count the traced pairs (i != j and not culled) in each group of RECV_PER_BLOCK receivers.
// Block (x, y) with threads (tx, ty) covers receiver j = x * RECV_PER_BLOCK + tx of row
// i = y * blockDim.y + ty.
__global__ void countTracedPairsKernel(const Triangle* tris, size_t N, size_t groups, uint32_t* groupCounts) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * RECV_PER_BLOCK + threadIdx.x;
    const bool valid = i < N && j < N;

    uint32_t flag = (valid && j != i && !isCulled(tris[i], tris[j])) ? 1u : 0u;
    // Reduce over the RECV_PER_BLOCK lanes of this row (segments of 16 lanes never straddle rows)
    for (unsigned off = RECV_PER_BLOCK / 2; off > 0; off >>= 1) {
        flag += __shfl_down_sync(0xffffffffu, flag, off, RECV_PER_BLOCK);
    }
    if (threadIdx.x == 0 && i < N) groupCounts[i * groups + blockIdx.x] = flag;
}

// Grid (x, y) = (receiver group, row offset from rowStart); block = RECV_PER_BLOCK receivers
// x NUM_RAYS rays. Thread (p, r) traces ray r for receiver p of the group. Each traced pair's
// draws start at its group's offset (within the chunk) plus its rank among the group's traced
// pairs. Ray contributions are summed in ray order, matching the sequential accumulation.
// Writes kijT[j * N + i] so the simulation can read coalesced across receivers.
__global__ void formFactorKernel(SceneView sc, size_t N, size_t groups, const val_t* draws,
                                 const uint64_t* rowOff, const uint32_t* groupDrawOff,
                                 uint64_t chunkOff, size_t rowStart, val_t* kijT) {
    __shared__ uint32_t traced[RECV_PER_BLOCK];
    __shared__ val_t terms[RECV_PER_BLOCK][NUM_RAYS];

    const size_t i = rowStart + blockIdx.y;
    const size_t group = blockIdx.x;
    const unsigned p = threadIdx.x / NUM_RAYS;
    const unsigned r = threadIdx.x % NUM_RAYS;
    const size_t j = group * RECV_PER_BLOCK + p;
    const bool valid = j < N;
    const Triangle& triI = sc.tris[i];
    const bool isTraced = valid && j != i && !isCulled(triI, sc.tris[j]);

    if (r == 0) traced[p] = isTraced ? 1u : 0u;
    __syncthreads();

    // Rank of this pair among the traced pairs of its group
    uint32_t rank = 0;
    for (unsigned q = 0; q < p; ++q) rank += traced[q];

    val_t term = ZERO;
    if (isTraced) {
        const val_t* draw = draws + (rowOff[i] - chunkOff) + groupDrawOff[i * groups + group]
                            + static_cast<size_t>(rank) * DRAWS_PER_PAIR + 4 * r;
        term = rayContribution(triI, sc.tris[j], i, j, sc, draw);
    }
    terms[p][r] = term;
    __syncthreads();

    if (r == 0 && valid) {
        val_t kij = ZERO;
        for (unsigned k = 0; k < NUM_RAYS; ++k) kij += terms[p][k];
        kijT[j * N + i] = kij * INV_NUM_RAYS;
    }
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__device__ int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// tauT[j * N + i] = delay from triangle j to triangle i
__global__ void computeTauKernel(const Vec3* centers, size_t N, int* tauT) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    tauT[j * N + i] = (i == j) ? 0 : computeTau(centers[i], centers[j]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;

    // Host data
    std::vector<Triangle> triangles;
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), copied back after simulation
    std::vector<val_t> distances;   // Computed distances from source, copied back after computation

    Octree octree;                  // Spatial acceleration structure (host build)

    // Device data
    DeviceArray<Triangle> dTriangles;
    DeviceArray<Vec3> dCenters;
    DeviceArray<DeviceNode> dNodes;
    DeviceArray<LeafTri> dLeafTris;
    DeviceArray<val_t> dAreas;      // Area of each triangle
    DeviceArray<val_t> dRho;        // Reflectivity (0.0 to 1.0)
    DeviceArray<val_t> dKij;        // Form factors, transposed (kijT[j * N + i] = Kij)
    DeviceArray<int> dTau;          // Time delays, transposed (tauT[j * N + i] = Tau_ij)
    DeviceArray<val_t> dRadE;       // Emission radiosity (T x N matrix)
    DeviceArray<val_t> dRadB;       // Reflected radiosity (T x N matrix)
    DeviceArray<val_t> dDistances;  // Computed distances from source

    SceneView sceneView() const {
        return {dTriangles.data(), dNodes.data(), dLeafTris.data(),
                static_cast<uint32_t>(dNodes.size())};
    }
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
    const size_t T = timesteps;

    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    FlatOctree flat;
    flattenOctree(state.octree, state.triangles, flat);

    // Host-side per-triangle data
    std::vector<Vec3> centers(N);
    std::vector<val_t> areas(N);
    for (size_t i = 0; i < N; ++i) {
        centers[i] = state.triangles[i].center();
        areas[i] = state.triangles[i].area();
    }
    std::vector<val_t> rho(N, reflectivity);

    // Source emission (active for first half of timesteps)
    std::vector<val_t> radE(T * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[t * N + state.sourceIndex] = 1.0f;
    }

    // Upload to device
    state.dTriangles = DeviceArray<Triangle>(N);
    state.dTriangles.upload(state.triangles);
    state.dCenters = DeviceArray<Vec3>(N);
    state.dCenters.upload(centers);
    state.dNodes = DeviceArray<DeviceNode>(flat.nodes.size());
    state.dNodes.upload(flat.nodes);
    state.dLeafTris = DeviceArray<LeafTri>(flat.leafTris.size());
    state.dLeafTris.upload(flat.leafTris);
    state.dAreas = DeviceArray<val_t>(N);
    state.dAreas.upload(areas);
    state.dRho = DeviceArray<val_t>(N);
    state.dRho.upload(rho);
    state.dRadE = DeviceArray<val_t>(T * N);
    state.dRadE.upload(radE);
    state.dRadB = DeviceArray<val_t>(T * N);
    state.dRadB.zero();
    state.dDistances = DeviceArray<val_t>(N);
    state.dKij = DeviceArray<val_t>(N * N);
    state.dTau = DeviceArray<int>(N * N);
    state.radB.assign(T * N, ZERO);
    state.distances.assign(N, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Kernel for the non-zero count of Kij (exact same threshold as the reference validation)
__global__ void countAboveEpsilonKernel(const val_t* data, size_t count, unsigned long long* total) {
    __shared__ unsigned long long partial[BLOCK_SIZE];
    unsigned long long local = 0;
    for (size_t k = blockIdx.x * BLOCK_SIZE + threadIdx.x; k < count; k += gridDim.x * BLOCK_SIZE) {
        if (data[k] > EPSILON) ++local;
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (unsigned s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
              static_cast<unsigned>((N + block.y - 1) / block.y));
    computeTauKernel<<<grid, block>>>(state.dCenters.data(), N, state.dTau.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const size_t groups = (N + RECV_PER_BLOCK - 1) / RECV_PER_BLOCK;

    // Pass 1: count traced pairs per (row, receiver group). Only traced pairs consume random
    // numbers in the sequential reference, so this fixes each pair's position in the RNG stream.
    const dim3 countBlock(RECV_PER_BLOCK, NUM_RAYS);
    DeviceArray<uint32_t> dGroupCounts(N * groups);
    countTracedPairsKernel<<<dim3(static_cast<unsigned>(groups),
                                  static_cast<unsigned>((N + countBlock.y - 1) / countBlock.y)),
                             countBlock>>>(state.dTriangles.data(), N, groups, dGroupCounts.data());
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> groupCounts;
    dGroupCounts.download(groupCounts);

    // rowOff[i]: draws consumed before row i. groupDrawOff[i * groups + g]: draws of row i
    // consumed before group g, relative to rowOff[i].
    std::vector<uint64_t> rowOff(N + 1, 0);
    std::vector<uint32_t> groupDrawOff(N * groups);
    for (size_t i = 0; i < N; ++i) {
        uint64_t acc = 0;
        for (size_t g = 0; g < groups; ++g) {
            groupDrawOff[i * groups + g] = static_cast<uint32_t>(acc);
            acc += static_cast<uint64_t>(groupCounts[i * groups + g]) * DRAWS_PER_PAIR;
        }
        rowOff[i + 1] = rowOff[i] + acc;
    }
    auto rowBegin = [&](size_t row) { return rowOff[row]; };

    uint64_t maxRowDraws = 0;
    for (size_t i = 0; i < N; ++i) {
        maxRowDraws = std::max(maxRowDraws, rowOff[i + 1] - rowOff[i]);
    }
    DeviceArray<uint64_t> dRowOff(N + 1);
    dRowOff.upload(rowOff);
    DeviceArray<uint32_t> dGroupDrawOff(N * groups);
    dGroupDrawOff.upload(groupDrawOff);

    // Pass 2: stream draws to the device in row chunks. The host generates the draws in
    // the same order the sequential code consumed them, while the previous chunk's kernel
    // runs. Two pinned staging buffers allow the host to run ahead of the copies.
    const uint64_t capacity = std::max(RNG_CHUNK_FLOATS, maxRowDraws);
    float* hostStage[2] = {nullptr, nullptr};
    cudaEvent_t staged[2];
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostStage[s]), capacity * sizeof(val_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&staged[s], cudaEventDisableTiming));
    }
    DeviceArray<val_t> dDraws(capacity);

    RandomGenerator rng(42);
    size_t rowStart = 0;
    int chunk = 0;
    while (rowStart < N) {
        size_t rowEnd = rowStart + 1;
        while (rowEnd < N && rowBegin(rowEnd + 1) - rowBegin(rowStart) <= capacity) ++rowEnd;

        const uint64_t chunkFloats = rowBegin(rowEnd) - rowBegin(rowStart);
        const int slot = chunk % 2;
        if (chunk >= 2) CUDA_CHECK(cudaEventSynchronize(staged[slot]));  // Staging buffer is free

        val_t* stage = hostStage[slot];
        for (uint64_t k = 0; k < chunkFloats; ++k) stage[k] = rng.rand();
        CUDA_CHECK(cudaMemcpyAsync(dDraws.data(), stage, chunkFloats * sizeof(val_t),
                                   cudaMemcpyHostToDevice, 0));
        CUDA_CHECK(cudaEventRecord(staged[slot], 0));

        formFactorKernel<<<dim3(static_cast<unsigned>(groups), static_cast<unsigned>(rowEnd - rowStart)),
                           BLOCK_SIZE>>>(
            state.sceneView(), N, groups, dDraws.data(), dRowOff.data(), dGroupDrawOff.data(),
            rowBegin(rowStart), rowStart, state.dKij.data());
        CUDA_CHECK(cudaGetLastError());

        for (size_t i = rowStart; i < rowEnd; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, N);
            }
        }

        rowStart = rowEnd;
        ++chunk;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaFreeHost(hostStage[s]));
        CUDA_CHECK(cudaEventDestroy(staged[s]));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One timestep t: thread i computes radB[t][i]. Reads only radB at earlier times
// (tau >= 1 for distinct triangles), so all i are independent within a timestep.
__global__ void simulationStepKernel(size_t t, size_t N, size_t sourceIndex,
                                     const int* tauT, const val_t* kijT, const val_t* areas,
                                     const val_t* rho, const val_t* radE, val_t* radB) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;
    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        const int tauij = tauT[j * N + i];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        const val_t kij = kijT[j * N + i];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        const size_t srcTime = t - static_cast<size_t>(tauij);
        const val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const unsigned blocks = static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (size_t t = 0; t < T; ++t) {
        simulationStepKernel<<<blocks, BLOCK_SIZE>>>(
            t, N, state.sourceIndex, state.dTau.data(), state.dKij.data(), state.dAreas.data(),
            state.dRho.data(), state.dRadE.data(), state.dRadB.data());
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    state.dRadB.download(state.radB);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// One thread per triangle i: lag search over t, correlating radB[.][i] with the source
// signal. Sums and the first-maximum tie-breaking follow the sequential order exactly.
__global__ void distanceKernel(size_t N, size_t T, size_t sourceIndex,
                               const val_t* radB, val_t* distances) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;

        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }

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

    distanceKernel<<<static_cast<unsigned>((N + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
        N, state.numTimesteps, state.sourceIndex, state.dRadB.data(), state.dDistances.data());
    CUDA_CHECK(cudaGetLastError());
    state.dDistances.download(state.distances);
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
    DeviceArray<unsigned long long> dCount(1);
    CUDA_CHECK(cudaMemset(dCount.data(), 0, sizeof(unsigned long long)));
    countAboveEpsilonKernel<<<1024, BLOCK_SIZE>>>(state.dKij.data(), state.dKij.size(), dCount.data());
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned long long> countHost;
    dCount.download(countHost);
    const int nonZeroKij = static_cast<int>(countHost[0]);

    const size_t N2 = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, N2,
           100.0f * nonZeroKij / static_cast<val_t>(N2));

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

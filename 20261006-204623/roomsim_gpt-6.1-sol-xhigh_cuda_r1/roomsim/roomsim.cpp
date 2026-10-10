/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
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
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
// CUDA storage, random sampling, and geometry kernels
// ============================================================================

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceBuffer {
    T* pointer = nullptr;
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflow");
        }
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "cudaMalloc");
    }
    T* get() const { return pointer; }
    void upload(const T* data, size_t count) {
        if (count) checkCuda(cudaMemcpy(pointer, data, count * sizeof(T), cudaMemcpyHostToDevice), "CUDA upload");
    }
    void download(T* data, size_t count) const {
        if (count) checkCuda(cudaMemcpy(data, pointer, count * sizeof(T), cudaMemcpyDeviceToHost), "CUDA download");
    }
};

class CudaStream {
    cudaStream_t value = nullptr;
public:
    CudaStream() { checkCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "Create CUDA stream"); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() { cudaStreamDestroy(value); }
    cudaStream_t get() const { return value; }
};

class CudaEvent {
    cudaEvent_t value = nullptr;
public:
    CudaEvent() { checkCuda(cudaEventCreateWithFlags(&value, cudaEventDisableTiming), "Create CUDA event"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(value); }
    void record(cudaStream_t stream) { checkCuda(cudaEventRecord(value, stream), "Record CUDA event"); }
    void wait(cudaStream_t stream) { checkCuda(cudaStreamWaitEvent(stream, value, 0), "Wait for CUDA event"); }
};

// Preorder nodes carry an escape index, so rays need neither recursion nor a
// per-thread traversal stack. Leaf triangle order matches the original octree.
struct DeviceOctreeNode {
    Vec3 center, halfExtent;
    uint32_t escape, begin, count;
};

void flattenOctree(const Octree& tree, std::vector<DeviceOctreeNode>& nodes,
                   std::vector<idx_t>& indices) {
    size_t nodeIndex = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size())});
    for (size_t index : tree.triangleIndices) indices.push_back(static_cast<idx_t>(index));
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[nodeIndex].escape = static_cast<uint32_t>(nodes.size());
}

constexpr int GPU_THREADS = 256;
// Bound scratch memory independently of mesh size (64 MiB of samples per buffer).
constexpr size_t PAIR_BATCH_SIZE = 262144;

struct DeviceSimulation {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<DeviceOctreeNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<val_t> areas, rho, kij, radE, radB, distances;
    DeviceBuffer<int> tau;
    // Pair matrices use the original row-major [receiver][emitter] layout.
    explicit DeviceSimulation(size_t n, size_t t, size_t nodeCount, size_t indexCount)
        : triangles(n), nodes(nodeCount), indices(indexCount), areas(n), rho(n),
          kij(n * n), radE(t * n), radB(t * n), distances(n), tau(n * n) {}
};

__device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2,
                               const DeviceOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir, const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DeviceOctreeNode* nodes, const idx_t* indices,
                             const Triangle* triangles, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;
    uint32_t nodeIndex = 0;
    uint32_t end = nodes[0].escape;
    while (nodeIndex < end) {
        const DeviceOctreeNode& node = nodes[nodeIndex];
        // The root is visited unconditionally, as in Octree::applyToTris.
        if (nodeIndex && !rayIntersectsBox(from, to, node)) {
            nodeIndex = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.count; ++k) {
            idx_t index = indices[node.begin + k];
            if (index == src || index == dst) continue;
            val_t dist = rayTriangleIntersect(from, dirNorm, triangles[index]);
            if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

__device__ Vec3 randomPointInTriangle(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t length = v.norm();
    if (length <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / length;
    return cosine > ZERO ? cosine : ZERO;
}

__global__ void timeDelaysKernel(const Triangle* triangles, int* tau, size_t n) {
    size_t count = n * n;
    for (size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < count; pair += static_cast<size_t>(gridDim.x) * blockDim.x) {
        size_t i = pair / n, j = pair % n;
        tau[pair] = i == j ? 0 : static_cast<int>(ceilf(
            (triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
    }
}

// Each block ranks just the pairs that consume random numbers in the original
// row-major loop. A small device scan supplies the offsets between blocks.
__global__ void rankPairsKernel(const Triangle* triangles, size_t n, size_t first,
                                size_t count, int* ranks, int* blockCounts) {
    using Scan = cub::BlockScan<int, GPU_THREADS>;
    __shared__ typename Scan::TempStorage storage;
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t pair = first + local;
    size_t i = pair / n, j = pair % n;
    int eligible = local < count && i != j &&
        !(triangles[i].normal().dot(triangles[j].normal()) > 0.99f);
    int rank, total;
    Scan(storage).ExclusiveSum(eligible, rank, total);
    if (local < count) ranks[local] = eligible ? rank : -1;
    if (threadIdx.x == 0) {
        blockCounts[blockIdx.x] = total;
        if (blockIdx.x + 1 == gridDim.x) blockCounts[gridDim.x] = 0;
    }
}

// The MT19937 twist has dependencies between three regions, but all 227
// updates within a region are parallel. Two shared buffers avoid in-place races.
__device__ void twistRandomState(uint32_t*& current, uint32_t*& next) {
    if (threadIdx.x < 227) {
        int k = threadIdx.x;
        uint32_t y = (current[k] & 0x80000000u) | (current[k + 1] & 0x7fffffffu);
        next[k] = current[k + 397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    __syncthreads();
    if (threadIdx.x < 227) {
        int k = threadIdx.x + 227;
        uint32_t y = (current[k] & 0x80000000u) | (current[k + 1] & 0x7fffffffu);
        next[k] = next[k - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    __syncthreads();
    if (threadIdx.x < 169) {
        int k = threadIdx.x + 454;
        uint32_t y = (current[k] & 0x80000000u) | (current[k + 1] & 0x7fffffffu);
        next[k] = next[k - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    if (threadIdx.x == 0) {
        uint32_t y = (current[623] & 0x80000000u) | (next[0] & 0x7fffffffu);
        next[623] = next[396] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }
    __syncthreads();
    uint32_t* swap = current; current = next; next = swap;
}

constexpr size_t RANDOM_CHUNK_SIZE = 624 * 16;

// Save sparse checkpoints of the original sequential stream. This block only
// advances states; the much larger sample output is generated by many blocks
// in the next kernel. A chunk spans whole MT periods, so all checkpoints share
// the same cursor. Unused words survive between pair batches.
__global__ void randomCheckpointsKernel(uint32_t* state, int* position,
                                        const int* offsets, int blocks,
                                        uint32_t* checkpoints, int* checkpointPosition) {
    __shared__ uint32_t stateA[624], stateB[624];
    for (int k = threadIdx.x; k < 624; k += blockDim.x) stateA[k] = state[k];
    __syncthreads();
    uint32_t* current = stateA;
    uint32_t* next = stateB;
    int cursor = *position;
    if (threadIdx.x == 0) *checkpointPosition = cursor;
    size_t count = static_cast<size_t>(offsets[blocks]) * (4 * NUM_RAYS);
    size_t generated = 0;
    size_t chunk = 0;
    while (generated < count) {
        for (int k = threadIdx.x; k < 624; k += blockDim.x) checkpoints[chunk * 624 + k] = current[k];
        __syncthreads();
        size_t end = generated + RANDOM_CHUNK_SIZE;
        if (end > count) end = count;
        while (generated < end) {
            if (cursor == 624) { twistRandomState(current, next); cursor = 0; }
            int available = static_cast<int>(end - generated < static_cast<size_t>(624 - cursor)
                ? end - generated : 624 - cursor);
            generated += available;
            cursor += available;
        }
        ++chunk;
    }
    for (int k = threadIdx.x; k < 624; k += blockDim.x) state[k] = current[k];
    if (threadIdx.x == 0) *position = cursor;
}

// Exactly reproduce uniform_real_distribution<float>'s conversion, including
// clamping the uint32 values whose rounded float representation would yield 1.
__global__ void randomSamplesKernel(const uint32_t* checkpoints, const int* checkpointPosition,
                                    const int* offsets, int blocks, val_t* samples) {
    size_t count = static_cast<size_t>(offsets[blocks]) * (4 * NUM_RAYS);
    size_t generated = static_cast<size_t>(blockIdx.x) * RANDOM_CHUNK_SIZE;
    if (generated >= count) return;
    size_t end = generated + RANDOM_CHUNK_SIZE;
    if (end > count) end = count;
    __shared__ uint32_t stateA[624], stateB[624];
    for (int k = threadIdx.x; k < 624; k += blockDim.x) stateA[k] = checkpoints[static_cast<size_t>(blockIdx.x) * 624 + k];
    __syncthreads();
    uint32_t* current = stateA;
    uint32_t* next = stateB;
    int cursor = *checkpointPosition;
    while (generated < end) {
        if (cursor == 624) { twistRandomState(current, next); cursor = 0; }
        int available = static_cast<int>(end - generated < static_cast<size_t>(624 - cursor)
            ? end - generated : 624 - cursor);
        for (int k = threadIdx.x; k < available; k += blockDim.x) {
            uint32_t y = current[cursor + k];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            val_t sample = static_cast<val_t>(y) * 0x1p-32f;
            samples[generated + k] = sample < ONE ? sample : 0x1.fffffep-1f;
        }
        generated += available;
        cursor += available;
        __syncthreads();
    }
}

__device__ val_t formFactorRay(const Triangle& triI, const Triangle& triJ,
                                const DeviceOctreeNode* nodes, const idx_t* indices,
                                const Triangle* triangles, size_t i, size_t j,
                                const val_t* sample) {
    Vec3 pI = randomPointInTriangle(triI, sample[0], sample[1]);
    Vec3 pJ = randomPointInTriangle(triJ, sample[2], sample[3]);
    if (isRayBlocked(pI, pJ, nodes, indices, triangles, i, j)) return ZERO;
    Vec3 v = pJ - pI;
    val_t distanceSquared = v.squaredNorm();
    if (distanceSquared < EPSILON) return ZERO;
    val_t cosineI = cosPhi(v, triI.normal()), cosineJ = cosPhi(-v, triJ.normal());
    if (cosineI <= ZERO || cosineJ <= ZERO) return ZERO;
    return (cosineI * cosineJ) / (PI * distanceSquared);
}

// Sixteen neighboring lanes trace the rays for one pair. Rays for the same
// geometry share cache lines, and random samples are read in contiguous groups.
// The first lane accumulates ray contributions in the original sampling order.
__global__ void formFactorsKernel(const Triangle* triangles,
                                  const DeviceOctreeNode* nodes, const idx_t* indices,
                                  size_t n, size_t first, size_t count,
                                  const int* ranks, const int* offsets,
                                  const val_t* samples, val_t* kij) {
    __shared__ val_t contributions[GPU_THREADS];
    int ray = threadIdx.x % NUM_RAYS;
    int groupStart = threadIdx.x - ray;
    size_t local = static_cast<size_t>(blockIdx.x) * (blockDim.x / NUM_RAYS) + threadIdx.x / NUM_RAYS;
    if (local >= count) return;
    size_t pair = first + local, i = pair / n, j = pair % n;
    int rank = ranks[local];
    if (rank < 0) {
        if (ray == 0) kij[pair] = ZERO;
        return;
    }
    size_t randomBase = static_cast<size_t>(rank + offsets[local / GPU_THREADS]) * (4 * NUM_RAYS);
    contributions[threadIdx.x] = formFactorRay(triangles[i], triangles[j], nodes, indices, triangles,
                                               i, j, samples + randomBase + 4 * ray);
    unsigned int mask = 0xffffu << (groupStart % 32);
    __syncwarp(mask);
    if (ray == 0) {
        val_t factor = ZERO;
        for (int r = 0; r < NUM_RAYS; ++r) factor += contributions[groupStart + r];
        kij[pair] = factor * INV_NUM_RAYS;
    }
}

// A warp cooperates on each receiver, loading 32 emitter contributions at a
// time. Its first lane adds them in original index order instead of performing
// a tree reduction. This fills the GPU even for small meshes and keeps all
// matrix reads coalesced without changing floating-point summation semantics.
constexpr int PROPAGATION_THREADS = 128;
__global__ void propagationKernel(const val_t* kij, const int* tau,
                                  const val_t* areas, const val_t* rho,
                                  const val_t* radE, val_t* radB, size_t n, size_t t) {
    __shared__ val_t contributions[PROPAGATION_THREADS];
    int lane = threadIdx.x % 32;
    int warpStart = threadIdx.x - lane;
    size_t i = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (i >= n) return;
    val_t sum = ZERO;
    for (size_t first = 0; first < n; first += 32) {
        size_t j = first + lane;
        val_t contribution = ZERO;
        if (j < n && i != j) {
            int delay = tau[i * n + j];
            if (static_cast<int>(t) >= delay) {
                val_t factor = kij[i * n + j];
                if (!(factor <= ZERO)) {
                    val_t radiosity = radB[(t - static_cast<size_t>(delay)) * n + j];
                    if (!(radiosity <= ZERO)) {
                        val_t weight = factor * areas[j];
                        contribution = (weight > ONE ? ONE : weight) * radiosity;
                    }
                }
            }
        }
        contributions[threadIdx.x] = contribution;
        __syncwarp();
        if (lane == 0) {
            for (int k = 0; k < 32; ++k) sum += contributions[warpStart + k];
        }
        __syncwarp();
    }
    if (lane == 0) radB[t * n + i] = rho[i] * sum + radE[t * n + i];
}

// Lags and receivers are independent. Each thread retains the sequential
// correlation sum; a second kernel selects the earliest lag on ties.
__global__ void correlationsKernel(const val_t* radB, val_t* correlations,
                                   size_t n, size_t timesteps, size_t source) {
    size_t count = n * timesteps;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        size_t i = index % n, lag = index / n;
        val_t sum = ZERO;
        for (size_t tt = lag; tt < timesteps; ++tt) {
            sum += radB[(tt - lag) * n + source] * radB[tt * n + i];
        }
        correlations[index] = sum;
    }
}

__global__ void distancesKernel(const val_t* correlations, val_t* distances,
                                size_t n, size_t timesteps) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maximum = ZERO;
    size_t best = 0;
    for (size_t lag = 0; lag < timesteps; ++lag) {
        val_t value = correlations[lag * n + i];
        if (value > maximum) { maximum = value; best = lag; }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(best);
}

__global__ void countFormFactorsKernel(const val_t* kij, size_t count,
                                       unsigned long long* nonzero) {
    __shared__ unsigned int counts[GPU_THREADS];
    unsigned int local = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (kij[i] > EPSILON) ++local;
    }
    counts[threadIdx.x] = local;
    __syncthreads();
    for (int stride = GPU_THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(nonzero, static_cast<unsigned long long>(counts[0]));
}

unsigned int gridSize(size_t count) {
    return static_cast<unsigned int>(std::min<size_t>((count + GPU_THREADS - 1) / GPU_THREADS, 65535));
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
    std::unique_ptr<DeviceSimulation> device; // Persistent GPU simulation data
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity, downloaded for validation
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

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
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    std::vector<DeviceOctreeNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(state.octree, nodes, indices);
    state.device = std::make_unique<DeviceSimulation>(state.numTriangles, timesteps,
                                                      nodes.size(), indices.size());
    auto& gpu = *state.device;
    gpu.triangles.upload(state.triangles.data(), state.numTriangles);
    gpu.nodes.upload(nodes.data(), nodes.size());
    gpu.indices.upload(indices.data(), indices.size());
    gpu.areas.upload(state.areas.data(), state.numTriangles);
    gpu.rho.upload(state.rho.data(), state.numTriangles);
    gpu.radE.upload(state.radE.data(), timesteps * state.numTriangles);
    if (timesteps) checkCuda(cudaMemset(gpu.radB.get(), 0, timesteps * state.numTriangles * sizeof(val_t)), "Initialize radiosity");
}

// ============================================================================
// Precomputation Phase
// ============================================================================

struct PairBatch {
    DeviceBuffer<int> ranks, counts, offsets;
    DeviceBuffer<val_t> samples;
    CudaEvent ready, consumed;
    PairBatch(size_t batchSize, int blocks)
        : ranks(batchSize), counts(blocks + 1), offsets(blocks + 1), samples(batchSize * (4 * NUM_RAYS)) {}
};

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    auto& gpu = *state.device;
    size_t pairs = state.numTriangles * state.numTriangles;
    size_t batchSize = std::min(pairs, PAIR_BATCH_SIZE);
    int maxBlocks = static_cast<int>((batchSize + GPU_THREADS - 1) / GPU_THREADS);
    int bufferCount = pairs > batchSize ? 2 : 1;
    std::array<std::unique_ptr<PairBatch>, 2> batches;
    for (int k = 0; k < bufferCount; ++k) batches[k] = std::make_unique<PairBatch>(batchSize, maxBlocks);
    DeviceBuffer<int> position(1), checkpointPosition(1);
    size_t randomChunks = (batchSize * (4 * NUM_RAYS) + RANDOM_CHUNK_SIZE - 1) / RANDOM_CHUNK_SIZE;
    DeviceBuffer<uint32_t> randomState(624), checkpoints(randomChunks * 624);
    std::array<uint32_t, 624> seed;
    seed[0] = 42;
    for (uint32_t k = 1; k < seed.size(); ++k) {
        seed[k] = 1812433253u * (seed[k - 1] ^ (seed[k - 1] >> 30)) + k;
    }
    randomState.upload(seed.data(), seed.size());
    int cursor = 624;
    position.upload(&cursor, 1);
    size_t scanBytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, batches[0]->counts.get(),
                                          batches[0]->offsets.get(), maxBlocks + 1), "Size pair scan");
    DeviceBuffer<unsigned char> scanStorage(scanBytes);
    CudaStream preparation, geometry;
    CudaEvent initialized;
    initialized.record(nullptr);
    initialized.wait(preparation.get());
    size_t batchIndex = 0;
    for (size_t first = 0; first < pairs; first += batchSize, ++batchIndex) {
        auto& batch = *batches[batchIndex % bufferCount];
        // Sampling the next batch overlaps visibility traversal of the current
        // batch. Events protect scratch reuse and preserve the single MT stream.
        if (batchIndex >= static_cast<size_t>(bufferCount)) batch.consumed.wait(preparation.get());
        size_t count = std::min(batchSize, pairs - first);
        int blocks = static_cast<int>((count + GPU_THREADS - 1) / GPU_THREADS);
        rankPairsKernel<<<blocks, GPU_THREADS, 0, preparation.get()>>>(
            gpu.triangles.get(), state.numTriangles, first, count, batch.ranks.get(), batch.counts.get());
        checkCuda(cudaGetLastError(), "Rank sampling pairs");
        checkCuda(cub::DeviceScan::ExclusiveSum(scanStorage.get(), scanBytes,
                                              batch.counts.get(), batch.offsets.get(), blocks + 1,
                                              preparation.get()), "Scan sampling pairs");
        randomCheckpointsKernel<<<1, GPU_THREADS, 0, preparation.get()>>>(
            randomState.get(), position.get(), batch.offsets.get(), blocks, checkpoints.get(), checkpointPosition.get());
        checkCuda(cudaGetLastError(), "Checkpoint random stream");
        randomSamplesKernel<<<static_cast<unsigned int>(randomChunks), GPU_THREADS, 0, preparation.get()>>>(
            checkpoints.get(), checkpointPosition.get(), batch.offsets.get(), blocks, batch.samples.get());
        checkCuda(cudaGetLastError(), "Generate random samples");
        batch.ready.record(preparation.get());
        batch.ready.wait(geometry.get());
        formFactorsKernel<<<static_cast<unsigned int>((count + GPU_THREADS / NUM_RAYS - 1) /
                                                        (GPU_THREADS / NUM_RAYS)), GPU_THREADS, 0, geometry.get()>>>(
            gpu.triangles.get(), gpu.nodes.get(), gpu.indices.get(), state.numTriangles, first, count,
            batch.ranks.get(), batch.offsets.get(), batch.samples.get(), gpu.kij.get());
        checkCuda(cudaGetLastError(), "Compute form factors");
        batch.consumed.record(geometry.get());
    }
    checkCuda(cudaDeviceSynchronize(), "Complete form factors");
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    timeDelaysKernel<<<gridSize(state.numTriangles * state.numTriangles), GPU_THREADS>>>(
        state.device->triangles.get(), state.device->tau.get(), state.numTriangles);
    checkCuda(cudaGetLastError(), "Compute time delays");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    auto& gpu = *state.device;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<static_cast<unsigned int>((state.numTriangles + PROPAGATION_THREADS / 32 - 1) /
                                                      (PROPAGATION_THREADS / 32)), PROPAGATION_THREADS>>>(
            gpu.kij.get(), gpu.tau.get(), gpu.areas.get(), gpu.rho.get(),
            gpu.radE.get(), gpu.radB.get(), state.numTriangles, t);
        checkCuda(cudaGetLastError(), "Propagate radiosity");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "Complete wave propagation");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = state.numTriangles, t = state.numTimesteps;
    auto& gpu = *state.device;
    DeviceBuffer<val_t> correlations(n * t);
    if (t) {
        correlationsKernel<<<gridSize(n * t), GPU_THREADS>>>(
            gpu.radB.get(), correlations.get(), n, t, state.sourceIndex);
        checkCuda(cudaGetLastError(), "Compute correlations");
    }
    distancesKernel<<<gridSize(n), GPU_THREADS>>>(correlations.get(), gpu.distances.get(), n, t);
    checkCuda(cudaGetLastError(), "Select distance lags");
    gpu.distances.download(state.distances.data(), n);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    state.radB.resize(state.numTimesteps * state.numTriangles);
    state.device->radB.download(state.radB.data(), state.radB.size());
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
    DeviceBuffer<unsigned long long> nonzero(1);
    checkCuda(cudaMemset(nonzero.get(), 0, sizeof(unsigned long long)), "Initialize validation count");
    size_t pairCount = state.numTriangles * state.numTriangles;
    countFormFactorsKernel<<<gridSize(pairCount), GPU_THREADS>>>(state.device->kij.get(), pairCount, nonzero.get());
    checkCuda(cudaGetLastError(), "Count form factors");
    unsigned long long nonZeroKij = 0;
    nonzero.download(&nonZeroKij, 1);
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, pairCount, 100.0 * nonZeroKij / static_cast<double>(pairCount));

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

int main(int argc, char** argv) try {
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
} catch (const std::exception& error) {
    fprintf(stderr, "roomsim: %s\n", error.what());
    return 1;
}

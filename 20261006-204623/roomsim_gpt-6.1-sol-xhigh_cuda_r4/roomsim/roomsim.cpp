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
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

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

    // Preserve the reference contraction order in geometric predicates.
    __host__ __device__ val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return __fmaf_rn(z, o.z, __fmaf_rn(x, o.x, __fmul_rn(y, o.y)));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {__fmaf_rn(-z, o.y, __fmul_rn(y, o.z)),
                __fmaf_rn(-x, o.z, __fmul_rn(z, o.x)),
                __fmaf_rn(x, o.y, -__fmul_rn(y, o.x))};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    __host__ __device__ val_t squaredNorm() const { return dot(*this); }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
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
// CUDA Storage and Geometry
// ============================================================================

void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class T>
class DeviceBuffer {
    T* ptr = nullptr;
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("CUDA allocation size overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    T* data() const { return ptr; }
};

// Pinned staging memory and events allow checkpoint construction to overlap
// the previous batch of visibility rays without copying or reusing live data.
template<class T>
class PinnedBuffer {
    T* ptr = nullptr;
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    void allocate(size_t count) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    T* data() const { return ptr; }
};

class CudaEvent {
    cudaEvent_t event = nullptr;
public:
    CudaEvent() { cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { if (event) cudaEventDestroy(event); }
    void record() { cudaCheck(cudaEventRecord(event)); }
    void wait() { cudaCheck(cudaEventSynchronize(event)); }
};

struct DeviceTriangle {
    Vec3 a, ab, ac, normal, center;
    val_t area;
};

// Preorder nodes use an escape index, so rays need neither recursion nor a stack.
struct DeviceNode {
    Vec3 center, halfExtent;
    uint32_t escape, firstTriangle, numTriangles;
};

void flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes,
                   std::vector<idx_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size())});
    for (size_t triangle : tree.triangleIndices) indices.push_back(static_cast<idx_t>(triangle));
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[index].escape = static_cast<uint32_t>(nodes.size());
}

struct CudaState {
    DeviceBuffer<DeviceTriangle> triangles;
    DeviceBuffer<DeviceNode> nodes;
    DeviceBuffer<idx_t> nodeTriangles;
    DeviceBuffer<val_t> weights, radB, correlations, distances;
    DeviceBuffer<int> tau, counters;
    size_t numNodes = 0;
};

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    val_t reflectivity = 0;
    int minDelay = 1, multiprocessors = 1;
    std::vector<Triangle> triangles;
    std::vector<val_t> distances;
    Octree octree;
    CudaState gpu;
    int nonZeroFormFactors = 0, receivedEnergy = 0;
};

// ============================================================================
// Exact MT19937 Stream Partitioning
// ============================================================================

constexpr int MT_WORDS = 624;
constexpr int MT_BLOCKS_PER_CHUNK = 128;
constexpr int RANDOM_CHUNK_WORDS = MT_WORDS * MT_BLOCKS_PER_CHUNK;
constexpr int RANDOM_WORDS_PER_PAIR = NUM_RAYS * 4;
constexpr int PAIR_BATCH_SIZE = 262144;

// Only sparse stream checkpoints are made on the host. Independent CUDA blocks
// expand them and temper the words. This preserves std::mt19937(42) exactly,
// including the fact that culled pairs consume no random numbers.
class RandomStream {
    std::array<uint32_t, MT_WORDS> words;
    int cursor = MT_WORDS;

    void twist() {
        for (int k = 0; k < 227; ++k) {
            uint32_t y = (words[k] & 0x80000000u) | (words[k + 1] & 0x7fffffffu);
            words[k] = words[k + 397] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
        }
        for (int k = 227; k < 623; ++k) {
            uint32_t y = (words[k] & 0x80000000u) | (words[k + 1] & 0x7fffffffu);
            words[k] = words[k - 227] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
        }
        uint32_t y = (words[623] & 0x80000000u) | (words[0] & 0x7fffffffu);
        words[623] = words[396] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
    }
public:
    RandomStream() {
        words[0] = 42;
        for (uint32_t k = 1; k < MT_WORDS; ++k)
            words[k] = 1812433253u * (words[k - 1] ^ (words[k - 1] >> 30)) + k;
    }

    int checkpoints(size_t count, std::vector<uint32_t>& seeds) {
        if (cursor == MT_WORDS) { twist(); cursor = 0; }
        int firstWord = cursor;
        size_t blocks = (count + cursor + MT_WORDS - 1) / MT_WORDS;
        seeds.resize(((blocks + MT_BLOCKS_PER_CHUNK - 1) / MT_BLOCKS_PER_CHUNK) * MT_WORDS);
        for (size_t block = 0; block < blocks; ++block) {
            if (block % MT_BLOCKS_PER_CHUNK == 0)
                std::copy(words.begin(), words.end(), seeds.begin() +
                          (block / MT_BLOCKS_PER_CHUNK) * MT_WORDS);
            if (block + 1 < blocks) twist();
        }
        cursor = static_cast<int>((count + firstWord) % MT_WORDS);
        if (cursor == 0) cursor = MT_WORDS;
        return firstWord;
    }
};

__global__ void generateRandomKernel(const uint32_t* __restrict__ seeds,
                                     val_t* __restrict__ random, int firstWord, int count) {
    __shared__ uint32_t words[MT_WORDS];
    for (int k = threadIdx.x; k < MT_WORDS; k += blockDim.x)
        words[k] = seeds[static_cast<size_t>(blockIdx.x) * MT_WORDS + k];
    __syncthreads();

    for (int block = 0; block < MT_BLOCKS_PER_CHUNK; ++block) {
        int base = static_cast<int>(blockIdx.x) * RANDOM_CHUNK_WORDS + block * MT_WORDS - firstWord;
        for (int k = threadIdx.x; k < MT_WORDS; k += blockDim.x) {
            int output = base + k;
            if (output >= 0 && output < count) {
                uint32_t y = words[k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                // libstdc++ uniform_real_distribution<float> rounds the integer
                // to float before scaling, and clamps a rounded value of 1.
                val_t value = __uint2float_rn(y) * 0x1p-32f;
                random[output] = value < ONE ? value : 0x1.fffffep-1f;
            }
        }
        if (block + 1 == MT_BLOCKS_PER_CHUNK || base + MT_WORDS >= count) break;
        __syncthreads();

        // In-place MT recurrence has three dependency waves. Separate reads
        // from writes so neighboring words cannot race within a wave.
        for (int wave = 0; wave < 3; ++wave) {
            int k = static_cast<int>(threadIdx.x) + wave * 227;
            uint32_t next = 0;
            if (threadIdx.x < 227 && k < 623) {
                uint32_t y = (words[k] & 0x80000000u) | (words[k + 1] & 0x7fffffffu);
                next = words[k < 227 ? k + 397 : k - 227] ^
                       (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
            }
            __syncthreads();
            if (threadIdx.x < 227 && k < 623) words[k] = next;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            uint32_t y = (words[623] & 0x80000000u) | (words[0] & 0x7fffffffu);
            words[623] = words[396] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
        }
        __syncthreads();
    }
}

// ============================================================================
// GPU Visibility and Form Factors
// ============================================================================

__device__ bool rayIntersectsBox(const Vec3& from, const Vec3& halfDirection,
                                 const Vec3& absHalfDirection, const DeviceNode& node) {
    Vec3 c = from + halfDirection - node.center;
    if (fabsf(c.x) > node.halfExtent.x + absHalfDirection.x ||
        fabsf(c.y) > node.halfExtent.y + absHalfDirection.y ||
        fabsf(c.z) > node.halfExtent.z + absHalfDirection.z) return false;
    if (fabsf(halfDirection.y * c.z - halfDirection.z * c.y) >
        node.halfExtent.y * absHalfDirection.z + node.halfExtent.z * absHalfDirection.y + EPSILON) return false;
    if (fabsf(halfDirection.z * c.x - halfDirection.x * c.z) >
        node.halfExtent.z * absHalfDirection.x + node.halfExtent.x * absHalfDirection.z + EPSILON) return false;
    if (fabsf(halfDirection.x * c.y - halfDirection.y * c.x) >
        node.halfExtent.x * absHalfDirection.y + node.halfExtent.y * absHalfDirection.x + EPSILON) return false;
    return true;
}

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                      const DeviceTriangle& tri) {
    Vec3 pvec = dir.cross(tri.ac);
    val_t det = tri.ab.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    val_t invDet = ONE / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;
    Vec3 qvec = tvec.cross(tri.ab);
    val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return tri.ac.dot(qvec) * invDet;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, idx_t src, idx_t dst,
                             const DeviceTriangle* triangles, const DeviceNode* nodes,
                             const idx_t* indices, uint32_t numNodes) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;
    Vec3 halfDirection = dir * 0.5f;
    Vec3 absHalfDirection(fabsf(halfDirection.x), fabsf(halfDirection.y), fabsf(halfDirection.z));
    uint32_t index = 0;
    while (index < numNodes) {
        const DeviceNode& node = nodes[index];
        if (index != 0 && !rayIntersectsBox(from, halfDirection, absHalfDirection, node)) {
            index = node.escape;
            continue;
        }
        for (uint32_t t = 0; t < node.numTriangles; ++t) {
            idx_t triangle = indices[node.firstTriangle + t];
            if (triangle == src || triangle == dst) continue;
            val_t distance = rayTriangleIntersect(from, dirNorm, triangles[triangle]);
            if (distance > EPSILON && distance < rayLen - EPSILON) return true;
        }
        ++index;
    }
    return false;
}

__device__ Vec3 randomPointInTriangle(const DeviceTriangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + tri.ab * u + tri.ac * v;
}

__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    return vNorm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(normal) / vNorm);
}

__global__ void classifyPairsKernel(const DeviceTriangle* triangles, size_t n,
                                    size_t begin, int count, int* active) {
    int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p < count) {
        size_t pair = begin + p;
        size_t i = pair / n, j = pair % n;
        active[p] = i != j && triangles[i].normal.dot(triangles[j].normal) <= 0.99f;
    }
    if (p == count) active[p] = 0;
}

__global__ void formFactorsKernel(const DeviceTriangle* __restrict__ triangles,
                                  const DeviceNode* __restrict__ nodes,
                                  const idx_t* __restrict__ indices, uint32_t numNodes,
                                  const val_t* __restrict__ random,
                                  const int* __restrict__ offsets, size_t n,
                                  size_t begin, int count, val_t* __restrict__ kij) {
    // Each half warp traces the 16 rays of one pair independently.
    int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int p = thread / NUM_RAYS;
    int ray = thread % NUM_RAYS;
    if (p >= count) return;
    size_t pair = begin + p;
    idx_t i = static_cast<idx_t>(pair / n), j = static_cast<idx_t>(pair % n);
    val_t contribution = ZERO;
    int rank = offsets[p];
    if (offsets[p + 1] != rank) {
        const DeviceTriangle& triI = triangles[i];
        const DeviceTriangle& triJ = triangles[j];
        const val_t* values = random + static_cast<size_t>(rank) * RANDOM_WORDS_PER_PAIR + ray * 4;
        Vec3 pI = randomPointInTriangle(triI, values[0], values[1]);
        Vec3 pJ = randomPointInTriangle(triJ, values[2], values[3]);
        if (!isRayBlocked(pI, pJ, i, j, triangles, nodes, indices, numNodes)) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosI = cosPhi(v, triI.normal);
                val_t cosJ = cosPhi(-v, triJ.normal);
                if (cosI > ZERO && cosJ > ZERO)
                    contribution = (cosI * cosJ) / (PI * distSqr);
            }
        }
    }
    // Preserve ascending ray order rather than changing the float reduction.
    unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
    val_t sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t value = __shfl_sync(mask, contribution, r, NUM_RAYS);
        if (ray == 0) sum += value;
    }
    if (ray == 0) kij[static_cast<size_t>(j) * n + i] = sum * INV_NUM_RAYS;
}

__device__ int warpSum(int value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void timeDelaysKernel(const DeviceTriangle* triangles, size_t n,
                                 int* tau) {
    for (size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < n * n; pair += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t i = pair % n, j = pair / n;
        int delay = i == j ? 0 : static_cast<int>(ceilf((triangles[i].center - triangles[j].center).norm() * INV_WAVE_SPEED));
        tau[pair] = delay;
    }
}

__global__ void prepareWeightsKernel(const DeviceTriangle* triangles, size_t n,
                                     val_t* weights, const int* tau, int* counters) {
    int count = 0, minimum = INT_MAX;
    for (size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < n * n; pair += static_cast<size_t>(blockDim.x) * gridDim.x) {
        val_t kij = weights[pair];
        count += kij > EPSILON;
        if (kij > ZERO) minimum = min(minimum, tau[pair]);
        // An inactive interaction stays inactive even for unusual CLI inputs.
        weights[pair] = kij > ZERO ? fminf(kij * triangles[pair / n].area, ONE) : ZERO;
    }
    count = warpSum(count);
    for (int offset = 16; offset; offset >>= 1)
        minimum = min(minimum, __shfl_down_sync(0xffffffffu, minimum, offset));
    if ((threadIdx.x & 31) == 0) {
        if (count) atomicAdd(counters + 1, count);
        atomicMin(counters, minimum);
    }
}

// ============================================================================
// Initialization and Precomputation
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // CUDA is mandatory: initialization fails explicitly if no device is usable.
    cudaCheck(cudaFree(nullptr));
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device));
    state.multiprocessors = properties.multiProcessorCount;
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    std::vector<DeviceTriangle> triangles;
    triangles.reserve(state.numTriangles);
    for (const Triangle& tri : state.triangles)
        triangles.push_back({tri.a, tri.b - tri.a, tri.c - tri.a,
                             tri.normal(), tri.center(), tri.area()});
    std::vector<DeviceNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(state.octree, nodes, indices);
    CudaState& gpu = state.gpu;
    gpu.numNodes = nodes.size();
    gpu.triangles.allocate(triangles.size());
    gpu.nodes.allocate(nodes.size());
    gpu.nodeTriangles.allocate(indices.size());
    cudaCheck(cudaMemcpy(gpu.triangles.data(), triangles.data(), triangles.size() * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.nodes.data(), nodes.data(), nodes.size() * sizeof(DeviceNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.nodeTriangles.data(), indices.data(), indices.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    size_t pairs = state.numTriangles * state.numTriangles;
    size_t samples = timesteps * state.numTriangles;
    gpu.weights.allocate(pairs);
    gpu.tau.allocate(pairs);
    gpu.radB.allocate(samples);
    gpu.correlations.allocate(samples);
    gpu.distances.allocate(state.numTriangles);
    gpu.counters.allocate(3);
    int counters[3] = {INT_MAX, 0, 0};
    cudaCheck(cudaMemcpy(gpu.counters.data(), counters, sizeof(counters), cudaMemcpyHostToDevice));
    if (samples) cudaCheck(cudaMemset(gpu.radB.data(), 0, samples * sizeof(val_t)));
    state.distances.resize(state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    size_t pairs = state.numTriangles * state.numTriangles;
    unsigned blocks = static_cast<unsigned>(std::min<size_t>((pairs + 255) / 256, 4096));
    timeDelaysKernel<<<blocks, 256>>>(state.gpu.triangles.data(), state.numTriangles,
                                      state.gpu.tau.data());
    cudaCheck(cudaGetLastError());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    size_t n = state.numTriangles, pairs = n * n;
    int capacity = static_cast<int>(std::min<size_t>(pairs, PAIR_BATCH_SIZE));
    DeviceBuffer<int> active, offsets[2];
    PinnedBuffer<int> activeCounts;
    PinnedBuffer<uint32_t> pinnedSeeds;
    CudaEvent ready[2];
    DeviceBuffer<uint32_t> seeds;
    DeviceBuffer<val_t> random;
    DeviceBuffer<unsigned char> scanStorage;
    active.allocate(capacity + 1);
    for (auto& buffer : offsets) buffer.allocate(capacity + 1);
    activeCounts.allocate(2);
    size_t maxWords = static_cast<size_t>(capacity) * RANDOM_WORDS_PER_PAIR;
    size_t maxChunks = (maxWords + MT_WORDS - 1 + RANDOM_CHUNK_WORDS - 1) / RANDOM_CHUNK_WORDS;
    size_t seedCapacity = maxChunks * MT_WORDS;
    seeds.allocate(seedCapacity);
    pinnedSeeds.allocate(2 * seedCapacity);
    random.allocate(maxWords);
    size_t scanBytes = 0;
    cudaCheck(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, active.data(), offsets[0].data(), capacity + 1));
    scanStorage.allocate(scanBytes);
    RandomStream rng;
    std::vector<uint32_t> checkpoints;
    auto classify = [&](size_t begin, int slot) {
        int count = static_cast<int>(std::min<size_t>(capacity, pairs - begin));
        classifyPairsKernel<<<(count + 1 + 255) / 256, 256>>>(state.gpu.triangles.data(), n, begin, count, active.data());
        cudaCheck(cudaGetLastError());
        cudaCheck(cub::DeviceScan::ExclusiveSum(scanStorage.data(), scanBytes, active.data(), offsets[slot].data(), count + 1));
        cudaCheck(cudaMemcpyAsync(activeCounts.data() + slot, offsets[slot].data() + count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
        ready[slot].record();
    };
    classify(0, 0);
    int slot = 0;
    for (size_t begin = 0; begin < pairs; begin += capacity, slot ^= 1) {
        int count = static_cast<int>(std::min<size_t>(capacity, pairs - begin));
        ready[slot].wait();
        int activePairs = activeCounts.data()[slot];
        // Classify the next batch before this batch's rays are queued. Its
        // readiness event then lets the host advance the RNG during ray tracing.
        if (begin + capacity < pairs) classify(begin + capacity, slot ^ 1);
        if (activePairs) {
            int words = activePairs * RANDOM_WORDS_PER_PAIR;
            int firstWord = rng.checkpoints(words, checkpoints);
            uint32_t* staging = pinnedSeeds.data() + slot * seedCapacity;
            std::copy(checkpoints.begin(), checkpoints.end(), staging);
            cudaCheck(cudaMemcpyAsync(seeds.data(), staging, checkpoints.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
            generateRandomKernel<<<checkpoints.size() / MT_WORDS, 256>>>(seeds.data(), random.data(), firstWord, words);
            cudaCheck(cudaGetLastError());
        }
        formFactorsKernel<<<(count * NUM_RAYS + 127) / 128, 128>>>(
            state.gpu.triangles.data(), state.gpu.nodes.data(), state.gpu.nodeTriangles.data(),
            static_cast<uint32_t>(state.gpu.numNodes), random.data(), offsets[slot].data(), n,
            begin, count, state.gpu.weights.data());
        cudaCheck(cudaGetLastError());
    }
    prepareWeightsKernel<<<static_cast<unsigned>(std::min<size_t>((pairs + 255) / 256, 4096)), 256>>>(
        state.gpu.triangles.data(), n, state.gpu.weights.data(), state.gpu.tau.data(), state.gpu.counters.data());
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&state.minDelay, state.gpu.counters.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (state.minDelay <= 0) throw std::runtime_error("Mesh has a nonpositive propagation delay");
    printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// GPU Wave Propagation and Cross-Correlation
// ============================================================================

__global__ void propagationKernel(size_t n, size_t timesteps, size_t source,
                                  val_t rho, size_t begin, size_t end,
                                  const val_t* __restrict__ weights,
                                  const int* __restrict__ tau, val_t* __restrict__ radB) {
    // A warp handles eight receivers and four emitters at a time. Receiver
    // loads stay coalesced, while independent emitters hide memory latency.
    unsigned lane = threadIdx.x & 31;
    unsigned receiver = lane & 7, emitter = lane >> 3;
    size_t warp = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    size_t index = warp * 8 + receiver;
    bool valid = index < (end - begin) * n;
    unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (!valid) return;
    size_t i = index % n, t = begin + index / n;
    val_t sumB = ZERO;
    for (size_t base = 0; base < n; base += 4) {
        size_t j = base + emitter;
        val_t weight = ZERO, radJ = ZERO;
        if (j < n && i != j) {
            size_t pair = j * n + i;
            int delay = tau[pair];
            if (t >= static_cast<size_t>(delay)) {
                val_t w = weights[pair];
                if (w > ZERO) {
                    val_t r = radB[(t - static_cast<size_t>(delay)) * n + j];
                    if (!(r <= ZERO)) { weight = w; radJ = r; }
                }
            }
        }
        // Feed the original ordered accumulation from the parallel loads,
        // rounding the product before each addition as in the reference.
        for (int e = 0; e < 4; ++e) {
            val_t w = __shfl_sync(mask, weight, e * 8 + receiver);
            val_t r = __shfl_sync(mask, radJ, e * 8 + receiver);
            if (emitter == 0) sumB = __fadd_rn(sumB, __fmul_rn(w, r));
        }
    }
    if (emitter == 0) {
        val_t emission = i == source && t < timesteps / 2 ? ONE : ZERO;
        radB[t * n + i] = rho * sumB + emission;
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    size_t n = state.numTriangles, timesteps = state.numTimesteps;
    // Timesteps separated by less than the smallest active travel time are independent.
    // A launch boundary synchronizes all earlier time windows on the device.
    for (size_t begin = 0; begin < timesteps; begin += state.minDelay) {
        size_t end = std::min(timesteps, begin + state.minDelay);
        size_t jobs = (end - begin) * n;
        unsigned blockSize = jobs >= static_cast<size_t>(state.multiprocessors) * 32 ? 128 : 32;
        size_t threads = ((jobs + 7) / 8) * 32;
        propagationKernel<<<static_cast<unsigned>((threads + blockSize - 1) / blockSize), blockSize>>>(
            n, timesteps, state.sourceIndex, state.reflectivity, begin, end,
            state.gpu.weights.data(), state.gpu.tau.data(), state.gpu.radB.data());
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    if (timesteps) printf("  Timestep %zu/%zu\n", timesteps, timesteps);
}

__global__ void correlationKernel(const val_t* __restrict__ radB, size_t n,
                                  size_t timesteps, size_t source, val_t* __restrict__ correlations) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * timesteps; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        size_t i = index % n, lag = index / n;
        val_t sum = ZERO;
        // The reference rounds each correlation product before adding it.
        // Prevent contraction from changing neighboring correlation peaks.
        for (size_t tt = lag; tt < timesteps; ++tt)
            sum = __fadd_rn(sum, __fmul_rn(radB[(tt - lag) * n + source], radB[tt * n + i]));
        correlations[index] = sum;
    }
}

__global__ void distancesKernel(const val_t* correlations, const val_t* radB,
                                size_t n, size_t timesteps, val_t* distances, int* receivedEnergy) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maxCorr = ZERO;
    size_t bestT = 0;
    bool received = false;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = correlations[t * n + i];
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
        received |= radB[t * n + i] > EPSILON;
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    if (received) atomicAdd(receivedEnergy, 1);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = state.numTriangles, samples = n * state.numTimesteps;
    if (samples) {
        correlationKernel<<<static_cast<unsigned>(std::min<size_t>((samples + 127) / 128, 65535)), 128>>>(
            state.gpu.radB.data(), n, state.numTimesteps, state.sourceIndex, state.gpu.correlations.data());
        cudaCheck(cudaGetLastError());
    }
    distancesKernel<<<static_cast<unsigned>((n + 127) / 128), 128>>>(
        state.gpu.correlations.data(), state.gpu.radB.data(), n, state.numTimesteps,
        state.gpu.distances.data(), state.gpu.counters.data() + 2);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(state.distances.data(), state.gpu.distances.data(), n * sizeof(val_t), cudaMemcpyDeviceToHost));
    int counters[3];
    cudaCheck(cudaMemcpy(counters, state.gpu.counters.data(), sizeof(counters), cudaMemcpyDeviceToHost));
    state.nonZeroFormFactors = counters[1];
    state.receivedEnergy = counters[2];
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
    int receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    int nonZeroKij = state.nonZeroFormFactors;
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
        uint32_t bits;
        memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

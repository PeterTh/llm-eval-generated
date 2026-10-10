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
#include <cstdint>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

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
// CUDA storage and spatial traversal
// ============================================================================

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template<class T>
class DeviceBuffer {
    T* data_ = nullptr;
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflow");
        }
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    T* data() const { return data_; }
    void upload(const T* source, size_t count) {
        if (count) CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* destination, size_t count) const {
        if (count) CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

struct CudaStream {
    cudaStream_t value = nullptr;
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(value); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

struct CudaEvent {
    cudaEvent_t value = nullptr;
    CudaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~CudaEvent() { cudaEventDestroy(value); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

struct FlatOctreeNode {
    Vec3 center, halfExtent;
    uint32_t firstTriangle, triangleCount, escape;
};

// Preorder traversal with an escape index avoids recursion and per-ray stacks.
void flattenOctree(const Octree& node, std::vector<FlatOctreeNode>& nodes,
                   std::vector<idx_t>& indices) {
    const size_t index = nodes.size();
    nodes.push_back({node.center, node.halfExtent, static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(node.triangleIndices.size()), 0});
    for (size_t triangle : node.triangleIndices) indices.push_back(static_cast<idx_t>(triangle));
    for (const auto& child : node.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[index].escape = static_cast<uint32_t>(nodes.size());
}

__device__ __forceinline__ bool intersectsBox(const Vec3& from, const Vec3& to,
                                             const FlatOctreeNode& node) {
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 c = from + d - node.center;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                     const Triangle& triangle) {
    const Vec3 e1 = triangle.b - triangle.a;
    const Vec3 e2 = triangle.c - triangle.a;
    const Vec3 pvec = dir.cross(e2);
    const val_t det = e1.dot(pvec);
    constexpr val_t noHit = FLT_MAX;
    if (fabsf(det) < EPSILON) return noHit;
    const val_t invDet = ONE / det;
    const Vec3 tvec = orig - triangle.a;
    const val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return noHit;
    const Vec3 qvec = tvec.cross(e1);
    const val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return noHit;
    return e2.dot(qvec) * invDet;
}

__device__ __forceinline__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                                            const Triangle* triangles,
                                            const FlatOctreeNode* nodes,
                                            const idx_t* indices, idx_t src, idx_t dst) {
    const Vec3 dir = to - from;
    const val_t length = dir.norm();
    if (length < EPSILON) return true;
    const Vec3 normalized = dir / length;
    uint32_t index = 0;
    while (index < nodes[0].escape) {
        const FlatOctreeNode node = nodes[index];
        // The original traversal enters the root without a bounding-box test.
        if (index != 0 && !intersectsBox(from, to, node)) {
            index = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.triangleCount; ++k) {
            const idx_t triangle = indices[node.firstTriangle + k];
            if (triangle == src || triangle == dst) continue;
            const val_t distance = rayTriangleIntersect(from, normalized, triangles[triangle]);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++index;
    }
    return false;
}

// ============================================================================
// Deterministic parallel sampling
// ============================================================================

constexpr unsigned PAIRS_PER_BATCH = 32768;
constexpr unsigned RANDOMS_PER_PAIR = NUM_RAYS * 4;

__global__ void classifyPairs(const Triangle* triangles, size_t n, size_t first,
                             unsigned count, uint32_t* flags) {
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local > count) return;
    if (local == count) {
        flags[local] = 0;
        return;
    }
    const size_t pair = first + local;
    const size_t i = pair / n, j = pair % n;
    flags[local] = i != j && !(triangles[i].normal().dot(triangles[j].normal()) > 0.99f);
}

__device__ __forceinline__ uint32_t mtMix(uint32_t a, uint32_t b) {
    const uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

// Match std::mt19937 and uniform_real_distribution<float>, including the
// sequential stream's conditional consumption. Algebraically expanding the
// in-place twist makes all 624 state updates independent within each round.
__global__ void generateSamples(uint32_t* state, const uint32_t* offsets,
                                unsigned count, val_t* samples) {
    __shared__ uint32_t words[2][624];
    for (unsigned i = threadIdx.x; i < 624; i += blockDim.x) words[0][i] = state[i];
    unsigned cursor = state[624];
    unsigned current = 0;
    const unsigned total = offsets[count] * RANDOMS_PER_PAIR;
    unsigned written = 0;
    __syncthreads();
    while (written < total) {
        if (cursor == 624) {
            const uint32_t* old = words[current];
            uint32_t* next = words[current ^ 1];
            for (unsigned i = threadIdx.x; i < 624; i += blockDim.x) {
                uint32_t value;
                if (i < 227) {
                    value = old[i + 397] ^ mtMix(old[i], old[i + 1]);
                } else if (i < 454) {
                    value = old[i + 170] ^ mtMix(old[i - 227], old[i - 226]) ^ mtMix(old[i], old[i + 1]);
                } else if (i < 623) {
                    value = old[i - 57] ^ mtMix(old[i - 454], old[i - 453]) ^
                            mtMix(old[i - 227], old[i - 226]) ^ mtMix(old[i], old[i + 1]);
                } else {
                    const uint32_t next0 = old[397] ^ mtMix(old[0], old[1]);
                    value = old[566] ^ mtMix(old[169], old[170]) ^
                            mtMix(old[396], old[397]) ^ mtMix(old[623], next0);
                }
                next[i] = value;
            }
            __syncthreads();
            current ^= 1;
            cursor = 0;
        }
        const unsigned available = min(624u - cursor, total - written);
        for (unsigned i = threadIdx.x; i < available; i += blockDim.x) {
            uint32_t y = words[current][cursor + i];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            const val_t sample = __uint2float_rn(y) * 0x1p-32f;
            samples[written + i] = sample < ONE ? sample : 0x1.fffffep-1f;
        }
        cursor += available;
        written += available;
        __syncthreads();
    }
    for (unsigned i = threadIdx.x; i < 624; i += blockDim.x) state[i] = words[current][i];
    if (threadIdx.x == 0) state[624] = cursor;
}

__device__ __forceinline__ Vec3 sampleTriangle(const Triangle& triangle, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v;
}

__global__ void formFactors(const Triangle* triangles, const FlatOctreeNode* nodes,
                            const idx_t* indices, size_t n, size_t first, unsigned count,
                            const uint32_t* flags, const uint32_t* offsets,
                            const val_t* samples, val_t* kij) {
    // Each half warp handles one pair, with one lane per visibility ray.
    const unsigned local = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_RAYS;
    const unsigned ray = threadIdx.x % NUM_RAYS;
    if (local >= count) return;
    const size_t pair = first + local;
    const idx_t i = static_cast<idx_t>(pair / n), j = static_cast<idx_t>(pair % n);
    if (!flags[local]) {
        if (ray == 0) kij[static_cast<size_t>(j) * n + i] = ZERO;
        return;
    }
    const Triangle triI = triangles[i], triJ = triangles[j];
    const val_t* random = samples + offsets[local] * RANDOMS_PER_PAIR + ray * 4;
    const Vec3 pI = sampleTriangle(triI, random[0], random[1]);
    const Vec3 pJ = sampleTriangle(triJ, random[2], random[3]);
    val_t contribution = ZERO;
    if (!isRayBlocked(pI, pJ, triangles, nodes, indices, i, j)) {
        const Vec3 v = pJ - pI;
        const val_t distanceSquared = v.squaredNorm();
        if (distanceSquared >= EPSILON) {
            const val_t norm = v.norm();
            const val_t cosI = norm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI.normal()) / norm);
            const val_t cosJ = norm <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(triJ.normal()) / norm);
            if (cosI > ZERO && cosJ > ZERO) contribution = (cosI * cosJ) / (PI * distanceSquared);
        }
    }
    // Preserve the original ray accumulation order, rather than a tree reduction.
    const unsigned mask = 0xffffu << ((threadIdx.x & 31u) >= 16u ? 16u : 0u);
    val_t sum = ZERO;
    #pragma unroll
    for (unsigned r = 0; r < NUM_RAYS; ++r) {
        const val_t value = __shfl_sync(mask, contribution, r, NUM_RAYS);
        if (ray == 0) sum += value;
    }
    // Transposed storage coalesces receiver-parallel propagation accesses.
    if (ray == 0) kij[static_cast<size_t>(j) * n + i] = sum * INV_NUM_RAYS;
}

__global__ void timeDelays(const Triangle* triangles, size_t n, int* tau, int* minimum) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int delay = INT_MAX;
    if (pair < n * n) {
        const size_t i = pair % n, j = pair / n;
        if (i != j) {
            const val_t distance = (triangles[i].center() - triangles[j].center()).norm();
            delay = static_cast<int>(ceilf(distance * INV_WAVE_SPEED));
            tau[pair] = delay;
        } else {
            tau[pair] = 0;
        }
    }
    for (int offset = 16; offset; offset /= 2) {
        delay = min(delay, __shfl_down_sync(0xffffffffu, delay, offset));
    }
    __shared__ int warpMinimum[8];
    if ((threadIdx.x & 31) == 0) warpMinimum[threadIdx.x / 32] = delay;
    __syncthreads();
    if (threadIdx.x == 0) {
        int result = warpMinimum[0];
        for (int warp = 1; warp < 8; ++warp) result = min(result, warpMinimum[warp]);
        atomicMin(minimum, result);
    }
}

__global__ void propagate(const val_t* kij, const int* tau, const val_t* areas,
                          const val_t* rho, val_t* radB, size_t n, size_t firstTime,
                          size_t count, size_t source, size_t emissionEnd) {
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= n * count) return;
    const size_t i = local % n, t = firstTime + local / n;
    val_t sum = ZERO;
    // Positive delays allow all receivers (and a window of timesteps) to run
    // independently. Keep emitter order to preserve float summation semantics.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        const int delay = tau[j * n + i];
        if (static_cast<int>(t) < delay) continue;
        const val_t factor = kij[j * n + i];
        if (factor <= ZERO) continue;
        const val_t radiosity = radB[(t - static_cast<size_t>(delay)) * n + j];
        if (radiosity <= ZERO) continue;
        sum += fminf(factor * areas[j], ONE) * radiosity;
    }
    const val_t emission = i == source && t < emissionEnd ? ONE : ZERO;
    radB[t * n + i] = rho[i] * sum + emission;
}

__global__ void correlate(const val_t* radB, val_t* correlations,
                          size_t n, size_t timesteps, size_t source) {
    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= n * timesteps) return;
    const size_t i = local % n, lag = local / n;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < timesteps; ++tt) {
        sum += radB[(tt - lag) * n + source] * radB[tt * n + i];
    }
    correlations[local] = sum;
}

__global__ void selectDistances(const val_t* correlations, val_t* distances,
                                size_t n, size_t timesteps) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maximum = ZERO;
    size_t best = 0;
    for (size_t lag = 0; lag < timesteps; ++lag) {
        const val_t value = correlations[lag * n + i];
        if (value > maximum) { maximum = value; best = lag; }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(best);
}

struct SamplingBatch {
    DeviceBuffer<uint32_t> flags{PAIRS_PER_BATCH + 1}, offsets{PAIRS_PER_BATCH + 1};
    DeviceBuffer<val_t> samples{PAIRS_PER_BATCH * RANDOMS_PER_PAIR};
    DeviceBuffer<unsigned char> scratch;
    size_t scratchBytes = 0;
    CudaEvent ready, consumed;
    bool used = false;
    SamplingBatch() {
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scratchBytes, flags.data(),
                                                offsets.data(), PAIRS_PER_BATCH + 1));
        scratch.allocate(scratchBytes);
    }
};

struct DeviceSimulation {
    CudaStream samplingStream, computeStream;
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<FlatOctreeNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<val_t> areas, rho, kij, radB, correlations, distances;
    DeviceBuffer<int> tau, minimumDelay{1};
    DeviceBuffer<uint32_t> randomState{625};
    SamplingBatch batches[2];
    int minDelay = 1;

    DeviceSimulation(const std::vector<Triangle>& mesh, const Octree& octree,
                     const std::vector<val_t>& triangleAreas,
                     const std::vector<val_t>& reflectivity, size_t timesteps) {
        const size_t n = mesh.size();
        triangles.allocate(n);
        triangles.upload(mesh.data(), n);
        std::vector<FlatOctreeNode> flatNodes;
        std::vector<idx_t> flatIndices;
        flattenOctree(octree, flatNodes, flatIndices);
        nodes.allocate(flatNodes.size());
        nodes.upload(flatNodes.data(), flatNodes.size());
        indices.allocate(flatIndices.size());
        indices.upload(flatIndices.data(), flatIndices.size());
        areas.allocate(n);
        areas.upload(triangleAreas.data(), n);
        rho.allocate(n);
        rho.upload(reflectivity.data(), n);
        kij.allocate(n * n);
        tau.allocate(n * n);
        radB.allocate(n * timesteps);
        correlations.allocate(n * timesteps);
        distances.allocate(n);
        uint32_t seed[625];
        seed[0] = 42;
        for (uint32_t i = 1; i < 624; ++i) {
            seed[i] = 1812433253u * (seed[i - 1] ^ (seed[i - 1] >> 30)) + i;
        }
        seed[624] = 624;
        randomState.upload(seed, 625);
        const int initialMinimum = std::numeric_limits<int>::max();
        minimumDelay.upload(&initialMinimum, 1);
        // The working streams are nonblocking; finish all initialization copies
        // before either stream starts reading their destinations.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Optional validation copy (transposed N x N)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    std::unique_ptr<DeviceSimulation> device;

    size_t sourceIndex;

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

    // Large working matrices stay on the GPU. Host copies are only needed for
    // optional validation; emission is represented by the source and time range.
    state.distances.resize(state.numTriangles, ZERO);
    state.device = std::make_unique<DeviceSimulation>(state.triangles, state.octree,
                                                       state.areas, state.rho, timesteps);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on CUDA...\n");
    DeviceSimulation& gpu = *state.device;
    const size_t pairs = state.numTriangles * state.numTriangles;
    size_t batchNumber = 0;
    for (size_t first = 0; first < pairs; first += PAIRS_PER_BATCH, ++batchNumber) {
        SamplingBatch& batch = gpu.batches[batchNumber % 2];
        const unsigned count = static_cast<unsigned>(std::min<size_t>(PAIRS_PER_BATCH, pairs - first));
        if (batch.used) CUDA_CHECK(cudaStreamWaitEvent(gpu.samplingStream.value, batch.consumed.value));
        classifyPairs<<<(count + 1 + 255) / 256, 256, 0, gpu.samplingStream.value>>>(
            gpu.triangles.data(), state.numTriangles, first, count, batch.flags.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(batch.scratch.data(), batch.scratchBytes,
                                                batch.flags.data(), batch.offsets.data(),
                                                count + 1, gpu.samplingStream.value));
        generateSamples<<<1, 256, 0, gpu.samplingStream.value>>>(
            gpu.randomState.data(), batch.offsets.data(), count, batch.samples.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(batch.ready.value, gpu.samplingStream.value));
        CUDA_CHECK(cudaStreamWaitEvent(gpu.computeStream.value, batch.ready.value));
        formFactors<<<(count + 7) / 8, 128, 0, gpu.computeStream.value>>>(
            gpu.triangles.data(), gpu.nodes.data(), gpu.indices.data(), state.numTriangles,
            first, count, batch.flags.data(), batch.offsets.data(), batch.samples.data(), gpu.kij.data());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(batch.consumed.value, gpu.computeStream.value));
        batch.used = true;
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.computeStream.value));
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on CUDA...\n");
    DeviceSimulation& gpu = *state.device;
    const size_t pairs = state.numTriangles * state.numTriangles;
    timeDelays<<<static_cast<unsigned>((pairs + 255) / 256), 256, 0, gpu.computeStream.value>>>(
        gpu.triangles.data(), state.numTriangles, gpu.tau.data(), gpu.minimumDelay.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(gpu.computeStream.value));
    gpu.minimumDelay.download(&gpu.minDelay, 1);
    if (gpu.minDelay < 1) throw std::runtime_error("Mesh contains a non-positive off-diagonal time delay");
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on CUDA...\n");
    DeviceSimulation& gpu = *state.device;
    const size_t window = static_cast<size_t>(gpu.minDelay);
    for (size_t t = 0; t < state.numTimesteps; t += window) {
        const size_t count = std::min(window, state.numTimesteps - t);
        propagate<<<static_cast<unsigned>((state.numTriangles * count + 255) / 256),
                    256, 0, gpu.computeStream.value>>>(
            gpu.kij.data(), gpu.tau.data(), gpu.areas.data(), gpu.rho.data(), gpu.radB.data(),
            state.numTriangles, t, count, state.sourceIndex, state.numTimesteps / 2);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.computeStream.value));
    printf("  Timestep %zu/%zu\n", state.numTimesteps, state.numTimesteps);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on CUDA...\n");
    DeviceSimulation& gpu = *state.device;
    const size_t entries = state.numTriangles * state.numTimesteps;
    if (entries) {
        correlate<<<static_cast<unsigned>((entries + 255) / 256), 256, 0, gpu.computeStream.value>>>(
            gpu.radB.data(), gpu.correlations.data(), state.numTriangles,
            state.numTimesteps, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
    }
    selectDistances<<<static_cast<unsigned>((state.numTriangles + 255) / 256), 256,
                      0, gpu.computeStream.value>>>(
        gpu.correlations.data(), gpu.distances.data(), state.numTriangles, state.numTimesteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(gpu.computeStream.value));
    gpu.distances.download(state.distances.data(), state.numTriangles);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    state.kij.resize(state.numTriangles * state.numTriangles);
    state.radB.resize(state.numTriangles * state.numTimesteps);
    state.device->kij.download(state.kij.data(), state.kij.size());
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

    if (timesteps < 0) throw std::runtime_error("Number of timesteps must be non-negative");

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
    fprintf(stderr, "Room simulation failed: %s\n", error.what());
    return 1;
}

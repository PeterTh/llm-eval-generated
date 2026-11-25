/**
 * Room Response Simulation Benchmark
 * 
 * This CUDA benchmark implements room impulse response
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
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
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return __fmaf_rn(z, o.z, __fmaf_rn(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {__fmaf_rn(y, o.z, -z * o.y), __fmaf_rn(z, o.x, -x * o.z),
                __fmaf_rn(x, o.y, -y * o.x)};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
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
// GPU scene and visibility
// ============================================================================

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
    T* data_ = nullptr;

public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }

    void allocate(size_t count, bool managed = false) {
        if (count == 0) return;
        checkCuda(managed ? cudaMallocManaged(&data_, count * sizeof(T))
                          : cudaMalloc(&data_, count * sizeof(T)),
                  "allocating CUDA memory");
    }

    void upload(const T* source, size_t count) {
        allocate(count);
        checkCuda(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
                  "uploading CUDA data");
    }

    void download(T* destination, size_t count) const {
        checkCuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                  "downloading CUDA results");
    }

    T* get() const { return data_; }
};

class CudaStream {
    cudaStream_t stream_ = nullptr;

public:
    CudaStream() {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "creating CUDA stream");
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() { cudaStreamDestroy(stream_); }

    cudaStream_t get() const { return stream_; }
};

class CudaEvent {
    cudaEvent_t event_ = nullptr;

public:
    CudaEvent() {
        checkCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "creating CUDA event");
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(event_); }

    void record(cudaStream_t stream) const {
        checkCuda(cudaEventRecord(event_, stream), "recording CUDA event");
    }

    void waitOn(cudaStream_t stream) const {
        checkCuda(cudaStreamWaitEvent(stream, event_), "ordering CUDA streams");
    }
};

struct RayTriangle {
    Vec3 a, ab, ac;
};

struct Patch {
    Vec3 center, normal;
    val_t area;
};

struct RayNode {
    Vec3 center, halfExtent;
    idx_t escape, first, count;
};

// Depth-first layout. The escape index skips a whole subtree on a box miss.
void flattenOctree(const Octree& tree, std::vector<RayNode>& nodes, std::vector<idx_t>& leaves) {
    const idx_t node = nodes.size();
    const idx_t first = leaves.size();
    const idx_t count = tree.triangleIndices.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, first, count});
    leaves.insert(leaves.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, leaves);
    }
    nodes[node].escape = nodes.size();
}

struct Scene {
    const RayTriangle* triangles;
    const Patch* patches;
    const RayNode* nodes;
    const idx_t* leaves;
    idx_t nodeCount;
    idx_t n;
};

__device__ bool intersectsBox(const RayNode& node, const Vec3& midpoint,
                               const Vec3& d, const Vec3& ad) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) {
        return false;
    }

    const Vec3 cross = d.cross(c);
    return fabsf(cross.x) <= __fmaf_rn(h.y, ad.z, h.z * ad.y) + EPSILON &&
           fabsf(cross.y) <= __fmaf_rn(h.z, ad.x, h.x * ad.z) + EPSILON &&
           fabsf(cross.z) <= __fmaf_rn(h.x, ad.y, h.y * ad.x) + EPSILON;
}

__device__ bool intersectsTriangle(const RayTriangle& tri, const Vec3& from,
                                    const Vec3& direction, val_t end) {
    const Vec3 p = direction.cross(tri.ac);
    const val_t det = tri.ab.dot(p);
    if (fabsf(det) < EPSILON) return false;

    const val_t invDet = ONE / det;
    const Vec3 delta = from - tri.a;
    const val_t u = delta.dot(p) * invDet;
    if (u < ZERO || u > ONE) return false;

    const Vec3 q = delta.cross(tri.ab);
    const val_t v = direction.dot(q) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    const val_t distance = tri.ac.dot(q) * invDet;
    return distance > EPSILON && distance < end;
}

__device__ bool rayBlocked(Scene scene, const Vec3& from, const Vec3& delta,
                           val_t length, idx_t i, idx_t j) {
    const Vec3 d = delta * 0.5f;
    const Vec3 midpoint = from + d;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3 direction = delta / length;
    const val_t end = length - EPSILON;

    // The CPU traversal entered the root without testing its bounds.
    for (idx_t node = 0; node < scene.nodeCount;) {
        const RayNode current = scene.nodes[node];
        if (node != 0 && !intersectsBox(current, midpoint, d, ad)) {
            node = current.escape;
            continue;
        }
        for (idx_t k = current.first; k < current.first + current.count; ++k) {
            const idx_t tri = scene.leaves[k];
            if (tri != i && tri != j && intersectsTriangle(scene.triangles[tri], from, direction, end)) {
                return true;
            }
        }
        ++node;
    }
    return false;
}

// ============================================================================
// Deterministic Monte Carlo sampling
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int PAIRS_PER_TILE = 128;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr int MT_WORDS = 624;
constexpr int MT_SHIFT = 397;
constexpr int MT_STEP = MT_WORDS - MT_SHIFT;
constexpr unsigned FULL_WARP = 0xffffffffu;

struct RandomCheckpoint {
    uint32_t words[MT_WORDS];
    int offset;
};

__device__ uint32_t twistBits(uint32_t a, uint32_t b) {
    const uint32_t bits = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (bits >> 1) ^ ((bits & 1u) ? 0x9908b0dfu : 0u);
}

__device__ uint32_t firstTwist(const uint32_t* old, int i) {
    return old[i + MT_SHIFT] ^ twistBits(old[i], old[i + 1]);
}

// Each thread follows one dependency chain of the in-place MT19937 twist.
// Only the final word needs results from other chains, which it recomputes.
__device__ void twist(const uint32_t* old, uint32_t* next) {
    for (int i = threadIdx.x; i < MT_STEP; i += blockDim.x) {
        const uint32_t first = firstTwist(old, i);
        const uint32_t second = first ^ twistBits(old[i + MT_STEP], old[i + MT_STEP + 1]);
        next[i] = first;
        next[i + MT_STEP] = second;
        if (i + 2 * MT_STEP < MT_WORDS - 1) {
            next[i + 2 * MT_STEP] = second ^ twistBits(old[i + 2 * MT_STEP], old[i + 2 * MT_STEP + 1]);
        }
    }
    if (threadIdx.x == 0) {
        constexpr int lastDependency = MT_SHIFT - 1 - MT_STEP;
        const uint32_t dependency = firstTwist(old, lastDependency) ^
                                    twistBits(old[MT_SHIFT - 1], old[MT_SHIFT]);
        next[MT_WORDS - 1] = dependency ^ twistBits(old[MT_WORDS - 1], firstTwist(old, 0));
    }
    __syncthreads();
}

__device__ val_t uniformFloat(uint32_t value) {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    value ^= value >> 18;
    // Match libstdc++ uniform_real_distribution<float> including its upper bound.
    return fminf(__uint2float_rn(value) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ bool samplesPair(Scene scene, size_t pair, idx_t& i, idx_t& j) {
    i = pair / scene.n;
    j = pair - size_t(i) * scene.n;
    return i < scene.n && i != j && scene.patches[i].normal.dot(scene.patches[j].normal) <= 0.99f;
}

__device__ unsigned warpSum(unsigned value) {
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        value += __shfl_down_sync(FULL_WARP, value, stride);
    }
    return value;
}

__global__ void countSampledPairs(Scene scene, size_t firstPair, idx_t* counts) {
    __shared__ idx_t warpCounts[PAIRS_PER_TILE / WARP_SIZE];
    idx_t i, j;
    const size_t pair = firstPair + size_t(blockIdx.x) * PAIRS_PER_TILE + threadIdx.x;
    const unsigned mask = __ballot_sync(FULL_WARP, samplesPair(scene, pair, i, j));
    if (threadIdx.x % WARP_SIZE == 0) warpCounts[threadIdx.x / WARP_SIZE] = __popc(mask);
    __syncthreads();
    if (threadIdx.x == 0) {
        idx_t sum = 0;
        for (idx_t count : warpCounts) sum += count;
        counts[blockIdx.x] = sum;
    }
}

// The original sampler uses one stream in receiver-major order. Save a state
// at each tile boundary so all tiles can replay their own draws concurrently.
// The terminal state is carried across batches to bound temporary memory.
__global__ void checkpointRandom(RandomCheckpoint* stream, const idx_t* counts,
                                 RandomCheckpoint* checkpoints, int tiles) {
    __shared__ uint32_t words[2][MT_WORDS];
    for (int i = threadIdx.x; i < MT_WORDS; i += blockDim.x) words[0][i] = stream->words[i];
    int offset = stream->offset;
    int current = 0;
    __syncthreads();

    if (offset == MT_WORDS) {
        twist(words[0], words[1]);
        current = 1;
        offset = 0;
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int draws = counts[tile] * DRAWS_PER_PAIR;
        if (draws == 0) continue;
        for (int i = threadIdx.x; i < MT_WORDS; i += blockDim.x) {
            checkpoints[tile].words[i] = words[current][i];
        }
        if (threadIdx.x == 0) checkpoints[tile].offset = offset;
        offset += draws;
        while (offset >= MT_WORDS) {
            twist(words[current], words[1 - current]);
            current = 1 - current;
            offset -= MT_WORDS;
        }
    }

    for (int i = threadIdx.x; i < MT_WORDS; i += blockDim.x) stream->words[i] = words[current][i];
    if (threadIdx.x == 0) stream->offset = offset;
}

__device__ Vec3 samplePoint(const RayTriangle& tri, uint32_t drawU, uint32_t drawV) {
    val_t u = uniformFloat(drawU);
    val_t v = uniformFloat(drawV);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {
        __fmaf_rn(tri.ac.x, v, __fmaf_rn(tri.ab.x, u, tri.a.x)),
        __fmaf_rn(tri.ac.y, v, __fmaf_rn(tri.ab.y, u, tri.a.y)),
        __fmaf_rn(tri.ac.z, v, __fmaf_rn(tri.ab.z, u, tri.a.z))
    };
}

__device__ val_t evaluateRay(Scene scene, idx_t i, idx_t j, const uint32_t* draws) {
    const Vec3 from = samplePoint(scene.triangles[i], draws[0], draws[1]);
    const Vec3 to = samplePoint(scene.triangles[j], draws[2], draws[3]);
    const Vec3 delta = to - from;
    const val_t squaredDistance = delta.dot(delta);
    if (squaredDistance < EPSILON) return ZERO;

    const val_t length = sqrtf(squaredDistance);
    const val_t cosI = delta.dot(scene.patches[i].normal) / length;
    const val_t cosJ = (-delta).dot(scene.patches[j].normal) / length;
    if (cosI <= ZERO || cosJ <= ZERO || rayBlocked(scene, from, delta, length, i, j)) return ZERO;
    return (cosI * cosJ) / (PI * squaredDistance);
}

__global__ __launch_bounds__(PAIRS_PER_TILE)
void precomputePairs(Scene scene, size_t firstPair, const RandomCheckpoint* checkpoints,
                     val_t* weights, uint8_t* delays, unsigned long long* nonZeroKij) {
    __shared__ idx_t warpCounts[PAIRS_PER_TILE / WARP_SIZE];
    __shared__ uint2 sampled[PAIRS_PER_TILE];
    constexpr int PAIRS_PER_ROUND = PAIRS_PER_TILE / NUM_RAYS;
    __shared__ uint32_t randomWords[2][PAIRS_PER_ROUND * DRAWS_PER_PAIR];
    __shared__ uint32_t words[2][MT_WORDS];
    __shared__ val_t contributions[PAIRS_PER_TILE];

    const int thread = threadIdx.x;
    const int lane = thread % WARP_SIZE;
    const int warp = thread / WARP_SIZE;
    const size_t tileStart = firstPair + size_t(blockIdx.x) * PAIRS_PER_TILE;
    idx_t i, j;
    const bool active = samplesPair(scene, tileStart + thread, i, j);
    const unsigned mask = __ballot_sync(FULL_WARP, active);
    if (lane == 0) warpCounts[warp] = __popc(mask);

    if (!active && i < scene.n) {
        const size_t index = tileStart + thread;
        weights[index] = ZERO;
        delays[index] = 0;
    }
    __syncthreads();

    int sampledCount = 0;
    int offsetInTile = 0;
    for (int w = 0; w < PAIRS_PER_TILE / WARP_SIZE; ++w) {
        if (w == warp) offsetInTile = sampledCount;
        sampledCount += warpCounts[w];
    }
    if (sampledCount == 0) return;
    if (active) {
        sampled[offsetInTile + __popc(mask & ((1u << lane) - 1u))] = make_uint2(i, j);
    }

    const RandomCheckpoint& checkpoint = checkpoints[blockIdx.x];
    for (int w = thread; w < MT_WORDS; w += PAIRS_PER_TILE) words[0][w] = checkpoint.words[w];
    __syncthreads();

    int wordOffset = checkpoint.offset;
    int current = 0;
    int buffer = 0;
    unsigned valid = 0;
    const int ray = thread % NUM_RAYS;
    const int pairInRound = thread / NUM_RAYS;
    const unsigned rayGroup = ((1u << NUM_RAYS) - 1u) << ((lane / NUM_RAYS) * NUM_RAYS);

    // Stage only the draws for the current group of rays to keep shared
    // memory usage low. The other buffer is still available to tracing warps.
    for (int first = 0; first < sampledCount; first += PAIRS_PER_ROUND) {
        const int pairCount = min(PAIRS_PER_ROUND, sampledCount - first);
        const int drawCount = pairCount * DRAWS_PER_PAIR;
        for (int written = 0; written < drawCount;) {
            if (wordOffset == MT_WORDS) {
                twist(words[current], words[1 - current]);
                current = 1 - current;
                wordOffset = 0;
            }
            const int count = min(drawCount - written, MT_WORDS - wordOffset);
            for (int w = thread; w < count; w += PAIRS_PER_TILE) {
                randomWords[buffer][written + w] = words[current][wordOffset + w];
            }
            written += count;
            wordOffset += count;
        }
        __syncthreads();

        if (pairInRound < pairCount) {
            const uint2 pair = sampled[first + pairInRound];
            const idx_t receiver = pair.x;
            const idx_t emitter = pair.y;
            const uint32_t* draws = randomWords[buffer] + pairInRound * DRAWS_PER_PAIR + 4 * ray;
            contributions[thread] = evaluateRay(scene, receiver, emitter, draws);
            __syncwarp(rayGroup);

            if (ray == 0) {
                val_t sum = ZERO;
                // Preserve the order of the serial floating-point sum.
                #pragma unroll
                for (int r = 0; r < NUM_RAYS; ++r) sum += contributions[thread + r];
                const val_t kij = sum * INV_NUM_RAYS;
                valid += kij > EPSILON;

                const Vec3 delta = scene.patches[receiver].center - scene.patches[emitter].center;
                const size_t index = size_t(receiver) * scene.n + emitter;
                weights[index] = fminf(kij * scene.patches[emitter].area, ONE);
                // The room has diameter 20 and the wave speed is 0.5.
                delays[index] = __float2int_ru(sqrtf(delta.dot(delta)) * INV_WAVE_SPEED);
            }
        }
        buffer = 1 - buffer;
    }

    if (nonZeroKij) {
        const unsigned subtotal = warpSum(valid);
        if (lane == 0) warpCounts[warp] = subtotal;
        __syncthreads();
        if (thread == 0) {
            unsigned sum = 0;
            for (idx_t count : warpCounts) sum += count;
            if (sum != 0) atomicAdd(nonZeroKij, static_cast<unsigned long long>(sum));
        }
    }
}

// ============================================================================
// Wave propagation and distance estimation
// ============================================================================

constexpr int PROPAGATION_WARPS = 4;

__global__ void propagateStep(const val_t* __restrict__ weights, const uint8_t* __restrict__ delays,
                              val_t* __restrict__ history, idx_t n, idx_t t, idx_t source,
                              idx_t timeOff, val_t reflectivity) {
    __shared__ val_t contributions[2][PROPAGATION_WARPS][WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const idx_t receiver = (blockIdx.x * blockDim.x) / WARP_SIZE + warp;
    if (receiver >= n) return;

    val_t sum = ZERO;
    const size_t row = size_t(receiver) * n;
    int buffer = 0;
    // A warp loads a contiguous group of emitters. Its first lane folds
    // their contributions in the same order as the serial implementation.
    for (idx_t base = 0; base < n && t != 0; base += WARP_SIZE) {
        const idx_t emitter = base + lane;
        val_t contribution = ZERO;
        if (emitter < n) {
            const idx_t delay = delays[row + emitter];
            if (delay != 0 && delay <= t) {
                const val_t weight = weights[row + emitter];
                if (weight > ZERO) {
                    const val_t past = history[size_t(t - delay) * n + emitter];
                    if (!(past <= ZERO)) contribution = weight * past;
                }
            }
        }

        contributions[buffer][warp][lane] = contribution;
        __syncwarp();
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < WARP_SIZE; ++j) sum += contributions[buffer][warp][j];
        }
        buffer = 1 - buffer;
    }

    if (lane == 0) {
        const val_t emission = t < timeOff && receiver == source ? ONE : ZERO;
        history[size_t(t) * n + receiver] = __fmaf_rn(reflectivity, sum, emission);
    }
}

constexpr int CORRELATION_WARPS = 8;

struct CorrelationPeak {
    val_t value;
    idx_t lag;

    __device__ void include(CorrelationPeak other) {
        if (other.value > value || (other.value == value && other.lag < lag)) {
            *this = other;
        }
    }
};

__global__ void correlateDistances(const val_t* __restrict__ history, val_t* __restrict__ distances,
                                   CorrelationPeak* __restrict__ partial, idx_t n, idx_t steps,
                                   idx_t source, idx_t* receivedEnergy) {
    __shared__ CorrelationPeak peaks[CORRELATION_WARPS][WARP_SIZE];
    const idx_t receiver = blockIdx.x * WARP_SIZE + threadIdx.x;
    const idx_t warp = threadIdx.y;
    CorrelationPeak best{ZERO, 0};
    bool received = false;

    if (receiver < n) {
        for (idx_t lag = blockIdx.y * CORRELATION_WARPS + warp;
             lag < steps; lag += gridDim.y * CORRELATION_WARPS) {
            val_t sum = ZERO;
            for (idx_t t = lag; t < steps; ++t) {
                const val_t reflected = history[size_t(t) * n + receiver];
                const val_t emitted = history[size_t(t - lag) * n + source];
                sum += emitted * reflected;
                if (lag == 0 && reflected > EPSILON) received = true;
            }
            best.include({sum, lag});
        }
    }
    peaks[warp][threadIdx.x] = best;
    const int energized = __syncthreads_count(received);
    if (warp == 0 && receiver < n) {
        for (int w = 1; w < CORRELATION_WARPS; ++w) best.include(peaks[w][threadIdx.x]);
        if (partial) {
            partial[size_t(blockIdx.y) * n + receiver] = best;
        } else {
            distances[receiver] = WAVE_SPEED * static_cast<val_t>(best.lag);
        }
        if (threadIdx.x == 0 && receivedEnergy && energized != 0) atomicAdd(receivedEnergy, energized);
    }
}

__global__ void combinePeaks(const CorrelationPeak* __restrict__ partial,
                             val_t* __restrict__ distances, idx_t n, idx_t tiles) {
    const idx_t receiver = blockIdx.x * blockDim.x + threadIdx.x;
    if (receiver >= n) return;

    CorrelationPeak best{ZERO, 0};
    for (idx_t tile = 0; tile < tiles; ++tile) best.include(partial[size_t(tile) * n + receiver]);
    distances[receiver] = WAVE_SPEED * static_cast<val_t>(best.lag);
}

// ============================================================================
// Host orchestration
// ============================================================================

struct SimulationState {
    idx_t numTriangles;
    idx_t numTimesteps;
    idx_t sourceIndex;
    val_t reflectivity;
    bool validate;
    std::vector<val_t> distances;
    idx_t receivedEnergy = 0;
    unsigned long long nonZeroKij = 0;

    DeviceBuffer<RayTriangle> triangles;
    DeviceBuffer<Patch> patches;
    DeviceBuffer<RayNode> nodes;
    DeviceBuffer<idx_t> leaves;
    DeviceBuffer<val_t> weights, history, gpuDistances;
    DeviceBuffer<uint8_t> delays;
    DeviceBuffer<unsigned long long> gpuNonZeroKij;
    DeviceBuffer<idx_t> gpuReceivedEnergy;
    idx_t nodeCount;

    SimulationState(int subdivisions, idx_t steps, size_t source, val_t rho, bool validate)
        : numTimesteps(steps), reflectivity(rho), validate(validate) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        numTriangles = mesh.triangles.size();
        sourceIndex = source % numTriangles;
        distances.resize(numTriangles);
        printf("Generated icosphere mesh with %u triangles\n", numTriangles);
        printf("Building octree...\n");

        Octree tree;
        tree.build(mesh.triangles);
        std::vector<RayNode> flatNodes;
        std::vector<idx_t> flatLeaves;
        flattenOctree(tree, flatNodes, flatLeaves);
        nodeCount = flatNodes.size();

        std::vector<RayTriangle> geometry;
        std::vector<Patch> surfaces;
        geometry.reserve(numTriangles);
        surfaces.reserve(numTriangles);
        for (const Triangle& tri : mesh.triangles) {
            geometry.push_back({tri.a, tri.b - tri.a, tri.c - tri.a});
            surfaces.push_back({tri.center(), tri.normal(), tri.area()});
        }
        triangles.upload(geometry.data(), geometry.size());
        patches.upload(surfaces.data(), surfaces.size());
        nodes.upload(flatNodes.data(), flatNodes.size());
        leaves.upload(flatLeaves.data(), flatLeaves.size());

        const size_t pairs = size_t(numTriangles) * numTriangles;
        const size_t samples = size_t(numTriangles) * numTimesteps;
        size_t freeBytes, totalBytes;
        checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "querying CUDA memory");
        const size_t matrixBytes = pairs * (sizeof(val_t) + sizeof(uint8_t));
        const size_t historyBytes = samples * sizeof(val_t);
        // Allow large inputs to use managed CUDA storage when device memory is tight.
        const bool managed = matrixBytes + historyBytes > freeBytes - freeBytes / 16;
        weights.allocate(pairs, managed);
        delays.allocate(pairs, managed);
        history.allocate(samples, managed);
        gpuDistances.allocate(numTriangles);
        if (validate) {
            gpuNonZeroKij.upload(&nonZeroKij, 1);
            gpuReceivedEnergy.upload(&receivedEnergy, 1);
        }
    }

    void precompute() {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");

        constexpr int MAX_TILES = 1024;
        const size_t pairs = size_t(numTriangles) * numTriangles;
        const size_t totalTiles = (pairs + PAIRS_PER_TILE - 1) / PAIRS_PER_TILE;
        const int batchTiles = std::min(totalTiles, size_t(MAX_TILES));
        DeviceBuffer<RandomCheckpoint> streamState;
        DeviceBuffer<idx_t> counts;
        counts.allocate(batchTiles);

        CudaStream sampling, tracing;
        struct Batch {
            DeviceBuffer<RandomCheckpoint> checkpoints;
            CudaEvent ready, consumed;
        };
        Batch batches[2];
        for (Batch& batch : batches) batch.checkpoints.allocate(batchTiles);

        RandomCheckpoint seed{};
        seed.words[0] = 42;
        for (int i = 1; i < MT_WORDS; ++i) {
            seed.words[i] = 1812433253u * (seed.words[i - 1] ^ (seed.words[i - 1] >> 30)) + i;
        }
        seed.offset = MT_WORDS;
        streamState.upload(&seed, 1);
        checkCuda(cudaStreamSynchronize(nullptr), "finishing precomputation uploads");
        const Scene scene{triangles.get(), patches.get(), nodes.get(), leaves.get(),
                          nodeCount, numTriangles};
        // Generate checkpoints for the next batch while other SMs trace
        // this one. A slot is only reused once tracing has consumed it.
        size_t batchIndex = 0;
        for (size_t tile = 0; tile < totalTiles; tile += batchTiles, ++batchIndex) {
            Batch& batch = batches[batchIndex % 2];
            const int tiles = std::min(totalTiles - tile, size_t(batchTiles));
            const size_t first = tile * PAIRS_PER_TILE;

            if (batchIndex >= 2) batch.consumed.waitOn(sampling.get());
            countSampledPairs<<<tiles, PAIRS_PER_TILE, 0, sampling.get()>>>(scene, first, counts.get());
            checkpointRandom<<<1, 256, 0, sampling.get()>>>(
                streamState.get(), counts.get(), batch.checkpoints.get(), tiles);
            checkCuda(cudaGetLastError(), "launching visibility sampling");
            batch.ready.record(sampling.get());

            batch.ready.waitOn(tracing.get());
            precomputePairs<<<tiles, PAIRS_PER_TILE, 0, tracing.get()>>>(
                scene, first, batch.checkpoints.get(), weights.get(), delays.get(), gpuNonZeroKij.get());
            checkCuda(cudaGetLastError(), "launching form-factor precomputation");
            batch.consumed.record(tracing.get());
        }
        checkCuda(cudaStreamSynchronize(tracing.get()), "computing delays and form factors");
        if (validate) gpuNonZeroKij.download(&nonZeroKij, 1);
    }

    void propagate() {
        printf("Running wave propagation simulation...\n");
        const int warps = numTriangles < 320 ? 1 : PROPAGATION_WARPS;
        const int blocks = (numTriangles + warps - 1) / warps;
        for (idx_t t = 0; t < numTimesteps; ++t) {
            propagateStep<<<blocks, warps * WARP_SIZE>>>(weights.get(), delays.get(), history.get(),
                                                numTriangles, t, sourceIndex, numTimesteps / 2,
                                                reflectivity);
            checkCuda(cudaGetLastError(), "launching wave propagation");
        }
        checkCuda(cudaDeviceSynchronize(), "running wave propagation");
    }

    void correlate() {
        printf("Computing distances via cross-correlation...\n");
        const idx_t receiverTiles = (numTriangles + WARP_SIZE - 1) / WARP_SIZE;
        // Long signals expose more independent lags than fit in one block
        // per receiver tile. Keep only each block's best lag in global memory.
        const idx_t lagTiles = std::max(1u, std::min({numTimesteps / 64, 128u, 2048u / receiverTiles}));
        DeviceBuffer<CorrelationPeak> partial;
        if (lagTiles > 1) partial.allocate(size_t(lagTiles) * numTriangles);

        correlateDistances<<<dim3(receiverTiles, lagTiles), dim3(WARP_SIZE, CORRELATION_WARPS)>>>(
            history.get(), gpuDistances.get(), partial.get(), numTriangles, numTimesteps,
            sourceIndex, gpuReceivedEnergy.get());
        checkCuda(cudaGetLastError(), "launching cross-correlation");
        if (lagTiles > 1) {
            constexpr int threads = 128;
            const int blocks = (numTriangles + threads - 1) / threads;
            combinePeaks<<<blocks, threads>>>(partial.get(), gpuDistances.get(), numTriangles, lagTiles);
            checkCuda(cudaGetLastError(), "combining cross-correlation peaks");
        }
        gpuDistances.download(distances.data(), distances.size());
        if (validate) gpuReceivedEnergy.download(&receivedEnergy, 1);
    }
};

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");
    const size_t n = state.numTriangles;

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < n; ++i) {
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    const idx_t receivedEnergy = state.receivedEnergy;

    printf("  Triangles receiving energy: %u/%zu\n", receivedEnergy, n);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const auto nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, n * n,
           100.0f * nonZeroKij / static_cast<val_t>(n * n));

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
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
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
    if (timesteps < 0) {
        fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
    }
    SimulationState state(subdivisions, timesteps, static_cast<size_t>(sourceIdx),
                          reflectivity, validate);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    state.precompute();

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    state.propagate();

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    state.correlate();

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
    size_t memTau = n * n * sizeof(uint8_t);
    size_t memRad = t * n * sizeof(val_t);
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

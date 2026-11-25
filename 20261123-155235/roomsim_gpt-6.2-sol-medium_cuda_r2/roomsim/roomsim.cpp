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

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

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

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
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
// GPU data and visibility
// ============================================================================

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }

    void allocate(size_t count, bool managed = false) {
        if (count == 0) return;
        if (managed) {
            CUDA_CHECK(cudaMallocManaged(&data, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }

    void upload(const T* source, size_t count) {
        CUDA_CHECK(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* destination, size_t count) const {
        CUDA_CHECK(cudaMemcpy(destination, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* data = nullptr;
};

template<typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(&data, count * sizeof(T)));
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() { cudaFreeHost(data); }

    T* data = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(event); }

    void record() { CUDA_CHECK(cudaEventRecord(event)); }
    void wait() const { CUDA_CHECK(cudaEventSynchronize(event)); }

private:
    cudaEvent_t event;
};

struct alignas(16) Patch {
    Vec3 vertex, edge1, edge2, normal, center;
    val_t area;
};

struct alignas(16) VisibilityNode {
    Vec3 center, halfExtent;
    // An internal node's end skips its subtree. A leaf's end is one
    // past its last triangle reference.
    int first;
    idx_t end;
};

void flattenOctree(const Octree& node, std::vector<VisibilityNode>& nodes,
                   std::vector<idx_t>& references) {
    size_t index = nodes.size();
    nodes.push_back({node.center, node.halfExtent, -1, 0});
    if (!node.triangleIndices.empty()) {
        nodes[index].first = static_cast<int>(references.size());
        references.insert(references.end(), node.triangleIndices.begin(), node.triangleIndices.end());
        nodes[index].end = static_cast<idx_t>(references.size());
    } else {
        for (const auto& child : node.children) {
            if (child) flattenOctree(*child, nodes, references);
        }
        nodes[index].end = static_cast<idx_t>(nodes.size());
    }
}

__device__ bool rayIntersectsBox(const Vec3& midpoint, const Vec3& d, const Vec3& ad,
                                 const VisibilityNode& node) {
    Vec3 c = midpoint - node.center;
    Vec3 half = node.halfExtent;
    if (fabsf(c.x) > half.x + ad.x || fabsf(c.y) > half.y + ad.y ||
        fabsf(c.z) > half.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > half.y * ad.z + half.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > half.z * ad.x + half.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > half.x * ad.y + half.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool rayIntersectsTriangle(const Vec3& origin, const Vec3& direction,
                                       val_t limit, const Patch& patch) {
    Vec3 pvec = direction.cross(patch.edge2);
    val_t det = patch.edge1.dot(pvec);
    if (fabsf(det) < EPSILON) return false;

    val_t invDet = 1.0f / det;
    Vec3 tvec = origin - patch.vertex;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return false;

    Vec3 qvec = tvec.cross(patch.edge1);
    val_t v = direction.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    val_t distance = patch.edge2.dot(qvec) * invDet;
    return distance > EPSILON && distance < limit;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& ray, val_t length,
                              idx_t src, idx_t dst, const Patch* patches,
                              const VisibilityNode* nodes, idx_t nodeCount,
                              const idx_t* references) {
    Vec3 direction = ray / length;
    Vec3 d = ray * 0.5f;
    Vec3 midpoint = from + d;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    val_t limit = length - EPSILON;

    // The original traversal always enters the root and tests child boxes.
    idx_t index = nodes[0].first < 0 ? 1 : 0;
    while (index < nodeCount) {
        const VisibilityNode& node = nodes[index];
        bool intersects = index == 0 || rayIntersectsBox(midpoint, d, ad, node);
        if (node.first < 0) {
            index = intersects ? index + 1 : node.end;
            continue;
        }
        if (intersects) {
            for (idx_t ref = node.first; ref < node.end; ++ref) {
                idx_t tri = references[ref];
                if (tri != src && tri != dst &&
                    rayIntersectsTriangle(from, direction, limit, patches[tri])) return true;
            }
        }
        ++index;
    }
    return false;
}

// ============================================================================
// Parallel sampling of the original MT19937 stream
// ============================================================================

constexpr int MT_WORDS = 624;
constexpr int MT_TAP = 397;
constexpr int MT_STRIDE = MT_WORDS - MT_TAP;
constexpr int TWISTS_PER_CHECKPOINT = 8;
constexpr int WORDS_PER_PAIR = 4 * NUM_RAYS;
constexpr int PAIRS_PER_CHECKPOINT = TWISTS_PER_CHECKPOINT * MT_WORDS / WORDS_PER_PAIR;
static_assert(TWISTS_PER_CHECKPOINT * MT_WORDS % WORDS_PER_PAIR == 0);
static_assert(MT_WORDS % 4 == 0);

using MtState = std::array<uint32_t, MT_WORDS>;

__host__ __device__ uint32_t twistWord(uint32_t first, uint32_t second, uint32_t tap) {
    uint32_t bits = (first & 0x80000000U) | (second & 0x7fffffffU);
    return tap ^ (bits >> 1) ^ ((0U - (bits & 1U)) & 0x9908b0dfU);
}

class RandomCheckpoints {
public:
    RandomCheckpoints() {
        words[0][0] = 42;
        for (int i = 1; i < MT_WORDS; ++i) {
            uint32_t prev = words[0][i - 1];
            words[0][i] = 1812433253U * (prev ^ (prev >> 30)) + i;
        }
    }

    void collect(size_t first, idx_t count, uint32_t* checkpoints) {
        for (idx_t i = 0; i < count; ++i) {
            // A batch may start in the last checkpoint used by its predecessor.
            while (position < first + i) {
                for (int twist = 0; twist < TWISTS_PER_CHECKPOINT; ++twist) advance();
                ++position;
            }
            std::copy_n(words[active].data(), MT_WORDS, checkpoints + i * MT_WORDS);
        }
    }

private:
    void advance() {
        const auto& prev = words[active];
        auto& next = words[1 - active];
        for (int i = 0; i < MT_STRIDE; ++i) {
            next[i] = twistWord(prev[i], prev[i + 1], prev[i + MT_TAP]);
        }
        for (int i = MT_STRIDE; i < 2 * MT_STRIDE; ++i) {
            next[i] = twistWord(prev[i], prev[i + 1], next[i - MT_STRIDE]);
        }
        for (int i = 2 * MT_STRIDE; i < MT_WORDS - 1; ++i) {
            next[i] = twistWord(prev[i], prev[i + 1], next[i - MT_STRIDE]);
        }
        next.back() = twistWord(prev.back(), next.front(), next[MT_TAP - 1]);
        active = 1 - active;
    }

    MtState words[2];
    size_t position = 0;
    int active = 0;
};

__device__ val_t temperedUniform(uint32_t value) {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680U;
    value ^= (value << 15) & 0xefc60000U;
    value ^= value >> 18;
    // uniform_real_distribution<float> uses one word and excludes 1.0.
    return fminf(__uint2float_rn(value) * 0x1p-32f, 0x1.fffffep-1f);
}

__global__ void generateSamples(const uint32_t* __restrict__ checkpoints, float4* __restrict__ samples) {
    __shared__ uint32_t states[2][MT_WORDS];
    const uint32_t* initial = checkpoints + blockIdx.x * MT_WORDS;
    float4* output = samples + blockIdx.x * TWISTS_PER_CHECKPOINT * (MT_WORDS / 4);
    int lane = threadIdx.x;

    for (int i = lane; i < MT_WORDS; i += blockDim.x) states[0][i] = initial[i];
    __syncthreads();

    // Each lane follows one chain of taps in a twist. Separate input and
    // output states allow all chains to run with a single barrier per twist.
    for (int twist = 0; twist < TWISTS_PER_CHECKPOINT; ++twist) {
        const uint32_t* prev = states[twist % 2];
        uint32_t* next = states[1 - twist % 2];
        if (lane < MT_STRIDE) {
            uint32_t word = twistWord(prev[lane], prev[lane + 1], prev[lane + MT_TAP]);
            next[lane] = word;

            int i = lane + MT_STRIDE;
            word = twistWord(prev[i], prev[i + 1], word);
            next[i] = word;

            i += MT_STRIDE;
            if (i < MT_WORDS - 1) {
                next[i] = twistWord(prev[i], prev[i + 1], word);
            } else if (i == MT_WORDS - 1) {
                uint32_t first = twistWord(prev[0], prev[1], prev[MT_TAP]);
                next[i] = twistWord(prev[i], first, word);
            }
        }
        __syncthreads();

        if (lane < MT_WORDS / 4) {
            const uint32_t* words = next + 4 * lane;
            output[lane] = make_float4(temperedUniform(words[0]), temperedUniform(words[1]),
                                       temperedUniform(words[2]), temperedUniform(words[3]));
        }
        output += MT_WORDS / 4;
    }
}

// ============================================================================
// Form factors and time delays
// ============================================================================

constexpr int FORM_FACTOR_THREADS = 128;
constexpr int PAIRS_PER_BLOCK = FORM_FACTOR_THREADS / NUM_RAYS;
constexpr int PAIR_BATCH_SIZE = 1 << 18;
static_assert(NUM_RAYS == 16);

// Use batch-local offsets so even a matrix with more than 2^32
// elements can be compacted and indexed using fast 32-bit division.
struct PairBatch {
    idx_t width, firstRow, firstColumn;

    __device__ size_t index(idx_t row, idx_t column) const {
        return static_cast<size_t>(row) * width + column;
    }

    __device__ void coordinates(idx_t offset, idx_t& row, idx_t& column) const {
        idx_t position = firstColumn + offset;
        row = firstRow + position / width;
        column = position % width;
    }
};

__global__ void classifyPairs(const Patch* __restrict__ patches, PairBatch batch, idx_t count,
                              uint8_t* __restrict__ flags, uint8_t* __restrict__ delays) {
    idx_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    idx_t i, j;
    batch.coordinates(local, i, j);
    bool sample = i != j && !(patches[i].normal.dot(patches[j].normal) > 0.99f);
    flags[local] = sample;
    if (!sample) delays[batch.index(i, j)] = 0;
}

__device__ Vec3 samplePoint(const Patch& patch, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return patch.vertex + patch.edge1 * u + patch.edge2 * v;
}

__device__ val_t rayContribution(const Patch& receiver, const Patch& emitter,
                                  const float4& samples, idx_t i, idx_t j,
                                  const Patch* patches, const VisibilityNode* nodes,
                                  idx_t nodeCount, const idx_t* references) {
    Vec3 from = samplePoint(receiver, samples.x, samples.y);
    Vec3 to = samplePoint(emitter, samples.z, samples.w);
    Vec3 ray = to - from;
    val_t distanceSquared = ray.squaredNorm();
    if (distanceSquared < EPSILON) return ZERO;

    val_t length = sqrtf(distanceSquared);
    val_t cosI = ray.dot(receiver.normal) / length;
    val_t cosJ = (-ray).dot(emitter.normal) / length;
    if (cosI <= ZERO || cosJ <= ZERO) return ZERO;
    if (isRayBlocked(from, ray, length, i, j, patches, nodes, nodeCount, references)) return ZERO;

    return (cosI * cosJ) / (PI * distanceSquared);
}

__global__ void formFactors(const Patch* __restrict__ patches, PairBatch batch,
                            const VisibilityNode* __restrict__ nodes, idx_t nodeCount,
                            const idx_t* __restrict__ references, const idx_t* __restrict__ pairs,
                            idx_t count, const float4* __restrict__ samples, val_t* __restrict__ weights,
                            uint8_t* __restrict__ delays, unsigned long long* __restrict__ nonzero) {
    __shared__ val_t rays[FORM_FACTOR_THREADS];
    int lane = threadIdx.x % NUM_RAYS;
    idx_t work = blockIdx.x * PAIRS_PER_BLOCK + threadIdx.x / NUM_RAYS;
    idx_t i, j;
    if (work < count) {
        batch.coordinates(pairs[work], i, j);
        rays[threadIdx.x] = rayContribution(patches[i], patches[j], samples[work * NUM_RAYS + lane],
                                            i, j, patches, nodes, nodeCount, references);
    }
    __syncwarp();

    bool positive = false;
    if (lane == 0 && work < count) {
        // Keep the original floating-point accumulation order.
        val_t kij = ZERO;
        #pragma unroll
        for (int ray = 0; ray < NUM_RAYS; ++ray) kij += rays[threadIdx.x + ray];
        kij *= INV_NUM_RAYS;
        size_t pair = batch.index(i, j);
        uint8_t delay = 0;
        if (kij > ZERO) {
            const Patch& receiver = patches[i];
            const Patch& emitter = patches[j];
            weights[pair] = fminf(kij * emitter.area, ONE);
            // Radius 10 and wave speed 0.5 bound the delay by 40.
            delay = static_cast<uint8_t>(ceilf((receiver.center - emitter.center).norm() * INV_WAVE_SPEED));
        }
        delays[pair] = delay;
        positive = kij > EPSILON;
    }
    if (nonzero) {
        int subtotal = __syncthreads_count(positive);
        if (threadIdx.x == 0 && subtotal) atomicAdd(nonzero, subtotal);
    }
}

// ============================================================================
// Wave propagation and cross-correlation
// ============================================================================

constexpr int PROPAGATION_THREADS = 128;
constexpr int PROPAGATION_TILE = 512;

__global__ void propagate(const val_t* __restrict__ weights, const uint8_t* __restrict__ delays,
                          val_t* __restrict__ history, idx_t n, int timestep, int emissionEnd,
                          idx_t source, val_t reflectivity) {
    idx_t receiver = blockIdx.x;
    size_t row = static_cast<size_t>(receiver) * n;
    val_t emission = receiver == source && timestep < emissionEnd ? ONE : ZERO;
    val_t sum = ZERO;
    __shared__ val_t terms[PROPAGATION_TILE];

    for (idx_t base = 0; timestep > 0 && base < n; base += PROPAGATION_TILE) {
        idx_t count = min(static_cast<idx_t>(PROPAGATION_TILE), n - base);
        for (idx_t offset = threadIdx.x; offset < count; offset += blockDim.x) {
            idx_t emitter = base + offset;
            int delay = delays[row + emitter];
            val_t term = ZERO;
            if (delay > 0 && delay <= timestep) {
                val_t incoming = history[static_cast<size_t>(timestep - delay) * n + emitter];
                if (!(incoming <= ZERO)) term = weights[row + emitter] * incoming;
            }
            terms[offset] = term;
        }
        __syncthreads();

        // Each receiver adds emitter contributions in mesh order.
        if (threadIdx.x == 0) {
            for (idx_t offset = 0; offset < count; ++offset) sum += terms[offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        history[static_cast<size_t>(timestep) * n + receiver] = fmaf(reflectivity, sum, emission);
    }
}

constexpr int DISTANCE_THREADS = 128;

struct Correlation {
    val_t value;
    int lag;
};

struct BestCorrelation {
    __device__ Correlation operator()(Correlation a, Correlation b) const {
        return b.value > a.value || (b.value == a.value && b.lag < a.lag) ? b : a;
    }
};

__global__ void correlate(const val_t* __restrict__ history, idx_t n, int timesteps, idx_t source,
                          val_t* __restrict__ distances, uint8_t* __restrict__ received) {
    idx_t receiver = blockIdx.x;
    Correlation best{ZERO, 0};
    bool hasEnergy = false;
    for (int lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        hasEnergy |= history[static_cast<size_t>(lag) * n + receiver] > EPSILON;
        val_t sum = ZERO;
        for (int t = lag; t < timesteps; ++t) {
            val_t sample = history[static_cast<size_t>(t) * n + receiver];
            val_t reference = history[static_cast<size_t>(t - lag) * n + source];
            sum = __fadd_rn(sum, __fmul_rn(sample, reference));
        }
        if (sum > best.value) best = {sum, lag};
    }

    using Reduction = cub::BlockReduce<Correlation, DISTANCE_THREADS>;
    __shared__ Reduction::TempStorage scratch;
    best = Reduction(scratch).Reduce(best, BestCorrelation{});
    if (threadIdx.x == 0) distances[receiver] = WAVE_SPEED * static_cast<val_t>(best.lag);
    if (received) {
        int any = __syncthreads_or(hasEnergy);
        if (threadIdx.x == 0) received[receiver] = any;
    }
}

struct SimulationState {
    size_t numTriangles;
    int numTimesteps;
    idx_t sourceIndex;
    val_t reflectivity;
    std::vector<val_t> distances;
    size_t nonZeroKij = 0;
    size_t receivedEnergy = 0;
    DeviceBuffer<val_t> weights, history;
    DeviceBuffer<uint8_t> delays;  // Zero marks a pair with no contribution.
    DeviceBuffer<Patch> patches;
    DeviceBuffer<VisibilityNode> nodes;
    DeviceBuffer<idx_t> references;
    idx_t nodeCount;
};

void initializeSimulation(SimulationState& state, int subdivisions, int timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.numTriangles = mesh.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.distances.resize(state.numTriangles);

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    printf("Building octree...\n");
    Octree octree;
    octree.build(mesh.triangles);

    std::vector<Patch> patches;
    patches.reserve(state.numTriangles);
    for (const Triangle& tri : mesh.triangles) {
        patches.push_back({tri.a, tri.b - tri.a, tri.c - tri.a, tri.normal(), tri.center(), tri.area()});
    }
    std::vector<VisibilityNode> nodes;
    std::vector<idx_t> references;
    flattenOctree(octree, nodes, references);
    state.nodeCount = static_cast<idx_t>(nodes.size());

    state.patches.allocate(patches.size());
    state.patches.upload(patches.data(), patches.size());
    state.nodes.allocate(nodes.size());
    state.nodes.upload(nodes.data(), nodes.size());
    state.references.allocate(references.size());
    state.references.upload(references.data(), references.size());
    state.history.allocate(state.numTriangles * timesteps);

    size_t pairCount = state.numTriangles * state.numTriangles;
    size_t available, total;
    CUDA_CHECK(cudaMemGetInfo(&available, &total));
    // Keep headroom for batched sampling and allow dense matrices larger
    // than the device's memory on devices that support oversubscription.
    bool managed = pairCount * (sizeof(val_t) + sizeof(uint8_t)) + (128u << 20) > available;
    state.weights.allocate(pairCount, managed);
    state.delays.allocate(pairCount, managed);
}

struct SamplingBatch {
    SamplingBatch(idx_t capacity, idx_t checkpointCapacity)
        : seeds(static_cast<size_t>(checkpointCapacity) * MT_WORDS), activePairs(1) {
        pairs.allocate(capacity);
    }

    DeviceBuffer<idx_t> pairs;
    PinnedBuffer<uint32_t> seeds;
    PinnedBuffer<idx_t> activePairs;
    CudaEvent pairsReady;
    PairBatch layout{};
};

void precompute(SimulationState& state, bool validate) {
    printf("Computing time delays (Tau)...\n");
    printf("Computing form factors (Kij)...\n");
    idx_t n = static_cast<idx_t>(state.numTriangles);
    size_t totalPairs = state.numTriangles * state.numTriangles;
    idx_t capacity = static_cast<idx_t>(std::min<size_t>(PAIR_BATCH_SIZE, totalPairs));
    // Allow a partially used checkpoint at the beginning and end of a batch.
    idx_t checkpointCapacity = (capacity + 2 * (PAIRS_PER_CHECKPOINT - 1)) / PAIRS_PER_CHECKPOINT;
    SamplingBatch batches[] = {{capacity, checkpointCapacity}, {capacity, checkpointCapacity}};

    DeviceBuffer<uint8_t> flags, selectScratch;
    DeviceBuffer<idx_t> selectedCount;
    DeviceBuffer<uint32_t> checkpoints;
    DeviceBuffer<float4> samples;
    DeviceBuffer<unsigned long long> nonzero;
    flags.allocate(capacity);
    selectedCount.allocate(1);
    checkpoints.allocate(static_cast<size_t>(checkpointCapacity) * MT_WORDS);
    samples.allocate(static_cast<size_t>(checkpointCapacity) * PAIRS_PER_CHECKPOINT * NUM_RAYS);
    if (validate) {
        nonzero.allocate(1);
        CUDA_CHECK(cudaMemset(nonzero.data, 0, sizeof(unsigned long long)));
    }

    size_t scratchBytes = 0;
    cub::CountingInputIterator<idx_t> indices(0);
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, scratchBytes, indices, flags.data,
                                         batches[0].pairs.data, selectedCount.data, capacity));
    selectScratch.allocate(scratchBytes);

    // Stable compaction preserves the row-major assignment of random draws.
    auto selectPairs = [&](size_t first, SamplingBatch& batch) {
        idx_t count = static_cast<idx_t>(std::min<size_t>(capacity, totalPairs - first));
        batch.layout = {n, static_cast<idx_t>(first / n), static_cast<idx_t>(first % n)};
        classifyPairs<<<(count + 255) / 256, 256>>>(state.patches.data, batch.layout, count, flags.data,
                                                   state.delays.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceSelect::Flagged(selectScratch.data, scratchBytes, indices, flags.data,
                                             batch.pairs.data, selectedCount.data, count));
        CUDA_CHECK(cudaMemcpyAsync(batch.activePairs.data, selectedCount.data, sizeof(idx_t),
                                   cudaMemcpyDeviceToHost));
        batch.pairsReady.record();
    };

    size_t nextProgress = std::min<size_t>(100, n);
    auto reportProgress = [&](size_t completedPairs) {
        while (nextProgress <= n && nextProgress * n <= completedPairs) {
            printf("  Progress: %zu/%u triangles\n", nextProgress, n);
            nextProgress = nextProgress == n ? nextProgress + 1 : std::min<size_t>(nextProgress + 100, n);
        }
    };

    RandomCheckpoints random;
    size_t sampledPairs = 0;
    selectPairs(0, batches[0]);
    // Selection for the next batch is queued ahead of the current ray
    // kernel. As that kernel runs, the host prepares the next checkpoints.
    // The selection event also guarantees that the prior use of a slot has ended.
    for (size_t first = 0, batchNumber = 0; first < totalPairs; first += capacity, ++batchNumber) {
        SamplingBatch& batch = batches[batchNumber % 2];
        batch.pairsReady.wait();
        if (first >= capacity) reportProgress(first - capacity);
        if (first + capacity < totalPairs) selectPairs(first + capacity, batches[(batchNumber + 1) % 2]);

        idx_t activePairs = *batch.activePairs.data;
        if (activePairs == 0) continue;
        idx_t offset = sampledPairs % PAIRS_PER_CHECKPOINT;
        idx_t countCheckpoints = (offset + activePairs + PAIRS_PER_CHECKPOINT - 1) / PAIRS_PER_CHECKPOINT;
        random.collect(sampledPairs / PAIRS_PER_CHECKPOINT, countCheckpoints, batch.seeds.data);
        CUDA_CHECK(cudaMemcpyAsync(checkpoints.data, batch.seeds.data,
                                   static_cast<size_t>(countCheckpoints) * MT_WORDS * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
        generateSamples<<<countCheckpoints, 256>>>(checkpoints.data, samples.data);
        CUDA_CHECK(cudaGetLastError());
        formFactors<<<(activePairs + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK, FORM_FACTOR_THREADS>>>(
            state.patches.data, batch.layout, state.nodes.data, state.nodeCount, state.references.data,
            batch.pairs.data, activePairs, samples.data + offset * NUM_RAYS,
            state.weights.data, state.delays.data, nonzero.data);
        CUDA_CHECK(cudaGetLastError());
        sampledPairs += activePairs;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    reportProgress(totalPairs);
    if (validate) {
        unsigned long long result;
        nonzero.download(&result, 1);
        state.nonZeroKij = result;
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    for (int t = 0; t < state.numTimesteps; ++t) {
        propagate<<<state.numTriangles, PROPAGATION_THREADS>>>(state.weights.data, state.delays.data,
            state.history.data, state.numTriangles, t, state.numTimesteps / 2, state.sourceIndex, state.reflectivity);
        CUDA_CHECK(cudaGetLastError());
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %d/%d\n", t + 1, state.numTimesteps);
        }
    }
}

void computeDistances(SimulationState& state, bool validate) {
    printf("Computing distances via cross-correlation...\n");
    DeviceBuffer<val_t> distances;
    DeviceBuffer<uint8_t> received;
    distances.allocate(state.numTriangles);
    if (validate) received.allocate(state.numTriangles);

    correlate<<<state.numTriangles, DISTANCE_THREADS>>>(state.history.data, state.numTriangles,
        state.numTimesteps, state.sourceIndex, distances.data, received.data);
    CUDA_CHECK(cudaGetLastError());
    distances.download(state.distances.data(), state.numTriangles);
    if (validate) {
        std::vector<uint8_t> energy(state.numTriangles);
        received.download(energy.data(), energy.size());
        state.receivedEnergy = std::count(energy.begin(), energy.end(), 1);
    }
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

    printf("  Triangles receiving energy: %zu/%zu\n", state.receivedEnergy, state.numTriangles);
    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    size_t pairCount = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n", state.nonZeroKij, pairCount,
           100.0f * state.nonZeroKij / static_cast<val_t>(pairCount));
    if (state.nonZeroKij == 0) {
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

    if (timesteps < 0) throw std::invalid_argument("Number of timesteps must be non-negative");
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
    initializeSimulation(state, subdivisions, timesteps,
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    precompute(state, validate);

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

    computeDistances(state, validate);

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
    fprintf(stderr, "RoomSim error: %s\n", error.what());
    return 1;
}

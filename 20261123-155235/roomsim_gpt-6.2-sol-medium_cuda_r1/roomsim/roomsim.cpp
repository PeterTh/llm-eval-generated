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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

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
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
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

// Nodes are stored in depth-first order. A missed box skips its entire subtree.
struct OctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    idx_t escape;
    idx_t leaf;
};

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

public:
    void flatten(std::vector<OctreeNode>& nodes, std::vector<idx_t>& leaves) const {
        const size_t node = nodes.size();
        nodes.push_back({center, halfExtent, 0, 0});
        if (!triangleIndices.empty()) {
            nodes[node].leaf = static_cast<idx_t>(leaves.size());
            leaves.push_back(static_cast<idx_t>(triangleIndices.size()));
            leaves.insert(leaves.end(), triangleIndices.begin(), triangleIndices.end());
        }
        for (const auto& child : children) {
            if (child) child->flatten(nodes, leaves);
        }
        nodes[node].escape = static_cast<idx_t>(nodes.size());
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
// CUDA storage and MT19937 stream
// ============================================================================

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T>
class DeviceBuffer {
    T* data_ = nullptr;

public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }

    T* data() const { return data_; }

    void allocate(size_t count, bool managed = false) {
        if (!count) return;
        if (managed) checkCuda(cudaMallocManaged(&data_, count * sizeof(T)));
        else checkCuda(cudaMalloc(&data_, count * sizeof(T)));
    }

    void upload(const std::vector<T>& values) {
        allocate(values.size());
        checkCuda(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

constexpr int MT_STATE = 624;
constexpr int MT_MIDDLE = 397;
constexpr int MT_PARALLEL_WORDS = MT_STATE - MT_MIDDLE;
constexpr int PAIRS_PER_CHECKPOINT = 64;
constexpr int WORDS_PER_PAIR = NUM_RAYS * 4;
constexpr int CHECKPOINT_WORDS = PAIRS_PER_CHECKPOINT * WORDS_PER_PAIR;
constexpr int CUDA_THREADS = 256;

__host__ __device__ uint32_t twist(uint32_t first, uint32_t second, uint32_t middle) {
    const uint32_t bits = (first & 0x80000000u) | (second & 0x7fffffffu);
    return middle ^ (bits >> 1) ^ ((0u - (bits & 1u)) & 0x9908b0dfu);
}

// Save the untempered stream at fixed intervals. Each GPU block can then
// expand its own draws concurrently, with no per-ray transfers or RNG atomics.
class RandomCheckpoints {
    std::array<uint32_t, MT_STATE + CHECKPOINT_WORDS> words;

public:
    RandomCheckpoints() {
        words[0] = 42;
        for (int i = 1; i < MT_STATE; ++i) {
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        }
    }

    void generate(uint32_t* output, int pairs) {
        while (pairs > 0) {
            std::copy_n(words.data(), MT_STATE, output);
            const int draws = std::min(pairs, PAIRS_PER_CHECKPOINT) * WORDS_PER_PAIR;
            for (int offset = 0; offset < draws; offset += MT_PARALLEL_WORDS) {
                const int end = std::min(offset + MT_PARALLEL_WORDS, draws);
                for (int i = offset; i < end; ++i) {
                    words[i + MT_STATE] = twist(words[i], words[i + 1], words[i + MT_MIDDLE]);
                }
            }
            std::copy_n(words.data() + draws, MT_STATE, words.data());
            pairs -= PAIRS_PER_CHECKPOINT;
            output += MT_STATE;
        }
    }
};

__device__ val_t uniformFloat(uint32_t bits) {
    bits ^= bits >> 11;
    bits ^= (bits << 7) & 0x9d2c5680u;
    bits ^= (bits << 15) & 0xefc60000u;
    bits ^= bits >> 18;
    // Match uniform_real_distribution<float> with one 32-bit MT19937 draw.
    return fminf(__uint2float_rn(bits) * 0x1p-32f, 0x1.fffffep-1f);
}

// ============================================================================
// CUDA form factors and propagation delays
// ============================================================================

struct Surface {
    Vec3 origin, edge1, edge2;
    Vec3 normal, center;
    val_t area;
};

struct ActivePair {
    const Surface* surfaces;
    size_t firstPair;
    idx_t n;

    __device__ bool operator()(idx_t localPair) const {
        const size_t pair = firstPair + localPair;
        const idx_t i = pair / n;
        const idx_t j = pair % n;
        return i != j && !(surfaces[i].normal.dot(surfaces[j].normal) > 0.99f);
    }
};

__device__ Vec3 samplePoint(const Surface& surface, const uint32_t* draws) {
    val_t u = uniformFloat(draws[0]);
    val_t v = uniformFloat(draws[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return surface.origin + surface.edge1 * u + surface.edge2 * v;
}

__device__ bool intersectsTriangle(const Vec3& origin, const Vec3& direction,
                                   val_t maxDistance, const Surface& tri) {
    const Vec3 pvec = direction.cross(tri.edge2);
    const val_t det = tri.edge1.dot(pvec);
    if (fabsf(det) < EPSILON) return false;

    const val_t invDet = ONE / det;
    const Vec3 tvec = origin - tri.origin;
    const val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return false;

    const Vec3 qvec = tvec.cross(tri.edge1);
    const val_t v = direction.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    const val_t distance = tri.edge2.dot(qvec) * invDet;
    return distance > EPSILON && distance < maxDistance;
}

__device__ bool intersectsBox(const OctreeNode& node, const Vec3& midpoint,
                              const Vec3& halfRay, const Vec3& absRay) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + absRay.x || fabsf(c.y) > h.y + absRay.y ||
        fabsf(c.z) > h.z + absRay.z) return false;

    const Vec3 cross = halfRay.cross(c);
    return !(fabsf(cross.x) > h.y * absRay.z + h.z * absRay.y + EPSILON ||
             fabsf(cross.y) > h.z * absRay.x + h.x * absRay.z + EPSILON ||
             fabsf(cross.z) > h.x * absRay.y + h.y * absRay.x + EPSILON);
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& ray, val_t length,
                             const Surface* surfaces, const OctreeNode* nodes,
                             const idx_t* leaves, idx_t i, idx_t j) {
    const Vec3 direction = ray / length;
    const Vec3 halfRay = ray * 0.5f;
    const Vec3 midpoint = from + halfRay;
    const Vec3 absRay = {fabsf(halfRay.x), fabsf(halfRay.y), fabsf(halfRay.z)};
    const val_t maxDistance = length - EPSILON;
    const idx_t end = nodes[0].escape;
    // The original traversal visits the root without a box test.
    for (idx_t k = nodes[0].leaf ? 0 : 1; k < end;) {
        const OctreeNode node = nodes[k];
        if (k != 0 && !intersectsBox(node, midpoint, halfRay, absRay)) {
            k = node.escape;
            continue;
        }
        if (node.leaf) {
            const idx_t* tri = leaves + node.leaf;
            const idx_t count = *tri++;
            for (idx_t l = 0; l < count; ++l) {
                const idx_t index = tri[l];
                if (index != i && index != j &&
                    intersectsTriangle(from, direction, maxDistance, surfaces[index])) return true;
            }
        }
        ++k;
    }
    return false;
}

__device__ val_t rayFormFactor(const Surface* surfaces, const OctreeNode* nodes,
                               const idx_t* leaves, idx_t i, idx_t j, const uint32_t* draws) {
    const Surface& a = surfaces[i];
    const Surface& b = surfaces[j];
    const Vec3 from = samplePoint(a, draws);
    const Vec3 ray = samplePoint(b, draws + 2) - from;
    const val_t distSqr = ray.squaredNorm();
    if (distSqr < EPSILON) return ZERO;

    const val_t length = sqrtf(distSqr);
    const val_t cosI = ray.dot(a.normal) / length;
    const val_t cosJ = (-ray).dot(b.normal) / length;
    if (cosI <= ZERO || cosJ <= ZERO ||
        isRayBlocked(from, ray, length, surfaces, nodes, leaves, i, j)) return ZERO;

    return (cosI * cosJ) / (PI * distSqr);
}

__global__ void computePairs(const Surface* __restrict__ surfaces,
                             const OctreeNode* __restrict__ nodes, const idx_t* __restrict__ leaves,
                             const idx_t* __restrict__ active, int numActive, size_t firstPair,
                             idx_t n, const uint32_t* __restrict__ checkpoints,
                             val_t* __restrict__ weights, uint8_t* __restrict__ delays,
                             unsigned long long* nonzeroFactors) {
    __shared__ uint32_t words[MT_STATE + CHECKPOINT_WORDS];
    __shared__ val_t factors[CUDA_THREADS];
    using Reduction = cub::BlockReduce<int, CUDA_THREADS>;
    __shared__ Reduction::TempStorage reduction;

    const int thread = threadIdx.x;
    const int first = blockIdx.x * PAIRS_PER_CHECKPOINT;
    const int pairs = min(PAIRS_PER_CHECKPOINT, numActive - first);
    const uint32_t* checkpoint = checkpoints + blockIdx.x * MT_STATE;
    for (int k = thread; k < MT_STATE; k += CUDA_THREADS) words[k] = checkpoint[k];
    __syncthreads();

    const int draws = pairs * WORDS_PER_PAIR;
    for (int offset = 0; offset < draws; offset += MT_PARALLEL_WORDS) {
        if (thread < MT_PARALLEL_WORDS && offset + thread < draws) {
            const int k = offset + thread;
            words[k + MT_STATE] = twist(words[k], words[k + 1], words[k + MT_MIDDLE]);
        }
        __syncthreads();
    }

    constexpr int pairsPerWave = CUDA_THREADS / NUM_RAYS;
    const int ray = thread % NUM_RAYS;
    const int group = thread / NUM_RAYS;
    const unsigned mask = 0xffffu << (group % 2 * NUM_RAYS);
    int nonzero = 0;
    for (int base = 0; base < pairs; base += pairsPerWave) {
        const int p = base + group;
        if (p < pairs) {
            const size_t pair = firstPair + active[first + p];
            const idx_t i = pair / n;
            const idx_t j = pair % n;
            factors[thread] = rayFormFactor(surfaces, nodes, leaves, i, j,
                                          words + MT_STATE + p * WORDS_PER_PAIR + ray * 4);
            __syncwarp(mask);
            if (ray == 0) {
                val_t kij = ZERO;
                // Preserve the serial sample order.
                #pragma unroll
                for (int r = 0; r < NUM_RAYS; ++r) kij += factors[thread + r];
                kij *= INV_NUM_RAYS;
                if (kij > ZERO) {
                    const val_t distance = (surfaces[i].center - surfaces[j].center).norm();
                    // The radius-10 mesh has at most 40 steps of flight time.
                    delays[pair] = static_cast<uint8_t>(ceilf(distance * INV_WAVE_SPEED));
                    weights[pair] = fminf(kij * surfaces[j].area, ONE);
                }
                nonzero += kij > EPSILON;
            }
            __syncwarp(mask);
        }
    }

    if (nonzeroFactors) {
        const int count = Reduction(reduction).Sum(nonzero);
        if (thread == 0) atomicAdd(nonzeroFactors, static_cast<unsigned long long>(count));
    }
}

// ============================================================================
// CUDA propagation and cross-correlation
// ============================================================================

__global__ void propagate(const val_t* __restrict__ weights, const uint8_t* __restrict__ delays,
                          val_t* __restrict__ radiosity, idx_t n, int t, int emissionEnd,
                          idx_t source, val_t reflectivity) {
    constexpr int warp = 32;
    __shared__ val_t weighted[CUDA_THREADS];
    __shared__ val_t incident[CUDA_THREADS];
    const int lane = threadIdx.x % warp;
    const idx_t i = blockIdx.x * (CUDA_THREADS / warp) + threadIdx.x / warp;
    if (i >= n) return;

    const size_t row = static_cast<size_t>(i) * n;
    val_t sum = ZERO;
    for (idx_t base = 0; t > 0 && base < n; base += warp) {
        const idx_t j = base + lane;
        const int delay = j < n ? delays[row + j] : 0;
        val_t prior = ZERO;
        if (delay && delay <= t) prior = radiosity[static_cast<size_t>(t - delay) * n + j];
        const unsigned active = __ballot_sync(0xffffffffu, !(prior <= ZERO));
        if (active) {
            if (!(prior <= ZERO)) {
                weighted[threadIdx.x] = weights[row + j];
                incident[threadIdx.x] = prior;
            }
            __syncwarp();
            if (lane == 0) {
                // Accumulate in emitter order, as in the serial simulation.
                const int first = threadIdx.x;
                for (unsigned remaining = active; remaining; remaining &= remaining - 1) {
                    const int index = first + __ffs(remaining) - 1;
                    sum = fmaf(weighted[index], incident[index], sum);
                }
            }
            __syncwarp();
        }
    }
    if (lane == 0) {
        const val_t emission = i == source && t < emissionEnd ? ONE : ZERO;
        radiosity[static_cast<size_t>(t) * n + i] = fmaf(reflectivity, sum, emission);
    }
}

struct Correlation {
    val_t value;
    int lag;
};

struct EarlierPeak {
    __device__ Correlation operator()(Correlation a, Correlation b) const {
        return b.value > a.value || (b.value == a.value && b.lag < a.lag) ? b : a;
    }
};

template<int Threads, bool CacheHistory>
__global__ void correlate(const val_t* __restrict__ radiosity, val_t* __restrict__ distances,
                          idx_t n, int timesteps, idx_t source, unsigned* received) {
    using Reduction = cub::BlockReduce<Correlation, Threads>;
    __shared__ typename Reduction::TempStorage reduction;
    extern __shared__ val_t history[];
    val_t* sourceHistory = history + timesteps;
    const idx_t triangle = blockIdx.x;

    if constexpr (CacheHistory) {
        for (int t = threadIdx.x; t < timesteps; t += Threads) {
            const size_t row = static_cast<size_t>(t) * n;
            history[t] = radiosity[row + triangle];
            sourceHistory[t] = radiosity[row + source];
        }
        __syncthreads();
    }

    Correlation best{ZERO, 0};
    bool energy = false;
    for (int lag = threadIdx.x; lag < timesteps; lag += Threads) {
        val_t sum = ZERO;
        for (int t = lag; t < timesteps; ++t) {
            const val_t other = CacheHistory ? history[t] : radiosity[static_cast<size_t>(t) * n + triangle];
            const val_t emitted = CacheHistory ? sourceHistory[t - lag]
                                                : radiosity[static_cast<size_t>(t - lag) * n + source];
            sum = fmaf(emitted, other, sum);
        }
        best = EarlierPeak{}(best, {sum, lag});
        if (received) {
            const val_t value = CacheHistory ? history[lag] : radiosity[static_cast<size_t>(lag) * n + triangle];
            energy |= value > EPSILON;
        }
    }

    if (received && __syncthreads_or(energy) && threadIdx.x == 0) atomicAdd(received, 1u);
    const Correlation peak = Reduction(reduction).Reduce(best, EarlierPeak{});
    if (threadIdx.x == 0) distances[triangle] = WAVE_SPEED * static_cast<val_t>(peak.lag);
}

// ============================================================================
// Host setup and CUDA dispatch
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;
    val_t reflectivity;
    std::vector<val_t> distances;
    DeviceBuffer<Surface> surfaces;
    DeviceBuffer<OctreeNode> nodes;
    DeviceBuffer<idx_t> leaves;
    DeviceBuffer<val_t> weights, radiosity, deviceDistances;
    DeviceBuffer<uint8_t> delays;
    DeviceBuffer<unsigned long long> nonzeroFactors;
    DeviceBuffer<unsigned> received;
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool validate) {
    const IcosphereMesh mesh(subdivisions, 10.0f);
    state.numTriangles = mesh.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.distances.resize(state.numTriangles);

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    printf("Building octree...\n");
    Octree octree;
    octree.build(mesh.triangles);

    std::vector<OctreeNode> nodes;
    std::vector<idx_t> leaves{0};  // Offset zero denotes an internal node.
    octree.flatten(nodes, leaves);
    state.nodes.upload(nodes);
    state.leaves.upload(leaves);

    std::vector<Surface> surfaces;
    surfaces.reserve(state.numTriangles);
    for (const Triangle& triangle : mesh.triangles) {
        surfaces.push_back({triangle.a, triangle.b - triangle.a, triangle.c - triangle.a,
                            triangle.normal(), triangle.center(), triangle.area()});
    }
    state.surfaces.upload(surfaces);
    state.radiosity.allocate(timesteps * state.numTriangles);
    state.deviceDistances.allocate(state.numTriangles);
    if (validate) {
        state.nonzeroFactors.allocate(1);
        state.received.allocate(1);
        checkCuda(cudaMemset(state.nonzeroFactors.data(), 0, sizeof(unsigned long long)));
        checkCuda(cudaMemset(state.received.data(), 0, sizeof(unsigned)));
    }
}

class PairBatch {
    DeviceBuffer<idx_t> active;
    DeviceBuffer<int> activeCount;
    DeviceBuffer<uint32_t> checkpoints;
    DeviceBuffer<unsigned char> temporary;
    size_t temporaryBytes = 0;
    uint32_t* hostData = nullptr;
    cudaStream_t stream = nullptr;

public:
    ~PairBatch() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        if (hostData) cudaFreeHost(hostData);
    }

    void allocate(int capacity, ActivePair predicate) {
        const int blocks = (capacity + PAIRS_PER_CHECKPOINT - 1) / PAIRS_PER_CHECKPOINT;
        const size_t words = static_cast<size_t>(blocks) * MT_STATE;
        active.allocate(capacity);
        activeCount.allocate(1);
        checkpoints.allocate(words);
        checkCuda(cub::DeviceSelect::If(nullptr, temporaryBytes, cub::CountingInputIterator<idx_t>(0),
                                      active.data(), activeCount.data(), capacity, predicate));
        temporary.allocate(temporaryBytes);
        checkCuda(cudaMallocHost(&hostData, (words + 1) * sizeof(uint32_t)));
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    void launch(const SimulationState& state, size_t first, int count, RandomCheckpoints& random) {
        const idx_t n = static_cast<idx_t>(state.numTriangles);
        // Stable compaction gives each surviving pair its original position in the RNG stream.
        checkCuda(cub::DeviceSelect::If(temporary.data(), temporaryBytes,
                                      cub::CountingInputIterator<idx_t>(0), active.data(),
                                      activeCount.data(), count,
                                      ActivePair{state.surfaces.data(), first, n}, stream));
        checkCuda(cudaMemcpyAsync(hostData, activeCount.data(), sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        const int selected = static_cast<int>(hostData[0]);
        if (!selected) return;

        uint32_t* hostCheckpoints = hostData + 1;
        random.generate(hostCheckpoints, selected);
        const int blocks = (selected + PAIRS_PER_CHECKPOINT - 1) / PAIRS_PER_CHECKPOINT;
        checkCuda(cudaMemcpyAsync(checkpoints.data(), hostCheckpoints,
                                  static_cast<size_t>(blocks) * MT_STATE * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, stream));
        computePairs<<<blocks, CUDA_THREADS, 0, stream>>>(
            state.surfaces.data(), state.nodes.data(), state.leaves.data(), active.data(), selected,
            first, n, checkpoints.data(), state.weights.data(), state.delays.data(),
            state.nonzeroFactors.data());
        checkCuda(cudaGetLastError());
    }
};

void precompute(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    printf("Computing form factors (Kij)...\n");

    constexpr int maxBatch = 1 << 19;
    const size_t n = state.numTriangles;
    const size_t pairs = n * n;
    const int capacity = static_cast<int>(std::min(pairs, static_cast<size_t>(maxBatch)));
    const ActivePair predicate{state.surfaces.data(), 0, static_cast<idx_t>(n)};
    std::array<PairBatch, 2> batches;
    batches[0].allocate(capacity, predicate);
    if (pairs > static_cast<size_t>(capacity)) batches[1].allocate(capacity, predicate);

    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    // Managed storage lets meshes larger than device memory use the same CUDA kernels.
    const bool oversubscribed = pairs > freeBytes / (sizeof(val_t) + sizeof(uint8_t));
    state.weights.allocate(pairs, oversubscribed);
    state.delays.allocate(pairs, oversubscribed);
    checkCuda(cudaMemset(state.delays.data(), 0, pairs * sizeof(uint8_t)));
    checkCuda(cudaDeviceSynchronize());

    // Generate the next batch's checkpoints while the previous batch traces rays.
    RandomCheckpoints random;
    int slot = 0;
    for (size_t first = 0; first < pairs; first += capacity) {
        const int count = static_cast<int>(std::min(pairs - first, static_cast<size_t>(capacity)));
        batches[slot].launch(state, first, count, random);
        slot ^= 1;
    }
    checkCuda(cudaDeviceSynchronize());
    printf("  Progress: %zu/%zu triangles\n", n, n);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const idx_t n = static_cast<idx_t>(state.numTriangles);
    const int blocks = (n + CUDA_THREADS / 32 - 1) / (CUDA_THREADS / 32);
    const int timesteps = static_cast<int>(state.numTimesteps);
    for (int t = 0; t < timesteps; ++t) {
        propagate<<<blocks, CUDA_THREADS>>>(state.weights.data(), state.delays.data(),
                                           state.radiosity.data(), n, t, timesteps / 2,
                                           static_cast<idx_t>(state.sourceIndex), state.reflectivity);
        checkCuda(cudaGetLastError());
        if ((t + 1) % 10 == 0 || t + 1 == timesteps) {
            printf("  Timestep %d/%d\n", t + 1, timesteps);
        }
    }
    checkCuda(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const idx_t n = static_cast<idx_t>(state.numTriangles);
    const idx_t source = static_cast<idx_t>(state.sourceIndex);
    const int timesteps = static_cast<int>(state.numTimesteps);
    const size_t historyBytes = 2 * static_cast<size_t>(timesteps) * sizeof(val_t);
    const val_t* radiosity = state.radiosity.data();
    val_t* distances = state.deviceDistances.data();
    unsigned* received = state.received.data();
    if (timesteps <= 64) {
        correlate<64, true><<<n, 64, historyBytes>>>(radiosity, distances, n, timesteps, source, received);
    } else if (timesteps <= 2048) {
        correlate<CUDA_THREADS, true><<<n, CUDA_THREADS, historyBytes>>>(
            radiosity, distances, n, timesteps, source, received);
    } else {
        correlate<CUDA_THREADS, false><<<n, CUDA_THREADS>>>(radiosity, distances, n, timesteps, source, received);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(state.distances.data(), state.deviceDistances.data(),
                         state.numTriangles * sizeof(val_t), cudaMemcpyDeviceToHost));
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
    unsigned receivedEnergy = 0;
    checkCuda(cudaMemcpy(&receivedEnergy, state.received.data(), sizeof(receivedEnergy),
                         cudaMemcpyDeviceToHost));
    printf("  Triangles receiving energy: %u/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = 0;
    checkCuda(cudaMemcpy(&nonZeroKij, state.nonzeroFactors.data(), sizeof(nonZeroKij),
                         cudaMemcpyDeviceToHost));
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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

    if (timesteps < 0) {
        fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
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
                         static_cast<size_t>(sourceIdx), reflectivity, validate);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    precompute(state);

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
    size_t memWeights = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(uint8_t);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memWeights + memTau + memRad;
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

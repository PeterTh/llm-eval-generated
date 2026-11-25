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
#include <math_constants.h>

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
// Random Number Generation
// ============================================================================

// Start independent CUDA blocks at checkpoints in the original MT19937
// stream. The host discards intervening draws in bulk. Only the first 624
// outputs of each chunk cross PCIe, regardless of how many pairs are culled.
constexpr int MT_WORDS = std::mt19937::state_size;
constexpr int MT_SHIFT = std::mt19937::shift_size;
constexpr int MT_GROUP = MT_WORDS - MT_SHIFT;
constexpr int RANDOM_CHUNK = 8192;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;

__device__ uint32_t untemper(uint32_t x) {
    x ^= x >> 18;
    x ^= (x << 15) & 0xefc60000u;
    uint32_t y = x;
    for (int k = 0; k < 4; ++k) y = x ^ ((y << 7) & 0x9d2c5680u);
    return y ^ (y >> 11) ^ (y >> 22);
}

__global__ void expandSamples(const uint32_t* checkpoints, int count, float* samples) {
    // x[k] depends on x[k-624], x[k-623], and x[k-227]. Keep just
    // the last 624 words, reading a full group before recycling slots.
    __shared__ uint32_t words[MT_WORDS];
    int start = blockIdx.x * RANDOM_CHUNK;
    int length = min(RANDOM_CHUNK, count - start);
    for (int k = threadIdx.x; k < MT_WORDS; k += blockDim.x)
        words[k] = untemper(checkpoints[blockIdx.x * MT_WORDS + k]);
    __syncthreads();
    for (int base = 0; base < length; base += MT_GROUP) {
        int k = base + threadIdx.x;
        int slot = k % MT_WORDS;
        uint32_t next = 0;
        if (threadIdx.x < MT_GROUP) {
            uint32_t x = words[slot];
            uint32_t joined = (x & 0x80000000u) | (words[(slot + 1) % MT_WORDS] & 0x7fffffffu);
            next = words[(slot + MT_SHIFT) % MT_WORDS] ^ (joined >> 1) ^
                   ((joined & 1) ? 0x9908b0dfu : 0u);
            // Keep the next state in a register until all warps finish reading.
            x ^= x >> 11;
            x ^= (x << 7) & 0x9d2c5680u;
            x ^= (x << 15) & 0xefc60000u;
            x ^= x >> 18;
            if (k < length)
                samples[start + k] = fminf(static_cast<float>(x) * 0x1p-32f, 0x1.fffffep-1f);
        }
        __syncthreads();
        if (threadIdx.x < MT_GROUP) words[slot] = next;
        __syncthreads();
    }
}

// Fold a uniform square sample into barycentric coordinates.
__device__ Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
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

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return CUDART_INF_F;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return CUDART_INF_F;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return CUDART_INF_F;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Preorder layout allows a ray to skip an entire subtree with one index.
struct Node {
    Vec3 center, halfExtent;
    int end, first, count;
};

void flatten(const Octree& tree, std::vector<Node>& nodes, std::vector<int>& indices) {
    size_t root = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flatten(*child, nodes, indices);
    }
    nodes[root].end = static_cast<int>(nodes.size());
}

__device__ bool intersectsBox(Vec3 c, Vec3 d, Vec3 ad, Vec3 halfExtent) {
    if (std::abs(c.x) > halfExtent.x + ad.x) return false;
    if (std::abs(c.y) > halfExtent.y + ad.y) return false;
    if (std::abs(c.z) > halfExtent.z + ad.z) return false;
    if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ val_t sampleRay(const Triangle* triangles, const Node* nodes, const int* indices,
                           int i, int j, float4 sample) {
    const Triangle& receiver = triangles[i];
    const Triangle& emitter = triangles[j];
    Vec3 from = randomPointInTriangle(receiver, sample.x, sample.y);
    Vec3 to = randomPointInTriangle(emitter, sample.z, sample.w);
    Vec3 ray = to - from;
    float squared = ray.squaredNorm();
    if (squared < EPSILON) return ZERO;
    float length = sqrtf(squared);
    float ci = ray.dot(receiver._normal) / length;
    float cj = (-ray).dot(emitter._normal) / length;
    if (ci <= ZERO || cj <= ZERO) return ZERO;

    Vec3 direction = ray / length;
    Vec3 half = ray * 0.5f;
    Vec3 middle = from + half;
    Vec3 extent(fabsf(half.x), fabsf(half.y), fabsf(half.z));
    for (int k = 0; k < nodes[0].end;) {
        const Node& node = nodes[k];
        if (k && !intersectsBox(middle - node.center, half, extent, node.halfExtent)) {
            k = node.end;
            continue;
        }
        for (int m = 0; m < node.count; ++m) {
            int other = indices[node.first + m];
            if (other == i || other == j) continue;
            const Triangle& tri = triangles[other];
            float dist = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < length - EPSILON) return ZERO;
        }
        ++k;
    }
    return (ci * cj) / (PI * squared);
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        if (!count) return;
        size_t bytes = count * sizeof(T);
        cudaError_t status = cudaMalloc(&data, bytes);
        // Large meshes can exceed local VRAM. On devices supporting demand
        // paging, let CUDA migrate those allocations while kernels run.
        if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            int device, canPage;
            checkCuda(cudaGetDevice(&device));
            checkCuda(cudaDeviceGetAttribute(&canPage, cudaDevAttrConcurrentManagedAccess, device));
            if (canPage) status = cudaMallocManaged(&data, bytes);
        }
        checkCuda(status);
    }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) checkCuda(cudaMemcpy(data, values.data(), values.size() * sizeof(T),
                                                  cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(data); }
};

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    float reflectivity;
    std::vector<Triangle> triangles;
    std::vector<val_t> distances, radB;
    DeviceBuffer<Triangle> geometry;
    DeviceBuffer<Node> nodes;
    DeviceBuffer<int> indices;
    DeviceBuffer<Vec3> centers;
    DeviceBuffer<float> areas, weights, history, distance;
    DeviceBuffer<unsigned char> delays;
    DeviceBuffer<unsigned long long> nonZero;
    size_t idxTN(size_t t, size_t i) const { return t * numTriangles + i; }
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    state.triangles = std::move(IcosphereMesh(subdivisions, 10.0f).triangles);
    size_t n = state.triangles.size();
    state.numTriangles = n;
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % n;
    state.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", n);
    Octree tree;
    tree.build(state.triangles);
    std::vector<Node> nodes;
    std::vector<int> indices;
    flatten(tree, nodes, indices);
    state.nodes.upload(nodes);
    state.indices.upload(indices);
    state.geometry.upload(state.triangles);
    std::vector<float> areas;
    std::vector<Vec3> centers;
    for (const auto& tri : state.triangles) {
        areas.push_back(tri.area());
        centers.push_back(tri.center());
    }
    state.areas.upload(areas);
    state.centers.upload(centers);
    state.weights.allocate(n * n);
    state.delays.allocate(n * n);
    state.history.allocate(timesteps * n);
    state.distance.allocate(n);
    state.distances.resize(n);
    state.nonZero.allocate(1);
    checkCuda(cudaMemset(state.nonZero.data, 0, sizeof(unsigned long long)));
    checkCuda(cudaStreamSynchronize(nullptr));
}

// One half warp traces the 16 samples for a pair. Sum in ray order to
// retain the serial accumulation and store the area-weighted coefficient once.
__global__ void formFactors(const Triangle* geometry, const Node* nodes, const int* indices,
                            const Vec3* centers, const float* areas, const float4* samples,
                            const int* sampleIndices, int n, size_t first, int count, float* weights, unsigned char* delays,
                            unsigned long long* nonZero) {
    __shared__ float contributions[128];
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = thread / NUM_RAYS;
    int ray = thread % NUM_RAYS;
    unsigned mask = __ballot_sync(0xffffffffu, pair < count);
    if (pair >= count) return;
    size_t linear = first + pair;
    int i = linear / n;
    int j = linear % n;
    int sampleIndex = sampleIndices[pair];
    float value = ZERO;
    if (sampleIndex >= 0)
        value = sampleRay(geometry, nodes, indices, i, j, samples[sampleIndex * NUM_RAYS + ray]);
    contributions[threadIdx.x] = value;
    __syncwarp(mask);
    float kij = ZERO;
    if (!ray) {
        for (int r = 0; r < NUM_RAYS; ++r) kij += contributions[threadIdx.x + r];
        kij *= INV_NUM_RAYS;
        weights[linear] = fminf(kij * areas[j], ONE);
        // Room diameter is 20 and speed is 0.5, so all delays fit in a byte.
        delays[linear] = static_cast<unsigned char>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
    }
    unsigned votes = __ballot_sync(mask, kij > EPSILON);
    if (votes && (threadIdx.x % 32 == __ffs(votes) - 1)) atomicAdd(nonZero, __popc(votes));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing time delays and form factors...\n");
    constexpr int batch = 16384;
    constexpr int maxChunks = (batch * DRAWS_PER_PAIR + RANDOM_CHUNK - 1) / RANDOM_CHUNK;
    struct Slot {
        int* hostIndices;
        uint32_t* hostCheckpoints;
        DeviceBuffer<int> indices;
        DeviceBuffer<uint32_t> checkpoints;
        DeviceBuffer<float4> samples;
        cudaStream_t stream;
        Slot() {
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            checkCuda(cudaMallocHost(&hostIndices, batch * sizeof(int)));
            checkCuda(cudaMallocHost(&hostCheckpoints, maxChunks * MT_WORDS * sizeof(uint32_t)));
            indices.allocate(batch);
            checkpoints.allocate(maxChunks * MT_WORDS);
            samples.allocate(batch * NUM_RAYS);
        }
        ~Slot() {
            cudaFreeHost(hostCheckpoints);
            cudaFreeHost(hostIndices);
            cudaStreamDestroy(stream);
        }
    };
    Slot slots[2];
    std::mt19937 rng(42);
    size_t n = state.numTriangles;
    for (size_t first = 0; first < n * n; first += batch) {
        Slot& slot = slots[(first / batch) % 2];
        checkCuda(cudaStreamSynchronize(slot.stream));
        int count = static_cast<int>(std::min<size_t>(batch, n * n - first));
        int active = 0;
        for (int p = 0; p < count; ++p) {
            size_t i = (first + p) / n, j = (first + p) % n;
            bool culled = i == j || state.triangles[i]._normal.dot(state.triangles[j]._normal) > 0.99f;
            slot.hostIndices[p] = culled ? -1 : active++;
        }
        int draws = active * DRAWS_PER_PAIR;
        int chunks = (draws + RANDOM_CHUNK - 1) / RANDOM_CHUNK;
        for (int c = 0; c < chunks; ++c) {
            int length = std::min(RANDOM_CHUNK, draws - c * RANDOM_CHUNK);
            int seedCount = std::min(MT_WORDS, length);
            uint32_t* seed = slot.hostCheckpoints + c * MT_WORDS;
            for (int k = 0; k < seedCount; ++k) seed[k] = rng();
            std::fill(seed + seedCount, seed + MT_WORDS, 0u);
            rng.discard(length - seedCount);
        }
        checkCuda(cudaMemcpyAsync(slot.indices.data, slot.hostIndices, count * sizeof(int),
                                  cudaMemcpyHostToDevice, slot.stream));
        if (chunks) {
            checkCuda(cudaMemcpyAsync(slot.checkpoints.data, slot.hostCheckpoints,
                                      chunks * MT_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice, slot.stream));
            expandSamples<<<chunks, 256, 0, slot.stream>>>(slot.checkpoints.data, draws,
                reinterpret_cast<float*>(slot.samples.data));
            checkCuda(cudaGetLastError());
        }
        formFactors<<<(count * NUM_RAYS + 127) / 128, 128, 0, slot.stream>>>(
            state.geometry.data, state.nodes.data, state.indices.data, state.centers.data,
            state.areas.data, slot.samples.data, slot.indices.data, n, first, count, state.weights.data,
            state.delays.data, state.nonZero.data);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaDeviceSynchronize());
}

__global__ void propagate(int n, size_t t, int source, size_t timeOff, float rho,
                          const float* weights, const unsigned char* delays, float* history) {
    int i = blockIdx.x;
    const size_t row = static_cast<size_t>(i) * n;
    constexpr int tile = 512;
    __shared__ float2 terms[tile];
    float sum = ZERO;
    // Gather the delayed history in parallel, then accumulate in emitter order.
    // Each block streams one contiguous row of the coupling matrices.
    for (int base = 0; base < n; base += tile) {
        int length = min(tile, n - base);
        for (int k = threadIdx.x; k < length; k += blockDim.x) {
            int j = base + k;
            int delay = delays[row + j];
            float weight = weights[row + j];
            float emitted = ZERO;
            if (delay && delay <= t && weight > ZERO) {
                float signal = history[(t - delay) * n + j];
                if (signal > ZERO) emitted = signal;
            }
            terms[k] = make_float2(weight, emitted);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int k = 0; k < length; ++k) sum += terms[k].x * terms[k].y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        history[t * n + i] = rho * sum + (i == source && t < timeOff ? ONE : ZERO);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    int n = state.numTriangles;
    for (size_t t = 0; t < state.numTimesteps; ++t)
        propagate<<<n, 128>>>(n, t, state.sourceIndex, state.numTimesteps / 2,
            state.reflectivity, state.weights.data, state.delays.data, state.history.data);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

__global__ void correlate(int n, int steps, int source, const float* history, float* distances) {
    int i = blockIdx.x;
    float best = ZERO;
    int lag = 0;
    for (int t = threadIdx.x; t < steps; t += blockDim.x) {
        float sum = ZERO;
        for (int tt = t; tt < steps; ++tt)
            sum += history[static_cast<size_t>(tt - t) * n + source] *
                   history[static_cast<size_t>(tt) * n + i];
        if (sum > best) { best = sum; lag = t; }
    }
    __shared__ float scores[64];
    __shared__ int lags[64];
    scores[threadIdx.x] = best;
    lags[threadIdx.x] = lag;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < 64; ++k) {
            if (scores[k] > best || (scores[k] == best && lags[k] < lag)) {
                best = scores[k];
                lag = lags[k];
            }
        }
        distances[i] = WAVE_SPEED * lag;
    }
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    correlate<<<state.numTriangles, 64>>>(state.numTriangles, state.numTimesteps, state.sourceIndex,
                                          state.history.data, state.distance.data);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(state.distances.data(), state.distance.data,
                         state.numTriangles * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
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
    state.radB.resize(state.numTriangles * state.numTimesteps);
    if (!state.radB.empty())
        checkCuda(cudaMemcpy(state.radB.data(), state.history.data, state.radB.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
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
    unsigned long long nonZeroKij;
    checkCuda(cudaMemcpy(&nonZeroKij, state.nonZero.data, sizeof(nonZeroKij), cudaMemcpyDeviceToHost));
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
    size_t memTau = n * n * sizeof(unsigned char);
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
}

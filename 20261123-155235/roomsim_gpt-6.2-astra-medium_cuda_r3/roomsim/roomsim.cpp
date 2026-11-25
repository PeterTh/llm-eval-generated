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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <cuda_runtime.h>
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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
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

struct FlatNode {
    Vec3 center, halfExtent;
    unsigned end, first, count;
    // Check if a ray intersects this node's bounding box
    __device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
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

__device__ Vec3 randomPointInTriangle(const Triangle& t, float u, float v) {
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

    if (std::abs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const Vec3& dirNorm,
                             float rayLen, const FlatNode* nodes, const unsigned* indices,
                             const Triangle* triangles, unsigned src, unsigned dst) {
    // Nodes are in preorder. A missed box skips its whole subtree.
    for (unsigned k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.rayIntersectsBox(from, to)) {
            k = node.end;
            continue;
        }
        for (unsigned i = node.first; i < node.first + node.count; ++i) {
            unsigned idx = indices[i];
            if (idx == src || idx == dst) continue;
            const Triangle& tri = triangles[idx];
            float d = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < rayLen - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }
    void allocate(size_t size) {
        if (size == 0) return;
        cudaError_t status = cudaMalloc(&data, size * sizeof(T));
        // The largest mesh can exceed device memory. Let CUDA page the
        // allocation on devices that support managed-memory oversubscription.
        if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            status = cudaMallocManaged(&data, size * sizeof(T));
        }
        cudaCheck(status);
    }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        cudaCheck(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<unsigned>& indices) {
    size_t k = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<unsigned>(indices.size()),
                     static_cast<unsigned>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flatten(*child, nodes, indices);
    }
    nodes[k].end = static_cast<unsigned>(nodes.size());
}

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    float reflectivity;
    bool validate;
    unsigned long long nonZeroKij = 0;
    DeviceBuffer<unsigned long long> count;
    std::vector<Triangle> triangles;
    DeviceBuffer<Triangle> mesh;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<unsigned> indices;
    DeviceBuffer<float> areas, weights, radiosity, distance;
    DeviceBuffer<unsigned char> delays;
    std::vector<float> distances, radB;
    size_t idxTN(size_t t, size_t j) const { return t * numTriangles + j; }
};

void initializeSimulation(SimulationState& s, int subdivisions, size_t timesteps, size_t source,
                          float reflectivity) {
    s.triangles = std::move(IcosphereMesh(subdivisions, 10.0f).triangles);
    s.numTriangles = s.triangles.size();
    s.numTimesteps = timesteps;
    s.sourceIndex = source % s.numTriangles;
    s.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", s.numTriangles);
    Octree tree;
    tree.build(s.triangles);
    std::vector<FlatNode> nodes;
    std::vector<unsigned> indices;
    flatten(tree, nodes, indices);
    s.nodes.upload(nodes);
    s.indices.upload(indices);
    s.mesh.upload(s.triangles);
    std::vector<float> areas;
    for (const Triangle& tri : s.triangles) areas.push_back(tri.area());
    s.areas.upload(areas);
    s.weights.allocate(s.numTriangles * s.numTriangles);
    s.delays.allocate(s.numTriangles * s.numTriangles);
    s.radiosity.allocate(timesteps * s.numTriangles);
    s.distances.resize(s.numTriangles);
    s.distance.allocate(s.numTriangles);
    if (s.validate) {
        s.count.allocate(1);
        cudaCheck(cudaMemset(s.count.data, 0, sizeof(s.nonZeroKij)));
    }
}

// Expand the same MT19937 stream as std::mt19937(42). The recurrence can
// vectorize across up to 227 consecutive words. Tempering runs on the GPU.
struct RandomStream {
    uint32_t history[624];
    RandomStream() {
        history[0] = 42;
        for (unsigned i = 1; i < 624; ++i)
            history[i] = 1812433253u * (history[i - 1] ^ (history[i - 1] >> 30)) + i;
    }
    void fill(uint32_t* buffer, size_t words) {
        std::memcpy(buffer, history, sizeof(history));
        for (size_t i = 0; i < words; ++i) {
            uint32_t y = (buffer[i] & 0x80000000u) | (buffer[i + 1] & 0x7fffffffu);
            buffer[i + 624] = buffer[i + 397] ^ (y >> 1) ^ ((0u - (y & 1)) & 0x9908b0dfu);
        }
        std::memcpy(history, buffer + words, sizeof(history));
    }
};

__device__ float uniformSample(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return fminf(__uint2float_rn(y) * 0x1p-32f, 0x1.fffffep-1f);
}

constexpr unsigned batchPairs = 32768;
constexpr unsigned threads = 128;

__global__ void formFactors(size_t first, unsigned pairs, unsigned n, const Triangle* mesh,
                            const FlatNode* nodes, const unsigned* indices, const int* offsets,
                            const uint4* samples, const float* areas, float* weights,
                            unsigned char* delays, unsigned long long* nonZeroCount) {
    unsigned ray = threadIdx.x % NUM_RAYS;
    unsigned pair = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_RAYS;
    unsigned lanes = __ballot_sync(0xffffffffu, pair < pairs);
    if (pair >= pairs) return;
    size_t index = first + pair;
    unsigned i = index / n, j = index % n;
    Triangle a = mesh[i], b = mesh[j];
    float contribution = 0;
    int offset = offsets[pair];
    if (offset >= 0) {
        uint4 sample = samples[offset * NUM_RAYS + ray];
        Vec3 from = randomPointInTriangle(a, uniformSample(sample.x), uniformSample(sample.y));
        Vec3 to = randomPointInTriangle(b, uniformSample(sample.z), uniformSample(sample.w));
        Vec3 v = to - from;
        float d2 = v.squaredNorm(), d = sqrtf(d2);
        if (d2 >= EPSILON) {
            float ci = v.dot(a._normal) / d, cj = (-v).dot(b._normal) / d;
            if (ci > 0 && cj > 0 && !isRayBlocked(from, to, v / d, d, nodes, indices, mesh, i, j))
                contribution = (ci * cj) / (PI * d2);
        }
    }
    __shared__ float terms[threads];
    terms[threadIdx.x] = contribution;
    __syncwarp(lanes);
    if (ray == 0) {
        float kij = 0;
        for (int r = 0; r < NUM_RAYS; ++r) kij += terms[threadIdx.x + r];
        kij *= INV_NUM_RAYS;
        weights[index] = fminf(kij * areas[j], ONE);
        // The radius-10 room has no delay greater than 40 timesteps.
        delays[index] =
            static_cast<unsigned char>(ceilf((a.center() - b.center()).norm() * INV_WAVE_SPEED));
        if (nonZeroCount && kij > EPSILON) atomicAdd(nonZeroCount, 1ULL);
    }
}

struct SampleBatch {
    int* offsets;
    uint32_t* random;
    DeviceBuffer<int> deviceOffsets;
    DeviceBuffer<uint4> samples;
    cudaStream_t stream;
    explicit SampleBatch(size_t capacity) {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaCheck(cudaMallocHost(&offsets, capacity * sizeof(int)));
        cudaCheck(cudaMallocHost(&random, (624 + capacity * NUM_RAYS * 4) * sizeof(uint32_t)));
        deviceOffsets.allocate(capacity);
        samples.allocate(capacity * NUM_RAYS);
    }
    ~SampleBatch() {
        cudaStreamDestroy(stream);
        cudaFreeHost(random);
        cudaFreeHost(offsets);
    }
};

void computeFormFactors(SimulationState& s) {
    printf("Computing time delays (Tau) and form factors (Kij)...\n");
    size_t n = s.numTriangles, total = n * n;
    size_t capacity = std::min<size_t>(batchPairs, total);
    SampleBatch batches[] = {SampleBatch(capacity), SampleBatch(capacity)};
    RandomStream rng;
    cudaCheck(cudaDeviceSynchronize());
    for (size_t first = 0; first < total; first += capacity) {
        // The other stream can trace its rays while the host prepares this batch.
        auto& batch = batches[(first / capacity) % 2];
        cudaCheck(cudaStreamSynchronize(batch.stream));
        unsigned pairs = std::min(capacity, total - first);
        unsigned active = 0;
        size_t i = first / n, j = first % n;
        for (unsigned p = 0; p < pairs; ++p) {
            bool culled = i == j || s.triangles[i]._normal.dot(s.triangles[j]._normal) > 0.99f;
            batch.offsets[p] = culled ? -1 : static_cast<int>(active++);
            if (++j == n) {
                j = 0;
                ++i;
            }
        }
        size_t words = active * NUM_RAYS * 4;
        rng.fill(batch.random, words);
        cudaCheck(cudaMemcpyAsync(batch.deviceOffsets.data, batch.offsets, pairs * sizeof(int),
                                  cudaMemcpyHostToDevice, batch.stream));
        cudaCheck(cudaMemcpyAsync(batch.samples.data, batch.random + 624, words * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice, batch.stream));
        formFactors<<<(pairs * NUM_RAYS + threads - 1) / threads, threads, 0, batch.stream>>>(
            first, pairs, n, s.mesh.data, s.nodes.data, s.indices.data, batch.deviceOffsets.data,
            batch.samples.data, s.areas.data, s.weights.data, s.delays.data, s.count.data);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    if (s.validate)
        cudaCheck(cudaMemcpy(&s.nonZeroKij, s.count.data, sizeof(s.nonZeroKij), cudaMemcpyDeviceToHost));
}

__global__ void propagate(size_t n, size_t t, size_t source, size_t timeOff, float rho,
                          const float* weights, const unsigned char* delays, float* rad) {
    size_t i = blockIdx.x;
    __shared__ float terms[threads];
    float sum = 0;
    for (size_t base = 0; base < n; base += threads) {
        size_t j = base + threadIdx.x;
        float term = 0;
        if (j < n && i != j) {
            size_t index = i * n + j;
            unsigned delay = delays[index];
            if (delay <= t && weights[index] > 0) {
                float r = rad[(t - delay) * n + j];
                if (!(r <= ZERO)) term = weights[index] * r;
            }
        }
        terms[threadIdx.x] = term;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned k = 0; k < threads && base + k < n; ++k) sum += terms[k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) rad[t * n + i] = rho * sum + float(i == source && t < timeOff);
}

void runSimulation(SimulationState& s) {
    printf("Running wave propagation simulation...\n");
    for (size_t t = 0; t < s.numTimesteps; ++t) {
        propagate<<<s.numTriangles, threads>>>(s.numTriangles, t, s.sourceIndex, s.numTimesteps / 2,
                                               s.reflectivity, s.weights.data, s.delays.data,
                                               s.radiosity.data);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

__global__ void correlate(size_t n, size_t steps, size_t source, const float* rad, float* scores) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * steps) return;
    size_t i = idx % n, lag = idx / n;
    float sum = 0;
    for (size_t t = lag; t < steps; ++t) sum += rad[(t - lag) * n + source] * rad[t * n + i];
    scores[idx] = sum;
}

__global__ void bestDistances(size_t n, size_t steps, const float* scores, float* distances) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t best = 0;
    float maxCorr = 0;
    for (size_t lag = 0; lag < steps; ++lag) {
        float score = scores[lag * n + i];
        if (score > maxCorr) {
            maxCorr = score;
            best = lag;
        }
    }
    distances[i] = WAVE_SPEED * best;
}

void computeDistances(SimulationState& s) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = s.numTriangles, cells = n * s.numTimesteps;
    DeviceBuffer<float> scores;
    scores.allocate(cells);
    if (cells) {
        correlate<<<(cells + threads - 1) / threads, threads>>>(n, s.numTimesteps, s.sourceIndex,
                                                                s.radiosity.data, scores.data);
        cudaCheck(cudaGetLastError());
    }
    bestDistances<<<(n + threads - 1) / threads, threads>>>(n, s.numTimesteps, scores.data,
                                                            s.distance.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(
        cudaMemcpy(s.distances.data(), s.distance.data, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (s.validate && cells) {
        s.radB.resize(cells);
        cudaCheck(
            cudaMemcpy(s.radB.data(), s.radiosity.data, cells * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = FLT_MAX;
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
    auto nonZeroKij = state.nonZeroKij;
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

    if (timesteps < 0) {
        fprintf(stderr, "Timesteps must be non-negative.\n");
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
    state.validate = validate;
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

    // Storage for the pair matrices, radiosity history and correlation scores.
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(unsigned char);
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

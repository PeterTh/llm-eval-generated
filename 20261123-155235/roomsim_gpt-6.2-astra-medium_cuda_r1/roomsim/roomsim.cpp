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

    __host__ __device__ val_t dot(const Vec3& o) const {
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return dot(*this); }
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

// Depth-first layout allows a ray to skip an entire subtree with one index.
struct FlatNode {
    Vec3 center, halfExtent;
    int end, first, count;

    __device__ bool intersects(Vec3 p1, Vec3 p2) const {
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

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<idx_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flatten(*child, nodes, indices);
    nodes[index].end = static_cast<int>(nodes.size());
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
// Random Number Generation
// ============================================================================

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

__device__ bool isRayBlocked(Vec3 from, Vec3 to, Vec3 direction, float length,
                             const FlatNode* nodes, const idx_t* indices,
                             const Triangle* triangles, idx_t i, idx_t j) {
    for (int k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.intersects(from, to)) {
            k = node.end;
            continue;
        }
        for (int n = 0; n < node.count; ++n) {
            idx_t index = indices[node.first + n];
            if (index == i || index == j) continue;
            const Triangle& tri = triangles[index];
            float d = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }

    void allocate(size_t n) {
        if (n) checkCuda(cudaMalloc(&data, n * sizeof(T)));
    }
    void upload(const std::vector<T>& host) {
        allocate(host.size());
        checkCuda(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    float reflectivity;
    std::vector<Triangle> triangles;
    std::vector<float> distances;
    DeviceBuffer<Triangle> geometry;
    DeviceBuffer<Vec3> centers;
    DeviceBuffer<float> areas, weights, radiosity, deviceDistances;
    DeviceBuffer<unsigned char> delays;  // Radius 10 and speed 0.5 give delays <= 40.
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<unsigned long long> nonzero;
};

void initializeSimulation(SimulationState& s, int subdivisions, size_t timesteps,
                          size_t source, float reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    s.triangles = std::move(mesh.triangles);
    size_t n = s.numTriangles = s.triangles.size();
    s.numTimesteps = timesteps;
    s.sourceIndex = source % n;
    s.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", n);
    Octree tree;
    tree.build(s.triangles);
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flatten(tree, nodes, indices);
    s.nodes.upload(nodes);
    s.indices.upload(indices);
    s.geometry.upload(s.triangles);

    std::vector<float> areas(n);
    std::vector<Vec3> centers(n);
    for (size_t i = 0; i < n; ++i) {
        areas[i] = s.triangles[i].area();
        centers[i] = s.triangles[i].center();
    }
    s.centers.upload(centers);
    s.areas.upload(areas);
    s.weights.allocate(n * n);
    s.delays.allocate(n * n);
    checkCuda(cudaMemset(s.weights.data, 0, n * n * sizeof(float)));
    checkCuda(cudaMemset(s.delays.data, 0, n * n));
    s.radiosity.allocate(timesteps * n);
    s.deviceDistances.allocate(n);
    s.distances.resize(n);
    s.nonzero.allocate(1);
    checkCuda(cudaMemset(s.nonzero.data, 0, sizeof(unsigned long long)));
}

// Match generate_canonical<float> for the 32-bit MT19937 engine, including
// the upper endpoint correction used by uniform_real_distribution.
__device__ float uniformSample(unsigned bits) {
    return fminf(__uint2float_rn(bits) * 0x1p-32f, 0x1.fffffep-1f);
}

constexpr int BLOCK = 128;
__global__ void formFactors(const Triangle* triangles, const FlatNode* nodes,
                            const idx_t* indices, const Vec3* centers, const float* areas,
                            const uint2* pairs, const uint4* draws, int count, size_t n,
                            float* weights, unsigned char* delays, unsigned long long* nonzero) {
    int thread = blockIdx.x * BLOCK + threadIdx.x;
    int pair = thread / NUM_RAYS;
    int ray = thread % NUM_RAYS;
    __shared__ float contributions[BLOCK];
    float term = 0;
    uint2 ij = {};
    if (pair < count) {
        ij = pairs[pair];
        const Triangle& receiver = triangles[ij.x];
        const Triangle& emitter = triangles[ij.y];
        uint4 r = draws[thread];
        Vec3 from = randomPointInTriangle(receiver, uniformSample(r.x), uniformSample(r.y));
        Vec3 to = randomPointInTriangle(emitter, uniformSample(r.z), uniformSample(r.w));
        Vec3 delta = to - from;
        float squared = delta.squaredNorm();
        float length = sqrtf(squared);
        if (squared >= EPSILON) {
            float a = delta.dot(receiver._normal) / length;
            float b = (-delta).dot(emitter._normal) / length;
            if (a > 0 && b > 0 && !isRayBlocked(from, to, delta / length, length,
                                                  nodes, indices, triangles, ij.x, ij.y))
                term = (a * b) / (PI * squared);
        }
    }
    contributions[threadIdx.x] = term;
    __syncwarp();
    bool positive = false;
    if (ray == 0 && pair < count) {
        float sum = 0;
        for (int k = 0; k < NUM_RAYS; ++k) sum += contributions[threadIdx.x + k];
        float kij = sum * INV_NUM_RAYS;
        size_t index = size_t(ij.y) * n + ij.x;
        weights[index] = fminf(kij * areas[ij.y], ONE);
        delays[index] = static_cast<unsigned char>(ceilf((centers[ij.x] - centers[ij.y]).norm() * INV_WAVE_SPEED));
        positive = kij > EPSILON;
    }
    unsigned mask = __ballot_sync(0xffffffff, positive);
    if ((threadIdx.x % 32) == 0 && mask) atomicAdd(nonzero, __popc(mask));
}

// Double-buffer the original random stream. The CPU can generate the next
// batch while CUDA transfers and traces earlier batches.
constexpr int BATCH = 8192;
struct RayBatch {
    uint2* pairs;
    uint4* draws;
    DeviceBuffer<uint2> devicePairs;
    DeviceBuffer<uint4> deviceDraws;
    cudaStream_t stream;
    explicit RayBatch(int capacity) {
        checkCuda(cudaStreamCreate(&stream));
        checkCuda(cudaMallocHost(&pairs, capacity * sizeof(uint2)));
        checkCuda(cudaMallocHost(&draws, capacity * NUM_RAYS * sizeof(uint4)));
        devicePairs.allocate(capacity);
        deviceDraws.allocate(capacity * NUM_RAYS);
    }
    ~RayBatch() {
        cudaFreeHost(pairs);
        cudaFreeHost(draws);
        cudaStreamDestroy(stream);
    }
    void submit(SimulationState& s, int count) {
        checkCuda(cudaMemcpyAsync(devicePairs.data, pairs, count * sizeof(uint2),
                                  cudaMemcpyHostToDevice, stream));
        checkCuda(cudaMemcpyAsync(deviceDraws.data, draws, count * NUM_RAYS * sizeof(uint4),
                                  cudaMemcpyHostToDevice, stream));
        formFactors<<<(count * NUM_RAYS + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            s.geometry.data, s.nodes.data, s.indices.data, s.centers.data, s.areas.data,
            devicePairs.data, deviceDraws.data, count, s.numTriangles,
            s.weights.data, s.delays.data, s.nonzero.data);
        checkCuda(cudaGetLastError());
    }
};

void computeFormFactors(SimulationState& s) {
    printf("Computing form factors (Kij) and time delays (Tau)...\n");
    std::mt19937 rng(42);
    int capacity = std::min<size_t>(BATCH, s.numTriangles * (s.numTriangles - 1));
    RayBatch batches[2] = {RayBatch(capacity), RayBatch(capacity)};
    int active = 0, count = 0;
    for (idx_t i = 0; i < s.numTriangles; ++i) {
        for (idx_t j = 0; j < s.numTriangles; ++j) {
            if (i == j || s.triangles[i]._normal.dot(s.triangles[j]._normal) > 0.99f) continue;
            RayBatch& batch = batches[active];
            batch.pairs[count] = make_uint2(i, j);
            for (int k = 0; k < NUM_RAYS; ++k) {
                batch.draws[count * NUM_RAYS + k] = {
                    unsigned(rng()), unsigned(rng()), unsigned(rng()), unsigned(rng())};
            }
            if (++count == capacity) {
                batch.submit(s, count);
                active ^= 1;
                checkCuda(cudaStreamSynchronize(batches[active].stream));
                count = 0;
            }
        }
    }
    if (count) batches[active].submit(s, count);
    checkCuda(cudaDeviceSynchronize());
}

__global__ void propagate(size_t n, size_t t, size_t source, float emission, float rho,
                          const float* weights, const unsigned char* delays, float* rad) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0;
    // Adjacent receivers read contiguous weights and delays. Each receiver
    // accumulates emitters in the original order.
    for (size_t j = 0; j < n; ++j) {
        size_t index = j * n + i;
        unsigned delay = delays[index];
        float weight = weights[index];
        if (weight > 0 && delay <= t) {
            float incoming = rad[(t - delay) * n + j];
            // Round each product before adding it to the running sum.
            if (incoming > 0) sum += __fmul_rn(weight, incoming);
        }
    }
    rad[t * n + i] = rho * sum + (i == source ? emission : ZERO);
}

void runSimulation(SimulationState& s) {
    printf("Running wave propagation simulation...\n");
    size_t n = s.numTriangles;
    for (size_t t = 0; t < s.numTimesteps; ++t) {
        propagate<<<(n + 31) / 32, 32>>>(n, t, s.sourceIndex, t < s.numTimesteps / 2,
            s.reflectivity, s.weights.data, s.delays.data, s.radiosity.data);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

struct Peak {
    float score;
    int lag;

    __device__ void consider(float candidate, int t) {
        if (candidate > score || (candidate == score && t < lag)) {
            score = candidate;
            lag = t;
        }
    }
};

// Each block evaluates a tile of lags for one receiver. Stage the two
// overlapping signal windows once for all lags in the block.
__global__ void correlate(const float* rad, size_t n, size_t steps, size_t source,
                          Peak* partial, float* distances) {
    size_t i = blockIdx.x;
    int lane = threadIdx.x;
    Peak best = {};
    __shared__ float sourceWindow[BLOCK], receiverWindow[2 * BLOCK];
    for (size_t base = size_t(blockIdx.y) * BLOCK; base < steps;
         base += size_t(gridDim.y) * BLOCK) {
        size_t lag = base + lane;
        float sum = 0;
        for (size_t k = 0; k < steps - base; k += BLOCK) {
            size_t t = k + lane;
            if (t < steps - base) {
                sourceWindow[lane] = rad[t * n + source];
                receiverWindow[lane] = rad[(base + t) * n + i];
            }
            if (t + BLOCK < steps - base)
                receiverWindow[BLOCK + lane] = rad[(base + t + BLOCK) * n + i];
            __syncthreads();
            for (int j = 0; j < BLOCK && lag + k + j < steps; ++j)
                sum += __fmul_rn(sourceWindow[j], receiverWindow[lane + j]);
            __syncthreads();
        }
        if (lag < steps) best.consider(sum, static_cast<int>(lag));
    }
    __shared__ Peak peaks[BLOCK];
    peaks[lane] = best;
    for (int stride = BLOCK / 2; stride; stride /= 2) {
        __syncthreads();
        if (lane < stride) {
            Peak other = peaks[lane + stride];
            best.consider(other.score, other.lag);
            peaks[lane] = best;
        }
    }
    if (lane == 0) {
        if (gridDim.y == 1) distances[i] = WAVE_SPEED * best.lag;
        else partial[size_t(blockIdx.y) * n + i] = best;
    }
}

__global__ void combinePeaks(const Peak* partial, size_t n, int tiles, float* distances) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Peak best = {};
    for (int tile = 0; tile < tiles; ++tile) {
        Peak peak = partial[size_t(tile) * n + i];
        best.consider(peak.score, peak.lag);
    }
    distances[i] = WAVE_SPEED * best.lag;
}

void computeDistances(SimulationState& s) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = s.numTriangles;
    int tiles = std::clamp<size_t>((s.numTimesteps + BLOCK - 1) / BLOCK, 1, 64);
    DeviceBuffer<Peak> partial;
    if (tiles > 1) partial.allocate(size_t(tiles) * n);
    correlate<<<dim3(n, tiles), BLOCK>>>(s.radiosity.data, n, s.numTimesteps,
                                         s.sourceIndex, partial.data, s.deviceDistances.data);
    checkCuda(cudaGetLastError());
    if (tiles > 1) {
        combinePeaks<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(partial.data, n, tiles, s.deviceDistances.data);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaMemcpy(s.distances.data(), s.deviceDistances.data,
                         n * sizeof(float), cudaMemcpyDeviceToHost));
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
    std::vector<float> radB(state.numTimesteps * state.numTriangles);
    if (!radB.empty()) checkCuda(cudaMemcpy(radB.data(), state.radiosity.data,
        radB.size() * sizeof(float), cudaMemcpyDeviceToHost));
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
    unsigned long long nonZeroKij;
    checkCuda(cudaMemcpy(&nonZeroKij, state.nonzero.data, sizeof(nonZeroKij), cudaMemcpyDeviceToHost));
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

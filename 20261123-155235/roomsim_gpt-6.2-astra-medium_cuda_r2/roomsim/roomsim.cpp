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
#include <cuda_runtime.h>
#include <math_constants.h>
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
constexpr val_t ROOM_RADIUS = 10.0f;
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
#ifdef __CUDA_ARCH__
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
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

    __host__ __device__ bool operator==(const Vec3& o) const {
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

// Depth-first order makes the end of a subtree its traversal skip address.
struct FlatNode {
    Vec3 center, halfExtent;
    unsigned end, first, count;

    __device__ bool intersects(const Vec3& midpoint, const Vec3& d, const Vec3& ad) const {
        Vec3 c = midpoint - center;

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

};

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<unsigned>& indices) {
    const size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<unsigned>(indices.size()),
                     static_cast<unsigned>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flatten(*child, nodes, indices);
    }
    nodes[index].end = static_cast<unsigned>(nodes.size());
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

__device__ Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v, val_t& minWeight) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    minWeight = fminf(fminf(u, v), 1.0f - (u + v));
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

    if (std::abs(det) < EPSILON) return CUDART_MAX_NORMAL_F;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return CUDART_MAX_NORMAL_F;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return CUDART_MAX_NORMAL_F;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

__device__ bool isRayBlocked(Vec3 from, Vec3 direction, val_t length,
                             const Triangle* triangles, const FlatNode* nodes,
                             const unsigned* indices, unsigned i, unsigned j) {
    const Vec3 half = direction * 0.5f;
    const Vec3 midpoint = from + half;
    const Vec3 extent(fabsf(half.x), fabsf(half.y), fabsf(half.z));
    const Vec3 unit = direction / length;
    for (unsigned k = 0; k < nodes[0].end;) {
        const FlatNode node = nodes[k];
        if (k && !node.intersects(midpoint, half, extent)) {
            k = node.end;
            continue;
        }
        for (unsigned p = node.first; p < node.first + node.count; ++p) {
            const unsigned id = indices[p];
            if (id == i || id == j) continue;
            const Triangle& tri = triangles[id];
            const val_t dist = rayTriangleIntersect(from, unit, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t count, bool managed = false) {
        if (!count) return;
        if (managed) check(cudaMallocManaged(&data, count * sizeof(T)));
        else check(cudaMalloc(&data, count * sizeof(T)));
    }
    void upload(const std::vector<T>& host) {
        allocate(host.size());
        check(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    val_t reflectivity;
    bool managedMatrices;
    std::vector<Triangle> triangles;
    std::vector<val_t> distances;
    DeviceBuffer<Triangle> mesh;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<unsigned> indices;
    DeviceBuffer<val_t> weights, radB, gpuDistances, interiorThreshold;
    DeviceBuffer<unsigned char> delays;
    DeviceBuffer<unsigned long long> nonZero;
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool validate) {
    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    state.triangles = std::move(mesh.triangles);
    const size_t n = state.triangles.size();
    state.numTriangles = n;
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % n;
    state.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %zu triangles\n", n);

    printf("Building octree...\n");
    Octree tree;
    tree.build(state.triangles);
    std::vector<FlatNode> nodes;
    std::vector<unsigned> indices;
    flatten(tree, nodes, indices);
    state.nodes.upload(nodes);
    state.indices.upload(indices);
    state.mesh.upload(state.triangles);
    state.interiorThreshold.allocate(n);
    // On meshes larger than device memory, CUDA can page the dense matrices.
    size_t available, total;
    check(cudaMemGetInfo(&available, &total));
    const size_t matrixBytes = n * n * (sizeof(float) + sizeof(unsigned char));
    const size_t historyBytes = n * timesteps * sizeof(float);
    state.managedMatrices = matrixBytes + historyBytes > available - available / 8;
    state.weights.allocate(n * n, state.managedMatrices);
    // The diameter is 20 units, hence all delays fit in 40 timesteps.
    state.delays.allocate(n * n, state.managedMatrices);
    state.radB.allocate(n * timesteps, state.managedMatrices);
    state.distances.resize(n);
    state.gpuDistances.allocate(n);
    if (validate) {
        state.nonZero.allocate(1);
        check(cudaMemset(state.nonZero.data, 0, sizeof(unsigned long long)));
    }
}

__global__ void computeDelays(unsigned n, const Triangle* mesh, unsigned char* delays) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        const val_t d = (mesh[i].center() - mesh[j].center()).norm();
        delays[size_t(i) * n + j] = static_cast<unsigned char>(ceilf(d * INV_WAVE_SPEED));
    }
}

// For a point with minimum barycentric weight w, at least 3*w of the
// point is the centroid, with the rest a convex combination of vertices.
// Bound its signed distance to every other face plane. Two endpoints on the
// interior side of every possible blocker imply the whole segment is clear.
// Samples too close to any plane still use the original octree test.
__global__ void computeInteriorThresholds(unsigned n, const Triangle* mesh, float* thresholds) {
    const unsigned i = blockIdx.x;
    const Triangle tri = mesh[i];
    float threshold = 0;
    for (unsigned j = threadIdx.x; j < n; j += blockDim.x) {
        if (i == j) continue;
        const Triangle other = mesh[j];
        const Vec3 normal = other.normal();
        const float a = (tri.a - other.a).dot(normal);
        const float b = (tri.b - other.a).dot(normal);
        const float c = (tri.c - other.a).dot(normal);
        const float low = fminf(a, fminf(b, c));
        const float center = (a + b + c) / 3.0f;
        // Allow for rounding in the sampled endpoints and intersection
        // determinants. The edge product / cross magnitude accounts for
        // conditioning when projecting onto the face plane.
        const Vec3 ab = other.b - other.a, ac = other.c - other.a;
        const float condition = ab.norm() * ac.norm() / ab.cross(ac).norm();
        const float margin = (256.0f * 0x1p-23f * 2.0f * ROOM_RADIUS) * condition;
        if (low <= margin) {
            const float needed = center > margin ? (margin - low) / (3.0f * (center - low)) : 1.0f;
            threshold = fmaxf(threshold, needed);
        }
    }
    __shared__ float bounds[128];
    bounds[threadIdx.x] = threshold;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            bounds[threadIdx.x] = fmaxf(bounds[threadIdx.x], bounds[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) thresholds[i] = bounds[0];
}

// Each half warp traces the 16 samples for one ordered pair. Keep the original
// order of addition when combining their contributions.
constexpr int FACTOR_THREADS = 128;
__global__ void formFactors(unsigned n, const Triangle* mesh, const FlatNode* nodes,
                            const unsigned* indices, const float* thresholds, const uint2* pairs,
                            const float4* samples, unsigned count, val_t* weights,
                            unsigned long long* nonZero) {
    __shared__ float values[FACTOR_THREADS];
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = thread / NUM_RAYS;
    const unsigned ray = thread % NUM_RAYS;
    float value = ZERO;
    uint2 ij{};
    if (pair < count) {
        ij = pairs[pair];
        const Triangle& a = mesh[ij.x];
        const Triangle& b = mesh[ij.y];
        const float4 sample = samples[thread];
        float wa, wb;
        const Vec3 from = randomPointInTriangle(a, sample.x, sample.y, wa);
        const Vec3 to = randomPointInTriangle(b, sample.z, sample.w, wb);
        const Vec3 v = to - from;
        const val_t squared = v.squaredNorm();
        const val_t length = sqrtf(squared);
        if (squared >= EPSILON) {
            const val_t ci = v.dot(a.normal()) / length;
            const val_t cj = (-v).dot(b.normal()) / length;
            const bool interior = wa > thresholds[ij.x] && wb > thresholds[ij.y];
            if (ci > ZERO && cj > ZERO &&
                (interior || !isRayBlocked(from, v, length, mesh, nodes, indices, ij.x, ij.y))) {
                value = (ci * cj) / (PI * squared);
            }
        }
    }
    values[threadIdx.x] = value;
    __syncthreads();
    bool significant = false;
    if (!ray && pair < count) {
        float sum = ZERO;
        for (int r = 0; r < NUM_RAYS; ++r) sum += values[threadIdx.x + r];
        const float kij = sum * INV_NUM_RAYS;
        significant = kij > EPSILON;
        weights[size_t(ij.x) * n + ij.y] = fminf(kij * mesh[ij.y].area(), ONE);
    }
    if (nonZero) {
        const int total = __syncthreads_count(significant);
        if (!threadIdx.x && total) atomicAdd(nonZero, static_cast<unsigned long long>(total));
    }
}

// Checkpoint the untempered MT19937 stream. Advancing the host state only
// requires the vectorizable twist recurrence. A CUDA block expands each
// checkpoint independently, in exactly the order of std::mt19937.
constexpr int MT_WORDS = std::mt19937::state_size;
constexpr int MT_OFFSET = std::mt19937::shift_size;
constexpr int MT_SPAN = MT_WORDS - MT_OFFSET;
constexpr int TWISTS_PER_GROUP = 8;
constexpr int PAIRS_PER_GROUP = TWISTS_PER_GROUP * MT_WORDS / (4 * NUM_RAYS);
constexpr unsigned BATCH_PAIRS = 128 * PAIRS_PER_GROUP;
static_assert(PAIRS_PER_GROUP * 4 * NUM_RAYS == TWISTS_PER_GROUP * MT_WORDS);

__host__ __device__ inline uint32_t twistBits(uint32_t high, uint32_t low) {
    const uint32_t bits = (high & 0x80000000u) | (low & 0x7fffffffu);
    return (bits >> 1) ^ ((0u - (bits & 1u)) & 0x9908b0dfu);
}

struct SampleStream {
    uint32_t words[MT_WORDS];

    explicit SampleStream(uint32_t seed) {
        words[0] = seed;
        for (int k = 1; k < MT_WORDS; ++k) {
            const uint32_t last = words[k - 1];
            words[k] = 1812433253u * (last ^ (last >> 30)) + k;
        }
    }

    void checkpoint(uint32_t* dest) {
        memcpy(dest, words, sizeof(words));
        for (int step = 0; step < TWISTS_PER_GROUP; ++step) {
            for (int k = 0; k < MT_SPAN; ++k)
                words[k] = words[k + MT_OFFSET] ^ twistBits(words[k], words[k + 1]);
            for (int k = MT_SPAN; k < MT_WORDS - 1; ++k)
                words[k] = words[k - MT_SPAN] ^ twistBits(words[k], words[k + 1]);
            words[MT_WORDS - 1] = words[MT_OFFSET - 1] ^ twistBits(words[MT_WORDS - 1], words[0]);
        }
    }
};

__device__ float uniformSample(uint32_t bits) {
    bits ^= bits >> 11;
    bits ^= (bits << 7) & 0x9d2c5680u;
    bits ^= (bits << 15) & 0xefc60000u;
    bits ^= bits >> 18;
    // generate_canonical<float> clamps values rounded up to one.
    return fminf(static_cast<float>(bits) * 0x1p-32f, 0x1.fffffep-1f);
}

__global__ void generateSamples(const uint32_t* checkpoints, float* samples) {
    __shared__ uint32_t states[2][MT_WORDS];
    const int lane = threadIdx.x;
    for (int k = lane; k < MT_WORDS; k += blockDim.x)
        states[0][k] = checkpoints[blockIdx.x * MT_WORDS + k];
    __syncthreads();
    for (int step = 0; step < TWISTS_PER_GROUP; ++step) {
        const uint32_t* prev = states[step % 2];
        uint32_t* next = states[1 - step % 2];
        // Within each span all words can be twisted simultaneously.
        for (int first = 0; first < MT_WORDS; first += MT_SPAN) {
            const int k = first + lane;
            if (lane < MT_SPAN && k < MT_WORDS) {
                const uint32_t offset = first ? next[k - MT_SPAN] : prev[k + MT_OFFSET];
                const uint32_t low = k + 1 < MT_WORDS ? prev[k + 1] : next[0];
                next[k] = offset ^ twistBits(prev[k], low);
                samples[(blockIdx.x * TWISTS_PER_GROUP + step) * MT_WORDS + k] = uniformSample(next[k]);
            }
            __syncthreads();
        }
    }
}

// Double-buffer bounded batches so host checkpoint generation overlaps tracing.
struct SampleBatch {
    uint2* pairs;
    uint32_t* checkpoints;
    DeviceBuffer<uint2> gpuPairs;
    DeviceBuffer<uint32_t> gpuCheckpoints;
    DeviceBuffer<float4> gpuSamples;
    cudaStream_t stream;

    explicit SampleBatch(unsigned groups) {
        const unsigned capacity = groups * PAIRS_PER_GROUP;
        const size_t words = groups * MT_WORDS;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMallocHost(&pairs, capacity * sizeof(uint2)));
        check(cudaMallocHost(&checkpoints, words * sizeof(uint32_t)));
        gpuPairs.allocate(capacity);
        gpuCheckpoints.allocate(words);
        gpuSamples.allocate(capacity * NUM_RAYS);
    }
    ~SampleBatch() {
        cudaStreamDestroy(stream);
        cudaFreeHost(checkpoints);
        cudaFreeHost(pairs);
    }
    void submit(const SimulationState& s, unsigned count, SampleStream& rng) {
        const unsigned groups = (count + PAIRS_PER_GROUP - 1) / PAIRS_PER_GROUP;
        for (unsigned k = 0; k < groups; ++k) rng.checkpoint(checkpoints + k * MT_WORDS);
        check(cudaMemcpyAsync(gpuPairs.data, pairs, count * sizeof(uint2),
                              cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(gpuCheckpoints.data, checkpoints, groups * MT_WORDS * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
        generateSamples<<<groups, 256, 0, stream>>>(gpuCheckpoints.data,
                                                      reinterpret_cast<float*>(gpuSamples.data));
        check(cudaGetLastError());
        formFactors<<<(count * NUM_RAYS + FACTOR_THREADS - 1) / FACTOR_THREADS,
                      FACTOR_THREADS, 0, stream>>>(s.numTriangles, s.mesh.data, s.nodes.data,
            s.indices.data, s.interiorThreshold.data, gpuPairs.data, gpuSamples.data, count,
            s.weights.data, s.nonZero.data);
        check(cudaGetLastError());
    }
};

void precompute(SimulationState& s) {
    const unsigned n = s.numTriangles;
    printf("Computing time delays (Tau)...\n");
    const dim3 block(32, 8);
    computeDelays<<<dim3((n + 31) / 32, (n + 7) / 8), block>>>(n, s.mesh.data, s.delays.data);
    check(cudaGetLastError());
    computeInteriorThresholds<<<n, 128>>>(n, s.mesh.data, s.interiorThreshold.data);
    check(cudaGetLastError());
    check(cudaMemset(s.weights.data, 0, size_t(n) * n * sizeof(float)));
    check(cudaDeviceSynchronize());
    printf("Computing form factors (Kij)...\n");
    SampleStream rng(42);
    const size_t pairs = size_t(n) * (n - 1);
    const unsigned capacity = std::min(size_t(BATCH_PAIRS), pairs);
    SampleBatch first((capacity + PAIRS_PER_GROUP - 1) / PAIRS_PER_GROUP);
    std::unique_ptr<SampleBatch> second;
    if (pairs >= BATCH_PAIRS) second = std::make_unique<SampleBatch>(BATCH_PAIRS / PAIRS_PER_GROUP);
    SampleBatch* batches[] = {&first, second.get()};
    unsigned current = 0, count = 0;
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            if (i == j || s.triangles[i].normal().dot(s.triangles[j].normal()) > 0.99f) continue;
            auto& batch = *batches[current];
            batch.pairs[count] = make_uint2(i, j);
            if (++count == BATCH_PAIRS) {
                batch.submit(s, count, rng);
                count = 0;
                current ^= 1;
                check(cudaStreamSynchronize(batches[current]->stream));
            }
        }
        if ((i + 1) % 100 == 0 || i + 1 == n) printf("  Progress: %u/%u triangles\n", i + 1, n);
    }
    if (count) batches[current]->submit(s, count, rng);
    check(cudaDeviceSynchronize());
}

__global__ void propagate(unsigned n, size_t time, size_t onTime, unsigned source,
                          float rho, const float* weights, const unsigned char* delays,
                          float* history) {
    const unsigned i = blockIdx.x;
    const size_t row = size_t(i) * n;
    // Gather products in parallel and accumulate in emitter order. Reassociating
    // this sum can change the winning lag when correlations are nearly tied.
    __shared__ float terms[128];
    float sum = ZERO;
    for (unsigned start = 0; start < n; start += blockDim.x) {
        const unsigned j = start + threadIdx.x;
        float term = ZERO;
        if (j < n) {
            const unsigned delay = delays[row + j];
            const float weight = weights[row + j];
            if (delay && delay <= time && weight > ZERO) {
                const float energy = history[(time - delay) * n + j];
                if (energy > ZERO) term = weight * energy;
            }
        }
        terms[threadIdx.x] = term;
        __syncthreads();
        if (!threadIdx.x) {
            for (unsigned k = 0; k < 128; ++k) sum += terms[k];
        }
        __syncthreads();
    }
    if (!threadIdx.x) history[time * n + i] = rho * sum + float(i == source && time < onTime);
}

void runSimulation(SimulationState& s) {
    printf("Running wave propagation simulation...\n");
    for (size_t t = 0; t < s.numTimesteps; ++t) {
        propagate<<<s.numTriangles, 128>>>(s.numTriangles, t, s.numTimesteps / 2,
            s.sourceIndex, s.reflectivity, s.weights.data, s.delays.data, s.radB.data);
        check(cudaGetLastError());
        if ((t + 1) % 10 == 0 || t + 1 == s.numTimesteps)
            printf("  Timestep %zu/%zu\n", t + 1, s.numTimesteps);
    }
    check(cudaDeviceSynchronize());
}

struct CorrelationPeak {
    float score;
    unsigned lag;
};

constexpr unsigned CORR_RECEIVERS = 8;
constexpr unsigned CORR_LAGS = 16;

// Adjacent lanes read adjacent receivers. Distributing the lag tiles across
// blocks also exposes parallelism for a small mesh with a long time history.
__global__ void correlate(unsigned n, unsigned steps, unsigned source, const float* history,
                          CorrelationPeak* peaks, float* distances) {
    const unsigned i = blockIdx.y * CORR_RECEIVERS + threadIdx.x;
    const unsigned lag = blockIdx.x * CORR_LAGS + threadIdx.y;
    float score = ZERO;
    if (i < n) {
        for (unsigned t = lag; t < steps; ++t)
            score += __fmul_rn(history[size_t(t - lag) * n + source], history[size_t(t) * n + i]);
    }
    __shared__ float scores[CORR_LAGS][CORR_RECEIVERS];
    scores[threadIdx.y][threadIdx.x] = score;
    __syncthreads();
    if (!threadIdx.y && i < n) {
        CorrelationPeak best{ZERO, 0};
        for (unsigned k = 0; k < CORR_LAGS; ++k) {
            const float candidate = scores[k][threadIdx.x];
            if (candidate > best.score) best = {candidate, blockIdx.x * CORR_LAGS + k};
        }
        if (gridDim.x == 1) distances[i] = WAVE_SPEED * best.lag;
        else peaks[size_t(blockIdx.x) * n + i] = best;
    }
}

__global__ void finishDistances(unsigned n, unsigned tiles, const CorrelationPeak* peaks,
                                float* distances) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    CorrelationPeak best{ZERO, 0};
    for (unsigned k = 0; k < tiles; ++k) {
        const CorrelationPeak candidate = peaks[size_t(k) * n + i];
        if (candidate.score > best.score) best = candidate;
    }
    distances[i] = WAVE_SPEED * best.lag;
}

void computeDistances(SimulationState& s) {
    printf("Computing distances via cross-correlation...\n");
    const unsigned n = s.numTriangles;
    const unsigned tiles = std::max(size_t(1), (s.numTimesteps + CORR_LAGS - 1) / CORR_LAGS);
    DeviceBuffer<CorrelationPeak> peaks;
    if (tiles > 1) peaks.allocate(size_t(n) * tiles, s.managedMatrices);
    correlate<<<dim3(tiles, (n + CORR_RECEIVERS - 1) / CORR_RECEIVERS),
                dim3(CORR_RECEIVERS, CORR_LAGS)>>>(n, s.numTimesteps, s.sourceIndex,
                                                  s.radB.data, peaks.data, s.gpuDistances.data);
    check(cudaGetLastError());
    if (tiles > 1) {
        finishDistances<<<(n + 127) / 128, 128>>>(n, tiles, peaks.data, s.gpuDistances.data);
        check(cudaGetLastError());
    }
    check(cudaMemcpy(s.distances.data(), s.gpuDistances.data, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

__global__ void countReceivers(unsigned n, unsigned steps, const float* history, unsigned* count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = 0;
    if (i < n) {
        while (t < steps && !(history[size_t(t) * n + i] > EPSILON)) ++t;
    }
    const unsigned received = __syncthreads_count(i < n && t < steps);
    if (!threadIdx.x) atomicAdd(count, received);
}

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
    DeviceBuffer<unsigned> count;
    count.allocate(1);
    check(cudaMemset(count.data, 0, sizeof(unsigned)));
    countReceivers<<<(state.numTriangles + 127) / 128, 128>>>(state.numTriangles, state.numTimesteps,
                                                             state.radB.data, count.data);
    check(cudaGetLastError());
    unsigned receivedEnergy;
    check(cudaMemcpy(&receivedEnergy, count.data, sizeof(unsigned), cudaMemcpyDeviceToHost));
    printf("  Triangles receiving energy: %u/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij;
    check(cudaMemcpy(&nonZeroKij, state.nonZero.data, sizeof(nonZeroKij), cudaMemcpyDeviceToHost));
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
        fprintf(stderr, "Number of timesteps must be non-negative.\n");
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

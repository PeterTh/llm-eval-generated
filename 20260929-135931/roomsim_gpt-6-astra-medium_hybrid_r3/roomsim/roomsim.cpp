/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
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
#include <random>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdarg>
#include <cfloat>

#include "../common/results_output.hpp"

// Only rank zero owns console output; MPI calls remain on the main thread.
int worldRank = 0, worldSize = 1;
int rootPrintf(const char* format, ...) {
    if (worldRank) return 0;
    va_list args;
    va_start(args, format);
    int result = vprintf(format, args);
    va_end(args);
    return result;
}
#define printf rootPrintf
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t n) { if (n) cudaCheck(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    void selectDevice() {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        cudaCheck(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

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

        #pragma omp parallel
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
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
                #pragma omp task firstprivate(i, childMin, childMax) shared(childIndices) if(indices.size() >= 256)
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
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

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(uint64_t n) { rng.discard(n); }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t begin = 0, localRows = 0;
    std::vector<int> counts, offsets;
    DeviceBuffer<val_t> deviceK, deviceB, deviceAreas, deviceDistances;
    DeviceBuffer<int> deviceTau;
    unsigned long long nonZeroKij = 0;
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
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
    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        state.offsets[r] = state.numTriangles * r / worldSize;
        state.counts[r] = state.numTriangles * (r + 1) / worldSize - state.offsets[r];
    }
    state.begin = state.offsets[worldRank];
    state.localRows = state.counts[worldRank];

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.deviceK.allocate(state.localRows * state.numTriangles);
    state.deviceTau.allocate(state.localRows * state.numTriangles);
    state.deviceB.allocate(timesteps * state.numTriangles);
    cudaCheck(cudaMemset(state.deviceB.data, 0, timesteps * state.numTriangles * sizeof(val_t)));
    state.deviceAreas.allocate(state.numTriangles);
    cudaCheck(cudaMemcpy(state.deviceAreas.data, state.areas.data(), state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    state.deviceDistances.allocate(state.localRows);
    if (!worldRank) state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // The CUDA propagation kernel evaluates the source emission analytically.
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// A depth-first flattened octree permits stackless traversal on the GPU.
struct FlatNode {
    Vec3 center, halfExtent;
    int end;
    size_t first, count;
    __device__ bool intersects(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
        if (fabsf(c.x) > halfExtent.x + ad.x || fabsf(c.y) > halfExtent.y + ad.y ||
            fabsf(c.z) > halfExtent.z + ad.z) return false;
        if (fabsf(d.y*c.z-d.z*c.y) > halfExtent.y*ad.z+halfExtent.z*ad.y+EPSILON) return false;
        if (fabsf(d.z*c.x-d.x*c.z) > halfExtent.z*ad.x+halfExtent.x*ad.z+EPSILON) return false;
        return fabsf(d.x*c.y-d.y*c.x) <= halfExtent.x*ad.y+halfExtent.y*ad.x+EPSILON;
    }
};
void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<size_t>& indices) {
    size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, indices.size(), tree.triangleIndices.size()});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) if (child) flatten(*child, nodes, indices);
    nodes[here].end = static_cast<int>(nodes.size());
}
__device__ bool blocked(const Vec3& from, const Vec3& to, size_t i, size_t j,
                        const Triangle* tris, const FlatNode* nodes, const size_t* indices) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    for (int n = 0; n < nodes[0].end;) {
        const FlatNode& node = nodes[n];
        if (n && !node.intersects(from, to)) { n = node.end; continue; }
        for (size_t k = 0; k < node.count; ++k) {
            size_t idx = indices[node.first+k];
            if (idx == i || idx == j) continue;
            const Triangle& tri = tris[idx];
            val_t distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length-EPSILON) return true;
        }
        ++n;
    }
    return false;
}
__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}
__global__ void formKernel(const Triangle* tris, const FlatNode* nodes, const size_t* indices,
                           const val_t* samples, val_t* kij, int* tau,
                           size_t n, size_t begin, size_t rows, size_t batchStart, size_t batchRows) {
    size_t pair = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair >= batchRows*n) return;
    size_t row = batchStart+pair/n, i = begin+row, j = pair%n;
    size_t output = j*rows+row; // Coalesced receiver access during propagation.
    const Triangle& a = tris[i];
    const Triangle& b = tris[j];
    tau[output] = i == j ? 0 : int(ceilf((a.center()-b.center()).norm()*INV_WAVE_SPEED));
    val_t sum = ZERO;
    if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
        for (int r = 0; r < NUM_RAYS; ++r) {
            const val_t* random = samples+pair*(4*NUM_RAYS)+4*r;
            Vec3 p = sampledPoint(a, random[0], random[1]);
            Vec3 q = sampledPoint(b, random[2], random[3]);
            if (blocked(p, q, i, j, tris, nodes, indices)) continue;
            Vec3 v = q-p;
            val_t d = v.squaredNorm();
            if (d < EPSILON) continue;
            val_t ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
            if (ci > ZERO && cj > ZERO) sum += (ci*cj)/(PI*d);
        }
    }
    kij[output] = sum*INV_NUM_RAYS;
}
__global__ void countFactors(const val_t* values, size_t count, unsigned long long* result) {
    __shared__ unsigned long long partial[256];
    unsigned long long local = 0;
    for (size_t k = size_t(blockIdx.x)*blockDim.x+threadIdx.x; k < count;
         k += size_t(blockDim.x)*gridDim.x)
        if (values[k] > EPSILON) ++local;
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(result, static_cast<unsigned long long>(partial[0]));
}
void computeTimeDelays(SimulationState&) {
    printf("Computing time delays and form factors on CUDA devices...\n");
}
void computeFormFactors(SimulationState& state) {
    const size_t n = state.numTriangles, rows = state.localRows;
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flatten(state.octree, nodes, indices);
    DeviceBuffer<Triangle> tris;
    DeviceBuffer<FlatNode> gpuNodes;
    DeviceBuffer<size_t> gpuIndices;
    tris.allocate(n); gpuNodes.allocate(nodes.size()); gpuIndices.allocate(indices.size());
    cudaCheck(cudaMemcpy(tris.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpuNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpuIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));

    // A culled pair consumes no random numbers in the reference implementation.
    // Save row-start engines before parallel sample generation; neither scheduling
    // nor MPI decomposition changes the Monte Carlo experiment.
    std::vector<size_t> draws(state.begin+rows);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < draws.size(); ++i) {
        size_t pairs = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) ++pairs;
        draws[i] = pairs*(4*NUM_RAYS);
    }
    RandomGenerator engine(42);
    for (size_t i = 0; i < state.begin; ++i) engine.discard(draws[i]);
    std::vector<RandomGenerator> engines;
    engines.reserve(rows);
    for (size_t row = 0; row < rows; ++row) {
        engines.push_back(engine);
        engine.discard(draws[state.begin+row]);
    }
    // Bounded staging storage, independent of the total matrix size.
    size_t batch = std::max(size_t(1), std::min(rows, size_t(32*1024*1024)/(n*4*NUM_RAYS*sizeof(val_t))));
    std::vector<val_t> samples(batch*n*4*NUM_RAYS);
    DeviceBuffer<val_t> gpuSamples;
    gpuSamples.allocate(samples.size());
    for (size_t first = 0; first < rows; first += batch) {
        size_t count = std::min(batch, rows-first);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < count; ++row) {
            size_t i = state.begin+first+row;
            auto rng = engines[first+row];
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                val_t* out = samples.data()+(row*n+j)*4*NUM_RAYS;
                for (int k = 0; k < 4*NUM_RAYS; ++k) out[k] = rng.rand();
            }
        }
        cudaCheck(cudaMemcpy(gpuSamples.data, samples.data(), count*n*4*NUM_RAYS*sizeof(val_t), cudaMemcpyHostToDevice));
        formKernel<<<(count*n+127)/128,128>>>(tris.data, gpuNodes.data, gpuIndices.data,
            gpuSamples.data, state.deviceK.data, state.deviceTau.data, n, state.begin, rows, first, count);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    DeviceBuffer<unsigned long long> counter;
    counter.allocate(1);
    cudaCheck(cudaMemset(counter.data, 0, sizeof(unsigned long long)));
    if (rows) {
        countFactors<<<std::min(size_t(1024), (rows*n+255)/256),256>>>(state.deviceK.data, rows*n, counter.data);
        cudaCheck(cudaGetLastError());
    }
    unsigned long long local = 0;
    cudaCheck(cudaMemcpy(&local, counter.data, sizeof(local), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

__global__ void propagate(const val_t* kij, const int* tau, const val_t* areas, val_t* history,
                          size_t n, size_t rows, size_t begin, size_t t, size_t timeOff,
                          size_t source, val_t rho) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    size_t i = begin+row;
    val_t sum = ZERO;
    // Preserve the original emitter accumulation order, including rounding.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t index = j*rows+row;
        int delay = tau[index];
        if (t < size_t(delay)) continue;
        val_t k = kij[index];
        if (k <= ZERO) continue;
        val_t value = history[(t-delay)*n+j];
        if (!(value <= ZERO)) sum += fminf(k*areas[j], ONE)*value;
    }
    history[t*n+i] = rho*sum + (i == source && t < timeOff ? ONE : ZERO);
}
void runSimulation(SimulationState& state) {
    printf("Running distributed CUDA wave propagation...\n");
    size_t n = state.numTriangles;
    val_t* exchange;
    cudaCheck(cudaMallocHost(&exchange, n*sizeof(val_t)));
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows) {
            propagate<<<(state.localRows+127)/128,128>>>(state.deviceK.data, state.deviceTau.data,
                state.deviceAreas.data, state.deviceB.data, n, state.localRows, state.begin, t,
                state.numTimesteps/2, state.sourceIndex, state.rho[0]);
            cudaCheck(cudaGetLastError());
        }
        // Host staging works with ordinary MPI as well as CUDA-aware MPI.
        if (worldSize > 1) {
            if (state.localRows) cudaCheck(cudaMemcpy(exchange+state.begin,
                state.deviceB.data+t*n+state.begin, state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange, state.counts.data(),
                          state.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            cudaCheck(cudaMemcpy(state.deviceB.data+t*n, exchange, n*sizeof(val_t), cudaMemcpyHostToDevice));
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFreeHost(exchange));
    // Only the reporting rank needs a host copy of the trajectory.
    if (!worldRank) cudaCheck(cudaMemcpy(state.radB.data(), state.deviceB.data,
        state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
}
__global__ void correlate(const val_t* history, val_t* distances, size_t n, size_t rows,
                          size_t begin, size_t steps, size_t source) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    val_t maxCorr = ZERO;
    size_t best = 0;
    for (size_t lag = 0; lag < steps; ++lag) {
        val_t sum = ZERO;
        for (size_t tt = lag; tt < steps; ++tt)
            sum += history[(tt-lag)*n+source]*history[tt*n+begin+row];
        if (sum > maxCorr) { maxCorr = sum; best = lag; }
    }
    distances[row] = WAVE_SPEED*static_cast<val_t>(best);
}
void computeDistances(SimulationState& state) {
    printf("Computing distances via CUDA cross-correlation...\n");
    if (state.localRows) {
        correlate<<<(state.localRows+127)/128,128>>>(state.deviceB.data, state.deviceDistances.data,
            state.numTriangles, state.localRows, state.begin, state.numTimesteps, state.sourceIndex);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(state.distances.data()+state.begin, state.deviceDistances.data,
                            state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.counts.data(),
                  state.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
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
    ParallelRuntime runtime(argc, argv);
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

    if (timesteps < 1) {
        printf("Timesteps must be positive\n");
        return 1;
    }
    runtime.selectDevice();
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
    long localTotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTotalTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
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
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Rank 0 device matrix/history memory: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults && !worldRank) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int success = 1;
    if (validate && !worldRank) success = validateResults(state);
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!success) return 1;

    return 0;
}

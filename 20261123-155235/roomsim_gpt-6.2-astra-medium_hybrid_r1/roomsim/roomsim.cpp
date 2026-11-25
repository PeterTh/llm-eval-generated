/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed, GPU-accelerated implementation of room impulse response
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
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            // Reverse winding to make normals point inward
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
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

// Preorder layout lets a ray skip a missed subtree with one index lookup.
void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<int>& indices) {
    const size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flatten(*child, nodes, indices);
    nodes[index].end = static_cast<int>(nodes.size());
}

__device__ bool blocked(Vec3 from, Vec3 to, Vec3 direction, float length,
                        int i, int j, const Triangle* triangles,
                        const FlatNode* nodes, const int* indices) {
    for (int k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.intersects(from, to)) {
            k = node.end;
            continue;
        }
        for (int p = node.first; p < node.first + node.count; ++p) {
            int other = indices[p];
            if (other == i || other == j) continue;
            const Triangle& tri = triangles[other];
            float hit = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (hit > EPSILON && hit < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

constexpr int blockSize = 128;
constexpr int batchPairs = 8192;

__global__ void formFactors(size_t first, int count, int n, int rowStart, int rows,
                            const Triangle* triangles, const float* areas,
                            const FlatNode* nodes, const int* indices, const float4* samples,
                            float* weights, unsigned char* delays, unsigned long long* nonzero, int* minDelay) {
    __shared__ float contributions[blockSize];
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = thread / NUM_RAYS;
    int lane = threadIdx.x % NUM_RAYS;
    unsigned mask = __ballot_sync(0xffffffffu, pair < count);
    if (pair >= count) return;
    size_t linear = first + pair;
    int row = linear / n;
    int i = row + rowStart, j = linear % n;
    float4 sample = samples[thread];
    float contribution = 0;
    if (sample.x >= 0) {
        const Triangle& a = triangles[i];
        const Triangle& b = triangles[j];
        Vec3 from = randomPointInTriangle(a, sample.x, sample.y);
        Vec3 to = randomPointInTriangle(b, sample.z, sample.w);
        Vec3 delta = to - from;
        float squared = delta.squaredNorm();
        if (squared >= EPSILON) {
            float length = sqrtf(squared);
            float ci = delta.dot(a._normal) / length;
            float cj = (-delta).dot(b._normal) / length;
            if (ci > 0 && cj > 0 &&
                !blocked(from, to, delta / length, length, i, j, triangles, nodes, indices))
                contribution = (ci * cj) / (PI * squared);
        }
    }
    contributions[threadIdx.x] = contribution;
    __syncwarp(mask);
    if (lane == 0) {
        float sum = 0;
        for (int k = 0; k < NUM_RAYS; ++k) sum += contributions[threadIdx.x + k];
        float kij = sum * INV_NUM_RAYS;
        size_t pos = static_cast<size_t>(j) * rows + row;
        weights[pos] = fminf(kij * areas[j], ONE);
        // The mesh radius is ten, so a delay is at most 40 timesteps.
        int delay = static_cast<int>(ceilf(
            (triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
        delays[pos] = static_cast<unsigned char>(delay);
        if (kij > 0) atomicMin(minDelay, delay);
        if (nonzero && kij > EPSILON) atomicAdd(nonzero, 1ULL);
    }
}

// Adjacent lanes own adjacent receivers. Each lane sums emitters in mesh order.
__global__ void propagate(int t, int steps, int n, int start, int rows, int source,
                          float rho, const float* weights, const unsigned char* delays,
                          float* history) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    t += blockIdx.y;
    if (row >= rows) return;
    float sum = 0;
    for (int j = 0; j < n; ++j) {
        size_t p = static_cast<size_t>(j) * rows + row;
        int delay = delays[p];
        if (delay > t || weights[p] <= 0) continue;
        float value = history[static_cast<size_t>(j) * steps + t - delay];
        if (value > 0) sum += weights[p] * value;
    }
    history[static_cast<size_t>(start + row) * steps + t] =
        rho * sum + (start + row == source && t < steps / 2 ? ONE : ZERO);
}

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "roomsim CUDA error: %s\n", cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

template<typename T>
struct DeviceArray {
    T* ptr = nullptr;
    explicit DeviceArray(size_t count) {
        if (count) check(cudaMalloc(&ptr, count * sizeof(T)));
    }
    DeviceArray(const DeviceArray&) = delete;
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    void upload(const std::vector<T>& data) {
        check(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    std::vector<float> distances;
    unsigned long long nonZeroKij = 0;
    int receivedEnergy = 0;
    int minDelay = 40;
};

struct Partition {
    std::vector<int> counts, offsets;
    int start, rows;
    Partition(int n, int rank, int ranks) : counts(ranks), offsets(ranks) {
        for (int r = 0; r < ranks; ++r) {
            counts[r] = n / ranks + (r < n % ranks);
            offsets[r] = (n / ranks) * r + std::min(r, n % ranks);
        }
        start = offsets[rank];
        rows = counts[rank];
    }
};

struct SampleBatch {
    float4* host;
    DeviceArray<float4> device{batchPairs * NUM_RAYS};
    cudaStream_t stream;
    SampleBatch() {
        check(cudaMallocHost(&host, batchPairs * NUM_RAYS * sizeof(float4)));
        check(cudaStreamCreate(&stream));
    }
    ~SampleBatch() { cudaStreamDestroy(stream); cudaFreeHost(host); }
};

// Supply the reconstructed state through the standard seed-sequence interface.
struct SamplingSeed {
    using result_type = uint32_t;
    std::array<result_type, std::mt19937::state_size> words{};
    template<class Iterator>
    void generate(Iterator begin, Iterator end) const {
        std::copy_n(words.begin(), end - begin, begin);
    }
};

// Jump the scalar reference MT19937 stream to a rank's first draw. For large
// offsets evaluate x^offset modulo its characteristic polynomial over GF(2).
// This takes logarithmic time in the offset, independently on each rank.
std::mt19937 samplingEngine(uint64_t offset) {
    std::mt19937 engine(42);
    if (offset < (1ULL << 23)) {
        engine.discard(offset);
        return engine;
    }
    constexpr int degree = 19937;
    constexpr int words = (degree + 63) / 64;
    // Non-leading exponents of the MT19937 characteristic polynomial.
    constexpr int terms[] = {
        0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681, 3908, 4135, 4362,
        4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752, 7979, 8206, 9505, 9901, 9969,
        10128, 10693, 10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374, 11384, 11485,
        11611, 11712, 11717, 11838, 11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509,
        12620, 12673, 12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243,
        13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093, 14151, 14209,
        14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774, 14779, 14953, 15001, 15006,
        15059, 15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682, 15909, 15967,
        16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590, 16595, 16817, 16822,
        16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783, 17841, 17952,
        18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314
    };
    std::array<uint64_t, words> remainder{};
    remainder[0] = 1;
    for (int bit = 63; bit >= 0; --bit) {
        if ((offset >> bit) == 0) continue;
        // In GF(2), squaring doubles each monomial's exponent.
        std::array<uint64_t, words * 2 + 2> square{};
        int extra = (offset >> bit) & 1;
        for (int i = 0; i < degree; ++i) {
            int exponent = 2 * i + extra;
            square[exponent / 64] |= ((remainder[i / 64] >> (i % 64)) & 1)
                                      << (exponent % 64);
        }
        // The second-highest degree is over 64 places below the leading
        // term, so an entire word of quotient bits can be reduced at once.
        for (int base = (degree / 64) * 64; base >= 0; base -= 64) {
            int top = degree + base;
            uint64_t quotient = (square[top / 64] >> (top % 64)) |
                                (square[top / 64 + 1] << (64 - top % 64));
            if (!quotient) continue;
            for (int term : terms) {
                int pos = base + term;
                square[pos / 64] ^= quotient << (pos % 64);
                if (pos % 64) square[pos / 64 + 1] ^= quotient >> (64 - pos % 64);
            }
        }
        std::copy_n(square.begin(), words, remainder.begin());
        remainder.back() &= (uint64_t{1} << (degree % 64)) - 1;
    }

    // The 624 untempered words preceding the next draw seed the standard
    // engine at the requested position.
    constexpr int width = std::mt19937::state_size;
    std::vector<uint32_t> raw(degree + width);
    raw[0] = 42;
    for (int i = 1; i < width; ++i)
        raw[i] = 1812433253u * (raw[i-1] ^ (raw[i-1] >> 30)) + i;
    for (size_t i = width; i < raw.size(); ++i) {
        uint32_t y = (raw[i-width] & 0x80000000u) | (raw[i-width+1] & 0x7fffffffu);
        raw[i] = raw[i-width+397] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0u);
    }
    SamplingSeed seed;
    for (int i = 0; i < degree; ++i)
        if ((remainder[i / 64] >> (i % 64)) & 1)
            for (int j = 0; j < width; ++j) seed.words[j] ^= raw[i + j];
    return std::mt19937(seed);
}

void computeFormFactors(const std::vector<Triangle>& mesh, const Partition& part, int rank,
                        DeviceArray<float>& weights, DeviceArray<unsigned char>& delays,
                        SimulationState& state, bool validate) {
    int n = mesh.size();
    Octree tree;
    tree.build(mesh);
    std::vector<FlatNode> flat;
    std::vector<int> indices;
    flatten(tree, flat, indices);
    std::vector<float> areas(n);
    unsigned long long active = 0;
    auto sampled = [&](int i, int j) {
        return i != j && mesh[i]._normal.dot(mesh[j]._normal) <= 0.99f;
    };
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < n; ++i) areas[i] = mesh[i].area();
        #pragma omp for reduction(+:active)
        for (int row = part.start; row < part.start + part.rows; ++row)
            for (int j = 0; j < n; ++j) active += sampled(row, j);
    }
    unsigned long long preceding = 0;
    MPI_Exscan(&active, &preceding, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    auto rng = samplingEngine(rank && part.rows ? preceding * NUM_RAYS * 4 : 0);
    std::uniform_real_distribution<float> uniform(0, 1);

    DeviceArray<Triangle> triangles(n);
    DeviceArray<float> deviceAreas(n);
    DeviceArray<FlatNode> nodes(flat.size());
    DeviceArray<int> deviceIndices(indices.size());
    DeviceArray<unsigned long long> nonzero(validate ? 1 : 0);
    DeviceArray<int> minDelay(1);
    check(cudaMemcpy(minDelay.ptr, &state.minDelay, sizeof(int), cudaMemcpyHostToDevice));
    triangles.upload(mesh);
    deviceAreas.upload(areas);
    nodes.upload(flat);
    deviceIndices.upload(indices);
    if (validate) check(cudaMemset(nonzero.ptr, 0, sizeof(unsigned long long)));

    SampleBatch buffers[2];
    size_t pairs = static_cast<size_t>(part.rows) * n;
    for (size_t first = 0; first < pairs; first += batchPairs) {
        int count = std::min<size_t>(batchPairs, pairs - first);
        SampleBatch& batch = buffers[(first / batchPairs) % 2];
        check(cudaStreamSynchronize(batch.stream));
        for (int p = 0; p < count; ++p) {
            int i = part.start + (first + p) / n, j = (first + p) % n;
            float4* rays = batch.host + p * NUM_RAYS;
            if (sampled(i, j)) {
                for (int k = 0; k < NUM_RAYS; ++k)
                    rays[k] = {uniform(rng), uniform(rng), uniform(rng), uniform(rng)};
            } else {
                for (int k = 0; k < NUM_RAYS; ++k) rays[k].x = -1;
            }
        }
        int rays = count * NUM_RAYS;
        check(cudaMemcpyAsync(batch.device.ptr, batch.host, rays * sizeof(float4),
                              cudaMemcpyHostToDevice, batch.stream));
        formFactors<<<(rays + blockSize - 1) / blockSize, blockSize, 0, batch.stream>>>(
            first, count, n, part.start, part.rows, triangles.ptr, deviceAreas.ptr, nodes.ptr,
            deviceIndices.ptr, batch.device.ptr, weights.ptr, delays.ptr, nonzero.ptr, minDelay.ptr);
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
    int localMin;
    check(cudaMemcpy(&localMin, minDelay.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&localMin, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate) {
        unsigned long long local;
        check(cudaMemcpy(&local, nonzero.ptr, sizeof(local), cudaMemcpyDeviceToHost));
        MPI_Reduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void runSimulation(const Partition& part, int ranks, const SimulationState& state, float rho,
                   const DeviceArray<float>& weights, const DeviceArray<unsigned char>& delays,
                   DeviceArray<float>& history) {
    int n = state.numTriangles;
    int steps = state.numTimesteps;
    int window = std::min(state.minDelay, steps);
    float* slice = nullptr;
    std::vector<int> counts(ranks), offsets(ranks);
    if (ranks > 1 && window > 0)
        check(cudaMallocHost(&slice, static_cast<size_t>(n) * window * sizeof(float)));
    // No nonzero edge can deliver a signal within the same window. All times
    // in it can run together and the cluster exchanges one contiguous batch.
    for (int t = 0; t < steps;) {
        int width = std::min(window, steps - t);
        if (part.rows) {
            dim3 grid((part.rows + blockSize - 1) / blockSize, width);
            propagate<<<grid, blockSize>>>(t, steps, n, part.start, part.rows, state.sourceIndex,
                                          rho, weights.ptr, delays.ptr, history.ptr);
            check(cudaGetLastError());
        }
        if (ranks > 1) {
            for (int r = 0; r < ranks; ++r) {
                counts[r] = part.counts[r] * width;
                offsets[r] = part.offsets[r] * width;
            }
            size_t pitch = static_cast<size_t>(steps) * sizeof(float);
            size_t bytes = width * sizeof(float);
            if (part.rows)
                check(cudaMemcpy2D(slice + part.start * width, bytes,
                                   history.ptr + static_cast<size_t>(part.start) * steps + t,
                                   pitch, bytes, part.rows, cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, slice, counts.data(), offsets.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            check(cudaMemcpy2D(history.ptr + t, pitch, slice, bytes, bytes, n, cudaMemcpyHostToDevice));
        }
        t += width;
    }
    check(cudaDeviceSynchronize());
    if (slice) check(cudaFreeHost(slice));
}

void computeDistances(const Partition& part, const DeviceArray<float>& history,
                      SimulationState& state, int rank, bool validate) {
    size_t steps = state.numTimesteps, n = state.numTriangles;
    std::vector<float> samples(steps * part.rows), source(steps), local(part.rows);
    if (steps) {
        if (part.rows)
            check(cudaMemcpy(samples.data(), history.ptr + part.start * steps,
                             samples.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(source.data(), history.ptr + state.sourceIndex * steps,
                         steps * sizeof(float), cudaMemcpyDeviceToHost));
    }
    int energy = 0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < part.rows; ++i) {
        if (!steps) continue;
        const float* signal = samples.data() + i * steps;
        float best = 0;
        size_t bestT = 0;
        for (size_t lag = 0; lag < steps; ++lag) {
            float sum = 0;
            for (size_t t = lag; t < steps; ++t) sum += source[t - lag] * signal[t];
            if (sum > best) { best = sum; bestT = lag; }
        }
        local[i] = WAVE_SPEED * bestT;
        if (validate)
            energy += std::any_of(signal, signal + steps, [](float v){ return v > EPSILON; });
    }
    if (validate) MPI_Reduce(&energy, &state.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) state.distances.resize(n);
    MPI_Gatherv(local.data(), part.rows, MPI_FLOAT, state.distances.data(), part.counts.data(),
                part.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    int receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
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

int benchmark(int argc, char** argv, int rank, int ranks) {
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (timesteps < 0) {
        if (!rank) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    check(cudaGetDeviceCount(&devices));
    if (!devices) check(cudaErrorNoDevice);
    check(cudaSetDevice(localRank % devices));

    IcosphereMesh mesh(subdivisions, 10.0f);
    SimulationState state{};
    state.numTriangles = mesh.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = static_cast<size_t>(sourceIdx) % state.numTriangles;
    Partition part(state.numTriangles, rank, ranks);
    if (!rank) {
        printf("Room Response Simulation Benchmark\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Timesteps: %d\nSource triangle: %d\nReflectivity: %.2f\n", timesteps, sourceIdx, reflectivity);
    }
    size_t pairs = state.numTriangles * part.rows;
    DeviceArray<float> weights(pairs), history(state.numTriangles * timesteps);
    DeviceArray<unsigned char> delays(pairs);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    computeFormFactors(mesh.triangles, part, rank, weights, delays, state, validate);
    MPI_Barrier(MPI_COMM_WORLD);
    double pre = MPI_Wtime();
    runSimulation(part, ranks, state, reflectivity, weights, delays, history);
    MPI_Barrier(MPI_COMM_WORLD);
    double sim = MPI_Wtime();
    computeDistances(part, history, state, rank, validate);
    if (rank) return 0;
    double end = MPI_Wtime();
    double totalTime = (end - start) * 1000;
    printf("Precomputation time: %.3f ms\n", (pre - start) * 1000);
    printf("Simulation time: %.3f ms\n", (sim - pre) * 1000);
    printf("Distance computation time: %.3f ms\n", (end - sim) * 1000);
    printf("Total computation time: %.3f ms\n", totalTime);

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

    printf("  Distributed form-factor and delay storage: %.2f MB\n",
           (sizeof(float) + sizeof(unsigned char)) * kijOps / (1024.0 * 1024.0));

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

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = benchmark(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}

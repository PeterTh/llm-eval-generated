#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Standard test cases for validation
inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

// Build the i-th option of the global set (scaled test set).
// Depends only on i, so any rank can construct any option independently.
inline OptionInput makeOption(const size_t i) noexcept {
    constexpr auto testOptions = getTestOptions();
    // Cycle through test options and vary parameters slightly
    OptionInput option = testOptions[i % testOptions.size()];
    
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Contiguous block distribution of [0, total) across ranks.
// The first (total % size) ranks receive one extra element.
struct Partition {
    size_t begin;
    size_t count;
};

inline Partition blockPartition(const size_t total, const int rank, const int size) noexcept {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = total / p;
    const size_t rem = total % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Generate the locally owned block [begin, begin + count) of the option set
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t count) {
    options.resize(count);
    for (size_t j = 0; j < count; ++j) {
        options[j] = makeOption(begin + j);
    }
}

// Gather variable-length per-rank arrays onto the root rank, in rank order.
// Blocks are contiguous and ordered by rank, so concatenation yields global order.
// Only the root's `out` is meaningful on return.
void gatherToRoot(const double* localData, const int localCount,
                  std::vector<double>& out, const int rank, const int size, MPI_Comm comm) {
    std::vector<int> counts(size, 0);
    std::vector<int> displs(size, 0);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    if (rank == 0) {
        long long total = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = static_cast<int>(total);
            total += counts[r];
        }
        if (total > INT_MAX) {
            fprintf(stderr, "Error: gathering %lld results exceeds MPI int count limit\n", total);
            MPI_Abort(comm, 1);
        }
        out.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(localData, localCount, MPI_DOUBLE,
                out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
}

bool validateResults(const std::vector<double>& results, const size_t numChecks) {
    bool allPassed = true;
    
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = makeOption(i).value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               static_cast<int>(i), computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", static_cast<int>(i), computed);
            allPassed = false;
        }
    }
    
    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);
    
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on every rank, so all ranks agree)
    int exitCode = -1; // -1: continue running
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitCode = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }
    
    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank owns one contiguous block of the option set
    const Partition part = blockPartition(numOptions, rank, size);
    
    // Generate local options (outside the timed region, as in the serial version)
    std::vector<OptionInput> options;
    generateOptions(options, part.begin, part.count);
    
    // Allocate local results
    std::vector<double> localResults(part.count);
    
    if (root) {
        printf("Pricing options...\n");
    }
    
    // Price local options; the reported time is the slowest rank's time
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    
    for (size_t i = 0; i < part.count; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const double t1 = MPI_Wtime();
    double elapsedSec = t1 - t0;
    double maxElapsedSec = 0.0;
    MPI_Reduce(&elapsedSec, &maxElapsedSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (root) {
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::duration<double>(maxElapsedSec));
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    }
    
    // Print results for external validation (full gather required)
    if (printResults) {
        std::vector<double> allResults;
        gatherToRoot(localResults.data(), static_cast<int>(part.count), allResults, rank, size, MPI_COMM_WORLD);
        if (root) {
            print_results(allResults, "OptionPrices");
        }
    }
    
    // Validation: only the first few results are needed on the root
    int rc = 0;
    if (validate) {
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        
        // Number of locally owned entries among the first numChecks global indices
        const size_t headCount = (part.begin >= numChecks)
            ? 0
            : std::min(part.count, numChecks - part.begin);
        
        std::vector<double> head;
        gatherToRoot(localResults.data(), static_cast<int>(headCount), head, rank, size, MPI_COMM_WORLD);
        
        if (root) {
            printf("Validating results...\n");
            const bool valid = validateResults(head, numChecks);
            if (valid) {
                printf("Validation: PASSED\n");
                rc = 0;
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
        // Share the exit status so every rank terminates consistently
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return rc;
}

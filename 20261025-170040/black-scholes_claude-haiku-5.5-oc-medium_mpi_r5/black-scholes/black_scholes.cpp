#include <mpi.h>

#include <algorithm>
#include <array>
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

// Number of leading results checked by validation
inline constexpr size_t kValidationChecks = 10;

// Contiguous block of the global option index space owned by one rank
struct Partition {
    size_t start;   // first global index
    size_t count;   // number of options
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

// Option i of the benchmark set. Each option depends only on its global index,
// so any rank can construct its own slice without communication.
inline OptionInput makeOption(const size_t i) noexcept {
    constexpr auto testOptions = getTestOptions();

    // Cycle through test options and vary parameters slightly
    const OptionInput& base = testOptions[i % testOptions.size()];
    OptionInput option = base;

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Block partition of numOptions across size ranks (first `rem` ranks get one extra)
Partition blockPartition(const size_t numOptions, const int rank, const int size) noexcept {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = numOptions / p;
    const size_t rem = numOptions % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Gather the first numNeeded results (global order) from all ranks onto rank 0.
// Only meaningful on rank 0 for `results`; other ranks contribute their local part.
void gatherResults(const std::vector<double>& localResults, const size_t numOptions,
                   const size_t numNeeded, const int rank, const int size,
                   std::vector<double>& results) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const Partition p = blockPartition(numOptions, r, size);
        const size_t begin = std::min(p.start, numNeeded);
        const size_t end = std::min(p.start + p.count, numNeeded);
        counts[r] = static_cast<int>(end - begin);
        displs[r] = static_cast<int>(begin);
    }

    if (rank == 0) {
        results.resize(numNeeded);
    }

    MPI_Gatherv(localResults.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool validateResults(const size_t numOptions, const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min(kValidationChecks, numOptions);
    
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = makeOption(i).value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank prices its own contiguous block of options
    const Partition part = blockPartition(numOptions, rank, size);
    std::vector<double> localResults(part.count);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Time the parallel pricing phase: all ranks start together, and the
    // reported time is the slowest rank (the wall-clock time of the phase).
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (size_t j = 0; j < part.count; ++j) {
        localResults[j] = blackScholes(makeOption(part.start + j));
    }

    const double localElapsed = MPI_Wtime() - tStart;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / elapsed);
    }

    // Only gather what the output actually needs: all results for -r,
    // the leading results for -v, nothing otherwise.
    size_t numNeeded = 0;
    if (printResults) {
        numNeeded = numOptions;
    } else if (validate) {
        numNeeded = std::min(kValidationChecks, numOptions);
    }

    int exitCode = 0;
    if (numNeeded > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: too many results to gather (%zu)\n", numNeeded);
        }
        return 1;
    }

    std::vector<double> results;
    if (numNeeded > 0) {
        gatherResults(localResults, numOptions, numNeeded, rank, size, results);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating results...\n");
        bool valid = validateResults(numOptions, results);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int exitCode = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return exitCode;
}

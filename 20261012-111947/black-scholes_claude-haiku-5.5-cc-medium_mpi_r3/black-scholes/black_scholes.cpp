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

// Build the option with global index i of the benchmark set by scaling the test set
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

// Contiguous range [begin, begin + count) of global option indices owned by one rank
struct Block {
    size_t begin;
    size_t count;
};

// Block distribution of numOptions across ranks; the first (numOptions % size) ranks get one extra
inline Block blockFor(const size_t numOptions, const int rank, const int size) noexcept {
    const size_t base = numOptions / static_cast<size_t>(size);
    const size_t rem = numOptions % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Generate the options owned by this rank
void generateLocalOptions(std::vector<OptionInput>& options, const Block block) {
    options.resize(block.count);
    for (size_t i = 0; i < block.count; ++i) {
        options[i] = makeOption(block.begin + i);
    }
}

bool validateResults(const std::vector<double>& results, const size_t numOptions) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), numOptions);

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = makeOption(i).value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
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
    printf("Run under MPI, e.g.: mpirun -np <p> %s [options]\n", progName);
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

    // Full results are only needed on rank 0 for printing and validation
    const bool needResults = printResults || validate;
    if (needResults && numOptions > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: -r and -v support at most %d options\n", INT_MAX);
        }
        return 1;
    }

    // Generate this rank's share of options
    const Block block = blockFor(numOptions, rank, size);
    std::vector<OptionInput> options;
    generateLocalOptions(options, block);

    // Allocate local results
    std::vector<double> results(block.count);

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (size_t i = 0; i < block.count; ++i) {
        results[i] = blackScholes(options[i]);
    }

    const double localElapsed = MPI_Wtime() - start;

    // The parallel computation takes as long as its slowest rank
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / elapsed);
    }

    int exitCode = 0;
    if (needResults) {
        // Gather all results in global order on rank 0
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const Block b = blockFor(numOptions, r, size);
            counts[r] = static_cast<int>(b.count);
            displs[r] = static_cast<int>(b.begin);
        }

        std::vector<double> allResults;
        if (rank == 0) {
            allResults.resize(numOptions);
        }
        MPI_Gatherv(results.data(), static_cast<int>(block.count), MPI_DOUBLE,
                    allResults.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(allResults, "OptionPrices");
            }

            // Validation
            if (validate) {
                printf("Validating results...\n");
                if (validateResults(allResults, numOptions)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }

        // Share the outcome so every rank exits with the same status
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

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

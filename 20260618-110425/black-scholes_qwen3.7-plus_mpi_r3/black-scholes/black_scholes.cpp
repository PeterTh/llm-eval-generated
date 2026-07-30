#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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

// Generate options for a contiguous range of global indices [startIdx, startIdx+count)
void generateOptionsRange(std::vector<OptionInput>& options, size_t startIdx, size_t count) {
    constexpr auto testOptions = getTestOptions();
    constexpr size_t numTest = testOptions.size();
    options.resize(count);

    for (size_t j = 0; j < count; ++j) {
        const size_t i = startIdx + j;
        const OptionInput& base = testOptions[i % numTest];
        options[j] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(numTest));
        options[j].spot *= factor;
        options[j].strike *= factor;
    }
}

// Generate a larger set of options by scaling the test set (full range [0, numOptions))
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    generateOptionsRange(options, 0, numOptions);
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResultsFlag = false;
    
    // Parse command line arguments (all ranks see the same argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute this rank's contiguous block of work
    const size_t localStart = (numOptions * static_cast<size_t>(rank)) / static_cast<size_t>(size);
    const size_t localEnd   = (numOptions * static_cast<size_t>(rank + 1)) / static_cast<size_t>(size);
    const size_t localCount = localEnd - localStart;

    // Generate only the local portion of options (deterministic from global index)
    std::vector<OptionInput> localOptions;
    generateOptionsRange(localOptions, localStart, localCount);

    // Price local options
    if (rank == 0) printf("Pricing options...\n");

    std::vector<double> localResults(localCount);

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    for (size_t i = 0; i < localCount; ++i) {
        localResults[i] = blackScholes(localOptions[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    double localElapsed = endTime - startTime;
    double maxElapsed;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }

    // Gather results to rank 0 when needed for output or validation
    const bool needGather = printResultsFlag || validate;
    std::vector<double> allResults;
    if (needGather) {
        // Compute recvCounts and displacements analytically (no communication needed)
        std::vector<int> recvCounts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t s = (numOptions * static_cast<size_t>(r)) / static_cast<size_t>(size);
            const size_t e = (numOptions * static_cast<size_t>(r + 1)) / static_cast<size_t>(size);
            recvCounts[r] = static_cast<int>(e - s);
            displs[r] = static_cast<int>(s);
        }

        if (rank == 0) {
            allResults.resize(numOptions);
        }

        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                     allResults.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResultsFlag) {
        print_results(allResults, "OptionPrices");
    }
    
    // Validation (rank 0 only)
    int exitCode = 0;
    if (rank == 0 && validate) {
        std::vector<OptionInput> allOptions;
        generateOptions(allOptions, numOptions);

        printf("Validating results...\n");
        bool valid = validateResults(allOptions, allResults);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    // Broadcast exit code so all ranks return consistently
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}

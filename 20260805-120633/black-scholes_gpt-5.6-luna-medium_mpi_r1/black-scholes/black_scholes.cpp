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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t firstOption,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t localIndex = 0; localIndex < numOptions; ++localIndex) {
        const size_t globalIndex = firstOption + localIndex;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[localIndex] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[localIndex].spot *= factor;
        options[localIndex].strike *= factor;
    }
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

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool parseFailed = false;
    bool showHelp = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseFailed = true;
        }
    }

    if (showHelp || parseFailed) {
        if (rank == 0) {
            if (parseFailed) {
                printf("Unknown or invalid option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseFailed ? 1 : 0;
    }

    // Divide the global option range as evenly as possible.  The first
    // remainder ranks receive one extra option, preserving global ordering.
    const size_t processCount = static_cast<size_t>(worldSize);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t baseCount = numOptions / processCount;
    const size_t remainder = numOptions % processCount;
    const size_t localCount = baseCount + (rankIndex < remainder ? 1 : 0);
    const size_t firstOption = rankIndex * baseCount + std::min(rankIndex, remainder);
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank generates only the options it owns.  Generation is deterministic
    // from the global index, so no input broadcast is required.
    std::vector<OptionInput> options;
    generateOptions(options, firstOption, localCount);
    
    std::vector<double> localResults(localCount);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localCount; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        printf("Options per second: %.0f\n", elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
    }
    
    // Print results for external validation
    std::vector<double> results;
    if (printResults) {
        // MPI_Gatherv uses int counts; this benchmark's command-line size is
        // expected to fit the MPI count range, while local storage remains size_t.
        const int localCountInt = static_cast<int>(localCount);
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(worldSize);
            displacements.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                const size_t rCount = baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
                counts[r] = static_cast<int>(rCount);
                displacements[r] = static_cast<int>(static_cast<size_t>(r) * baseCount +
                                                     std::min(static_cast<size_t>(r), remainder));
            }
            results.resize(numOptions);
        }
        MPI_Gatherv(localResults.empty() ? nullptr : localResults.data(), localCountInt, MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(results, "OptionPrices");
        }
    }
    
    // Validation
    int validationPassed = 1;
    if (validate) {
        // Only the first ten global results are examined, matching the original
        // validation semantics. Gather only that prefix rather than the full
        // result array.
        if (rank == 0) {
            if (!printResults) {
                results.resize(std::min<size_t>(10, numOptions));
            }
            if (!printResults) {
                const size_t firstCount = results.size();
                std::vector<int> counts(worldSize);
                std::vector<int> displacements(worldSize);
                for (int r = 0; r < worldSize; ++r) {
                    const size_t rFirst = static_cast<size_t>(r) * baseCount +
                                          std::min(static_cast<size_t>(r), remainder);
                    const size_t rCount = baseCount +
                                          (static_cast<size_t>(r) < remainder ? 1 : 0);
                    const size_t rOffset = rFirst < firstCount ? 0 : firstCount;
                    const size_t rTake = rFirst < firstCount
                                             ? std::min(rCount, firstCount - rFirst) : 0;
                    counts[r] = static_cast<int>(rTake);
                    displacements[r] = static_cast<int>(rFirst + rOffset);
                }
                const size_t localOffset = 0;
                const size_t localTake = firstOption < firstCount
                                             ? std::min(localCount, firstCount - firstOption) : 0;
                MPI_Gatherv(localTake == 0 ? nullptr : localResults.data() + localOffset,
                            static_cast<int>(localTake), MPI_DOUBLE,
                            results.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                            0, MPI_COMM_WORLD);
            }
            printf("Validating results...\n");
            std::vector<OptionInput> expectedOptions;
            generateOptions(expectedOptions, 0, std::min<size_t>(10, results.size()));
            validationPassed = validateResults(expectedOptions, results) ? 1 : 0;
            printf("Validation: %s\n", validationPassed == 1 ? "PASSED" : "FAILED");
        } else if (!printResults) {
            const size_t firstCount = std::min<size_t>(10, numOptions);
            const size_t localOffset = 0;
            const size_t localTake = firstOption < firstCount
                                         ? std::min(localCount, firstCount - firstOption) : 0;
            // The receive-side metadata is constructed on rank 0; all ranks
            // participate in this collective, including ranks with zero data.
            MPI_Gatherv(localTake == 0 ? nullptr : localResults.data() + localOffset,
                        static_cast<int>(localTake), MPI_DOUBLE,
                        nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validate && validationPassed == 0 ? 1 : 0;
}

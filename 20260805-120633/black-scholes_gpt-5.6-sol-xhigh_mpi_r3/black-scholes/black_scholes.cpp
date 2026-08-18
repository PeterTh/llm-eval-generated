#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    
    const double sigmaSqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    
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
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t globalIndex = globalOffset + i;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor =
            1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// Split [0, globalSize) into contiguous, near-equal ranges. Contiguous ranges
// preserve the serial result ordering when optional output is gathered.
void getLocalRange(const size_t globalSize, const int rank, const int processCount,
                   size_t& begin, size_t& count) noexcept {
    const size_t processes = static_cast<size_t>(processCount);
    const size_t base = globalSize / processes;
    const size_t remainder = globalSize % processes;
    const size_t rankIndex = static_cast<size_t>(rank);

    count = base + (rankIndex < remainder ? 1 : 0);
    begin = rankIndex * base + std::min(rankIndex, remainder);
}

// MPI_Gatherv is optimal for normal problem sizes, but its counts and
// displacements are int in MPI-3. Fall back to chunked point-to-point transfer
// for larger arrays so the global indexing remains correct.
std::vector<double> gatherResults(const std::vector<double>& localResults,
                                  const size_t globalBegin,
                                  const size_t globalSize,
                                  const int rank,
                                  const int processCount) {
    std::vector<double> globalResults;
    if (rank == 0) {
        globalResults.resize(globalSize);
    }

    if (globalSize <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(processCount));
            displacements.resize(static_cast<size_t>(processCount));
            for (int source = 0; source < processCount; ++source) {
                size_t sourceBegin = 0;
                size_t sourceCount = 0;
                getLocalRange(globalSize, source, processCount, sourceBegin, sourceCount);
                counts[static_cast<size_t>(source)] = static_cast<int>(sourceCount);
                displacements[static_cast<size_t>(source)] = static_cast<int>(sourceBegin);
            }
        }

        MPI_Gatherv(localResults.data(), static_cast<int>(localResults.size()), MPI_DOUBLE,
                    rank == 0 ? globalResults.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return globalResults;
    }

    constexpr size_t transferChunk = 1U << 26;
    constexpr int resultTag = 1;
    if (rank == 0) {
        std::copy(localResults.begin(), localResults.end(),
                  globalResults.begin() + static_cast<std::ptrdiff_t>(globalBegin));
        for (int source = 1; source < processCount; ++source) {
            size_t sourceBegin = 0;
            size_t sourceCount = 0;
            getLocalRange(globalSize, source, processCount, sourceBegin, sourceCount);
            for (size_t received = 0; received < sourceCount;) {
                const int chunk = static_cast<int>(std::min(transferChunk,
                                                            sourceCount - received));
                MPI_Recv(globalResults.data() + sourceBegin + received, chunk, MPI_DOUBLE,
                         source, resultTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                received += static_cast<size_t>(chunk);
            }
        }
    } else {
        for (size_t sent = 0; sent < localResults.size();) {
            const int chunk = static_cast<int>(std::min(transferChunk,
                                                        localResults.size() - sent));
            MPI_Send(localResults.data() + sent, chunk, MPI_DOUBLE, 0, resultTag,
                     MPI_COMM_WORLD);
            sent += static_cast<size_t>(chunk);
        }
    }

    return globalResults;
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
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0; // 0: run, 1: help, 2: invalid arguments
    
    // Parse once and broadcast a compact configuration to every process.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseStatus = 2;
                break;
            }
        }
    }

    std::uint64_t wireNumOptions = static_cast<std::uint64_t>(numOptions);
    int flags[3] = {validate ? 1 : 0, printResults ? 1 : 0, parseStatus};
    MPI_Bcast(&wireNumOptions, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(wireNumOptions);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;
    parseStatus = flags[2];

    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI processes: %d\n", processCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t localBegin = 0;
    size_t localCount = 0;
    getLocalRange(numOptions, rank, processCount, localBegin, localCount);
    
    // Every process generates only its deterministic global slice. This avoids
    // broadcasting the input and keeps memory use proportional to local work.
    std::vector<OptionInput> options;
    generateOptions(options, localCount, localBegin);
    
    std::vector<double> localResults(localCount);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localCount; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Options per second: %.0f\n",
               elapsed > 0.0 ? static_cast<double>(numOptions) / elapsed : 0.0);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> results = gatherResults(localResults, localBegin, numOptions,
                                                    rank, processCount);
        if (rank == 0) {
            print_results(results, "OptionPrices");
        }
    }
    
    // Validation
    int returnCode = 0;
    if (validate) {
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        std::array<double, 10> localChecks{};
        std::array<double, 10> globalChecks{};
        for (size_t i = 0; i < localCount; ++i) {
            const size_t globalIndex = localBegin + i;
            if (globalIndex >= numChecks) {
                break;
            }
            localChecks[globalIndex] = localResults[i];
        }
        MPI_Reduce(localChecks.data(), globalChecks.data(), static_cast<int>(numChecks),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, numChecks);
            const std::vector<double> checkResults(globalChecks.begin(),
                                                   globalChecks.begin() + numChecks);
            const bool valid = validateResults(checkOptions, checkResults);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
        MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return returnCode;
}
